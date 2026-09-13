"""
How much faster the device is, measured on one brain load.

WHY THESE FIVE NUMBERS
The roamer asks the brain for 60 steps (12 ms of brain time) per 50 ms world
step, so (a) and (b) are that call: numpy against the device, one run at a
time, the way roam and plume make it. (c) is the batch the port exists for,
32 seeds of the same window stepped as 32 columns so the weights are read
once per step for all of them. (d) is 250 steps, 50 ms of brain time, the
run that would make the brain real time against the world clock; numpy is
timed on it too so the gap is on record. (e) is what learning costs on the
device: mushroom.MushroomBody.apply() writes its KC-to-MBON weights into
wdata and the next run must see them, so the push sits on the roamer's path,
not in a one-off setup, and its cost belongs next to a run's.

One brain per process, because the connectome is about 2.5 GB in RAM: the
numpy numbers are flysim.FlyBrain.run on the same FlyBrainGPU object, on the
same weights, so the comparison has no second load and no second copy. The
last block repeats the real-brain test's comparison (20 seeds, per-population
counts within 3 SE, cells fired within 5 %) so the JSON carries the
equivalence numbers next to the timings they qualify: a speed-up over a
simulator that fires different neurons would be worth nothing.

  py bench_gpu.py                the connectome, the roamer's eye drive, calibrated gains -> build/gpu_bench.json
  py bench_gpu.py --fake         a 2,000-neuron synthetic graph: no connectome, no RAM check, JSON to a temp file
  py bench_gpu.py --cpu-runs 10 --gpu-runs 50 --batch 32 --long-steps 250 --device cuda --out some.json
"""
import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import flysim
from flysim import BUILD
from flysim_gpu import FlyBrainGPU

ROAMER_STEPS = 60          # roam.FlyPilot(sim_steps=60): one world step of brain
WORLD_STEP_MS = 50.0       # the roamer's and plume's world step
REALTIME_STEPS = 250       # 50 ms of brain at dt = 0.2 ms
RAM_FLOOR_GB = 6.0         # plume_experiment's floor: never load a brain below it


def fake_graph(n=2000, seed=0):
    """A synthetic graph in build_graph.py's shape, about 80 synapses a cell, like the real one."""
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), 80)
    post = rng.integers(0, n, size=pre.size)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    count = rng.integers(1, 6, size=pre.size).astype(np.float32)
    sign = np.where(rng.random(n) < 0.8, 1.0, -1.0).astype(np.float32)
    W = sp.csr_matrix(((count * 0.275 * sign[pre]).astype(np.float32), (post, pre)), shape=(n, n))
    W.sum_duplicates()
    blank = np.array([""] * n, dtype=str)
    path = Path(tempfile.mkdtemp(prefix="bench_gpu_")) / "graph.npz"
    np.savez(path, data=W.data, indices=W.indices.astype(np.int32), indptr=W.indptr.astype(np.int32),
             shape=np.array(W.shape, dtype=np.int64), bodies=np.arange(n, dtype=np.int64), sign=sign,
             types=np.array([f"T{i % 40:02d}" for i in range(n)], dtype=str),
             superclass=blank, subclass=blank, receptor=blank, fru=blank,
             nt=np.where(sign > 0, "acetylcholine", "gaba").astype(str))
    return path


# ---- clocks ----------------------------------------------------------------

def sync(fb):
    """The device's work is asynchronous; a clock stopped before it is done lies."""
    if fb.device.type == "cuda":
        torch.cuda.synchronize(fb.device)


def stats(times_s, steps, runs_per_call=1):
    """
    One timing block: per call, per run (a column of a batch is a run) and
    per step, in milliseconds. The median is the number to quote, the mean is
    what a loop of them costs, min and max show the jitter.
    """
    t = np.asarray(times_s, dtype=float) * 1000.0
    per_run = t / runs_per_call
    return {
        "repeats": int(len(t)), "steps": int(steps), "runs_per_call": int(runs_per_call),
        "ms_per_call_mean": float(t.mean()), "ms_per_call_median": float(np.median(t)),
        "ms_per_run_mean": float(per_run.mean()), "ms_per_run_median": float(np.median(per_run)),
        "ms_per_run_min": float(per_run.min()), "ms_per_run_max": float(per_run.max()),
        "ms_per_step_mean": float(per_run.mean() / steps),
        "ms_per_step_median": float(np.median(per_run) / steps),
    }


def time_cpu(fb, drive, steps, gains, record, repeats):
    """(a) numpy, one run at a time, a different seed each so the work is the roamer's, not a cache's."""
    times, spikes, fired = [], [], []
    for s in range(repeats):
        t0 = time.perf_counter()
        r = flysim.FlyBrain.run(fb, drive, steps, gains=gains, record=record, seed=s)
        times.append(time.perf_counter() - t0)
        spikes.append(r["_spikes_per_sec"])
        fired.append(len(r["_fired"]))
    out = stats(times, steps)
    out.update(spikes_per_sec_mean=float(np.mean(spikes)), cells_fired_mean=float(np.mean(fired)))
    return out


def time_gpu_single(fb, drive, steps, gains, record, repeats, warm=3):
    """(b), (d) the device, one run at a time, warm: the allocator and cuSPARSE have seen this shape."""
    for s in range(warm):
        fb.run(drive, steps, gains=gains, record=record, seed=s)
    sync(fb)
    times, spikes, fired = [], [], []
    for s in range(repeats):
        sync(fb)
        t0 = time.perf_counter()
        r = fb.run(drive, steps, gains=gains, record=record, seed=s)
        sync(fb)
        times.append(time.perf_counter() - t0)
        spikes.append(r["_spikes_per_sec"])
        fired.append(len(r["_fired"]))
    out = stats(times, steps)
    out.update(spikes_per_sec_mean=float(np.mean(spikes)), cells_fired_mean=float(np.mean(fired)))
    return out


def time_gpu_batch(fb, drive, steps, gains, record, B, repeats, warm=1):
    """(c) B seeds as B columns; the per-run figure is the batch's time over B."""
    for _ in range(warm):
        fb.run_batch(drive, steps, gains=gains, record=record, seeds=list(range(B)))
    sync(fb)
    times, spikes = [], []
    for k in range(repeats):
        sync(fb)
        t0 = time.perf_counter()
        outs = fb.run_batch(drive, steps, gains=gains, record=record, seeds=list(range(k * B, (k + 1) * B)))
        sync(fb)
        times.append(time.perf_counter() - t0)
        spikes.append(np.mean([o["_spikes_per_sec"] for o in outs]))
    out = stats(times, steps, runs_per_call=B)
    out.update(batch=int(B), spikes_per_sec_mean=float(np.mean(spikes)))
    return out


# ---- learned weights ---------------------------------------------------------

def learning_positions(fb):
    """
    The positions the learning circuit writes: the real MushroomBody's
    KC-to-MBON synapses, built on this brain with a throw-away store so the
    bench neither reads nor writes the roamer's learned gains. A graph with
    no KC (the fake one) gets random positions of the same count instead.
    """
    try:
        from mushroom import MushroomBody
        mb = MushroomBody(fb, store=Path(tempfile.mkdtemp(prefix="bench_gpu_mb_")) / "gains.npz")
        if len(mb.pos):
            return mb, mb.pos, f"MushroomBody KC->MBON synapses ({mb.stats()['synapses']:,})"
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
    else:
        why = "no KC/MBON in this graph"
    k = min(44042, len(fb.wdata))
    pos = np.sort(np.random.default_rng(0).choice(len(fb.wdata), size=k, replace=False))
    return None, pos, f"random positions ({why})"


def time_push(fb, mb, pos, drive, steps, gains, record, repeats):
    """
    (e) what a change of learned weights costs before the next run may go:
    the host write (MushroomBody.apply's wdata[pos] = base * gain), the push
    of the whole weight array to the device (one copy, one gather, one CSR
    tensor: the position count does not matter to it, the 10 M synapses do),
    and a 60-step run made right after a write, which includes the push run()
    makes on its own. The weights are restored afterwards, and the device is
    checked to hold exactly what wdata holds.
    """
    base = fb.wdata[pos].copy()
    rng = np.random.default_rng(1)
    t_write, t_push, t_run_after = [], [], []
    for _ in range(repeats):
        gain = rng.uniform(0.25, 1.0, len(pos)).astype(np.float32)
        t0 = time.perf_counter()
        if mb is not None:
            mb.gain = gain
            mb.apply()
        else:
            fb.wdata[pos] = base * gain
        t_write.append(time.perf_counter() - t0)
        sync(fb)
        t0 = time.perf_counter()
        fb.push_weights()
        sync(fb)
        t_push.append(time.perf_counter() - t0)
        # the same write again, then the run that has to notice it
        if mb is not None:
            mb.gain = gain * np.float32(0.99)
            mb.apply()
        else:
            fb.wdata[pos] = base * gain * np.float32(0.99)
        sync(fb)
        t0 = time.perf_counter()
        fb.run(drive, steps, gains=gains, record=record, seed=0)
        sync(fb)
        t_run_after.append(time.perf_counter() - t0)
    if mb is not None:
        mb.gain = np.ones(len(pos), dtype=np.float32)
        mb.apply()
    else:
        fb.wdata[pos] = base
    fb.push_weights()
    restored = bool(np.array_equal(fb.wdata[pos], base)) and fb.weights_are_current()
    ms = lambda t: float(np.median(t) * 1000.0)
    return {
        "positions": int(len(pos)), "synapses_pushed": int(len(fb.wdata)), "repeats": int(repeats),
        "ms_host_write_median": ms(t_write), "ms_push_median": ms(t_push), "ms_push_max": float(np.max(t_push) * 1000.0),
        "ms_run_after_write_median": ms(t_run_after), "run_after_write_steps": int(steps),
        "restored_and_current": restored,
    }


# ---- equivalence -------------------------------------------------------------

def equivalence(fb, drive, steps, gains, seeds, pops):
    """
    The real-brain test's comparison, repeated here so the numbers travel with
    the timings: the same seeds on both simulators, per-population spike
    counts within 3 SE of the difference, cells fired within 5 %, and how
    many seeds fired exactly the same set (float32 summation order can part
    them; on the connectome it has not).
    """
    record = {k: fb.where(type_re=v) for k, v in pops.items()}
    secs = steps * fb.p.dt / 1000.0
    cpu = [flysim.FlyBrain.run(fb, drive, steps, gains=gains, record=record, seed=s) for s in seeds]
    gpu = fb.run_batch(drive, steps, gains=gains, record=record, seeds=list(seeds))
    counts = lambda res, k: np.array([float(np.asarray(r[k]).sum() * secs) for r in res])
    out = {"seeds": int(len(seeds)), "steps": int(steps), "populations": {}, "pass": True}
    for k in pops:
        a, b = counts(cpu, k), counts(gpu, k)
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        z = abs(a.mean() - b.mean()) / se if se > 0 else 0.0
        ok = abs(a.mean() - b.mean()) <= 3.0 * se + 1e-9
        out["populations"][k] = {"cells": int(len(record[k])), "cpu_mean_spikes": float(a.mean()),
                                 "gpu_mean_spikes": float(b.mean()), "se_diff": se, "z": float(z),
                                 "within_3se": bool(ok)}
        out["pass"] &= bool(ok)
    fa = np.array([len(r["_fired"]) for r in cpu], dtype=float)
    fg = np.array([len(r["_fired"]) for r in gpu], dtype=float)
    ratio = float(fg.mean() / fa.mean()) if fa.mean() else 1.0
    out["fired_cpu_mean"] = float(fa.mean())
    out["fired_gpu_mean"] = float(fg.mean())
    out["fired_ratio"] = ratio
    out["fired_within_5pct"] = bool(abs(ratio - 1.0) < 0.05)
    out["pass"] &= out["fired_within_5pct"]
    out["seeds_with_identical_fired_sets"] = int(sum(np.array_equal(x["_fired"], y["_fired"]) for x, y in zip(cpu, gpu)))
    out["seed0_fired_cpu"] = int(len(cpu[0]["_fired"]))
    out["seed0_fired_gpu"] = int(len(gpu[0]["_fired"]))
    return out


# ---- environment -------------------------------------------------------------

def free_ram_gb():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out) / (1024 * 1024)
    except Exception:
        return float("nan")


def nvidia_smi_used_mib():
    """What the whole card holds right now, this process and the desktop alike; None without the tool."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip().split(",")
        return int(float(out[0])), out[1].strip()
    except Exception:
        return None, None


def gpu_memory(fb):
    if fb.device.type != "cuda":
        return {"device": str(fb.device), "note": "torch on the CPU: no device memory"}
    free, total = torch.cuda.mem_get_info(fb.device)
    return {"device": str(fb.device), "name": torch.cuda.get_device_name(fb.device),
            "allocated_mib": torch.cuda.memory_allocated(fb.device) / 2**20,
            "reserved_mib": torch.cuda.memory_reserved(fb.device) / 2**20,
            "max_allocated_mib": torch.cuda.max_memory_allocated(fb.device) / 2**20,
            "card_free_mib": free / 2**20, "card_total_mib": total / 2**20}


def reset_peak(fb):
    if fb.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(fb.device)


# ---- main --------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fake", action="store_true", help="a synthetic 2,000-cell graph instead of the connectome")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=ROAMER_STEPS, help="steps per roamer run (a, b, c, e)")
    ap.add_argument("--long-steps", type=int, default=REALTIME_STEPS, help="steps of the real-time run (d)")
    ap.add_argument("--cpu-runs", type=int, default=10)
    ap.add_argument("--gpu-runs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batch-runs", type=int, default=5)
    ap.add_argument("--long-runs", type=int, default=10)
    ap.add_argument("--long-cpu-runs", type=int, default=3)
    ap.add_argument("--push-runs", type=int, default=10)
    ap.add_argument("--equiv-seeds", type=int, default=20)
    ap.add_argument("--out", default=None, help="JSON path (default build/gpu_bench.json; a temp file with --fake)")
    a = ap.parse_args(argv)

    smi_before, driver = nvidia_smi_used_mib()
    ram = free_ram_gb()
    t0 = time.perf_counter()
    if a.fake:
        fb = FlyBrainGPU(fake_graph(), device=a.device)
        drive, how = {tuple(range(0, 100)): 200.0, tuple(range(100, 200)): 60.0}, "synthetic: 200 cells at 200/60 Hz"
        gains, setting = None, None
        pops = {"T00": r"^T00$", "T01": r"^T01$", "T02": r"^T02$", "T03": r"^T03$", "all": r"."}
    else:
        # the test's drive and populations, so the equivalence block here is the test's
        from test_flysim_gpu import RealBrain, eye_drive
        import calibration
        if not ram > RAM_FLOOR_GB:
            raise SystemExit(f"{ram:.1f} GB free; the connectome needs more than {RAM_FLOOR_GB} GB")
        fb = FlyBrainGPU(device=a.device)
        drive, how = eye_drive(fb)
        setting = calibration.CHOSEN
        gains = calibration.gains_for(fb, setting)
        pops = dict(RealBrain.POPS)
    load_s = time.perf_counter() - t0
    driven = int(sum(len(k) for k in drive))
    record = {k: fb.where(type_re=v) for k, v in pops.items()}
    print(f"{fb.n:,} neurons, {len(fb.wdata):,} synapses on {fb.device}, loaded in {load_s:.1f} s "
          f"(free RAM before load {ram:.1f} GB)")
    print(f"drive: {how}, {driven:,} driven neurons; gains: {setting}; record: "
          + ", ".join(f"{k} {len(v)}" for k, v in record.items()))
    mem_after_load = gpu_memory(fb)
    steps, long_steps = a.steps, a.long_steps

    print(f"\n(a) numpy FlyBrain.run, {steps} steps x {a.cpu_runs}")
    cpu60 = time_cpu(fb, drive, steps, gains, record, a.cpu_runs)
    print(f"    {cpu60['ms_per_run_median']:.1f} ms/run (mean {cpu60['ms_per_run_mean']:.1f}, "
          f"min {cpu60['ms_per_run_min']:.1f}, max {cpu60['ms_per_run_max']:.1f}), "
          f"{cpu60['ms_per_step_median']:.3f} ms/step, {cpu60['cells_fired_mean']:,.0f} cells fired")

    print(f"(b) FlyBrainGPU.run, {steps} steps x {a.gpu_runs}, warm")
    reset_peak(fb)
    gpu60 = time_gpu_single(fb, drive, steps, gains, record, a.gpu_runs)
    mem_single = gpu_memory(fb)
    print(f"    {gpu60['ms_per_run_median']:.2f} ms/run (mean {gpu60['ms_per_run_mean']:.2f}, "
          f"min {gpu60['ms_per_run_min']:.2f}, max {gpu60['ms_per_run_max']:.2f}), "
          f"{gpu60['ms_per_step_median']:.3f} ms/step, {gpu60['cells_fired_mean']:,.0f} cells fired")

    print(f"(c) FlyBrainGPU.run_batch, {a.batch} seeds x {steps} steps, x {a.batch_runs}")
    reset_peak(fb)
    batch = time_gpu_batch(fb, drive, steps, gains, record, a.batch, a.batch_runs)
    mem_batch = gpu_memory(fb)
    print(f"    {batch['ms_per_call_median']:.1f} ms/batch = {batch['ms_per_run_median']:.3f} ms/run, "
          f"{batch['ms_per_step_median']:.4f} ms/step/run")

    print(f"(d) {long_steps} steps = {long_steps * fb.p.dt:.0f} ms of brain: GPU x {a.long_runs}, numpy x {a.long_cpu_runs}")
    reset_peak(fb)
    gpu_long = time_gpu_single(fb, drive, long_steps, gains, record, a.long_runs)
    mem_long = gpu_memory(fb)
    cpu_long = time_cpu(fb, drive, long_steps, gains, record, a.long_cpu_runs)
    print(f"    GPU {gpu_long['ms_per_run_median']:.1f} ms/run ({gpu_long['ms_per_step_median']:.3f} ms/step); "
          f"numpy {cpu_long['ms_per_run_median']:.0f} ms/run ({cpu_long['ms_per_step_median']:.3f} ms/step)")

    print(f"(e) learned weights: write, push, run-after-write, x {a.push_runs}")
    mb, pos, pos_how = learning_positions(fb)
    push = time_push(fb, mb, pos, drive, steps, gains, record, a.push_runs)
    push["positions_from"] = pos_how
    print(f"    {push['positions']:,} positions ({pos_how}): host write {push['ms_host_write_median']:.2f} ms, "
          f"push {push['ms_push_median']:.2f} ms (max {push['ms_push_max']:.2f}), "
          f"{steps}-step run after a write {push['ms_run_after_write_median']:.2f} ms; restored {push['restored_and_current']}")

    print(f"(=) equivalence, {a.equiv_seeds} seeds x {steps} steps, numpy against the device")
    eq = equivalence(fb, drive, steps, gains, list(range(a.equiv_seeds)), pops)
    print(f"    {'population':10} {'cells':>6} {'CPU mean':>10} {'GPU mean':>10} {'SE diff':>8} {'z':>6}")
    for k, e in eq["populations"].items():
        print(f"    {k:10} {e['cells']:6d} {e['cpu_mean_spikes']:10.1f} {e['gpu_mean_spikes']:10.1f} "
              f"{e['se_diff']:8.2f} {e['z']:6.2f}")
    print(f"    _fired     {'':6} {eq['fired_cpu_mean']:10.1f} {eq['fired_gpu_mean']:10.1f}   ratio {eq['fired_ratio']:.4f}; "
          f"{eq['seeds_with_identical_fired_sets']}/{eq['seeds']} seeds fired identical sets; pass {eq['pass']}")

    smi_after, _ = nvidia_smi_used_mib()
    world_ms = WORLD_STEP_MS
    speed = {
        "gpu_single_vs_numpy": cpu60["ms_per_run_median"] / gpu60["ms_per_run_median"],
        "gpu_batch_per_run_vs_numpy": cpu60["ms_per_run_median"] / batch["ms_per_run_median"],
        "gpu_batch_per_run_vs_gpu_single": gpu60["ms_per_run_median"] / batch["ms_per_run_median"],
        "gpu_long_vs_numpy": cpu_long["ms_per_run_median"] / gpu_long["ms_per_run_median"],
        "realtime_factor_gpu_long": world_ms / gpu_long["ms_per_run_median"],
        "realtime_factor_numpy_long": world_ms / cpu_long["ms_per_run_median"],
        "realtime_factor_gpu_roamer": world_ms / gpu60["ms_per_run_median"],
        "realtime_factor_numpy_roamer": world_ms / cpu60["ms_per_run_median"],
    }
    print(f"\nspeed-ups (medians): single {speed['gpu_single_vs_numpy']:.1f}x, batch of {a.batch} per run "
          f"{speed['gpu_batch_per_run_vs_numpy']:.1f}x, {long_steps} steps {speed['gpu_long_vs_numpy']:.1f}x; "
          f"{long_steps} steps against a {world_ms:.0f} ms world step: GPU {speed['realtime_factor_gpu_long']:.2f}x real time, "
          f"numpy {speed['realtime_factor_numpy_long']:.3f}x")
    mem = {"after_load": mem_after_load, "peak_single": mem_single, "peak_batch": mem_batch, "peak_long": mem_long,
           "nvidia_smi_card_used_mib_before": smi_before, "nvidia_smi_card_used_mib_after": smi_after}
    if fb.device.type == "cuda":
        print(f"GPU memory: {mem_after_load['allocated_mib']:.0f} MiB after load, peak {mem_single['max_allocated_mib']:.0f} MiB "
              f"single / {mem_batch['max_allocated_mib']:.0f} MiB batch of {a.batch} / {mem_long['max_allocated_mib']:.0f} MiB "
              f"{long_steps} steps; reserved {mem_batch['reserved_mib']:.0f} MiB; card {smi_before} -> {smi_after} MiB (nvidia-smi)")

    result = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "graph": {"fake": bool(a.fake), "neurons": int(fb.n), "synapses": int(len(fb.wdata)), "load_s": load_s},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(fb.device),
                        "gpu": torch.cuda.get_device_name(fb.device) if fb.device.type == "cuda" else None,
                        "driver": driver, "platform": platform.platform(), "free_ram_gb_before_load": ram},
        "drive": {"how": how, "driven_neurons": driven, "gains": setting,
                  "record": {k: int(len(v)) for k, v in record.items()}},
        "a_numpy_roamer_run": cpu60, "b_gpu_roamer_run": gpu60, "c_gpu_batch": batch,
        "d_gpu_realtime_run": gpu_long, "d_numpy_realtime_run": cpu_long, "e_push_weights": push,
        "speedups": speed, "world_step_ms": world_ms, "gpu_memory": mem, "equivalence": eq,
    }
    out = Path(a.out) if a.out else (Path(tempfile.mkdtemp(prefix="bench_gpu_")) / "gpu_bench_fake.json"
                                     if a.fake else BUILD / "gpu_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    main()
