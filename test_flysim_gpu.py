"""
flysim_gpu.FlyBrainGPU against flysim.FlyBrain, the ground truth.

Everything but the last class runs on a 200-neuron synthetic graph written in
the shape build_graph.py writes, on torch's CPU device and, when there is one,
on CUDA. The graph's weights are whole multiples of 0.25 mV and the gains of
0.25, so every synaptic sum is exact in float32 whatever order cuSPARSE adds
it in; that is what lets the tests ask for the same fired set every step
rather than "close". Membrane potentials are then compared to the bit as
well, which they need not be in general (flysim_gpu's PRECISION note), so a
last-bit difference on some future torch would show up here first and only
here.

  py -m pytest -q test_flysim_gpu.py
  FLY_GPU_REAL_BRAIN=1 py -m pytest -q -s test_flysim_gpu.py -k RealBrain
"""
import copy
import json
import os
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

import flysim
import flysim_gpu
from flysim_gpu import FlyBrainGPU

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
N = 200
TYPES = ["KCab", "KCg", "MBON01", "MBON11", "T04", "T05", "T06", "T07"]
GAINS = np.array([0.5, 1.0, 1.5, 2.0, 0.75, 1.25, 1.0, 0.5], dtype=np.float32)


def write_graph(path, n=N, seed=0, unit=0.25, edges=None):
    """
    A graph.npz like build_graph.py's: n neurons in eight types, six random
    outgoing synapses each, three quarters excitatory, weights count * unit
    with count in 1..8. unit=0.25 keeps every sum exact in float32 (see the
    module docstring); unit=0.275 is the real graph's quantum. `edges` gives
    an explicit (pre, post, weight) list instead, for graphs built by hand.
    """
    if edges is None:
        rng = np.random.default_rng(seed)
        pre = np.repeat(np.arange(n), 6)
        post = rng.integers(0, n, size=pre.size)
        keep = pre != post
        pre, post = pre[keep], post[keep]
        count = rng.integers(1, 9, size=pre.size).astype(np.float32)
        sign = np.where(rng.random(n) < 0.75, 1.0, -1.0).astype(np.float32)
        w = (count * unit * sign[pre]).astype(np.float32)
    else:
        pre, post, w = (np.asarray(x) for x in zip(*edges))
        pre, post, w = pre.astype(np.int64), post.astype(np.int64), w.astype(np.float32)
        sign = np.ones(n, dtype=np.float32)
    W = sp.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    typ = np.array([TYPES[i % len(TYPES)] for i in range(n)], dtype=str)
    blank = np.array([""] * n, dtype=str)
    np.savez(
        path,
        data=W.data.astype(np.float32), indices=W.indices.astype(np.int32),
        indptr=W.indptr.astype(np.int32), shape=np.array(W.shape, dtype=np.int64),
        bodies=np.arange(n, dtype=np.int64) + 1000, sign=sign,
        types=typ, superclass=blank, subclass=blank, receptor=blank, fru=blank,
        nt=np.where(sign > 0, "acetylcholine", "gaba").astype(str),
    )
    return path


def graph_path(**kw):
    return write_graph(Path(tempfile.mkdtemp(prefix="flysim_gpu_")) / "graph.npz", **kw)


def brains(path=None, p=None, **kw):
    """The CPU brain and one GPU brain per available device, all on one graph."""
    path = path or graph_path(**kw)
    p = p or flysim.Params()
    cpu = flysim.FlyBrain(path, p)
    return cpu, {d: FlyBrainGPU(path, p, device=d) for d in DEVICES}


def drive():
    """A scalar-rate group and a per-neuron-rate group, so both forms are exercised every run."""
    return {tuple(range(0, 30)): 300.0,
            tuple(range(30, 60)): np.linspace(20.0, 400.0, 30).astype(np.float32)}


def record():
    return {"a": np.arange(0, 50), "b": np.arange(50, 200, 3), "c": [7, 8, 9]}


def assert_same_run(tc, a, b, rec=None, spikes=True):
    """Every returned quantity equal, arrays to the element, the state included."""
    rec = record() if rec is None else rec
    for k in rec:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        tc.assertEqual(b[k].shape, (len(rec[k]),))
    np.testing.assert_array_equal(a["_fired"], b["_fired"])
    tc.assertEqual(a["_mean_mv"], b["_mean_mv"])
    tc.assertEqual(a["_total_hz"], b["_total_hz"])
    tc.assertEqual(a["_spikes_per_sec"], b["_spikes_per_sec"])
    np.testing.assert_array_equal(a["_state"]["v"], b["_state"]["v"])
    np.testing.assert_array_equal(a["_state"]["refr"], b["_state"]["refr"])
    if spikes:
        tc.assertEqual(len(a["_spikes"]), len(b["_spikes"]))
        for step, (x, y) in enumerate(zip(a["_spikes"], b["_spikes"])):
            np.testing.assert_array_equal(x, y, err_msg=f"step {step}")


class SameDrawsSameSpikes(unittest.TestCase):
    """With the same uniform draws the GPU fires exactly the CPU's neurons, every step."""

    def test_same_seed_is_the_same_run(self):
        """The default path: a numpy generator per run, so seed 7 means the same numbers on both."""
        cpu, gpus = brains()
        for gains in (None, GAINS):
            a = cpu.run(drive(), 300, gains=gains, record=record(), seed=7, spike_log=True)
            self.assertGreater(len(a["_fired"]), 20)          # the graph is alive
            self.assertGreater(a["_spikes_per_sec"], 100.0)
            for dev, fb in gpus.items():
                with self.subTest(device=dev, gains=gains is not None):
                    b = fb.run(drive(), 300, gains=gains, record=record(), seed=7, spike_log=True)
                    assert_same_run(self, a, b)
                    self.assertEqual(a["_state"]["rng"], b["_state"]["rng"])

    def test_injected_draws_reproduce_the_cpu(self):
        """
        The exactness path the spec asks for: a callable hands both simulators
        the same draws, one call per step of exactly the driven count, and the
        CPU is handed it by standing in for numpy's default_rng.
        """
        class Draws:
            def __init__(self, seed):
                self.rng, self.calls = np.random.default_rng(seed), []

            def __call__(self, k):
                self.calls.append(int(k))
                return self.rng.random(k)

        class FakeRng:
            def __init__(self, draws):
                self.random = draws
                self.bit_generator = types.SimpleNamespace(state={"fake": True})

        cpu, gpus = brains()
        n_ext = 60
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                d_gpu = Draws(2024)
                b = fb.run(drive(), 150, gains=GAINS, record=record(), seed=0,
                           spike_log=True, random_source=d_gpu)
                d_cpu = Draws(2024)
                with mock.patch.object(np.random, "default_rng", lambda seed=None: FakeRng(d_cpu)):
                    a = cpu.run(drive(), 150, gains=GAINS, record=record(), seed=0, spike_log=True)
                self.assertEqual(d_cpu.calls, [n_ext] * 150)
                self.assertEqual(d_gpu.calls, d_cpu.calls)
                assert_same_run(self, a, b)

    def test_a_numpy_generator_as_the_source_is_carried_in_the_state(self):
        cpu, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                g = np.random.default_rng(5)
                b = fb.run(drive(), 50, record=record(), seed=99, random_source=g, spike_log=True)
                a = cpu.run(drive(), 50, record=record(), seed=5, spike_log=True)
                assert_same_run(self, a, b)
                self.assertEqual(b["_state"]["rng"], g.bit_generator.state)

    def test_no_drive_draws_nothing_and_stays_at_rest(self):
        """
        A plain function that raises, not a Mock: a Mock has every attribute,
        so it once passed for a numpy Generator and its .random child took
        the calls, which left the assertion below with nothing to catch.
        """
        cpu, gpus = brains()
        calls = []

        def never(k):
            calls.append(int(k))
            raise AssertionError("no draw expected without drive")

        a = cpu.run({}, 40, record=record(), seed=3, spike_log=True)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                b = fb.run({}, 40, record=record(), seed=3, spike_log=True, random_source=never)
                assert_same_run(self, a, b)
                self.assertEqual(b["_mean_mv"], -52.0)
                self.assertEqual(len(b["_fired"]), 0)
                self.assertEqual(calls, [])
                # the untouched generator for seed 3 is what the state carries
                self.assertEqual(b["_state"]["rng"], np.random.default_rng(3).bit_generator.state)
                self.assertEqual(b["_state"]["rng"], a["_state"]["rng"])

    def test_a_random_source_must_be_a_generator_or_a_callable(self):
        """Duck typing let a Mock through as a Generator; the check is now isinstance, and anything else is a TypeError."""
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                for bad in (3, "rng", np.random.RandomState(0)):
                    with self.assertRaises(TypeError):
                        fb.run(drive(), 2, random_source=bad)
                g = np.random.default_rng(4)
                self.assertEqual(fb.run(drive(), 2, random_source=g)["_state"]["rng"], g.bit_generator.state)
                self.assertNotEqual(g.bit_generator.state, np.random.default_rng(4).bit_generator.state)

    def test_other_membrane_parameters_are_honoured(self):
        """decay and refr_steps come from p, as on the CPU; a faster, briefer cell still matches."""
        p = flysim.Params()
        p.tau_m, p.refractory, p.v_thresh = 8.0, 0.6, -47.0
        cpu, gpus = brains(p=p)
        self.assertEqual(cpu.refr_steps, 3)
        a = cpu.run(drive(), 200, gains=GAINS, record=record(), seed=1, spike_log=True)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                self.assertEqual(fb.refr_steps, 3)
                self.assertEqual(fb.decay, cpu.decay)
                b = fb.run(drive(), 200, gains=GAINS, record=record(), seed=1, spike_log=True)
                assert_same_run(self, a, b)


class Batched(unittest.TestCase):
    """run_batch's columns are the separate runs, seed for seed and spike for spike."""

    def test_batch_equals_separate_runs(self):
        cpu, gpus = brains()
        other = {tuple(range(100, 140)): 250.0}
        drives = [drive(), drive(), other, {}]
        seeds = [3, 11, 3, 5]
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                outs = fb.run_batch(drives, 120, gains=GAINS, record=record(), seeds=seeds, spike_log=True)
                self.assertEqual(len(outs), 4)
                for d, s, o in zip(drives, seeds, outs):
                    one = fb.run(d, 120, gains=GAINS, record=record(), seed=s, spike_log=True)
                    assert_same_run(self, one, o)
                    ref = cpu.run(d, 120, gains=GAINS, record=record(), seed=s, spike_log=True)
                    assert_same_run(self, ref, o)
                # two columns with the same drive and different seeds are different runs
                self.assertFalse(all(np.array_equal(x, y)
                                     for x, y in zip(outs[0]["_spikes"], outs[1]["_spikes"])))
                # the column without drive drew nothing: its generator is still the seeded one
                self.assertEqual(outs[3]["_state"]["rng"], np.random.default_rng(5).bit_generator.state)
                self.assertEqual(outs[0]["_state"]["rng"],
                                 cpu.run(drive(), 120, gains=GAINS, seed=3)["_state"]["rng"])

    def test_one_drive_for_every_column(self):
        cpu, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                outs = fb.run_batch(drive(), 80, record=record(), seeds=[4, 5, 6])
                self.assertEqual(len(outs), 3)
                for s, o in zip([4, 5, 6], outs):
                    assert_same_run(self, cpu.run(drive(), 80, record=record(), seed=s), o, spikes=False)
                by_int = fb.run_batch([drive()] * 3, 80, record=record(), seeds=4)
                for a, b in zip(outs, by_int):
                    assert_same_run(self, a, b, spikes=False)

    def test_states_per_column(self):
        cpu, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                st = [cpu.run(drive(), 30, record=record(), seed=s)["_state"] for s in (1, 2)]
                outs = fb.run_batch([drive(), drive(), drive()], 30, record=record(),
                                    seeds=[0, 0, 9], states=[st[0], st[1], None], spike_log=True)
                for s, o in zip(st, outs[:2]):
                    assert_same_run(self, cpu.run(drive(), 30, record=record(), state=s, spike_log=True), o)
                assert_same_run(self, cpu.run(drive(), 30, record=record(), seed=9, spike_log=True), outs[2])

    def test_mismatched_column_counts_are_refused(self):
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                with self.assertRaises(ValueError):
                    fb.run_batch([drive(), drive()], 5, seeds=[1])
                with self.assertRaises(ValueError):
                    fb.run_batch([drive(), drive()], 5, states=[None])


class StateCarry(unittest.TestCase):
    """Chunked equals whole on the device, and a state crosses between the simulators."""

    def chained(self, fb, chunks, seed=7, state=None, **kw):
        outs = []
        for k in chunks:
            r = fb.run(drive(), k, gains=GAINS, record=record(), seed=seed, spike_log=True, state=state, **kw)
            outs.append(r)
            state = r["_state"]
        return outs

    def merged(self, outs):
        secs = sum(len(o["_spikes"]) for o in outs) * 0.2 / 1000.0
        counts = {k: sum(np.rint(o[k] * (len(o["_spikes"]) * 0.2 / 1000.0)) for o in outs) for k in record()}
        fired = np.unique(np.concatenate([o["_fired"] for o in outs]))
        return counts, fired, [f for o in outs for f in o["_spikes"]], secs

    def test_chunked_equals_whole_on_the_device(self):
        _, gpus = brains()
        for dev, fb in gpus.items():
            for chunks in ([30, 30, 30, 30], [7, 50, 63], [1] * 25):
                with self.subTest(device=dev, chunks=chunks):
                    n = sum(chunks)
                    whole = fb.run(drive(), n, gains=GAINS, record=record(), seed=7, spike_log=True)
                    parts = self.chained(fb, chunks)
                    counts, fired, log, secs = self.merged(parts)
                    self.assertEqual(len(log), n)
                    for step, (a, b) in enumerate(zip(whole["_spikes"], log)):
                        np.testing.assert_array_equal(a, b, err_msg=f"step {step}")
                    np.testing.assert_array_equal(whole["_fired"], fired)
                    for k in record():
                        np.testing.assert_array_equal(np.rint(whole[k] * secs), counts[k], err_msg=k)
                    self.assertEqual(whole["_mean_mv"], parts[-1]["_mean_mv"])
                    np.testing.assert_array_equal(whole["_state"]["v"], parts[-1]["_state"]["v"])
                    np.testing.assert_array_equal(whole["_state"]["refr"], parts[-1]["_state"]["refr"])
                    self.assertEqual(whole["_state"]["rng"], parts[-1]["_state"]["rng"])
                    self.assertGreater(sum(len(f) for f in log), 0)

    def test_a_cpu_state_continues_on_the_gpu_and_back(self):
        cpu, gpus = brains()
        whole = cpu.run(drive(), 120, gains=GAINS, record=record(), seed=7, spike_log=True)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                first = cpu.run(drive(), 40, gains=GAINS, record=record(), seed=7, spike_log=True)
                second = fb.run(drive(), 40, gains=GAINS, record=record(), state=first["_state"], spike_log=True)
                third = cpu.run(drive(), 40, gains=GAINS, record=record(), state=second["_state"], spike_log=True)
                log = first["_spikes"] + second["_spikes"] + third["_spikes"]
                for step, (a, b) in enumerate(zip(whole["_spikes"], log)):
                    np.testing.assert_array_equal(a, b, err_msg=f"step {step}")
                np.testing.assert_array_equal(whole["_state"]["v"], third["_state"]["v"])
                np.testing.assert_array_equal(whole["_state"]["refr"], third["_state"]["refr"])
                self.assertEqual(whole["_state"]["rng"], third["_state"]["rng"])

    def test_the_state_has_the_cpu_shape(self):
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                st = fb.run(drive(), 10, record=record(), seed=1)["_state"]
                self.assertEqual(set(st), {"v", "refr", "rng"})
                self.assertIsInstance(st["v"], np.ndarray)
                self.assertEqual((st["v"].shape, st["v"].dtype), ((N,), np.float32))
                self.assertEqual((st["refr"].shape, st["refr"].dtype), ((N,), np.int32))
                self.assertEqual(st["rng"]["bit_generator"], "PCG64")

    def test_the_state_passed_in_is_not_modified_nor_aliased(self):
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                st = fb.run(drive(), 40, record=record(), seed=2)["_state"]
                before = copy.deepcopy(st)
                nxt = fb.run(drive(), 40, record=record(), state=st)["_state"]
                np.testing.assert_array_equal(st["v"], before["v"])
                np.testing.assert_array_equal(st["refr"], before["refr"])
                self.assertEqual(st["rng"], before["rng"])
                self.assertFalse(np.shares_memory(st["v"], nxt["v"]))
                self.assertFalse(np.shares_memory(st["refr"], nxt["refr"]))

    def test_seed_is_ignored_with_a_state_and_a_wrong_size_is_refused(self):
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                st = fb.run(drive(), 20, record=record(), seed=2)["_state"]
                a = fb.run(drive(), 20, record=record(), state=st, seed=1, spike_log=True)
                b = fb.run(drive(), 20, record=record(), state=st, seed=999, spike_log=True)
                assert_same_run(self, a, b)
                with self.assertRaises(ValueError):
                    fb.run(drive(), 5, record=record(), state=dict(st, v=st["v"][:-1]))

    def test_a_state_and_a_generator_source_together_are_refused(self):
        """
        A state carries its own generator. A Generator handed in as well used
        to replace it silently, so the run continued the membranes but not
        the stream and no longer matched the CPU's continuation. It is now a
        ValueError; a plain callable is still allowed, and the state's
        generator is then carried untouched.
        """
        cpu, gpus = brains()
        st = cpu.run(drive(), 20, gains=GAINS, record=record(), seed=1)["_state"]
        whole = cpu.run(drive(), 40, gains=GAINS, record=record(), seed=1, spike_log=True)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                with self.assertRaises(ValueError):
                    fb.run(drive(), 20, state=st, random_source=np.random.default_rng(77))
                with self.assertRaises(ValueError):
                    fb.run_batch([drive(), drive()], 20, states=[None, st],
                                 random_sources=[None, np.random.default_rng(77)])
                # the callable form: the state's own stream, handed in by hand,
                # continues the whole run; the state's generator stays untouched
                g = np.random.default_rng()
                g.bit_generator.state = st["rng"]
                b = fb.run(drive(), 20, gains=GAINS, record=record(), state=st, spike_log=True,
                           random_source=g.random)
                for step, (x, y) in enumerate(zip(whole["_spikes"][20:], b["_spikes"])):
                    np.testing.assert_array_equal(x, y, err_msg=f"step {step}")
                np.testing.assert_array_equal(whole["_state"]["v"], b["_state"]["v"])
                self.assertEqual(b["_state"]["rng"], st["rng"])
                self.assertEqual(g.bit_generator.state, whole["_state"]["rng"])


class LearnedWeights(unittest.TestCase):
    """What mushroom.apply() writes into wdata is what the next run steps on."""

    # 0 -> 1 and 0 -> 2 strong enough to fire on their own (7 mV gap), 2 -> 3 the
    # same, 1 -| 4 inhibitory, and neuron 5 alone. Driven at 5000 Hz neuron 0 is
    # kicked every step.
    EDGES = [(0, 1, 8.0), (0, 2, 8.0), (2, 3, 8.0), (1, 4, -3.0), (5, 4, 0.5)]

    def hand_graph(self):
        return graph_path(n=6, edges=self.EDGES)

    def rates(self, fb, **kw):
        r = fb.run({(0,): 5000.0}, 100, record={"n": np.arange(6)}, seed=0, **kw)
        return r["n"]

    def test_a_silenced_synapse_no_longer_excites_its_target(self):
        path = self.hand_graph()
        cpu = flysim.FlyBrain(path)
        for dev in DEVICES:
            with self.subTest(device=dev):
                fb = FlyBrainGPU(path, device=dev)
                before = self.rates(fb)
                np.testing.assert_array_equal(before, self.rates(cpu))
                self.assertTrue(before[1] > 0 and before[2] > 0 and before[3] > 0)
                # the way mushroom.apply() writes: positions in wdata, assigned in place
                k = fb.indptr[0] + np.flatnonzero(fb.indices[fb.indptr[0]:fb.indptr[1]] == 1)[0]
                self.assertEqual(fb.wdata[k], 8.0)
                fb.wdata[[k]] = np.array([0.0], dtype=np.float32) * np.float32(1.0)
                self.assertNotEqual(fb._pushed_version, fb._wdata_version)
                after = self.rates(fb)
                self.assertTrue(fb.weights_are_current())
                self.assertEqual(after[1], 0.0)
                self.assertEqual(after[2], before[2])
                self.assertEqual(after[3], before[3])
                cpu.wdata[k] = 0.0                                       # the CPU on the same weights
                np.testing.assert_array_equal(after, self.rates(cpu))
                cpu.wdata[k] = 8.0
                fb.wdata[k] = 8.0
                np.testing.assert_array_equal(self.rates(fb), before)

    def test_the_real_learning_circuit_writes_are_seen(self):
        """mushroom.MushroomBody on a FlyBrainGPU: a dopamine event, then apply(), then a run on the depressed weights."""
        from mushroom import MushroomBody
        tmp = Path(tempfile.mkdtemp(prefix="flysim_gpu_mb_"))
        sides = tmp / "mb_sides.json"
        sides.write_text(json.dumps({"types": {"MBON01": {"side": "PAM"}, "MBON11": {"side": "PPL1"}}}),
                         encoding="utf-8")
        path = graph_path(seed=4)
        cpu = flysim.FlyBrain(path)
        a = cpu.run(drive(), 100, record=record(), seed=3, spike_log=True)   # untouched weights
        for dev in DEVICES:
            with self.subTest(device=dev):
                fb = FlyBrainGPU(path, device=dev)
                mb = MushroomBody(fb, sides=sides, store=tmp / dev / "mb_gains.v2.npz",
                                  clock=lambda: 1e9, calibration="test")
                self.assertGreater(len(mb.pos), 0)
                mb.observe(fb.where(type_re=r"^KC"))
                self.assertGreater(mb.dopamine(+1, 1.0), 0)
                pushed = fb._pushed_version
                mb.apply()
                self.assertGreater(fb._wdata_version, pushed)
                self.assertFalse(np.array_equal(fb.wdata[mb.pos], mb.base))
                b = fb.run(drive(), 100, record=record(), seed=3, spike_log=True)     # depressed weights
                self.assertTrue(fb.weights_are_current())
                cpu.wdata[mb.pos] = fb.wdata[mb.pos]
                c = cpu.run(drive(), 100, record=record(), seed=3, spike_log=True)
                cpu.wdata[mb.pos] = mb.base
                assert_same_run(self, c, b)
                self.assertFalse(np.array_equal(a["_state"]["v"], b["_state"]["v"]))

    def test_every_write_path_numpy_routes_through_python_is_counted(self):
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                def bumped(write):
                    was = fb._wdata_version
                    write()
                    return fb._wdata_version > was

                self.assertTrue(bumped(lambda: fb.wdata.__setitem__(slice(0, 10), 0.0)))
                self.assertTrue(bumped(lambda: fb.wdata.__imul__(0.5)))
                self.assertTrue(bumped(lambda: np.multiply(fb.wdata, 2.0, out=fb.wdata)))
                self.assertTrue(bumped(lambda: fb.wdata[5:9].__iadd__(1.0)))      # a view
                self.assertTrue(bumped(lambda: fb.wdata.fill(0.25)))
                self.assertTrue(bumped(lambda: fb.wdata.put([1, 2], [0.5, 0.5])))
                self.assertTrue(bumped(lambda: np.add.at(fb.wdata, [3], 0.25)))
                self.assertTrue(bumped(lambda: setattr(fb, "wdata", np.zeros(len(fb.wdata), np.float32))))
                self.assertFalse(bumped(lambda: fb.wdata * 2.0))                    # a read
                self.assertIs(type(fb.wdata * 2.0), np.ndarray)
                self.assertFalse(fb.weights_are_current())
                fb.run({}, 1)
                self.assertTrue(fb.weights_are_current())
                # the window and its views are read-only to numpy, yet the
                # counted paths above went through: each unlocks for one write
                self.assertFalse(fb.wdata.flags.writeable)
                self.assertFalse(fb.wdata[5:9].flags.writeable)
                self.assertIsInstance(fb.wdata[5:9], flysim_gpu._TrackedWeights)

    def test_an_untracked_numpy_write_is_refused_not_stale(self):
        """
        np.copyto, .flat, putmask, place, sort and assignment through a plain
        view never reach Python, so the counter cannot see them. They used to
        change the host weights and leave the device on the old ones: numpy
        fired the synapse the host said was silenced and torch fired it
        anyway. The window is read-only now, so numpy refuses every one of
        them, nothing changes, and the device run still equals the numpy run.
        """
        path = self.hand_graph()
        cpu = flysim.FlyBrain(path)
        n = len(cpu.wdata)
        k = cpu.indptr[0] + np.flatnonzero(cpu.indices[cpu.indptr[0]:cpu.indptr[1]] == 1)[0]
        before = self.rates(cpu)
        zeroed = np.where(np.arange(n) == k, 0.0, cpu.wdata).astype(np.float32)
        refused = {
            "copyto": lambda fb: np.copyto(fb.wdata, zeroed),
            "copyto where": lambda fb: np.copyto(fb.wdata, 0.0, where=np.arange(n) == k),
            "flat": lambda fb: fb.wdata.flat.__setitem__(int(k), 0.0),
            "putmask": lambda fb: np.putmask(fb.wdata, np.arange(n) == k, 0.0),
            "place": lambda fb: np.place(fb.wdata, np.arange(n) == k, 0.0),
            "sort": lambda fb: fb.wdata.sort(),
            "plain view": lambda fb: fb.wdata.view(np.ndarray).__setitem__(k, 0.0),
            "asarray": lambda fb: np.asarray(fb.wdata).__setitem__(k, 0.0),
            "ufunc out= plain view": lambda fb: np.multiply(fb.wdata, 0.0, out=fb.wdata.view(np.ndarray)),
            "base-class setitem": lambda fb: np.ndarray.__setitem__(fb.wdata, k, 0.0),
            "memoryview": lambda fb: memoryview(fb.wdata).__setitem__(int(k), 0.0),
        }
        # np.put and np.ndarray.fill are not on that list: numpy routes them
        # to the methods, which the window intercepts, so they are counted
        counted = {
            "np.put": lambda fb: np.put(fb.wdata, [k], [0.0]),
            "setitem": lambda fb: fb.wdata.__setitem__(k, 0.0),
            "np.multiply at": lambda fb: np.multiply.at(fb.wdata, [k], 0.0),
        }
        for dev in DEVICES:
            for name, write in refused.items():
                with self.subTest(device=dev, path=name):
                    fb = FlyBrainGPU(path, device=dev)               # fresh, so one path cannot hide another
                    version = fb._wdata_version
                    with self.assertRaises((ValueError, TypeError)):
                        write(fb)
                    self.assertEqual(fb.wdata[k], 8.0)
                    self.assertEqual(fb._wdata_version, version)
                    self.assertTrue(fb.weights_are_current())
                    np.testing.assert_array_equal(self.rates(fb), before)
                    np.testing.assert_array_equal(flysim.FlyBrain.run(fb, {(0,): 5000.0}, 100,
                                                                      record={"n": np.arange(6)}, seed=0)["n"], before)
            for name, write in counted.items():
                with self.subTest(device=dev, path=name):
                    fb = FlyBrainGPU(path, device=dev)
                    version = fb._wdata_version
                    write(fb)
                    self.assertEqual(fb.wdata[k], 0.0)
                    self.assertGreater(fb._wdata_version, version)
                    self.assertFalse(fb.wdata.flags.writeable)         # locked again after the write
                    self.assertEqual(self.rates(fb)[1], 0.0)
                    self.assertTrue(fb.weights_are_current())

    def test_a_copy_of_the_weights_is_plain_and_uncounted(self):
        """
        mushroom keeps base = fb.wdata[pos].copy() and bench_gpu the same; a
        fancy-index result used to keep the brain as its owner, so writing
        to such a private copy pushed 10 M weights for nothing. A copy is
        now a plain, writeable ndarray that the brain never hears about.
        """
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                fb.run({}, 1)
                version = fb._wdata_version
                for name, c in {"fancy": fb.wdata[[0, 1, 2]], "fancy copy": fb.wdata[[0, 1]].copy(),
                                "copy": fb.wdata.copy(), "bool mask": fb.wdata[fb.wdata > 0],
                                "ufunc": fb.wdata * 2.0}.items():
                    with self.subTest(kind=name):
                        self.assertIs(type(c), np.ndarray)
                        self.assertTrue(c.flags.writeable)
                        self.assertFalse(np.shares_memory(c, fb.wdata))
                        c[0] = 99.0
                        self.assertEqual(fb._wdata_version, version)
                self.assertNotEqual(fb.wdata[0], 99.0)
                self.assertTrue(fb.weights_are_current())
                self.assertIsInstance(fb.wdata[3], np.floating)

    def test_a_raw_buffer_write_needs_an_explicit_push(self):
        """The raw path: a plain view the caller unlocks on purpose, or fb.wdata.base; both are invisible until push_weights()."""
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                w = fb.wdata.view(np.ndarray)
                with self.assertRaises(ValueError):
                    w[0] = 123.0                          # a plain view is refused as it comes
                w.flags.writeable = True                  # the caller unlocks it: numpy allows that
                w[0] = 123.0                              # because the buffer beneath is writeable
                self.assertEqual(fb.wdata[0], 123.0)
                self.assertEqual(fb._pushed_version, fb._wdata_version)
                self.assertFalse(fb.weights_are_current())
                fb.push_weights()
                self.assertTrue(fb.weights_are_current())
                fb.wdata.base[1] = 124.0                  # the buffer itself, the same deal
                self.assertFalse(fb.weights_are_current())
                fb.push_weights()
                self.assertTrue(fb.weights_are_current())
                self.assertEqual(fb.run({}, 1)["_mean_mv"], -52.0)


class Rewired(unittest.TestCase):
    """
    The device operator follows the CSC arrays. lesion.lesioned makes a
    copy.copy of the brain and replaces W, indptr, indices and wdata on it;
    the copy used to share the original's device buffers and its frozen
    structure, so a lesion either died in copy_ (fewer synapses) or ran the
    new weights through the old wiring (the same count), and its push
    clobbered the original's staging buffer either way.
    """

    def lesioned(self, fb, keep):
        """The real consumer: copy.copy, then W, indptr, indices and wdata replaced on the copy."""
        try:
            from lesion import lesioned
        except ImportError:
            self.skipTest("lesion.py (the launcher's lesion helper) is not part of this tree")
        return lesioned(fb, keep)

    def test_the_lesion_pattern_runs_on_the_new_wiring(self):
        cpu, gpus = brains()
        keep = np.ones(N, dtype=bool)
        keep[::3] = False
        ref = self.lesioned(cpu, keep)
        a = ref.run(drive(), 150, gains=GAINS, record=record(), seed=5, spike_log=True)
        untouched = cpu.run(drive(), 150, gains=GAINS, record=record(), seed=5, spike_log=True)
        self.assertFalse(np.array_equal(a["_state"]["v"], untouched["_state"]["v"]))
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                before = fb.run(drive(), 150, gains=GAINS, record=record(), seed=5, spike_log=True)
                lb = self.lesioned(fb, keep)
                self.assertLess(len(lb.wdata), len(fb.wdata))
                b = lb.run(drive(), 150, gains=GAINS, record=record(), seed=5, spike_log=True)
                assert_same_run(self, a, b)
                assert_same_run(self, flysim.FlyBrain.run(lb, drive(), 150, gains=GAINS, record=record(),
                                                          seed=5, spike_log=True), b)
                self.assertTrue(lb.weights_are_current())
                # the original is as it was: its own buffers, its own run
                self.assertTrue(fb.weights_are_current())
                self.assertIsNot(lb._wdata_dev, fb._wdata_dev)
                self.assertIsNot(lb._M, fb._M)
                assert_same_run(self, before, fb.run(drive(), 150, gains=GAINS, record=record(), seed=5,
                                                     spike_log=True))
                assert_same_run(self, untouched, before)

    def test_a_same_count_rewiring_is_seen(self):
        """Rows reversed: the same number of synapses on other targets, which the frozen permutation used to hide."""
        cpu, gpus = brains()
        Wr = cpu.W.tocsr()[::-1].tocsc()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                lb = copy.copy(fb)
                lb.W = Wr
                lb.indptr, lb.indices, lb.wdata = Wr.indptr, Wr.indices, Wr.data.astype(np.float32)
                self.assertEqual(len(lb.wdata), len(fb.wdata))
                self.assertFalse(lb.weights_are_current())
                b = lb.run(drive(), 100, gains=GAINS, record=record(), seed=2, spike_log=True)
                a = flysim.FlyBrain.run(lb, drive(), 100, gains=GAINS, record=record(), seed=2, spike_log=True)
                assert_same_run(self, a, b)
                self.assertTrue(lb.weights_are_current())
                self.assertFalse(np.array_equal(
                    b["_state"]["v"], fb.run(drive(), 100, gains=GAINS, record=record(), seed=2)["_state"]["v"]))

    def test_the_copy_owns_its_weights_and_device_state(self):
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                lb = copy.copy(fb)
                self.assertIsInstance(lb, FlyBrainGPU)
                self.assertFalse(np.shares_memory(lb.wdata, fb.wdata))
                np.testing.assert_array_equal(lb.wdata, fb.wdata)
                v_fb, v_lb = fb._wdata_version, lb._wdata_version
                lb.wdata[0] = 0.5                           # counted on the copy only
                self.assertEqual(fb._wdata_version, v_fb)
                self.assertNotEqual(lb._wdata_version, v_lb)
                fb.wdata[1] = 0.5                           # and the other way round
                self.assertEqual(lb.wdata[1], fb.wdata[0])
                self.assertNotEqual(lb.wdata[1], fb.wdata[1])
                lb.run({}, 1)
                fb.run({}, 1)
                self.assertTrue(lb.weights_are_current() and fb.weights_are_current())
                self.assertIsNot(lb._wdata_dev, fb._wdata_dev)
                self.assertIsNot(lb._M, fb._M)
                # the copy's device state was built on its first run, from its own wdata
                self.assertEqual(float(lb._wdata_dev[0]), 0.5)
                self.assertNotEqual(float(fb._wdata_dev[0]), 0.5)

    def test_an_in_place_write_to_the_index_arrays_is_refused(self):
        """
        Only the setters of indptr and indices mark the operator stale, so a
        rewiring written into them in place (the same count, other targets)
        used to run the device on the old targets while numpy on the same
        object took the new ones, and weights_are_current() saw nothing. The
        arrays are read-only views now: refused, like the untracked weight
        writes. scipy's own arrays on fb.W stay writeable, and a replacement
        through the setters is the tracked way (the tests above).
        """
        cpu, gpus = brains()
        self.assertTrue(cpu.indices.flags.writeable)                 # the numpy brain is as it was
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                fb.run({}, 1)
                for name in ("indices", "indptr"):
                    arr = getattr(fb, name)
                    with self.assertRaises(ValueError, msg=name):
                        arr[:] = (arr + 1) % N
                    with self.assertRaises(ValueError, msg=name):
                        arr[0] = arr[0]
                    self.assertFalse(getattr(fb, name)[:3].flags.writeable, name)
                np.testing.assert_array_equal(fb.indices, cpu.indices)
                np.testing.assert_array_equal(fb.indptr, cpu.indptr)
                self.assertTrue(fb.W.indices.flags.writeable)        # the raw path, scipy's arrays
                self.assertTrue(fb.weights_are_current())
                assert_same_run(self, cpu.run(drive(), 60, gains=GAINS, record=record(), seed=4, spike_log=True),
                                fb.run(drive(), 60, gains=GAINS, record=record(), seed=4, spike_log=True))
                # the tracked way: replaced through the setters, the operator follows
                lb = copy.copy(fb)
                Wp = cpu.W.tocsr()[::-1].tocsc()
                lb.W, lb.indptr, lb.indices, lb.wdata = Wp, Wp.indptr, Wp.indices, Wp.data.astype(np.float32)
                self.assertFalse(lb.indices.flags.writeable)
                self.assertTrue(Wp.indices.flags.writeable)          # the caller's array is untouched
                assert_same_run(self, flysim.FlyBrain.run(lb, drive(), 60, gains=GAINS, record=record(), seed=4,
                                                          spike_log=True),
                                lb.run(drive(), 60, gains=GAINS, record=record(), seed=4, spike_log=True))

    def test_inconsistent_arrays_are_refused_by_name(self):
        _, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                lb = copy.copy(fb)
                lb.wdata = fb.wdata[:-1]                    # one synapse short of the wiring
                with self.assertRaises(ValueError) as c:
                    lb.run({}, 1)
                self.assertIn("wdata", str(c.exception))
                self.assertIn("indptr", str(c.exception))
                lb = copy.copy(fb)
                lb.indptr = fb.indptr[:-1]
                with self.assertRaises(ValueError):
                    lb.push_weights()
                # a whole-array replacement of the same count is a push, not a rebuild
                lb = copy.copy(fb)
                lb.run({}, 1)
                perm = lb._perm
                lb.wdata = np.zeros(len(fb.wdata), np.float32)
                lb.run({}, 1)
                self.assertIs(lb._perm, perm)
                self.assertTrue(lb.weights_are_current())


class SpikeLog(unittest.TestCase):
    """
    The spike log is staged on the device in chunks. On the CPU device .cpu()
    returned the live buffer itself, so every chunk copied out was a view of
    it and later steps overwrote the earlier chunks' rows: a log longer than
    one chunk reported the last steps twice. Now every chunk is a copy on
    both devices, checked here across many boundaries by shrinking the chunk
    and once at the real chunk size.
    """

    def test_a_log_longer_than_one_chunk_is_the_cpus_log(self):
        cpu, gpus = brains()
        drives = [drive(), drive(), {tuple(range(100, 140)): 250.0}, drive()]
        seeds = [3, 11, 3, 5]
        refs = [cpu.run(d, 123, gains=GAINS, record=record(), seed=s, spike_log=True) for d, s in zip(drives, seeds)]
        for dev, fb in gpus.items():
            for chunk in (1, 5, 123, 500):
                with self.subTest(device=dev, chunk=chunk):
                    with mock.patch.object(flysim_gpu, "LOG_BUFFER_BYTES", chunk * N * len(drives)):
                        outs = fb.run_batch(drives, 123, gains=GAINS, record=record(), seeds=seeds, spike_log=True)
                        one = fb.run(drive(), 123, gains=GAINS, record=record(), seed=3, spike_log=True)
                    for ref, out in zip(refs, outs):
                        assert_same_run(self, ref, out)
                    assert_same_run(self, refs[0], one)
                    self.assertEqual(len(one["_spikes"]), 123)

    def test_at_the_real_chunk_size(self):
        """n * B large enough that 64 MiB holds fewer steps than the run: two chunks, the second partial."""
        cpu, gpus = brains()
        B, steps = 1000, 400
        chunk = flysim_gpu.LOG_BUFFER_BYTES // (N * B)
        self.assertLess(chunk, steps)
        self.assertNotEqual(steps % chunk, 0)
        check = [0, chunk, B - 1]
        refs = {b: cpu.run(drive(), steps, gains=GAINS, record=record(), seed=b, spike_log=True) for b in check}
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                outs = fb.run_batch(drive(), steps, gains=GAINS, record=record(), seeds=list(range(B)), spike_log=True)
                self.assertEqual(len(outs), B)
                for b in check:
                    assert_same_run(self, refs[b], outs[b])
                    self.assertEqual(len(outs[b]["_spikes"]), steps)
                # the tell-tale of the aliasing: the first chunk's rows equal to the last chunk's
                first, last = outs[0]["_spikes"][:steps - chunk], outs[0]["_spikes"][chunk:]
                self.assertFalse(all(np.array_equal(x, y) for x, y in zip(first, last)))


class DriveForms(unittest.TestCase):
    """The drive dict is read exactly as the CPU reads it."""

    def test_scalar_and_per_neuron_rates_and_array_keys(self):
        cpu, gpus = brains()
        d = {(1, 2, 3): 400.0, (10, 11): np.array([100.0, 900.0]),
             tuple(range(20, 60)): np.full(40, 60.0), (70,): 0.0}
        a = cpu.run(d, 150, record=record(), seed=9, spike_log=True)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                assert_same_run(self, a, fb.run(d, 150, record=record(), seed=9, spike_log=True))

    def test_mismatched_lengths_raise_the_cpu_error(self):
        cpu, gpus = brains()
        bad = {(1, 2, 3): np.array([1.0, 2.0])}
        with self.assertRaises(ValueError) as c:
            cpu.run(bad, 1)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                with self.assertRaises(ValueError) as g:
                    fb.run(bad, 1)
                self.assertEqual(str(g.exception), str(c.exception))

    def test_a_neuron_driven_by_two_keys(self):
        """Any hit kicks it; the accumulate path handles the duplicate, the CPU's fancy assignment does."""
        cpu, gpus = brains()
        d = {tuple(range(0, 30)): 150.0, tuple(range(20, 50)): 150.0}
        a = cpu.run(d, 200, record=record(), seed=2, spike_log=True)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                assert_same_run(self, a, fb.run(d, 200, record=record(), seed=2, spike_log=True))

    def test_rates_above_the_step_are_clipped_to_certainty(self):
        cpu, gpus = brains()
        d = {(0, 1): 9000.0}
        a = cpu.run(d, 30, record=record(), seed=2, spike_log=True)
        self.assertTrue(all(0 in f and 1 in f for f in a["_spikes"][::11]))
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                assert_same_run(self, a, fb.run(d, 30, record=record(), seed=2, spike_log=True))


class Outputs(unittest.TestCase):
    """The dict a caller gets is the CPU's dict: keys, types, shapes, values."""

    def test_keys_types_and_values(self):
        cpu, gpus = brains()
        a = cpu.run(drive(), 100, record=record(), seed=8, spike_log=True)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                b = fb.run(drive(), 100, record=record(), seed=8, spike_log=True)
                self.assertEqual(set(a), set(b))
                assert_same_run(self, a, b)
                self.assertEqual(b["_fired"].dtype, a["_fired"].dtype)
                self.assertEqual(b["a"].dtype, a["a"].dtype)
                self.assertTrue(all(f.dtype == np.int32 for f in b["_spikes"]))
                self.assertIsInstance(b["_mean_mv"], float)
                self.assertNotIn("_spikes", fb.run(drive(), 10, record=record(), seed=8))
                self.assertEqual(fb.run(drive(), 10, seed=8).keys(),
                                 {"_total_hz", "_spikes_per_sec", "_fired", "_mean_mv", "_state"})

    def test_it_is_a_flybrain_with_the_same_annotations(self):
        cpu, gpus = brains()
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                self.assertIsInstance(fb, flysim.FlyBrain)
                for attr in ("n", "bodies", "types", "superclass", "subclass", "receptor", "fru", "nt",
                             "type_names", "type_code", "n_types", "body_to_i", "indptr", "indices",
                             "wdata", "W", "p", "decay", "refr_steps"):
                    self.assertTrue(hasattr(fb, attr), attr)
                np.testing.assert_array_equal(fb.where(type_re=r"^KC"), cpu.where(type_re=r"^KC"))
                np.testing.assert_array_equal(fb.wdata, cpu.wdata)
                np.testing.assert_array_equal(fb.indices, cpu.indices)
                self.assertEqual(fb.n_types, cpu.n_types)
                self.assertEqual(fb.device.type, dev)

    def test_the_realistic_quantum_still_agrees_on_a_short_window(self):
        """
        Weights in the real graph's 0.275 mV steps: not exact in float32, so
        this checks only a short window, where the sums stay identical in
        practice, and the statistical claim is left to RealBrain.
        """
        cpu, gpus = brains(unit=0.275)
        a = cpu.run(drive(), 40, gains=GAINS, record=record(), seed=6, spike_log=True)
        for dev, fb in gpus.items():
            with self.subTest(device=dev):
                b = fb.run(drive(), 40, gains=GAINS, record=record(), seed=6, spike_log=True)
                for step, (x, y) in enumerate(zip(a["_spikes"], b["_spikes"])):
                    np.testing.assert_array_equal(x, y, err_msg=f"step {step}")
                np.testing.assert_allclose(a["_state"]["v"], b["_state"]["v"], atol=1e-4)

    def test_open_picks_the_simulator(self):
        path = graph_path()
        self.assertIs(type(flysim_gpu.open("numpy", graph_path=path)), flysim.FlyBrain)
        for dev in DEVICES:
            fb = flysim_gpu.open(dev, graph_path=path)
            self.assertIsInstance(fb, FlyBrainGPU)
            self.assertEqual(fb.device.type, dev)

    @unittest.skipIf(torch.cuda.is_available(), "only meaningful without a GPU")
    def test_cuda_without_a_gpu_is_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            FlyBrainGPU(graph_path(), device="cuda")


def free_ram_gb():
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out) / (1024 * 1024)
    except Exception:
        return float("nan")


def eye_drive(fb):
    """
    The roamer's own eye on a uniform mid-grey frame, if pumpui's FlyEye can
    be built here (it needs data/body-annotations.feather and pandas); else
    the same two populations, L1 and L2, at the rates that frame would give.
    """
    try:
        from pumpui import FlyEye
        img = np.full((600, 800), 0.5, dtype=np.float32)
        return FlyEye(fb).look(img, 400, 300), "FlyEye on a uniform 0.5 frame"
    except Exception as exc:
        return ({tuple(fb.where(type_re=r"^L1$")): 90.0,
                 tuple(fb.where(type_re=r"^L2$")): 54.0}, f"L1/L2 uniform ({type(exc).__name__})")


@unittest.skipUnless(os.environ.get("FLY_GPU_REAL_BRAIN") == "1",
                     "set FLY_GPU_REAL_BRAIN=1 to load the connectome")
class RealBrain(unittest.TestCase):
    """
    The connectome, loaded once as a FlyBrainGPU, which is also the CPU brain:
    FlyBrain.run on it is the ground truth. 20 seeds each way with the
    calibrated gains and the roamer's eye drive; the per-population spike
    counts must agree within 3 SE of the difference and the number of cells
    that fired within 5 %.
    """

    POPS = {"DNa01": r"^DNa01$", "DNa02": r"^DNa02$", "MDN": r"^MDN$", "DNp09": r"^DNp09$",
            "KC": r"^KC", "MBON": r"^MBON"}
    SEEDS = list(range(20))
    STEPS = 60

    def test_rates_agree_within_noise(self):
        ram = free_ram_gb()
        print(f"\nfree RAM before load: {ram:.1f} GB")
        self.assertGreater(ram, 6.0, "need more than 6 GB free to load a FlyBrain")
        import calibration
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.perf_counter()
        fb = FlyBrainGPU(device=dev)
        print(f"loaded {fb.n:,} neurons / {len(fb.wdata):,} synapses on {dev} in {time.perf_counter() - t0:.1f} s")
        gains = calibration.gains_for(fb, calibration.CHOSEN)
        drive, how = eye_drive(fb)
        print(f"drive: {how}, {sum(len(k) for k in drive)} driven neurons")
        record = {k: fb.where(type_re=v) for k, v in self.POPS.items()}
        secs = self.STEPS * fb.p.dt / 1000.0

        t0 = time.perf_counter()
        cpu = [flysim.FlyBrain.run(fb, drive, self.STEPS, gains=gains, record=record, seed=s)
               for s in self.SEEDS]
        t_cpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        gpu = fb.run_batch(drive, self.STEPS, gains=gains, record=record, seeds=self.SEEDS)
        t_batch = time.perf_counter() - t0
        t0 = time.perf_counter()
        one = fb.run(drive, self.STEPS, gains=gains, record=record, seed=0)
        t_one = time.perf_counter() - t0
        print(f"CPU {len(self.SEEDS)} runs x {self.STEPS} steps: {t_cpu:.2f} s "
              f"({1000 * t_cpu / len(self.SEEDS) / self.STEPS:.2f} ms/step); "
              f"GPU batch of {len(self.SEEDS)}: {t_batch:.2f} s "
              f"({1000 * t_batch / len(self.SEEDS) / self.STEPS:.2f} ms/step/run, "
              f"{t_cpu / t_batch:.1f}x); GPU one run: {t_one:.2f} s")

        def counts(res, k):
            return np.array([float(np.asarray(r[k]).sum() * secs) for r in res])

        print(f"{'population':10} {'cells':>6} {'CPU mean':>10} {'GPU mean':>10} {'SE diff':>8} {'z':>6}")
        for k in self.POPS:
            a, b = counts(cpu, k), counts(gpu, k)
            se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            z = abs(a.mean() - b.mean()) / se if se > 0 else 0.0
            print(f"{k:10} {len(record[k]):6d} {a.mean():10.1f} {b.mean():10.1f} {se:8.2f} {z:6.2f}")
            self.assertLessEqual(abs(a.mean() - b.mean()), 3.0 * se + 1e-9, k)
        fa = np.array([len(r["_fired"]) for r in cpu], dtype=float)
        fg = np.array([len(r["_fired"]) for r in gpu], dtype=float)
        print(f"_fired     {'':6} {fa.mean():10.1f} {fg.mean():10.1f}   ratio {fg.mean() / fa.mean():.4f}")
        self.assertLess(abs(fg.mean() - fa.mean()) / fa.mean(), 0.05)
        # the same seed on both simulators is the same draws; how far the two
        # walk together before float32 summation order parts them is reported,
        # not asserted
        same = int(np.array_equal(one["_fired"], cpu[0]["_fired"]))
        print(f"seed 0 fired sets identical across simulators: {bool(same)} "
              f"({len(cpu[0]['_fired'])} vs {len(one['_fired'])} cells)")


if __name__ == "__main__":
    unittest.main()
