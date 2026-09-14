"""
Headless brain + eye + motor proof, for a throwaway cloud sandbox.

No browser, no server, no network, no chain. This loads the real 165,122-neuron
male-CNS connectome (built by build_graph.py from the CC-BY release), points the
fly's eye at a synthetic dark "table" of bright cards, steps the leaky
integrate-and-fire simulation, and reads the cursor + click straight out of the
descending neurons a fly actually walks with (DNa02 steer, DNa01 forward, MDN
reverse, DNp09 stop/click). It is the same closed loop roam.py runs, with the
live web page replaced by an image in memory - so it proves the sim, the
retinotopic eye sampling and the motor readout all work end to end, on a machine
that can touch nothing.

Run:  python sandbox_demo.py
Exits non-zero if the brain does not fire, the eye does not drive it, or the
motor produces no readout - so a green run is real evidence, not just "no crash".
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from flysim import FlyBrain
from flyeye import FlyPilot

ROOT = Path(__file__).parent
OUT = ROOT / "build" / "sandbox"
OUT.mkdir(parents=True, exist_ok=True)

H, W = 720, 1280          # the synthetic screen, in pixels
N_STEPS = 30              # control steps (each is a burst of brain time)
START = (W // 2, H // 2)  # cursor starts centre-screen


def synthetic_screen():
    """
    A dark page with a few bright rectangular 'cards' - the stand-in for a
    Pythia table of legal moves. Dark background because the fly's retina is a
    luminance map tuned against dark UI (README: light mode breaks it). Returns
    a float32 image in [0, 1], plus the card rectangles for the screenshot.
    """
    img = np.full((H, W), 0.05, dtype=np.float32)
    cards = [
        (140, 180, 360, 380),   # (x0, y0, x1, y1)
        (470, 160, 690, 360),
        (800, 200, 1020, 400),
        (300, 460, 560, 640),
        (720, 440, 980, 620),
    ]
    for (x0, y0, x1, y1) in cards:
        img[y0:y1, x0:x1] = 0.92
        # a dim inner border so there is structure inside each card too
        img[y0 + 8:y0 + 14, x0:x1] = 0.4
    return img, cards


def main():
    t0 = time.time()
    print("=" * 72)
    print("flycoinrh (stripped fork) - headless brain+eye+motor sandbox proof")
    print("=" * 72)

    graph = ROOT / "build" / "graph.npz"
    if not graph.exists():
        print(f"FAIL: {graph} missing - run build_graph.py first", file=sys.stderr)
        return 2

    print(f"loading connectome from {graph} ...")
    fb = FlyBrain()
    print(f"  brain: {fb.n:,} neurons, {fb.W.nnz:,} signed synapses, "
          f"{fb.n_types:,} cell types")
    if fb.n < 100_000:
        print(f"FAIL: expected ~165k neurons, got {fb.n}", file=sys.stderr)
        return 2

    print("wiring the eye and the descending-neuron motor readout ...")
    pilot = FlyPilot(fb)                      # loads body-annotations.feather
    motor = {k: int(len(v)) for k, v in pilot.motor.items()}
    print(f"  eye: {len(pilot.eye.on_idx):,} L1 (ON) + {len(pilot.eye.off_idx):,} "
          f"L2 (OFF) retinotopic columns")
    print(f"  motor neurons: {motor}")
    if len(pilot.eye.on_idx) == 0 or sum(motor.values()) == 0:
        print("FAIL: eye or motor population is empty", file=sys.stderr)
        return 2

    img, cards = synthetic_screen()
    cx, cy = START
    traj = [(float(cx), float(cy))]
    clicks = []
    rows = []
    total_visual = 0
    total_motor = 0

    print(f"\ndriving the fly for {N_STEPS} control steps across the {W}x{H} "
          f"screen ...\n")
    hdr = f"{'step':>4} {'cx':>6} {'cy':>6} {'dx':>7} {'dy':>7} {'click':>6} " \
          f"{'fired':>7} {'vis':>5} {'mot':>5} {'spk/s':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in range(N_STEPS):
        dx, dy, click, hz, info = pilot.step(img, int(cx), int(cy),
                                             seed=s, detail=True)
        cx = float(np.clip(cx + dx, 0, W - 1))
        cy = float(np.clip(cy + dy, 0, H - 1))
        traj.append((cx, cy))
        total_visual += info["visual"]
        total_motor += info["motor"]
        if click:
            clicks.append((cx, cy, s))
        rows.append({
            "step": s, "cx": round(cx, 1), "cy": round(cy, 1),
            "dx": round(float(dx), 2), "dy": round(float(dy), 2),
            "click": bool(click),
            "fired": info["firing"], "visual": info["visual"],
            "motor": info["motor"],
            "spikes_per_sec": round(info["spikes_per_sec"], 1),
            "hz": {k: round(float(v), 1) for k, v in hz.items()},
        })
        print(f"{s:>4} {cx:>6.0f} {cy:>6.0f} {dx:>7.2f} {dy:>7.2f} "
              f"{str(click):>6} {info['firing']:>7} {info['visual']:>5} "
              f"{info['motor']:>5} {info['spikes_per_sec']:>9.1f}")

    fired_any = any(r["fired"] > 0 for r in rows)
    moved = any(abs(r["dx"]) > 1e-6 or abs(r["dy"]) > 1e-6 for r in rows)
    dt = time.time() - t0

    print("\n" + "=" * 72)
    ok = fired_any and total_visual > 0 and moved
    summary = {
        "result": "PASS" if ok else "FAIL",
        "neurons": int(fb.n),
        "synapses": int(fb.W.nnz),
        "eye_columns": int(len(pilot.eye.on_idx) + len(pilot.eye.off_idx)),
        "motor_neurons": motor,
        "control_steps": N_STEPS,
        "cursor_start": list(START),
        "cursor_end": [round(cx, 1), round(cy, 1)],
        "clicks": [{"x": round(x, 1), "y": round(y, 1), "step": st}
                   for (x, y, st) in clicks],
        "visual_spikes_total": int(total_visual),
        "motor_spikes_total": int(total_motor),
        "brain_fired": fired_any,
        "cursor_moved": moved,
        "seconds": round(dt, 1),
        "checks": {
            "brain_fired": fired_any,
            "eye_drove_brain": total_visual > 0,
            "motor_moved_cursor": moved,
        },
        "steps": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"brain fired          : {fired_any}")
    print(f"eye drove the brain  : {total_visual > 0}  ({total_visual} visual spikes)")
    print(f"motor moved cursor   : {moved}  ({len(clicks)} clicks)")
    print(f"cursor {START} -> ({cx:.0f}, {cy:.0f})  in {dt:.1f}s")
    print(f"wrote {OUT / 'summary.json'}")

    try:
        render(img, cards, traj, clicks)
        print(f"wrote {OUT / 'trajectory.png'}")
    except Exception as exc:  # a missing screenshot must not fail the proof
        print(f"(screenshot skipped: {type(exc).__name__}: {exc})")

    print("=" * 72)
    print("RESULT:", summary["result"])
    return 0 if ok else 1


def render(img, cards, traj, clicks):
    """Draw the fly's cursor path over the synthetic screen and save a PNG."""
    from PIL import Image, ImageDraw
    rgb = np.stack([(img * 255).astype(np.uint8)] * 3, axis=-1)
    im = Image.fromarray(rgb, "RGB")
    d = ImageDraw.Draw(im)
    for (x0, y0, x1, y1) in cards:
        d.rectangle([x0, y0, x1, y1], outline=(90, 90, 110), width=2)
    pts = [(int(x), int(y)) for (x, y) in traj]
    if len(pts) > 1:
        d.line(pts, fill=(80, 200, 255), width=3)
    for (x, y) in pts:
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(80, 200, 255))
    sx, sy = pts[0]
    d.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], outline=(120, 255, 120), width=3)
    for (x, y, _s) in clicks:
        x, y = int(x), int(y)
        d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 90, 90), width=3)
    d.text((12, 12), "fly cursor path (brain+eye+motor, no browser, no chain)",
           fill=(220, 220, 220))
    im.save(OUT / "trajectory.png")


if __name__ == "__main__":
    sys.exit(main())
