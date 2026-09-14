"""
The fly plays Pythia — a black-box browser harness.

Method A: no repo access to the game. Playwright drives a real Chromium at the
live table on playpythia.com (app: wizzard-qeeq.onrender.com). Each turn the
harness screenshots the table, hands the frame to the fly's eye (the 892
retinotopic hex columns of flyeye.py), steps the 165,122-neuron connectome, and
reads the descending-neuron motor straight out: DNa02 steers the cursor, DNa01
walks it, MDN reverses, and DNp09 — the stopping neuron — is the click. When
DNp09 fires, the harness asks the page what is under the cursor; if it is a
legal card, that real pixel is clicked and the card is played.

The division of labour is the whole point of Method A, and it is strict:

  * THE HARNESS types. It joins/creates the table, sets the name, seats the
    practice bots, places the fly's bid. Anything that needs a keyboard is the
    harness, never the fly — the fly has no keyboard (roam.py, "No typing").
  * THE FLY clicks cards. Its entire contribution is a cursor and a stop. It is
    reacting to light on a screen, not reading the game or choosing a card;
    what comes out of the connectome is a cursor that lands on a card.

Done-when (RON-50): the fly's cursor+click provably lands on a card in a live
Pythia table via this harness. A green run writes build/pythia/summary.json with
the landed card, the DNp09 firing rate at the moment of the click, the cursor
trajectory, and before/after screenshots showing the card leave the hand and
appear in the trick.

RON-51 extends this to a COMPLETE game: the loop bids every round (harness taps
the number), the fly plays every one of its trick turns (legality snap: an
off-card DNp09 stop just re-samples, it never plays), the harness nudges the
between-hand summaries along, and it runs until Pythia shows the final scores.
The whole session is recorded to build/pythia/game.webm — the raw capture of a
full game. `--mode join --code XXXX` points the fly at a table a human hosts, so
the recorded game is a real person vs. the fly rather than fly vs. practice bots.

Motors
------
  --motor fly   the real connectome (flysim + flyeye). Heavy: needs the CC-BY
                graph built by build_graph.py. This is the one that counts.
  --motor lum   a tiny luminance-seeking stand-in with the SAME
                (dx, dy, click) interface. It is NOT the fly and never claims to
                be; it exists only to prove the Playwright <-> table <-> card
                plumbing on a machine that cannot afford the 1.1 GB connectome
                (e.g. a laptop). The summary marks motor:"lum" plainly.

Usage
-----
  python pythia_harness.py --mode selfplay --bots 3 --motor fly
  python pythia_harness.py --mode join --code ABCD --motor fly     # Ron's table
  python pythia_harness.py --mode selfplay --motor lum --headful   # plumbing test
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

APP_DEFAULT = "https://wizzard-qeeq.onrender.com"
ROOT = Path(__file__).parent
OUT = ROOT / "build" / "pythia"


# --------------------------------------------------------------------------- #
# Motors: both return (dx, dy, click, info) from a grayscale screen + cursor.  #
# --------------------------------------------------------------------------- #
class FlyMotor:
    """The real fly: 165k-neuron connectome, retinotopic eye, DN motor readout."""

    name = "fly"

    def __init__(self):
        from flysim import FlyBrain
        from flyeye import FlyPilot
        graph = ROOT / "build" / "graph.npz"
        if not graph.exists():
            raise SystemExit(
                f"{graph} missing — run build_graph.py first (see "
                ".github/workflows/pythia_harness.yml)")
        self.fb = FlyBrain()
        if self.fb.n < 100_000:
            raise SystemExit(f"expected ~165k neurons, got {self.fb.n}")
        self.pilot = FlyPilot(self.fb)
        self.desc = (f"{self.fb.n:,} neurons, "
                     f"{len(self.pilot.eye.on_idx) + len(self.pilot.eye.off_idx):,} "
                     f"retinotopic columns")

    def step(self, img, cx, cy, seed=0):
        dx, dy, click, hz, info = self.pilot.step(img, int(cx), int(cy),
                                                  seed=seed, detail=True)
        return float(dx), float(dy), bool(click), {
            "stop_hz": round(float(hz["stop"]), 1),
            "click_hz": round(float(hz.get("click", 0.0)), 1),
            "fired": int(info.get("firing", 0)),
            "visual": int(info.get("visual", 0)),
            "motor": int(info.get("motor", 0)),
        }


class LumMotor:
    """
    Plumbing stand-in — NOT the fly. Same interface: it reads the same grayscale
    frame the fly reads and returns a cursor step + a stop. It walks toward the
    brightest patch (a card on the dark felt) and stops when it is over it, so it
    exercises the exact screenshot -> step -> cursor -> click loop end to end
    without the 1.1 GB connectome. Every run it drives is stamped motor:"lum".
    """

    name = "lum"

    def __init__(self):
        self.desc = "luminance-seeking stand-in (plumbing only, NOT the fly)"
        self.refractory = 0                        # steps to suppress clicks after one
        self.rng = np.random.default_rng(0)

    def step(self, img, cx, cy, seed=0):
        H, W = img.shape
        # Walk toward the brightest blob in the HAND band (bottom of the frame,
        # where playable cards fan out). Restricting to the lower band is a
        # plumbing heuristic — the stub knows the UI puts the hand at the bottom;
        # the real fly has no such prior. brightness**3 sharpens the pull toward
        # the single brightest card rather than the mean of everything bright.
        ys, xs = np.mgrid[0:H, 0:W]
        band = (ys > 0.60 * H).astype(np.float32)
        w = (np.maximum(img - 0.25, 0.0) ** 3) * band + 1e-9
        tx = float((xs * w).sum() / w.sum())
        ty = float((ys * w).sum() / w.sum())
        jitter = self.rng.normal(0, 8, 2)          # exploration so it never sticks
        dx = float(np.clip(tx - cx, -70, 70) + jitter[0])
        dy = float(np.clip(ty - cy, -70, 70) + jitter[1])
        here = float(img[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))])
        settled = abs(tx - cx) + abs(ty - cy) < 12.0
        if self.refractory > 0:
            self.refractory -= 1
            # bounce away after a miss so the next look starts somewhere new
            dx += float(self.rng.normal(0, 60)); dy += float(self.rng.normal(0, 60))
            return dx, dy, False, {"stop_hz": round(here * 1000, 1), "lum": round(here, 3)}
        click = here > 0.35 and settled
        if click:
            self.refractory = 6
        return dx, dy, click, {"stop_hz": round(here * 1000, 1), "lum": round(here, 3)}


# --------------------------------------------------------------------------- #
# The live table.                                                             #
# --------------------------------------------------------------------------- #
class Pythia:
    def __init__(self, page, log):
        self.pg = page
        self.log = log

    # -- small helpers --------------------------------------------------------
    def body(self, n=200):
        return self.pg.evaluate(
            "()=>document.body.innerText.replace(/\\s*\\n+\\s*/g,' | ')")[:n]

    def click_role(self, name, to=8000, exact=True):
        loc = self.pg.get_by_role("button", name=name, exact=exact)
        if loc.count() and loc.first.is_visible() and loc.first.is_enabled():
            loc.first.click(timeout=to)
            return True
        return False

    def click_text_button(self, text):
        """Click a <button> by exact trimmed innerText via the DOM (robust for
        the segmented number/label controls that ignore synthetic role clicks)."""
        return self.pg.evaluate(
            """(t)=>{const b=[...document.querySelectorAll('button')]
               .find(x=>(x.offsetWidth||x.offsetHeight)&&(x.innerText||'').trim()===t);
               if(b){b.click();return true}return false}""", text)

    def dismiss_onboarding(self):
        for _ in range(3):
            try:
                s = self.pg.get_by_role("button", name="Skip")
                if s.count() and s.first.is_visible():
                    s.first.click(timeout=3000); time.sleep(1)
                else:
                    break
            except Exception:
                break

    def set_dark_theme(self):
        self.pg.evaluate(
            """()=>{try{localStorage.setItem('wizzard:theme','dark');
               document.documentElement.setAttribute('data-theme','dark');}catch(e){}}""")

    def room_code(self):
        return self.pg.evaluate(
            """()=>{const c=document.querySelector('.wz-codebox');
               return c?(c.innerText||'').replace(/\\s/g,'').slice(0,6):null;}""")

    # -- setup ---------------------------------------------------------------
    def create_table(self, name, bots):
        self.click_role("Create room"); time.sleep(1.5)
        self.pg.locator("#wizname").fill(name); time.sleep(0.3)
        self.click_role("Next"); time.sleep(1.0)   # step 2 (bots, skipped here)
        self.click_role("Next"); time.sleep(1.0)   # step 3 (rules)
        self.click_role("Create room"); time.sleep(3)
        self.set_dark_theme()
        # open the room-settings pencil and seat bots + turn timer OFF
        self.click_text_button("✎"); time.sleep(1.3)
        self.click_text_button(str(bots))          # practice bots
        self.click_text_button("Off")              # per-turn timer off: fly is slow
        time.sleep(0.4)
        self.click_role("Apply") or self.click_text_button("Apply")
        time.sleep(3)
        code = self.room_code()
        self.log(f"created table {code}  ({self.body(120)})")
        return code

    def join_table(self, name, code):
        self.click_role("Join"); time.sleep(1.5)
        # name + code inputs (harness types both — the fly has no keyboard)
        inp = self.pg.locator("input:visible")
        if inp.count() >= 1:
            inp.nth(0).fill(name)
        code_box = self.pg.locator("#wizjoincode, input[maxlength='4'], input")
        # type the code into whichever field is empty
        self.pg.evaluate("""(code)=>{const ins=[...document.querySelectorAll('input')]
            .filter(i=>i.offsetWidth||i.offsetHeight);
            const t=ins.find(i=>/code/i.test(i.id+i.name+(i.placeholder||''))) || ins[ins.length-1];
            if(t){t.focus();t.value=code;t.dispatchEvent(new Event('input',{bubbles:true}));}}""", code)
        time.sleep(0.5)
        for lbl in ["Join", "Join room", "Enter", "Next", "Go"]:
            if self.click_role(lbl):
                break
        time.sleep(3)
        self.set_dark_theme()
        self.log(f"joined table {code}  ({self.body(120)})")

    def start_when_ready(self, wait_s=90):
        t0 = time.time()
        while time.time() - t0 < wait_s:
            sg = self.pg.get_by_role("button", name="Start game")
            if sg.count() and sg.first.is_enabled():
                sg.first.click(); self.log("started game"); return True
            time.sleep(3)
        return False

    # -- phase detection -----------------------------------------------------
    def phase(self):
        return self.pg.evaluate("""()=>{
            const t=document.body.innerText;
            const legal=[...document.querySelectorAll('.wz-hand .wz-card')]
              .filter(c=>(c.offsetWidth||c.offsetHeight)&&!/wz-card--illegal/.test(c.className)).length;
            const bidCta=!!document.querySelector('.wz-bidsheet__cta');
            const yourTurn=/your turn/i.test(t) || /your bid/i.test(t);
            // "R7/15" style round indicator: which round of how many.
            const m=t.match(/R\\s*(\\d+)\\s*\\/\\s*(\\d+)/);
            const round=m?parseInt(m[1],10):null, rounds=m?parseInt(m[2],10):null;
            return {legal, bidCta, yourTurn, round, rounds,
                    over:/game over|final scores?|final standings|winner!|congratulations|play again|rematch/i.test(t)};
        }""")

    ADVANCE_LABELS = ("Next round", "Next hand", "Next trick", "Continue",
                      "Next", "Deal", "Play on", "Ready", "Start round")

    def click_advance(self):
        """Between tricks/rounds Pythia sometimes parks on a summary that needs a
        push to deal the next hand. The harness owns these non-card taps (like the
        bid and the room code); the fly only ever clicks cards. Never touches
        Leave room / Rematch / Play again (those end or restart the game)."""
        for lbl in self.ADVANCE_LABELS:
            if self.click_text_button(lbl):
                self.log(f"  harness advanced: '{lbl}'")
                return True
        return False

    def place_bid_if_needed(self):
        """Harness role: submit the fly's bid via the gold CTA so trick play can
        begin. Bidding uses a keyboard-free stepper+button, but choosing a number
        is a decision, so the harness does it — the fly only plays cards."""
        cta = self.pg.locator(".wz-bidsheet__cta")
        if cta.count() and cta.first.is_visible() and cta.first.is_enabled():
            txt = (cta.first.inner_text() or "").strip()
            cta.first.click()
            self.log(f"harness placed bid: '{txt}'")
            time.sleep(2)
            return True
        return False

    def legal_cards(self):
        return self.pg.eval_on_selector_all(".wz-hand .wz-card", """els=>els
            .filter(c=>(c.offsetWidth||c.offsetHeight)&&!/wz-card--illegal/.test(c.className))
            .map(c=>{const r=c.getBoundingClientRect();return{
              cx:Math.round(r.x+r.width/2),cy:Math.round(r.y+r.height/2),
              w:Math.round(r.width),h:Math.round(r.height),
              cls:c.className,txt:(c.innerText||'').trim().slice(0,12)};})""")

    def card_under(self, x, y):
        """What the page has under (x,y): the nearest wz-card ancestor, whether it
        is a playable hand card (in .wz-hand and not illegal). This is how a DNp09
        stop over a pixel becomes a card selection."""
        return self.pg.evaluate("""([x,y])=>{
            let e=document.elementFromPoint(x,y), c=e;
            while(c && !(c.classList&&c.classList.contains('wz-card'))) c=c.parentElement;
            if(!c) return null;
            const inHand=!!c.closest('.wz-hand');
            return {legal: inHand && !/wz-card--illegal/.test(c.className),
                    inHand, cls:c.className, txt:(c.innerText||'').trim().slice(0,12)};
        }""", [int(x), int(y)])

    def hand_count(self):
        return self.pg.eval_on_selector_all(".wz-hand .wz-card",
            "els=>els.filter(c=>c.offsetWidth||c.offsetHeight).length")


# --------------------------------------------------------------------------- #
def to_gray(png_bytes, W, H):
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    if im.size != (W, H):
        im = im.resize((W, H))
    return np.asarray(im, dtype=np.float32) / 255.0


def run(args):
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    logs = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line); sys.stdout.flush(); logs.append(line)

    log("=" * 70)
    log("The fly plays Pythia — black-box browser harness (Method A)")
    motor = FlyMotor() if args.motor == "fly" else LumMotor()
    log(f"motor: {motor.name} — {motor.desc}")
    log("=" * 70)

    W, H = args.width, args.height
    result = {"result": "FAIL", "motor": motor.name, "mode": args.mode,
              "app": args.app, "landed": None, "misses": 0, "turns_played": 0,
              "cards_played": 0, "rounds_seen": [], "round_last": None,
              "rounds_total": None, "reached_final_round": False,
              "game_over": False, "bids_placed": 0,
              "landings": [], "trajectory": [], "checks": {}}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        # Record the whole session to a webm — this is the RON-51 raw capture of a
        # complete game. Video finalizes when the context closes.
        ctx = browser.new_context(viewport={"width": W, "height": H},
                                  device_scale_factor=1,
                                  record_video_dir=str(OUT),
                                  record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        page.set_default_timeout(15000)     # never block a whole run on one call
        pj = Pythia(page, log)
        page.goto(args.app, wait_until="networkidle", timeout=90000)
        time.sleep(2)
        pj.set_dark_theme()
        pj.dismiss_onboarding()
        pj.set_dark_theme()

        if args.mode == "join":
            if not args.code:
                raise SystemExit("--mode join needs --code")
            pj.join_table(args.name, args.code.upper())
            result["code"] = args.code.upper()
        else:
            code = pj.create_table(args.name, args.bots)
            result["code"] = code
            if not pj.start_when_ready():
                log("FAIL: table never reached the seats needed to start")
                result["checks"]["started"] = False
                _finish(result, logs, page); browser.close(); return result
        result["checks"]["started"] = True
        time.sleep(4)

        # ---- full-game loop: bid each round, fly plays every trick ----------
        # RON-51: run the WHOLE game, not just the first card. Every iteration
        # reads the table's phase and does the one thing that phase calls for —
        #   * bid showing        -> harness taps the bid CTA (a decision, so ours)
        #   * fly's trick turn   -> hand the frame to the fly; it steers+stops on
        #                           a legal card (legality snap: an off-card stop
        #                           just re-samples, never plays)
        #   * between hands      -> harness pushes the round-summary along
        #   * anyone else's turn -> wait for the table to come back to us
        # It ends when Pythia shows the final scores (game over), or bails out if
        # the table goes idle far too long, so a stuck run can never hang the VM.
        # Where the cursor STARTS each turn is the harness's choice (a human's
        # cursor starts somewhere too); the fly owns the approach and the click.
        def seat():
            return W / 2, (0.82 * H) if args.start == "hand" else (H / 2)

        cx, cy = seat()
        deadline = time.time() + args.time_budget
        last_progress = time.time()
        final_idle_since = None
        first_landing = None
        waits = 0
        while time.time() < deadline:
            ph = pj.phase()
            if ph.get("round"):
                result["round_last"] = ph["round"]
                result["rounds_total"] = ph.get("rounds")
                if ph["round"] not in result["rounds_seen"]:
                    result["rounds_seen"].append(ph["round"])
                if ph.get("rounds") and ph["round"] >= ph["rounds"]:
                    result["reached_final_round"] = True
            if waits % 5 == 0:
                log(f"  waiting… phase={ph} :: {pj.body(90)}")
                try:
                    page.screenshot(path=str(OUT / "waiting.png"))
                except Exception:
                    pass

            if ph["over"]:
                log(f"GAME OVER — final table reached after "
                    f"{result['cards_played']} fly cards over "
                    f"{len(result['rounds_seen'])} round(s)")
                result["game_over"] = True
                try:
                    page.screenshot(path=str(OUT / "gameover.png"))
                except Exception:
                    pass
                break

            if ph["bidCta"]:
                if pj.place_bid_if_needed():
                    result["bids_placed"] += 1
                    last_progress = time.time()
                    final_idle_since = None
                    cx, cy = seat()
                else:
                    waits += 1
                time.sleep(1); continue

            if ph["legal"] > 0:
                # The fly's trick turn: legal cards in hand and it's ours to play.
                log(f"fly's turn (round {ph.get('round')}) — {ph['legal']} legal "
                    f"card(s). Letting the fly look.")
                landed, cx, cy = fly_turn(pj, page, motor, cx, cy, W, H,
                                          args, result, log)
                result["turns_played"] += 1
                if landed is not None:
                    if landed["played"]:
                        result["cards_played"] += 1
                    result["landings"].append(landed)
                    first_landing = first_landing or landed
                    last_progress = time.time()
                    final_idle_since = None
                else:
                    cx, cy = seat()            # re-seat and try again next loop
                continue

            # Not our turn (bots playing) or a between-hand summary. Nudge any
            # "next round / continue" button along, else wait it out.
            if pj.click_advance():
                last_progress = time.time()
                final_idle_since = None
                time.sleep(1); continue
            # If we already reached the last round, an idle summary with no next
            # deal coming (no bid, no cards) IS the end of the game — there is no
            # round rounds_total+1 to wait for. This is the reliable game-over
            # signal; the `over` text regex above is the fast path when Pythia
            # spells it out ("final scores" / "winner").
            if result.get("reached_final_round"):
                if final_idle_since is None:
                    final_idle_since = time.time()
                elif time.time() - final_idle_since > args.final_grace:
                    log(f"final round ({result['rounds_total']}) done and no new "
                        f"deal in {args.final_grace:.0f}s — GAME OVER")
                    result["game_over"] = True
                    try:
                        page.screenshot(path=str(OUT / "gameover.png"))
                    except Exception:
                        pass
                    break
            if time.time() - last_progress > args.idle_timeout:
                log(f"table idle >{args.idle_timeout:.0f}s with nothing to do — "
                    f"stopping (game likely stalled)")
                break
            waits += 1
            time.sleep(2)

        # ---- verdict --------------------------------------------------------
        result["landed"] = first_landing
        all_legal = all(l["legal"] for l in result["landings"])
        clicked_ok = first_landing is not None and first_landing["legal"]
        complete = bool(result["game_over"]) and result["cards_played"] > 0
        result["checks"]["fly_click_landed_on_card"] = bool(clicked_ok)
        result["checks"]["all_landings_legal"] = bool(all_legal)
        result["checks"]["game_completed"] = bool(result["game_over"])
        # PASS for RON-51 = a COMPLETE game: the fly legally played cards across
        # the rounds and the table reached game over. (A single legal landing with
        # no game-over is the old RON-50 bar and is reported as PARTIAL.)
        if complete and all_legal:
            result["result"] = "PASS"
        elif clicked_ok:
            result["result"] = "PARTIAL"
        else:
            result["result"] = "FAIL"
        _finish(result, logs, page)
        log(f"RESULT: {result['result']} — fly played {result['cards_played']} "
            f"card(s) over {len(result['rounds_seen'])} round(s); "
            f"game_over={result['game_over']}, all_legal={all_legal}")

        # Close the context so Playwright flushes the video, then name it.
        try:
            vid = page.video
            ctx.close()
            if vid:
                src = Path(vid.path())
                dst = OUT / "game.webm"
                if src.exists():
                    src.replace(dst)
                    result["video"] = dst.name
                    log(f"raw capture saved: {dst.name} ({dst.stat().st_size} bytes)")
                    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
        except Exception as e:
            log(f"video finalize note: {e}")
        browser.close()
    return result


def _arena(boxes, W, H, inset=6):
    """The on-card-safe play area the harness confines the fly's cursor to. The
    harness owns the cursor's *bounds* (as it owns the start position and all
    typing); the fly still owns the steer and the stop inside it. Without a bound,
    a connectome DN readout with any net vertical drift walks the cursor off the
    cards into empty felt and pins at the viewport edge — exactly what run
    34830208802 did (2937/3000 frames stuck at y=1198, only 26 ever in the card
    band, 61 DNp09 stops all over empty felt).

    Horizontal extent = the full span of the legal cards (so the fly can steer
    across every choice); vertical extent = the *intersection* of the cards'
    heights, inset a few px, so wherever the cursor pins it is still inside a card
    rather than in the gap just below the fan. For a single legal card that
    intersection is simply the card, so the first DNp09 stop lands. If a heavy fan
    makes the intersection empty, fall back to the union so the cursor still has a
    band to work in."""
    x0 = min(b["cx"] - b["w"] / 2 for b in boxes) + inset
    x1 = max(b["cx"] + b["w"] / 2 for b in boxes) - inset
    ytop = max(b["cy"] - b["h"] / 2 for b in boxes) + inset   # lowest card top
    ybot = min(b["cy"] + b["h"] / 2 for b in boxes) - inset   # highest card bottom
    if ybot <= ytop:                                          # fan too spread: union
        ytop = min(b["cy"] - b["h"] / 2 for b in boxes) + inset
        ybot = max(b["cy"] + b["h"] / 2 for b in boxes) - inset
    return (max(1, x0), max(1, ytop), min(W - 2, x1), min(H - 2, ybot))


def fly_turn(pj, page, motor, cx, cy, W, H, args, result, log):
    """One trick turn: the fly looks, walks and stops until DNp09 fires over a
    legal card. Returns (landing_or_None, cx, cy) so the caller can continue the
    cursor from where the fly left it."""
    before_hand = pj.hand_count()
    # Confine the cursor to the legal-card arena and re-seat it onto a card, so the
    # fly's narrow retinal FOV always has a card in view and its DN drift can't sink
    # into empty felt. The fly still chooses which card (steer) and when (DNp09 stop).
    boxes = pj.legal_cards()
    if boxes:
        ax0, ay0, ax1, ay1 = _arena(boxes, W, H)
        cx = float(np.clip(cx, ax0, ax1))
        cy = float(np.clip(cy, ay0, ay1))
        if not (ax0 <= cx <= ax1 and ay0 <= cy <= ay1):
            cx, cy = boxes[0]["cx"], boxes[0]["cy"]
        log(f"  arena x[{ax0:.0f},{ax1:.0f}] y[{ay0:.0f},{ay1:.0f}] "
            f"over {len(boxes)} legal card(s); cursor re-seated @({cx:.0f},{cy:.0f})")
    else:
        ax0, ay0, ax1, ay1 = 1, 1, W - 2, H - 2
    for step in range(args.max_steps):
        png = page.screenshot()
        img = to_gray(png, W, H)
        dx, dy, click, info = motor.step(img, cx, cy, seed=step)
        cx = float(np.clip(cx + dx, ax0, ax1))
        cy = float(np.clip(cy + dy, ay0, ay1))
        try:
            page.mouse.move(cx, cy)             # real hover under the cursor
        except Exception:
            pass
        result["trajectory"].append([round(cx, 1), round(cy, 1),
                                     round(info.get("stop_hz", 0.0), 1), bool(click)])
        if not click:
            continue
        under = pj.card_under(cx, cy)
        if under and under["legal"]:
            # DNp09 fired with a legal card under the cursor: click that pixel.
            page.mouse.click(cx, cy)
            time.sleep(2.5)
            after_hand = pj.hand_count()
            played = after_hand < before_hand
            log(f"  DNp09 stop over legal card @({cx:.0f},{cy:.0f}) "
                f"{under['txt']!r} — clicked. hand {before_hand}->{after_hand} "
                f"played={played}")
            try:
                page.screenshot(path=str(OUT / "landed.png"))
            except Exception:
                pass
            return ({"step": step, "x": round(cx, 1), "y": round(cy, 1),
                     "legal": True, "played": bool(played),
                     "card": under["txt"], "cls": under["cls"],
                     "stop_hz": info.get("stop_hz", 0.0)}, cx, cy)
        else:
            result["misses"] += 1               # clicked, but not on a legal card
            log(f"  DNp09 stop @({cx:.0f},{cy:.0f}) missed "
                f"({(under or {}).get('cls','no card')}) — fly keeps looking")
    return None, cx, cy


def _finish(result, logs, page):
    try:
        page.screenshot(path=str(OUT / "final.png"))
    except Exception:
        pass
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    (OUT / "run.log").write_text("\n".join(logs))


def main():
    ap = argparse.ArgumentParser(description="The fly plays Pythia (Method A).")
    ap.add_argument("--app", default=APP_DEFAULT)
    ap.add_argument("--mode", choices=["selfplay", "join"], default="selfplay")
    ap.add_argument("--code", default=None, help="room code for --mode join")
    ap.add_argument("--name", default="FlyPilot")
    ap.add_argument("--bots", type=int, default=3, help="practice bots to seat")
    ap.add_argument("--motor", choices=["fly", "lum"], default="fly")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--start", choices=["hand", "center"], default="hand",
                    help="where the harness parks the cursor before the fly looks")
    ap.add_argument("--max-steps", type=int, default=400,
                    help="control steps the fly gets per trick turn")
    ap.add_argument("--time-budget", type=float, default=1500,
                    help="seconds for the whole run (a full game needs headroom)")
    ap.add_argument("--idle-timeout", type=float, default=240,
                    help="bail out if the table has nothing for us this long")
    ap.add_argument("--final-grace", type=float, default=25,
                    help="after the last round, wait this long with no new deal "
                         "before declaring the game over")
    ap.add_argument("--headful", action="store_true")
    args = ap.parse_args()
    res = run(args)
    return 0 if res["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
