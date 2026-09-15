#!/usr/bin/env python3
"""
RON-50 join-mode proof — the fly's seat readies up so the host can start.

This is a *self-contained* regression test for the exact bug Ron reported: in
join mode the fly's seat lands NOT READY and the human host cannot start the
game until it is ready, but the harness only ever matched the literal button
"Ready" and so never tapped the lobby's real "I'm ready" button (curly
apostrophe). The fix taught the harness to ready the seat up (Pythia.ready_up).

Unlike pythia_harness.py this needs NO connectome and NO fly brain — readying is
purely a harness lobby tap, so we exercise only that path. Two browser contexts
in one Chromium play the two human roles that Method A assigns to the harness:

  * HOST  — creates a table, seats one practice bot (so the game is startable
            with 3 players once everyone is ready), never readies (the host
            starts, it doesn't ready).
  * FLY   — the joiner seat. Joins with the code, then the harness readies it up
            exactly as pythia_harness.py does in join mode.

The proof is the gate flip: while the fly seat is NOT READY the host's "Start
game" is disabled; the instant the harness readies the fly, it enables and the
host can start. Emits build/pythia/join_ready.json with result PASS/FAIL.
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Reuse the exact production harness object under test — same join/create/ready
# code paths pythia_harness.py runs, so a pass here is a pass for the harness.
from pythia_harness import Pythia, APP_DEFAULT

OUT = Path(__file__).parent / "build" / "pythia"
W, H = 900, 1200


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    sys.stdout.flush()


def start_enabled(page):
    """Is the host's 'Start game' button present AND enabled right now?"""
    return page.evaluate(r"""()=>{
        const b=[...document.querySelectorAll('button')].find(x=>
          (x.offsetWidth||x.offsetHeight) && /start game/i.test(x.innerText||''));
        return b ? !b.disabled : false;
    }""")


def shot(page, name):
    try:
        page.screenshot(path=str(OUT / name))
    except Exception as e:
        log(f"  (screenshot {name} failed: {e})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"result": "FAIL", "test": "join_ready", "app": APP_DEFAULT,
           "code": None, "checks": {}}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # -- HOST -------------------------------------------------------------
        host_ctx = browser.new_context(viewport={"width": W, "height": H})
        host = host_ctx.new_page()
        host.set_default_timeout(15000)
        pj_host = Pythia(host, log)
        host.goto(APP_DEFAULT, wait_until="networkidle", timeout=90000)
        time.sleep(2)
        pj_host.set_dark_theme(); pj_host.dismiss_onboarding(); pj_host.set_dark_theme()
        # 1 bot -> host + bot + fly = 3 players, a legal Wizard table; the bot
        # auto-readies, so the only thing gating Start is the fly's seat.
        code = pj_host.create_table("HostRon", 1)
        res["code"] = code
        log(f"HOST created table {code}")
        if not code:
            res["checks"]["created"] = False
            _write(res); browser.close(); return res
        res["checks"]["created"] = True
        shot(host, "jr_host_created.png")

        # -- FLY (joiner) -----------------------------------------------------
        fly_ctx = browser.new_context(viewport={"width": W, "height": H})
        fly = fly_ctx.new_page()
        fly.set_default_timeout(15000)
        pj_fly = Pythia(fly, log)
        fly.goto(APP_DEFAULT, wait_until="networkidle", timeout=90000)
        time.sleep(2)
        pj_fly.set_dark_theme(); pj_fly.dismiss_onboarding(); pj_fly.set_dark_theme()
        pj_fly.join_table("FlyPilot", code)
        time.sleep(3)

        # -- GATE, BEFORE READY ----------------------------------------------
        res["checks"]["fly_not_ready_on_join"] = (not pj_fly.is_ready())
        time.sleep(2)  # let the host see the joiner seat appear
        enabled_before = start_enabled(host)
        res["checks"]["host_start_disabled_before_ready"] = (not enabled_before)
        log(f"BEFORE ready: fly.is_ready={pj_fly.is_ready()}  "
            f"host Start enabled={enabled_before}")
        shot(fly, "jr_fly_before.png"); shot(host, "jr_host_before.png")

        # -- THE FIX UNDER TEST: harness readies the fly up ------------------
        readied = pj_fly.ready_up(wait_s=25)
        res["checks"]["readied"] = bool(readied)
        log(f"harness ready_up -> {readied}; fly.is_ready={pj_fly.is_ready()}")
        shot(fly, "jr_fly_readied.png")

        # -- GATE, AFTER READY (poll: host UI updates over the socket) -------
        enabled_after = False
        for _ in range(20):
            enabled_after = start_enabled(host)
            if enabled_after:
                break
            time.sleep(1)
        res["checks"]["host_start_enabled_after_ready"] = bool(enabled_after)
        log(f"AFTER ready: host Start enabled={enabled_after}")
        shot(host, "jr_host_after.png")

        # -- and it actually starts ------------------------------------------
        started = False
        if enabled_after:
            started = pj_host.start_when_ready(wait_s=20)
            time.sleep(4)
            # both seats should have left the lobby copy
            left_lobby = (not pj_fly.in_lobby())
            res["checks"]["host_started_game"] = bool(started)
            res["checks"]["fly_left_lobby"] = bool(left_lobby)
            log(f"host start_when_ready={started}; fly_left_lobby={left_lobby}")
            shot(host, "jr_started_host.png"); shot(fly, "jr_started_fly.png")

        c = res["checks"]
        res["result"] = "PASS" if (
            c.get("created")
            and c.get("readied")
            and c.get("host_start_disabled_before_ready")
            and c.get("host_start_enabled_after_ready")
            and c.get("host_started_game")
        ) else "FAIL"
        _write(res)
        browser.close()
    return res


def _write(res):
    (OUT / "join_ready.json").write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 70)
    print(json.dumps(res, indent=2))
    print("=" * 70)
    print(f"RESULT: {res['result']}")


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["result"] == "PASS" else 1)
