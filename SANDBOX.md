# Stripped fork — brain + eye + motor only

This is a fork of [`fruitflydev/flycoinrh`](https://github.com/fruitflydev/flycoinrh)
with **every wallet, chain, trade and coin-launch path removed**. The point of
the upstream project — a real fruit-fly connectome driving a cursor — is kept
intact; the point of *this* fork is that there is nothing left that can spend.

## What was removed

Deleted outright:

- **Wallet / signer:** `rhwallet.py`, `rhlive.py`, `rhdryrun.py`,
  `rhprovider.py`, `check_wallet.py`
- **Chain / trade:** `pons.py` (ABI/contract calls), `executor.py` (the paper
  executor that talked to the chain RPC and was the trade path), `tradebook.py`
- **Coin "backroom"** (the one place a click became a commit): `backroom.py`,
  `backroom_screen.py`, `backrooms*.py`, `backrooms_relay/`
- **Social / narrator** (outbound posting): `voice.py`, `xpost.py`,
  `voice_prompt.md`
- **The live token site / deploy:** `site/`, coin-card web pages
- **Chain libraries** from `requirements.txt`: `eth-account`, `eth-utils`,
  `eth-abi` (and the executor's `requirements-executor.txt`)

Neutralised in place:

- `roam.py` still drives the cursor, but its **backroom is hard-disabled at the
  root** (`backroom_on()` returns `False`, and `load_room()` returns `None`).
  No env flag can reopen a commit path, and the deleted modules are never
  imported.

## What was kept

The connectome sim and everything the fly needs to look and move:

- `flysim.py` — the leaky integrate-and-fire brain (165,122 neurons)
- `flyeye.py` — retinotopic eye + descending-neuron motor readout
- `roam.py` — the motor loop (browser roaming only; reads pages, holds no key)
- `mushroom.py` — the learning circuit, `calibration.py`, `envcfg.py`,
  `build_graph.py`, `mb_sides.py`

## The safe-to-run proof

`sandbox_demo.py` is a headless closed loop: load the connectome, point the
eye at a synthetic dark "table" of bright cards, step the sim, and read the
cursor + click straight out of the descending neurons (DNa02/DNa01/MDN/DNp09) —
the same loop `roam.py` runs, with the live web page replaced by an image in
memory. No browser, no server, no network, no chain.

It runs on a **throwaway cloud sandbox, never a local laptop**:
`.github/workflows/sandbox.yml` spins up a fresh, ephemeral GitHub-hosted
Ubuntu VM (destroyed when the job ends), downloads the CC-BY connectome, builds
the graph, asserts that no wallet/chain file or library survived, runs the sim,
and uploads the log, a JSON summary and a cursor-trajectory screenshot as
artifacts. Trigger it from the repo's **Actions → sandbox-brain-eye-motor →
Run workflow**.

## Connectome attribution

The connectome data is **CC-BY** (© HHMI Janelia FlyEM, the Cambridge
Connectomics Group and collaborators) and is not ours to relicense. It is not
committed here; the workflow fetches it from the public bucket. Keep the
attribution — it is the whole reason any of this is real. Code is MIT, per
upstream `LICENSE`.
