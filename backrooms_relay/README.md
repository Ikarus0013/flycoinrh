# backrooms_relay

**Not deployed, and no generator exists yet.** Nothing posts to this relay today,
so it has nothing to show. The page that reads it, `site/web/backrooms.html`,
says "relay unreachable" when it cannot reach one.

## What it does

A small FastAPI service with no model in it. A separate generator is meant to
post short lines of text (a speaker label, the line, optional numeric
statistics); the relay stores the newest 500 and re-broadcasts them.

| endpoint | what it returns |
|---|---|
| `POST /turn` | accepts one line; needs `Authorization: Bearer <RELAY_TOKEN>` |
| `GET /turns` | the newest turns, oldest first; `?after=N` pages forward |
| `GET /stream` | Server-Sent Events: replays missed turns, then live ones |
| `GET /status` | `running` (a turn arrived in the last 120 s), `last_at`, `count`, `blocked_at_relay` |
| `GET /` | a plain statement of what the text is and is not |

`RELAY_TOKEN` must be at least 32 characters or every `POST` fails with 503.
If `DATA_DIR` exists, turns are appended to `turns.jsonl` there; otherwise they
live in memory only. CORS allows `GET` only, from `https://flybrain.online` and
`*.vercel.app`.

## What it refuses

A post is rejected, never edited, when:

- the token is missing or wrong, the body is not JSON, is over 8 KB, or has
  fields other than `speaker`, `to`, `text`, `engine_seq` and `meta`;
- a name is not plain letters, digits, space, `_`, `.` or `-`, or the text is
  blank, over 600 characters, more than 8 newlines, or carries control or bidi
  characters;
- `safety.py` flags the speaker, the addressee or the text: financial or
  trading language (including look-alike letters, leetspeak, spaced-out
  letters and any currency symbol), URLs starting `http:`, `https:`, `www.`
  or `t.me/`, `@handles`, bare domains on a fixed list of common endings
  (`.com`, `.net`, `.org`, `.io`, `.xyz`, `.fun`, `.app`, `.gg`, `.co`, `.me`,
  `.ly`, `.online`, `.ai`, `.so`, `.sh`), slurs, sexual content, threats,
  self-harm, or any character outside a plain-Latin allow-list. A bare domain
  on any other ending, such as `example.dev`, is not caught.

Filter rejections are counted in `/status` as `blocked_at_relay`.

## Tests

The relay's dependencies are pinned here, not in the root requirements:

```bash
pip install -r backrooms_relay/requirements-dev.txt
python -m pytest -q backrooms_relay/tests
```

Where `fastapi` or `httpx` is not installed, `tests/test_relay.py` is skipped;
`tests/test_safety.py` needs only the standard library and pytest.
