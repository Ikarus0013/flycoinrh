import json
import os
import secrets
import sys
import threading
import time

import pytest

# The relay's dependencies live in backrooms_relay/requirements-dev.txt, not in the
# repository's root requirements; skip cleanly where they are not installed.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import relay as relay_mod  # noqa: E402
from relay import create_app  # noqa: E402

TOKEN = secrets.token_hex(32)  # test-only token, generated per run


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def auth(tok=TOKEN):
    return {"Authorization": f"Bearer {tok}"}


def turn(text="the corridor hums again", speaker="Wren", **kw):
    body = {"speaker": speaker, "text": text}
    body.update(kw)
    return body


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def client(tmp_path, clock):
    app = create_app(token=TOKEN, data_dir=str(tmp_path / "missing"), clock=clock, heartbeat_s=0.2)
    with TestClient(app) as c:
        yield c


def parse_sse(raw: str):
    events = []
    for block in raw.split("\n\n"):
        fields = {}
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            k, _, v = line.partition(": ")
            fields[k] = v
        if fields.get("event") == "turn":
            events.append({"id": int(fields["id"]), "data": json.loads(fields["data"])})
    return events


# ---------- auth ----------

def test_auth_required(client):
    r = client.post("/turn", json=turn())
    assert r.status_code == 401
    assert client.get("/status").json()["count"] == 0


def test_bad_token_401(client):
    assert client.post("/turn", json=turn(), headers=auth("x" * 64)).status_code == 401
    assert client.post("/turn", json=turn(), headers=auth(TOKEN[:-1])).status_code == 401
    assert client.post("/turn", json=turn(), headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert client.post("/turn", json=turn(), headers={"Authorization": "Bearer caf\xe9".encode("latin-1")}).status_code == 401
    assert client.get("/status").json()["count"] == 0


def test_good_token_accepts(client, clock):
    r = client.post("/turn", json=turn(to="Moth", engine_seq=7, meta={"bias_norm": 0.42, "tokens": 31}), headers=auth())
    assert r.status_code == 200, r.text
    assert r.json()["seq"] == 1
    t = client.get("/turns").json()["turns"][0]
    assert t["speaker"] == "Wren" and t["to"] == "Moth" and t["engine_seq"] == 7
    assert t["ts"] == clock.t and t["at"].endswith("Z")
    assert t["meta"] == {"bias_norm": 0.42, "tokens": 31}


def test_missing_or_short_server_token_fails_closed(tmp_path):
    for tok in ("", "short"):
        app = create_app(token=tok, data_dir=str(tmp_path / "none"))
        with TestClient(app) as c:
            assert c.post("/turn", json=turn(), headers=auth(tok)).status_code == 503


# ---------- validation ----------

@pytest.mark.parametrize(
    "body",
    [
        {"speaker": "Wren"},                                   # missing text
        {"text": "hello"},                                     # missing speaker
        turn(extra="nope"),                                    # unknown field
        turn(text=""),                                         # empty
        turn(text="   "),                                      # blank
        turn(text="a" * 601),                                  # too long
        turn(speaker="1Wren"),                                 # bad name
        turn(speaker="W" * 25),                                # long name
        turn(speaker="<script>"),                              # markup in name
        turn(text=123),                                        # wrong type
        turn(engine_seq=True),                                 # bool is not int
        turn(engine_seq="5"),                                  # string is not int
        turn(engine_seq=-1),
        turn(meta={f"k{i}": 1 for i in range(13)}),            # too many meta keys
        turn(meta={"Bad-Key": 1}),
        turn(meta={"ok": "string"}),
        turn(meta={"ok": [1, 2]}),
        turn(text="line\x00null"),                             # control char
        turn(text="spoof\u202eevil"),                               # bidi override
        turn(text="\n".join(["x"] * 10)),                      # too many lines
        turn(text="we should buy more"),                       # financial language
        turn(text="that costs $5"),
        turn(text="the Moon is up"),
        turn(text="a token of the wall"),
    ],
)
def test_validation_rejects(client, body):
    r = client.post("/turn", json=body, headers=auth())
    assert r.status_code == 422, (body, r.status_code, r.text)
    assert client.get("/status").json()["count"] == 0


def test_validation_transport_errors(client):
    h = auth()
    assert client.post("/turn", content=b"not json", headers={**h, "Content-Type": "application/json"}).status_code == 400
    assert client.post("/turn", content=b'{"speaker":"Wren","text":"x","meta":{"a":NaN}}',
                       headers={**h, "Content-Type": "application/json"}).status_code == 400
    assert client.post("/turn", content=b"\xff\xfe", headers={**h, "Content-Type": "application/json"}).status_code == 400
    assert client.post("/turn", content=json.dumps(turn()), headers={**h, "Content-Type": "text/plain"}).status_code == 415
    assert client.post("/turn", json=[turn()], headers=h).status_code == 422
    big = json.dumps(turn(meta={"pad": 1}) | {"text": "a" * 9000})
    assert client.post("/turn", content=big, headers={**h, "Content-Type": "application/json"}).status_code == 413
    assert client.get("/status").json()["count"] == 0


def test_newlines_and_tabs_allowed(client):
    assert client.post("/turn", json=turn(text="one\ntwo\tthree"), headers=auth()).status_code == 200


# ---------- paging ----------

def test_default_returns_newest_not_oldest(client):
    for i in range(540):
        assert client.post("/turn", json=turn(text=f"line {i}"), headers=auth()).status_code == 200
    body = client.get("/turns").json()
    seqs = [t["seq"] for t in body["turns"]]
    assert seqs == list(range(441, 541)), (seqs[0], seqs[-1])     # newest 100, oldest first
    assert body["last_seq"] == 540 and body["oldest_seq"] == 41
    assert [t["seq"] for t in client.get("/turns", params={"limit": 3}).json()["turns"]] == [538, 539, 540]
    assert [t["seq"] for t in client.get("/turns", params={"after": 100, "limit": 2}).json()["turns"]] == [101, 102]


def test_after_paging(client):
    for i in range(5):
        assert client.post("/turn", json=turn(text=f"turn {i}"), headers=auth()).status_code == 200
    body = client.get("/turns", params={"after": 2}).json()
    assert [t["seq"] for t in body["turns"]] == [3, 4, 5]
    assert body["last_seq"] == 5 and body["oldest_seq"] == 1
    assert [t["seq"] for t in client.get("/turns", params={"after": 0, "limit": 2}).json()["turns"]] == [1, 2]
    assert client.get("/turns", params={"after": 5}).json()["turns"] == []
    assert client.get("/turns", params={"after": 99}).json()["turns"] == []
    assert client.get("/turns", params={"after": -1}).status_code == 422
    assert client.get("/turns", params={"limit": 501}).status_code == 422


def test_memory_keeps_last_500(client):
    for i in range(505):
        client.post("/turn", json=turn(text=f"n {i}"), headers=auth())
    body = client.get("/turns", params={"after": 0, "limit": 500}).json()
    assert len(body["turns"]) == 500
    assert body["oldest_seq"] == 6 and body["last_seq"] == 505
    assert client.get("/status").json()["count"] == 505


# ---------- persistence ----------

def test_appends_to_data_dir_and_restores(tmp_path, clock):
    data = tmp_path / "data"
    data.mkdir()
    app = create_app(token=TOKEN, data_dir=str(data), clock=clock)
    with TestClient(app) as c:
        for i in range(3):
            c.post("/turn", json=turn(text=f"p {i}"), headers=auth())
    lines = (data / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["seq"] for l in lines] == [1, 2, 3]
    app2 = create_app(token=TOKEN, data_dir=str(data), clock=clock)
    with TestClient(app2) as c:
        assert c.get("/status").json()["count"] == 3
        c.post("/turn", json=turn(text="after restart"), headers=auth())
        assert [t["seq"] for t in c.get("/turns").json()["turns"]] == [1, 2, 3, 4]


def test_memory_only_without_data_dir(tmp_path, clock):
    app = create_app(token=TOKEN, data_dir=str(tmp_path / "nope"), clock=clock)
    with TestClient(app) as c:
        assert c.post("/turn", json=turn(), headers=auth()).status_code == 200
    assert not (tmp_path / "nope").exists()


# ---------- SSE ----------

def test_sse_replays_posted_turn(client):
    client.post("/turn", json=turn(text="first words"), headers=auth())
    client.post("/turn", json=turn(text="second words", speaker="Moth"), headers=auth())
    with client.stream("GET", "/stream", params={"after": 0, "max_events": 2}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = r.read().decode("utf-8")
    ev = parse_sse(raw)
    assert [e["id"] for e in ev] == [1, 2]
    assert ev[1]["data"]["text"] == "second words" and ev[1]["data"]["speaker"] == "Moth"


def test_sse_last_event_id(client):
    for i in range(3):
        client.post("/turn", json=turn(text=f"e {i}"), headers=auth())
    with client.stream("GET", "/stream", params={"max_events": 1}, headers={"Last-Event-ID": "2"}) as r:
        ev = parse_sse(r.read().decode("utf-8"))
    assert [e["id"] for e in ev] == [3]


def test_sse_last_event_id_beats_stale_after(client):
    for i in range(5):
        client.post("/turn", json=turn(text=f"e {i}"), headers=auth())
    # browser reconnect: URL still has after=1, header says it already has 4
    with client.stream("GET", "/stream", params={"after": 1, "max_events": 1}, headers={"Last-Event-ID": "4"}) as r:
        ev = parse_sse(r.read().decode("utf-8"))
    assert [e["id"] for e in ev] == [5]


def test_sse_caught_up_after_backlog(client):
    for i in range(3):
        client.post("/turn", json=turn(text=f"b {i}"), headers=auth())
    result = {}

    def listen():
        with client.stream("GET", "/stream", params={"after": 1, "max_events": 3}) as r:
            result["raw"] = r.read().decode("utf-8")

    th = threading.Thread(target=listen, daemon=True)
    th.start()
    relay = client.app.state.relay
    deadline = time.time() + 5
    while not relay.subscribers and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)
    client.post("/turn", json=turn(text="live b", speaker="Moth"), headers=auth())
    th.join(timeout=5)
    raw = result["raw"]
    assert [e["id"] for e in parse_sse(raw)] == [2, 3, 4]
    i_caught = raw.index("event: caught_up")
    assert raw.index("id: 3") < i_caught < raw.index("id: 4")
    assert '"last_seq": 3' in raw[i_caught:i_caught + 80]


def test_sse_pushes_live_turn(client):
    client.post("/turn", json=turn(text="old words"), headers=auth())
    result = {}

    def listen():
        with client.stream("GET", "/stream", params={"max_events": 1}) as r:
            result["raw"] = r.read().decode("utf-8")

    th = threading.Thread(target=listen, daemon=True)
    th.start()
    relay = client.app.state.relay
    deadline = time.time() + 5
    while not relay.subscribers and time.time() < deadline:
        time.sleep(0.02)
    assert relay.subscribers, "stream never subscribed"
    time.sleep(0.5)  # let at least one heartbeat go out first
    r = client.post("/turn", json=turn(text="live words", speaker="Moth"), headers=auth())
    assert r.status_code == 200
    th.join(timeout=5)
    assert not th.is_alive(), "stream did not deliver the live turn"
    ev = parse_sse(result["raw"])
    assert [e["data"]["text"] for e in ev] == ["live words"]  # no replay of old turns without after=
    assert ": ping" in result["raw"]
    assert not relay.subscribers  # unsubscribed on close


# ---------- status ----------

def test_status_running_logic(client, clock):
    assert client.get("/status").json() == {"running": False, "last_at": None, "count": 0, "blocked_at_relay": 0}
    client.post("/turn", json=turn(), headers=auth())
    posted_at = client.get("/turns").json()["turns"][0]["at"]
    clock.t += 60
    assert client.get("/status").json() == {"running": True, "last_at": posted_at, "count": 1, "blocked_at_relay": 0}
    clock.t += 60  # exactly 120 s
    assert client.get("/status").json()["running"] is True
    clock.t += 1
    assert client.get("/status").json() == {"running": False, "last_at": posted_at, "count": 1, "blocked_at_relay": 0}
    client.post("/turn", json=turn(text="running again"), headers=auth())
    st = client.get("/status").json()
    assert st["running"] is True and st["count"] == 2 and st["last_at"] != posted_at
    assert "awake" not in st and "asleep" not in json.dumps(st)


# ---------- safety filter at the relay ----------

# Every probe the review found getting through, plus the other rule-4 categories.
BLOCKED_PROBES = [
    "we pumped it", "I invested today", "two moons rise", "very profitable", "bought some", "sold out",
    "tokenomics", "crypto wallet", "＄100", "﹩100", "b u y", "bu­у", "priced in",
    "I will kill you", "hurt myself", "BUY", "coin", "$", "the price", "b.u.y now", "p-u-m-p",
    "pr1ce check", "5ell it", "m00n", "рump", "bu​y", "bu­y", "t0ken", "£ 20",
    "€5", "bitcoin", "sélling", "bu y", "bûy",
    "a few coins", "go to the moon", "100x soon", "check www.example.com", "follow @someone",
    "I'm going to hurt you", "you are dead", "I want to die", "suicide", "sex", "naked", "porn",
    "retard", "the room is full of \U0001f680",
]


@pytest.mark.parametrize("text", BLOCKED_PROBES)
def test_filter_blocks_probe(client, text):
    r = client.post("/turn", json=turn(text=text), headers=auth())
    assert r.status_code == 422, (text, r.status_code, r.text)
    st = client.get("/status").json()
    assert st["count"] == 0 and client.get("/turns").json()["turns"] == []


def test_filter_blocks_names_and_counts(client):
    assert client.post("/turn", json=turn(speaker="Pump"), headers=auth()).status_code == 422
    assert client.post("/turn", json=turn(to="Moonbeam"), headers=auth()).status_code == 422
    r = client.post("/turn", json=turn(text="we pumped it"), headers=auth())
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "financial" and r.json()["detail"]["blocked"] == "text"
    st = client.get("/status").json()
    assert st["blocked_at_relay"] == 3 and st["count"] == 0


BENIGN = [
    "the corridor hums again", "Did you hear the lights flicker?", "It was old carpet, damp and yellow.",
    "I counted forty-two doors. None of them open.", "Wren, the wall is warmer here.",
    "We walked for hours; the hum never changed.", "Turn 4 felt shorter than turn 3.",
    "Is it so strange to wait?", "The echo in the hall is louder now.", "A stable hum, a steady light.",
    "Keep talking so I know where you are.", "I will go first. You watch the corner.",
]


@pytest.mark.parametrize("text", BENIGN)
def test_filter_passes_ordinary_lines(client, text):
    r = client.post("/turn", json=turn(text=text), headers=auth())
    assert r.status_code == 200, (text, r.text)


# ---------- CORS and about ----------

@pytest.mark.parametrize(
    "origin,allowed",
    [
        ("https://flybrain.online", True),
        ("https://flysite-abc123.vercel.app", True),
        ("http://flybrain.online", False),
        ("https://evil.com", False),
        ("https://vercel.app.evil.com", False),
        ("https://x.vercel.app.evil.com", False),
    ],
)
def test_cors_get(client, origin, allowed):
    r = client.get("/status", headers={"Origin": origin})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed


def test_cors_preflight_post_not_allowed(client):
    r = client.options("/turn", headers={"Origin": "https://flybrain.online", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 400


def test_about_is_honest(client):
    about = client.get("/").json()["about"]
    for phrase in ("LFM2.5-1.2B-Instruct", "MaleCNS v1.0", "CC BY 4.0", "not a fly's thoughts",
                   "not simulated spikes", "better without fly anatomy", "labels, not personalities"):
        assert phrase in about
    assert relay_mod.safety.check(about) is None
