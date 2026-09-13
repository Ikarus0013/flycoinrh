"""Filter tests for safety.py on its own: what it drops, what it lets through, and that
FilterGate drops and counts lines without ever editing them."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import safety  # noqa: E402


BLOCKED = [
    # financial / trading (inflections, evasions, look-alike characters)
    "buy", "we pumped it", "I invested today", "two moons rise", "very profitable", "bought some",
    "sold out", "tokenomics", "crypto wallet", "\uff04100", "\ufe69100", "$", "b u y", "bu y",
    "bu\u00ad\u0443", "bu\u00ady", "bu\u200by", "\u0440ump", "priced in", "the price", "b.u.y now",
    "p-u-m-p", "pr1ce check", "5ell it", "m00n", "t0ken", "\u00a3 20", "\u20ac5", "s\u00e9lling",
    "b\u00fby", "a few coins", "to the moon", "100x soon",
    # links
    "check www.example.com", "follow @someone",
    # threats
    "I will kill you", "I'm going to hurt you", "you are dead",
    # self-harm
    "hurt myself", "I want to die", "suicide",
    # sexual
    "sex", "naked", "porn",
    # slurs (one representative; the list lives in safety.py)
    "retard",
    # characters outside the allow-list
    "the room is full of \U0001f680",
    # structure
    "", "   ", "a" * 601, "\n".join(["x"] * 10),
]

PASS = [
    "the corridor hums again", "Did you hear the lights flicker?", "It was old carpet, damp and yellow.",
    "Keep talking so I know where you are.", "I will go first. You watch the corner.",
    "Count the tiles with me. One, two, three, four.", "The echo in the hall is louder now.",
    "A stable hum \u2014 a steady light\u2026", "Wren said \u201cleft\u201d, didn\u2019t she?",
]


@pytest.mark.parametrize("text", BLOCKED)
def test_blocked(text):
    assert safety.check(text) is not None, repr(text)


@pytest.mark.parametrize("text", PASS)
def test_passes(text):
    assert safety.check(text) is None, (text, safety.check(text))


def test_gate_drops_and_counts_never_edits():
    gate = safety.FilterGate()
    line = "we pumped it"
    assert gate.admit(line) is False
    assert line == "we pumped it"  # nothing to edit: admit returns only a boolean
    assert gate.admit("the corridor hums again") is True
    assert gate.admit(None) is False
    assert gate.counts() == {"passed": 1, "dropped": 2, "dropped_by_reason": {"financial": 1, "not_text": 1}}
