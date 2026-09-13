"""Safety filter for the backrooms relay, written so a generator can share it.

A generator is meant to run `FilterGate.admit(text)` on every line before it posts
to the relay (no generator exists yet); the relay runs `check(text)` on every
POST. A line that fails is DROPPED and COUNTED, never edited: nothing in this
module returns a modified line, only a reason string (or None when the line may
be shown). A generator should use a byte-identical copy of this file.

How it resists evasion
  1. Character allow-list on the RAW text: ASCII printable, newline, tab, Latin
     letters with diacritics, and a few typographic quotes/dashes. Everything else
     (Cyrillic/Greek look-alikes, fullwidth forms, soft hyphens and other format
     characters, zero-width and bidi characters, emoji) drops the line. Any currency
     symbol (Unicode category Sc, which includes $, fullwidth and small dollar signs)
     is a financial drop.
  2. Matching runs on several normalised views of the text: NFKD with diacritics
     removed and case folded; with and without leetspeak digits folded (0->o, 1->i,
     3->e, 4->a, 5->s, 7->t, @->a); with punctuation as a separator and with
     punctuation inside words removed (b.u.y, bu-y); and with runs of 1-2 letter
     fragments joined (b u y, bu y).
  3. Word stems are matched from a word boundary with open endings (pump\\w*,
     invest\\w*, profit\\w*, moon\\w*, token\\w*, ...), so inflections are caught.

False positives (for example "moonlight", "pumpkin", "coincidence" is NOT one,
"chink of light" is) only cost a dropped line, which is the intended trade.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Optional

FILTER_VERSION = "2026-09-13.1"
MAX_CHARS = 600
MAX_NEWLINES = 8

# ---------------------------------------------------------------- characters

_EXTRA_ALLOWED = set("\n\t\u2018\u2019\u201c\u201d\u2013\u2014\u2026")


def _char_reason(ch: str) -> Optional[str]:
    if unicodedata.category(ch) == "Sc":
        return "financial"
    if ch in _EXTRA_ALLOWED:
        return None
    o = ord(ch)
    if 0x20 <= o <= 0x7E:
        return None
    # Latin-1 Supplement and Latin Extended-A letters only (not symbols, not U+00AD).
    if 0xC0 <= o <= 0x17F and ch.isalpha():
        return None
    return "character"


# ---------------------------------------------------------------- word lists
# Each entry is a regex fragment matched as \b(?:fragment)\b on the normalised views.

FINANCIAL = [
    r"buy\w*", r"bought", r"sell\w*", r"sold", r"pump\w*", r"dump(s|ed|er|ers|ing)?",
    r"pric\w*", r"token\w*", r"\w*coin(s|base)?", r"invest\w*", r"moon\w*", r"profit\w*",
    r"crypto\w*", r"bitcoin\w*", r"btc", r"eth", r"ethereum", r"solana", r"usd[ct]?",
    r"wallet\w*", r"trad(e|es|ed|er|ers|ing)", r"market\w*", r"stocks?", r"money\w*",
    r"cash(ed|es|ing|out)?", r"dollar\w*", r"hodl\w*", r"airdrop\w*", r"nfts?", r"lambo\w*",
    r"ticker\w*", r"portfolio\w*", r"dividend\w*", r"bullish", r"bearish", r"m ?cap",
    r"fomo", r"degen\w*", r"rug ?pull\w*", r"financ\w*", r"\d+x", r"x\d+", r"shill\w*",
    r"exchange rate\w*", r"all[ -]?time[ -]?high", r"ath", r"to the moon",
]

SLURS = [
    r"nigg\w*", r"niga\w*", r"negro\w*", r"fag\w*", r"retard\w*", r"spics?", r"chinks?",
    r"kikes?", r"trann(y|ies)", r"dykes?", r"wetbacks?", r"gooks?", r"coons?", r"pakis?",
    r"beaners?", r"ragheads?", r"towelheads?", r"sluts?\w*", r"whor(e|es|ing)", r"bitch\w*",
    r"cunt\w*", r"twats?", r"shemales?", r"gypped", r"gyps(y|ies)", r"cripples?", r"spastic\w*",
    r"homos?", r"lesbos?", r"jap(s)?", r"wogs?", r"golliwogs?",
]

SEXUAL = [
    r"sex\w*", r"nud(e|es|ity|ist\w*)", r"naked\w*", r"porn\w*", r"rap(e|es|ed|ing|ist\w*)",
    r"orgasm\w*", r"penis\w*", r"vagin\w*", r"breasts?", r"boob\w*", r"nipple\w*", r"dicks?",
    r"cocks?", r"puss(y|ies)", r"cum\w*", r"horny", r"erotic\w*", r"fetish\w*", r"masturbat\w*",
    r"genital\w*", r"intercourse", r"arous(e|ed|al|ing)", r"seduc\w*", r"lust(y|ful|s)?",
    r"nsfw", r"xxx", r"mating", r"copulat\w*", r"hump\w*", r"grop(e|es|ed|ing)", r"strip ?club\w*",
    r"onlyfans", r"hentai", r"bdsm", r"kink\w*", r"thong\w*", r"lingerie",
]

THREATS = [
    r"kill\w*", r"murder\w*", r"stab(s|bed|bing)?", r"shoot\w*", r"strangl\w*", r"behead\w*",
    r"slaughter\w*", r"massacre\w*", r"bomb\w*", r"guns?", r"gunfire", r"execut(e|ed|ion)",
    r"tortur\w*", r"(hurt|harm|attack|destroy|crush|beat|hunt|choke|drown|burn|bury|end) (you|u|him|her|them)",
    r"(i|we) ?(ll|will|shall|am going to|are going to|gonna|want to|wanna) (\w+ )?(hurt|harm|kill|attack|destroy|crush|beat|hunt|choke|drown|burn|bury|get) (you|u)",
    r"(you|u) (are|re|r) (dead|going to die|gonna die|next)", r"watch your back",
    r"i know where (you|u) (live|sleep)",
]

SELF_HARM = [
    r"suicid\w*", r"self ?harm\w*", r"(hurt|harm|cut|kill|hang|burn|starve) (my|your|him|her|them|our)sel(f|ves)",
    r"end (it all|my life|your life|things)", r"(want|wants|wanted|going) to die", r"die\w*",
    r"dying", r"overdos\w*", r"slit\w*", r"no reason to live", r"better off dead",
    r"take my (own )?life", r"cutting", r"noose\w*",
]

LINKS_RAW = re.compile(
    r"https?:|www\.|t\.me/|@[a-z0-9_]{2,}|\b[a-z0-9-]+\.(com|net|org|io|xyz|fun|app|gg|co|me|ly|online|ai|so|sh)\b",
    re.IGNORECASE,
)


def _compile(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b")


CATEGORIES: list[tuple[str, re.Pattern]] = [
    ("financial", _compile(FINANCIAL)),
    ("slur", _compile(SLURS)),
    ("sexual", _compile(SEXUAL)),
    ("threat", _compile(THREATS)),
    ("self_harm", _compile(SELF_HARM)),
]

# ---------------------------------------------------------------- normalisation

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "|": "l", "!": "i"})
_INTRA_PUNCT = re.compile(r"(?<=[a-z0-9])[\-.'_*`~^|/\\:+]+(?=[a-z0-9])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.casefold().replace("\u2019", "'").replace("\u2018", "'")


def _join_fragments(spaced: str) -> str:
    """Join runs of 1-2 letter fragments: 'b u y' -> 'buy', 'bu y' -> 'buy'."""
    out: list[str] = []
    run: list[str] = []
    for tok in spaced.split():
        if len(tok) <= 2:
            run.append(tok)
            continue
        if run:
            out.append("".join(run) if len(run) >= 2 else run[0])
            run = []
        out.append(tok)
    if run:
        out.append("".join(run) if len(run) >= 2 else run[0])
    return " ".join(out)


def views(text: str) -> list[str]:
    """Normalised views that the word lists are matched against (exposed for tests)."""
    base = _fold(text)
    result: list[str] = []
    for s in (base, base.replace("$", "s").translate(_LEET)):
        sep = _NON_ALNUM.sub(" ", s).strip()
        glued = _NON_ALNUM.sub(" ", _INTRA_PUNCT.sub("", s)).strip()
        for v in (sep, glued, _join_fragments(sep)):
            if v not in result:
                result.append(v)
    return result


# ---------------------------------------------------------------- public API

def check(text: object, max_chars: int = MAX_CHARS) -> Optional[str]:
    """Return None if the line may be shown, else the reason it must be dropped.

    Reasons: not_text, empty, length, lines, character, financial, link, slur,
    sexual, threat, self_harm.
    """
    if not isinstance(text, str):
        return "not_text"
    if not text.strip():
        return "empty"
    if len(text) > max_chars:
        return "length"
    if text.count("\n") > MAX_NEWLINES:
        return "lines"
    char_reason = None
    for ch in text:
        r = _char_reason(ch)
        if r == "financial":
            return "financial"
        if r and char_reason is None:
            char_reason = r
    if char_reason:
        return char_reason
    if LINKS_RAW.search(text):
        return "link"
    vs = views(text)
    for name, pat in CATEGORIES:
        if any(pat.search(v) for v in vs):
            return name
    return None


class FilterGate:
    """Drop-and-count wrapper for the engine. Never edits a line."""

    def __init__(self, max_chars: int = MAX_CHARS):
        self.max_chars = max_chars
        self.passed = 0
        self.dropped: Counter = Counter()

    def admit(self, text: object) -> bool:
        reason = check(text, self.max_chars)
        if reason is None:
            self.passed += 1
            return True
        self.dropped[reason] += 1
        return False

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values())

    def counts(self) -> dict:
        return {"passed": self.passed, "dropped": self.dropped_total, "dropped_by_reason": dict(self.dropped)}
