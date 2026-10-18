"""Rule-made training texts for ext-2 (ideas 3-6; the ideas are credited to Delvefall).

Each generator maps a fit card's text to a variant with a known relation to it:
  flip     (3) one mechanism word swapped (tap/untap, enters/dies, ...): a hard negative
  opener   (4) a real card with the same trigger opener and no shared tag: a hard negative
  numbers  (5) the same text with only its numbers changed: a positive
  wording  (6) older Oracle wording for the same rule: a positive
Pure functions of the text and a seed; the caller removes exam families (exam.exam_cards).
"""

from __future__ import annotations

import re

import numpy as np

from .exam import FLIPS, opener
from .util import sha256_json

# "you"/"opponent" makes ungrammatical text ("opponent control"), so it is mined, not made
SYNTH_FLIPS = [p for p in FLIPS if p != ("you", "opponent")]
_FLIP_OF = {a: b for a, b in SYNTH_FLIPS} | {b: a for a, b in SYNTH_FLIPS}
_FLIP_RE = re.compile(r"(?<![\w+/-])(" + "|".join(
    re.escape(w) for w in sorted(_FLIP_OF, key=len, reverse=True)) + r")(?![\w/])", re.I)
_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven"]
# a number not inside "+1/+1"-style counters or power/toughness ("2/2")
_NUMBER = re.compile(r"(?<![\w+/-])(\d+|" + "|".join(_WORDS[1:11]) + r")(?![\w/])", re.I)
WORDING = [(re.compile(r"\benters(?! the battlefield)\b"), "enters the battlefield"),
           (re.compile(r"\bdies\b"), "is put into a graveyard from the battlefield"),
           (re.compile(r"\bmana value\b"), "converted mana cost"),
           (re.compile(r"\bcan't be blocked\b"), "is unblockable")]


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(int(sha256_json(["augment", *key])[:15], 16))


def flip(text: str, seed: int, key: str) -> str | None:
    """Swap one seeded mechanism word (case kept); None if the text has none."""
    hits = list(_FLIP_RE.finditer(text))
    if not hits:
        return None
    m = hits[int(_rng("flip", seed, key).integers(len(hits)))]
    w = _FLIP_OF[m.group(0).lower()]
    w = w[0].upper() + w[1:] if m.group(0)[0].isupper() else w
    return text[: m.start()] + w + text[m.end():]


def numbers(text: str) -> str | None:
    """Every free number moved up by one (digits and number words); None if none."""
    def bump(m):
        t = m.group(0)
        if t.isdigit():
            return str(int(t) + 1)
        i = _WORDS.index(t.lower()) + 1
        return _WORDS[i].capitalize() if t[0].isupper() else _WORDS[i]
    out = _NUMBER.sub(bump, text)
    return out if out != text else None


def wording(text: str) -> str | None:
    """Every older-wording rule applied; None if no rule applies."""
    out = text
    for pat, old in WORDING:
        out = pat.sub(old, out)
    return out if out != text else None


def opener_negatives(ids: list[str], texts: list[str], families: list[str],
                     tags: dict[str, frozenset[str]], seed: int) -> dict[str, str]:
    """One-line card -> a seeded real card with the same opener, another family and no
    shared tag. Cards without such a partner are left out."""
    by_open: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        if "\n" not in t and (o := opener(t)):
            by_open.setdefault(o, []).append(i)
    out = {}
    for members in by_open.values():
        for i in members:
            cand = [j for j in members if families[j] != families[i]
                    and not tags.get(ids[i], frozenset()) & tags.get(ids[j], frozenset())]
            if cand:
                out[ids[i]] = ids[cand[int(_rng("opener", seed, ids[i]).integers(len(cand)))]]
    return out
