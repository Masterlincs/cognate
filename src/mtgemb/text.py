"""`no_reminder` v1 text representation (MASTER_PLAN §3).

Pure functions only: the encoder input is exactly what `no_reminder_v1` returns.
"""

from __future__ import annotations

import re

VERSION = "no_reminder_v1"
FACE_BOUNDARY = "\n//\n"

_SPACES = re.compile(r"[ \t ]+")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:])")


def strip_parentheticals(text: str) -> tuple[str | None, dict]:
    """Remove every balanced (...) span, nested spans included.

    Returns (stripped, info). If the parentheses are unbalanced the text is left
    untouched and `stripped` is None so the caller can quarantine it; we never
    delete everything after an unmatched bracket.
    """
    out: list[str] = []
    depth = 0
    spans = nested = whole_lines = 0
    span_start = 0
    removed: list[str] = []
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                span_start = i
                spans += 1
            else:
                nested += 1
            depth += 1
        elif ch == ")":
            if depth == 0:
                return None, {"unbalanced": True, "reason": f"unmatched ')' at {i}"}
            depth -= 1
            if depth == 0:
                removed.append(text[span_start : i + 1])
        elif depth == 0:
            out.append(ch)
    if depth:
        return None, {"unbalanced": True, "reason": f"{depth} unclosed '('"}
    for line in text.split("\n"):
        # a whole parenthetical ability: non-empty line with nothing left outside its spans
        kept, d = [], 0
        for ch in line:
            d += ch == "("
            if d == 0:
                kept.append(ch)
            d -= ch == ")"
        whole_lines += bool(line.strip()) and not "".join(kept).strip()
    info = {
        "unbalanced": False,
        "spans": spans,
        "nested": nested,
        "whole_line_spans": whole_lines,
        "removed": removed,
    }
    return "".join(out), info


def normalise_whitespace(text: str) -> str:
    """Deterministic whitespace cleanup: LF line endings, single spaces, no space before
    , . ; : , trimmed lines, no empty lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = _SPACE_BEFORE_PUNCT.sub(r"\1", _SPACES.sub(" ", line)).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def no_reminder_v1(face_texts: list[str]) -> dict:
    """Transform ordered face rules texts into the v1 encoder text.

    status: "ok" (non-empty text), "empty" (nothing left; discoverable but no vector),
    or "quarantined" (unbalanced parentheses; excluded from embedding pending review).
    """
    raw = FACE_BOUNDARY.join(t or "" for t in face_texts)
    parts: list[str] = []
    stats = {"spans": 0, "nested": 0, "whole_line_spans": 0, "removed": []}
    for t in face_texts:
        stripped, info = strip_parentheticals(t or "")
        if stripped is None:
            return {"raw": raw, "text": None, "status": "quarantined", "reason": info["reason"]}
        for k in ("spans", "nested", "whole_line_spans"):
            stats[k] += info[k]
        stats["removed"] += info["removed"]
        face = normalise_whitespace(stripped)
        if face:
            parts.append(face)
    text = FACE_BOUNDARY.join(parts)
    status = "ok" if text else "empty"
    reason = None
    if not text:
        reason = "no rules text" if not raw.replace(FACE_BOUNDARY, "").strip() else (
            "only parenthetical text"
        )
    return {"raw": raw, "text": text, "status": status, "reason": reason, **stats}


_WORD = re.compile(r"[a-z0-9]+")


def word_tokens(text: str) -> frozenset[str]:
    """Frozen tokenisation for the hard-lexical view: lowercase [a-z0-9]+ runs."""
    return frozenset(_WORD.findall(text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Word-token Jaccard. Defined as 0.0 when either set is empty (empty texts are
    excluded before ranking anyway)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------- no_reminder_v2 (#7)

VERSION_V2 = "no_reminder_v2"
_SUBTYPE_NOUNS = ("Equipment", "Aura", "Vehicle", "Saga", "Class", "Case", "Room", "Siege")
_TYPE_NOUNS = (("Creature", "creature"), ("Planeswalker", "planeswalker"),
               ("Battle", "battle"), ("Land", "land"), ("Artifact", "artifact"),
               ("Enchantment", "enchantment"), ("Instant", "spell"), ("Sorcery", "spell"))


def self_noun(type_line: str) -> str:
    """Oracle-style self reference for one face: "this creature", "this Equipment", ...

    Creature wins (bestow Auras and Vehicles that are creatures read "this creature"), then
    the capitalised subtypes Oracle uses, then the card type; anything else is "this card".
    """
    types, _, subs = (type_line or "").partition("—")
    if "Creature" in types:
        return "this creature"
    for s in _SUBTYPE_NOUNS:
        if s in subs.split():
            return f"this {s}"
    for t, noun in _TYPE_NOUNS:
        if t in types:
            return f"this {noun}"
    return "this card"


def replace_self_names(text: str, names: list[str], noun: str) -> str:
    """Own names (longest first) -> noun; "named <Name>" is a name reference and is kept.

    Limitation: a bare word match; a card whose name is also an ordinary rules word would
    be rewritten wherever that word appears. Measured before use.
    """
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(rf"(?<!named )(?<!\w){re.escape(n)}(?!\w)", noun, text)
    # sentence starts read "This creature ..."
    return re.sub(rf"(^|(?<=[.!?] )|(?<=: )){re.escape(noun)}", "T" + noun[1:], text,
                  flags=re.M)


def no_reminder_v2(faces: list[dict], names: list[str]) -> dict:
    """v1 after replacing each face's own names with that face's Oracle-style noun.

    `faces`: dicts with oracle_text and type_line; `names`: the card's self names (full
    name, face names, the part before ", "), as data.self_names returns them.
    """
    return no_reminder_v1([replace_self_names(f.get("oracle_text") or "", names,
                                              self_noun(f.get("type_line", "")))
                           for f in faces])
