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
