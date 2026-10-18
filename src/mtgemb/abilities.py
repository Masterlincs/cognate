"""Per-ability neighbours.

An ability is one non-empty line of the model input text (`no_reminder_v1`); the face
boundary line "//" is not an ability. Each ability is encoded alone with the release
encoder's exact recipe (prefix, pooling, normalisation, max_length). A query ability scores
every candidate card by its best-matching ability line (max cosine); the query's own card
is excluded. Rows are catalogue text rows in Oracle ID order, so ties break by Oracle ID.
"""

from __future__ import annotations

import numpy as np


def split_abilities(text: str) -> list[str]:
    return [s for s in (x.strip() for x in text.split("\n")) if s and s != "//"]


def flatten(texts: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """All ability lines in row order, the owning row of each line, and each row's first
    line offset (for np.maximum.reduceat). Every row must have at least one ability."""
    lines, owner = [], []
    for r, t in enumerate(texts):
        a = split_abilities(t)
        if not a:
            raise ValueError(f"text row {r} has no ability line")
        lines += a
        owner += [r] * len(a)
    owner = np.array(owner, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    return lines, owner, starts


def maxsim_topk(q: np.ndarray, q_owner: np.ndarray, line_v: np.ndarray, starts: np.ndarray,
                k: int, block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k candidate rows per query line by max line cosine (unit rows in).

    Scores a (block, n_lines) slab at a time; ties -> lower row index (stable sort).
    """
    n = len(starts)
    k = min(k, n - 1)
    idx = np.empty((len(q), k), dtype=np.int64)
    sco = np.empty((len(q), k), dtype=line_v.dtype)
    for s in range(0, len(q), block):
        e = min(len(q), s + block)
        card = np.maximum.reduceat(q[s:e] @ line_v.T, starts, axis=1)
        card[np.arange(e - s), q_owner[s:e]] = -np.inf
        order = np.argsort(-card, axis=1, kind="stable")[:, :k]
        idx[s:e] = order
        sco[s:e] = np.take_along_axis(card, order, axis=1)
    return idx, sco


def maxsim_reference(q_row: np.ndarray, own: int, line_v: np.ndarray, owner: np.ndarray,
                     n: int) -> np.ndarray:
    """Independent float64 card scores for one query line (np.maximum.at, not reduceat)."""
    out = np.full(n, -np.inf)
    np.maximum.at(out, owner, line_v.astype(np.float64) @ q_row.astype(np.float64))
    out[own] = -np.inf
    return out
