"""Which of a card's tags belongs to which line (idea 8, credited to Delvefall).

A tag's centroid is the mean deployed vector of the fit one-line cards that carry it (a
one-line card's tags all belong to its line). Each tag of a multi-line card goes to the line
nearest its centroid. Tags with fewer than MIN_SUPPORT one-line examples stay card-level.

Check (automatic, no labels invented): a "twin" is a multi-line card line whose number-masked
text equals a one-line card's whole text. When exactly one of a card's lines has a twin that
carries tag t, and the card carries t, that line is t's gold line. The twin is left out of
t's centroid for this check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .abilities import split_abilities
from .exam import mask_numbers

MIN_SUPPORT = 3


def centroids(ids: list[str], texts: list[str], split: list[str], vectors: np.ndarray,
              mem: pd.DataFrame, tag_ids: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """(sums, counts) per tag over fit one-line cards, plus each tag's member rows."""
    pos = {o: i for i, o in enumerate(ids)}
    one = {i for i, (t, s) in enumerate(zip(texts, split, strict=True))
           if s == "fit" and len(split_abilities(t)) == 1}
    tpos = {t: j for j, t in enumerate(tag_ids)}
    sums = np.zeros((len(tag_ids), vectors.shape[1]))
    counts = np.zeros(len(tag_ids), dtype=int)
    members: dict[int, set[int]] = {}
    for o, t in zip(mem["oracle_id"], mem["tag_id"], strict=True):
        i, j = pos.get(o), tpos.get(t)
        if i in one and j is not None:
            sums[j] += vectors[i]
            counts[j] += 1
            members.setdefault(j, set()).add(i)
    return sums, counts, members


def attribute(line_v: np.ndarray, tag_rows: list[int], sums: np.ndarray,
              counts: np.ndarray) -> dict[int, int | None]:
    """tag row -> index of the nearest line, or None when the tag lacks support."""
    out = {}
    for j in tag_rows:
        if counts[j] < MIN_SUPPORT:
            out[j] = None
            continue
        c = sums[j] / np.linalg.norm(sums[j])
        out[j] = int(np.argmax(line_v @ c))
    return out


def check(ids, texts, split, vectors, line_v, line_owner, mem, tag_ids) -> dict:
    """Accuracy on gold lines from one-line twins (leave-twin-out), vs a uniform line pick."""
    pos = {o: i for i, o in enumerate(ids)}
    tpos = {t: j for j, t in enumerate(tag_ids)}
    sums, counts, members = centroids(ids, texts, split, vectors, mem, tag_ids)
    one_line = {mask_numbers(t): i for i, t in enumerate(texts) if len(split_abilities(t)) == 1}
    card_tags: dict[int, set[int]] = {}
    for o, t in zip(mem["oracle_id"], mem["tag_id"], strict=True):
        if o in pos and t in tpos:
            card_tags.setdefault(pos[o], set()).add(tpos[t])
    starts = np.flatnonzero(np.r_[True, line_owner[1:] != line_owner[:-1]])
    hit = n = chance = 0.0
    for r, t in enumerate(texts):
        lines = split_abilities(t)
        if len(lines) < 2:
            continue
        twins = [one_line.get(mask_numbers(x)) for x in lines]
        for j in card_tags.get(r, ()):
            gold = [k for k, w in enumerate(twins) if w is not None and j in card_tags.get(w, ())]
            if len(gold) != 1:
                continue
            w = twins[gold[0]]
            s, c = sums[j].copy(), counts[j]
            if w in members.get(j, ()):
                s, c = s - vectors[w], c - 1
            if c < MIN_SUPPORT:
                continue
            lv = line_v[starts[r]: starts[r] + len(lines)]
            hit += int(np.argmax(lv @ (s / np.linalg.norm(s)))) == gold[0]
            n += 1
            chance += 1 / len(lines)
    return {"n": int(n), "accuracy": hit / n if n else None,
            "uniform_chance": chance / n if n else None}


def release_tags(texts: list[str], split: list[str], ids: list[str], vectors: np.ndarray,
                 line_v: np.ndarray, mem: pd.DataFrame, tag_ids: list[str]) -> dict:
    """Per text row: all its tags (card level) and, for multi-line rows, the tags on each line.

    Returns {"tag_ids": used tags (sorted), "card": [[tag index...]], "lines": [[[...]...]]};
    single-line rows have lines == []. Indices point into the returned tag_ids.
    """
    sums, counts, _ = centroids(ids, texts, split, vectors, mem, tag_ids)
    pos = {o: i for i, o in enumerate(ids)}
    tpos = {t: j for j, t in enumerate(tag_ids)}
    per: list[set[int]] = [set() for _ in ids]
    for o, t in zip(mem["oracle_id"], mem["tag_id"], strict=True):
        if o in pos and t in tpos:
            per[pos[o]].add(tpos[t])
    used = sorted(set().union(*per))
    remap = {j: k for k, j in enumerate(used)}
    card, lines, start = [], [], 0
    for r, t in enumerate(texts):
        m = len(split_abilities(t))
        card.append(sorted(remap[j] for j in per[r]))
        if m >= 2:
            got = attribute(line_v[start:start + m], sorted(per[r]), sums, counts)
            lines.append([sorted(remap[j] for j, li in got.items() if li == k)
                          for k in range(m)])
        else:
            lines.append([])
        start += m
    if start != len(line_v):
        raise ValueError("line vectors do not match the texts")
    return {"tag_ids": [tag_ids[j] for j in used], "card": card, "lines": lines}
