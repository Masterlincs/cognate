"""Line-to-tag training rows (ideas 1 and 2; the ideas are credited to Delvefall).

A one-line card's tags all belong to its single ability line, so each (line, tag) is a
labelled example nobody had to annotate. Fit cards only; trap-exam families are excluded;
tags are the pair-eligible functional tags (fit df 2..500). Built deterministically
on load; the row hash enters the run identity.
"""

from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp

from .abilities import split_abilities
from .util import sha256_json

TAG_CAP = 64  # rows per tag at most (seeded choice); keeps broad tags from dominating
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def tag_text(label: str, description: str | None) -> str:
    """Label plus the first sentence of the description, markdown links reduced to text."""
    if not isinstance(description, str) or not description.strip():
        return label
    first = re.split(r"(?<=[.!?])\s", _LINK.sub(r"\1", description).strip(), maxsplit=1)[0]
    return f"{label}: {first}"


def build(ds: dict, elig: set[str], exclude: set[str], seed: int = 0) -> dict:
    cards = ds["cards"]
    # exclude whole families: a same-text twin of an exam card would leak its line
    fams = set(cards.loc[cards["oracle_id"].isin(exclude), "family_id"])
    fit = cards[(cards["split"] == "fit") & (cards["text_status"] == "ok")
                & ~cards["family_id"].isin(fams)].sort_values("oracle_id")
    fit = fit[fit["text"].map(lambda t: len(split_abilities(t)) == 1)].reset_index(drop=True)
    mem = ds["memberships"]
    m = mem[mem["oracle_id"].isin(set(fit["oracle_id"])) & mem["tag_id"].isin(elig)]
    m = m[["oracle_id", "tag_id"]].drop_duplicates().sort_values(["tag_id", "oracle_id"])
    rng = np.random.default_rng(int(sha256_json(["linetag-cap", seed])[:15], 16))
    keep = []
    for _, g in m.groupby("tag_id", sort=True):
        idx = g.index.to_numpy()
        keep.append(idx if len(idx) <= TAG_CAP else np.sort(rng.choice(idx, TAG_CAP, False)))
    m = m.loc[np.concatenate(keep)] if keep else m.iloc[:0]
    lines = fit["text"].tolist()  # one line each
    lpos = {o: i for i, o in enumerate(fit["oracle_id"])}
    tags = ds["tags"].set_index("tag_id")
    tag_ids = sorted(set(m["tag_id"]))
    tpos = {t: j for j, t in enumerate(tag_ids)}
    texts = [tag_text(tags.at[t, "label"], tags.at[t, "description"]) for t in tag_ids]
    li = m["oracle_id"].map(lpos).to_numpy()
    tj = m["tag_id"].map(tpos).to_numpy()
    # every eligible tag a line carries (not only the capped rows), for in-batch masking
    full = mem[mem["oracle_id"].isin(lpos) & mem["tag_id"].isin(tpos)]
    carries = sp.csr_matrix((np.ones(len(full), dtype=np.float32),
                             (full["oracle_id"].map(lpos), full["tag_id"].map(tpos))),
                            shape=(len(lines), len(tag_ids)))
    return {"lines": lines, "line_ids": fit["oracle_id"].tolist(), "tag_ids": tag_ids,
            "tag_texts": texts, "row_line": li, "row_tag": tj, "carries": carries,
            "hash": sha256_json([fit["oracle_id"].tolist(), tag_ids, texts,
                                 li.tolist(), tj.tolist()])[:16]}


def schedule(aux: dict, batch: int, updates: int, data_seed: int) -> list[np.ndarray]:
    """Row indices per update: no tag and no line twice in a batch; epochs reshuffle."""
    n = len(aux["row_line"])
    out, epoch = [], 0
    while len(out) < updates:
        rng = np.random.default_rng(int(sha256_json(["linetag", data_seed, epoch])[:15], 16))
        queue, deferred, made = iter(rng.permutation(n).tolist()), [], 0
        while len(out) < updates:
            rows, lines, tags, keep = [], set(), set(), []

            def add(r, rows=rows, lines=lines, tags=tags) -> bool:
                ln, tg = int(aux["row_line"][r]), int(aux["row_tag"][r])
                if ln in lines or tg in tags:
                    return False
                rows.append(r)
                lines.add(ln)
                tags.add(tg)
                return True

            for r in deferred:
                if not (len(rows) < batch and add(r)):
                    keep.append(r)
            deferred = keep
            while len(rows) < batch and (r := next(queue, None)) is not None:
                if not add(r):
                    deferred.append(r)
            if len(rows) < batch:
                break  # epoch tail
            out.append(np.array(rows))
            made += 1
        if not made:
            raise ValueError("line-tag rows cannot fill one batch")
        epoch += 1
    return out


def masks(aux: dict, rows: np.ndarray) -> np.ndarray:
    """known[i, j]: row i's line also carries row j's tag (j != i). Transpose for tag->line."""
    li, tj = aux["row_line"][rows], aux["row_tag"][rows]
    known = aux["carries"][li][:, tj].toarray() > 0
    np.fill_diagonal(known, False)
    return known
