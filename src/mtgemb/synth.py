"""Fit-only rule-made positive/negative rows (ideas 3-6; exam exclusions)."""

from __future__ import annotations

import numpy as np

from . import augment
from .data import tag_sets
from .util import sha256_json


def build(ds: dict, elig: set[str], exclude: set[str], seed: int = 0) -> dict:
    """One seeded positive and at most one negative per eligible fit card.

    elig contains pair-eligible tag IDs, as in TrainData and linetag.build.
    Opener partners use the same fit/eligibility/exam-family filters as anchors.
    """
    cards = ds["cards"]
    fams = set(cards.loc[cards["oracle_id"].isin(exclude), "family_id"])
    mem = ds["memberships"]
    eligible_ids = set(mem.loc[mem["tag_id"].isin(elig), "oracle_id"])
    tags = tag_sets(mem)  # shared broad functional tags still rule out an opener negative
    fit = cards[(cards["split"] == "fit") & (cards["text_status"] == "ok")
                & cards["oracle_id"].isin(eligible_ids) & ~cards["family_id"].isin(fams)]
    fit = fit.sort_values("oracle_id")
    ids, texts, families = (fit[c].tolist() for c in ("oracle_id", "text", "family_id"))
    openers = augment.opener_negatives(ids, texts, families, tags, seed)
    text_of = dict(zip(ids, texts, strict=True))
    rng = np.random.default_rng(int(sha256_json(["synth-build", seed])[:15], 16))
    rows = []
    for oid, text, family in zip(ids, texts, families, strict=True):
        positives = [(kind, p) for kind, p in (("numbers", augment.numbers(text)),
                                               ("wording", augment.wording(text)))
                     if p is not None and p != text]
        if not positives:
            continue
        kind, positive = positives[int(rng.integers(len(positives)))]
        negative = augment.flip(text, seed, key=oid)
        if negative is None or negative == text:
            negative = text_of.get(openers.get(oid))
        rows.append({"anchor_id": oid, "anchor": text, "positive": positive,
                     "negative": negative, "family": family, "positive_kind": kind})
    return {"anchor_ids": [r["anchor_id"] for r in rows],
            "anchors": [r["anchor"] for r in rows],
            "positives": [r["positive"] for r in rows],
            "negatives": [r["negative"] for r in rows],
            "families": [r["family"] for r in rows],
            "positive_kinds": [r["positive_kind"] for r in rows],
            "hash": sha256_json(rows)[:16]}


def schedule(rows: dict, batch: int, updates: int, data_seed: int) -> list[np.ndarray]:
    """Cycle shuffled row permutations, dropping each short epoch tail."""
    n = len(rows["anchors"])
    if batch < 1 or n < batch:
        raise ValueError("synth rows cannot fill one batch")
    rng = np.random.default_rng([data_seed, 7919])
    out = []
    while len(out) < updates:
        perm = rng.permutation(n)
        for start in range(0, n - batch + 1, batch):
            out.append(perm[start:start + batch])
            if len(out) == updates:
                break
    return out
