"""Fresh-card test set: cards that did not exist in the frozen snapshot.

No model, rule or threshold was fitted or selected on them, so this is the clean held-out
test left after the sealed split was spent. Fresh = Oracle IDs absent from the whole
frozen raw snapshot (every card, not only the corpus), in the corpus policy, with text,
released at least LAG_DAYS before the new snapshot (Tagger needs time). Their tags come from
the new snapshot, restricted to the frozen functional taxonomy; weights stay fit-derived.
Universe = the frozen dev universe plus the fresh cards; anchors = fresh cards with a
fit-represented tag. Nothing is built below MIN_ANCHORS.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .data import card_row, corpus_reason, load_dataset, load_raw, name_normalised
from .evaluate import BENCHMARK_VERSION, VIEWS, K
from .util import DATA, atomic_write_json, log_event, sha256_json, utcnow

LAG_DAYS = 45
MIN_ANCHORS = 200
FRESH_VIEWS = VIEWS[:-1] + ("fresh_candidates",)


def collect(ds: dict, old_raw: Path, new_raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    old_cards, _, _ = load_raw(Path(old_raw))
    new_cards, new_tags, man = load_raw(Path(new_raw))
    snap = date.fromisoformat(man["snapshot"])
    old_ids = {c["oracle_id"] for c in old_cards}
    rows, why = [], {"new_ids": 0, "excluded_by_policy": 0, "too_recent": 0, "no_text": 0}
    for c in new_cards:
        if c["oracle_id"] in old_ids:
            continue
        why["new_ids"] += 1
        if corpus_reason(c):
            why["excluded_by_policy"] += 1
            continue
        if (snap - date.fromisoformat(c["released_at"])).days < LAG_DAYS:
            why["too_recent"] += 1
            continue
        r = card_row(c)
        if r["text_status"] != "ok":
            why["no_text"] += 1
            continue
        rows.append(r)
    cards = pd.DataFrame(rows)
    if len(cards):
        # a fresh card with a frozen card's (name-normalised) text joins that card's family
        old = ds["cards"]
        key = {name_normalised(t, n, json.loads(f)): fam for t, n, f, fam in zip(
            old["text"].fillna(""), old["name"], old["faces"], old["family_id"], strict=True)}
        cards["family_id"] = [key.get(name_normalised(t, n, json.loads(f)), o) for t, n, f, o in
                              zip(cards["text"], cards["name"], cards["faces"],
                                  cards["oracle_id"], strict=True)]
        cards["split"] = "fresh"
        cards = cards.sort_values("oracle_id").reset_index(drop=True)
    functional = set(ds["tags"].loc[ds["tags"]["functional"], "tag_id"])
    ids = set(cards["oracle_id"]) if len(cards) else set()
    mem = pd.DataFrame([{"oracle_id": g["oracle_id"], "tag_id": t["id"]}
                        for t in new_tags if t.get("type") == "oracle" and t["id"] in functional
                        for g in t["taggings"] if g.get("oracle_id") in ids],
                       columns=["oracle_id", "tag_id"]).drop_duplicates()
    why.update(fresh_cards=len(cards), snapshot=man["snapshot"],
               fresh_memberships=len(mem))
    return cards, mem, why


def build(ds_dir, old_raw: Path, new_raw: Path) -> dict:
    """Write data/fresh/<snapshot>/ (cards, memberships, benchmark spec) when large enough."""
    ds = load_dataset(ds_dir)
    cards, mem, rep = collect(ds, old_raw, new_raw)
    fit_tags = set(ds["tags"].loc[ds["tags"]["df_fit"] > 0, "tag_id"])
    anchors = sorted(set(mem.loc[mem["tag_id"].isin(fit_tags), "oracle_id"]))
    rep["anchors"] = len(anchors)
    if len(anchors) < MIN_ANCHORS:
        log_event("data", "FRESH_TOO_SMALL", **rep, min_anchors=MIN_ANCHORS)
        return {"built": False, **rep}
    old = ds["cards"]
    universe = sorted(old.loc[old["text_status"] == "ok", "oracle_id"].tolist() +
                      cards["oracle_id"].tolist())
    spec = {"dataset_id": ds["ident"]["dataset_id"], "fresh_snapshot": rep["snapshot"],
            "universe": universe, "anchors": anchors,
            "rules": {"version": BENCHMARK_VERSION, "k": K, "views": FRESH_VIEWS,
                      "held_split": "fresh", "lag_days": LAG_DAYS,
                      "anchors": "fresh cards with >=1 fit-represented tag",
                      "weights": "frozen fit idf; fresh tags from the new snapshot"},
            "counts": rep, "created_utc": utcnow()}
    spec["benchmark_id"] = sha256_json([spec["dataset_id"], universe, anchors,
                                        spec["rules"], rep["snapshot"]])[:16]
    out = DATA / "fresh" / rep["snapshot"]
    if (out / "benchmark_fresh.json").exists():
        raise FileExistsError(f"{out} is already built; a change is a new version")
    out.mkdir(parents=True, exist_ok=True)
    cards.to_parquet(out / "cards.parquet", index=False)
    mem.to_parquet(out / "memberships.parquet", index=False)
    atomic_write_json(out / "benchmark_fresh.json", spec)
    log_event("data", "FRESH_BUILT", benchmark_id=spec["benchmark_id"], **rep)
    return {"built": True, "benchmark_id": spec["benchmark_id"], **rep}

