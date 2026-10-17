"""Fit-only positive pairs and frozen lexical-negative pools (MASTER_PLAN §3, §6).

Run after the split is frozen. Everything here reads fit cards only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset, tag_sets
from .lexical import LEXICAL_RECIPE, fit_vectorizer, topk_sparse
from .util import atomic_write_json, log_event, sha256_file, sha256_json, utcnow

PAIRS_VERSION = "pairs_v1"
PAIR_CFG = {
    "seed": "mtgemb-pairs-v1",
    "min_df": 2,  # a tag needs two fit members to form a pair
    "max_df": 500,  # broader tags still count in the metric but generate no pairs
    "rounds": 3,  # random perfect matchings per tag -> about 3 pairs per member per tag
    "card_cap": 24,  # max pairs any card appears in
    "family_cap": 48,  # max pairs touching any family component
    "neg_pool": 200,  # stored lexical-negative ranks per anchor
    "spec_clip": [0.5, 2.0],
}


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(int(sha256_json([PAIR_CFG["seed"], *key])[:15], 16))


def candidate_pairs(members: list[str], rounds: int, key: str) -> list[tuple[str, str, int]]:
    """All pairs if the tag is tiny, else `rounds` seeded random matchings."""
    n = len(members)
    if n - 1 <= rounds:
        return [(a, b, 0) for i, a in enumerate(members) for b in members[i + 1 :]]
    out, seen = [], set()
    rng = _rng("tag", key)
    for r in range(rounds):
        perm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            a, b = sorted((members[perm[i]], members[perm[i + 1]]))
            if (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b, r))
    return out


def build_pairs(cards: pd.DataFrame, mem: pd.DataFrame, tags: pd.DataFrame, cfg=PAIR_CFG):
    """Deterministic, capped positive pairs over fit cards with usable text.

    Tags are processed rarest first so broad tags cannot crowd out rare effects
    before the per-card and per-family caps are reached.
    """
    fit = cards[(cards["split"] == "fit") & (cards["text_status"] == "ok")]
    fit_ids = set(fit["oracle_id"])
    fam = dict(zip(fit["oracle_id"], fit["family_id"], strict=True))
    fm = mem[mem["oracle_id"].isin(fit_ids)]
    df = fm.groupby("tag_id").size()
    elig = set(df[(df >= cfg["min_df"]) & (df <= cfg["max_df"])].index)
    all_sets = tag_sets(fm)
    elig_sets = {k: v & elig for k, v in all_sets.items()}
    idf = dict(zip(tags["tag_id"], tags["idf_fit"], strict=True))
    label = dict(zip(tags["tag_id"], tags["label"], strict=True))
    strong = set(map(tuple, fm.loc[fm["weight"] == "very_strong", ["oracle_id", "tag_id"]].values))

    members = {t: sorted(g) for t, g in fm[fm["tag_id"].isin(elig)].groupby("tag_id")["oracle_id"]}
    order = sorted(members, key=lambda t: (len(members[t]), t))
    card_n, fam_n = Counter(), Counter()
    rows, seen = [], set()
    stats = Counter()
    for t in order:
        cands = candidate_pairs(members[t], cfg["rounds"], t)
        stats["candidates"] += len(cands)
        for a, b, r in cands:
            if (a, b) in seen:
                stats["dup_across_tags"] += 1
                continue
            if fam[a] == fam[b]:
                stats["same_family"] += 1
                continue
            if card_n[a] >= cfg["card_cap"] or card_n[b] >= cfg["card_cap"]:
                stats["card_cap"] += 1
                continue
            fa, fb = fam[a], fam[b]
            if fam_n[fa] >= cfg["family_cap"] or fam_n[fb] >= cfg["family_cap"]:
                stats["family_cap"] += 1
                continue
            seen.add((a, b))
            card_n[a] += 1
            card_n[b] += 1
            fam_n[fa] += 1
            fam_n[fb] += 1
            shared_all = sorted(all_sets[a] & all_sets[b])
            rows.append({
                "a": a, "b": b, "split": "fit", "tag_id": t, "tag_label": label[t],
                "tag_df_fit": len(members[t]),
                "sampling_reason": f"tag_matching_round_{r}" if len(members[t]) - 1 > cfg[
                    "rounds"] else "tag_all_pairs",
                "relation_source": "scryfall_tagger_oracle_tag",
                "confidence": "very_strong" if (a, t) in strong and (b, t) in strong else "median",
                "shared_eligible": sorted(elig_sets[a] & elig_sets[b]),
                "shared_all": shared_all,
                "mean_idf_shared": float(np.mean([idf[x] for x in shared_all])),
            })
    pairs = pd.DataFrame(rows)
    # slot-7 specificity weight: mean fit-IDF of shared retained tags, mean-one, clipped
    w = pairs["mean_idf_shared"] / pairs["mean_idf_shared"].mean()
    pairs["spec_weight"] = w.clip(*cfg["spec_clip"])
    stats["pairs"] = len(pairs)
    stats["eligible_tags"] = len(elig)
    stats["tags_with_pairs"] = int(pairs["tag_id"].nunique())
    stats["fit_cards_ok"] = len(fit_ids)
    stats["fit_cards_in_pairs"] = len(card_n)
    stats["fit_cards_with_eligible_tag"] = sum(bool(v) for v in elig_sets.values())
    stats["card_exposure_max"] = max(card_n.values())
    stats["card_exposure_mean"] = float(np.mean(list(card_n.values())))
    stats["spec_weight_clipped_frac"] = float(
        ((w < cfg["spec_clip"][0]) | (w > cfg["spec_clip"][1])).mean())
    return pairs, elig, dict(stats)


def lexical_negatives(cards: pd.DataFrame, mem: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict]:
    """For each tagged fit card, the k lexically nearest tagged fit cards that share no
    retained functional tag and are not in its family. Ranks are frozen fit-only.

    Tag non-overlap is a sampling heuristic, not a semantic guarantee; untagged cards
    are never negatives because missing tags are unknown.
    """
    fit = cards[(cards["split"] == "fit") & (cards["text_status"] == "ok")]
    vec = fit_vectorizer(fit["text"].tolist())
    sets = tag_sets(mem[mem["oracle_id"].isin(set(fit["oracle_id"]))])
    tagged = fit[fit["oracle_id"].isin(sets)].reset_index(drop=True)
    ids = tagged["oracle_id"].tolist()
    x = vec.transform(tagged["text"].tolist())
    fam = tagged["family_id"].to_numpy()
    tag_index = defaultdict(list)
    for i, oid in enumerate(ids):
        for t in sets[oid]:
            tag_index[t].append(i)

    def exclude(rows, sc):
        for j, i in enumerate(rows):
            sc[j, i] = -np.inf
            sc[j, fam == fam[i]] = -np.inf
            for t in sets[ids[i]]:
                sc[j, tag_index[t]] = -np.inf

    idx, val = topk_sparse(x, x, k, exclude=exclude)
    a = np.repeat(np.arange(len(ids)), k)
    neg = pd.DataFrame({
        "anchor": np.asarray(ids)[a],
        "negative": np.asarray(ids)[idx.ravel()],
        "rank": np.tile(np.arange(1, k + 1), len(ids)),
        "lex_cosine": val.ravel(),
        "source": "char3-5_tfidf_fit_no_shared_tag",
    })
    bad = ~np.isfinite(neg["lex_cosine"])
    stats = {"anchors": len(ids), "k": k, "non_finite_dropped": int(bad.sum()),
             "lex_cosine_rank1_median": float(np.median(val[:, 0])),
             "lex_cosine_rank20_median": float(np.median(val[:, min(19, k - 1)])),
             "lex_cosine_rank100_median": float(np.median(val[:, min(99, k - 1)]))}
    return neg[~bad].reset_index(drop=True), stats


def build_training_data(ds_dir: Path) -> dict:
    ds = load_dataset(ds_dir)
    out = Path(ds_dir) / PAIRS_VERSION
    if (out / "pairs.json").exists():
        raise FileExistsError(f"{out} already finalised")
    out.mkdir(exist_ok=True)
    log_event("data", "STARTED", step="pairs", dataset_id=ds["ident"]["dataset_id"])
    cards, mem, tags = ds["cards"], ds["memberships"], ds["tags"]
    pairs, elig, pstats = build_pairs(cards, mem, tags)
    neg, nstats = lexical_negatives(cards, mem, PAIR_CFG["neg_pool"])
    # leakage checks: every id in pairs/negatives must be fit
    split = dict(zip(cards["oracle_id"], cards["split"], strict=True))
    for col, frame in (("a", pairs), ("b", pairs), ("anchor", neg), ("negative", neg)):
        bad = frame[col].map(split) != "fit"
        if bad.any():
            raise AssertionError(f"non-fit id in {col}: {frame.loc[bad, col].head().tolist()}")
    pairs.to_parquet(out / "pairs.parquet", index=False)
    neg.to_parquet(out / "negatives.parquet", index=False)
    pd.DataFrame({"tag_id": sorted(elig)}).to_parquet(out / "pair_tags.parquet", index=False)
    rec = {
        "version": PAIRS_VERSION, "dataset_id": ds["ident"]["dataset_id"], "config": PAIR_CFG,
        "lexical_recipe": LEXICAL_RECIPE, "pair_stats": pstats, "negative_stats": nstats,
        "files": {p.name: sha256_file(p) for p in sorted(out.glob("*.parquet"))},
        "created_utc": utcnow(),
    }
    rec["pairs_id"] = sha256_json(rec["files"])[:16]
    atomic_write_json(out / "pairs.json", rec)
    log_event("data", "COMPLETED", step="pairs", pairs_id=rec["pairs_id"], **pstats)
    return rec

