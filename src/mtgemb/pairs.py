"""Fit-only positive pairs and frozen lexical-negative pools.

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

# Tags about a card's own cost, casting mode or printing rather than what it does. In
# pairs_v2 they make no positive pairs, mask no in-batch negative and do not stop a card from
# being a lexical negative. Tags about other cards' costs (cost reducer, free-cast-another)
# and "cares about" conditions (threshold, morbid) are effects and stay.
MODIFIER_LABELS = (
    # own cost: alternative, additional, reduced, paid in life, cards, counters or taps
    "alternative cost", "additional cost", "manaless value", "potentially free",
    "spell with no casting cost", "discount-self", "life payment", "upkeep cost",
    "graveyard fuel", "graveyard fuel-artifact", "mm counter cost", "tap fuel-land",
    "tap fuel-artifact", "sacrifice cost land", "offcolor additional cost", "strive",
    "phyrexian mana cost", "phyrexian mana ability", "multicolor kicker",
    "multiple kicker costs", "distinct echo cost", "alternate-equip-cost",
    "alternate-cost-bounce", "alternate-cost-sacrifice", "alternate-cost-life",
    "alternate-cost-gain-life", "emerge-from-creature", "emerge-from-artifact",
    "convoke", "delve", "improvise", "affinity for artifacts", "affinity for spells",
    "affinity undergrowth", "affinity for attacking", "affinity for land type",
    "affinity for party", "affinity for domain", "affinity for equipment",
    "affinity for creatures", "affinity for graveyard", "affinity for devotion",
    "affinity for enchantments", "x doesn't matter", "cr 107.3f x card",
    # casting mode or condition on casting it
    "madness", "ninjutsu", "exhaust", "casting restriction", "synergy-commander",
    # printing and format, not function
    "commander set booster cards", "face-commander", "useless in singleton formats",
    "stronger in singleton formats",
)
VERSIONS = {"pairs_v1": PAIR_CFG,
            "pairs_v2": {**PAIR_CFG, "exclude_labels": list(MODIFIER_LABELS)}}


def drop_modifier_tags(mem: pd.DataFrame, tags: pd.DataFrame, labels) -> tuple[pd.DataFrame, set]:
    """Memberships without the tags carrying these labels; an unknown label is an error."""
    by_label = tags.groupby("label")["tag_id"].apply(set)
    missing = sorted(set(labels) - set(by_label.index))
    if missing:
        raise ValueError(f"no tag labelled {missing}")
    drop = set().union(*(by_label[x] for x in labels))
    return mem[~mem["tag_id"].isin(drop)].reset_index(drop=True), drop


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
                "sampling_reason": (f"tag_matching_round_{r}" if len(members[t]) - 1 > cfg["rounds"]
                                    else "tag_all_pairs"),
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


def rider_twin_stats(neg: pd.DataFrame, mem: pd.DataFrame, dropped: set, band: int = 30) -> dict:
    """How many anchors have, in lexical ranks 1..band, a negative sharing a dropped tag."""
    rider = mem[mem["tag_id"].isin(dropped)].groupby("oracle_id")["tag_id"].apply(set)
    top = neg[neg["rank"] <= band]
    hit = [bool(rider.get(a, set()) & rider.get(n, set()))
           for a, n in zip(top["anchor"], top["negative"], strict=True)]
    hits = top.loc[hit, "anchor"]
    return {f"rider_twin_negatives_top{band}": len(hits),
            f"anchors_with_rider_twin_top{band}": int(hits.nunique())}


def build_training_data(ds_dir: Path, version: str = PAIRS_VERSION) -> dict:
    ds = load_dataset(ds_dir)
    cfg = VERSIONS[version]
    out = Path(ds_dir) / version
    if (out / "pairs.json").exists():
        raise FileExistsError(f"{out} already finalised")
    out.mkdir(exist_ok=True)
    log_event("data", "STARTED", step="pairs", version=version,
              dataset_id=ds["ident"]["dataset_id"])
    cards, mem, tags = ds["cards"], ds["memberships"], ds["tags"]
    dropped = set()
    if cfg.get("exclude_labels"):
        mem, dropped = drop_modifier_tags(mem, tags, cfg["exclude_labels"])
    pairs, elig, pstats = build_pairs(cards, mem, tags, cfg)
    neg, nstats = lexical_negatives(cards, mem, cfg["neg_pool"])
    if dropped:
        fit = set(cards.loc[cards["split"] == "fit", "oracle_id"])
        full = ds["memberships"][ds["memberships"]["oracle_id"].isin(fit)]
        nstats.update(rider_twin_stats(neg, full, dropped), excluded_tags=len(dropped))
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
        "version": version, "dataset_id": ds["ident"]["dataset_id"], "config": cfg,
        "excluded_tag_ids": sorted(dropped),
        "lexical_recipe": LEXICAL_RECIPE, "pair_stats": pstats, "negative_stats": nstats,
        "files": {p.name: sha256_file(p) for p in sorted(out.glob("*.parquet"))},
        "created_utc": utcnow(),
    }
    rec["pairs_id"] = sha256_json(rec["files"])[:16]
    atomic_write_json(out / "pairs.json", rec)
    log_event("data", "COMPLETED", step="pairs", version=version, pairs_id=rec["pairs_id"],
              **pstats)
    return rec


def audit_pairs(ds_dir: Path, per_band: int = 40) -> dict:
    """Structural audit of sampled pairs and negatives, plus a readable sample file.

    Automated checks establish consistency only; semantic quality stays unverified unless
    a person judges the written sample.
    """
    ds = load_dataset(ds_dir)
    pdir = Path(ds_dir) / PAIRS_VERSION
    pairs = pd.read_parquet(pdir / "pairs.parquet")
    neg = pd.read_parquet(pdir / "negatives.parquet")
    cards = ds["cards"].set_index("oracle_id")
    sets = tag_sets(ds["memberships"])
    rng = np.random.default_rng(20261017)
    bands = {
        "broad_tag_df_200_500": pairs[pairs["tag_df_fit"] >= 200],
        "mid_tag_df_11_199": pairs[(pairs["tag_df_fit"] > 10) & (pairs["tag_df_fit"] < 200)],
        "rare_tag_df_le_10": pairs[pairs["tag_df_fit"] <= 10],
    }
    checks = Counter()
    lines = ["# Pair audit sample", "", "Structural checks are automated. Semantic quality is "
             "unverified unless marked as judged.", ""]
    for name, frame in bands.items():
        sample = frame.iloc[rng.choice(len(frame), min(per_band, len(frame)), replace=False)]
        lines += [f"## {name} ({len(sample)} of {len(frame)})", ""]
        for r in sample.itertuples():
            a, b = cards.loc[r.a], cards.loc[r.b]
            checks["pairs_checked"] += 1
            checks["both_fit"] += a["split"] == "fit" and b["split"] == "fit"
            checks["tag_on_both"] += r.tag_id in sets[r.a] and r.tag_id in sets[r.b]
            checks["different_family"] += a["family_id"] != b["family_id"]
            checks["shared_all_consistent"] += set(r.shared_all) == set(sets[r.a] & sets[r.b])
            lines += [f"- tag `{r.tag_label}` (df {r.tag_df_fit}); shared {len(r.shared_all)}",
                      f"  - {a['name']}: {a['text'][:160]!r}",
                      f"  - {b['name']}: {b['text'][:160]!r}"]
        lines.append("")
    for lo, hi in ((1, 30), (31, 100), (101, 200)):
        band = neg[(neg["rank"] >= lo) & (neg["rank"] <= hi)]
        sample = band.iloc[rng.choice(len(band), 15, replace=False)]
        lines += [f"## lexical negatives ranks {lo}-{hi}", ""]
        for r in sample.itertuples():
            a, b = cards.loc[r.anchor], cards.loc[r.negative]
            checks["negatives_checked"] += 1
            checks["neg_no_shared_tag"] += not (sets[r.anchor] & sets[r.negative])
            checks["neg_both_fit"] += a["split"] == "fit" and b["split"] == "fit"
            checks["neg_different_family"] += a["family_id"] != b["family_id"]
            lines += [f"- rank {r.rank}, lex {r.lex_cosine:.2f}",
                      f"  - {a['name']}: {a['text'][:160]!r}",
                      f"  - {b['name']}: {b['text'][:160]!r}"]
        lines.append("")
    (pdir / "audit_sample.md").write_text("\n".join(lines), encoding="utf-8")
    rec = {"checks": dict(checks), "semantic_quality": "unverified", "created_utc": utcnow()}
    atomic_write_json(pdir / "audit.json", rec)
    return rec
