"""Run tables, pilot advancement and registered promotion contrasts (MASTER_PLAN §5).

Reads only immutable run records (runs/<id>/) and evaluation records
(runs/eval/<benchmark_id>/<name>/). Nothing here selects a checkpoint.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .evaluate import PILOT_THRESHOLDS, PROMOTION, n_boot_for, paired_bootstrap
from .util import RUNS, read_json

BOOT_SEED = 20261017  # 


def run_row(run_id: str) -> dict:
    d = RUNS / run_id
    man = read_json(d / "manifest.json")
    cfg = man["config"]
    steps = [json.loads(x) for x in (d / "metrics.jsonl").read_text("utf-8").splitlines()
             if x and '"eval"' not in x]
    res = read_json(d / "result.json") if (d / "result.json").exists() else {}
    sel = res.get("selected") or {}
    return {"run_id": run_id, "slot": man["slot"], "backbone": cfg["backbone"],
            "lr": cfg["lr"], "scale": cfg["scale"], "objective": cfg["objective"],
            "weighting": cfg["pair_weighting"], "band": tuple(cfg["neg_band"]),
            "updates": len(steps), "finished": bool(res.get("finished")),
            "sel_update": sel.get("update"), "full": sel.get("full"),
            "hard_lexical": sel.get("hard_lexical"),
            "guardrail_passed": res.get("guardrail_passed"),
            "train_hours": round(sum(s["sec"] for s in steps) / 3600, 2),
            "peak_gb": max((s["peak_mem_gb"] or 0) for s in steps) if steps else None,
            "skipped_rows": sum(s["skipped"] for s in steps)}


def run_table(prefix: str = "initial-") -> pd.DataFrame:
    ids = sorted(p.name for p in RUNS.glob(f"{prefix}*") if (p / "manifest.json").exists())
    return pd.DataFrame([run_row(i) for i in ids])


def advances(cand: dict, matched: dict) -> bool:
    """Pilot advancement over the matched recipe (practical thresholds, not significance)."""
    return bool(cand["guardrail_passed"]
                and cand["full"] - matched["full"] >= PILOT_THRESHOLDS["full_ndcg_gain"]
                and cand["hard_lexical"] - matched["hard_lexical"]
                >= PILOT_THRESHOLDS["hard_lexical_min_change"])


def _ndcg(bench_id: str, name: str, view: str) -> pd.Series:
    per = pd.read_parquet(RUNS / "eval" / bench_id / name / "per_anchor.parquet")
    per = per[(per["view"] == view) & (per["n_rel"] > 0)]
    return per.set_index("anchor")["ndcg"]


def contrast(cand: dict[int, str], comp: dict[int, str] | str, view: str, bench_id: str,
             m: int, families: pd.Series) -> dict:
    """Paired candidate-minus-comparator contrast.

    cand: {seed: eval name}. comp: {seed: eval name} for a trained comparator (seed sets
    must match; pairs are aligned by seed id, never by list order), or a single eval name
    for an untrained comparator, which is broadcast to every candidate seed.
    """
    seeds = sorted(cand)
    if isinstance(comp, dict):
        if sorted(comp) != seeds:
            raise ValueError(f"seed sets differ: {seeds} vs {sorted(comp)}")
        comp_names = [comp[s] for s in seeds]
    else:
        comp_names = [comp]
    a = [_ndcg(bench_id, cand[s], view) for s in seeds]
    b = [_ndcg(bench_id, n, view) for n in comp_names]
    anchors = a[0].index
    for x in a + b:
        if not x.index.equals(anchors):
            raise ValueError("systems were scored on different anchor sets")
    out = paired_bootstrap(np.stack([x.to_numpy() for x in a]),
                           np.stack([x.to_numpy() for x in b]),
                           families.loc[anchors].to_numpy(), n_boot=n_boot_for(m),
                           seed=BOOT_SEED, lower_q=PROMOTION["alpha"] / m)
    out.update(view=view, seeds=seeds, candidate=[cand[s] for s in seeds],
               comparator=comp_names, cand_seed_means=[float(x.mean()) for x in a],
               comp_seed_means=[float(x.mean()) for x in b])
    return out


def promotion(full: dict, hard_lexical: dict) -> bool:
    """One comparator's promotion test; a candidate must pass against both baselines."""
    return bool(full["point"] >= PROMOTION["full_point"]
                and full["lower_adjusted"] > PROMOTION["full_lower"]
                and hard_lexical["point"] >= PROMOTION["hard_lexical_point"]
                and hard_lexical["lower_adjusted"] > PROMOTION["hard_lexical_lower"])


def best_recipe(rows: list[dict]) -> dict | None:
    """Best E5 recipe among pilot rows.

    Walk recipes in preference order (lower lr, then lower scale). The first one that
    finished and passed its guardrail is the incumbent; a later recipe replaces it only if
    it advances over the incumbent by the pilot thresholds. None if no recipe qualifies.
    """
    best = None
    for r in sorted(rows, key=lambda r: (r["lr"], r["scale"])):
        if not (r["finished"] and r["guardrail_passed"]):
            continue
        if best is None or advances(r, best):
            best = r
    return best
