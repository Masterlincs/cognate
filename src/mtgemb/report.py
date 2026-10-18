"""Run tables, pilot advancement and registered promotion contrasts (MASTER_PLAN §5).

Reads only immutable run records (runs/<id>/) and evaluation records
(runs/eval/<benchmark_id>/<name>/). Nothing here selects a checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import PILOT_THRESHOLDS, PROMOTION, n_boot_for, paired_bootstrap
from .util import RUNS, atomic_write_json, read_json, utcnow

BOOT_SEED = 20261017  # 
# frozen confirmation design: one E5 candidate, trained-BGE control, seeds 1-3
CONFIRM = {"candidate": "confirm-e5-s{}", "bge": "confirm-bge-s{}",
           "pretrained": "pretrained-e5-base-v2", "seeds": (1, 2, 3), "deploy_seed": 1}
VIEWS = ("full", "hard_lexical")
QUALIFIER = "provisional tag-proxy promotion on selection-exposed development"


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


def confirmation_report(bench_id: str) -> dict:
    """Registered P7 contrasts and the promotion decision (MASTER_PLAN §5).

    Every candidate and BGE eval must come from the matching confirmation run's final
    weights. Written once to runs/confirmation/<bench_id>/report.json; never overwritten.
    """
    out = RUNS / "confirmation" / bench_id / "report.json"
    if out.exists():
        raise FileExistsError(out)
    seeds = CONFIRM["seeds"]
    cand = {s: CONFIRM["candidate"].format(s) for s in seeds}
    bge = {s: CONFIRM["bge"].format(s) for s in seeds}
    for name in [*cand.values(), *bge.values()]:
        res = read_json(RUNS / name / "result.json")
        meta = read_json(RUNS / "eval" / bench_id / name / "summary.json")["meta"]
        if not (res["finished"] and res["selection"] == "final"):
            raise ValueError(f"{name}: not a finished selection='final' run")
        if Path(meta["model_path"]).resolve() != (RUNS / name / "final").resolve():
            raise ValueError(f"{name}: eval was not made from runs/{name}/final")
    per = pd.read_parquet(RUNS / "eval" / bench_id / cand[seeds[0]] / "per_anchor.parquet")
    families = per[per["view"] == "full"].set_index("anchor")["family"]
    comps = {"pretrained_e5": CONFIRM["pretrained"], "trained_bge": bge}
    m = len(comps) * len(VIEWS)
    contrasts = {}
    for key, comp in comps.items():
        c = {v: contrast(cand, comp, v, bench_id, m, families) for v in VIEWS}
        contrasts[key] = {**c, "passes": promotion(c["full"], c["hard_lexical"])}
    promoted = all(c["passes"] for c in contrasts.values())
    dep = cand[CONFIRM["deploy_seed"]]
    rep = {"benchmark_id": bench_id, "m_contrasts": m, "contrasts": contrasts,
           "decision": "promote" if promoted else "not promoted",
           "qualifier": QUALIFIER, "deployment": {
               "seed": CONFIRM["deploy_seed"], "run_id": dep,
               **{v: float(_ndcg(bench_id, dep, v).mean()) for v in VIEWS}},
           "design": {**CONFIRM, "decisions": ["", ""]}, "created_utc": utcnow()}
    atomic_write_json(out, rep)
    return rep
