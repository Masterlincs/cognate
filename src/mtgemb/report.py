"""Run tables, pilot advancement and registered promotion contrasts (MASTER_PLAN §5).

Reads only immutable run records (runs/<id>/) and evaluation records
(runs/eval/<benchmark_id>/<name>/). Nothing here selects a checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import PILOT_THRESHOLDS, PROMOTION, n_boot_for, paired_bootstrap
from .util import RUNS, read_json, utcnow

BOOT_SEED = 20261017  # 
# frozen confirmation design: one E5 candidate, trained-BGE control, seeds 1-3
CONFIRM = {"candidate": "confirm-e5-s{}", "bge": "confirm-bge-s{}",
           "pretrained": "pretrained-e5-base-v2", "seeds": (1, 2, 3), "deploy_seed": 1,
           "recipe_from": {"candidate": "initial-1-e5-lr3e-05-s20",
                           "bge": "initial-4-bge-lr3e-05-s20"},
           "pretrained_backbone": "e5-base-v2", "pretrained_max_length": 94}
# the 4000-update recipe (x-1) under the same design, plus the deployed 2000-update E5
# recipe as a third comparator (a swap must beat it too); report under confirmation/<id>/x1/
CONFIRM_X1 = {**CONFIRM, "candidate": "confirm-x1-s{}",
              "recipe_from": {**CONFIRM["recipe_from"], "candidate": "x-1-e5-4000",
                              "incumbent": "initial-1-e5-lr3e-05-s20"},
              "incumbent": "confirm-e5-s{}", "tag": "x1"}
DESIGNS = {"d18": CONFIRM, "x1": CONFIRM_X1}
# config keys a confirmation run may change from its frozen pilot recipe
CONFIRM_FREE = {"run_id", "slot", "hypothesis", "comparator", "seed", "data_seed", "selection"}
MIN_CLUSTERS = 100  # fewer eligible families than this makes a contrast inconclusive
VIEWS = ("full", "hard_lexical")
QUALIFIER = "provisional tag-proxy promotion on selection-exposed development"
SEALED_QUALIFIER = ("registered gate re-tested once on sealed anchors (tag proxy); "
                    "does not change the P7 decision")
SEALED_GATE = {"promote": "cleared", "reject": "not_cleared", "inconclusive": "inconclusive"}


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


def _check_eval(bench_id: str, name: str, backbone: str, max_length: int,
                model_path: Path | None, recipe: dict) -> None:
    rec = read_json(RUNS / "eval" / bench_id / name / "summary.json")
    meta = rec["meta"]
    got = Path(meta["model_path"]).resolve() if meta["model_path"] else None
    want = model_path.resolve() if model_path else None
    if (rec["benchmark_id"], meta["backbone"], meta["max_length"], meta["recipe"], got) != (
            bench_id, backbone, max_length, recipe, want):
        raise ValueError(f"{name}: eval was not made from {want or 'the pretrained model'} "
                         f"with {backbone} {recipe} at max_length {max_length} on {bench_id}")


def _check_confirm_run(bench_id: str, name: str, seed: int, recipe_run: str,
                       eval_bench: str | None = None) -> None:
    res = read_json(RUNS / name / "result.json")
    man = read_json(RUNS / name / "manifest.json")
    ref = read_json(RUNS / recipe_run / "manifest.json")
    cfg = man["config"]
    if not (res["finished"] and res["selection"] == "final" == cfg["selection"]):
        raise ValueError(f"{name}: not a finished selection='final' run")
    if cfg["seed"] != seed or cfg["data_seed"] != seed:
        raise ValueError(f"{name}: seed/data_seed {cfg['seed']}/{cfg['data_seed']} != {seed}")
    diff = sorted(k for k in cfg.keys() | ref["config"].keys()
                  if k not in CONFIRM_FREE and cfg.get(k) != ref["config"].get(k))
    # identity: dataset, pairs, model recipe/revision, trainer code, benchmark ("config" is
    # the hash of the config, which differs by design in the CONFIRM_FREE keys)
    diff += sorted(f"identity.{k}" for k in man["identity"].keys() | ref["identity"].keys()
                   if k != "config" and man["identity"].get(k) != ref["identity"].get(k))
    if man["identity"].get("benchmark_id") != bench_id:
        diff.append("identity.benchmark_id != report benchmark")
    if diff:
        raise ValueError(f"{name}: differs from frozen recipe {recipe_run} in {diff}")
    _check_eval(eval_bench or bench_id, name, cfg["backbone"], cfg["max_length"],
                RUNS / name / "final", man["identity"]["recipe"])


def confirmation_report(bench_id: str, train_bench: str | None = None,
                        design: dict = CONFIRM) -> dict:
    """Registered P7 contrasts and the promotion decision (MASTER_PLAN §5).

    Every confirmation run must be the frozen recipe at its registered seed, and every eval
    must come from that run's final weights (pretrained: the pinned backbone). Decision:
    inconclusive if any contrast has < MIN_CLUSTERS families; else promote only if both
    comparators pass; else reject. Written once to runs/confirmation/<bench_id>/report.json.

    train_bench: the runs were trained and selected on that benchmark and bench_id is
    the sealed one. The same contrasts are then re-tested and reported as a sealed gate
    (cleared / not_cleared / inconclusive), written once to runs/sealed/<bench_id>/.
    """
    sealed = train_bench is not None
    train_bench = train_bench or bench_id
    if sealed and design is not CONFIRM:
        raise ValueError("the sealed gate was run once, for the design only")
    out = RUNS / ("sealed" if sealed else "confirmation") / bench_id
    out = (out / design["tag"] if "tag" in design else out) / "report.json"
    if out.exists():
        raise FileExistsError(out)
    seeds, rf = design["seeds"], design["recipe_from"]
    cand = {s: design["candidate"].format(s) for s in seeds}
    bge = {s: design["bge"].format(s) for s in seeds}
    inc = {s: design["incumbent"].format(s) for s in seeds} if "incumbent" in design else None
    for s in seeds:
        _check_confirm_run(train_bench, cand[s], s, rf["candidate"], bench_id)
        _check_confirm_run(train_bench, bge[s], s, rf["bge"], bench_id)
        if inc:
            _check_confirm_run(train_bench, inc[s], s, rf["incumbent"], bench_id)
    e5_recipe = read_json(RUNS / rf["candidate"] / "manifest.json")["identity"]["recipe"]
    _check_eval(bench_id, design["pretrained"], design["pretrained_backbone"],
                design["pretrained_max_length"], None, e5_recipe)
    per = pd.read_parquet(RUNS / "eval" / bench_id / cand[seeds[0]] / "per_anchor.parquet")
    families = per[per["view"] == "full"].set_index("anchor")["family"]
    excluded = {v: int(((per["view"] == v) & (per["n_rel"] == 0)).sum()) for v in VIEWS}
    # relevance (n_rel) comes from the frozen benchmark, so eligibility is the same for
    # every system; check adequacy before bootstrapping (an empty cohort cannot be resampled)
    eligible = per[per["n_rel"] > 0]
    clusters = {v: int(eligible.loc[eligible["view"] == v, "family"].nunique()) for v in VIEWS}
    comps = {"pretrained_e5": design["pretrained"], "trained_bge": bge}
    if inc:
        comps["incumbent_e5"] = inc
    m = len(comps) * len(VIEWS)
    adequate = min(clusters.values()) >= MIN_CLUSTERS
    contrasts = {}
    for key, comp in comps.items() if adequate else ():
        c = {v: contrast(cand, comp, v, bench_id, m, families) for v in VIEWS}
        for x in c.values():
            x["diff_seed_range"] = [min(x["per_seed"]), max(x["per_seed"])]
            x["cand_seed_range"] = [min(x["cand_seed_means"]), max(x["cand_seed_means"])]
        contrasts[key] = {**c, "passes": promotion(c["full"], c["hard_lexical"])}
    if not adequate:
        decision = "inconclusive"
    elif all(c["passes"] for c in contrasts.values()):
        decision = "promote"
    else:
        decision = "reject"
    dep = cand[design["deploy_seed"]]
    rep = {"benchmark_id": bench_id, "m_contrasts": m, "contrasts": contrasts,
           "decision": decision, "qualifier": QUALIFIER,
           "excluded_zero_positive": excluded, "eligible_clusters": clusters,
           "min_clusters": MIN_CLUSTERS,
           "deployment": {"seed": CONFIRM["deploy_seed"], "run_id": dep,
                          **{v: float(_ndcg(bench_id, dep, v).mean()) for v in VIEWS}},
           "design": {**design, "decisions": ["", "", ""] + ([""] if inc else [])},
           "created_utc": utcnow()}
    if sealed:
        rep["sealed_gate"] = SEALED_GATE[rep.pop("decision")]
        rep.update(qualifier=SEALED_QUALIFIER, train_benchmark_id=train_bench)
        rep["design"]["decisions"].append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"report.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, indent=2), "utf-8")
    try:
        os.link(tmp, out)  # no overwrite: fails if a concurrent call already wrote it
    finally:
        tmp.unlink()
    return rep
