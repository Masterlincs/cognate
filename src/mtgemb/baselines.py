"""Evaluate fixed systems (lexical, pretrained encoders, trained checkpoints) on the benchmark
and store immutable evaluation records under runs/eval/<benchmark_id>/<name>/."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import encoder as enc
from .evaluate import Benchmark, cohort_summary, evaluate, load_benchmark
from .lexical import LEXICAL_RECIPE, fit_vectorizer
from .train import DeviceLock
from .util import (
    ARTIFACTS,
    RUNS,
    atomic_write_json,
    git_identity,
    log_event,
    read_json,
    sha256_file,
    utcnow,
)


def save_eval(name: str, bm: Benchmark, res: dict, meta: dict, vectors=None,
              truncated: set[str] | None = None) -> dict:
    out = RUNS / "eval" / bm.spec["benchmark_id"] / name
    if (out / "summary.json").exists():
        raise FileExistsError(f"evaluation {name} already recorded; use a new name")
    out.mkdir(parents=True, exist_ok=True)
    res["per_anchor"].to_parquet(out / "per_anchor.parquet", index=False)
    files = {"per_anchor.parquet": sha256_file(out / "per_anchor.parquet")}
    if vectors is not None and not hasattr(vectors, "tocsr"):
        np.save(out / "vectors.npy", vectors)
        files["vectors.npy"] = sha256_file(out / "vectors.npy")
    rec = {"name": name, "benchmark_id": bm.spec["benchmark_id"], "summary": res["summary"],
           "cohorts": cohort_summary(bm, res["per_anchor"], bm.cohorts(truncated)),
           "meta": meta, "files": files, "code": git_identity(), "created_utc": utcnow(),
           "claim_scope": "tag-proxy relevance on selection-exposed development anchors"}
    atomic_write_json(out / "summary.json", rec)
    log_event("eval", "COMPLETED", name=name, full=res["summary"]["full"]["ndcg@10"],
              hard_lexical=res["summary"]["hard_lexical"]["ndcg@10"])
    return rec


def _lexical_vectors(bm: Benchmark, ds: dict):
    c = ds["cards"]
    fit = c[(c["split"] == "fit") & (c["text_status"] == "ok")]
    return fit_vectorizer(fit["text"].tolist()).transform(bm.texts)


def eval_lexical(ds_dir, name="lexical-char35-tfidf") -> dict:
    bm, ds = load_benchmark(ds_dir)
    res = evaluate(bm, _lexical_vectors(bm, ds))
    return save_eval(name, bm, res, {"system": "lexical", "recipe": LEXICAL_RECIPE,
                                     "fitted_on": "fit text only"})


def eval_encoder(ds_dir, name: str, backbone: str, max_length: int,
                 model_path: str | None = None, batch_size: int = 128) -> dict:
    bm, _ = load_benchmark(ds_dir)
    recipe = enc.RECIPES[backbone]
    tok, model = enc.load(recipe, model_path)
    dev = enc.device()
    lengths = enc.token_lengths(tok, bm.texts, recipe)
    truncated = {bm.ids[i] for i in np.flatnonzero(lengths > max_length)}
    with DeviceLock():
        model.to(dev)
        t0 = time.time()
        v = enc.encode(model, tok, bm.texts, recipe, max_length, batch_size=batch_size, dev=dev)
        secs = time.time() - t0
    res = evaluate(bm, v)
    meta = {"system": "encoder", "backbone": backbone, "recipe": recipe,
            "max_length": max_length, "model_path": model_path, "device": str(dev),
            "torch": torch.__version__, "encode_seconds": round(secs, 1),
            "truncated_universe": len(truncated),
            "truncated_dev_anchors": int(sum(bm.ids[i] in truncated for i in bm.anchor_idx))}
    return save_eval(name, bm, res, meta, vectors=v, truncated=truncated)


def replay_evals(ds_dir, tol: float = 1e-9) -> dict:
    """P10 evaluation replay from pinned embeddings; no encoding, no training.

    For every recorded evaluation on the current benchmark: verify the recorded file hashes,
    re-score the saved vectors (lexical: refit on fit text), and compare per-anchor metrics
    with the record. For every exported release with a run_id, check its vectors equal that
    run's evaluated vectors row for row. Writes runs/replay/<benchmark_id>/<utc>.json.
    """
    bm, ds = load_benchmark(ds_dir)
    bid = bm.spec["benchmark_id"]
    evals, vecs = {}, {}
    for d in sorted(p for p in (RUNS / "eval" / bid).iterdir() if (p / "summary.json").exists()):
        rec = read_json(d / "summary.json")
        bad_hash = sorted(f for f, h in rec["files"].items() if sha256_file(d / f) != h)
        if "vectors.npy" in rec["files"]:
            vecs[d.name] = np.load(d / "vectors.npy")
            res = evaluate(bm, vecs[d.name])
        elif rec["meta"].get("system") == "lexical":
            res = evaluate(bm, _lexical_vectors(bm, ds))
        else:
            evals[d.name] = {"hash_mismatch": bad_hash, "replayed": False}
            continue
        old = pd.read_parquet(d / "per_anchor.parquet").set_index(["anchor", "view"]).sort_index()
        new = res["per_anchor"].set_index(["anchor", "view"]).sort_index()
        cols = [c for c in old.columns if pd.api.types.is_float_dtype(old[c])]
        ints = ["n_rel", "n_candidates"]
        # rows, NaN pattern (zero-positive anchors) and integer counts must all match exactly
        same_rows = bool(old.index.equals(new.index)
                         and (old[cols].isna().to_numpy() == new[cols].isna().to_numpy()).all()
                         and (old[ints].to_numpy() == new[ints].to_numpy()).all())
        diff = (float(np.nanmax(np.abs(old[cols].to_numpy() - new[cols].to_numpy())))
                if same_rows else None)
        evals[d.name] = {"hash_mismatch": bad_hash, "replayed": True, "same_rows": same_rows,
                         "max_abs_diff": diff,
                         "recorded": {v: rec["summary"][v]["ndcg@10"] for v in ("full",
                                                                               "hard_lexical")},
                         "replayed_ndcg@10": {v: res["summary"][v]["ndcg@10"] for v in
                                              ("full", "hard_lexical")}}
    releases = {}
    for meta_p in sorted(Path(ARTIFACTS / "releases").glob("*/release_meta.json")):
        meta = read_json(meta_p)
        rows = ds["cards"].sort_values("oracle_id")
        rows = rows.loc[rows["text_status"] == "ok", "oracle_id"].tolist()
        v = np.load(meta_p.parent / "vectors.npy")
        r = {"run_id": meta["run_id"], "vectors_sha256_ok":
             sha256_file(meta_p.parent / "vectors.npy") == meta["vectors_sha256"],
             "rows_are_benchmark_universe": rows == list(bm.ids)}
        if meta["run_id"] in vecs and r["rows_are_benchmark_universe"]:
            r["max_abs_diff_vs_eval"] = float(np.abs(v - vecs[meta["run_id"]]).max())
        releases[meta["release_id"]] = r
    ok = all(not e["hash_mismatch"] and e.get("replayed") and e["same_rows"]
             and e["max_abs_diff"] <= tol for e in evals.values()) and all(
        r["vectors_sha256_ok"] and r.get("max_abs_diff_vs_eval", 0.0) <= tol
        for r in releases.values())
    out = {"benchmark_id": bid, "dataset_id": ds["ident"]["dataset_id"], "tolerance": tol,
           "passed": ok, "evals": evals, "releases": releases, "code": git_identity(),
           "created_utc": utcnow(),
           "scope": "evaluation replay from pinned embeddings; not a retraining replay"}
    # unique name: a replay report is never overwritten by a later replay
    path = RUNS / "replay" / bid / f"{out['created_utc'].replace(':', '')}-{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, out)
    log_event("eval", "REPLAYED", benchmark_id=bid, passed=ok, report=str(path))
    return out
