"""Evaluate fixed systems (lexical, pretrained encoders, trained checkpoints) on the benchmark
and store immutable evaluation records under runs/eval/<name>/."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import encoder as enc
from .data import load_dataset
from .evaluate import Benchmark, cohort_summary, evaluate
from .lexical import LEXICAL_RECIPE, fit_vectorizer
from .util import RUNS, atomic_write_json, git_identity, log_event, read_json, sha256_file, utcnow


def load_benchmark(ds_dir: str | Path) -> tuple[Benchmark, dict]:
    ds = load_dataset(ds_dir)
    spec = read_json(Path(ds_dir) / "benchmark.json")
    return Benchmark(spec, ds["cards"], ds["memberships"], ds["tags"]), ds


def save_eval(name: str, bm: Benchmark, res: dict, meta: dict, vectors=None,
              truncated: set[str] | None = None) -> dict:
    out = RUNS / "eval" / name
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


def eval_lexical(ds_dir, name="lexical-char35-tfidf") -> dict:
    bm, ds = load_benchmark(ds_dir)
    c = ds["cards"]
    fit = c[(c["split"] == "fit") & (c["text_status"] == "ok")]
    vec = fit_vectorizer(fit["text"].tolist())
    x = vec.transform(bm.texts)
    res = evaluate(bm, x)
    return save_eval(name, bm, res, {"system": "lexical", "recipe": LEXICAL_RECIPE,
                                     "fitted_on": "fit text only"})


def eval_encoder(ds_dir, name: str, backbone: str, max_length: int,
                 model_path: str | None = None, batch_size: int = 128) -> dict:
    bm, _ = load_benchmark(ds_dir)
    recipe = enc.RECIPES[backbone]
    tok, model = enc.load(recipe, model_path)
    dev = enc.device()
    model.to(dev)
    lengths = enc.token_lengths(tok, bm.texts, recipe)
    truncated = {bm.ids[i] for i in np.flatnonzero(lengths > max_length)}
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
