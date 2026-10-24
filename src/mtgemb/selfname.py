"""Self-name normalisation, `no_reminder_v2` (idea 7, credited to Delvefall).

Shared training/exam text swap and an inference-only diagnosis over the selection-exposed
dev benchmark. Its views and labels stay fixed; only encoder inputs change. A training
run on v2 would be ext-3 and needs registration.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from .abilities import split_abilities
from .data import self_names
from .evaluate import cohort_summary, evaluate, load_benchmark
from .text import VERSION, VERSION_V2, no_reminder_v2
from .util import RUNS, atomic_write_json, git_identity, log_event, utcnow


def v2_texts(cards: pd.DataFrame) -> pd.Series:
    """oracle_id -> v2 text (None where v2 is not "ok")."""
    out = {}
    for r in cards.itertuples():
        faces = json.loads(r.faces)
        t = no_reminder_v2(faces, self_names(r.name, faces))
        out[r.oracle_id] = t["text"] if t["status"] == "ok" else None
    return pd.Series(out)


def apply_text_version(cards: pd.DataFrame, version: str) -> pd.DataFrame:
    """Copy cards, changing only usable encoder texts; ability indices stay fixed."""
    if version not in (VERSION, VERSION_V2):
        raise ValueError(f"unknown text version: {version}")
    out = cards.copy()
    if version == VERSION:
        return out
    ok = cards["text_status"] == "ok"
    texts = cards.loc[ok, "oracle_id"].map(v2_texts(cards.loc[ok]))
    for oid, before, after in zip(cards.loc[ok, "oracle_id"], cards.loc[ok, "text"],
                                  texts, strict=True):
        if after is None or pd.isna(after):
            raise ValueError(f"{oid}: v2 text is not ok")
        if len(split_abilities(before)) != len(split_abilities(after)):
            raise ValueError(f"{oid}: v2 changed ability-line count")
    out.loc[ok, "text"] = texts
    return out


def diagnose(ds_dir, name: str, backbone: str, max_length: int,
             model_path: str | None = None, cpu: bool = False) -> dict:
    import torch

    from . import encoder as enc

    bm, ds = load_benchmark(ds_dir)
    cards = apply_text_version(ds["cards"], VERSION_V2)
    t2 = cards.set_index("oracle_id").loc[bm.ids, "text"].tolist()
    changed = [a != b for a, b in zip(bm.texts, t2, strict=True)]
    recipe = enc.RECIPES[backbone]
    log_event("eval", "STARTED", task="selfname-diagnosis", name=name, model_path=model_path)
    tok, model = enc.load(recipe, model_path)
    dev = torch.device("cpu") if cpu else enc.device()  # XPU segfaults happen
    model.to(dev)
    res = {}
    for ver, texts in (("v1", bm.texts), ("v2", t2)):
        v = enc.encode(model, tok, texts, recipe, max_length, batch_size=128, dev=dev)
        r = evaluate(bm, v)
        res[ver] = {"summary": {k: r["summary"][k]["ndcg@10"] for k in ("full", "hard_lexical")},
                    "cohorts": cohort_summary(bm, r["per_anchor"], {
                        "anchor_text_changed": [changed[i] for i in bm.anchor_idx],
                        "anchor_text_same": [not changed[i] for i in bm.anchor_idx]})}
    out = {"name": name, "backbone": backbone, "model_path": model_path,
           "max_length": max_length, "device": str(dev), "benchmark_id": bm.spec["benchmark_id"],
           "universe_texts_changed": int(sum(changed)), "universe": len(bm.ids),
           "results": res, "code": git_identity(), "created_utc": utcnow(),
           "note": "selection-exposed dev diagnosis; inference only; not a promotion test"}
    rec = RUNS / "diag" / "selfname" / f"{name}-{time.time_ns()}.json"
    rec.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rec, out)
    log_event("eval", "SELFNAME_DIAGNOSED", name=name, record=str(rec),
              v1=res["v1"]["summary"], v2=res["v2"]["summary"])
    return out
