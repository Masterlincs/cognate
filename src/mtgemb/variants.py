"""scoring variants on the frozen dev benchmark (diagnostics, not evaluation records).

Face match: a multi-face card is one text and one vector, so a card whose second face is
Lightning Bolt is diluted by its first face. Here each card also gets one vector per face
(cards with two or more non-empty faces) and q scores c by the largest cosine over all
their vectors. Pairs of single-face cards keep the whole-card cosine.

Effect-only relevance: the same universe, anchors and views, with the modifier tags
(own cost, casting mode, printing) removed from the tag-proxy gains, so a model is not
rewarded for matching a cost rider.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .text import FACE_BOUNDARY
from .util import RUNS, atomic_write_json, log_event, read_json, sha256_file, utcnow


def face_texts(text: str) -> list[str]:
    """The non-empty face texts of a card, or [] when it has fewer than two."""
    parts = [p for p in text.split(FACE_BOUNDARY) if p.strip()]
    return parts if len(parts) > 1 else []


def face_index(texts: list[str]) -> tuple[list[str], np.ndarray]:
    """All face texts in card order, and the row of the card each belongs to."""
    out, owner = [], []
    for i, t in enumerate(texts):
        f = face_texts(t)
        out += f
        owner += [i] * len(f)
    return out, np.array(owner, dtype=np.int64)


def face_max(W: np.ndarray, F: np.ndarray, owner: np.ndarray, rows: np.ndarray,
             query_faces: bool = True) -> np.ndarray:
    """(len(rows), n) best cosine over {whole} U faces of the query and of each candidate.

    W: (n, d) unit whole-card rows; F: (m, d) unit face rows; owner: (m,) card row of each
    face, non-decreasing (as face_index returns it). query_faces=False keeps the query's
    whole-card vector only (the site's Face match: "which cards have a face like this card").
    """
    s = W[rows] @ W.T
    if not len(owner):
        return s
    cols, starts = np.unique(owner, return_index=True)

    def best_faces(q):  # (r, n_multi): best face cosine per multi-face candidate
        return np.maximum.reduceat(q @ F.T, starts, axis=1)

    s[:, cols] = np.maximum(s[:, cols], best_faces(W[rows]))
    if not query_faces:
        return s
    at = dict(zip(cols.tolist(), starts.tolist(), strict=True))
    ends = dict(zip(cols.tolist(), np.r_[starts[1:], len(owner)].tolist(), strict=True))
    for j, r in enumerate(rows.tolist()):
        if r in at:
            q = F[at[r]:ends[r]]
            s[j] = np.maximum(s[j], (q @ W.T).max(0))
            s[j, cols] = np.maximum(s[j, cols], best_faces(q).max(0))
    return s


def effect_only(bm, tags: pd.DataFrame, labels) -> object:
    """A copy of the benchmark whose gains ignore tags carrying these labels."""
    w = tags.loc[tags["df_fit"] > 0, ["tag_id", "label"]]  # Benchmark's column order
    drop = set(tags.loc[tags["label"].isin(set(labels)), "tag_id"])
    keep = sp.diags((~w["tag_id"].isin(drop)).to_numpy(dtype=float))
    out = copy.copy(bm)
    out.A, out.Aw = sp.csr_matrix(bm.A @ keep), sp.csr_matrix(bm.Aw @ keep)
    out.card_tags = {k: [t for t in v if t not in drop] for k, v in bm.card_tags.items()}
    return out


def diagnose(ds_dir, name: str, batch_size: int = 32) -> dict:
    """Score a recorded dev evaluation's vectors as whole card, face match on both sides and
    face match on the candidate side, each under standard and effect-only relevance. Face
    vectors are encoded on the CPU with the recorded encoder. v1 (2026-10-22, both-sided
    only) stays in runs/diag/d54/<name>; this version writes runs/diag/d54/v2/<name>."""
    import torch

    from . import encoder as enc
    from .evaluate import cohort_summary, evaluate, load_benchmark
    from .pairs import MODIFIER_LABELS

    bm, ds = load_benchmark(ds_dir)
    bid = bm.spec["benchmark_id"]
    d = RUNS / "eval" / bid / name
    rec = read_json(d / "summary.json")
    if sha256_file(d / "vectors.npy") != rec["files"]["vectors.npy"]:
        raise ValueError(f"{name}: saved vectors changed")
    W = np.load(d / "vectors.npy", allow_pickle=False)
    m = rec["meta"]
    texts, owner = face_index(bm.texts)
    log_event("diag", "STARTED", step="d54_variants", name=name, faces=len(texts))
    tok, model = enc.load(m["recipe"], m["model_path"])
    F = enc.encode(model, tok, texts, m["recipe"], m["max_length"], batch_size=batch_size,
                   dev=torch.device("cpu"))
    eo = effect_only(bm, ds["tags"], MODIFIER_LABELS)
    out_dir = RUNS / "diag" / "d54" / "v2" / name
    if (out_dir / "summary.json").exists():
        raise FileExistsError(f"{out_dir} already recorded")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "face_vectors.npy", F)
    res = {}
    both = lambda a: face_max(W, F, owner, a)  # noqa: E731
    cand = lambda a: face_max(W, F, owner, a, query_faces=False)  # noqa: E731
    for key, b, scorer in (("whole", bm, None), ("face_match", bm, both),
                           ("face_match_candidate", bm, cand), ("whole_effect_only", eo, None),
                           ("face_match_effect_only", eo, both),
                           ("face_match_candidate_effect_only", eo, cand)):
        r = evaluate(b, None if scorer else W, scorer=scorer)
        r["per_anchor"].to_parquet(out_dir / f"{key}.parquet", index=False)
        res[key] = {"summary": {v: r["summary"][v] for v in ("full", "hard_lexical")},
                    "multi_face": cohort_summary(b, r["per_anchor"],
                                                 {"multi_face": bm.cohorts()["multi_face"]}
                                                 )["multi_face"]}
    recorded = rec["summary"]["full"]["ndcg@10"]
    if abs(res["whole"]["summary"]["full"]["ndcg@10"] - recorded) > 1e-9:
        raise ValueError("whole-card scoring does not reproduce the recorded evaluation")
    out = {"name": name, "benchmark_id": bid, "faces": len(texts),
           "multi_face_cards": int(len(np.unique(owner))), "results": res,
           "face_vectors_sha256": sha256_file(out_dir / "face_vectors.npy"),
           "excluded_labels": list(MODIFIER_LABELS), "created_utc": utcnow(),
           "claim_scope": "diagnostic on selection-exposed development anchors"}
    atomic_write_json(out_dir / "summary.json", out)
    log_event("diag", "COMPLETED", step="d54_variants", name=name,
              **{k: v["summary"]["full"]["ndcg@10"] for k, v in res.items()})
    return out
