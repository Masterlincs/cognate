"""Line-based whole-card scores (, inference only).

line_mean:  mean over the anchor's lines of the best cosine against any candidate line.
line_invdf: the same, each anchor line weighted 1/df, df = universe cards that have that
            line after number masking, so a clause many cards share cannot swamp a rare one.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import exam
from .abilities import flatten, split_abilities
from .evaluate import evaluate, load_benchmark
from .exam import mask_numbers
from .util import RUNS, atomic_write_json, git_identity, log_event, read_json, sha256_file, utcnow

VARIANTS = ("line_mean", "line_invdf")


class LineScorer:
    def __init__(self, texts: list[str], line_v: np.ndarray, variant: str):
        """texts: universe rows (each with >=1 line); line_v: unit vectors of every line in
        row order (abilities.split_abilities), as the release stores them."""
        if variant not in VARIANTS:
            raise ValueError(variant)
        lines, self.owner, self.starts = flatten(texts)
        if len(lines) != len(line_v):
            raise ValueError("line vectors do not match the universe texts")
        self.v = line_v
        self.n = len(texts)
        if variant == "line_mean":
            self.w = np.ones(len(lines))
        else:
            keys = [mask_numbers(x) for x in lines]
            cards: dict[str, set[int]] = {}
            for k, o in zip(keys, self.owner.tolist(), strict=True):
                cards.setdefault(k, set()).add(o)
            self.w = np.array([1.0 / len(cards[k]) for k in keys])

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        """(len(rows), n) card scores for anchor rows."""
        q = np.concatenate([np.arange(self.starts[r], self._end(r)) for r in rows])
        best = np.maximum.reduceat(self.v[q] @ self.v.T, self.starts, axis=1)
        w = self.w[q][:, None]
        seg = np.r_[0, np.cumsum([self._end(r) - self.starts[r] for r in rows])[:-1]]
        return np.add.reduceat(best * w, seg, axis=0) / np.add.reduceat(w, seg, axis=0)

    def _end(self, r: int) -> int:
        return self.starts[r + 1] if r + 1 < self.n else len(self.owner)


def run(ds_dir, release_dir) -> dict:
    """whole-card baseline vs both variants on dev (selection-exposed) and the exam."""
    rd = Path(release_dir)
    bm, _ = load_benchmark(ds_dir)
    meta = read_json(rd / "release_meta.json")
    for f, k in (("vectors.npy", "vectors_sha256"), ("ability_vectors.npy",
                                                     "ability_vectors_sha256")):
        if sha256_file(rd / f) != meta[k]:
            raise ValueError(f"{f} does not match its release hash")
    owners = [x["oracle_id"] for x in read_json(rd / "ability_lines.json")]
    if list(dict.fromkeys(owners)) != list(bm.ids):
        raise ValueError("release rows are not the dev benchmark universe in order")
    whole = np.load(rd / "vectors.npy")
    line_v = np.load(rd / "ability_vectors.npy")
    log_event("eval", "STARTED", task="linescore", release=meta["release_id"])
    pos = {o: i for i, o in enumerate(bm.ids)}
    items = [it for it in exam.parse(exam.EXAM.read_text("utf-8"))
             if it["status"] == "accepted" and it.get("line") is None]
    single = np.array([len(split_abilities(bm.texts[i])) == 1 for i in bm.anchor_idx])
    out = {}
    for name in ("whole", *VARIANTS):
        sc = (lambda a: whole[a] @ whole.T) if name == "whole" else LineScorer(
            bm.texts, line_v, name)
        r = evaluate(bm, None if name != "whole" else whole, scorer=sc)
        per = r["per_anchor"]
        res = {v: r["summary"][v]["ndcg@10"] for v in ("full", "hard_lexical")}
        full = per[(per["view"] == "full") & (per["n_rel"] > 0)].set_index("anchor")["ndcg"]
        for lab, mask in (("single_line", single), ("multi_line", ~single)):
            ids = [bm.ids[i] for i, m in zip(bm.anchor_idx, mask, strict=True) if m]
            sub = full.loc[full.index.intersection(ids)]
            res[f"full_{lab}"] = {"anchors": len(sub), "ndcg@10": float(sub.mean())}
        ex = []
        for it in items:  # whole-card items only; line anchors are max-over-lines already
            s = sc(np.array([pos[it["anchor"]]]))[0]
            ex.append({"id": it["id"], "margin": float(s[pos[it["match"]]] - s[pos[it["trap"]]])})
        res["exam_whole_items"] = {"n": len(ex), "passed": sum(e["margin"] > 0 for e in ex),
                                   "failed": [e["id"] for e in ex if e["margin"] <= 0]}
        out[name] = res
    rec = {"decision": "", "release_id": meta["release_id"],
           "benchmark_id": bm.spec["benchmark_id"], "results": out,
           "worksheet_sha256": sha256_file(exam.EXAM), "code": git_identity(),
           "created_utc": utcnow(), "note": "selection-exposed dev; inference only"}
    p = RUNS / "diag" / "linescore" / f"{time.time_ns()}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, rec)
    log_event("eval", "LINESCORE_DONE", record=str(p),
              **{k: {m: v[m] for m in ("full", "hard_lexical")} for k, v in out.items()})
    return rec


def topk(scorer: LineScorer, k: int, block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k rows per row under the scorer, own row excluded; ties -> lower index."""
    n = scorer.n
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    sco = np.empty((n, k))
    for s in range(0, n, block):
        rows = np.arange(s, min(n, s + block))
        sc = scorer(rows)
        sc[np.arange(len(rows)), rows] = -np.inf
        order = np.argsort(-sc, axis=1, kind="stable")[:, :k]
        idx[rows] = order
        sco[rows] = np.take_along_axis(sc, order, axis=1)
    return idx, sco
