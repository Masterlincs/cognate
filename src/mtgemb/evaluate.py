"""Tag-proxy retrieval evaluation (MASTER_PLAN §5).

Benchmark: a frozen candidate universe U (fit+dev cards with usable text; untagged cards
stay in with zero proxy gain), sorted by Oracle ID, and frozen development anchors. Every system
supplies one L2-normalisable row per U card; scores are cosine; ties break by Oracle ID
ascending. Each view's mask is applied before both the retrieved and the ideal ranking.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text import word_tokens
from .util import atomic_write_json, read_json, sha256_json, utcnow

BENCHMARK_VERSION = "benchmark_v3"  # v2 + dataset identity bound into the id
K = 10
VIEWS = ("full", "hard", "hard_lexical", "cross_colour", "cross_cost", "cross_body", "strict",
         "dev_candidates")
SEALED_VIEWS = VIEWS[:-1] + ("sealed_candidates",)  # same views, sealed diagnostic
HELD = ("dev", "sealed")
HARD_LEXICAL_JACCARD = 0.8
CROSS_COST_MIN = 2.0
PILOT_THRESHOLDS = {"full_ndcg_gain": 0.005, "hard_lexical_min_change": -0.005,
                    "checkpoint_guardrail": -0.005}
PROMOTION = {"full_point": 0.005, "full_lower": 0.0, "hard_lexical_point": 0.0,
             "hard_lexical_lower": -0.010, "alpha": 0.05}
BROAD_DF = 500
RARE_DF = 10


# ---------------------------------------------------------------- benchmark

def build_benchmark(cards: pd.DataFrame, mem: pd.DataFrame, tags: pd.DataFrame,
                    dataset_id: str, held: str = "dev") -> dict:
    """Freeze universe, anchors and rules, bound to the dataset that supplies relevance
    inputs (tags, weights, text, metadata): the id changes if any of them change.

    held="sealed" (, authorised final evaluation only): sealed cards join the universe
    and are the anchors; weights, taxonomy and view rules stay the fit-derived ones. The
    held="dev" spec is unchanged by this option.
    """
    if held not in HELD:
        raise ValueError(f"held must be one of {HELD}")
    fit_tags = set(tags.loc[tags["df_fit"] > 0, "tag_id"])
    tagged = set(mem["oracle_id"])
    splits = ["fit", "dev"] + (["sealed"] if held == "sealed" else [])
    u = cards[cards["split"].isin(splits) & (cards["text_status"] == "ok")
              ].sort_values("oracle_id")
    fit_tagged = set(mem.loc[mem["tag_id"].isin(fit_tags), "oracle_id"])
    anchors = u[(u["split"] == held) & u["oracle_id"].isin(fit_tagged)]["oracle_id"].tolist()
    dev_all = cards[(cards["split"] == held) & (cards["text_status"] == "ok")]
    dev_mem = mem[mem["oracle_id"].isin(set(dev_all["oracle_id"]))]
    bench = {
        "dataset_id": dataset_id,
        "universe": u["oracle_id"].tolist(),
        "anchors": anchors,
        "rules": {
            "version": BENCHMARK_VERSION,
            "universe": "split in {fit, dev}, text_status ok; untagged cards have zero gain",
            "anchors": "dev cards in universe with >=1 tag represented in fit",
            "ranking": "cosine desc, then oracle_id asc", "k": K,
            "views": VIEWS, "hard_lexical_jaccard": HARD_LEXICAL_JACCARD,
            "jaccard_tokens": "lowercase [a-z0-9]+ runs; empty set -> 0",
            "cross_cost_min": CROSS_COST_MIN,
            "creature_policy": "'Creature' in first face type line",
            "zero_idcg": "excluded per view, counted",
            "pilot_thresholds": PILOT_THRESHOLDS, "promotion": PROMOTION,
        },
        "counts": {
            "universe": len(u), "universe_untagged": int((~u["oracle_id"].isin(tagged)).sum()),
            "anchors": len(anchors),
            f"{held}_ok_cards": len(dev_all),
            f"{held}_untagged": int((~dev_all["oracle_id"].isin(tagged)).sum()),
            f"{held}_tags_unseen_in_fit": int(dev_mem.loc[~dev_mem["tag_id"].isin(fit_tags),
                                                          "tag_id"].nunique()),
            f"{held}_memberships_unseen_tag_frac": float(
                (~dev_mem["tag_id"].isin(fit_tags)).mean()) if len(dev_mem) else 0.0,
        },
        "created_utc": utcnow(),
    }
    if held == "sealed":
        bench["rules"].update(
            universe="split in {fit, dev, sealed}, text_status ok; untagged cards have zero gain",
            anchors="sealed cards in universe with >=1 tag represented in fit",
            views=SEALED_VIEWS, held_split="sealed")
    bench["benchmark_id"] = _bench_id(bench)
    return bench


def _bench_id(bench: dict) -> str:
    return sha256_json([bench["dataset_id"], bench["universe"], bench["anchors"],
                        bench["rules"]])[:16]


def load_benchmark(ds_dir, held: str = "dev") -> tuple[Benchmark, dict]:
    """Frozen benchmark plus its dataset; refuses a spec edited or built on other inputs.

    Only held="sealed" (the authorised final evaluation) parses sealed rows.
    """
    from .data import load_dataset

    if held not in HELD:
        raise ValueError(f"held must be one of {HELD}")
    ds = load_dataset(ds_dir, include_sealed=held == "sealed")
    spec = read_json(bench_path(ds_dir, held))
    if spec.get("dataset_id") != ds["ident"]["dataset_id"] or _bench_id(spec) != spec[
            "benchmark_id"] or spec["rules"].get("held_split", "dev") != held:
        raise ValueError("benchmark does not match its dataset or has been edited")
    return Benchmark(spec, ds["cards"], ds["memberships"], ds["tags"]), ds


class Benchmark:
    """Precomputed relevance and view masks for a frozen benchmark."""

    def __init__(self, bench: dict, cards: pd.DataFrame, mem: pd.DataFrame, tags: pd.DataFrame):
        self.spec = bench
        self.held = bench["rules"].get("held_split", "dev")
        self.views = tuple(bench["rules"]["views"])
        self.ids = bench["universe"]
        pos = {o: i for i, o in enumerate(self.ids)}
        self.anchor_idx = np.array([pos[a] for a in bench["anchors"]], dtype=np.int64)
        c = cards.set_index("oracle_id").loc[self.ids]
        self.split = c["split"].to_numpy()
        self.family = c["family_id"].to_numpy()
        self.colour = c["color_identity"].to_numpy()
        self.mv = c["mv"].to_numpy(dtype=float)
        self.creature = c["is_creature"].to_numpy(dtype=bool)
        self.texts = c["text"].tolist()
        self.n_faces = c["n_faces"].to_numpy()
        # weighted tag incidence over fit-represented tags: gain = A_q diag(w) A_c^T
        w = tags.loc[tags["df_fit"] > 0].set_index("tag_id")["idf_fit"]
        self.df_fit = tags.set_index("tag_id")["df_fit"]
        m = mem[mem["oracle_id"].isin(pos) & mem["tag_id"].isin(w.index)]
        tcol = {t: j for j, t in enumerate(w.index)}
        rows = m["oracle_id"].map(pos).to_numpy()
        cols = m["tag_id"].map(tcol).to_numpy()
        self.A = sp.csr_matrix((np.ones(len(m)), (rows, cols)), shape=(len(self.ids), len(w)))
        self.Aw = self.A @ sp.diags(w.to_numpy())
        self.card_tags = m.groupby("oracle_id")["tag_id"].apply(list).to_dict()
        tok = CountVectorizer(analyzer=lambda s: sorted(word_tokens(s)), binary=True)
        self.T = tok.fit_transform(self.texts).astype(np.float32).tocsr()
        self.tsize = np.asarray(self.T.sum(1)).ravel()

    def gains(self, a: np.ndarray) -> np.ndarray:
        return (self.Aw[a] @ self.A.T).toarray()

    def jaccard(self, a: np.ndarray) -> np.ndarray:
        inter = (self.T[a] @ self.T.T).toarray()
        union = self.tsize[a][:, None] + self.tsize[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.where(union > 0, inter / union, 0.0)
        j[(self.tsize[a] == 0)[:, None] | (self.tsize == 0)[None, :]] = 0.0
        return j

    def masks(self, a: np.ndarray) -> dict[str, np.ndarray]:
        n = len(self.ids)
        base = np.ones((len(a), n), dtype=bool)
        base[np.arange(len(a)), a] = False
        hard = base & (self.family[a][:, None] != self.family[None, :])
        same_col = self.colour[a][:, None] == self.colour[None, :]
        same_body = self.creature[a][:, None] == self.creature[None, :]
        return {
            "full": base,
            "hard": hard,
            "hard_lexical": hard & (self.jaccard(a) < HARD_LEXICAL_JACCARD),
            "cross_colour": base & ~same_col,
            "cross_cost": base & (np.abs(self.mv[a][:, None] - self.mv[None, :])
                                  >= CROSS_COST_MIN),
            "cross_body": base & ~same_body,
            "strict": base & same_col & same_body & (self.mv[a][:, None] == self.mv[None, :]),
            f"{self.held}_candidates": base & (self.split == self.held)[None, :],
        }

    def cohorts(self, truncated: set[str] | None = None) -> dict[str, np.ndarray]:
        """Boolean masks over anchors for diagnostics."""
        out = {"multi_face": [], "rare_effect": [], "broad_only": [], "multi_ability": [],
               "long_text": []}
        lens = np.array([len(t) for t in self.texts])
        long_cut = np.quantile(lens[self.anchor_idx], 0.9)
        for i in self.anchor_idx:
            oid = self.ids[i]
            dfs = [int(self.df_fit[t]) for t in self.card_tags.get(oid, [])]
            out["multi_face"].append(self.n_faces[i] > 1)
            out["rare_effect"].append(bool(dfs) and min(dfs) <= RARE_DF)
            out["broad_only"].append(bool(dfs) and min(dfs) > BROAD_DF)
            out["multi_ability"].append(self.texts[i].count("\n") >= 2)
            out["long_text"].append(lens[i] >= long_cut)
        res = {k: np.array(v, dtype=bool) for k, v in out.items()}
        if truncated is not None:
            res["truncated"] = np.array([self.ids[i] in truncated for i in self.anchor_idx])
        return res


# ---------------------------------------------------------------- metrics

_DISC = 1.0 / np.log2(np.arange(2, K + 2))


def rank_metrics(ranked_gains: np.ndarray, ideal_gains: np.ndarray, n_rel: int) -> dict:
    """Metrics for one query from the top-K gains (retrieved order) and top-K ideal gains.

    ranked_gains/ideal_gains may be shorter than K when fewer candidates are permitted.
    """
    g = np.asarray(ranked_gains, dtype=float)
    ig = np.asarray(ideal_gains, dtype=float)
    dcg = float((g * _DISC[: len(g)]).sum())
    idcg = float((ig * _DISC[: len(ig)]).sum())
    rel = g > 0
    hits = np.cumsum(rel)
    ap = float((hits / np.arange(1, len(g) + 1) * rel).sum() / min(n_rel, K)) if n_rel else 0.0
    first = np.flatnonzero(rel)
    return {
        "ndcg": dcg / idcg if idcg > 0 else math.nan,
        "ap": ap,
        "mrr": float(1.0 / (first[0] + 1)) if len(first) else 0.0,
        "recall": float(rel.sum() / n_rel) if n_rel else math.nan,
        "n_rel": int(n_rel),
    }


def validate_vectors(x, n: int) -> object:
    """Reject wrong shape, non-finite or zero rows; return L2-normalised rows."""
    if x.shape[0] != n:
        raise ValueError(f"expected {n} rows, got {x.shape[0]}")
    if sp.issparse(x):
        x = sp.csr_matrix(x, dtype=np.float64)
        if not np.isfinite(x.data).all():
            raise ValueError("non-finite values in sparse vectors")
        norms = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
        if (norms == 0).any():
            raise ValueError(f"{int((norms == 0).sum())} zero vectors")
        return sp.diags(1.0 / norms) @ x
    x = np.asarray(x, dtype=np.float64)  # float64 norms: no overflow for finite float32 input
    if x.ndim != 2:
        raise ValueError("vectors must be 2-D")
    if not np.isfinite(x).all():
        raise ValueError("non-finite values in vectors")
    norms = np.linalg.norm(x, axis=1)
    if not (norms > 0).all() or not np.isfinite(norms).all():
        raise ValueError(f"{int((~(norms > 0)).sum())} zero or unnormalisable vectors")
    return (x / norms[:, None]).astype(np.float32)


def evaluate(bm: Benchmark, vectors, block: int = 256, top_n: int = 0) -> dict:
    """Score every anchor in every view. Returns per-anchor rows and macro means.

    top_n > 0 also returns the full-view top-n neighbour ids per anchor (for error analysis).
    """
    x = validate_vectors(vectors, len(bm.ids))
    xt = x.T
    rows, tops = [], {}
    a_all = bm.anchor_idx
    for s in range(0, len(a_all), block):
        a = a_all[s : s + block]
        sc = x[a] @ xt
        sc = sc.toarray() if sp.issparse(sc) else np.asarray(sc)
        gains = bm.gains(a)
        masks = bm.masks(a)
        for j, ai in enumerate(a):
            order = np.argsort(-sc[j], kind="stable")  # ties -> lower index = lower oracle id
            g = gains[j]
            if top_n:
                o = order[masks["full"][j][order]][:top_n]
                tops[bm.ids[ai]] = [bm.ids[k] for k in o]
            for v in bm.views:
                m = masks[v][j]
                ranked = g[order[m[order]][:K]]
                gm = g[m]
                n_rel = int((gm > 0).sum())
                ideal = -np.sort(-gm)[:K] if n_rel else np.zeros(0)
                r = rank_metrics(ranked, ideal, n_rel)
                r.update(anchor=bm.ids[ai], view=v, n_candidates=int(m.sum()),
                         family=bm.family[ai])
                rows.append(r)
    per = pd.DataFrame(rows)
    return {"per_anchor": per, "summary": summarise(per), "top": tops}


def summarise(per: pd.DataFrame) -> dict:
    out = {}
    for v, g in per.groupby("view", sort=False):
        ok = g[g["n_rel"] > 0]
        out[v] = {
            "anchors_total": len(g), "anchors_scored": len(ok),
            "zero_positive_excluded": int((g["n_rel"] == 0).sum()),
            "ndcg@10": float(ok["ndcg"].mean()), "map@10": float(ok["ap"].mean()),
            "mrr@10": float(ok["mrr"].mean()), "recall@10": float(ok["recall"].mean()),
            "median_candidates": float(g["n_candidates"].median()),
        }
    return out


def cohort_summary(bm: Benchmark, per: pd.DataFrame, cohorts: dict, view="full") -> dict:
    g = per[(per["view"] == view) & (per["n_rel"] > 0)].set_index("anchor")
    out = {}
    for name, mask in cohorts.items():
        ids = [bm.ids[i] for i, keep in zip(bm.anchor_idx, mask, strict=True) if keep]
        sub = g.loc[g.index.intersection(ids)]
        out[name] = {"anchors": len(sub), "ndcg@10": float(sub["ndcg"].mean()) if len(sub)
                     else None}
    return out


# ---------------------------------------------------------------- paired bootstrap

def paired_bootstrap(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, *, n_boot: int,
                     seed: int, lower_q: float = 0.05) -> dict:
    """Paired difference mean(a) - mean(b) with whole-cluster resampling.

    a, b: (n_seeds, n_anchors) per-anchor metric (n_seeds=1 for untrained systems; a
    single-seed b is broadcast). Seeds are matched by index and resampled jointly with
    clusters: each replicate draws seed indices with replacement and clusters with
    replacement, then averages the matched per-anchor differences.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    if b.shape[0] == 1 and a.shape[0] > 1:
        b = np.repeat(b, a.shape[0], 0)
    if a.shape[0] == 1 and b.shape[0] > 1:
        a = np.repeat(a, b.shape[0], 0)
    if a.shape != b.shape or a.shape[1] != len(clusters):
        raise ValueError("a, b and clusters must align")
    d = a - b  # (S, n)
    uc, inv = np.unique(clusters, return_inverse=True)
    nc, ns = len(uc), d.shape[0]
    csum = np.zeros((ns, nc))
    for s in range(ns):
        csum[s] = np.bincount(inv, weights=d[s], minlength=nc)
    csize = np.bincount(inv, minlength=nc).astype(float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    chunk = max(1, 2_000_000 // nc)
    for s0 in range(0, n_boot, chunk):
        e = min(n_boot, s0 + chunk)
        cnt = rng.multinomial(nc, np.full(nc, 1.0 / nc), size=e - s0)  # cluster draws
        sd = rng.integers(0, ns, size=(e - s0, ns))  # matched seed draws
        sbar = csum[sd].mean(1)  # (B, nc) seed-averaged cluster sums
        stats[s0:e] = (cnt * sbar).sum(1) / (cnt @ csize)
    point = float(d.mean())
    tail = int(np.floor(lower_q * n_boot))
    return {
        "point": point,
        "per_seed": [float(x) for x in d.mean(1)],
        "lower_adjusted": float(np.quantile(stats, lower_q)),
        "lower_q": lower_q,
        "ci95": [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))],
        "n_boot": n_boot, "tail_samples": tail, "seed": seed,
        "n_clusters": int(nc), "n_anchors": int(d.shape[1]), "n_seeds": int(ns),
    }


def n_boot_for(m_contrasts: int, alpha: float = 0.05, tail: int = 100) -> int:
    """Enough resamples for >= `tail` samples below the alpha/M quantile."""
    return int(math.ceil(tail * m_contrasts / alpha))


def bench_path(ds_dir, held: str = "dev") -> Path:
    return Path(ds_dir) / (f"{BENCHMARK_VERSION}.json" if held == "dev"
                           else f"{BENCHMARK_VERSION}_{held}.json")


def save_benchmark(bench: dict, path: Path) -> None:
    if Path(path).exists():
        old = read_json(path)
        if old["benchmark_id"] != bench["benchmark_id"]:
            raise FileExistsError(f"{path} holds a different frozen benchmark")
        return
    atomic_write_json(path, bench)
