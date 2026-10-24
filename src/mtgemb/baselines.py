"""Evaluate fixed systems (lexical, pretrained encoders, trained checkpoints) on the benchmark
and store immutable evaluation records under runs/eval/<benchmark_id>/<name>/."""

from __future__ import annotations

import json
import numbers
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import encoder as enc
from .evaluate import Benchmark, bench_path, cohort_summary, evaluate, load_benchmark
from .lexical import LEXICAL_RECIPE, fit_vectorizer
from .train import DeviceLock
from .util import (
    ARTIFACTS,
    ROOT,
    RUNS,
    atomic_write_json,
    git_identity,
    log_event,
    read_json,
    sha256_file,
    utcnow,
)
from .variants import face_index, face_max

V4_K = 50  # v4 bundles store 50 line and ability matches; export.K grew to 200 in v5
V5_K = 200  # v5+ store 200 whole-card, line and (v6) face matches; ability matches stay 50
WUBRG = "WUBRG"
TOPK_ATOL = 1e-5  # float32 export vs float64 replay; export's own reference check uses 1e-6
CLAIM_SCOPE = {"dev": "tag-proxy relevance on selection-exposed development anchors",
               "sealed": "tag-proxy relevance on sealed anchors (authorised final evaluation)"}
COS_MIN = 0.9999  # sealed re-encoding must reproduce the frozen dev vectors


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
           "claim_scope": CLAIM_SCOPE[bm.held]}
    atomic_write_json(out / "summary.json", rec)
    log_event("eval", "COMPLETED", name=name, full=res["summary"]["full"]["ndcg@10"],
              hard_lexical=res["summary"]["hard_lexical"]["ndcg@10"])
    return rec


def _lexical_vectors(bm: Benchmark, ds: dict):
    c = ds["cards"]
    fit = c[(c["split"] == "fit") & (c["text_status"] == "ok")]
    return fit_vectorizer(fit["text"].tolist()).transform(bm.texts)


def eval_lexical(ds_dir, name="lexical-char35-tfidf", held: str = "dev") -> dict:
    bm, ds = load_benchmark(ds_dir, held)
    res = evaluate(bm, _lexical_vectors(bm, ds))
    return save_eval(name, bm, res, {"system": "lexical", "recipe": LEXICAL_RECIPE,
                                     "fitted_on": "fit text only"})


def same_weights(v: np.ndarray, ids: list[str], ds_dir, name: str) -> float:
    """Min row cosine between v and the saved dev-eval vectors of `name` on the dev universe.

    Raises below COS_MIN: a sealed score must come from exactly the frozen weights.
    """
    dev = read_json(bench_path(ds_dir))
    ref_dir = RUNS / "eval" / dev["benchmark_id"] / name
    rec = read_json(ref_dir / "summary.json")
    if sha256_file(ref_dir / "vectors.npy") != rec["files"]["vectors.npy"]:
        raise ValueError(f"{name}: saved dev vectors changed")
    ref = np.load(ref_dir / "vectors.npy", allow_pickle=False)
    pos = {o: i for i, o in enumerate(ids)}
    x = v[[pos[o] for o in dev["universe"]]]
    cos = (x * ref).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(ref, axis=1))
    lo = float(cos.min())
    if not lo >= COS_MIN:
        raise ValueError(f"{name}: re-encoded dev vectors differ (min cosine {lo:.6f})")
    return lo


def eval_encoder(ds_dir, name: str, backbone: str, max_length: int,
                 model_path: str | None = None, batch_size: int = 128,
                 held: str = "dev") -> dict:
    bm, _ = load_benchmark(ds_dir, held)
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
    # sealed: prove the weights are the frozen ones before any sealed anchor is scored
    min_cos = same_weights(v, bm.ids, ds_dir, name) if held != "dev" else None
    res = evaluate(bm, v)
    meta = {"system": "encoder", "backbone": backbone, "recipe": recipe,
            "max_length": max_length, "model_path": model_path, "device": str(dev),
            "torch": torch.__version__, "encode_seconds": round(secs, 1),
            "truncated_universe": len(truncated),
            f"truncated_{held}_anchors": int(sum(bm.ids[i] in truncated
                                                 for i in bm.anchor_idx))}
    if min_cos is not None:
        meta["dev_vectors_min_cosine"] = min_cos
    return save_eval(name, bm, res, meta, vectors=v, truncated=truncated)


METRICS = ["ndcg", "ap", "mrr", "recall"]
COUNTS = ["n_rel", "n_candidates"]
SCHEMAS = [f"mtgemb-release-v{i}" for i in range(1, 7)]


def _bundle_needs(ver: int, shards: int) -> list[str]:
    """Files a bundle of schema version ver must list."""
    files = ["names.json"] + ["ids.json"] * (ver >= 2) + ["tags.json"] * (ver >= 4)
    dirs = ["cards", "neighbours"] + ["abilities"] * (ver >= 3) + ["lines"] * (ver >= 4) + [
        "faces"] * (ver >= 6)
    return files + [f"{d}/{i:03d}.json" for d in dirs for i in range(shards)]


def _is_bool(x) -> bool:
    return isinstance(x, bool | np.bool_)


def _num_diff(a, b) -> float:
    """Largest numeric difference between two JSON-like values; inf on any structural or
    non-numeric mismatch (NaN matches only NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max((_num_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max((_num_diff(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    if _is_bool(a) or _is_bool(b):  # True must not match the count 1
        return 0.0 if _is_bool(a) and _is_bool(b) and bool(a) == bool(b) else float("inf")
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        if np.isnan(a) or np.isnan(b):
            return 0.0 if np.isnan(a) and np.isnan(b) else float("inf")
        return abs(float(a) - float(b))
    return 0.0 if type(a) is type(b) and a == b else float("inf")


def _canon(x) -> str:
    """Canonical JSON: True never equals 1, 3.0 never equals 3, key order is irrelevant."""
    return json.dumps(x, sort_keys=True)


def _path_or_null(x) -> bool:
    """model_path: null means pretrained; anything else must be a real path string."""
    return x is None or (isinstance(x, str) and bool(x.strip()))


def _encoder_key(recipe, max_length, model_path) -> str:
    return _canon([recipe, max_length,
                   None if model_path is None else str(_under_root(model_path))])


def _under_root(p) -> Path:
    p = Path(p)
    return (p if p.is_absolute() else ROOT / p).resolve()


def _bad_vectors(v) -> str | None:
    """Saved embeddings must be a finite float matrix (integers overflow when subtracted)."""
    if v.ndim != 2 or not np.issubdtype(v.dtype, np.floating):
        return f"vectors are not a float matrix: {v.dtype} {v.shape}"
    if not np.isfinite(v).all():
        return "vectors contain non-finite values"
    return None


def _replay_one(d: Path, bm: Benchmark, ds: dict, bid: str, tol: float) -> tuple[dict, object]:
    """Replay one evaluation record; returns (result with problems, vectors or None)."""
    if not (d / "summary.json").exists():
        return {"ok": False, "problems": ["no summary.json"]}, None
    rec = read_json(d / "summary.json")
    probs = []
    if rec.get("benchmark_id") != bid:
        probs.append(f"benchmark_id {rec.get('benchmark_id')} != {bid}")
    # a record must say what produced it, or no release can be attributed to it
    m = rec.get("meta") or {}
    if m.get("system") == "lexical":
        # the refit below always uses LEXICAL_RECIPE, so the record must claim exactly that
        if _canon(m.get("recipe")) != _canon(LEXICAL_RECIPE):
            probs.append("lexical recipe differs from the refit recipe")
    elif m.get("system") == "encoder":
        # the encoder contract: exactly one of the recipes the encoder module can run
        if _canon(m.get("recipe")) not in {_canon(r) for r in enc.RECIPES.values()}:
            probs.append("encoder recipe is not a known encoder.RECIPES entry")
        if type(m.get("max_length")) is not int or m["max_length"] <= 0:
            probs.append("incomplete metadata: encoder max_length required")
        if "model_path" not in m or not _path_or_null(m["model_path"]):
            # absent is unknown weights, not pretrained: pretrained is an explicit null
            probs.append("incomplete metadata: encoder model_path required (null if pretrained)")
    else:
        probs.append("incomplete metadata: system and recipe required")
    for f, h in rec["files"].items():
        if not (d / f).exists() or sha256_file(d / f) != h:
            probs.append(f"hash mismatch or missing: {f}")
    if "per_anchor.parquet" not in rec["files"]:
        probs.append("per_anchor.parquet not recorded")
    v = None
    if "vectors.npy" in rec["files"] and (d / "vectors.npy").exists():
        v = np.load(d / "vectors.npy", allow_pickle=False)
        if bad := _bad_vectors(v):
            return {"ok": False, "problems": probs + [bad]}, None
        if m.get("system") == "encoder" and isinstance(m.get("recipe"), dict) \
                and m["recipe"].get("dim") != v.shape[1]:
            probs.append(f"recipe dim {m['recipe'].get('dim')} != vector width {v.shape[1]}")
        res = evaluate(bm, v)
    elif m.get("system") == "lexical":
        res = evaluate(bm, _lexical_vectors(bm, ds))
    else:
        return {"ok": False, "problems": probs + ["no saved vectors to replay"]}, None
    out = {"replayed_ndcg@10": {k: res["summary"][k]["ndcg@10"]
                                for k in ("full", "hard_lexical")}}
    if not (d / "per_anchor.parquet").exists():
        return {"ok": False, "problems": probs + ["per_anchor.parquet missing"], **out}, v
    old = pd.read_parquet(d / "per_anchor.parquet")
    need = METRICS + COUNTS + ["anchor", "view", "family"]
    missing = [c for c in need if c not in old.columns]
    if missing:
        return {"ok": False, "problems": probs + [f"columns missing: {missing}"], **out}, v
    # numeric, never bool: True must not pass as the count 1 after the float cast below
    bad = [c for c in METRICS + COUNTS if not pd.api.types.is_numeric_dtype(old[c])
           or pd.api.types.is_bool_dtype(old[c])]
    if bad:
        return {"ok": False, "problems": probs + [f"non-numeric columns: {bad}"], **out}, v
    old = old.set_index(["anchor", "view"]).sort_index()
    new = res["per_anchor"].set_index(["anchor", "view"]).sort_index()
    if not old.index.equals(new.index):
        return {"ok": False, "problems": probs + ["anchor/view rows differ"], **out}, v
    # compare as float whatever the stored dtype; NaN (zero-positive anchors) only matches NaN
    a, b = old[METRICS].to_numpy(dtype=float), new[METRICS].to_numpy(dtype=float)
    if (np.isnan(a) != np.isnan(b)).any():
        probs.append("NaN pattern differs")
    out["max_abs_diff"] = float(np.nanmax(np.abs(a - b), initial=0.0))
    if not out["max_abs_diff"] <= tol:  # NaN-safe
        probs.append(f"per-anchor metrics differ by {out['max_abs_diff']}")
    if not (old[COUNTS].to_numpy(dtype=float) == new[COUNTS].to_numpy(dtype=float)).all():
        probs.append("candidate/relevant counts differ")
    if not (old["family"].astype(str).to_numpy() == new["family"].astype(str).to_numpy()).all():
        probs.append("families differ")
    out["summary_max_diff"] = _num_diff(rec["summary"], res["summary"])
    if not out["summary_max_diff"] <= tol:
        probs.append(f"recorded summary differs by {out['summary_max_diff']}")
    return {"ok": not probs, "problems": probs, **out}, v


def _check_ranked(pairs, k: int, n: int, has_text: list, self_index: int, what: str) -> None:
    """A stored ranking: at most k [card, score] pairs of other text cards, no repeats,
    finite scores in non-increasing order."""
    if not isinstance(pairs, list) or len(pairs) > k:
        raise ValueError(f"invalid {what} list length")
    seen, previous = set(), float("inf")
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"invalid {what} pair")
        j, score = pair
        if (type(j) is not int or not 0 <= j < n or has_text[j] is not True or
                j == self_index or j in seen or
                type(score) not in (int, float) or not np.isfinite(score) or
                score > previous):
            raise ValueError(f"invalid {what} index, self, duplicate or order")
        seen.add(j)
        previous = score


def _check_v4_shards(bundle: Path, man: dict, n: int, ver: int = 4) -> list[str]:
    """Validate tag attribution and the stored rankings (v4+), beyond file integrity; v5+
    also the display fields and the 200 whole-card neighbours, v6 the face-match lists."""
    try:
        labels = read_json(bundle / "tags.json")["labels"]
        if not isinstance(labels, list) or not all(isinstance(t, str) for t in labels):
            raise ValueError("invalid tag labels")
        ss, k = man["shard_size"], man["line_k"]
        want = V4_K if ver == 4 else V5_K
        counts = [("line_k", want), ("ability_k", V4_K)] + [("k", V5_K)] * (ver >= 5) + [
            ("face_k", V5_K)] * (ver >= 6)
        if type(ss) is not int or ss <= 0 or any(
                type(man.get(key)) is not int or man[key] != value for key, value in counts):
            raise ValueError("invalid line_k, ability_k, k, face_k or shard_size")
        for key in ("line_status", "tags_status", "abilities_status") + (
                ("face_status",) if ver >= 6 else ()):
            if not isinstance(man.get(key), str) or not man[key].startswith("exploratory"):
                raise ValueError(f"invalid {key}: must start with exploratory")
        card_shards = [read_json(bundle / "cards" / f"{start // ss:03d}.json")
                       for start in range(0, n, ss)]
        has_text = [card["has_text"] for cards in card_shards for card in cards]
        if len(has_text) != n:
            raise ValueError("v4 card count mismatch")

        def tags(value):
            if (not isinstance(value, list) or
                    any(type(t) is not int or not 0 <= t < len(labels) for t in value) or
                    value != sorted(set(value))):
                raise ValueError("tag indices invalid or out of range")
            return set(value)

        for start in range(0, n, ss):
            name = f"{start // ss:03d}.json"
            cards = card_shards[start // ss]
            abilities = read_json(bundle / "abilities" / name)
            lines = read_json(bundle / "lines" / name)
            ranked = {"neighbours": read_json(bundle / "neighbours" / name)} if ver >= 5 else {}
            if ver >= 6:
                ranked["faces"] = read_json(bundle / "faces" / name)
            size = min(ss, n - start)
            if any(not isinstance(x, list) or len(x) != size
                   for x in (cards, abilities, lines, *ranked.values())):
                raise ValueError("v4 shard length mismatch")
            triples = zip(cards, abilities, lines, strict=True)
            for offset, (card, ability, line) in enumerate(triples):
                for what, rows in ranked.items():
                    _check_ranked(rows[offset], V5_K, n, has_text, start + offset, what)
                    if not card["has_text"] and rows[offset]:
                        raise ValueError(f"textless card has {what}")
                if ver >= 5:
                    c = card["colors"]
                    if not isinstance(c, str) or c != "".join(x for x in WUBRG if x in c):
                        raise ValueError("invalid colors")
                    b = card["image_back"]
                    if b is not None and not (isinstance(b, str) and b.strip()):
                        raise ValueError("invalid image_back")
                ct = tags(card["tags"])
                if not isinstance(ability, list):
                    raise ValueError("invalid ability list")
                for a in ability:
                    if not tags(a["tags"]).issubset(ct):
                        raise ValueError("ability tags are not a subset of card tags")
                _check_ranked(line, k, n, has_text, start + offset, "line neighbour")
                if not card["has_text"] and (ct or ability or line):
                    raise ValueError("textless card has tags or line neighbours")
        return []
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"invalid v{ver} bundle: {e}"]


def _check_scores(bundle: Path, man: dict, v: np.ndarray, art: Path, meta: dict,
                  cards: pd.DataFrame, ver: int, n_rows: int = 20, tol: float = 1e-4
                  ) -> list[str]:
    """Stored rankings against the release vectors (v5+), for seeded sample cards.

    Every stored score must equal the recomputed score of its card within tol (the bundle
    rounds to 4 decimals), and the recomputed scores of the stored cards must be the true
    top-k within TOPK_ATOL (tie-aware: only float32 near-ties may swap). Whole-card lists
    are cosine; v6 face-match lists score a candidate by the best of its whole card and its
    own faces against the query's whole card.
    """
    try:
        return _scores_problems(bundle, man, v, art, meta, cards.reset_index(drop=True), ver,
                                n_rows, tol)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
        return [f"stored rankings unreadable: {type(e).__name__}: {e}"]


def _scores_problems(bundle, man, v, art, meta, cards, ver, n_rows, tol) -> list[str]:
    has_text = (cards["text_status"] == "ok").to_numpy()
    ti = np.flatnonzero(has_text)
    row_of = {int(c): r for r, c in enumerate(ti)}
    if v.shape[0] != len(ti):
        return ["release vectors do not have one row per text card"]
    W = v.astype(np.float64)
    rows = np.sort(np.random.default_rng(20261023).choice(
        len(ti), size=min(n_rows, len(ti)), replace=False))
    kinds = {"neighbours": W[rows] @ W.T}
    probs = []
    if ver >= 6:
        texts, owner = face_index(cards.loc[has_text, "text"].tolist())
        p = art / "face_vectors.npy"
        F = np.load(p, allow_pickle=False) if p.is_file() else None
        if (F is None or meta.get("n_faces") != len(texts) or F.shape != (len(texts), v.shape[1])
                or not np.issubdtype(F.dtype, np.floating) or not np.isfinite(F).all()
                or not np.allclose(np.linalg.norm(F.astype(np.float64), axis=1), 1, atol=1e-3,
                                   rtol=0)):
            return probs + ["face vectors are not unit-norm floats, one row per catalogue face"]
        kinds["faces"] = face_max(W, F.astype(np.float64), owner, rows, query_faces=False)
    ss, shards = man["shard_size"], {}
    for what, S in kinds.items():
        k = min(man["face_k" if what == "faces" else "k"], len(ti) - 1)
        for i, r in enumerate(rows.tolist()):
            c = int(ti[r])
            key = (what, c // ss)
            if key not in shards:
                shards[key] = read_json(bundle / what / f"{c // ss:03d}.json")
            stored = shards[key][c % ss]
            s = S[i].copy()
            s[r] = -np.inf
            top = np.sort(s)[::-1][:k]
            got = [(row_of.get(j), score) for j, score in stored]
            if (len(stored) != k or any(j is None or abs(s[j] - score) > tol for j, score in got)
                    or np.abs(s[[j for j, _ in got]] - top).max(initial=0.0) > TOPK_ATOL):
                probs.append(f"{what} scores of card {cards['oracle_id'].iloc[c]} differ from "
                             "the release vectors")
    return probs


def _check_release(meta_p: Path, bm: Benchmark, ds: dict, metas: dict, vecs: dict,
                   tol: float) -> dict:
    """Release integrity, and identity with the evaluated encoder it was exported from."""
    if not meta_p.is_file():
        return {"ok": False, "problems": ["no readable release_meta.json"]}
    meta = read_json(meta_p)
    probs, ver = [], 0  # ver: bundle schema version, 0 when unreadable or unknown
    if meta.get("dataset_id") != ds["ident"]["dataset_id"]:
        probs.append(f"dataset_id {meta.get('dataset_id')} != {ds['ident']['dataset_id']}")
    vp = meta_p.parent / "vectors.npy"
    if not vp.exists() or sha256_file(vp) != meta["vectors_sha256"]:
        probs.append("vectors.npy hash mismatch or missing")
    rows = ds["cards"].sort_values("oracle_id")
    if rows.loc[rows["text_status"] == "ok", "oracle_id"].tolist() != list(bm.ids):
        probs.append("release rows are not the benchmark universe")
    bundle = _under_root(meta["bundle_dir"])
    if not (bundle / "release.json").exists():
        probs.append(f"bundle missing: {bundle}")
    elif sha256_file(bundle / "release.json") != meta["release_json_sha256"]:
        probs.append("release.json hash mismatch")
    else:
        man = read_json(bundle / "release.json")
        for f, e in man["files"].items():
            p = bundle / f
            if not p.exists() or (sha256_file(p), p.stat().st_size) != (e["sha256"], e["bytes"]):
                probs.append(f"bundle file mismatch or missing: {f}")
        # the manifest must be complete for its schema, not merely self-consistent
        if man.get("release_id") != meta.get("release_id"):
            probs.append(f"bundle release_id {man.get('release_id')} != {meta.get('release_id')}")
        n_cat = len(ds["cards"])  # catalogue: every fit+dev card, with or without vectors
        if man.get("n_cards") != n_cat:
            probs.append(f"bundle n_cards {man.get('n_cards')} != {n_cat}")
        ss = man.get("shard_size")
        if man.get("schema") not in SCHEMAS:
            probs.append(f"unknown bundle schema {man.get('schema')}")
        elif type(ss) is not int or ss <= 0:
            probs.append(f"bundle shard_size {ss!r} is not a positive integer")
        else:
            ver = SCHEMAS.index(man["schema"]) + 1
            need = _bundle_needs(ver, -(-n_cat // ss))
            if absent := sorted(set(need) - set(man["files"])):
                probs.append(f"bundle manifest incomplete: {absent[:5]}")
        arts = [("ability_vectors.npy", "ability_vectors_sha256"),
                ("ability_lines.json", "ability_lines_sha256")] * (ver >= 3) + [
                ("face_vectors.npy", "face_vectors_sha256")] * (ver >= 6)
        for f, k in arts:
            p = meta_p.parent / f
            if not meta.get(k) or not p.exists() or sha256_file(p) != meta[k]:
                probs.append(f"{f} hash mismatch or missing")
        if ver >= 4:
            probs.extend(_check_v4_shards(bundle, man, n_cat, ver))
    enc_meta = meta["encoder"]
    enc_meta = dict(enc_meta) if isinstance(enc_meta, dict) else None  # never coerce a list
    ml = enc_meta.pop("max_length", None) if enc_meta is not None else None
    key = None  # matches no evaluation
    if enc_meta is None:
        probs.append("release encoder must be a JSON object")
    elif not _path_or_null(meta["model_path"]):
        probs.append("release model_path must be null (pretrained) or a path")
    else:  # the evaluation(s) of exactly this encoder: same recipe, length and weights
        key = _encoder_key(enc_meta, ml, meta["model_path"])
    match = sorted(n for n, k in metas.items() if k == key)
    out = {"run_id": meta["run_id"], "compared_with": match, "max_abs_diff_vs_eval": {}}
    if not match:
        probs.append("no recorded evaluation of this encoder to compare with")
    if meta["run_id"] is not None and meta["run_id"] not in match:
        probs.append(f"run_id {meta['run_id']} has no matching evaluation")
    if meta.get("release_id") != meta_p.parent.name:
        probs.append(f"release_id {meta.get('release_id')} != directory {meta_p.parent.name}")
    if not vp.exists():
        return {"ok": False, "problems": probs, **out}
    v = np.load(vp, allow_pickle=False)
    if bad := _bad_vectors(v):
        return {"ok": False, "problems": probs + [f"release {bad}"], **out}
    if ver >= 5:
        probs.extend(_check_scores(bundle, read_json(bundle / "release.json"), v,
                                   meta_p.parent, meta, rows, ver))
    for n in match:
        a, b = v.astype(np.float64), vecs[n].astype(np.float64)
        diff = float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
        out["max_abs_diff_vs_eval"][n] = diff
        if not diff <= tol:  # NaN-safe
            probs.append(f"vectors differ from evaluation {n}")
    return {"ok": not probs, "problems": probs, **out}


def replay_evals(ds_dir, tol: float = 1e-9) -> dict:
    """Evaluation replay from pinned embeddings; no encoding, no training.

    Every directory under runs/eval/<benchmark_id>/ must be a complete record: its benchmark
    id and file hashes are verified, its saved vectors re-scored (lexical: refit on fit text),
    and every per-anchor metric, count and family and the recorded summary compared. Every
    exported release must belong to this dataset, have intact vectors and bundle files, and
    hold vectors equal to a recorded evaluation of the same encoder (recipe, length, weights).
    Passes only if at least one evaluation replays and nothing has a problem. Cohort tables
    are not replayed. Writes runs/replay/<benchmark_id>/<utc>-<ns>.json, never overwritten.
    """
    def guarded(fn, *a):  # a corrupt record is a failed check, never a lost report
        try:
            return fn(*a)
        except Exception as e:
            return {"ok": False, "problems": [f"replay error: {type(e).__name__}: {e}"]}

    bm, ds = load_benchmark(ds_dir)
    bid = bm.spec["benchmark_id"]
    evals, vecs, metas = {}, {}, {}
    root = RUNS / "eval" / bid

    def replay_eval(d):  # everything that reads the record runs inside guarded()
        res, v = _replay_one(d, bm, ds, bid, tol)
        if v is not None and res["ok"]:
            m = read_json(d / "summary.json")["meta"]
            if m["system"] == "encoder":  # only encoders are exported as releases
                metas[d.name] = _encoder_key(m["recipe"], m["max_length"], m["model_path"])
                vecs[d.name] = v
        return res

    # every entry is a record, checked inside guarded(): an unreadable one fails, never vanishes
    for d in sorted(root.iterdir()) if root.exists() else []:
        evals[d.name] = guarded(replay_eval, d)
    # likewise every release entry, except unpublished exports: export publishes by renaming
    # .partial dirs, and failed ones are kept aside as .failed-N
    rel = ARTIFACTS / "releases"
    dirs = sorted(rel.iterdir()) if rel.exists() else []
    unpublished = [p.name for p in dirs if p.name.endswith(".partial") or ".failed-" in p.name]
    # keyed by directory, so two releases claiming one release_id cannot hide each other
    releases = {p.name: guarded(_check_release, p / "release_meta.json", bm, ds, metas, vecs,
                                tol) for p in dirs if p.name not in unpublished}
    ok = bool(evals) and all(e["ok"] for e in evals.values()) and all(
        r["ok"] for r in releases.values())
    out = {"benchmark_id": bid, "dataset_id": ds["ident"]["dataset_id"], "tolerance": tol,
           "passed": ok, "evals": evals, "releases": releases,
           "unpublished_release_dirs": unpublished, "code": git_identity(),
           "created_utc": utcnow(),
           "scope": "evaluation replay from pinned embeddings; not a retraining replay; "
                    "cohort tables not replayed"}
    # unique name: a replay report is never overwritten by a later replay
    path = RUNS / "replay" / bid / f"{out['created_utc'].replace(':', '')}-{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, out)
    log_event("eval", "REPLAYED", benchmark_id=bid, passed=ok, report=str(path))
    return out
