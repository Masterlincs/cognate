"""Release export (P8): encode the permitted catalogue, exact top-k neighbours, static bundle.

The catalogue is every fit+dev card (sealed rows are never loaded). Cards without usable
text are listed but get no vector and no neighbours. The bundle follows web/SCHEMA.md;
vectors, ID order and parity results go to artifacts/releases/<release_id>/.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from . import encoder as enc
from .abilities import flatten, maxsim_reference, maxsim_topk
from .data import load_dataset
from .train import DeviceLock
from .util import (
    ARTIFACTS,
    ROOT,
    atomic_write_bytes,
    atomic_write_json,
    git_identity,
    log_event,
    sha256_file,
    sha256_json,
    utcnow,
)

SCHEMA = "mtgemb-release-v3"  # per-ability neighbours
K = 50
SHARD = 256
ATTRIBUTION = ("Card data and images © Wizards of the Coast; data from Scryfall (scryfall.com). "
               "Not affiliated with or endorsed by Wizards of the Coast or Scryfall. "
               "Tags from Scryfall Tagger.")


def topk_neighbours(v: np.ndarray, k: int = K, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by cosine for unit rows, excluding self; ties -> lower row index.

    Rows must be in Oracle ID order so the index tie-break is the Oracle ID tie-break.
    Scores a (block, n) slab at a time; never builds the n x n matrix.
    """
    n = len(v)
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    sco = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        e = min(n, s + block)
        sc = v[s:e] @ v.T
        sc[np.arange(e - s), np.arange(s, e)] = -np.inf
        order = np.argsort(-sc, axis=1, kind="stable")[:, :k]
        idx[s:e] = order
        sco[s:e] = np.take_along_axis(sc, order, axis=1)
    return idx, sco


def reference_check(v: np.ndarray, nidx: np.ndarray, rows: np.ndarray, atol: float = 1e-6):
    """Tie-aware check of the block top-k against an independent float64 full sort.

    Passes a row when the float64 scores of the block's picks equal the true top-k scores
    within atol (float32 near-ties may swap order). Returns (reference lists, failures,
    exact-list matches).
    """
    v64 = v.astype(np.float64)
    k = nidx.shape[1]
    ref, bad, exact = {}, [], 0
    for r in rows:
        sc = v64 @ v64[r]
        sc[r] = -np.inf
        order = np.lexsort((np.arange(len(v)), -sc))[:k]
        ref[int(r)] = order.tolist()
        if not np.allclose(sc[nidx[r]], sc[order], rtol=0, atol=atol):
            bad.append(int(r))
        exact += int(nidx[r].tolist() == ref[int(r)])
    return ref, bad, exact


def ranking_parity(q: np.ndarray, v: np.ndarray, rows: np.ndarray, nidx: np.ndarray,
                   k: int = 10, tol: float = 1e-4) -> np.ndarray:
    """Tie-aware top-k agreement per probe: CPU query rows q (for catalogue rows `rows`)
    ranked against the device catalogue v, versus the device top-k in nidx.

    A CPU pick counts when it is in the device top-k or its device score is within tol of
    the device k-th score (a near-tie swapped across the cut-off). tol is a chosen
    tolerance, not derived from the vector parity bound.
    """
    k = min(k, len(v) - 1)
    if k < 1:
        return np.ones(len(rows))  # no other card to rank
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        sc = v @ q[i]
        sc[r] = -np.inf
        top = np.argsort(-sc, kind="stable")[:k]
        dev_sc = v @ v[r]
        kth = dev_sc[nidx[r, k - 1]]
        dev_set = set(nidx[r, :k].tolist())
        out[i] = np.mean([j in dev_set or dev_sc[j] >= kth - tol for j in top.tolist()])
    return out


def release_digest(dataset_id: str, ids: list[str], encoder: dict, model_hash: str,
                   vec_hash: str, schema: str = SCHEMA,
                   ability_vec_hash: str | None = None) -> str:
    """Everything that defines a release's content and layout; same inputs -> same id."""
    content = [schema, dataset_id, sha256_json(ids), encoder, model_hash, vec_hash]
    if ability_vec_hash is not None:
        content.append(ability_vec_hash)
    return sha256_json(content)[:8]


def ability_neighbours(texts: list[str], encode, parity_n: int = 64,
                       ref_n: int = 20) -> dict:
    """Encode lines and check max-sim picks; encode(lines, cpu=bool) is injectable.

    Neighbour indices and owners here are text rows, mapped to catalogue rows on export.
    The caller holds DeviceLock for both CPU and device encoding.
    """
    lines, owner, starts = flatten(texts)
    rng = np.random.default_rng(20261017)
    probe = np.sort(rng.choice(len(lines), size=min(max(1, parity_n), len(lines)), replace=False))
    cpu = encode([lines[i] for i in probe], cpu=True)
    v = encode(lines, cpu=False)
    cos = (cpu * v[probe]).sum(1)
    if not np.isfinite(cos).all() or cos.min() < 0.9999:
        raise ValueError(f"ability device/CPU parity failed: min cosine {cos.min():.6f}")
    counts = np.diff(np.r_[starts, len(lines)])
    queries = np.flatnonzero(counts[owner] >= 2)
    idx, sco = maxsim_topk(v[queries], owner[queries], v, starts, K)
    ref_rows = np.sort(rng.choice(len(queries), size=min(max(20, ref_n), len(queries)),
                                 replace=False))
    exact = 0
    for r in ref_rows:
        q = queries[r]
        sc = maxsim_reference(v[q], int(owner[q]), v, owner, len(texts))
        order = np.lexsort((np.arange(len(texts)), -sc))[:idx.shape[1]]
        if not np.allclose(sc[idx[r]], sc[order], rtol=0, atol=1e-6):
            raise ValueError(f"ability block top-k disagrees with float64 reference on line {q}")
        exact += int(np.array_equal(idx[r], order))
    rows: list[list] = [[] for _ in texts]
    for r, q in enumerate(queries):
        rows[owner[q]].append({"text": lines[q], "neighbours": [
            [int(j), round(float(s), 4)] for j, s in zip(idx[r], sco[r], strict=True)]})
    return {"lines": lines, "owner": owner, "vectors": v, "rows": rows,
            "n_queries": int(len(queries)),
            "parity": {"cpu_vs_device_min_cos": float(cos.min()), "n_probe": int(len(probe)),
                       "topk_reference_rows": int(len(ref_rows)), "topk_reference_failures": 0,
                       "topk_reference_exact_lists": exact}}


def finish_interrupted(final_out: Path, art: Path, final_art: Path) -> None:
    """Complete a publication interrupted between the bundle and artifact renames.

    The bundle is published first and the artifacts last, so final_art existing means the
    release is complete. Only artifacts that name this exact release.json are adopted.
    """
    meta_p = art / "release_meta.json"
    if not meta_p.exists() or json.loads(meta_p.read_text("utf-8"))[
            "release_json_sha256"] != sha256_file(final_out / "release.json"):
        raise FileExistsError(f"bundle {final_out} exists without matching artifacts")
    os.replace(art, final_art)


def _dump(path: Path, obj) -> dict:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    atomic_write_bytes(path, data)
    return {"sha256": sha256_file(path), "bytes": len(data)}


def export(ds_dir, backbone: str, max_length: int, label: str, status: str,
           model_path: str | None = None, run_id: str | None = None,
           web_dir: str | Path | None = None, parity_n: int = 64, ref_n: int = 20) -> dict:
    if status not in ("promoted", "experimental", "pretrained-fallback"):
        raise ValueError(status)
    ds = load_dataset(ds_dir)
    cards = ds["cards"].sort_values("oracle_id").reset_index(drop=True)
    if set(cards["split"]) - {"fit", "dev"}:
        raise ValueError("catalogue must contain only fit and dev cards")
    has_text = (cards["text_status"] == "ok").to_numpy()
    ti = np.flatnonzero(has_text)
    texts = cards.loc[ti, "text"].tolist()

    recipe = enc.RECIPES[backbone]
    tok, model = enc.load(recipe, model_path)
    dev = enc.device()
    rng = np.random.default_rng(20261017)
    probe = np.sort(rng.choice(len(ti), size=min(parity_n, len(ti)), replace=False))
    with DeviceLock():
        cpu = enc.encode(model, tok, [texts[i] for i in probe], recipe, max_length,
                         dev=torch.device("cpu"))
        model.to(dev)
        v = enc.encode(model, tok, texts, recipe, max_length, batch_size=128, dev=dev)

        def encode_lines(lines, *, cpu):
            target = torch.device("cpu") if cpu else dev
            model.to(target)
            return enc.encode(model, tok, lines, recipe, max_length, batch_size=256, dev=target)

        ability = ability_neighbours(texts, encode_lines, parity_n, ref_n)
    cos = (cpu * v[probe]).sum(1)
    if cos.min() < 0.9999:
        raise ValueError(f"device/CPU parity failed: min cosine {cos.min():.6f}")
    nidx, nsco = topk_neighbours(v)
    overlap = ranking_parity(cpu, v, probe, nidx)
    if overlap.mean() < 0.98:
        raise ValueError(f"CPU/device top-10 overlap {overlap.mean():.4f} < 0.98")
    ref_rows = np.sort(rng.choice(len(ti), size=min(ref_n, len(ti)), replace=False))
    ref, bad, exact = reference_check(v, nidx, ref_rows)
    if bad:
        raise ValueError(f"block top-k disagrees with float64 reference on rows {bad[:5]}")

    vec_hash = hashlib.sha256(v.tobytes()).hexdigest()
    ability_vec_hash = hashlib.sha256(ability["vectors"].tobytes()).hexdigest()
    model_hash = sha256_json(recipe) if model_path is None else sha256_file(
        Path(model_path) / "model.safetensors")
    snapshot = ds["ident"].get("snapshot") or Path(ds_dir).name.split("_")[0]
    encoder_meta = {**recipe, "max_length": max_length}
    release_id = f"{snapshot}-{backbone}-" + release_digest(
        ds["ident"]["dataset_id"], cards["oracle_id"].tolist(), encoder_meta, model_hash,
        vec_hash, ability_vec_hash=ability_vec_hash)
    web = Path(web_dir or ROOT / "web" / "public")
    final_out = web / "data" / release_id
    final_art = ARTIFACTS / "releases" / release_id
    # write into unpublished .partial dirs; rename bundle first, artifacts last
    out, art = (p.with_name(p.name + ".partial") for p in (final_out, final_art))
    if final_art.exists():
        raise FileExistsError(f"release {release_id} already exported; never overwritten")
    if final_out.exists():
        finish_interrupted(final_out, art, final_art)
        log_event("export", "COMPLETED", release_id=release_id, recovered=True)
        return {"release_id": release_id, "bundle": str(final_out),
                "artifacts": str(final_art), "parity": json.loads(
                    (final_art / "release_meta.json").read_text("utf-8"))["parity"]}
    for p in (out, art):
        if p.exists():
            shutil.rmtree(p)  # leftover from a failed export; never published
    (out / "cards").mkdir(parents=True)
    (out / "neighbours").mkdir(parents=True)
    (out / "abilities").mkdir(parents=True)
    art.mkdir(parents=True)

    neigh: list[list] = [[] for _ in range(len(cards))]
    ability_rows: list[list] = [[] for _ in range(len(cards))]
    for r, ci in enumerate(ti):  # text-subset row r is catalogue card ci
        neigh[ci] = [[int(ti[j]), round(float(s), 4)] for j, s in zip(nidx[r], nsco[r],
                                                                      strict=True)]
        ability_rows[ci] = [{"text": a["text"], "neighbours": [
            [int(ti[j]), score] for j, score in a["neighbours"]]} for a in ability["rows"][r]]
    files = {}
    aliases = []
    recs = []
    for i, c in cards.iterrows():
        faces = json.loads(c["faces"])
        if len(faces) > 1:
            aliases += [[f["name"], i] for f in faces if f["name"] and f["name"] != c["name"]]
        recs.append({"id": c["oracle_id"], "name": c["name"], "layout": c["layout"],
                     "faces": [{k: f.get(k) or "" for k in
                                ("name", "mana_cost", "type_line", "oracle_text")}
                               for f in faces],
                     "color_identity": c["color_identity"], "mv": float(c["mv"]),
                     "type_line": c["type_line"], "is_creature": bool(c["is_creature"]),
                     "image": c["image"] or None, "scryfall_uri": c["scryfall_uri"],
                     "has_text": bool(has_text[i])})
    # names.json is loaded at startup; the poorly compressing ids load beside it
    files["names.json"] = _dump(out / "names.json",
                                {"names": cards["name"].tolist(), "aliases": aliases})
    files["ids.json"] = _dump(out / "ids.json", {"ids": cards["oracle_id"].tolist()})
    for s in range(0, len(cards), SHARD):
        nn = f"{s // SHARD:03d}.json"
        files[f"cards/{nn}"] = _dump(out / "cards" / nn, recs[s:s + SHARD])
        files[f"neighbours/{nn}"] = _dump(out / "neighbours" / nn, neigh[s:s + SHARD])
        files[f"abilities/{nn}"] = _dump(out / "abilities" / nn, ability_rows[s:s + SHARD])
    model_meta = {"label": label, "status": status, "backbone": recipe["model_id"],
                  "revision": recipe["revision"], "run_id": run_id}
    if status == "promoted":
        model_meta["decision"] = ("provisional tag-proxy promotion on selection-exposed "
                                  "development")
    release = {"schema": SCHEMA, "release_id": release_id, "fixture": False,
               "snapshot_date": snapshot, "created_utc": utcnow(), "model": model_meta,
               "n_cards": len(cards), "k": K, "shard_size": SHARD, "files": files,
               "ability_k": K,
               "abilities_status": "exploratory: not evaluated on the tag proxy",
               "attribution": ATTRIBUTION}
    atomic_write_json(out / "release.json", release)

    np.save(art / "vectors.npy", v)
    np.save(art / "ability_vectors.npy", ability["vectors"])
    atomic_write_json(art / "ability_lines.json", [
        {"text": line, "oracle_id": cards.loc[ti[own], "oracle_id"]}
        for line, own in zip(ability["lines"], ability["owner"], strict=True)])
    meta = {"release_id": release_id, "dataset_id": ds["ident"]["dataset_id"],
            "encoder": encoder_meta, "model_path": model_path,
            "model_hash": model_hash, "run_id": run_id, "dim": int(v.shape[1]),
            "vector_rows": "catalogue cards with has_text, in oracle_id order",
            "vectors_sha256": sha256_file(art / "vectors.npy"),
            "n_cards": len(cards), "n_with_text": int(len(ti)),
            "ability_vectors_sha256": sha256_file(art / "ability_vectors.npy"),
            "ability_lines_sha256": sha256_file(art / "ability_lines.json"),
            "n_ability_lines": len(ability["lines"]), "n_ability_queries": ability["n_queries"],
            "ability_parity": {"device": str(dev), **ability["parity"]},
            "parity": {"device": str(dev), "cpu_vs_device_min_cos": float(cos.min()),
                       "n_probe": int(len(probe)),
                       "cpu_query_top10_overlap_mean": float(overlap.mean()),
                       "cpu_query_top10_overlap_min": float(overlap.min()),
                       "topk_reference_rows": int(len(ref_rows)),
                       "topk_reference_failures": 0, "topk_reference_exact_lists": exact},
            "reference_neighbours": {cards.loc[ti[r], "oracle_id"]:
                                     [cards.loc[ti[j], "oracle_id"] for j in ref[int(r)]]
                                     for r in ref_rows},
            "bundle_dir": str(final_out), "release_json_sha256": sha256_file(out / "release.json"),
            "code": git_identity(), "created_utc": utcnow()}
    atomic_write_json(art / "release_meta.json", meta)
    os.replace(out, final_out)
    os.replace(art, final_art)
    log_event("export", "COMPLETED", release_id=release_id, status=status, run_id=run_id)
    return {"release_id": release_id, "bundle": str(final_out), "artifacts": str(final_art),
            "parity": meta["parity"]}
