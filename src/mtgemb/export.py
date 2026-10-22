"""Release export (P8): encode the permitted catalogue, exact top-k neighbours, static bundle.

The catalogue is every fit+dev card (sealed rows are never loaded). Cards without usable
text are listed but get no vector and no neighbours. The bundle follows web/SCHEMA.md;
vectors, ID order and parity results go to artifacts/releases/<release_id>/.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from . import encoder as enc
from . import linescore, tagattr
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
from .variants import face_index, face_max

SCHEMA = "mtgemb-release-v6"  # v5 plus face-match neighbours
K = 200  # whole-card, line-match and face-match neighbours stored per card
ABILITY_K = 50
WUBRG = "WUBRG"
SHARD = 256
ATTRIBUTION = ("Card data and images © Wizards of the Coast; data from Scryfall (scryfall.com). "
               "Not affiliated with or endorsed by Wizards of the Coast or Scryfall. "
               "Tags from Scryfall Tagger.")


def card_display(card: dict) -> dict:
    """Display-only fields from a raw Scryfall card: colours and the back-face image.

    Colours are the card's printed colours (top level, or the union of faces for cards that
    keep them per face). Only cards printed with two faces (transform, modal DFC) have a back
    image; adventure, split and flip cards are one image.
    """
    faces = card.get("card_faces") or []
    cols = set(card["colors"]) if "colors" in card else {c for f in faces
                                                         for c in f.get("colors") or []}
    back = None
    if not card.get("image_uris") and len(faces) > 1:
        back = (faces[1].get("image_uris") or {}).get("normal")
    return {"colors": "".join(c for c in WUBRG if c in cols), "image_back": back}


def display_fields(raw_dir: Path, ids: list[str], manifest_sha256: str) -> dict[str, dict]:
    """card_display for the catalogue cards, from the dataset's own raw snapshot.

    Other records (sealed cards among them) are parsed only for their oracle_id and dropped.
    """
    raw_dir = Path(raw_dir)
    if sha256_file(raw_dir / "manifest.json") != manifest_sha256:
        raise ValueError("raw manifest does not match the dataset")
    f = json.loads((raw_dir / "manifest.json").read_text("utf-8"))["files"]["oracle_cards"]
    if sha256_file(raw_dir / f["path"]) != f["sha256"]:
        raise ValueError("raw card file hash mismatch")
    want, out = set(ids), {}
    with gzip.open(raw_dir / f["path"], "rt", encoding="utf-8") as fh:
        for line in fh:
            card = json.loads(line)
            if card.get("oracle_id") in want:
                out[card["oracle_id"]] = card_display(card)
    if missing := want - out.keys():
        raise ValueError(f"{len(missing)} catalogue cards missing from the raw snapshot")
    return out


def topk_neighbours(v: np.ndarray, k: int = K, block: int = 512,
                    faces: tuple[np.ndarray, np.ndarray] | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by cosine for unit rows, excluding self; ties -> lower row index.

    Rows must be in Oracle ID order so the index tie-break is the Oracle ID tie-break.
    Scores a (block, n) slab at a time; never builds the n x n matrix. faces=(F, owner)
    scores face match: a candidate scores its best of whole card and own faces
    against the query's whole-card vector.
    """
    n = len(v)
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    sco = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        e = min(n, s + block)
        sc = v[s:e] @ v.T if faces is None else face_max(v, *faces, np.arange(s, e),
                                                          query_faces=False)
        sc[np.arange(e - s), np.arange(s, e)] = -np.inf
        order = np.argsort(-sc, axis=1, kind="stable")[:, :k]
        idx[s:e] = order
        sco[s:e] = np.take_along_axis(sc, order, axis=1)
    return idx, sco


def reference_check(v: np.ndarray, nidx: np.ndarray, rows: np.ndarray, atol: float = 1e-6,
                    faces: tuple[np.ndarray, np.ndarray] | None = None):
    """Tie-aware check of the block top-k against an independent float64 full sort
    (face match when faces=(F, owner) is given).

    Passes a row when the float64 scores of the block's picks equal the true top-k scores
    within atol (float32 near-ties may swap order). Returns (reference lists, failures,
    exact-list matches).
    """
    v64 = v.astype(np.float64)
    k = nidx.shape[1]
    ref, bad, exact = {}, [], 0
    for r in rows:
        sc = v64 @ v64[r]
        if faces is not None:
            np.maximum.at(sc, faces[1], faces[0].astype(np.float64) @ v64[r])
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
                   ability_vec_hash: str | None = None,
                   line_hash: str | None = None, tags_hash: str | None = None,
                   face_hash: str | None = None) -> str:
    """Everything that defines a release's content and layout; same inputs -> same id."""
    content = [schema, dataset_id, sha256_json(ids), encoder, model_hash, vec_hash]
    if ability_vec_hash is not None:
        content.append(ability_vec_hash)
    if line_hash is not None or tags_hash is not None:
        content.extend([line_hash, tags_hash])
    if face_hash is not None:
        content.append(face_hash)
    return sha256_json(content)[:8]


def ability_neighbours(texts: list[str], encode, parity_n: int = 64,
                       ref_n: int = 20) -> dict:
    """Encode lines and check max-sim picks; encode(lines, cpu=bool) is injectable.

    Neighbour indices and owners here are text rows, mapped to catalogue rows on export.
    The caller holds DeviceLock for both CPU and device encoding.
    """
    lines, _, _ = flatten(texts)
    rng = np.random.default_rng(20261017)
    probe = np.sort(rng.choice(len(lines), size=min(max(1, parity_n), len(lines)), replace=False))
    cpu = encode([lines[i] for i in probe], cpu=True)
    v = encode(lines, cpu=False)
    cos = (cpu * v[probe]).sum(1)
    if not np.isfinite(cos).all() or cos.min() < 0.9999:
        raise ValueError(f"ability device/CPU parity failed: min cosine {cos.min():.6f}")
    result = rank_ability_neighbours(texts, v, ref_n)
    result["parity"].update(cpu_vs_device_min_cos=float(cos.min()), n_probe=int(len(probe)))
    return result


def rank_ability_neighbours(texts: list[str], v: np.ndarray, ref_n: int = 20) -> dict:
    """Rank encoded ability lines, including an independent float64 reference check."""
    lines, owner, starts = flatten(texts)
    if v.ndim != 2 or len(v) != len(lines) or not np.isfinite(v).all():
        raise ValueError("ability vectors do not match the texts or are non-finite")
    rng = np.random.default_rng(20261017)
    counts = np.diff(np.r_[starts, len(lines)])
    queries = np.flatnonzero(counts[owner] >= 2)
    idx, sco = maxsim_topk(v[queries], owner[queries], v, starts, ABILITY_K)
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
            "parity": {"topk_reference_rows": int(len(ref_rows)), "topk_reference_failures": 0,
                       "topk_reference_exact_lists": exact}}


def _reuse_vectors(release_id: str, dataset_id: str, model_hash: str, encoder: dict,
                   texts: list[str], ids: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    if not release_id or Path(release_id).name != release_id or release_id in (".", ".."):
        raise ValueError("from_release must be a release ID")
    root = ARTIFACTS / "releases" / release_id
    meta = json.loads((root / "release_meta.json").read_text("utf-8"))
    for key, expected in (("dataset_id", dataset_id), ("model_hash", model_hash),
                          ("encoder", encoder)):
        if meta.get(key) != expected:
            raise ValueError(f"source release {key} mismatch")
    for name, key in (("vectors.npy", "vectors_sha256"),
                      ("ability_vectors.npy", "ability_vectors_sha256"),
                      ("ability_lines.json", "ability_lines_sha256")):
        p = root / name
        if not p.is_file() or sha256_file(p) != meta.get(key):
            raise ValueError(f"source release {name} hash mismatch or missing")
    lines, owner, _ = flatten(texts)
    expected_lines = [{"text": line, "oracle_id": ids[o]}
                      for line, o in zip(lines, owner, strict=True)]
    if json.loads((root / "ability_lines.json").read_text("utf-8")) != expected_lines:
        raise ValueError("source release ability lines or oracle ids mismatch")
    v = np.load(root / "vectors.npy", allow_pickle=False)
    lv = np.load(root / "ability_vectors.npy", allow_pickle=False)
    for values, n in ((v, len(texts)), (lv, len(lines))):
        if (values.ndim != 2 or values.shape[0] != n or
                not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all()):
            raise ValueError("source release vectors have invalid shape, dtype or values")
    if v.shape[1] != lv.shape[1] or v.shape[1] != encoder["dim"]:
        raise ValueError("source release vector dimensions mismatch")
    for values in (v, lv):
        if not np.allclose(np.linalg.norm(values.astype(np.float64), axis=1), 1,
                           atol=1e-3, rtol=0):
            raise ValueError("source release vectors are not unit-norm")
    return v, lv, meta


def _reuse_faces(release_id: str, meta: dict, n: int, dim: int) -> np.ndarray | None:
    """A source release's face vectors, or None when it has none (pre-v6 releases)."""
    if "face_vectors_sha256" not in meta:
        return None
    p = ARTIFACTS / "releases" / release_id / "face_vectors.npy"
    if not p.is_file() or sha256_file(p) != meta["face_vectors_sha256"]:
        raise ValueError("source release face_vectors.npy hash mismatch or missing")
    f = np.load(p, allow_pickle=False)
    if f.shape != (n, dim) or not np.isfinite(f).all():
        raise ValueError("source release face vectors do not match the catalogue faces")
    return f


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


def model_meta(label: str, status: str, recipe: dict, run_id: str | None,
               decision: str | None = None) -> dict:
    """The release's model block; `decision` names the promotion decision (promoted only)."""
    check_decision(status, run_id, decision)
    meta = {"label": label, "status": status, "backbone": recipe["model_id"],
            "revision": recipe["revision"], "run_id": run_id}
    if status == "promoted":
        meta["decision"] = (decision or "").strip() or (
            "tag-proxy promotion on selection-exposed development "
            "; gate also cleared on one sealed run")
    return meta


def check_decision(status: str, run_id: str | None, decision: str | None) -> None:
    if decision is not None and status != "promoted":
        raise ValueError("a decision is only recorded for a promoted release")
    # only the first promoted run predates the argument; its text claims a sealed
    # run, so no other promoted model may fall back to it, including with a blank decision
    if status == "promoted" and not (decision or "").strip() and run_id != "confirm-e5-s1":
        raise ValueError("a promoted release needs an explicit decision")


def export(ds_dir, backbone: str, max_length: int, label: str, status: str,
           model_path: str | None = None, run_id: str | None = None,
           web_dir: str | Path | None = None, parity_n: int = 64, ref_n: int = 20,
           from_release: str | None = None, decision: str | None = None) -> dict:
    check_decision(status, run_id, decision)  # fail before encoding
    if status not in ("promoted", "experimental", "pretrained-fallback"):
        raise ValueError(status)
    ds = load_dataset(ds_dir)
    cards = ds["cards"].sort_values("oracle_id").reset_index(drop=True)
    if set(cards["split"]) - {"fit", "dev"}:
        raise ValueError("catalogue must contain only fit and dev cards")
    has_text = (cards["text_status"] == "ok").to_numpy()
    ti = np.flatnonzero(has_text)
    texts = cards.loc[ti, "text"].tolist()
    snapshot = ds["ident"].get("snapshot") or Path(ds_dir).name.split("_")[0]
    display = display_fields(ROOT / "data" / "raw" / snapshot, cards["oracle_id"].tolist(),
                             ds["ident"].get("raw_manifest_sha256"))

    recipe = enc.RECIPES[backbone]
    model_hash = sha256_json(recipe) if model_path is None else sha256_file(
        Path(model_path) / "model.safetensors")
    encoder_meta = {**recipe, "max_length": max_length}
    rng = np.random.default_rng(20261017)
    probe = np.sort(rng.choice(len(ti), size=min(parity_n, len(ti)), replace=False))
    source_meta = None
    if from_release is not None:
        v, line_v, source_meta = _reuse_vectors(
            from_release, ds["ident"]["dataset_id"], model_hash, encoder_meta,
            texts, cards.loc[ti, "oracle_id"].tolist())
        ability = rank_ability_neighbours(texts, line_v, ref_n)
    else:
        tok, model = enc.load(recipe, model_path)
        dev = enc.device()
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
    # face match: faces are short and few, so they are always encoded on the CPU
    face_texts, face_owner = face_index(texts)
    F = None if source_meta is None else _reuse_faces(from_release, source_meta,
                                                      len(face_texts), v.shape[1])
    if F is None:
        if source_meta is not None:
            tok, model = enc.load(recipe, model_path)
        model.to(torch.device("cpu"))
        F = enc.encode(model, tok, face_texts, recipe, max_length, dev=torch.device("cpu"))
    nidx, nsco = topk_neighbours(v)
    fidx, fsco = topk_neighbours(v, faces=(F, face_owner))
    if source_meta is None:
        overlap = ranking_parity(cpu, v, probe, nidx)
        if overlap.mean() < 0.98:
            raise ValueError(f"CPU/device top-10 overlap {overlap.mean():.4f} < 0.98")
    ref_rows = np.sort(rng.choice(len(ti), size=min(ref_n, len(ti)), replace=False))
    ref, bad, exact = reference_check(v, nidx, ref_rows)
    if bad:
        raise ValueError(f"block top-k disagrees with float64 reference on rows {bad[:5]}")
    _, fbad, fexact = reference_check(v, fidx, ref_rows, faces=(F, face_owner))
    if fbad:
        raise ValueError(f"face top-k disagrees with float64 reference on rows {fbad[:5]}")

    vec_hash = hashlib.sha256(v.tobytes()).hexdigest()
    ability_vec_hash = hashlib.sha256(ability["vectors"].tobytes()).hexdigest()
    tags = tagattr.release_tags(
        texts, cards.loc[ti, "split"].tolist(), cards.loc[ti, "oracle_id"].tolist(),
        v, ability["vectors"], ds["memberships"],
        sorted(ds["tags"].loc[ds["tags"]["functional"], "tag_id"].tolist()))
    lidx, lsco = linescore.topk(linescore.LineScorer(texts, ability["vectors"], "line_invdf"), K)
    line_hash = sha256_json([lidx.tolist(), lsco.tolist()])
    release_id = f"{snapshot}-{backbone}-" + release_digest(
        ds["ident"]["dataset_id"], cards["oracle_id"].tolist(), encoder_meta, model_hash,
        vec_hash, ability_vec_hash=ability_vec_hash, line_hash=line_hash,
        tags_hash=sha256_json(tags), face_hash=hashlib.sha256(F.tobytes()).hexdigest())
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
    (out / "lines").mkdir(parents=True)
    (out / "faces").mkdir(parents=True)
    art.mkdir(parents=True)

    neigh: list[list] = [[] for _ in range(len(cards))]
    line_rows: list[list] = [[] for _ in range(len(cards))]
    face_rows: list[list] = [[] for _ in range(len(cards))]
    card_tags: list[list] = [[] for _ in range(len(cards))]
    ability_rows: list[list] = [[] for _ in range(len(cards))]
    for r, ci in enumerate(ti):  # text-subset row r is catalogue card ci
        neigh[ci] = [[int(ti[j]), round(float(s), 4)] for j, s in zip(nidx[r], nsco[r],
                                                                      strict=True)]
        line_rows[ci] = [[int(ti[j]), round(float(score), 4)]
                         for j, score in zip(lidx[r], lsco[r], strict=True)]
        face_rows[ci] = [[int(ti[j]), round(float(s), 4)] for j, s in zip(fidx[r], fsco[r],
                                                                          strict=True)]
        card_tags[ci] = tags["card"][r]
        ability_rows[ci] = [{"text": a["text"], "tags": tags["lines"][r][k], "neighbours": [
            [int(ti[j]), score] for j, score in a["neighbours"]]} for k, a in
            enumerate(ability["rows"][r])]
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
                     "has_text": bool(has_text[i]), "tags": card_tags[i],
                     **display[c["oracle_id"]]})
    # names.json is loaded at startup; the poorly compressing ids load beside it
    files["names.json"] = _dump(out / "names.json",
                                {"names": cards["name"].tolist(), "aliases": aliases})
    files["ids.json"] = _dump(out / "ids.json", {"ids": cards["oracle_id"].tolist()})
    labels = ds["tags"].set_index("tag_id")["label"]
    files["tags.json"] = _dump(out / "tags.json", {"labels": [labels[t] for t in tags["tag_ids"]]})
    for s in range(0, len(cards), SHARD):
        nn = f"{s // SHARD:03d}.json"
        files[f"cards/{nn}"] = _dump(out / "cards" / nn, recs[s:s + SHARD])
        files[f"neighbours/{nn}"] = _dump(out / "neighbours" / nn, neigh[s:s + SHARD])
        files[f"lines/{nn}"] = _dump(out / "lines" / nn, line_rows[s:s + SHARD])
        files[f"faces/{nn}"] = _dump(out / "faces" / nn, face_rows[s:s + SHARD])
        files[f"abilities/{nn}"] = _dump(out / "abilities" / nn, ability_rows[s:s + SHARD])
    meta_model = model_meta(label, status, recipe, run_id, decision)
    release = {"schema": SCHEMA, "release_id": release_id, "fixture": False,
               "snapshot_date": snapshot, "created_utc": utcnow(), "model": meta_model,
               "n_cards": len(cards), "k": K, "shard_size": SHARD, "files": files,
               "ability_k": ABILITY_K, "line_k": K, "face_k": K,
               "face_status": "exploratory: face match; about level with the whole-card "
                              "ranking on the dev tag proxy",
               "line_status": "exploratory: line match; below the whole-card ranking on the "
                              "dev tag proxy",
               "tags_status": "exploratory: per-line tags by nearest one-line-card centroid",
               "abilities_status": "exploratory: not evaluated on the tag proxy",
               "attribution": ATTRIBUTION}
    atomic_write_json(out / "release.json", release)

    np.save(art / "vectors.npy", v)
    np.save(art / "ability_vectors.npy", ability["vectors"])
    np.save(art / "face_vectors.npy", F)
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
            "face_vectors_sha256": sha256_file(art / "face_vectors.npy"),
            "n_faces": len(face_texts), "face_rows": "face texts of catalogue text cards with "
            "two or more non-empty faces, in card then face order, CPU-encoded",
            "face_topk_reference_exact_lists": fexact,
            "n_ability_lines": len(ability["lines"]), "n_ability_queries": ability["n_queries"],
            "ability_parity": source_meta["ability_parity"] if source_meta is not None else
                              {"device": str(dev), **ability["parity"]},
            "parity": source_meta["parity"] if source_meta is not None else {
                       "device": str(dev), "cpu_vs_device_min_cos": float(cos.min()),
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
    if source_meta is not None:
        meta["vectors_from"] = from_release
    atomic_write_json(art / "release_meta.json", meta)
    os.replace(out, final_out)
    os.replace(art, final_art)
    log_event("export", "COMPLETED", release_id=release_id, status=status, run_id=run_id)
    return {"release_id": release_id, "bundle": str(final_out), "artifacts": str(final_art),
            "parity": meta["parity"]}
