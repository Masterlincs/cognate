"""Static data for the Cognate site.

Three outputs, all derived from existing records:
- a 2D map of a release's card vectors (t-SNE) with region labels from Tagger tags;
- pretrained e5-base-v2 neighbours for the before-and-after comparison;
- metrics.json: every number the site shows, each with its source record.

Site data goes to a new folder web/public/data/site-<release_id>/; the release is untouched.
Only fit and dev cards are read (load_dataset drops sealed rows).
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .data import load_dataset
from .export import SHARD, topk_neighbours
from .util import ARTIFACTS, ROOT, RUNS, atomic_write_json, read_json, sha256_file

DATASET = ROOT / "data" / "processed" / "2026-10-16_no_reminder_v1"
WEB = ROOT / "web"
DEV_BENCH = "8bf2f48c0bba3b2c"
SEALED_BENCH = "c98bb619ea613672"
SEEDS = (1, 2, 3)
DEV_MODELS = {
    "character n-gram TF-IDF": ["lexical-char35-tfidf"],
    "pretrained BGE-small": ["pretrained-bge-small-en-v1.5"],
    "pretrained E5-base": ["pretrained-e5-base-v2"],
    "fine-tuned BGE-small": [f"confirm-bge-s{s}" for s in SEEDS],
    "fine-tuned E5, 2000 updates": [f"confirm-e5-s{s}" for s in SEEDS],
}
DEV_MODELS_FLAT = [n for names in DEV_MODELS.values() for n in names]
X1_LABEL = "fine-tuned E5, 4000 updates"
X1_RUNS = [f"confirm-x1-s{s}" for s in SEEDS]
TEXTLESS = -32768
SCALE = 32000
WUBRG = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16}


def scale_xy(xy: np.ndarray) -> np.ndarray:
    """Centre each axis and scale both by one factor so the longer half-range is SCALE."""
    xy = np.asarray(xy, dtype=np.float64)
    lo, hi = xy.min(0), xy.max(0)
    half = max(float((hi - lo).max()) / 2, 1e-12)
    return (xy - (lo + hi) / 2) * (SCALE / half)


def pack_map(xy: np.ndarray, card_rows: np.ndarray, n_cards: int, colors: list[str]) -> bytes:
    """5 bytes per card in card order: int16 x, int16 y, uint8 WUBRG mask (little-endian).

    xy rows belong to the cards in card_rows; every other card gets x = y = TEXTLESS.
    """
    pts = np.full((n_cards, 2), TEXTLESS, dtype=np.int64)
    pts[card_rows] = np.rint(scale_xy(xy)).astype(np.int64)
    out = bytearray()
    for i in range(n_cards):
        mask = sum(WUBRG[c] for c in colors[i] if c in WUBRG)
        out += struct.pack("<hhB", int(pts[i, 0]), int(pts[i, 1]), mask)
    return bytes(out)


def region_labels(xy: np.ndarray, tag_sets: list[frozenset], labels: dict, k: int = 36,
                  min_support: float = 0.15) -> list[dict]:
    """KMeans regions labelled by the tag with the highest lift (cluster share / overall
    share) among tags held by at least min_support of the cluster; ties -> larger count,
    then label. Clusters with no such tag are dropped."""
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(xy)
    n = len(tag_sets)
    overall = Counter(t for ts in tag_sets for t in ts)
    out = []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        counts = Counter(t for i in members for t in tag_sets[i])
        ok = [(cnt / len(members) / (overall[t] / n), cnt, labels[t]) for t, cnt in counts.items()
              if cnt >= min_support * len(members)]
        if not ok:
            continue
        best = max(ok)
        cx, cy = xy[members].mean(0)
        out.append({"x": round(float(cx)), "y": round(float(cy)), "label": best[2],
                    "n": int(len(members))})
    return out


_PAREN = re.compile(r"\([^()]*\)")
# what may follow a keyword: costs, numbers, "from red"
_KEYWORD_TAIL = re.compile(
    r"^(?:\s*(?:\{[^}]+\}|[0-9x+\-/]+|from [a-z]+(?: and from [a-z]+)*|—))*\s*$")


def keyword_vocabulary(ids: set[str]) -> list[str]:
    """Scryfall keywords of the catalogue cards, longest first. The raw snapshot also holds
    sealed cards, so a record is dropped by oracle_id before anything else is read."""
    path = next((ROOT / "data" / "raw" / "2026-10-16").glob("oracle-cards-*.jsonl.gz"))
    vocab: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("oracle_id") in ids:
                vocab.update(k.lower() for k in rec.get("keywords", []))
    return sorted(vocab, key=len, reverse=True)


def is_keyword_only(oracle_text: str, vocab: list[str]) -> bool:
    """True when, reminder text aside, every line is a list of keyword abilities
    (a "French vanilla" card). No text at all is not keyword-only: those cards have no vector."""
    lines = [_PAREN.sub("", ln).strip() for ln in oracle_text.split("\n")]
    items = [i.strip().lower() for ln in lines for i in re.split(r"[;,]", ln) if i.strip()]

    def keyword_item(item: str) -> bool:
        return any(item.startswith(k) and (len(item) == len(k) or item[len(k)] in " {—-:")
                   and _KEYWORD_TAIL.match(item[len(k):]) for k in vocab)

    return bool(items) and all(keyword_item(i) for i in items)


def keyword_only_indices(release_id: str) -> list[int]:
    """Card indices (release order) of keyword-only cards, from the release's public card files."""
    rel = WEB / "public" / "data" / release_id
    ids = _read(rel / "ids.json")["ids"]
    vocab = keyword_vocabulary(set(ids))
    out, i = [], 0
    for shard in sorted((rel / "cards").glob("*.json")):
        for card in _read(shard):
            text = "\n".join(f.get("oracle_text") or "" for f in card["faces"])
            if is_keyword_only(text, vocab):
                out.append(i)
            i += 1
    if i != len(ids):
        raise ValueError("card files do not match ids.json")
    return out


def write_keyword_only(release_id: str) -> Path:
    """site-<id>/keyword_only.json, for a site folder built before this list existed."""
    if not release_id or Path(release_id).name != release_id:
        raise ValueError("release must be a release ID")
    out = WEB / "public" / "data" / f"site-{release_id}" / "keyword_only.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; site data is never overwritten")
    atomic_write_json(out, {"release_id": release_id, "indices": keyword_only_indices(release_id)})
    return out


def _cpu_encoder(max_length: int = 94):
    import torch

    from . import encoder as enc

    recipe = enc.RECIPES["e5-base-v2"]
    tok, model = enc.load(recipe)
    return lambda texts: enc.encode(model, tok, texts, recipe, max_length, batch_size=64,
                                    dev=torch.device("cpu"))


def pretrained_neighbours(texts: list[str], ti: np.ndarray, n_cards: int, k: int = 10,
                          encode=None) -> list[list]:
    """Top-k pretrained neighbours per card index ([] for textless cards)."""
    v = (encode or _cpu_encoder())(texts)
    idx, sco = topk_neighbours(np.asarray(v, dtype=np.float32), k=k)
    out: list[list] = [[] for _ in range(n_cards)]
    for r, ci in enumerate(ti):
        out[ci] = [[int(ti[j]), round(float(s), 4)] for j, s in zip(idx[r], sco[r], strict=True)]
    return out


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing record: {path}")
    return read_json(path)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _dev_row(names: list[str]) -> dict:
    paths = [RUNS / "eval" / DEV_BENCH / n / "summary.json" for n in names]
    sums = [_read(p)["summary"] for p in paths]
    return {"full": [s["full"]["ndcg@10"] for s in sums],
            "hard_lexical": [s["hard_lexical"]["ndcg@10"] for s in sums],
            "runs": names, "source": [_rel(p) for p in paths]}


def _report(path: Path) -> dict:
    r = _read(path)
    views = ("full", "hard_lexical")
    return {k: r[k] for k in ("decision", "qualifier", "m_contrasts", "deployment", "sealed_gate")
            if k in r} | {
        "contrasts": {name: {**{v: {k: c[v][k] for k in ("point", "lower_adjusted", "ci95")}
                                for v in views}, "passes": c.get("passes")}
                      for name, c in r["contrasts"].items()},
        "source": _rel(path)}


def _final_full(run: str) -> float | None:
    p = RUNS / run / "result.json"
    return _read(p)["evals"][-1]["full"] if p.is_file() else None


TRAIN_KEYS = ("objective", "lr", "batch_pairs", "hard_negatives", "total_updates", "scale",
              "warmup_frac", "max_length", "seed")


def _deployed() -> dict | None:
    """The deployed release's model and its training config, or None without a release."""
    cur = WEB / "public" / "data" / "current.json"
    if not cur.is_file():
        return None
    rid = _read(cur)["release_id"]
    model = _read(WEB / "public" / "data" / rid / "release.json")["model"]
    meta = ARTIFACTS / "releases" / rid / "release_meta.json"
    out = {"release_id": rid, "label": model["label"], "run_id": model.get("run_id"),
           "encoder": {k: v for k, v in _read(meta)["encoder"].items()
                       if k in ("model_id", "prefix", "pooling", "dim", "max_length")}}
    man = RUNS / str(model.get("run_id")) / "manifest.json"
    if man.is_file():
        cfg = _read(man)["config"]
        out["training"] = {k: cfg[k] for k in TRAIN_KEYS if k in cfg} | {"source": _rel(man)}
    return out


def site_metrics(showcase: list[dict] | None = None) -> dict:
    """Every figure the site shows, read from run and evaluation records."""
    dev = {label: _dev_row(names) for label, names in DEV_MODELS.items()}
    if all((RUNS / "eval" / DEV_BENCH / n / "summary.json").is_file() for n in X1_RUNS):
        dev[X1_LABEL] = _dev_row(X1_RUNS)
    x1 = RUNS / "confirmation" / DEV_BENCH / "x1" / "report.json"
    exam = {}
    for d in sorted((RUNS / "exam").glob("*/")):
        recs = sorted(d.glob("*.json"))
        if recs:
            e = _read(recs[-1])
            exam[d.name] = {"passed": e["passed"], "n": e["n"], "source": _rel(recs[-1]),
                            "failed": [{"id": i["id"], "section": i["section"],
                                        "margin": i["margin"]}
                                       for i in e.get("items", []) if not i["passed"]]}
    steps = {}
    for s in SEEDS:
        a, b = _final_full(f"confirm-e5-s{s}"), _final_full(f"confirm-x1-s{s}")
        if a is not None and b is not None:
            steps[str(s)] = {"2000": a, "4000": b}
    ds = _read(DATASET / "dataset.json")["report"]
    return {
        "dev": dev, "dev_benchmark": DEV_BENCH,
        "promotion": {"d18": _report(RUNS / "confirmation" / DEV_BENCH / "report.json"),
                      "x1": _report(x1) if x1.is_file() else None},
        "sealed": _report(RUNS / "sealed" / SEALED_BENCH / "report.json"),
        "exam": exam, "steps": steps,
        "data": {k: ds.get(k) for k in ("n_cards", "split_counts", "tags_functional",
                                        "functional_memberships", "families", "multi_face")},
        "deployed": _deployed(),
        "showcase": showcase or [],
    }


def write_metrics(path: Path | None = None) -> Path:
    path = path or WEB / "src" / "lib" / "generated" / "metrics.json"
    showcase = _read(path).get("showcase", []) if path.is_file() else []
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, site_metrics(showcase))
    return path


def build_site_data(release_id: str, perplexity: float = 30.0) -> dict:
    """Map, regions and pretrained neighbours for one release, into a new folder."""
    if not release_id or Path(release_id).name != release_id:
        raise ValueError("release must be a release ID")
    out = WEB / "public" / "data" / f"site-{release_id}"
    if out.exists():
        raise FileExistsError(f"{out} exists; site data is never overwritten")
    meta = _read(ARTIFACTS / "releases" / release_id / "release_meta.json")
    ds = load_dataset(DATASET)
    if ds["ident"]["dataset_id"] != meta["dataset_id"]:
        raise ValueError("release was built from a different dataset")
    cards = ds["cards"].sort_values("oracle_id").reset_index(drop=True)
    ids = _read(WEB / "public" / "data" / release_id / "ids.json")["ids"]
    if cards["oracle_id"].tolist() != ids:
        raise ValueError("dataset card order does not match the release")
    ti = np.flatnonzero((cards["text_status"] == "ok").to_numpy())
    v = np.load(ARTIFACTS / "releases" / release_id / "vectors.npy", allow_pickle=False)
    if len(v) != len(ti):
        raise ValueError("release vectors do not match the text cards")

    xy = TSNE(perplexity=perplexity, init="pca", random_state=0).fit_transform(
        PCA(n_components=50, random_state=0).fit_transform(v))
    func = ds["tags"].loc[ds["tags"]["functional"]]
    labels = dict(zip(func["tag_id"], func["label"], strict=True))
    mem = ds["memberships"][ds["memberships"]["tag_id"].isin(labels)]
    by_card = mem.groupby("oracle_id")["tag_id"].agg(frozenset).to_dict()
    tag_sets = [by_card.get(o, frozenset()) for o in cards.loc[ti, "oracle_id"]]
    regions = region_labels(scale_xy(xy), tag_sets, labels)
    neigh = pretrained_neighbours(cards.loc[ti, "text"].tolist(), ti, len(cards))

    part = out.with_name(out.name + ".partial")
    (part / "pretrained").mkdir(parents=True, exist_ok=True)
    (part / "map.bin").write_bytes(pack_map(xy, ti, len(cards), cards["color_identity"].tolist()))
    for s in range(0, len(cards), SHARD):
        atomic_write_json(part / "pretrained" / f"{s // SHARD:03d}.json", neigh[s:s + SHARD])
    info = {"release_id": release_id, "n_cards": len(cards), "regions": regions,
            "projection": {"method": "PCA 50 then t-SNE", "perplexity": perplexity,
                           "random_state": 0, "scale": SCALE, "textless": TEXTLESS},
            "region_rule": {"clusters": "KMeans k=36 on the map",
                            "label": "highest-lift Tagger tag",
                            "min_support": 0.15},
            "pretrained": {"model": "intfloat/e5-base-v2", "max_length": 94, "k": 10,
                           "shard_size": SHARD}}
    atomic_write_json(part / "map.json", info)
    atomic_write_json(part / "keyword_only.json",
                      {"release_id": release_id, "indices": keyword_only_indices(release_id)})
    os.replace(part, out)
    return {"site_data": _rel(out), "regions": len(regions), "metrics": _rel(write_metrics())}


def copy_site_data(release_id: str, source_id: str) -> dict:
    """Site data for a release whose card vectors and card order equal an earlier release's
    (e.g. a re-export with --from-release): the map and pretrained neighbours are copied."""
    for rid in (release_id, source_id):
        if not rid or Path(rid).name != rid:
            raise ValueError("releases must be release IDs")
    data = WEB / "public" / "data"
    out, src = data / f"site-{release_id}", data / f"site-{source_id}"
    if out.exists():
        raise FileExistsError(f"{out} exists; site data is never overwritten")
    metas = [_read(ARTIFACTS / "releases" / r / "release_meta.json")
             for r in (release_id, source_id)]
    # same dataset and ids -> same text cards, i.e. the same card behind each vector row
    if metas[0]["dataset_id"] != metas[1]["dataset_id"]:
        raise ValueError("the releases were built from different datasets")
    for r, m in zip((release_id, source_id), metas, strict=True):
        if sha256_file(ARTIFACTS / "releases" / r / "vectors.npy") != m["vectors_sha256"]:
            raise ValueError(f"{r}: vectors.npy does not match its recorded hash")
    if metas[0]["vectors_sha256"] != metas[1]["vectors_sha256"]:
        raise ValueError("the releases have different card vectors")
    if _read(data / release_id / "ids.json") != _read(data / source_id / "ids.json"):
        raise ValueError("the releases have a different card order")
    part = out.with_name(out.name + ".partial")
    if part.exists():
        shutil.rmtree(part)  # leftover from a failed copy; never published
    shutil.copytree(src, part)
    info = {**_read(src / "map.json"), "release_id": release_id, "copied_from": source_id}
    atomic_write_json(part / "map.json", info)
    os.replace(part, out)
    return {"site_data": _rel(out), "copied_from": source_id, "metrics": _rel(write_metrics())}
