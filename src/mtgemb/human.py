"""Optional human pilot (MASTER_PLAN §9, IMPLEMENTATION_PLAN P11; ).

A blinded pool of development anchors with candidates pooled from frozen systems. The
annotation tool sees only pool.json (cards, items, presentation order); key.json maps each
item to the systems and ranks that proposed it and is read only by the analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .abilities import split_abilities
from .baselines import _lexical_vectors
from .evaluate import load_benchmark
from .util import RUNS, atomic_write_json, read_json, sha256_json, utcnow

SEED = 20261018
N_SCORED = 50  # per stratum (single-ability, multi-ability)
N_CAL = 5  # per stratum; calibration anchors are shown first and never scored
DEPTH = 3  # top-k pooled from each card-level system
ABILITY_DEPTH = 3  # top-k pooled from each ability list of a multi-ability anchor
N_RANDOM = 1
REPEAT_FRAC = 0.10
CARD_SYSTEMS = {"lexical": None, "pretrained-e5": "pretrained-e5-base-v2",
                "trained-bge-s1": "confirm-bge-s1", "finetuned-e5-s1": "confirm-e5-s1"}
RUBRIC = {"3": "same main effect with aligned decisive features",
          "2": "related main effect with a meaningful difference",
          "1": "incidental shared effect",
          "0": "unrelated or complementary only",
          "unknown": "insufficient confidence"}


def sample_anchors(multi: list[str], single: list[str], rng) -> dict[str, list[str]]:
    """Seeded stratified draw: calibration then scored anchors per stratum, disjoint."""
    out = {"calibration": [], "scored": []}
    for ids in (single, multi):
        perm = [ids[i] for i in rng.permutation(len(ids))[:N_CAL + N_SCORED]]
        out["calibration"] += perm[:N_CAL]
        out["scored"] += perm[N_CAL:]
    for k in out:
        out[k] = [out[k][i] for i in rng.permutation(len(out[k]))]  # interleave strata
    return out


def top_k(v: np.ndarray, a: int, k: int) -> list[int]:
    """Full-view ranking for anchor row a: cosine desc, row (= Oracle ID) asc, self out."""
    s = v @ v[a].T
    s = np.asarray(s.todense()).ravel() if hasattr(s, "todense") else np.asarray(s).ravel()
    s[a] = -np.inf
    return np.argsort(-s, kind="stable")[:k].tolist()


def assemble(anchors: dict[str, list[str]], proposals: dict[str, dict[str, list[str]]],
             universe: list[str], rng) -> tuple[list[dict], dict[str, dict]]:
    """Items in presentation order plus the blinding key.

    proposals[anchor][source] = ranked candidate ids. Each anchor's unique candidates plus
    N_RANDOM random universe cards are shuffled. Repeats of REPEAT_FRAC of scored items
    are inserted after the first and second half of the scored anchors, each drawn from
    the half just completed.
    """
    items, key = [], {}

    def block(anchor: str, phase: str) -> list[dict]:
        src: dict[str, list] = {}
        for s, ids in proposals[anchor].items():
            for r, c in enumerate(ids, 1):
                src.setdefault(c, []).append([s, r])
        pool = set(src) | {anchor}
        rest = [u for u in universe if u not in pool]
        for c in (rest[i] for i in rng.choice(len(rest), N_RANDOM, replace=False)):
            src[c] = [["random", None]]
        cands = sorted(src)
        out = []
        for c in (cands[i] for i in rng.permutation(len(cands))):
            iid = f"{phase[0]}{len(items) + len(out):05d}"
            key[iid] = {"sources": src[c]}
            out.append({"item_id": iid, "phase": phase, "anchor": anchor, "candidate": c})
        return out

    for a in anchors["calibration"]:
        items += block(a, "calibration")
    half = len(anchors["scored"]) // 2
    for part in (anchors["scored"][:half], anchors["scored"][half:]):
        start = len(items)
        for a in part:
            items += block(a, "scored")
        done = items[start:]
        n = max(1, round(REPEAT_FRAC * len(done)))
        for i in sorted(rng.choice(len(done), n, replace=False)):
            src = done[i]
            iid = f"r{len(items):05d}"
            key[iid] = {"repeat_of": src["item_id"], "sources": key[src["item_id"]]["sources"]}
            items.append({**src, "item_id": iid, "phase": "repeat"})
    return items, key


def _card(c, n_abilities: int) -> dict:
    faces = json.loads(c["faces"])
    return {"name": c["name"], "type_line": c["type_line"], "image": c["image"] or None,
            "faces": [{k: f.get(k) or "" for k in ("name", "mana_cost", "type_line",
                                                   "oracle_text")} for f in faces],
            "abilities": split_abilities(c["text"]) if n_abilities >= 2 else []}


def build_pool(ds_dir, release_dir) -> dict:
    """Write runs/labels/<pool_id>/{pool.json,key.json} once; returns the pool summary."""
    bm, ds = load_benchmark(ds_dir)  # development benchmark; sealed rows never loaded
    bid = bm.spec["benchmark_id"]
    cards = ds["cards"].set_index("oracle_id")
    release_dir = Path(release_dir)
    rel = read_json(release_dir / "release.json")
    if rel["schema"] != "mtgemb-release-v3":
        raise ValueError("ability lists need a v3 release")
    cat = read_json(release_dir / "ids.json")["ids"]
    n_ab = {o: len(split_abilities(cards.at[o, "text"])) for o in bm.spec["anchors"]}
    multi = [o for o in bm.spec["anchors"] if n_ab[o] >= 2]
    single = [o for o in bm.spec["anchors"] if n_ab[o] == 1]
    rng = np.random.default_rng(SEED)
    anchors = sample_anchors(multi, single, rng)

    pos = {o: i for i, o in enumerate(bm.ids)}
    vecs = {s: (_lexical_vectors(bm, ds) if n is None else
                np.load(RUNS / "eval" / bid / n / "vectors.npy", allow_pickle=False))
            for s, n in CARD_SYSTEMS.items()}
    proposals: dict[str, dict[str, list[str]]] = {}
    shard_cache: dict[int, list] = {}
    for a in anchors["calibration"] + anchors["scored"]:
        p = {s: [bm.ids[j] for j in top_k(v, pos[a], DEPTH)] for s, v in vecs.items()}
        if n_ab[a] >= 2:
            ci = cat.index(a)
            sh = ci // rel["shard_size"]
            if sh not in shard_cache:
                shard_cache[sh] = read_json(release_dir / "abilities" / f"{sh:03d}.json")
            entries = shard_cache[sh][ci % rel["shard_size"]]
            if len(entries) != n_ab[a]:
                raise ValueError(f"{a}: release abilities disagree with the dataset text")
            for t, e in enumerate(entries):
                p[f"ability-{t}"] = [cat[j] for j, _ in e["neighbours"][:ABILITY_DEPTH]]
        proposals[a] = p
    items, key = assemble(anchors, proposals, bm.ids, rng)

    used = {x for it in items for x in (it["anchor"], it["candidate"])}
    drawn = anchors["calibration"] + anchors["scored"]
    pool = {"design": {"decision": "", "seed": SEED, "n_scored_per_stratum": N_SCORED,
                       "n_calibration_per_stratum": N_CAL, "depth": DEPTH,
                       "ability_depth": ABILITY_DEPTH, "n_random": N_RANDOM,
                       "repeat_frac": REPEAT_FRAC, "rubric": RUBRIC,
                       "direction": "judge the candidate as a similar card FOR the anchor"},
            "benchmark_id": bid, "release_id": rel["release_id"],
            "cards": {o: _card(cards.loc[o], n_ab[o] if o in drawn else 0)
                      for o in sorted(used)},
            "items": items}
    pool_id = sha256_json(pool)[:12]
    out = RUNS / "labels" / pool_id
    if (out / "pool.json").exists():
        raise FileExistsError(out)
    pool["pool_id"], pool["created_utc"] = pool_id, utcnow()
    atomic_write_json(out / "key.json", {"pool_id": pool_id, "systems": CARD_SYSTEMS,
                                         "items": key, "anchors": anchors,
                                         "strata": {o: ("multi" if n_ab[o] >= 2 else "single")
                                                    for o in drawn}})
    atomic_write_json(out / "pool.json", pool)
    counts = {ph: sum(it["phase"] == ph for it in items)
              for ph in ("calibration", "scored", "repeat")}
    return {"pool_id": pool_id, "dir": str(out), "items": counts}
