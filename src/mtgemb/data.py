"""Corpus, tags, families, split and fit-only statistics (MASTER_PLAN §3-4).

`prepare()` turns a raw snapshot into an immutable processed dataset directory.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .text import FACE_BOUNDARY, VERSION, no_reminder_v1
from .util import DATA, atomic_write_json, log_event, read_json, sha256_file, sha256_json, utcnow

# ---------------------------------------------------------------- corpus policy

EXCLUDED_LAYOUTS = {"token", "double_faced_token", "emblem", "art_series"}
CORPUS_POLICY = (
    "One row per Oracle ID. Include iff 'paper' in games, vintage legality is not 'not_legal' "
    "(legal/restricted/banned = a tournament card), and layout not in "
    f"{sorted(EXCLUDED_LAYOUTS)}. Excludes tokens, emblems, art cards, digital-only cards, "
    "unreleased previews, planes/schemes/vanguards and non-tournament novelty cards."
)
WUBRG = "WUBRG"


def corpus_reason(card: dict) -> str | None:
    """None if the card is included, else the exclusion reason."""
    if card.get("layout") in EXCLUDED_LAYOUTS:
        return f"layout:{card['layout']}"
    if "paper" not in card.get("games", []):
        return "not_paper"
    if card.get("legalities", {}).get("vintage", "not_legal") == "not_legal":
        return "not_tournament_legal"
    return None


def face_list(card: dict) -> list[dict]:
    """Ordered faces. Single-faced cards are one face built from the top-level fields."""
    keys = ("name", "mana_cost", "type_line", "oracle_text", "power", "toughness", "loyalty",
            "defense")
    faces = card.get("card_faces")
    if faces:
        return [{k: f.get(k) for k in keys} for f in faces]
    return [{k: card.get(k) for k in keys}]


def card_row(card: dict) -> dict:
    faces = face_list(card)
    tr = no_reminder_v1([f["oracle_text"] or "" for f in faces])
    front_type = faces[0]["type_line"] or card.get("type_line") or ""
    img = card.get("image_uris") or (card.get("card_faces") or [{}])[0].get("image_uris") or {}
    return {
        "oracle_id": card["oracle_id"],
        "name": card["name"],
        "layout": card["layout"],
        "faces": json.dumps(faces, ensure_ascii=False),
        "n_faces": len(faces),
        "type_line": card.get("type_line") or " // ".join(f["type_line"] or "" for f in faces),
        "mana_cost": card.get("mana_cost") or " // ".join(f["mana_cost"] or "" for f in faces),
        "color_identity": "".join(c for c in WUBRG if c in card.get("color_identity", [])),
        "mv": float(card.get("cmc", 0.0)),
        # frozen ordered-face type policy: creature status of the first face
        "is_creature": "Creature" in front_type,
        "set": card.get("set"),
        "released_at": card.get("released_at"),
        "image": img.get("normal"),
        "scryfall_uri": card.get("scryfall_uri"),
        "text_raw": tr["raw"],
        "text": tr["text"],
        "text_status": tr["status"],
        "text_reason": tr["reason"],
        "paren_spans": tr.get("spans", 0),
        "paren_nested": tr.get("nested", 0),
        "paren_whole_lines": tr.get("whole_line_spans", 0),
    }


def check_raw_cards(cards: list[dict]) -> None:
    """Reject duplicate/missing Oracle IDs and malformed core fields instead of guessing."""
    seen: dict[str, dict] = {}
    for c in cards:
        oid = c.get("oracle_id")
        if not oid:
            raise ValueError(f"card without oracle_id: {c.get('name')}")
        if oid in seen and seen[oid] != c:
            raise ValueError(f"conflicting duplicate oracle_id {oid}")
        seen[oid] = c
        if not c.get("name") or "layout" not in c:
            raise ValueError(f"malformed card {oid}")
        if c.get("card_faces") is not None and not c["card_faces"]:
            raise ValueError(f"empty card_faces on {oid}")


# ---------------------------------------------------------------- tag policy

TAG_POLICY_VERSION = "functional_tags_v1"
_DENY_EXACT = {
    "alliteration", "namesake spell", "portmanteau", "anagram", "vanity card", "meme",
    "fun ruling", "virtual legendary", "potentially black border", "digital-only mechanics",
    "paper-compatible", "digital to paper", "digital replacement", "unprinted token",
    "aesthetic counter", "counter fuel-aesthetic", "misnomer", "tongue twister", "roman numeral",
    "wordle", "multi character card", "legendary team-up", "alt-commander", "mirrored knight",
    "invitational card", "one-off", "fulfilled futureshift", "mechanical foreshadow",
    "playtest forecast", "references keyword", "notorious templating", "sneaky-self-trigger",
    "day zero errata", "intervening if clause", "untracked indefinite effect",
    "unnoted tracked information", "noted tracked information", "faux targeting", "front-card",
    "weaker in singleton formats", "pair-commander", "creature type ship", "creature type hero",
    "mana source (type)", "counts as a type", "mix-and-match", "keyword soup",
    "card game reference", "typal-mutant-ninja-turtle", "unstable killbot",
    "unstable secret base", "nemesis-mega-cycle", "5c set mechanic commander", "color break",
    "offcolor ability", "cheaper than mv", "more expensive than mv", "phyrexian token",
    "token without a card", "hate-set-mechanic", "eponymous", "eponymous planeswalker",
    "bear with set's mechanic", "wind drake with set's mechanic",
    "staple with set's mechanic",
}
_DENY_PATTERNS = [
    (re.compile(r"^(cycle|supercycle|vertical-cycle)\b"), "cycle"),
    (re.compile(r"^(type errata|type addition|keyword errata|token errata|deprecated )"), "errata"),
    (re.compile(r"^unique[ -]"), "printing-unique"),
    (re.compile(r"^un[- ]"), "un-set"),
    (re.compile(r"^(dnd |doctor who|40k |inscryption|fallout |marvel |preexisting dnd)"),
     "flavour-reference"),
    (re.compile(r"storyline-in-cards$"), "flavour-reference"),
    (re.compile(r"with set'?s? mechanic$"), "set-relative"),
    (re.compile(r"(^| )name$"), "name"),
    (re.compile(r"^shares name with"), "name"),
]


_KEEP_EXACT = {"tutors by name", "change name"}  # functional despite matching a deny pattern


def tag_reason(label: str) -> str | None:
    """None if the tag is retained as functional, else the exclusion category."""
    lab = label.strip().lower()
    if lab in _KEEP_EXACT:
        return None
    if lab in _DENY_EXACT:
        return "metadata"
    for pat, why in _DENY_PATTERNS:
        if pat.search(lab):
            return why
    return None


# ---------------------------------------------------------------- families

def _self_names(name: str, faces: list[dict]) -> list[str]:
    names = {name} | {f["name"] for f in faces if f.get("name")}
    for n in list(names):
        if ", " in n:
            names.add(n.split(", ")[0])
    return sorted((n for n in names if len(n) >= 3), key=len, reverse=True)


def name_normalised(text: str, name: str, faces: list[dict]) -> str:
    """Grouping key only (never an encoder input): own names -> CARDNAME."""
    for n in _self_names(name, faces):
        text = re.sub(rf"(?<!\w){re.escape(n)}(?!\w)", "CARDNAME", text)
    return text


class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if rb < ra:
            ra, rb = rb, ra
        self.p[rb] = ra
        return True


def build_families(cards: pd.DataFrame, meld_links: list[tuple[str, str]]) -> tuple[dict, list]:
    """Connected components over exact-text, self-name-template and meld edges.

    Empty texts never link (otherwise every vanilla creature would join one component).
    Returns ({oracle_id: family_id}, edges[(a, b, reason)]). family_id = min oracle_id.
    """
    ids = sorted(cards["oracle_id"])
    dsu = DSU(ids)
    edges: list[tuple[str, str, str]] = []
    ok = cards[cards["text_status"] == "ok"]

    def link_groups(keys: pd.Series, reason: str):
        groups: dict[str, list[str]] = defaultdict(list)
        for oid, k in zip(ok["oracle_id"], keys, strict=True):
            groups[k].append(oid)
        for members in groups.values():
            members = sorted(members)
            for m in members[1:]:
                if dsu.union(members[0], m):
                    edges.append((members[0], m, reason))

    link_groups(ok["text"], "exact_text")
    templ = [
        name_normalised(t, n, json.loads(f))
        for t, n, f in zip(ok["text"], ok["name"], ok["faces"], strict=True)
    ]
    link_groups(pd.Series(templ, index=ok.index), "self_name_template")
    present = set(ids)
    for a, b in meld_links:
        if a in present and b in present and dsu.union(a, b):
            edges.append((min(a, b), max(a, b), "meld"))
    return {i: dsu.find(i) for i in ids}, edges


# ---------------------------------------------------------------- split

SPLIT_SEED = "mtgemb-split-v1"
SPLIT_TARGETS = (("fit", 0.8), ("dev", 0.9), ("sealed", 1.0))


def split_of(family_id: str, seed: str = SPLIT_SEED) -> str:
    """Whole components go to one partition by a seeded hash of the component key."""
    u = int(sha256_json([seed, family_id])[:16], 16) / 2**64
    for name, edge in SPLIT_TARGETS:
        if u < edge:
            return name
    return SPLIT_TARGETS[-1][0]


# ---------------------------------------------------------------- tag weights

def idf_weights(memberships: pd.DataFrame, fit_ids: set[str]) -> dict[str, float]:
    """w(t) = 1 + ln((N + 1) / (df(t) + 1)) over fit cards; only tags present in fit."""
    n = len(fit_ids)
    m = memberships[memberships["oracle_id"].isin(fit_ids)]
    df = m.groupby("tag_id")["oracle_id"].nunique()
    return {t: 1.0 + math.log((n + 1) / (d + 1)) for t, d in df.items()}


# ---------------------------------------------------------------- prepare

def load_raw(raw_dir: Path) -> tuple[list[dict], list[dict], dict]:
    manifest = read_json(raw_dir / "manifest.json")
    for f in manifest["files"].values():
        if sha256_file(raw_dir / f["path"]) != f["sha256"]:
            raise ValueError(f"raw file hash mismatch: {f['path']}")

    def jl(key):
        with gzip.open(raw_dir / manifest["files"][key]["path"], "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    return jl("oracle_cards"), jl("oracle_tags"), manifest


def prepare(raw_dir: Path, out_dir: Path | None = None) -> dict:
    raw_dir = Path(raw_dir)
    raw_cards, raw_tags, manifest = load_raw(raw_dir)
    check_raw_cards(raw_cards)
    out_dir = Path(out_dir or DATA / "processed" / f"{manifest['snapshot']}_{VERSION}")
    if (out_dir / "dataset.json").exists():
        raise FileExistsError(f"{out_dir} already finalised; a change needs a new version")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_event("data", "STARTED", step="prepare", raw=str(raw_dir), out=str(out_dir))

    excluded = Counter()
    rows, meld_links = [], []
    for c in raw_cards:
        r = corpus_reason(c)
        if r:
            excluded[r] += 1
            continue
        rows.append(card_row(c))
        for p in c.get("all_parts") or []:
            if p.get("component") in ("meld_part", "meld_result"):
                # all_parts gives printing ids; map via name below
                meld_links.append((c["oracle_id"], p["name"]))
    cards = pd.DataFrame(rows).sort_values("oracle_id").reset_index(drop=True)
    if cards["name"].duplicated().any():
        dups = cards.loc[cards["name"].duplicated(), "name"].tolist()
        raise ValueError(f"duplicate names: {dups}")
    by_name = dict(zip(cards["name"], cards["oracle_id"], strict=True))
    meld_links = [(a, by_name[n]) for a, n in meld_links if n in by_name and by_name[n] != a]

    # tags
    tag_rows, mem_rows, orphans = [], [], 0
    present = set(cards["oracle_id"])
    for t in raw_tags:
        if t.get("type") != "oracle":
            continue
        reason = tag_reason(t["label"])
        tag_rows.append({
            "tag_id": t["id"], "label": t["label"], "slug": t["slug"],
            "description": t.get("description"),
            "parent_ids": json.dumps(t.get("parent_ids") or []),
            "functional": reason is None, "exclusion": reason, "n_taggings_raw": len(t["taggings"]),
        })
        for g in t["taggings"]:
            if g.get("oracle_id") is None:
                orphans += 1
                continue
            if g["oracle_id"] in present:
                mem_rows.append({"oracle_id": g["oracle_id"], "tag_id": t["id"],
                                 "weight": g.get("weight", "median")})
    tags = pd.DataFrame(tag_rows).sort_values("tag_id").reset_index(drop=True)
    mem = pd.DataFrame(mem_rows).drop_duplicates(["oracle_id", "tag_id"])
    mem = mem[mem["tag_id"].isin(set(tags.loc[tags["functional"], "tag_id"]))]
    mem = mem.sort_values(["oracle_id", "tag_id"]).reset_index(drop=True)

    # families and split
    fam, edges = build_families(cards, meld_links)
    cards["family_id"] = cards["oracle_id"].map(fam)
    cards["split"] = cards["family_id"].map(split_of)
    cards["n_tags"] = cards["oracle_id"].map(mem.groupby("oracle_id").size()).fillna(0).astype(int)

    fit_ids = set(cards.loc[cards["split"] == "fit", "oracle_id"])
    w = idf_weights(mem, fit_ids)
    df_fit = mem[mem["oracle_id"].isin(fit_ids)].groupby("tag_id").size()
    tags["df_fit"] = tags["tag_id"].map(df_fit).fillna(0).astype(int)
    tags["idf_fit"] = tags["tag_id"].map(w)

    cards.to_parquet(out_dir / "cards.parquet", index=False)
    tags.to_parquet(out_dir / "tags.parquet", index=False)
    mem.to_parquet(out_dir / "memberships.parquet", index=False)
    pd.DataFrame(edges, columns=["a", "b", "reason"]).to_parquet(
        out_dir / "family_edges.parquet", index=False)

    fam_sizes = cards.groupby("family_id").size()
    split_counts = cards["split"].value_counts().to_dict()
    report = {
        "excluded_by_policy": dict(excluded),
        "n_cards": len(cards),
        "text_status": cards["text_status"].value_counts().to_dict(),
        "text_reason": cards["text_reason"].value_counts().to_dict(),
        "cards_with_parentheticals": int((cards["paren_spans"] > 0).sum()),
        "cards_with_nested": int((cards["paren_nested"] > 0).sum()),
        "cards_with_whole_line_parenthetical": int((cards["paren_whole_lines"] > 0).sum()),
        "multi_face": int((cards["n_faces"] > 1).sum()),
        "tags_total": len(tags),
        "tags_functional": int(tags["functional"].sum()),
        "tags_excluded_by_reason": tags["exclusion"].value_counts().to_dict(),
        "functional_memberships": len(mem),
        "orphan_taggings": orphans,
        "cards_with_functional_tag": int((cards["n_tags"] > 0).sum()),
        "families": int(fam_sizes.size),
        "family_edges_by_reason": Counter(e[2] for e in edges),
        "largest_families": fam_sizes.sort_values(ascending=False).head(10).to_dict(),
        "split_counts": split_counts,
        "split_fraction": {k: v / len(cards) for k, v in split_counts.items()},
        "fit_tags_represented": int((tags["df_fit"] > 0).sum()),
    }
    ident = {
        "snapshot": manifest["snapshot"],
        "raw_manifest_sha256": sha256_file(raw_dir / "manifest.json"),
        "raw_files": manifest["files"],
        "terms": manifest["terms"],
        "corpus_policy": CORPUS_POLICY,
        "representation": VERSION,
        "face_boundary": FACE_BOUNDARY,
        "tag_policy": TAG_POLICY_VERSION,
        "split_seed": SPLIT_SEED,
        "split_targets": SPLIT_TARGETS,
        "split_algorithm": "component key = min oracle_id; u = sha256(json([seed,key]))[:16]/2^64",
        "files": {p.name: sha256_file(p) for p in sorted(out_dir.glob("*.parquet"))},
        "created_utc": utcnow(),
        "report": report,
    }
    ident["dataset_id"] = sha256_json({k: ident[k] for k in ("raw_manifest_sha256", "files")})[:16]
    atomic_write_json(out_dir / "dataset.json", ident)
    log_event("data", "COMPLETED", step="prepare", out=str(out_dir), dataset_id=ident["dataset_id"])
    return ident


def load_dataset(ds_dir: Path) -> dict:
    """Load a finalised dataset, verifying file hashes."""
    ds_dir = Path(ds_dir)
    ident = read_json(ds_dir / "dataset.json")
    for name, h in ident["files"].items():
        if sha256_file(ds_dir / name) != h:
            raise ValueError(f"dataset file changed: {name}")
    out = {"ident": ident, "dir": ds_dir}
    for name in ident["files"]:
        out[name.removesuffix(".parquet")] = pd.read_parquet(ds_dir / name)
    return out


def tag_sets(mem: pd.DataFrame, tag_ids: set[str] | None = None) -> dict[str, frozenset[str]]:
    m = mem if tag_ids is None else mem[mem["tag_id"].isin(tag_ids)]
    return {k: frozenset(v) for k, v in m.groupby("oracle_id")["tag_id"]}

