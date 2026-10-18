"""Trap exam (idea 9; the idea is credited to Delvefall).

Each item is an anchor (a whole card, or one of its ability lines), a true match and a
near-identical trap. A model passes an item when it scores the match above the trap.

The worksheet `exam/traps.md` is the source of truth. Items are proposed by rule-based
mining or from user reports; only the user turns `proposed` into `accepted` or `rejected`.
Only accepted items are scored. Exam cards are never used for training (`exam_cards`).
Sealed cards are never loaded, so they cannot appear.

Mined sections (one-line cards only, so every tag on a card belongs to its single line):
- rider: the trap shares the anchor's opening trigger/condition clause but no functional
  tag; the match has a different opener and shares the most idf weight with the anchor.
- mechanism: the trap's text differs from the anchor's by exactly one flipped word
  (tap/untap, enters/dies, ...); the match shares the most idf weight with the anchor.
- parameters: the match has the anchor's text up to numbers; the trap is the lexically
  nearest card that shares no functional tag.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .abilities import split_abilities

EXAM = Path("exam/traps.md")
STATUSES = ("proposed", "accepted", "rejected")
SECTIONS = ("user", "rider", "mechanism", "parameters")
PER_SECTION = 20

OPENER = re.compile(r"^(At the beginning of [^,]+,|Whenever [^,]+,|When [^,]+,|If [^,]+,"
                    r"|As long as [^,]+,)")
FLIPS = [("tap", "untap"), ("tapped", "untapped"), ("enters", "dies"), ("gain", "lose"),
         ("gains", "loses"), ("attacks", "blocks"), ("attacking", "blocking"),
         ("hand", "graveyard"), ("draw", "discard"), ("you", "opponent"),
         ("creature", "land"), ("+1/+1", "-1/-1"), ("top", "bottom")]
FLIP_OF = {a: b for a, b in FLIPS} | {b: a for a, b in FLIPS}
_NUM = re.compile(r"\b(\d+|X|one|two|three|four|five|six|seven|eight|nine|ten)\b")
_TOKEN = re.compile(r"[^\s.,;:]+")


def opener(line: str) -> str | None:
    m = OPENER.match(line)
    return m.group(0) if m else None


def mask_numbers(text: str) -> str:
    return _NUM.sub("#", text)


# ---------------------------------------------------------------- worksheet

_HEAD = re.compile(r"^## (T\d{3}) · (\w+) · (\w+)\s*$")
_ROLE = re.compile(r"^(anchor|match|trap): ([0-9a-f-]{36})(?: line (\d+))?\b")


def parse(md: str) -> list[dict]:
    """Items from the worksheet; malformed entries raise rather than being skipped."""
    items, cur = [], None
    for n, raw in enumerate(md.splitlines(), 1):
        line = raw.strip()
        if h := _HEAD.match(line):
            if h.group(2) not in SECTIONS or h.group(3) not in STATUSES:
                raise ValueError(f"line {n}: unknown section or status: {line}")
            cur = {"id": h.group(1), "section": h.group(2), "status": h.group(3)}
            items.append(cur)
        elif cur is not None and (r := _ROLE.match(line)):
            cur[r.group(1)] = r.group(2)
            if r.group(1) == "anchor":
                cur["line"] = int(r.group(3)) if r.group(3) else None
    for it in items:
        if not all(k in it for k in ("anchor", "match", "trap")):
            raise ValueError(f"{it['id']}: needs anchor, match and trap")
    if len({it["id"] for it in items}) != len(items):
        raise ValueError("duplicate item id")
    return items


def render(item: dict, cards) -> str:
    """One worksheet entry. `cards` is indexed by oracle_id (name, split, text)."""
    def role(r):
        c = cards.loc[item[r]]
        ln = f" line {item['line']}" if r == "anchor" and item.get("line") is not None else ""
        text = (split_abilities(c["text"])[item["line"] - 1] if ln
                else c["text"].replace("\n", " / "))
        return f"{r}: {item[r]}{ln} | {c['name']} ({c['split']}) | {text}"
    lines = [f"## {item['id']} · {item['section']} · {item['status']}",
             role("anchor"), role("match"), role("trap")]
    if item.get("note"):
        lines.append(f"note: {item['note']}")
    return "\n".join(lines) + "\n"


def exam_cards(path: Path = EXAM) -> set[str]:
    """Every card named in the worksheet, whatever its status: excluded from training."""
    if not path.exists():
        return set()
    return {it[r] for it in parse(path.read_text("utf-8")) for r in ("anchor", "match", "trap")}


# ---------------------------------------------------------------- mining

def mine(ds: dict, k: int = PER_SECTION) -> list[dict]:
    """Proposed items from fit+dev one-line cards. Dev anchors first, then fit."""
    from .data import tag_sets
    from .lexical import fit_vectorizer

    cards = ds["cards"]
    ok = cards[cards["text_status"] == "ok"].sort_values("oracle_id")
    one = ok[ok["text"].map(lambda t: len(split_abilities(t)) == 1)].reset_index(drop=True)
    tags = ds["tags"].set_index("tag_id")
    func = set(tags.index[tags["functional"]])
    ts = tag_sets(ds["memberships"], func)
    idf = tags["idf_fit"].dropna().to_dict()
    ids, texts = one["oracle_id"].tolist(), one["text"].tolist()
    tset = [ts.get(o, frozenset()) for o in ids]
    tagged = np.array([bool(t) for t in tset])
    fam = one["family_id"].to_numpy()
    order = np.argsort(one["split"].map({"dev": 0, "fit": 1}).to_numpy(), kind="stable")
    vec = fit_vectorizer(ok.loc[ok["split"] == "fit", "text"].tolist())
    lex = vec.transform(texts)

    def overlap(i, j):  # weighted Jaccard: shared idf mass over union idf mass
        union = sum(idf.get(t, 0.0) for t in tset[i] | tset[j])
        return sum(idf.get(t, 0.0) for t in tset[i] & tset[j]) / union if union else 0.0

    def best_match(i, allowed):
        cand = [j for j in allowed if j != i and fam[j] != fam[i] and tagged[j]]
        sc = [(overlap(i, j), -j) for j in cand]
        if not sc:
            return None
        s, j = max(sc)
        return -j if s > 0 and len(tset[i] & tset[-j]) >= 2 else None

    def lexical_nearest(i, cand):
        cand = [j for j in cand if fam[j] != fam[i]]
        if not cand:
            return None
        sims = (lex[cand] @ lex[i].T).toarray().ravel()
        return cand[int(np.argmax(sims))]

    out = {s: [] for s in SECTIONS[1:]}
    used: set[int] = set()
    n = len(one)
    opens = [opener(t) for t in texts]
    masked = [mask_numbers(t) for t in texts]
    by_open: dict[str, list[int]] = {}
    for j, o in enumerate(opens):
        if o:
            by_open.setdefault(o, []).append(j)
    by_mask: dict[str, list[int]] = {}
    for j, m in enumerate(masked):
        by_mask.setdefault(m, []).append(j)
    toks = [tuple(_TOKEN.findall(t.lower())) for t in texts]
    by_toks: dict[tuple, list[int]] = {}
    for j, t in enumerate(toks):
        by_toks.setdefault(t, []).append(j)
    seen_open, seen_flip = set(), set()
    for i in order:
        if not tagged[i] or i in used:
            continue
        o = opens[i]
        if len(out["rider"]) < k and o and o not in seen_open:
            traps = [j for j in by_open[o] if tagged[j] and not (tset[i] & tset[j])]
            trap = lexical_nearest(i, traps)
            match = best_match(i, [j for j in range(n) if opens[j] != o])
            if trap is not None and match is not None and not {trap, match} & used:
                out["rider"].append((i, match, trap))
                used |= {i, match, trap}
                seen_open.add(o)
                continue
        if len(out["mechanism"]) < k:
            ti = toks[i]
            flips = [(j, (w, FLIP_OF[w])) for p, w in enumerate(ti) if w in FLIP_OF
                     for j in by_toks.get(ti[:p] + (FLIP_OF[w],) + ti[p + 1:], [])]
            flips = [(j, f) for j, f in flips if frozenset(f) not in seen_flip and j not in used]
            if flips:
                trap, f = flips[0]
                match = best_match(i, [j for j in range(n) if j not in (i, trap)])
                if match is not None and match not in used:
                    out["mechanism"].append((i, match, trap))
                    used |= {i, match, trap}
                    seen_flip.add(frozenset(f))
                    continue
        if len(out["parameters"]) < k:
            twins = [j for j in by_mask[masked[i]]
                     if j != i and fam[j] != fam[i] and texts[j] != texts[i] and j not in used]
            if twins:
                traps = [j for j in range(n) if tagged[j] and not (tset[i] & tset[j])
                         and j not in used]
                trap = lexical_nearest(i, traps)
                if trap is not None:
                    out["parameters"].append((i, twins[0], trap))
                    used |= {i, twins[0], trap}
    items = []
    for s in SECTIONS[1:]:
        for i, m, t in out[s]:
            items.append({"section": s, "status": "proposed", "anchor": ids[i], "line": None,
                          "match": ids[m], "trap": ids[t]})
    return items


HEADER = """# Trap exam

Idea credited to Delvefall (github.com/Ballchinian/Delvefall); items here are our own.
Rule: **same mechanism, flexible parameters.** Numbers, colours and riders are forgivable;
a different or flipped mechanism is not.

Each item: an anchor (a whole card, or `line N` of it), a true **match** and a **trap**.
A model passes when it scores the match above the trap. Only the user changes `proposed` to
`accepted` or `rejected` in an item's heading. Only accepted items are scored. Never edit
ids or card ids; to change an item, reject it and add a new one with the next id.
Every card named here is excluded from training. Sealed cards never appear.

"""


def init_worksheet(ds: dict, user_items: list[dict], path: Path = EXAM) -> list[dict]:
    """Write the first worksheet (user reports first, then mined items). Never overwrites."""
    if path.exists():
        raise FileExistsError(f"{path} exists; append new items instead of rebuilding")
    items = [*user_items, *mine(ds)]
    cards = ds["cards"].set_index("oracle_id")
    for n, it in enumerate(items, 1):
        it["id"] = f"T{n:03d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "\n".join(render(it, cards) for it in items), "utf-8")
    assert [it["id"] for it in parse(path.read_text("utf-8"))] == [it["id"] for it in items]
    return items


# ---------------------------------------------------------------- scoring

def score(items: list[dict], cards, encode) -> dict:
    """Pass rates over accepted items. `encode(texts) -> unit vectors`.

    Whole-card anchors compare whole texts; a line anchor scores each candidate by its
    best-matching ability line (the per-ability view's rule).
    """
    acc = [it for it in items if it["status"] == "accepted"]
    rows = []
    for it in acc:
        a = cards.loc[it["anchor"], "text"]
        if it.get("line") is not None:
            q = encode([split_abilities(a)[it["line"] - 1]])[0]
            s = [float(np.max(encode(split_abilities(cards.loc[it[r], "text"])) @ q))
                 for r in ("match", "trap")]
        else:
            v = encode([a, cards.loc[it["match"], "text"], cards.loc[it["trap"], "text"]])
            s = [float(v[1] @ v[0]), float(v[2] @ v[0])]
        rows.append({"id": it["id"], "section": it["section"], "match": s[0], "trap": s[1],
                     "margin": s[0] - s[1], "passed": s[0] > s[1]})
    by = {}
    for sec in SECTIONS:
        r = [x for x in rows if x["section"] == sec]
        if r:
            by[sec] = {"n": len(r), "passed": sum(x["passed"] for x in r),
                       "mean_margin": float(np.mean([x["margin"] for x in r]))}
    return {"n": len(rows), "passed": sum(x["passed"] for x in rows), "sections": by,
            "items": rows}


def exam_score(ds_dir, name: str, backbone: str, max_length: int,
               model_path: str | None = None, path: Path = EXAM) -> dict:
    """Score one encoder (on CPU) on the accepted items; one new record per call."""
    import time

    import torch

    from . import encoder as enc
    from .data import load_dataset
    from .util import RUNS, atomic_write_json, git_identity, log_event, sha256_file, utcnow

    items = parse(path.read_text("utf-8"))
    cards = load_dataset(ds_dir)["cards"].set_index("oracle_id")
    recipe = enc.RECIPES[backbone]
    tok, model = enc.load(recipe, model_path)
    cpu = torch.device("cpu")
    model.to(cpu)
    out = score(items, cards, lambda t: enc.encode(model, tok, t, recipe, max_length, dev=cpu))
    out.update(name=name, backbone=backbone, max_length=max_length, model_path=model_path,
               worksheet_sha256=sha256_file(path), n_proposed=sum(
                   it["status"] == "proposed" for it in items),
               code=git_identity(), created_utc=utcnow())
    rec = RUNS / "exam" / name / f"{out['created_utc'].replace(':', '')}-{time.time_ns()}.json"
    rec.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rec, out)
    log_event("eval", "EXAM_SCORED", name=name, passed=out["passed"], n=out["n"],
              record=str(rec))
    return out
