"""Trap exam (idea 9, credited to Delvefall).

Each item is an anchor (a whole card, or one of its ability lines), a true match and a
near-identical trap. A model passes an item when it scores the match above the trap.

The worksheet `exam/traps.md` is the source of truth. Items are proposed by rule-based
mining or from user reports; only a human reviewer turns `proposed` into `accepted` or `rejected`.
Only accepted items are scored. Exam card families are kept out of new training inputs;
fit exam cards remain in the existing card pairs (`exam_cards`).
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
import time
from pathlib import Path

import numpy as np

from .abilities import split_abilities
from .util import ROOT, RUNS, atomic_write_json, git_identity, log_event, sha256_file, utcnow

EXAM = ROOT / "exam" / "traps.md"  # absolute: a wrong cwd must not empty the exclusions
STATUSES = ("proposed", "accepted", "rejected")
SECTIONS = ("user", "rider", "mechanism", "parameters")

OPENER = re.compile(r"^(At the beginning of [^,]+,|Whenever [^,]+,|When [^,]+,|If [^,]+,"
                    r"|As long as [^,]+,)")
FLIPS = [("tap", "untap"), ("tapped", "untapped"), ("enters", "dies"), ("gain", "lose"),
         ("gains", "loses"), ("attacks", "blocks"), ("attacking", "blocking"),
         ("hand", "graveyard"), ("draw", "discard"), ("you", "opponent"),
         ("creature", "land"), ("+1/+1", "-1/-1"), ("top", "bottom")]
_NUM = re.compile(r"\b(\d+|X|one|two|three|four|five|six|seven|eight|nine|ten)\b")


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
        if line.startswith("##"):  # an item heading; a malformed one must not be skipped
            if not (h := _HEAD.match(line)):
                raise ValueError(f"line {n}: malformed item heading: {line}")
            if h.group(2) not in SECTIONS or h.group(3) not in STATUSES:
                raise ValueError(f"line {n}: unknown section or status: {line}")
            cur = {"id": h.group(1), "section": h.group(2), "status": h.group(3)}
            items.append(cur)
        elif cur is not None and (r := _ROLE.match(line)):
            if r.group(1) in cur:
                raise ValueError(f"line {n}: {cur['id']} repeats its {r.group(1)}")
            cur[r.group(1)] = r.group(2)
            if r.group(1) == "anchor":
                cur["line"] = int(r.group(3)) if r.group(3) else None
                if cur["line"] == 0:
                    raise ValueError(f"line {n}: ability lines are numbered from 1")
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
    """Every card named in the worksheet, whatever its status.

    Their families are kept out of every NEW training input (aux rows, augmentations).
    Fit exam cards stay in the existing card pairs, shared by the comparator and the new
    runs, so the exam is not held out for them; dev-only items are reported separately.
    """
    if not path.exists():
        return set()
    return {it[r] for it in parse(path.read_text("utf-8")) for r in ("anchor", "match", "trap")}


HEADER = """# Trap exam

Idea credited to Delvefall (github.com/Ballchinian/Delvefall); items here are our own.
Rule: **same mechanism, flexible parameters.** Numbers, colours and riders are forgivable;
a different or flipped mechanism is not.

Each item: an anchor (a whole card, or `line N` of it), a true **match** and a **trap**.
A model passes when it scores the match above the trap. Only a human reviewer changes `proposed` to
`accepted` or `rejected` in an item's heading. Only accepted items are scored. Never edit
ids or card ids; to change an item, reject it and add a new one with the next id.
Every card named here (and its family) is kept out of new training inputs.
Sealed cards never appear.

"""


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
            lines = split_abilities(a)
            if not 1 <= it["line"] <= len(lines):
                raise ValueError(f"{it['id']}: anchor has no line {it['line']}")
            q = encode([lines[it["line"] - 1]])[0]
            s = [float(np.max(encode(split_abilities(cards.loc[it[r], "text"])) @ q))
                 for r in ("match", "trap")]
        else:
            v = encode([a, cards.loc[it["match"], "text"], cards.loc[it["trap"], "text"]])
            s = [float(v[1] @ v[0]), float(v[2] @ v[0])]
        rows.append({"id": it["id"], "section": it["section"],
                     "item": [it["anchor"], it.get("line"), it["match"], it["trap"]],
                     "match": s[0], "trap": s[1], "margin": s[0] - s[1], "passed": s[0] > s[1],
                     "dev_only": all(cards.loc[it[r], "split"] == "dev"
                                     for r in ("anchor", "match", "trap"))})
    by = {}
    for sec in SECTIONS:
        r = [x for x in rows if x["section"] == sec]
        if r:
            by[sec] = {"n": len(r), "passed": sum(x["passed"] for x in r),
                       "mean_margin": float(np.mean([x["margin"] for x in r]))}
    dev = [x for x in rows if x["dev_only"]]  # never trained on by any run
    return {"n": len(rows), "passed": sum(x["passed"] for x in rows), "sections": by,
            "dev_only": {"n": len(dev), "passed": sum(x["passed"] for x in dev)},
            "items": rows}


def guard(old: dict, new: dict) -> dict:
    """Regression guard (idea 10, Delvefall): items both records scored, by identical item.

    An item edited between the two records is not compared (it is a different item).
    """
    before = {(r["id"], tuple(r["item"])): r["passed"] for r in old["items"]}
    after = {(r["id"], tuple(r["item"])): r["passed"] for r in new["items"]}
    both = sorted(before.keys() & after.keys())
    return {"n_compared": len(both),
            "regressed": [k[0] for k in both if before[k] and not after[k]],
            "fixed": [k[0] for k in both if after[k] and not before[k]],
            "only_old": sorted(k[0] for k in before.keys() - after.keys()),
            "only_new": sorted(k[0] for k in after.keys() - before.keys())}


def exam_score(ds_dir, name: str, backbone: str, max_length: int,
               model_path: str | None = None, path: Path = EXAM,
               text_version: str = "no_reminder_v1") -> dict:
    """Score one encoder (on CPU) on the accepted items; one new record per call."""
    import torch

    from . import encoder as enc
    from .data import load_dataset
    from .selfname import apply_text_version

    items = parse(path.read_text("utf-8"))
    cards = apply_text_version(load_dataset(ds_dir)["cards"], text_version).set_index("oracle_id")
    recipe = enc.RECIPES[backbone]
    tok, model = enc.load(recipe, model_path)
    cpu = torch.device("cpu")
    model.to(cpu)
    out = score(items, cards, lambda t: enc.encode(model, tok, t, recipe, max_length, dev=cpu))
    out.update(name=name, backbone=backbone, max_length=max_length, model_path=model_path,
               text_version=text_version,
               worksheet_sha256=sha256_file(path), n_proposed=sum(
                   it["status"] == "proposed" for it in items),
               code=git_identity(), created_utc=utcnow())
    rec = RUNS / "exam" / name / f"{out['created_utc'].replace(':', '')}-{time.time_ns()}.json"
    rec.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rec, out)
    log_event("eval", "EXAM_SCORED", name=name, passed=out["passed"], n=out["n"],
              text_version=text_version, record=str(rec))
    return out
