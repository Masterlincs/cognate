"""Contrastive fine-tuning with exact resume.

The batch schedule is a pure function of (config, data), so the sampler position is just
the update counter. Checkpoints hold model, optimiser, scheduler, RNG states and counters,
and are immutable generations (ckpt-<update>.pt) published by atomically rewriting a single
pointer file (last.json) that carries the hash resume validates.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pydantic import BaseModel, ConfigDict, model_validator

from . import encoder as enc
from . import exam, linetag, selfname, synth
from .data import load_dataset
from .evaluate import Benchmark, evaluate, load_benchmark
from .util import (
    RUNS,
    atomic_write_bytes,
    atomic_write_json,
    git_identity,
    log_event,
    read_json,
    sha256_file,
    sha256_json,
    utcnow,
)


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    slot: str  # e.g. "initial-1", "smoke-3", "confirm-7"
    hypothesis: str = ""
    comparator: str = ""
    backbone: str = "e5-base-v2"
    max_length: int = 94
    lr: float = 3e-5
    scale: float = 20.0
    batch_pairs: int = 64
    hard_negatives: int = 1  # per row, drawn from the anchor's lexical pool
    neg_band: tuple[int, int] = (1, 30)  # inclusive lexical ranks
    weight_decay: float = 0.01
    total_updates: int = 1000
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    seed: int = 0  # model-side randomness (dropout)
    data_seed: int = 0  # batch schedule
    objective: Literal["masked_infonce", "multi_positive"] = "masked_infonce"
    pair_weighting: Literal["none", "specificity", "tag_idf"] = "none"
    pairs_version: Literal["pairs_v1", "pairs_v2"] = "pairs_v1"  # v2 is effect-only
    batch_mode: Literal["random", "tag_grouped"] = "random"
    text_version: Literal["no_reminder_v1", "no_reminder_v2"] = "no_reminder_v1"
    # ideas 1-2 (Delvefall): a line<->tag stream beside the unchanged card schedule
    aux: Literal["none", "line_tag"] = "none"
    aux_weight: float = 1.0
    aux_batch: int = 32
    synth: bool = False
    synth_weight: float = 1.0
    synth_batch: int = 32
    precision: Literal["fp32", "bf16"] = "fp32"
    grad_checkpointing: bool = False
    max_skip_rate: float = 0.01
    guardrail_hard_lexical: float | None = None  # pretrained hard-lexical nDCG
    # guarded_best: pilot rule. final: confirmation runs, no selection
    selection: Literal["guarded_best", "final"] = "guarded_best"
    dataset_dir: str
    stop_after: int | None = None  # engineering: stop (resumably) after this update

    @model_validator(mode="after")
    def _confirmation_uses_final(self):
        if self.slot.startswith("confirm") and self.selection != "final":
            raise ValueError("confirmation runs must use selection='final'")
        return self


# ---------------------------------------------------------------- data


class TrainData:
    """Fit cards with text, positive pairs, eligible-tag incidence and negative pools."""

    def __init__(self, ds_dir: str | Path, text_version: str = "no_reminder_v1",
                 pairs_version: str = "pairs_v1"):
        ds = load_dataset(ds_dir)
        cards = selfname.apply_text_version(ds["cards"], text_version)
        fit_ok = (cards["split"] == "fit") & (cards["text_status"] == "ok")
        self.fit_texts_changed = int((cards.loc[fit_ok, "text"] !=
                                     ds["cards"].loc[fit_ok, "text"]).sum())
        self.text_version, self.pairs_version = text_version, pairs_version
        ds["cards"] = cards
        pdir = Path(ds_dir) / pairs_version
        self.pairs_rec = read_json(pdir / "pairs.json")
        for name, h in self.pairs_rec["files"].items():
            if sha256_file(pdir / name) != h:
                raise ValueError(f"pairs file changed: {name}")
        fit = cards[fit_ok].sort_values("oracle_id").reset_index(drop=True)
        self.ids = fit["oracle_id"].tolist()
        self.texts = fit["text"].tolist()
        pos = {o: i for i, o in enumerate(self.ids)}
        self.family = pd.factorize(fit["family_id"])[0]
        pairs = pd.read_parquet(pdir / "pairs.parquet")
        self.a = pairs["a"].map(pos).to_numpy()
        self.b = pairs["b"].map(pos).to_numpy()
        self.pair_tag = pd.factorize(pairs["tag_id"])[0]
        self.spec_weight = pairs["spec_weight"].to_numpy(dtype=np.float32)
        self.pair_tag_id = pairs["tag_id"].to_numpy()
        elig = pd.read_parquet(pdir / "pair_tags.parquet")["tag_id"]
        mem = ds["memberships"]
        m = mem[mem["oracle_id"].isin(pos) & mem["tag_id"].isin(set(elig))]
        tcol = {t: j for j, t in enumerate(sorted(set(elig)))}
        self.E = sp.csr_matrix(
            (np.ones(len(m), dtype=np.float32),
             (m["oracle_id"].map(pos).to_numpy(), m["tag_id"].map(tcol).to_numpy())),
            shape=(len(self.ids), len(tcol)))
        neg = pd.read_parquet(pdir / "negatives.parquet")
        neg = neg[neg["anchor"].isin(pos) & neg["negative"].isin(pos)]
        neg = neg.assign(ai=neg["anchor"].map(pos), ni=neg["negative"].map(pos))
        neg = neg.sort_values(["ai", "rank"])
        ai, ni, rk = (neg[c].to_numpy() for c in ("ai", "ni", "rank"))
        starts = np.flatnonzero(np.r_[True, ai[1:] != ai[:-1]])
        ends = np.r_[starts[1:], len(ai)]
        self.neg_pool = {int(ai[s]): ni[s:e] for s, e in zip(starts, ends, strict=True)}
        self.neg_rank = {int(ai[s]): rk[s:e] for s, e in zip(starts, ends, strict=True)}
        self.identity = {"dataset_id": ds["ident"]["dataset_id"],
                         "pairs_id": self.pairs_rec["pairs_id"]}
        self._ds, self._elig = ds, set(elig)

    def tag_weights(self) -> np.ndarray:
        """Each pair weighted by the fit IDF of the tag that made it (not the
        mean over all shared tags, as slot 7 did), mean-one, clipped like spec_weight."""
        return tag_idf_weights(self.pair_tag_id, self._ds["tags"])

    def line_tag_rows(self) -> dict:
        """Aux rows; every trap-exam card's family is kept out (exam.exam_cards)."""
        return linetag.build(self._ds, self._elig, exam.exam_cards())

    def synth_rows(self) -> dict:
        """Rule-made rows, with whole exam families excluded."""
        rows = synth.build(self._ds, self._elig, exam.exam_cards())
        pos = {o: i for i, o in enumerate(self.ids)}
        rows["card_rows"] = np.array([pos[o] for o in rows["anchor_ids"]])
        return rows

    def known_pos(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Shares at least one pair-eligible tag."""
        return (self.E[rows] @ self.E[cols].T).toarray() > 0


def tag_idf_weights(pair_tag_id, tags: pd.DataFrame, clip=(0.5, 2.0)) -> np.ndarray:
    idf = tags.set_index("tag_id")["idf_fit"].reindex(pair_tag_id).to_numpy(dtype=np.float64)
    if not np.isfinite(idf).all():
        raise ValueError("a pair's tag has no fit IDF")
    return np.clip(idf / idf.mean(), *clip).astype(np.float32)


# ---------------------------------------------------------------- schedule


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(int(sha256_json(list(key))[:15], 16))


def build_schedule(cfg: TrainConfig, data: TrainData) -> list[dict]:
    """Deterministic list of batches for all `total_updates`.

    Each batch: anchors, positives (row-aligned), hard negatives, pair indices.
    No card or family appears twice in a batch except a designated pair's two members
    (which are different families by construction). Pairs that collide are deferred.
    """
    out: list[dict] = []
    n_pairs = len(data.a)
    lo, hi = cfg.neg_band
    epoch = 0
    by_tag: dict[int, list[int]] = {}
    if cfg.batch_mode == "tag_grouped":
        for i, t in enumerate(data.pair_tag):
            by_tag.setdefault(int(t), []).append(i)
    while len(out) < cfg.total_updates:
        rng = _rng("schedule", cfg.data_seed, epoch)
        perm = rng.permutation(n_pairs)
        flip = rng.random(n_pairs) < 0.5  # orientation: which member is the anchor
        used_pair = np.zeros(n_pairs, dtype=bool)
        tag_cursor: dict[int, int] = {}
        if cfg.batch_mode == "tag_grouped":
            tag_order = {t: rng.permutation(v).tolist() for t, v in by_tag.items()}
        deferred: list[int] = []
        queue = iter(perm.tolist())
        exhausted = False
        while len(out) < cfg.total_updates and not exhausted:
            cards: set[int] = set()
            fams: set[int] = set()
            rows: list[int] = []

            def try_add(p: int, cards=cards, fams=fams, rows=rows, used_pair=used_pair) -> bool:
                x, y = int(data.a[p]), int(data.b[p])
                fx, fy = int(data.family[x]), int(data.family[y])
                if x in cards or y in cards or fx in fams or fy in fams:
                    return False
                cards.update((x, y))
                fams.update((fx, fy))
                rows.append(p)
                used_pair[p] = True
                return True

            keep = []
            for p in deferred:
                if len(rows) < cfg.batch_pairs and not used_pair[p] and try_add(p):
                    continue
                if not used_pair[p]:
                    keep.append(p)
            deferred = keep
            while len(rows) < cfg.batch_pairs:
                p = next(queue, None)
                if p is None:
                    exhausted = True
                    break
                if used_pair[p]:
                    continue
                if not try_add(p):
                    deferred.append(p)
                    continue
                if cfg.batch_mode == "tag_grouped" and len(rows) < cfg.batch_pairs:
                    # add one more pair from the same tag to create in-batch extra positives
                    t = int(data.pair_tag[p])
                    lst = tag_order[t]
                    c = tag_cursor.get(t, 0)
                    while c < len(lst):
                        q = lst[c]
                        c += 1
                        if not used_pair[q] and try_add(q):
                            break
                    tag_cursor[t] = c
            if len(rows) < cfg.batch_pairs:
                if not any(bb["epoch"] == epoch for bb in out[-1:]):
                    raise ValueError("an epoch produced no full batch; batch_pairs too large")
                break  # epoch tail: drop the short batch, start a new epoch
            rows_a = np.array(rows)
            f = flip[rows_a]
            anc = np.where(f, data.b[rows_a], data.a[rows_a])
            pos_ = np.where(f, data.a[rows_a], data.b[rows_a])
            negs: list[int] = []
            if cfg.hard_negatives:
                nrng = _rng("negs", cfg.data_seed, epoch, len(out))
                for x in anc:
                    pool = data.neg_pool.get(int(x))
                    if pool is None:
                        continue
                    ranks = data.neg_rank[int(x)]
                    cand = pool[(ranks >= lo) & (ranks <= hi)]
                    cand = cand[nrng.permutation(len(cand))]
                    got = 0
                    for n in cand:
                        n = int(n)
                        if n in cards or int(data.family[n]) in fams:
                            continue
                        cards.add(n)
                        fams.add(int(data.family[n]))
                        negs.append(n)
                        got += 1
                        if got == cfg.hard_negatives:
                            break
            out.append({"pairs": rows_a, "anchors": anc, "positives": pos_,
                        "negatives": np.array(negs, dtype=np.int64), "epoch": epoch})
        epoch += 1
    return out


def schedule_hash(batches: list[dict]) -> str:
    return sha256_json([[b["anchors"].tolist(), b["positives"].tolist(),
                         b["negatives"].tolist()] for b in batches])


# ---------------------------------------------------------------- loss


def contrastive_loss(q: torch.Tensor, c: torch.Tensor, known: torch.Tensor,
                     excluded: torch.Tensor, scale: float, objective: str,
                     weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """One direction of the batch loss.

    q: (R, d) normalised rows; c: (C, d) normalised candidates, C >= R; the designated
    positive of row i is column i. known[i, j]: j is another known positive of row i.
    excluded[i, j]: j is the same card or the same family as row i (never a negative and
    never an extra positive). Neither mask may be set on the diagonal.

    masked_infonce: -log softmax at the designated positive over {designated} U allowed.
    multi_positive: -log (sum of softmax mass on {designated} U known) over the same set
    plus known positives. Rows with no allowed negative are skipped and counted.
    """
    r, cn = q.shape[0], c.shape[0]
    logits = scale * (q @ c.T)
    diag = torch.zeros(r, cn, dtype=torch.bool, device=q.device)
    diag[torch.arange(r), torch.arange(r)] = True
    allowed = ~(known | excluded | diag)
    if objective == "masked_infonce":
        keep = allowed | diag
        pos = diag
    elif objective == "multi_positive":
        pos = diag | (known & ~excluded)
        keep = allowed | pos
    else:
        raise ValueError(objective)
    neg_inf = torch.finfo(logits.dtype).min
    denom = torch.logsumexp(logits.masked_fill(~keep, neg_inf), dim=1)
    numer = torch.logsumexp(logits.masked_fill(~pos, neg_inf), dim=1)
    eff_neg = allowed.sum(1)
    valid = eff_neg > 0
    row_loss = denom - numer
    w = torch.ones(r, device=q.device) if weights is None else weights
    w = w * valid
    loss = (row_loss * w).sum() / w.sum().clamp(min=1e-12)
    stats = {"rows": r, "skipped": int((~valid).sum()),
             "eff_neg_mean": float(eff_neg.float().mean()),
             "masked_frac": float((known | excluded).float().sum() / max(1, r * (cn - 1))),
             "multi_pos_rows": int(((known & ~excluded).sum(1) > 0).sum())}
    return loss, stats


def batch_masks(data: TrainData, rows: np.ndarray, cols: np.ndarray):
    """known/excluded masks for row cards vs candidate cards, diagonal cleared."""
    known = data.known_pos(rows, cols)
    excluded = (rows[:, None] == cols[None, :]) | (
        data.family[rows][:, None] == data.family[cols][None, :])
    k = min(len(rows), len(cols))
    known[np.arange(k), np.arange(k)] = False
    excluded[np.arange(k), np.arange(k)] = False
    return torch.from_numpy(known), torch.from_numpy(excluded)


# ---------------------------------------------------------------- checkpointing


def _rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.xpu.is_available():
        st["xpu"] = torch.xpu.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "xpu" in st and torch.xpu.is_available():
        torch.xpu.set_rng_state_all(st["xpu"])


def save_checkpoint(out: Path, state: dict, identity: dict) -> Path:
    """Write a new generation (temp, fsync, rename), then atomically repoint last.json.

    An interruption at any point leaves last.json naming a complete, matching generation.
    Superseded generations are removed only after the pointer has moved.
    """
    out = Path(out)
    path = out / f"ckpt-{state['update']:07d}.pt"
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    with open(tmp, "r+b") as f:  # Windows needs a writable handle to fsync
        os.fsync(f.fileno())
    os.replace(tmp, path)
    atomic_write_json(out / "last.json",
                      {"file": path.name, "sha256": sha256_file(path), "identity": identity,
                       "saved_utc": utcnow(), "update": state["update"]})
    for old in out.glob("ckpt-*.pt"):
        if old.name != path.name:
            old.unlink()
    return path


def load_checkpoint(out: Path, identity: dict) -> dict | None:
    """Latest published generation, or None. Rejects changed bytes, config or inputs."""
    ptr = Path(out) / "last.json"
    if not ptr.exists():
        return None
    side = read_json(ptr)
    path = Path(out) / side["file"]
    if sha256_file(path) != side["sha256"]:
        raise ValueError(f"checkpoint {path} is corrupted (hash mismatch)")
    if side["identity"] != identity:
        diff = {k for k in set(side["identity"]) | set(identity)
                if side["identity"].get(k) != identity.get(k)}
        raise ValueError(f"checkpoint identity mismatch on {sorted(diff)}; fork a new run")
    # weights_only=False: the state holds RNG tuples/numpy arrays; the file's sha256 was
    # verified above against the sidecar this code wrote, so it is our own pickle
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- device lock


class DeviceLock:
    """One GPU workload at a time: a lock file holding the owner's pid."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or RUNS / "gpu.lock")

    def __enter__(self):
        if self.path.exists():
            pid = int(json.loads(self.path.read_text())["pid"])
            if _pid_alive(pid) and pid != os.getpid():
                raise RuntimeError(f"device locked by live pid {pid}: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"pid": os.getpid(), "since": utcnow()}))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------- training


def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "bias" in n or "norm" in n.lower() else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(total: int, warmup_frac: float):
    warm = max(1, int(round(total * warmup_frac)))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total - step) / max(1, total - warm))

    return f


def select_checkpoint(evals: list[dict], guardrail: float | None, tol: float = 0.005):
    """Registered rule: best full nDCG among checkpoints with hard-lexical >= guardrail-tol;
    ties -> higher hard-lexical, then earlier update. None qualifies -> (final, failed)."""
    trained = [e for e in evals if e["update"] > 0]
    ok = [e for e in trained if guardrail is None or e["hard_lexical"] >= guardrail - tol]
    if not ok:
        return max(trained, key=lambda e: e["update"]), False
    best = sorted(ok, key=lambda e: (-e["full"], -e["hard_lexical"], e["update"]))[0]
    return best, True


SELECTION_RULES = {
    "guarded_best": "best dev full nDCG with hard-lexical >= pretrained-0.005; "
                    "ties: higher hard-lexical, earlier update",
    "final": "fixed final update; no checkpoint selection on development results",
}


def finalise_selection(evals: list[dict], cfg: TrainConfig, tol: float = 0.005):
    """(selected eval, guardrail passed) under the run's registered selection mode."""
    if cfg.selection == "final":
        sel = max(evals, key=lambda e: e["update"])
        g = cfg.guardrail_hard_lexical
        return sel, g is None or sel["hard_lexical"] >= g - tol
    return select_checkpoint(evals, cfg.guardrail_hard_lexical, tol)


_CODE_FILES = ("train.py", "encoder.py", "evaluate.py", "text.py", "linetag.py",
               "synth.py", "augment.py", "selfname.py")


def code_identity() -> str:
    """Hash of the source that defines training semantics; resume refuses a change."""
    here = Path(__file__).parent
    return sha256_json({f: sha256_file(here / f) for f in _CODE_FILES})[:16]


def train(cfg: TrainConfig, bench: dict | None = None, *, dev: torch.device | None = None,
          data: TrainData | None = None) -> dict:
    """Train (or resume) one run. Returns the run record."""
    dev = dev or enc.device()
    out = RUNS / cfg.run_id
    out.mkdir(parents=True, exist_ok=True)
    data = data or TrainData(cfg.dataset_dir, cfg.text_version, cfg.pairs_version)
    if getattr(data, "text_version", "no_reminder_v1") != cfg.text_version:
        raise ValueError("training data text_version does not match config")
    if getattr(data, "pairs_version", "pairs_v1") != cfg.pairs_version:
        raise ValueError("training data pairs_version does not match config")
    recipe = enc.RECIPES[cfg.backbone]
    if bench and cfg.selection == "guarded_best" and cfg.guardrail_hard_lexical is None:
        raise ValueError("guarded_best selection needs guardrail_hard_lexical "
                         "(the backbone's pretrained hard-lexical nDCG)")
    cfg_hash = sha256_json(cfg.model_dump(mode="json", exclude={"stop_after"}))
    aux = data.line_tag_rows() if cfg.aux == "line_tag" else None
    synth_rows = data.synth_rows() if cfg.synth else None
    identity = {"config": cfg_hash, **data.identity, "recipe": recipe, "code": code_identity(),
                "benchmark_id": bench["benchmark_id"] if bench else None}
    if aux is not None:
        identity["aux_rows"] = aux["hash"]
    if synth_rows is not None:
        identity["synth_rows"] = synth_rows["hash"]
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        man = read_json(manifest_path)
        if man["identity"] != identity:
            raise ValueError(f"run {cfg.run_id} exists with a different identity; fork it")
    else:
        man = {"run_id": cfg.run_id, "slot": cfg.slot, "config": cfg.model_dump(mode="json"),
               "identity": identity, "code": git_identity(), "created_utc": utcnow(),
               "device": str(dev), "torch": torch.__version__,
               "hypothesis": cfg.hypothesis, "comparator": cfg.comparator,
               "selection_rule": SELECTION_RULES[cfg.selection]}
        atomic_write_json(manifest_path, man)

    batches = build_schedule(cfg, data)
    card_hash = sched_hash = schedule_hash(batches)
    if aux is not None:  # the card schedule is untouched; its hash is logged separately
        aux_sched = linetag.schedule(aux, cfg.aux_batch, cfg.total_updates, cfg.data_seed)
        for b, rows in zip(batches, aux_sched, strict=True):
            b["aux"] = rows
        sched_hash = sha256_json([card_hash, [r.tolist() for r in aux_sched]])
    if synth_rows is not None:
        synth_sched = synth.schedule(synth_rows, cfg.synth_batch, cfg.total_updates, cfg.data_seed)
        for b, rows in zip(batches, synth_sched, strict=True):
            b["synth"] = rows
        sched_hash = sha256_json([sched_hash, [r.tolist() for r in synth_sched]])
    with DeviceLock():
        return _train_locked(cfg, bench, dev, data, recipe, identity, out, batches, sched_hash,
                             aux, card_hash, synth_rows)


def aux_loss(model, tok, recipe, max_length, aux: dict, rows: np.ndarray, scale: float,
             dev, autocast) -> tuple[torch.Tensor, dict]:
    """Symmetric masked InfoNCE between one-line texts and their tag texts (ideas 1-2).

    Row i pairs line i with tag i; another in-batch tag the line also carries is masked,
    never a negative. Lines and tags are unique within a batch, so nothing else is excluded.
    """
    texts = [aux["lines"][i] for i in aux["row_line"][rows]]
    texts += [aux["tag_texts"][j] for j in aux["row_tag"][rows]]
    tb = {k: v.to(dev) for k, v in enc.tokenize(tok, texts, recipe, max_length).items()}
    with autocast:
        v = enc.embed(model, tb, recipe).float()
    n = len(rows)
    lines, tags = v[:n], v[n:]
    known = torch.from_numpy(linetag.masks(aux, rows)).to(dev)
    none = torch.zeros_like(known)
    a1, s1 = contrastive_loss(lines, tags, known, none, scale, "masked_infonce")
    a2, s2 = contrastive_loss(tags, lines, known.T.contiguous(), none, scale, "masked_infonce")
    return 0.5 * (a1 + a2), {"skipped": s1["skipped"] + s2["skipped"],
                             "masked_frac": s1["masked_frac"]}


def synth_loss(model, tok, recipe, max_length, synth_rows: dict, rows: np.ndarray, scale: float,
               dev, autocast, shared: np.ndarray | None = None) -> tuple[torch.Tensor, dict]:
    """Anchor->positive InfoNCE; all negatives compete.

    Another row's positive is masked (never a negative) when its anchor is in the same
    family or, via `shared` (n x n), shares a pair-eligible tag, as in the card loss.
    """
    texts = [synth_rows["anchors"][i] for i in rows]
    texts += [synth_rows["positives"][i] for i in rows]
    texts += [synth_rows["negatives"][i] for i in rows if synth_rows["negatives"][i] is not None]
    tb = {k: v.to(dev) for k, v in enc.tokenize(tok, texts, recipe, max_length).items()}
    with autocast:
        v = enc.embed(model, tb, recipe).float()
    n = len(rows)
    families = np.array(synth_rows["families"])[rows]
    known = torch.zeros(n, len(texts) - n, dtype=torch.bool, device=dev)
    same = families[:, None] == families[None, :]
    if shared is not None:
        same = same | shared
    known[:, :n] = torch.from_numpy(same).to(dev)
    known[torch.arange(n), torch.arange(n)] = False
    return contrastive_loss(v[:n], v[n:], known, torch.zeros_like(known), scale, "masked_infonce")


def _train_locked(cfg, bench, dev, data, recipe, identity, out, batches, sched_hash,
                  aux, card_hash, synth_rows):
    tag_w = data.tag_weights() if cfg.pair_weighting == "tag_idf" else None
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tok, model = enc.load(recipe)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(dev)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg.total_updates, cfg.warmup_frac))
    start, evals, best = 0, [], None
    skipped_total = rows_total = 0
    st = load_checkpoint(out, identity)
    if st is not None:
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])
        _set_rng_state(st["rng"])
        start, evals, best = st["update"], st["evals"], st.get("best")
        skipped_total, rows_total = st["skipped_total"], st["rows_total"]
        if st["schedule_hash"] != sched_hash:
            raise ValueError("batch schedule changed since the checkpoint")
        _truncate_metrics(out / "metrics.jsonl", start)
        log_event("train", "RESUMED", run_id=cfg.run_id, update=start)
    else:
        _truncate_metrics(out / "metrics.jsonl", 0)  # restart before any checkpoint
        log_event("train", "STARTED", run_id=cfg.run_id, slot=cfg.slot,
                  schedule_hash=sched_hash, card_schedule_hash=card_hash,
                  aux_rows=aux["hash"] if aux else None,
                  synth_rows=synth_rows["hash"] if synth_rows else None,
                  synth_count=len(synth_rows["anchors"]) if synth_rows else 0,
                  text_version=cfg.text_version,
                  fit_texts_changed=getattr(data, "fit_texts_changed", 0),
                  config=cfg.model_dump(mode="json"))
    bm = None
    if bench:
        bm, ds = load_benchmark(cfg.dataset_dir)
        if bm.spec["benchmark_id"] != bench["benchmark_id"]:
            raise ValueError("benchmark changed since the run was configured")
        # Keep v1 lexical masks/labels frozen, as in selfname.diagnose; swap encoder inputs only.
        cards = selfname.apply_text_version(ds["cards"], cfg.text_version)
        bm.texts = cards.set_index("oracle_id").loc[bm.ids, "text"].tolist()
    metrics_f = open(out / "metrics.jsonl", "a", encoding="utf-8")
    autocast = torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                              enabled=cfg.precision == "bf16")
    t_last = time.time()
    done = start
    try:
        for upd in range(start, cfg.total_updates):
            if cfg.stop_after is not None and upd >= cfg.stop_after:
                break
            b = batches[upd]
            model.train()
            anc, pos_, neg = b["anchors"], b["positives"], b["negatives"]
            cand = np.concatenate([pos_, neg])
            texts = [data.texts[i] for i in np.concatenate([anc, cand])]
            tb = {k: v.to(dev) for k, v in enc.tokenize(tok, texts, recipe,
                                                        cfg.max_length).items()}
            with autocast:
                v = enc.embed(model, tb, recipe).float()
            q, c = v[: len(anc)], v[len(anc):]
            k1, x1 = batch_masks(data, anc, cand)
            k2, x2 = batch_masks(data, pos_, anc)
            w = None
            if cfg.pair_weighting == "specificity":
                w = torch.from_numpy(data.spec_weight[b["pairs"]]).to(dev)
            elif tag_w is not None:
                w = torch.from_numpy(tag_w[b["pairs"]]).to(dev)
            l1, s1 = contrastive_loss(q, c, k1.to(dev), x1.to(dev), cfg.scale, cfg.objective, w)
            l2, s2 = contrastive_loss(c[: len(anc)], q, k2.to(dev), x2.to(dev), cfg.scale,
                                      cfg.objective, w)
            loss = 0.5 * (l1 + l2)
            if aux is not None:
                la, sa = aux_loss(model, tok, recipe, cfg.max_length, aux, b["aux"], cfg.scale,
                                  dev, autocast)
                loss = loss + cfg.aux_weight * la
            if synth_rows is not None:
                cr = synth_rows["card_rows"][b["synth"]]
                ls, ss = synth_loss(model, tok, recipe, cfg.max_length, synth_rows, b["synth"],
                                    cfg.scale, dev, autocast, data.known_pos(cr, cr))
                loss = loss + cfg.synth_weight * ls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            loss_v = float(loss.detach())
            if not math.isfinite(gn) or not math.isfinite(loss_v):
                raise FloatingPointError(f"non-finite loss/grad at update {upd}")
            opt.step()
            sch.step()
            skipped_total += s1["skipped"] + s2["skipped"]
            rows_total += s1["rows"] + s2["rows"]
            if aux is not None:  # aux rows count toward the same skip guard
                skipped_total += sa["skipped"]
                rows_total += 2 * len(b["aux"])
            if synth_rows is not None:
                skipped_total += ss["skipped"]
                rows_total += ss["rows"]
            with torch.no_grad():
                off = (q @ q.T).fill_diagonal_(float("nan"))
                off_mean = float(torch.nanmean(off))
            rec = {"update": upd + 1, "loss": loss_v, "grad_norm": gn,
                   "lr": sch.get_last_lr()[0], "skipped": s1["skipped"] + s2["skipped"],
                   "eff_neg_a2b": s1["eff_neg_mean"], "eff_neg_b2a": s2["eff_neg_mean"],
                   "masked_frac_a2b": s1["masked_frac"], "multi_pos_rows": s1["multi_pos_rows"],
                   "n_neg": int(len(neg)), "offdiag_cos": off_mean,
                   **({"aux_loss": float(la.detach()), "aux_skipped": sa["skipped"],
                       "aux_masked_frac": sa["masked_frac"]} if aux is not None else {}),
                   **({"synth_loss": float(ls.detach()), "synth_skipped": ss["skipped"]}
                      if synth_rows is not None else {}),
                   "sec": round(time.time() - t_last, 3),
                   "peak_mem_gb": round(torch.xpu.max_memory_allocated() / 1e9, 2)
                   if dev.type == "xpu" else None}
            t_last = time.time()
            metrics_f.write(json.dumps(rec) + "\n")
            done = upd + 1
            if rows_total and skipped_total / rows_total > cfg.max_skip_rate and done >= 50:
                raise RuntimeError(f"skipped-row rate {skipped_total / rows_total:.3f} exceeds "
                                   f"{cfg.max_skip_rate}")
            is_eval = bm is not None and (done % cfg.eval_every == 0 or done == cfg.total_updates)
            if is_eval:
                e = eval_model(model, tok, recipe, cfg.max_length, bm, dev)
                e["update"] = done
                evals.append(e)
                metrics_f.write(json.dumps({"eval": e}) + "\n")
                cur, ok = select_checkpoint(evals, cfg.guardrail_hard_lexical)
                if cfg.selection == "guarded_best" and ok and cur["update"] == done:
                    best = dict(e)
                    _save_weights(model, tok, out / "best")
                log_event("train", "EVAL", run_id=cfg.run_id, **e)
            if is_eval or done == cfg.total_updates or (cfg.stop_after and done == cfg.stop_after):
                metrics_f.flush()
                save_checkpoint(out, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                      "scheduler": sch.state_dict(), "rng": _rng_state(),
                                      "update": done, "evals": evals, "best": best,
                                      "skipped_total": skipped_total, "rows_total": rows_total,
                                      "schedule_hash": sched_hash}, identity)
    except BaseException as ex:
        metrics_f.close()
        log_event("train", "INTERRUPTED" if isinstance(ex, KeyboardInterrupt) else "FAILED",
                  run_id=cfg.run_id, error=repr(ex)[:500])
        raise
    metrics_f.close()
    finished = cfg.stop_after is None or cfg.stop_after >= cfg.total_updates
    result = {"run_id": cfg.run_id, "updates_done": done,
              "evals": evals, "schedule_hash": sched_hash, "card_schedule_hash": card_hash,
              "finished": finished}
    if finished and evals:
        sel, passed = finalise_selection(evals, cfg)
        _save_weights(model, tok, out / "final")
        use_best = cfg.selection == "guarded_best" and passed
        result.update(selected=sel, guardrail_passed=passed, selection=cfg.selection,
                      selected_dir=str(out / ("best" if use_best else "final")))
        atomic_write_json(out / "result.json", result)
        log_event("train", "COMPLETED", run_id=cfg.run_id, selected=sel, guardrail_passed=passed)
    elif finished:
        _save_weights(model, tok, out / "final")
        log_event("train", "COMPLETED", run_id=cfg.run_id, note="no evaluation configured")
    else:
        log_event("train", "PAUSED", run_id=cfg.run_id, update=cfg.stop_after)
    return result


def _save_weights(model, tok, path: Path) -> None:
    """Atomic directory replace: save to sibling temp dir, then swap."""
    tmp = path.with_name(path.name + ".tmp")
    old = path.with_name(path.name + ".old")
    if old.exists():  # an interrupted earlier swap
        if path.exists():
            shutil.rmtree(old)  # new copy was published; backup is stale
        else:
            os.replace(old, path)  # interrupted between the two renames: restore
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    if path.exists():
        os.replace(path, old)
    os.replace(tmp, path)
    if old.exists():
        shutil.rmtree(old)


def _truncate_metrics(path: Path, upto: int) -> None:
    """Drop records logged after the checkpoint being resumed (they will be redone)."""
    if not path.exists():
        return
    keep = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if n == len(lines) - 1:
                break  # torn final record from an interrupted write
            raise
        u = r["eval"]["update"] if "eval" in r else r["update"]
        if u <= upto:
            keep.append(line)
    atomic_write_bytes(path, "".join(line + "\n" for line in keep).encode())


def eval_model(model, tok, recipe, max_length, bm: Benchmark, dev) -> dict:
    v = enc.encode(model, tok, bm.texts, recipe, max_length, batch_size=128, dev=dev)
    r = evaluate(bm, v)["summary"]
    norms_ok = bool(np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3))
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(v), 2000)
    j = rng.integers(0, len(v), 2000)
    return {"full": r["full"]["ndcg@10"], "hard_lexical": r["hard_lexical"]["ndcg@10"],
            "hard": r["hard"]["ndcg@10"], "mean_random_cos": float((v[i] * v[j]).sum(1).mean()),
            "unit_norm": norms_ok}
