"""The authorised one-shot sealed evaluation of the frozen systems.

Re-scores the nine dev-evaluated systems, unchanged, on a sealed benchmark (sealed anchors,
fit+dev+sealed universe), then re-tests the registered P7 contrasts there. Nothing here
trains, tunes or selects; the P7 decision and the release are not touched.
"""

from __future__ import annotations

import os

from .baselines import eval_encoder, eval_lexical
from .data import load_dataset
from .evaluate import bench_path, build_benchmark, save_benchmark
from .report import CONFIRM, confirmation_report
from .util import RUNS, canonical_json, git_identity, log_event, read_json, utcnow

SYSTEMS = ("lexical-char35-tfidf", "pretrained-e5-base-v2", "pretrained-bge-small-en-v1.5",
           *(CONFIRM["candidate"].format(s) for s in CONFIRM["seeds"]),
           *(CONFIRM["bge"].format(s) for s in CONFIRM["seeds"]))


def sealed_eval(ds_dir) -> dict:
    """Fail closed: a lock file claims the single run and is never removed, so a second
    launch, concurrent or after a crash, refuses. A crashed run needs a human decision."""
    dev_bid = read_json(bench_path(ds_dir))["benchmark_id"]
    if not (RUNS / "confirmation" / dev_bid / "report.json").exists():
        raise FileNotFoundError("no P7 confirmation report: nothing frozen to test")
    lock = RUNS / "sealed" / "run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise FileExistsError(f"{lock} exists: the sealed evaluation was already started "
                              "(allows one run)") from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(canonical_json({"pid": os.getpid(), "started_utc": utcnow(),
                                "code": git_identity()}) + "\n")
    path = bench_path(ds_dir, "sealed")
    if not path.exists():
        ds = load_dataset(ds_dir, include_sealed=True)
        save_benchmark(build_benchmark(ds["cards"], ds["memberships"], ds["tags"],
                                       ds["ident"]["dataset_id"], held="sealed"), path)
    bid = read_json(path)["benchmark_id"]
    if (RUNS / "eval" / bid).exists() or (RUNS / "sealed" / bid).exists():
        raise FileExistsError(f"sealed records for {bid} already exist; refusing")
    log_event("sealed", "STARTED", benchmark_id=bid, train_benchmark_id=dev_bid)
    for name in SYSTEMS:
        m = read_json(RUNS / "eval" / dev_bid / name / "summary.json")["meta"]
        if m["system"] == "lexical":
            eval_lexical(ds_dir, name, held="sealed")
        else:
            eval_encoder(ds_dir, name, m["backbone"], m["max_length"], m["model_path"],
                         held="sealed")
    rep = confirmation_report(bid, train_bench=dev_bid)
    log_event("sealed", "COMPLETED", benchmark_id=bid, sealed_gate=rep["sealed_gate"])
    return rep
