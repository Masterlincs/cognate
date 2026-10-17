"""Command line entry points. Each command runs a real, implemented pipeline step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .util import read_json

DEFAULT_DS = "data/processed/2026-10-16_no_reminder_v1"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mtgemb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("acquire", help="download the current Scryfall bulk snapshot")
    p = sub.add_parser("prepare", help="build the processed dataset from a raw snapshot")
    p.add_argument("raw_dir")
    p.add_argument("--out")
    for name in ("pairs", "audit", "benchmark", "eval-lexical"):
        q = sub.add_parser(name)
        q.add_argument("--ds", default=DEFAULT_DS)
    q = sub.add_parser("eval-encoder", help="evaluate a pretrained or fine-tuned encoder")
    q.add_argument("name")
    q.add_argument("backbone")
    q.add_argument("--max-length", type=int, required=True)
    q.add_argument("--model-path")
    q.add_argument("--ds", default=DEFAULT_DS)
    q = sub.add_parser("export", help="encode the catalogue and write a release bundle")
    q.add_argument("backbone")
    q.add_argument("--max-length", type=int, required=True)
    q.add_argument("--label", required=True)
    q.add_argument("--status", required=True,
                   choices=["promoted", "experimental", "pretrained-fallback"])
    q.add_argument("--model-path")
    q.add_argument("--run-id")
    q.add_argument("--web-dir")
    q.add_argument("--ds", default=DEFAULT_DS)
    q = sub.add_parser("train", help="train or resume a run from a JSON config")
    q.add_argument("config")
    a = ap.parse_args(argv)

    if a.cmd == "acquire":
        from .acquire import acquire

        out = acquire()
    elif a.cmd == "prepare":
        from .data import prepare

        out = prepare(Path(a.raw_dir), Path(a.out) if a.out else None)["report"]
    elif a.cmd == "pairs":
        from .pairs import build_training_data

        out = build_training_data(Path(a.ds))
    elif a.cmd == "audit":
        from .pairs import audit_pairs

        out = audit_pairs(Path(a.ds))
    elif a.cmd == "benchmark":
        from .data import load_dataset
        from .evaluate import bench_path, build_benchmark, save_benchmark

        ds = load_dataset(a.ds)
        out = build_benchmark(ds["cards"], ds["memberships"], ds["tags"],
                              ds["ident"]["dataset_id"])
        save_benchmark(out, bench_path(a.ds))
        out = {k: out[k] for k in ("benchmark_id", "counts")}
    elif a.cmd == "eval-lexical":
        from .baselines import eval_lexical

        out = eval_lexical(a.ds)["summary"]
    elif a.cmd == "eval-encoder":
        from .baselines import eval_encoder

        out = eval_encoder(a.ds, a.name, a.backbone, a.max_length, a.model_path)["summary"]
    elif a.cmd == "export":
        from .export import export

        out = export(a.ds, a.backbone, a.max_length, a.label, a.status, a.model_path,
                     a.run_id, a.web_dir)
    elif a.cmd == "train":
        from .evaluate import bench_path
        from .train import TrainConfig, train

        cfg = TrainConfig(**read_json(a.config))
        bench = read_json(bench_path(cfg.dataset_dir)) if cfg.eval_every else None
        out = train(cfg, bench)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
