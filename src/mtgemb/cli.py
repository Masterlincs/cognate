"""Command-line entry points for each pipeline step."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .util import log_event, read_json

DEFAULT_DS = "data/processed/2026-10-16_no_reminder_v1"


def gpu_limits(env=None, clock=None, sleep=None) -> dict:
    """Share the laptop GPU with the desktop, from env vars; training maths is unchanged.

    MTGEMB_GPU_MEM_GB caps the XPU memory the process may hold. MTGEMB_GPU_REST=r makes
    the trainer idle r times as long as each update took after every optimizer step
    (r=1 keeps the GPU at most about half busy). Lives here, not in train.py, so the code
    identity of a running run does not change and it can resume under the limits.
    """
    import torch

    env = os.environ if env is None else env
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    mem, rest = env.get("MTGEMB_GPU_MEM_GB"), float(env.get("MTGEMB_GPU_REST") or 0)
    if mem and torch.xpu.is_available():
        total = torch.xpu.get_device_properties(0).total_memory / 1e9
        torch.xpu.set_per_process_memory_fraction(min(1.0, float(mem) / total))
    hook = None
    if rest > 0:
        last = [clock()]

        def pause(*_):
            if torch.xpu.is_available():
                torch.xpu.synchronize()
            # an update that includes an evaluation is long; cap the rest so it stays short
            sleep(min(rest * (clock() - last[0]), 60.0))
            last[0] = clock()

        from torch.optim.optimizer import register_optimizer_step_post_hook

        hook = register_optimizer_step_post_hook(pause)
    return {"mem_gb": float(mem) if mem else None, "rest": rest, "hook": hook}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mtgemb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("acquire", help="download the Scryfall bulk snapshot")
    p = sub.add_parser("prepare", help="build the processed dataset from a raw snapshot")
    p.add_argument("raw_dir")
    p.add_argument("--out")
    for name in ("pairs", "audit", "benchmark", "eval-lexical"):
        q = sub.add_parser(name)
        q.add_argument("--ds", default=DEFAULT_DS)
        if name == "pairs":
            q.add_argument("--version", default="pairs_v1", choices=["pairs_v1", "pairs_v2"])
    q = sub.add_parser("eval-encoder", help="evaluate a pretrained or fine-tuned encoder on dev")
    q.add_argument("name")
    q.add_argument("backbone")
    q.add_argument("--max-length", type=int, required=True)
    q.add_argument("--model-path")
    q.add_argument("--ds", default=DEFAULT_DS)
    q = sub.add_parser("replay-eval", help="re-score recorded evaluations from saved vectors")
    q.add_argument("--ds", default=DEFAULT_DS)
    q = sub.add_parser("variants", help="face-match and effect-only scores for a dev eval")
    q.add_argument("name")
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
    q.add_argument("--from-release", metavar="ID")
    q.add_argument("--decision", help="promotion decision text for a promoted release")
    q.add_argument("--ds", default=DEFAULT_DS)
    q = sub.add_parser(
        "confirm-report", help="compute confirmation contrasts and print the promotion decision"
    )
    q.add_argument("--ds", default=DEFAULT_DS)
    q.add_argument("--design", choices=["d18", "x1"], default="d18")
    q = sub.add_parser("site-data", help="build map, pretrained neighbours, and site metrics")
    q.add_argument("--release", metavar="ID")
    q.add_argument("--metrics-only", action="store_true")
    q.add_argument(
        "--copy-from",
        metavar="ID",
        help="copy the map of an earlier release with identical card vectors",
    )
    q = sub.add_parser("sealed-eval", help="run the authorised one-shot sealed evaluation")
    q.add_argument("--ds", default=DEFAULT_DS)
    q = sub.add_parser("train", help="train or resume a run from a JSON config")
    q.add_argument("config")
    q = sub.add_parser("exam-score", help="score an encoder on the accepted trap-exam items")
    q.add_argument("name")
    q.add_argument("backbone")
    q.add_argument("--max-length", type=int, required=True)
    q.add_argument("--model-path")
    q.add_argument("--text-version", choices=["no_reminder_v1", "no_reminder_v2"],
                   default="no_reminder_v1")
    q.add_argument("--ds", default=DEFAULT_DS)
    q = sub.add_parser("exam-guard", help="find trap-exam regressions between two models")
    q.add_argument("old")
    q.add_argument("new")
    q = sub.add_parser("annotate", help="run the local web app to label a blinded human pilot pool")
    q.add_argument("pool_dir")
    q.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    if a.cmd == "annotate":
        from .annotate import serve

        serve(a.pool_dir, a.port)
        return
    elif a.cmd == "exam-score":
        from .exam import exam_score

        out = exam_score(a.ds, a.name, a.backbone, a.max_length, a.model_path,
                         text_version=a.text_version)
        out = {k: out[k] for k in ("name", "n", "passed", "sections", "dev_only", "n_proposed")}
    elif a.cmd == "exam-guard":
        from .exam import guard

        out = guard(read_json(a.old), read_json(a.new))
    elif a.cmd == "acquire":
        from .acquire import acquire

        out = acquire()
    elif a.cmd == "prepare":
        from .data import prepare

        out = prepare(Path(a.raw_dir), Path(a.out) if a.out else None)["report"]
    elif a.cmd == "pairs":
        from .pairs import build_training_data

        out = build_training_data(Path(a.ds), a.version)
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
    elif a.cmd == "replay-eval":
        from .baselines import replay_evals

        out = replay_evals(a.ds)
    elif a.cmd == "variants":
        from .variants import diagnose

        out = diagnose(a.ds, a.name)
    elif a.cmd == "export":
        from .export import export

        out = export(a.ds, a.backbone, a.max_length, a.label, a.status, a.model_path,
                     a.run_id, a.web_dir, from_release=a.from_release,
                     decision=a.decision)
    elif a.cmd == "confirm-report":
        from .evaluate import bench_path
        from .report import DESIGNS, confirmation_report

        out = confirmation_report(read_json(bench_path(a.ds))["benchmark_id"],
                                  design=DESIGNS[a.design])
    elif a.cmd == "site-data":
        from .sitedata import build_site_data, copy_site_data, write_metrics

        out = (str(write_metrics()) if a.metrics_only else
               copy_site_data(a.release, a.copy_from) if a.copy_from else
               build_site_data(a.release))
    elif a.cmd == "sealed-eval":
        from .sealed import sealed_eval

        out = sealed_eval(a.ds)
    elif a.cmd == "train":
        from .evaluate import bench_path
        from .train import TrainConfig, train

        cfg = TrainConfig(**read_json(a.config))
        limits = gpu_limits()
        if limits["mem_gb"] or limits["rest"]:
            log_event("train", "GPU_LIMITS", run_id=cfg.run_id, mem_gb=limits["mem_gb"],
                      rest=limits["rest"])
        bench = read_json(bench_path(cfg.dataset_dir)) if cfg.eval_every else None
        out = train(cfg, bench)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
