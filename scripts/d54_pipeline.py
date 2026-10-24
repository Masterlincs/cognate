"""One-off: score each finished run on the GPU before the next run is queued.

Waits for x-3, then runs the dev evals of x-2 and of x-1's selected checkpoint (the
effect-only baseline; both re-run because the first attempts died in a native XPU abort on
2026-10-22), scores x-3 (dev eval and trap exam) and queues x-4. The CPU-only variants
of x-1 to x-3 run while x-4 trains; then x-4 is scored. A failed step is logged and skipped;
it never holds back the queue. GPU steps run with faulthandler and up to two retries.
Usage: uv run python scripts/d54_pipeline.py
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from mtgemb.util import log_event

ROOT = Path(__file__).resolve().parents[1]
ENC = ["e5-base-v2", "--max-length", "94"]


def step(*args: str) -> bool:
    name = re.sub(r"[^\w.-]+", "-", "_".join(a for a in args if not a.startswith("-")))[:80]
    for attempt in range(1 if args[0] == "variants" else 3):
        with open(ROOT / "runs/logs" / f"d54-{name}.log", "a", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, "-X", "faulthandler", "-m", "mtgemb.cli",
                                    *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        log_event("diag", "D54_STEP", args=list(args), exit_code=code, attempt=attempt)
        if code == 0:
            return True
        time.sleep(60)
    return False


def wait(run_id: str) -> str:
    res = ROOT / "runs" / run_id / "result.json"
    while not res.exists():
        time.sleep(120)
    return json.loads(res.read_text())["selected_dir"]


def score(run_id: str, model_dir: str) -> None:
    step("eval-encoder", run_id, *ENC, "--model-path", model_dir)
    step("exam-score", run_id, *ENC, "--model-path", model_dir)


def queue(cfg: str) -> None:
    with open(ROOT / "runs/queue.txt", "a", encoding="utf-8") as f:
        f.write(cfg + "\n")
    log_event("train", "D54_QUEUED", config=cfg)


x3 = wait("x-3-e5-effect")
step("eval-encoder", "x-2-e5-hn3", *ENC, "--model-path", wait("x-2-e5-hn3"))
step("eval-encoder", "x-1-e5-4000", *ENC, "--model-path", "runs/x-1-e5-4000/best")
score("x-3-e5-effect", x3)
queue("configs/x-4-e5-tagidf.json")
for name in ("x-2-e5-hn3", "x-1-e5-4000", "x-3-e5-effect"):
    step("variants", name)
score("x-4-e5-tagidf", wait("x-4-e5-tagidf"))
step("variants", "x-4-e5-tagidf")
log_event("diag", "D54_PIPELINE_DONE")
