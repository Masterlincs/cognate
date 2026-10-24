"""One-off: score each finished run on the GPU before the next run is queued.

After x-2: dev eval (saved vectors) and trap exam for x-2, a dev eval
of x-1's selected checkpoint (the effect-only baseline), then merge d54-train (train.py is
in x-2's code identity, so it waits) and queue x-3. After x-3: score it, queue x-4. After
x-4: score it. CPU-only variants run after the next run is queued. A failed step is
logged and skipped; it never holds back the queue. Usage: uv run python scripts/d54_pipeline.py

GPU steps run with faulthandler and up to two retries.
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
