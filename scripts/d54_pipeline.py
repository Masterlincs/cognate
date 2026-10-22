"""One-off: score each finished run on the GPU before the next run is queued.

Replaces d54_watch.py. After x-2: dev eval (saved vectors) and trap exam for x-2, a dev eval
of x-1's selected checkpoint (the effect-only baseline), then merge d54-train (train.py is
in x-2's code identity, so it waits) and queue x-3. After x-3: score it, queue x-4. After
x-4: score it. CPU-only variants run after the next run is queued. A failed step is
logged and skipped; it never holds back the queue. Usage: uv run python scripts/d54_pipeline.py
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
    with open(ROOT / "runs/logs" / f"d54-{name}.log", "a", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, "-m", "mtgemb.cli", *args], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT)
    log_event("diag", "D54_STEP", args=list(args), exit_code=code)
    return code == 0


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


score("x-2-e5-hn3", wait("x-2-e5-hn3"))
step("eval-encoder", "x-1-e5-4000", *ENC, "--model-path", "runs/x-1-e5-4000/best")
r = subprocess.run(["git", "merge", "--no-edit", "d54-train"], cwd=ROOT, capture_output=True,
                   text=True)
if r.returncode:
    log_event("train", "D54_MERGE_FAILED", error=(r.stdout + r.stderr)[-500:])
    raise SystemExit(1)
queue("configs/x-3-e5-effect.json")
for name in ("x-2-e5-hn3", "x-1-e5-4000"):
    step("variants", name)
score("x-3-e5-effect", wait("x-3-e5-effect"))
queue("configs/x-4-e5-tagidf.json")
step("variants", "x-3-e5-effect")
score("x-4-e5-tagidf", wait("x-4-e5-tagidf"))
step("variants", "x-4-e5-tagidf")
log_event("diag", "D54_PIPELINE_DONE")
