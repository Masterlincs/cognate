"""One-off: after x-2 finishes, merge the d54-train branch and queue x-3, x-4.

train.py is part of a run's code identity, so the branch must not land while x-2 could
still need to resume. Usage: uv run python scripts/d54_watch.py (runs until done).
"""

import subprocess
import time
from pathlib import Path

from mtgemb.util import log_event

ROOT = Path(__file__).resolve().parents[1]
DONE = ROOT / "runs/x-2-e5-hn3/result.json"
CONFIGS = ["configs/x-3-e5-effect.json", "configs/x-4-e5-tagidf.json"]

while not DONE.exists():
    time.sleep(120)
r = subprocess.run(["git", "merge", "--no-edit", "d54-train"], cwd=ROOT, capture_output=True,
                   text=True)
if r.returncode:
    log_event("train", "D54_WATCH_FAILED", step="merge", error=(r.stdout + r.stderr)[-500:])
    raise SystemExit(1)
with open(ROOT / "runs/queue.txt", "a", encoding="utf-8") as f:
    f.write("".join(c + "\n" for c in CONFIGS))
log_event("train", "D54_WATCH_QUEUED", configs=CONFIGS)
