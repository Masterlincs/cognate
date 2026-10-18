"""Run training configs one after another. Usage: uv run python scripts/run_queue.py

Reads runs/queue.txt (one config path per line) again before each run, so lines can be
appended while it works. Skips a run that has result.json, waits while the GPU lock is held
by a live process, and resumes a crashed run once. Stops when runs/queue.stop exists or
nothing is left after an hour of idling.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from mtgemb.train import _pid_alive
from mtgemb.util import log_event

ROOT = Path(__file__).resolve().parents[1]
QUEUE, STOP, LOCK = ROOT / "runs/queue.txt", ROOT / "runs/queue.stop", ROOT / "runs/gpu.lock"


def pending(failed: dict) -> list[str]:
    out = []
    for line in QUEUE.read_text().splitlines():
        cfg = line.strip()
        if not cfg or cfg.startswith("#"):
            continue
        run_id = json.loads((ROOT / cfg).read_text())["run_id"]
        if not (ROOT / "runs" / run_id / "result.json").exists() and failed.get(cfg, 0) < 2:
            out.append(cfg)
    return out


def gpu_busy() -> bool:
    return LOCK.exists() and _pid_alive(int(json.loads(LOCK.read_text())["pid"]))


def main() -> None:
    failed: dict[str, int] = {}
    idle = 0
    while not STOP.exists():
        todo = pending(failed)
        if not todo or gpu_busy():
            idle = idle + 1 if not todo else 0
            if idle > 60:
                break
            time.sleep(60)
            continue
        cfg = todo[0]
        log_event("train", "QUEUE_LAUNCH", config=cfg, attempt=failed.get(cfg, 0) + 1)
        name = Path(cfg).stem
        with open(ROOT / "runs/logs" / f"{name}.log", "a", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, "-m", "mtgemb.cli", "train", cfg],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code:
            failed[cfg] = failed.get(cfg, 0) + 1
            log_event("train", "QUEUE_FAILED", config=cfg, exit_code=code,
                      attempt=failed[cfg])
            time.sleep(30)
    log_event("train", "QUEUE_EXIT", stop_file=STOP.exists())


if __name__ == "__main__":
    main()
