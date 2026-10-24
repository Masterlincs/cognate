"""Hashing, atomic writes and the append-only event log."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
ARTIFACTS = ROOT / "artifacts"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temp file in the same directory, fsync, then rename over the target.

    A crash before the rename leaves the previous file intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log_event(log: str, event: str, **fields: Any) -> dict:
    """Append one line to runs/<log>.jsonl. Never pass credentials here."""
    RUNS.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "event": event, **fields}
    with open(RUNS / f"{log}.jsonl", "a", encoding="utf-8") as f:
        f.write(canonical_json(rec) + "\n")
    return rec


def git_identity() -> dict:
    """Commit hash and whether the tracked tree is dirty."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, capture_output=True, text=True,
                                    check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": head, "dirty": dirty}
