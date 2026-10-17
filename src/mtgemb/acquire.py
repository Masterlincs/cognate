"""Download pinned Scryfall bulk files (Oracle cards + Oracle tags) with a provenance manifest."""

from __future__ import annotations

import time
from pathlib import Path

import requests

from .util import DATA, atomic_write_json, log_event, sha256_file, utcnow

API = "https://api.scryfall.com/bulk-data"
HEADERS = {"User-Agent": "mtgemb/0.1", "Accept": "application/json"}
BULK_TYPES = ("oracle_cards", "oracle_tags")
TERMS = (
    "Scryfall API data guidelines (https://scryfall.com/docs/api): free for research and "
    "Magic software; no paywalling; no plain repackaging/republishing; no implied Scryfall "
    "endorsement. Oracle tags are community data from Scryfall Tagger. Card text is (c) Wizards "
    "of the Coast, used under the WotC Fan Content Policy."
)


def _get(url: str, *, stream: bool = False, tries: int = 4) -> requests.Response:
    """GET with bounded retries, Retry-After handling and a minimum spacing."""
    for attempt in range(tries):
        time.sleep(0.2)  # aggregate throttle: far below Scryfall's 10 req/s guidance
        try:
            r = requests.get(url, headers=HEADERS, stream=stream, timeout=60)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After", 2**attempt))
            time.sleep(min(wait, 60))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"giving up on {url} after {tries} attempts")


def acquire(out_dir: Path | None = None) -> dict:
    """Download each bulk file once into data/raw/<snapshot>/; raw files are never overwritten."""
    index = _get(API).json()["data"]
    by_type = {b["type"]: b for b in index}
    stamp = by_type["oracle_cards"]["updated_at"][:10]
    out_dir = Path(out_dir or DATA / "raw" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"snapshot": stamp, "retrieved_utc": utcnow(), "terms": TERMS, "files": {}}
    for t in BULK_TYPES:
        b = by_type[t]
        url = b["jsonl_download_uri"]
        dest = out_dir / Path(url).name
        if not dest.exists():
            log_event("data", "STARTED", step="download", url=url)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with _get(url, stream=True) as r, open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
        manifest["files"][t] = {
            "url": url,
            "bulk_id": b["id"],
            "updated_at": b["updated_at"],
            "path": dest.name,
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
        }
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        atomic_write_json(mpath, manifest)
    log_event("data", "COMPLETED", step="acquire", snapshot=stamp, files=manifest["files"])
    return manifest
