"""Local, blinded annotation of an existing pool; never opens the blinding key."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_server(pool_dir: str | Path, port: int = 8765) -> ThreadingHTTPServer:
    """Create a loopback-only server. The caller owns shutdown and server_close."""
    pool_dir = Path(pool_dir)
    pool_bytes = (pool_dir / "pool.json").read_bytes()
    pool = json.loads(pool_bytes)
    items = {item["item_id"]: item for item in pool["items"]}
    page = Path(__file__).with_name("annotate.html").read_bytes()
    labels_path = pool_dir / "labels.jsonl"
    lock = threading.Lock()
    session = uuid.uuid4().hex

    class Handler(BaseHTTPRequestHandler):
        timeout = 10  # seconds: a client that stalls mid-body is dropped, not left holding a thread

        def reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def json_reply(self, status: int, value) -> None:
            self.reply(status, json.dumps(value).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/pool":
                self.reply(200, pool_bytes, "application/json; charset=utf-8")
            elif self.path == "/api/labels":
                with lock:
                    body = labels_path.read_bytes() if labels_path.exists() else b""
                self.reply(200, body, "application/x-ndjson; charset=utf-8")
            else:
                self.json_reply(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/api/label":
                self.json_reply(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                # read before rejecting: closing on unread data resets the connection on
                # Windows, so the client would see an abort instead of the 400
                body = self.rfile.read(length) if 0 < length <= 1 << 20 else b""
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json")
                if not 0 < length <= 16384:
                    raise ValueError("Invalid body length (maximum 16384 bytes)")
                label = json.loads(body)
                if not isinstance(label, dict) or set(label) != {
                    "item_id", "grade", "ability", "reason"
                }:
                    raise ValueError("Expected item_id, grade, ability and reason only")
                iid, grade = label["item_id"], label["grade"]
                if not isinstance(iid, str) or iid not in items:
                    raise ValueError("Unknown item_id")
                if not (type(grade) is int and grade in (0, 1, 2, 3)) and grade != "unknown":
                    raise ValueError("Grade must be 0, 1, 2, 3 or unknown")
                ability = label["ability"]
                abilities = pool["cards"][items[iid]["anchor"]]["abilities"]
                if ability is not None and not (abilities and (
                    ability == "whole" or type(ability) is int and 0 <= ability < len(abilities)
                )):
                    raise ValueError("Ability must be null, whole or a valid anchor ability index")
                if not isinstance(label["reason"], str) or len(label["reason"]) > 500:
                    raise ValueError("Reason must be a string of at most 500 characters")
            except (ValueError, UnicodeError, RecursionError) as exc:
                self.json_reply(400, {"error": str(exc)})
                return
            label.update(ts=datetime.now(UTC).isoformat(), session=session)
            try:
                with lock, labels_path.open("ab") as stream:
                    stream.write((json.dumps(label) + "\n").encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                self.json_reply(503, {"error": f"Could not save judgment: {exc}"})
                return
            self.json_reply(200, label)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(pool_dir: str | Path, port: int = 8765) -> None:
    with make_server(pool_dir, port) as server:
        print(f"Annotate at http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
