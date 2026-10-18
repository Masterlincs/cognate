"""HTTP contract checks using only synthetic, blinded pool data."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.client import HTTPConnection
from pathlib import Path

import pytest

from mtgemb.annotate import make_server


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    pool = {
        "design": {"rubric": {"3": "same", "2": "related", "1": "incidental",
                              "0": "unrelated", "unknown": "unsure"}},
        "cards": {
            "multi": {"name": "Two abilities", "abilities": ["Draw a card.", "Flying"],
                      "faces": [{"name": "Front", "mana_cost": "{U}",
                                 "type_line": "Creature", "oracle_text": "Draw a card.\nFlying"},
                                {"name": "Back", "mana_cost": "", "type_line": "Land",
                                 "oracle_text": "{T}: Add {U}."}], "image": None},
            "single": {"name": "One ability", "abilities": [], "faces": [
                {"name": "One ability", "mana_cost": "{U}", "type_line": "Sorcery",
                 "oracle_text": "Draw a card."}], "image": None},
        },
        "items": [{"item_id": "c0", "phase": "calibration", "anchor": "multi",
                   "candidate": "single"},
                  {"item_id": "s1", "phase": "scored", "anchor": "single",
                   "candidate": "multi"},
                  {"item_id": "r2", "phase": "repeat", "anchor": "multi",
                   "candidate": "single"}],
    }
    raw = json.dumps(pool, indent=2).encode()
    (tmp_path / "pool.json").write_bytes(raw)
    (tmp_path / "key.json").write_text('{"secret": "not for annotation"}', encoding="utf-8")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name != "key.json", "annotation must never open the blinding key"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path, raw
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def request(pilot, path, payload=None, raw=None, content_type="application/json"):
    server = pilot[0]
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        body = raw if raw is not None else json.dumps(payload) if payload is not None else None
        connection.request("POST" if body is not None else "GET", path, body=body,
                           headers={"Content-Type": content_type})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def judgment(**changes):
    return {"item_id": "c0", "grade": 3, "ability": None, "reason": "aligned", **changes}


def test_append_and_rejudge(pilot, monkeypatch):
    import mtgemb.annotate as annotate

    synced = []
    fsync = annotate.os.fsync

    def observed_fsync(fd):
        synced.append(fd)
        fsync(fd)

    monkeypatch.setattr(annotate.os, "fsync", observed_fsync)
    assert pilot[0].server_address[0] == "127.0.0.1"
    assert request(pilot, "/api/labels") == (200, b"")
    status, body = request(pilot, "/api/label", judgment(ability=1))
    assert status == 200
    first = json.loads(body)
    assert {key: first[key] for key in judgment()} == judgment(ability=1)
    assert datetime.fromisoformat(first["ts"]).utcoffset().total_seconds() == 0
    assert isinstance(first["session"], str) and first["session"]
    path = pilot[1] / "labels.jsonl"
    original = path.read_bytes()
    assert original.endswith(b"\n") and len(original.splitlines()) == 1
    status, body = request(pilot, "/api/label", judgment(grade="unknown", ability="whole"))
    assert status == 200
    second = json.loads(body)
    assert second["session"] == first["session"]
    assert path.read_bytes().startswith(original)
    assert [json.loads(line) for line in path.read_bytes().splitlines()] == [first, second]
    assert request(pilot, "/api/labels") == (200, path.read_bytes())
    assert len(synced) == 2


@pytest.mark.parametrize("changes", [
    {"item_id": "absent"}, {"item_id": []}, {"grade": 4}, {"grade": -1},
    {"grade": "3"}, {"grade": True}, {"grade": False}, {"grade": 2.0},
    {"grade": None}, {"grade": []}, {"grade": {}},
    {"ability": 2}, {"ability": -1}, {"ability": True}, {"ability": 0.0},
    {"ability": "0"}, {"ability": []}, {"ability": {}},
    {"item_id": "s1", "ability": 0}, {"item_id": "s1", "ability": "whole"},
    {"reason": "x" * 501}, {"reason": None}, {"unexpected": "field"},
])
def test_invalid_label_preserves_file(pilot, changes):
    assert request(pilot, "/api/label", judgment())[0] == 200
    path = pilot[1] / "labels.jsonl"
    original = path.read_bytes()
    status, body = request(pilot, "/api/label", judgment(**changes))
    assert status == 400
    assert json.loads(body)["error"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("raw", [b"{", b"null", b"[]", b"{}", b"\xff", b"x" * 16385])
def test_malformed_body_does_not_create_labels(pilot, raw):
    assert request(pilot, "/api/label", raw=raw)[0] == 400
    assert not (pilot[1] / "labels.jsonl").exists()


def test_routes_and_blinding(pilot):
    assert request(pilot, "/api/pool") == (200, pilot[2])
    status, page = request(pilot, "/")
    assert status == 200 and b"Grading rubric" in page
    for path in ("/key.json", "/api/key", "/pool.json", "/../key.json", "/%6bey.json"):
        assert request(pilot, path)[0] == 404
    assert request(pilot, "/api/key", judgment())[0] == 404
    assert request(pilot, "/api/label", judgment(), content_type="text/plain")[0] == 400


def test_all_grades_and_concurrent_append(pilot):
    payloads = [judgment(grade=grade, ability=ability, reason="x" * 500)
                for grade in (0, 1, 2, 3, "unknown") for ability in (None, 0, 1, "whole")]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda label: request(pilot, "/api/label", label), payloads))
    assert all(status == 200 for status, _ in results)
    lines = (pilot[1] / "labels.jsonl").read_text().splitlines()
    assert len(lines) == len(payloads)
    assert {json.loads(line)["session"] for line in lines} == {
        json.loads(results[0][1])["session"]}


def test_existing_labels_and_new_session(pilot):
    path = pilot[1] / "labels.jsonl"
    original = json.dumps({**judgment(), "ts": "old", "session": "previous"}) + "\n"
    path.write_text(original, encoding="utf-8")
    original_bytes = path.read_bytes()
    assert request(pilot, "/api/labels")[1] == original_bytes
    status, body = request(pilot, "/api/label", judgment(item_id="s1", grade=0))
    assert status == 200 and json.loads(body)["session"] != "previous"
    assert path.read_bytes().startswith(original_bytes)


def test_cli_dispatch(monkeypatch):
    from mtgemb import annotate, cli

    calls = []
    monkeypatch.setattr(annotate, "serve", lambda *args: calls.append(args))
    cli.main(["annotate", "synthetic"])
    cli.main(["annotate", "synthetic", "--port", "9999"])
    assert calls == [("synthetic", 8765), ("synthetic", 9999)]
