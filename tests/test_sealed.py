import json

import pytest

from mtgemb import sealed


def _setup(tmp_path, monkeypatch):
    runs, ds = tmp_path / "runs", tmp_path / "ds"
    ds.mkdir()
    (ds / "benchmark_v3.json").write_text(json.dumps({"benchmark_id": "B"}), "utf-8")
    (ds / "benchmark_v3_sealed.json").write_text(json.dumps({"benchmark_id": "S"}), "utf-8")
    (runs / "confirmation" / "B").mkdir(parents=True)
    (runs / "confirmation" / "B" / "report.json").write_text("{}", "utf-8")
    for name in sealed.SYSTEMS:
        d = runs / "eval" / "B" / name
        d.mkdir(parents=True)
        meta = {"system": "lexical"} if name.startswith("lexical") else {
            "system": "encoder", "backbone": "b", "max_length": 8, "model_path": None}
        (d / "summary.json").write_text(json.dumps({"meta": meta}), "utf-8")
    calls = []
    monkeypatch.setattr(sealed, "RUNS", runs)
    monkeypatch.setattr(sealed, "eval_lexical", lambda *a, **k: calls.append(a[1]))
    monkeypatch.setattr(sealed, "eval_encoder", lambda *a, **k: calls.append(a[1]))
    monkeypatch.setattr(sealed, "confirmation_report",
                        lambda bid, train_bench: {"sealed_gate": "cleared"})
    return runs, ds, calls


def test_sealed_eval_runs_once_then_refuses(tmp_path, monkeypatch):
    runs, ds, calls = _setup(tmp_path, monkeypatch)
    assert sealed.sealed_eval(ds)["sealed_gate"] == "cleared"
    assert calls == list(sealed.SYSTEMS) and (runs / "sealed" / "run.lock").exists()
    with pytest.raises(FileExistsError, match="already started"):
        sealed.sealed_eval(ds)  # a second launch, concurrent or after a crash
    assert len(calls) == len(sealed.SYSTEMS)


def test_sealed_eval_refuses_partial_records(tmp_path, monkeypatch):
    runs, ds, calls = _setup(tmp_path, monkeypatch)
    (runs / "eval" / "S" / "lexical-char35-tfidf").mkdir(parents=True)  # stale, no lock
    with pytest.raises(FileExistsError, match="already exist"):
        sealed.sealed_eval(ds)
    assert calls == []
