import numpy as np
from test_evaluate import _fixture

from mtgemb import baselines
from mtgemb.evaluate import Benchmark, build_benchmark, evaluate
from mtgemb.util import sha256_file


def _setup(tmp_path, monkeypatch):
    cards, mem, tags = _fixture()
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    ds = {"cards": cards[cards["split"] != "sealed"], "ident": {"dataset_id": "ds"}}
    monkeypatch.setattr(baselines, "load_benchmark", lambda _ds: (bm, ds))
    monkeypatch.setattr(baselines, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(baselines, "ARTIFACTS", tmp_path / "art")
    v = np.random.default_rng(0).normal(size=(len(bm.ids), 4)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    baselines.save_eval("enc", bm, evaluate(bm, v), {"system": "encoder"}, vectors=v)
    return bm, v, tmp_path / "runs" / "eval" / bm.spec["benchmark_id"] / "enc"


def _release(tmp_path, v, run_id="enc"):
    d = tmp_path / "art" / "releases" / "r1"
    d.mkdir(parents=True)
    np.save(d / "vectors.npy", v)
    (d / "release_meta.json").write_text(
        f'{{"release_id": "r1", "run_id": "{run_id}", '
        f'"vectors_sha256": "{sha256_file(d / "vectors.npy")}"}}', "utf-8")


def test_replay_matches_untouched_records_and_release(tmp_path, monkeypatch):
    _, v, _ = _setup(tmp_path, monkeypatch)
    _release(tmp_path, v)
    r = baselines.replay_evals("ds")
    assert r["passed"] and r["evals"]["enc"]["max_abs_diff"] == 0.0
    assert r["releases"]["r1"]["max_abs_diff_vs_eval"] == 0.0
    baselines.replay_evals("ds")  # a second replay writes a new report, never overwrites
    assert len(list((tmp_path / "runs" / "replay").rglob("*.json"))) == 2


def test_replay_fails_on_tampered_vectors(tmp_path, monkeypatch):
    _, v, d = _setup(tmp_path, monkeypatch)
    np.save(d / "vectors.npy", v[::-1].copy())  # different vectors, stale recorded hash
    r = baselines.replay_evals("ds")
    assert not r["passed"] and r["evals"]["enc"]["hash_mismatch"] == ["vectors.npy"]


def test_replay_fails_when_release_is_not_the_evaluated_model(tmp_path, monkeypatch):
    _, v, _ = _setup(tmp_path, monkeypatch)
    _release(tmp_path, -v)
    r = baselines.replay_evals("ds")
    assert not r["passed"] and r["releases"]["r1"]["max_abs_diff_vs_eval"] > 1
