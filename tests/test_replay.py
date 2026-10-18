import json

import numpy as np
import pandas as pd
import pytest
from test_evaluate import _fixture

from mtgemb import baselines
from mtgemb.evaluate import Benchmark, build_benchmark, evaluate
from mtgemb.util import sha256_file

RECIPE = {"model_id": "m", "revision": "r", "prefix": "query: ", "pooling": "mean",
          "normalize": True, "dim": 4}  # registered as a known recipe in _setup


def _setup(tmp_path, monkeypatch):
    cards, mem, tags = _fixture()
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    ds = {"cards": cards[cards["split"] != "sealed"], "ident": {"dataset_id": "ds"}}
    monkeypatch.setattr(baselines, "load_benchmark", lambda _ds: (bm, ds))
    monkeypatch.setattr(baselines, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(baselines, "ARTIFACTS", tmp_path / "art")
    monkeypatch.setitem(baselines.enc.RECIPES, "test", RECIPE)
    v = np.random.default_rng(0).normal(size=(len(bm.ids), 4)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    meta = {"system": "encoder", "recipe": RECIPE, "max_length": 8, "model_path": None}
    baselines.save_eval("enc", bm, evaluate(bm, v), meta, vectors=v)
    return v, tmp_path / "runs" / "eval" / bm.spec["benchmark_id"] / "enc"


def _release(tmp_path, v, run_id=None, max_length=8, dataset_id="ds", rid="r1", dirname="r1"):
    d, b = tmp_path / "art" / "releases" / dirname, tmp_path / f"bundle-{dirname}"
    d.mkdir(parents=True)
    b.mkdir()
    np.save(d / "vectors.npy", v)
    files = {}
    for f in ("names.json", "ids.json", "cards/000.json", "neighbours/000.json"):
        (b / f).parent.mkdir(exist_ok=True)
        (b / f).write_text("[]", "utf-8")
        files[f] = {"sha256": sha256_file(b / f), "bytes": (b / f).stat().st_size}
    n = int((_fixture()[0]["split"] != "sealed").sum())  # the fit+dev catalogue
    man = {"schema": "mtgemb-release-v2", "release_id": rid, "n_cards": n,
           "shard_size": n, "files": files}  # one shard
    (b / "release.json").write_text(json.dumps(man), "utf-8")
    meta = {"release_id": rid, "run_id": run_id, "dataset_id": dataset_id,
            "encoder": {**RECIPE, "max_length": max_length}, "model_path": None,
            "vectors_sha256": sha256_file(d / "vectors.npy"), "bundle_dir": str(b),
            "release_json_sha256": sha256_file(b / "release.json")}
    (d / "release_meta.json").write_text(json.dumps(meta), "utf-8")
    return d, b


def _edit_json(path, fn):
    rec = json.loads(path.read_text("utf-8"))
    fn(rec)
    path.write_text(json.dumps(rec), "utf-8")


def test_replay_passes_on_untouched_records_and_matching_release(tmp_path, monkeypatch):
    v, _ = _setup(tmp_path, monkeypatch)
    _release(tmp_path, v)  # run_id None: still compared, via recipe/length/weights
    for name in ("r2.partial", "r2.failed-1"):  # unpublished exports: listed, not checked
        (tmp_path / "art" / "releases" / name).mkdir()
    r = baselines.replay_evals("ds")
    assert r["passed"] and r["evals"]["enc"]["max_abs_diff"] == 0.0
    assert r["unpublished_release_dirs"] == ["r2.failed-1", "r2.partial"]
    assert list(r["releases"]) == ["r1"]
    assert r["releases"]["r1"]["compared_with"] == ["enc"]
    assert r["releases"]["r1"]["max_abs_diff_vs_eval"] == {"enc": 0.0}
    baselines.replay_evals("ds")  # a second replay writes a new report, never overwrites
    assert len(list((tmp_path / "runs" / "replay").rglob("*.json"))) == 2


def _tamper_vectors(tmp_path, v, d):
    np.save(d / "vectors.npy", v[::-1].copy())


def _drop_summary(tmp_path, v, d):
    (d / "summary.json").unlink()


def _wrong_benchmark(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r.update(benchmark_id="other"))


def _wrong_summary(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["summary"]["full"].update({"ndcg@10": 123}))


def _int_metric(tmp_path, v, d):
    per = pd.read_parquet(d / "per_anchor.parquet")
    per["ap"] = 123  # integer dtype, rehashed so only the comparison can catch it
    per.to_parquet(d / "per_anchor.parquet", index=False)
    _edit_json(d / "summary.json",
               lambda r: r["files"].update({"per_anchor.parquet":
                                            sha256_file(d / "per_anchor.parquet")}))


def _release_other_vectors(tmp_path, v, d):
    _release(tmp_path, -v, run_id="enc")


def _release_unevaluated_encoder(tmp_path, v, d):
    _release(tmp_path, v, max_length=9)


def _release_wrong_dataset(tmp_path, v, d):
    _release(tmp_path, v, dataset_id="other")


def _release_corrupt_bundle(tmp_path, v, d):
    _, b = _release(tmp_path, v)
    (b / "names.json").write_text('{"names": ["x"]}', "utf-8")


def _release_wrong_release_hash(tmp_path, v, d):
    r, _ = _release(tmp_path, v)
    _edit_json(r / "release_meta.json", lambda m: m.update(release_json_sha256="0" * 64))


def _release_nan_vectors(tmp_path, v, d):
    bad = v.copy()
    bad[0, 0] = np.nan
    _release(tmp_path, bad)  # recorded hash matches the NaN file


def _release_duplicate_id(tmp_path, v, d):
    _release(tmp_path, v, dataset_id="other", dirname="a")
    _release(tmp_path, v, dirname="r1")  # valid release sorted after, same release_id


def _bool_in_summary(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["summary"]["full"].update(
        {"anchors_total": True}))  # fixture value is 1


def _drop_per_anchor(tmp_path, v, d):
    (d / "per_anchor.parquet").unlink()


def _bool_counts(tmp_path, v, d):
    per = pd.read_parquet(d / "per_anchor.parquet")
    per["n_rel"] = per["n_rel"] > 0  # bool, rehashed: True would compare equal to 1.0
    per.to_parquet(d / "per_anchor.parquet", index=False)
    _edit_json(d / "summary.json",
               lambda r: r["files"].update({"per_anchor.parquet":
                                            sha256_file(d / "per_anchor.parquet")}))


def _release_missing_vectors(tmp_path, v, d):
    r, _ = _release(tmp_path, v)
    (r / "vectors.npy").unlink()


def _rehash(d, f):
    _edit_json(d / "summary.json", lambda r: r["files"].update({f: sha256_file(d / f)}))


def _truncated_per_anchor(tmp_path, v, d):
    f = d / "per_anchor.parquet"
    f.write_bytes(f.read_bytes()[:100])
    _rehash(d, "per_anchor.parquet")  # hash matches, so only the read can fail


def _nan_eval_vectors(tmp_path, v, d):
    bad = v.copy()
    bad[0, 0] = np.nan
    np.save(d / "vectors.npy", bad)
    _rehash(d, "vectors.npy")


def _int_release_vectors(tmp_path, v, d):
    _release(tmp_path, np.zeros(v.shape, np.int8), run_id="enc")


def _truncated_release_vectors(tmp_path, v, d):
    r, _ = _release(tmp_path, v)
    (r / "vectors.npy").write_bytes((r / "vectors.npy").read_bytes()[:100])
    _edit_json(r / "release_meta.json",
               lambda m: m.update(vectors_sha256=sha256_file(r / "vectors.npy")))


def _release_without_meta(tmp_path, v, d):
    r, _ = _release(tmp_path, v, dataset_id="other")  # would fail, so must not vanish
    (r / "release_meta.json").unlink()


def _release_incomplete_manifest(tmp_path, v, d):
    r, b = _release(tmp_path, v)  # Sol r7: drop names.json and re-pin release.json
    (b / "names.json").unlink()
    _edit_json(b / "release.json", lambda m: m["files"].pop("names.json"))
    _edit_json(r / "release_meta.json",
               lambda m: m.update(release_json_sha256=sha256_file(b / "release.json")))


def _stray_eval_entry(tmp_path, v, d):
    (d.parent / "stray").write_text("", "utf-8")  # not a directory: still a record


def _empty_meta(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r.update(meta={}))  # Sol r8


def _no_max_length(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["meta"].pop("max_length"))


def _no_model_path(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["meta"].pop("model_path"))  # Sol r9
    _release(tmp_path, v)  # a pretrained release it would otherwise match


def _fake_recipe(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["meta"].update(recipe={"x": 1}))  # Sol r10


def _null_model_id(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["meta"]["recipe"].update(model_id=None))


def _wrong_dim(tmp_path, v, d):
    monkey = {**RECIPE, "dim": 5}  # a known recipe whose dim disagrees with the vectors
    baselines.enc.RECIPES["test"] = monkey  # key monkeypatched in _setup, so restored
    _edit_json(d / "summary.json", lambda r: r["meta"].update(recipe=monkey))


def _lexical_loose_types(tmp_path, v, d):
    from mtgemb.lexical import LEXICAL_RECIPE  # Sol r11: 1 for True, 3.0 for 3
    loose = {**json.loads(json.dumps(LEXICAL_RECIPE)), "lowercase": 1}
    _edit_json(d / "summary.json", lambda r: r["meta"].update(system="lexical", recipe=loose))


def _encoder_labelled_lexical(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["meta"].update(system="lexical"))
    _release(tmp_path, v)  # would match the mislabelled record before r10


def _empty_model_path(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["meta"].update(model_path=""))


def _release_false_model_path(tmp_path, v, d):
    r, _ = _release(tmp_path, v)
    _edit_json(r / "release_meta.json", lambda m: m.update(model_path=False))


def _release_bad_shard_size(tmp_path, v, d):
    r, b = _release(tmp_path, v)  # an exporter bug: written and pinned consistently
    _edit_json(b / "release.json", lambda m: m.update(shard_size=-1))
    _edit_json(r / "release_meta.json",
               lambda m: m.update(release_json_sha256=sha256_file(b / "release.json")))


def _release_encoder_pairs(tmp_path, v, d):
    r, _ = _release(tmp_path, v)  # Sol r12: dict(pairs) must not normalise it
    _edit_json(r / "release_meta.json",
               lambda m: m.update(encoder=[list(x) for x in m["encoder"].items()]))


def _drop_meta(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r.pop("meta"))


def _bad_model_path(tmp_path, v, d):
    _edit_json(d / "summary.json", lambda r: r["meta"].update(model_path=["x"]))


@pytest.mark.parametrize("tamper,problem", [
    (_tamper_vectors, "hash mismatch or missing: vectors.npy"),
    (_drop_summary, "no summary.json"),
    (_wrong_benchmark, "benchmark_id other"),
    (_wrong_summary, "recorded summary differs"),
    (_int_metric, "per-anchor metrics differ"),
    (_release_other_vectors, "vectors differ from evaluation enc"),
    (_release_unevaluated_encoder, "no recorded evaluation of this encoder"),
    (_release_wrong_dataset, "dataset_id other"),
    (_release_corrupt_bundle, "bundle file mismatch or missing: names.json"),
    (_release_wrong_release_hash, "release.json hash mismatch"),
    (_release_nan_vectors, "non-finite"),
    (_release_duplicate_id, "dataset_id other"),
    (_bool_in_summary, "recorded summary differs"),
    (_drop_per_anchor, "per_anchor.parquet missing"),
    (_bool_counts, "non-numeric columns: ['n_rel']"),
    (_release_missing_vectors, "vectors.npy hash mismatch or missing"),
    (_truncated_per_anchor, "replay error"),
    (_nan_eval_vectors, "vectors contain non-finite values"),
    (_int_release_vectors, "release vectors are not a float matrix"),
    (_truncated_release_vectors, "replay error"),
    (_bad_model_path, "incomplete metadata: encoder model_path"),
    (_drop_meta, "incomplete metadata: system and recipe"),
    (_release_without_meta, "no readable release_meta.json"),
    (_release_incomplete_manifest, "bundle manifest incomplete"),
    (_stray_eval_entry, "no summary.json"),
    (_empty_meta, "incomplete metadata: system and recipe"),
    (_no_max_length, "incomplete metadata: encoder max_length"),
    (_no_model_path, "incomplete metadata: encoder model_path"),
    (_fake_recipe, "encoder recipe is not a known"),
    (_null_model_id, "encoder recipe is not a known"),
    (_wrong_dim, "recipe dim 5 != vector width 4"),
    (_lexical_loose_types, "lexical recipe differs"),
    (_encoder_labelled_lexical, "lexical recipe differs from the refit recipe"),
    (_empty_model_path, "incomplete metadata: encoder model_path"),
    (_release_false_model_path, "release model_path must be null"),
    (_release_bad_shard_size, "is not a positive integer"),
    (_release_encoder_pairs, "release encoder must be a JSON object"),
])
def test_replay_fails_on_each_tampering(tmp_path, monkeypatch, tamper, problem):
    v, d = _setup(tmp_path, monkeypatch)
    tamper(tmp_path, v, d)
    r = baselines.replay_evals("ds")
    probs = [p for x in [*r["evals"].values(), *r["releases"].values()] for p in x["problems"]]
    assert not r["passed"] and any(problem in p for p in probs), probs


def test_replay_with_no_evaluations_does_not_pass(tmp_path, monkeypatch):
    _, d = _setup(tmp_path, monkeypatch)
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    assert not baselines.replay_evals("ds")["passed"]
