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


def _v3_release(tmp_path, v, missing=None):
    art, bundle = _release(tmp_path, v)
    man = json.loads((bundle / "release.json").read_text("utf-8"))
    man.update(schema="mtgemb-release-v3", shard_size=(man["n_cards"] + 1) // 2)
    for name in ("cards/001.json", "neighbours/001.json",
                 "abilities/000.json", "abilities/001.json"):
        path = bundle / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("[]", "utf-8")
        if name != missing:
            man["files"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    (bundle / "release.json").write_text(json.dumps(man), "utf-8")
    np.save(art / "ability_vectors.npy", v)
    (art / "ability_lines.json").write_text("[]", "utf-8")
    _edit_json(art / "release_meta.json", lambda m: m.update(
        release_json_sha256=sha256_file(bundle / "release.json"),
        ability_vectors_sha256=sha256_file(art / "ability_vectors.npy"),
        ability_lines_sha256=sha256_file(art / "ability_lines.json")))
    return art


@pytest.mark.parametrize("missing", [None, "abilities/000.json", "abilities/001.json"])
def test_replay_v3_requires_every_ability_shard(tmp_path, monkeypatch, missing):
    v, _ = _setup(tmp_path, monkeypatch)
    _v3_release(tmp_path, v, missing)
    result = baselines.replay_evals("ds")
    assert result["passed"] == (missing is None)
    if missing:
        assert any("bundle manifest incomplete" in p and missing in p
                   for p in result["releases"]["r1"]["problems"])


@pytest.mark.parametrize("fault", ["vectors_tampered", "lines_missing", "hash_absent"])
def test_replay_v3_verifies_ability_artifacts(tmp_path, monkeypatch, fault):
    v, _ = _setup(tmp_path, monkeypatch)
    art = _v3_release(tmp_path, v)
    if fault == "vectors_tampered":
        np.save(art / "ability_vectors.npy", v[::-1].copy())
    elif fault == "lines_missing":
        (art / "ability_lines.json").unlink()
    else:
        _edit_json(art / "release_meta.json", lambda m: m.pop("ability_vectors_sha256"))
    result = baselines.replay_evals("ds")
    assert not result["passed"]
    name = "ability_lines.json" if fault == "lines_missing" else "ability_vectors.npy"
    assert f"{name} hash mismatch or missing" in result["releases"]["r1"]["problems"]


@pytest.mark.parametrize("fault,problem", [
    (None, None),
    ("tag_range", "tag indices invalid or out of range"),
    ("ability_subset", "ability tags are not a subset"),
    ("line_self", "invalid line neighbour index"),
    ("line_range", "invalid line neighbour index"),
    ("line_order", "invalid line neighbour index"),
    ("line_long", "invalid line neighbour list length"),
    ("line_textless", "invalid line neighbour index"),
    ("line_nan", "invalid line neighbour index"),
    ("line_inf", "invalid line neighbour index"),
    ("line_duplicate", "invalid line neighbour index"),
    ("textless_tags", "textless card has tags"),
    ("textless_lines", "textless card has tags"),
    ("line_k_zero", "invalid line_k"),
    ("line_k_bool", "invalid line_k"),
    ("ability_k_wrong", "invalid line_k"),
    ("ability_k_missing", "invalid line_k"),
    *[(f"{key}_{fault}", f"invalid {key}")
      for key in ("line_status", "tags_status", "abilities_status")
      for fault in ("missing", "empty", "wrong", "type")],
    ("missing_line", "bundle manifest incomplete"),
    ("tags_hash", "bundle file mismatch or missing: tags.json"),
])
def test_replay_v4_checks_tags_and_line_neighbours(tmp_path, monkeypatch, fault, problem):
    v, _ = _setup(tmp_path, monkeypatch)
    art = _v3_release(tmp_path, v)
    bundle = tmp_path / "bundle-r1"
    man = json.loads((bundle / "release.json").read_text("utf-8"))
    man.update(schema="mtgemb-release-v4", line_k=50, ability_k=50,
               line_status="exploratory: line match", tags_status="exploratory: tags",
               abilities_status="exploratory: abilities")
    payloads = {"tags.json": {"labels": ["first", "second"]}}
    n, ss = man["n_cards"], man["shard_size"]
    for start in range(0, n, ss):
        name = f"{start // ss:03d}.json"
        size = min(ss, n - start)
        payloads[f"cards/{name}"] = [{"tags": [0], "has_text": True} for _ in range(size)]
        payloads[f"abilities/{name}"] = [[{"tags": [0]}] for _ in range(size)]
        payloads[f"lines/{name}"] = [[] for _ in range(size)]
    # A textless card in the second shard must never be a line-match target.
    payloads["cards/001.json"][-1] = {"tags": [], "has_text": False}
    payloads["abilities/001.json"][-1] = []
    payloads["lines/000.json"][0] = [[1, 0.8], [2, 0.5]]
    # Mutations are rehashed, so semantic checks must catch them.
    if fault == "tag_range":
        payloads["cards/000.json"][0]["tags"] = [2]
    elif fault == "ability_subset":
        payloads["abilities/000.json"][0][0]["tags"] = [1]
    elif fault == "line_self":
        payloads["lines/000.json"][0] = [[0, 0.5]]
    elif fault == "line_range":
        payloads["lines/000.json"][0] = [[n, 0.5]]
    elif fault == "line_order":
        payloads["lines/000.json"][0] = [[1, 0.2], [2, 0.8]]
    elif fault == "line_long":
        payloads["lines/000.json"][0] = [[1, 0.8]] * 51
    elif fault == "line_textless":
        payloads["lines/000.json"][0] = [[n - 1, 0.8]]
    elif fault in ("line_nan", "line_inf"):
        payloads["lines/000.json"][0] = [[1, float(fault.removeprefix("line_"))]]
    elif fault == "line_duplicate":
        payloads["lines/000.json"][0] = [[1, 0.8], [1, 0.5]]
    elif fault == "textless_tags":
        payloads["cards/001.json"][-1]["tags"] = [0]
    elif fault == "textless_lines":
        payloads["lines/001.json"][-1] = [[0, 0.5]]
    elif fault == "line_k_zero":
        man["line_k"] = 0
        for name, rows in payloads.items():
            if name.startswith("lines/"):
                rows[:] = [[] for _ in rows]
    elif fault == "line_k_bool":
        man["line_k"] = True
    elif fault == "ability_k_wrong":
        man["ability_k"] = 49
    elif fault == "ability_k_missing":
        man.pop("ability_k")
    elif fault and "_status_" in fault:
        key, value = fault.rsplit("_", 1)
        if value == "missing":
            man.pop(key)
        else:
            man[key] = {"empty": "", "wrong": "validated", "type": 1}[value]
    for name, value in payloads.items():
        if fault == "missing_line" and name == "lines/001.json":
            continue
        path = bundle / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value), "utf-8")
        man["files"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if fault == "tags_hash":
        (bundle / "tags.json").write_text('{"labels": ["changed"]}', "utf-8")
    (bundle / "release.json").write_text(json.dumps(man), "utf-8")
    _edit_json(art / "release_meta.json", lambda m: m.update(
        release_json_sha256=sha256_file(bundle / "release.json")))
    result = baselines.replay_evals("ds")
    assert result["passed"] == (fault is None)
    if fault is not None:
        assert any(problem in p for p in result["releases"]["r1"]["problems"])


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


def test_same_weights_refuses_a_changed_encoder(tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "benchmark_v3.json").write_text(json.dumps({"benchmark_id": "B",
                                                      "universe": ["b", "c"]}), "utf-8")
    monkeypatch.setattr(baselines, "RUNS", tmp_path / "runs")
    d = tmp_path / "runs" / "eval" / "B" / "enc"
    d.mkdir(parents=True)
    np.save(d / "vectors.npy", np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32))
    (d / "summary.json").write_text(json.dumps(
        {"files": {"vectors.npy": sha256_file(d / "vectors.npy")}}), "utf-8")
    v = np.array([[0.0, 1.0], [2.0, 0.0], [0.6, 0.8]])  # sealed universe a, b, c; scale-free
    assert baselines.same_weights(v, ["a", "b", "c"], ds, "enc") == pytest.approx(1.0)
    v[2] = [0.8, 0.6]
    with pytest.raises(ValueError, match="differ"):
        baselines.same_weights(v, ["a", "b", "c"], ds, "enc")
    np.save(d / "vectors.npy", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="changed"):
        baselines.same_weights(v, ["a", "b", "c"], ds, "enc")


def _v6_release(tmp_path, v, ver=6, fault=None):
    """A v5/v6 bundle whose stored rankings are computed here, independently, by brute force:
    whole-card neighbours by cosine, face match as the best of a candidate's whole card and
    its own faces against the query's whole card."""
    from mtgemb.text import FACE_BOUNDARY

    art = _v3_release(tmp_path, v)
    bundle = tmp_path / "bundle-r1"
    ds = baselines.load_benchmark("ds")[1]
    two = f"draw a card{FACE_BOUNDARY}scry 1"  # c2 gets two faces
    ds["cards"] = ds["cards"].assign(text=ds["cards"]["text"].where(
        ds["cards"]["oracle_id"] != "c2", two))
    cards = ds["cards"].sort_values("oracle_id").reset_index(drop=True)
    n = len(cards)
    c2 = int(cards.index[cards["oracle_id"] == "c2"][0])
    F = np.random.default_rng(1).normal(size=(2, 4))
    F = (F / np.linalg.norm(F, axis=1, keepdims=True)).astype(np.float32)
    W = v.astype(np.float64)

    def ranked(r, faces):
        score = {j: float(W[r] @ W[j]) for j in range(n) if j != r}
        if faces and r != c2:
            score[c2] = max(score[c2], *(float(W[r] @ f) for f in F.astype(np.float64)))
        order = sorted(score, key=lambda j: (-score[j], j))
        return [[j, round(score[j], 4)] for j in order]

    man = json.loads((bundle / "release.json").read_text("utf-8"))
    man.update(schema=f"mtgemb-release-v{ver}", k=200, line_k=200, ability_k=50,
               line_status="exploratory: line match", tags_status="exploratory: tags",
               abilities_status="exploratory: abilities")
    if ver == 6:
        man.update(face_k=200, face_status="exploratory: face match")
    payloads = {"tags.json": {"labels": ["first"]}}
    ss = man["shard_size"]
    for start in range(0, n, ss):
        name, rows = f"{start // ss:03d}.json", range(start, min(n, start + ss))
        payloads[f"cards/{name}"] = [{"tags": [0], "has_text": True, "colors": "WU",
                                      "image_back": None} for _ in rows]
        payloads[f"abilities/{name}"] = [[{"tags": [0]}] for _ in rows]
        payloads[f"lines/{name}"] = [[] for _ in rows]
        payloads[f"neighbours/{name}"] = [ranked(r, False) for r in rows]
        if ver == 6:
            payloads[f"faces/{name}"] = [ranked(r, True) for r in rows]
    payloads["cards/000.json"][0]["image_back"] = "https://example.org/back.jpg"
    whole, face = payloads["neighbours/000.json"][0], payloads.get("faces/000.json", [[]])[0]
    if fault == "neighbour_score":
        whole[-1][1] = round(whole[-1][1] - 0.01, 4)
    elif fault == "neighbour_swap":
        whole[0][0], whole[1][0] = whole[1][0], whole[0][0]
    elif fault == "neighbour_self":
        whole[0][0] = 0
    elif fault == "neighbour_short":
        whole.pop()
    elif fault == "face_score":
        face[-1][1] = round(face[-1][1] - 0.01, 4)
    elif fault == "face_as_whole":
        payloads["faces/000.json"][0] = [list(p) for p in whole]
    elif fault == "face_order":
        face[0], face[1] = face[1], face[0]
    elif fault == "colors_order":
        payloads["cards/000.json"][0]["colors"] = "UW"
    elif fault == "colors_letter":
        payloads["cards/000.json"][0]["colors"] = "X"
    elif fault == "image_back_type":
        payloads["cards/000.json"][0]["image_back"] = 1
    elif fault == "k_wrong":
        man["k"] = 50
    elif fault == "face_k_wrong":
        man["face_k"] = 50
    elif fault == "face_status_missing":
        man.pop("face_status")
    for name, value in payloads.items():
        if fault == "missing_face_shard" and name == "faces/001.json":
            continue
        path = bundle / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value), "utf-8")
        man["files"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    (bundle / "release.json").write_text(json.dumps(man), "utf-8")
    meta = {"release_json_sha256": sha256_file(bundle / "release.json")}
    if ver == 6:
        np.save(art / "face_vectors.npy", 2 * F if fault == "face_vectors_norm" else F)
        meta.update(face_vectors_sha256=sha256_file(art / "face_vectors.npy"), n_faces=2)
        if fault == "face_vectors_tampered":
            np.save(art / "face_vectors.npy", F[::-1].copy())
        elif fault == "n_faces_wrong":
            meta["n_faces"] = 3
    _edit_json(art / "release_meta.json", lambda m: m.update(meta))


@pytest.mark.parametrize("ver,fault,problem", [
    (6, None, None),
    (5, None, None),
    (5, "neighbour_score", "neighbours scores of card"),
    (6, "neighbour_score", "neighbours scores of card"),
    (6, "neighbour_swap", "neighbours scores of card"),
    (6, "neighbour_self", "invalid neighbours index"),
    (6, "neighbour_short", "neighbours scores of card"),
    (6, "face_score", "faces scores of card"),
    (6, "face_as_whole", "faces scores of card"),
    (6, "face_order", "invalid faces index"),
    (6, "colors_order", "invalid colors"),
    (6, "colors_letter", "invalid colors"),
    (6, "image_back_type", "invalid image_back"),
    (5, "k_wrong", "invalid line_k"),
    (6, "face_k_wrong", "invalid line_k"),
    (6, "face_status_missing", "invalid face_status"),
    (6, "missing_face_shard", "bundle manifest incomplete"),
    (6, "face_vectors_tampered", "face_vectors.npy hash mismatch or missing"),
    (6, "face_vectors_norm", "face vectors"),
    (6, "n_faces_wrong", "face vectors"),
])
def test_replay_v5_v6_checks_rankings_against_release_vectors(tmp_path, monkeypatch, ver,
                                                              fault, problem):
    v, _ = _setup(tmp_path, monkeypatch)
    _v6_release(tmp_path, v, ver, fault)
    result = baselines.replay_evals("ds")
    probs = result["releases"]["r1"]["problems"]
    assert result["passed"] == (fault is None), probs
    if fault is not None:
        assert any(problem in p for p in probs), probs
