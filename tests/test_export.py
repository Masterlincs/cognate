import hashlib
import json

import numpy as np
import pytest

from mtgemb.export import (
    ability_neighbours,
    finish_interrupted,
    ranking_parity,
    reference_check,
    release_digest,
    topk_neighbours,
)
from mtgemb.util import sha256_file, sha256_json


def test_topk_excludes_self_breaks_ties_by_index_and_blocks_agree():
    v = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8], [1, 0]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx, sco = topk_neighbours(v, k=3, block=2)
    assert idx[0].tolist() == [2, 4, 3]  # exact ties 2, 4 by index, then cos 0.6
    assert idx[1].tolist() == [3, 0, 2]  # cos 0.8, then tied zeros by index
    assert np.allclose(sco[0], [1, 1, 0.6])
    assert all(i not in row for i, row in enumerate(idx.tolist()))
    big = np.random.default_rng(0).normal(size=(300, 8)).astype(np.float32)
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    a, _ = topk_neighbours(big, k=10, block=7)
    b, _ = topk_neighbours(big, k=10, block=300)
    assert (a == b).all()
    _, bad, exact = reference_check(big, a, np.arange(20))
    assert bad == [] and exact == 20


def test_face_topk_scores_candidates_by_their_best_face():
    # card 1 is creature // bolt; its whole vector is diluted, its bolt face is not
    v = np.array([[1, 0], [0.6, 0.8], [0, 1], [0.8, 0.6]], dtype=np.float32)
    F = np.array([[0, 1], [1, 0]], dtype=np.float32)  # card 1's faces
    owner = np.array([1, 1])
    whole, _ = topk_neighbours(v, k=3)
    assert whole[0].tolist() == [3, 1, 2]
    idx, sco = topk_neighbours(v, k=3, block=1, faces=(F, owner))
    assert idx[0].tolist() == [1, 3, 2] and np.allclose(sco[0], [1.0, 0.8, 0.0])
    # the query keeps its whole vector: card 1 ranks as [0.6, 0.8] does
    assert idx[1].tolist() == [3, 2, 0] and np.allclose(sco[1], [0.96, 0.8, 0.6])
    assert idx[2].tolist() == [1, 3, 0]  # card 1's creature face matches card 2 exactly
    _, bad, exact = reference_check(v, idx, np.arange(4), faces=(F, owner))
    assert bad == [] and exact == 4
    _, bad, _ = reference_check(v, whole, np.arange(4), faces=(F, owner))
    assert bad == [0]  # card 0's whole-card list is wrong under face match (card 2's is the same)


def test_ranking_parity_counts_top10_overlap():
    v = np.random.default_rng(1).normal(size=(40, 6)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    nidx, _ = topk_neighbours(v, k=10)
    rows = np.array([0, 5])
    assert ranking_parity(v[rows], v, rows, nidx).tolist() == [1.0, 1.0]
    bad_q = -v[rows]  # a CPU encoder that disagrees: picks the device's least similar cards
    assert (ranking_parity(bad_q, v, rows, nidx) < 0.5).all()
    # fewer than k other cards: identical rankings still score 1
    small = v[:5]
    sidx, _ = topk_neighbours(small, k=10)
    assert ranking_parity(small[:1], small, np.array([0]), sidx).tolist() == [1.0]
    one = v[:1]
    assert ranking_parity(one, one, np.array([0]), np.empty((1, 0), np.int64)).tolist() == [1.0]


def test_ranking_parity_tolerates_near_tie_at_cutoff():
    # anchor e0; cards 1..3 score 0.9, 0.5, 0.5 - 1e-6: with k=2, a CPU query that ranks
    # card 3 above card 2 is a near-tie swap, not a disagreement
    def unit(c):
        return np.array([c, np.sqrt(1 - c * c), 0.0])
    v = np.stack([[1.0, 0, 0], unit(0.9), unit(0.5), unit(0.5 - 1e-6)])
    nidx, _ = topk_neighbours(v, k=2)
    assert nidx[0].tolist() == [1, 2]
    q = np.array([[1.0, 0, 0]])
    swapped = nidx.copy()
    swapped[0] = [1, 3]  # pretend the device chose 3; CPU picks 2, within tol of the cut
    assert ranking_parity(q, v, np.array([0]), swapped, k=2).tolist() == [1.0]


def test_release_digest_binds_catalogue_and_encoder():
    base = ("ds", ["a", "b"], {"model_id": "m", "max_length": 94}, "mh", "vh")
    d = release_digest(*base)
    assert d == release_digest(*base)
    assert d != release_digest("ds2", *base[1:])
    assert d != release_digest("ds", ["b", "a"], *base[2:])
    assert d != release_digest("ds", base[1], {"model_id": "m", "max_length": 92}, *base[3:])
    # a bundle layout change is a new release, never the old id with new files
    assert d != release_digest(*base, schema="mtgemb-release-v1")
    # Omitting the optional ability hash preserves the original v2 digest formula.
    old = sha256_json(["mtgemb-release-v2", "ds", sha256_json(base[1]), *base[2:]])[:8]
    assert release_digest(*base, schema="mtgemb-release-v2") == old
    old_v3 = sha256_json(["mtgemb-release-v3", "ds", sha256_json(base[1]), *base[2:], "av"])[:8]
    assert release_digest(*base, schema="mtgemb-release-v3", ability_vec_hash="av") == old_v3
    assert release_digest(*base, ability_vec_hash="av", line_hash="lh", tags_hash="th") != old_v3
    assert release_digest(*base, line_hash="a") != release_digest(*base, line_hash="b")
    assert release_digest(*base, tags_hash="a") != release_digest(*base, tags_hash="b")
    assert release_digest(*base, ability_vec_hash="a") != release_digest(
        *base, ability_vec_hash="b")


def test_ability_neighbours_hand_worked_maxsim_and_single_line_candidates():
    vectors = {"A": [1, 0], "B": [0, 1], "C": [0.6, 0.8], "D": [1, 0]}

    def encode(lines, *, cpu):
        return np.array([vectors[line] for line in lines], dtype=np.float32)

    result = ability_neighbours(["A\n//\nB", "C", "D"], encode)
    assert result["lines"] == ["A", "B", "C", "D"]
    assert result["owner"].tolist() == [0, 0, 1, 2]
    assert result["rows"] == [[
        {"text": "A", "neighbours": [[2, 1.0], [1, 0.6]]},
        {"text": "B", "neighbours": [[1, 0.8], [2, 0.0]]}], [], []]
    assert result["parity"]["topk_reference_rows"] == 2  # all available query lines
    assert result["parity"]["cpu_vs_device_min_cos"] == pytest.approx(1)


def test_ability_neighbours_checks_at_least_twenty_query_lines():
    def encode(lines, *, cpu):
        return np.tile(np.array([[1.0, 0.0]], np.float32), (len(lines), 1))

    result = ability_neighbours(["A\nB"] * 15, encode, ref_n=1)
    assert result["parity"]["topk_reference_rows"] == 20
    assert all(own not in [j for j, _ in a["neighbours"]]
               for own, row in enumerate(result["rows"]) for a in row)
    assert ability_neighbours(["A"], encode)["rows"] == [[]]


def test_ability_neighbours_rejects_parity_failure():
    def encode(lines, *, cpu):
        return np.tile(np.array([[1.0, 0.0] if cpu else [0.0, 1.0]], np.float32),
                       (len(lines), 1))

    with pytest.raises(ValueError, match="ability device/CPU parity failed"):
        ability_neighbours(["A\nB", "C"], encode)


def test_ability_neighbours_rejects_incorrect_block_ranking(monkeypatch):
    import mtgemb.export as exporter

    def wrong(q, q_owner, line_v, starts, k):
        return np.array([[2, 1], [1, 2]]), np.zeros((2, 2))

    monkeypatch.setattr(exporter, "maxsim_topk", wrong)
    values = {"A": [1, 0], "B": [0, 1], "C": [1, 0], "D": [0, 1]}
    with pytest.raises(ValueError, match="ability block top-k disagrees"):
        ability_neighbours(["A\nB", "C", "D"], lambda lines, **_: np.array(
            [values[line] for line in lines], np.float32))


def test_export_v5_shards_alignment_mapping_and_artifacts(tmp_path, monkeypatch):
    from contextlib import nullcontext

    import pandas as pd
    import torch

    import mtgemb.export as exporter

    # Textless cards before/between text rows expose accidental text-row bundle indices.
    texts = ["", "A\n//\nB", "", "C", "D", "E"]
    cards = pd.DataFrame([
        {"oracle_id": str(i), "split": "dev" if i == 1 else "fit",
         "text_status": "ok" if text else "empty",
         "text": text, "name": str(i), "faces": "[]", "layout": "normal",
         "color_identity": "", "mv": 0, "type_line": "", "is_creature": False,
         "image": "", "scryfall_uri": ""} for i, text in enumerate(texts)])
    monkeypatch.setattr(exporter, "load_dataset", lambda _: {
        "cards": cards, "ident": {"dataset_id": "test", "snapshot": "test"},
        "memberships": pd.DataFrame({"oracle_id": ["1", "3", "4", "5", "4"],
                                     "tag_id": ["t", "t", "t", "t", "a"]}),
        "tags": pd.DataFrame({"tag_id": ["t", "a"], "label": ["test tag", "other"],
                              "functional": [True, True]})})
    monkeypatch.setattr(exporter, "DeviceLock", nullcontext)
    monkeypatch.setattr(exporter, "display_fields", lambda _dir, ids, _sha: {
        i: {"colors": "R" if i == "3" else "", "image_back": "b.jpg" if i == "1" else None}
        for i in ids})
    monkeypatch.setattr(exporter, "ARTIFACTS", tmp_path / "art")
    monkeypatch.setattr(exporter, "SHARD", 2)
    monkeypatch.setattr(exporter, "log_event", lambda *a, **kw: None)
    monkeypatch.setattr(exporter, "git_identity", lambda: {})
    monkeypatch.setattr(exporter.enc, "load", lambda *a: (None, torch.nn.Identity()))
    monkeypatch.setattr(exporter.enc, "device", lambda: torch.device("cpu"))
    monkeypatch.setitem(exporter.enc.RECIPES, "e5-base-v2",
                        {**exporter.enc.RECIPES["e5-base-v2"], "dim": 2})
    vectors = {"A": [1, 0], "B": [0, 1], "A\n//\nB": [1, 0],
               "C": [0.6, 0.8], "D": [1, 0], "E": [1, 0]}
    monkeypatch.setattr(exporter.enc, "encode", lambda model, tok, lines, *a, **kw: np.array(
        [vectors[line] for line in lines], np.float32))
    result = exporter.export("unused", "e5-base-v2", 94, "fake", "experimental",
                             web_dir=tmp_path / "bundle")
    from pathlib import Path

    bundle, art = Path(result["bundle"]), Path(result["artifacts"])
    release = json.loads((bundle / "release.json").read_text("utf-8"))
    assert release["schema"] == "mtgemb-release-v6" and release["ability_k"] == 50
    assert release["k"] == release["line_k"] == release["face_k"] == 200
    assert release["face_status"].endswith("")
    assert release["abilities_status"] == "exploratory: not evaluated on the tag proxy"
    expected = [[], [{"text": "A", "tags": [1], "neighbours": [[4, 1.0], [5, 1.0], [3, 0.6]]},
                     {"text": "B", "tags": [], "neighbours": [[3, 0.8], [4, 0.0], [5, 0.0]]}],
                [], [], [], []]
    rows = []
    for i, length in enumerate([2, 2, 2]):
        name = f"abilities/{i:03d}.json"
        path = bundle / name
        shard = json.loads(path.read_text("utf-8"))
        assert len(shard) == length
        assert release["files"][name] == {"sha256": sha256_file(path),
                                          "bytes": path.stat().st_size}
        rows += shard
    assert rows == expected
    assert not (tmp_path / "bundle" / "data" / "current.json").exists()
    lines = json.loads((art / "ability_lines.json").read_text("utf-8"))
    assert lines == [{"text": "A", "oracle_id": "1"}, {"text": "B", "oracle_id": "1"},
                     {"text": "C", "oracle_id": "3"}, {"text": "D", "oracle_id": "4"},
                     {"text": "E", "oracle_id": "5"}]
    assert np.array_equal(np.load(art / "ability_vectors.npy"),
                          np.array([[1, 0], [0, 1], [0.6, 0.8], [1, 0], [1, 0]], np.float32))
    meta = json.loads((art / "release_meta.json").read_text("utf-8"))
    assert meta["n_ability_lines"] == 5 and meta["n_ability_queries"] == 2
    for name in ("ability_vectors", "ability_lines"):
        suffix = ".npy" if name == "ability_vectors" else ".json"
        assert meta[name + "_sha256"] == sha256_file(art / (name + suffix))
    assert meta["ability_parity"]["topk_reference_failures"] == 0

    assert release["line_status"].endswith("dev tag proxy")
    assert release["tags_status"].endswith("")
    assert json.loads((bundle / "tags.json").read_text("utf-8")) == {
        "labels": ["other", "test tag"]}
    records, line_rows = [], []
    for i in range(3):
        records += json.loads((bundle / f"cards/{i:03d}.json").read_text("utf-8"))
        name = f"lines/{i:03d}.json"
        path = bundle / name
        line_rows += json.loads(path.read_text("utf-8"))
        assert release["files"][name] == {"sha256": sha256_file(path),
                                          "bytes": path.stat().st_size}
    assert [c["tags"] for c in records] == [[], [1], [], [1], [0, 1], [1]]
    assert [c["colors"] for c in records] == ["", "", "", "R", "", ""]
    assert [c["image_back"] for c in records] == [None, "b.jpg", None, None, None, None]
    assert line_rows[0] == line_rows[2] == []
    assert line_rows[1] == [[3, 0.7], [4, 0.5], [5, 0.5]]
    assert all(i != j for i, row in enumerate(line_rows) for j, _ in row)
    assert all(row == sorted(row, key=lambda x: -x[1]) for row in line_rows)
    assert release["files"]["tags.json"]["sha256"] == sha256_file(bundle / "tags.json")
    face_rows, whole_rows = [], []
    for i in range(3):
        name = f"faces/{i:03d}.json"
        face_rows += json.loads((bundle / name).read_text("utf-8"))
        whole_rows += json.loads((bundle / f"neighbours/{i:03d}.json").read_text("utf-8"))
        assert release["files"][name]["sha256"] == sha256_file(bundle / name)
    # card 3 (C, [0.6, 0.8]) meets card 1's face B ([0, 1]) at 0.8; whole card 1 is 0.6
    assert whole_rows[3] == [[1, 0.6], [4, 0.6], [5, 0.6]]
    assert face_rows[3] == [[1, 0.8], [4, 0.6], [5, 0.6]]
    assert face_rows[1] == whole_rows[1] == [[4, 1.0], [5, 1.0], [3, 0.6]]
    assert face_rows[0] == face_rows[2] == []
    assert np.array_equal(np.load(art / "face_vectors.npy"), np.array([[1, 0], [0, 1]], np.float32))
    assert meta["n_faces"] == 2 and meta["face_vectors_sha256"] == sha256_file(
        art / "face_vectors.npy")

    # Reuse a v3-shaped source with the exact pinned vectors; encoding must never run.
    import shutil
    reuse_root = tmp_path / "reuse-art"
    source = reuse_root / "releases" / "source-v3"
    shutil.copytree(art, source)
    monkeypatch.setattr(exporter.enc, "load", lambda *a: pytest.fail("reuse loaded a model"))
    monkeypatch.setattr(exporter.enc, "encode", lambda *a, **kw: pytest.fail("reuse encoded"))
    monkeypatch.setattr(exporter, "ARTIFACTS", reuse_root)
    reused = exporter.export("unused", "e5-base-v2", 94, "fake", "experimental",
                             web_dir=tmp_path / "reuse", from_release="source-v3")
    reused_art = Path(reused["artifacts"])
    reused_meta = json.loads((reused_art / "release_meta.json").read_text("utf-8"))
    assert reused_meta["vectors_from"] == "source-v3"
    for key in ("parity", "ability_parity"):
        assert reused_meta[key] == meta[key]
    for filename in ("vectors.npy", "ability_vectors.npy", "ability_lines.json"):
        assert (reused_art / filename).read_bytes() == (source / filename).read_bytes()
    source_meta = json.loads((source / "release_meta.json").read_text("utf-8"))
    source_meta["model_hash"] = "wrong"
    (source / "release_meta.json").write_text(json.dumps(source_meta), "utf-8")
    with pytest.raises(ValueError, match="model_hash mismatch"):
        exporter.export("unused", "e5-base-v2", 94, "fake", "experimental",
                        from_release="source-v3")
    source_meta["model_hash"] = meta["model_hash"]
    (source / "release_meta.json").write_text(json.dumps(source_meta), "utf-8")
    (source / "vectors.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="vectors.npy hash mismatch"):
        exporter.export("unused", "e5-base-v2", 94, "fake", "experimental",
                        from_release="source-v3")

    # A pre-v6 source has no face vectors: only the face texts are encoded, on the CPU.
    old = tmp_path / "reuse5-art" / "releases" / "source-v5"
    shutil.copytree(art, old)
    monkeypatch.setattr(exporter, "ARTIFACTS", tmp_path / "reuse5-art")
    (old / "face_vectors.npy").unlink()
    old_meta = json.loads((old / "release_meta.json").read_text("utf-8"))
    del old_meta["face_vectors_sha256"]
    (old / "release_meta.json").write_text(json.dumps(old_meta), "utf-8")
    encoded = []
    monkeypatch.setattr(exporter.enc, "load", lambda *a: (None, torch.nn.Identity()))
    monkeypatch.setattr(exporter.enc, "encode", lambda model, tok, lines, *a, dev, **kw: (
        encoded.append((lines, dev.type)) or np.array([vectors[t] for t in lines], np.float32)))
    v5 = Path(exporter.export("unused", "e5-base-v2", 94, "fake", "experimental",
                              web_dir=tmp_path / "reuse5", from_release="source-v5")["bundle"])
    assert encoded == [(["A", "B"], "cpu")]
    assert [json.loads((v5 / f"faces/{i:03d}.json").read_text("utf-8")) for i in range(3)] == [
        face_rows[s:s + 2] for s in (0, 2, 4)]


@pytest.mark.parametrize("fault", [None, "dataset_id", "model_hash", "recipe", "max_length",
                                   "oracle_id", "line_text", "line_hash", "line_vectors_hash",
                                   "wrong_width", "whole_width", "line_width", "whole_norm",
                                   "line_norm", "zero_whole", "zero_line", "near_unit"])
def test_reuse_rejects_incompatible_source(tmp_path, monkeypatch, fault):
    import mtgemb.export as exporter

    monkeypatch.setattr(exporter, "ARTIFACTS", tmp_path)
    root = tmp_path / "releases" / "source"
    root.mkdir(parents=True)
    v = np.array([[1.0, 0.0]], np.float32)
    lv = v.copy()
    if fault in ("wrong_width", "whole_width"):
        v = np.array([[1.0, 0.0, 0.0]], np.float32)
    if fault in ("wrong_width", "line_width"):
        lv = np.array([[1.0, 0.0, 0.0]], np.float32)
    if fault == "whole_norm":
        v *= 1.002
    elif fault == "line_norm":
        lv *= 0.998
    elif fault == "zero_whole":
        v *= 0
    elif fault == "zero_line":
        lv *= 0
    elif fault == "near_unit":
        v *= 1.0005
        lv *= 0.9995
    np.save(root / "vectors.npy", v)
    np.save(root / "ability_vectors.npy", lv)
    lines = [{"text": "A", "oracle_id": "a"}]
    if fault == "oracle_id":
        lines[0]["oracle_id"] = "other"
    elif fault == "line_text":
        lines[0]["text"] = "B"
    (root / "ability_lines.json").write_text(json.dumps(lines), "utf-8")
    meta = {"dataset_id": "ds", "model_hash": "model",
            "encoder": {"model_id": "m", "max_length": 94, "dim": 2},
            "vectors_sha256": sha256_file(root / "vectors.npy"),
            "ability_vectors_sha256": sha256_file(root / "ability_vectors.npy"),
            "ability_lines_sha256": sha256_file(root / "ability_lines.json")}
    if fault in ("dataset_id", "model_hash"):
        meta[fault] = "other"
    elif fault == "recipe":
        meta["encoder"]["model_id"] = "other"
    elif fault == "max_length":
        meta["encoder"]["max_length"] = 95
    elif fault == "line_hash":
        meta["ability_lines_sha256"] = "other"
    elif fault == "line_vectors_hash":
        meta["ability_vectors_sha256"] = "other"
    (root / "release_meta.json").write_text(json.dumps(meta), "utf-8")
    encoder = {"model_id": "m", "max_length": 94, "dim": 2}
    if fault in (None, "near_unit"):
        actual_v, actual_lv, _ = exporter._reuse_vectors(
            "source", "ds", "model", encoder, ["A"], ["a"])
        np.testing.assert_array_equal(actual_v, v)
        np.testing.assert_array_equal(actual_lv, lv)
        return
    problem = "unit-norm" if fault in (
        "whole_norm", "line_norm", "zero_whole", "zero_line") else "mismatch"
    with pytest.raises(ValueError, match=problem):
        exporter._reuse_vectors("source", "ds", "model", encoder,
                                ["A"], ["a"])


def test_export_cli_forwards_from_release(monkeypatch, capsys):
    import mtgemb.cli as cli
    import mtgemb.export as exporter

    received = {}

    def fake_export(*args, **kwargs):
        received.update(kwargs)
        return {}

    monkeypatch.setattr(exporter, "export", fake_export)
    cli.main(["export", "e5-base-v2", "--max-length", "94", "--label", "fake",
              "--status", "experimental", "--from-release", "source-v3"])
    assert received == {"from_release": "source-v3", "decision": None}
    assert capsys.readouterr().out.strip() == "{}"


def test_finish_interrupted_adopts_only_matching_artifacts(tmp_path):
    out, art, final_art = tmp_path / "out", tmp_path / "art.partial", tmp_path / "art"
    out.mkdir()
    (out / "release.json").write_text("{}", "utf-8")
    art.mkdir()
    (art / "release_meta.json").write_text(json.dumps({"release_json_sha256": "x"}), "utf-8")
    with pytest.raises(FileExistsError):
        finish_interrupted(out, art, final_art)
    good = hashlib.sha256(b"{}").hexdigest()
    (art / "release_meta.json").write_text(json.dumps({"release_json_sha256": good}), "utf-8")
    finish_interrupted(out, art, final_art)
    assert (final_art / "release_meta.json").exists() and not art.exists()
    with pytest.raises(FileExistsError):  # no partial artifacts left
        finish_interrupted(out, tmp_path / "missing", tmp_path / "art2")


def test_model_meta_decision_text():
    import mtgemb.export as exporter

    recipe = {"model_id": "m", "revision": "r"}
    old = exporter.model_meta("x", "promoted", recipe, "confirm-e5-s1")
    assert "" in old["decision"] and "" in old["decision"]
    # the sealed-run wording belongs to that one run; any other promoted model must say its own
    with pytest.raises(ValueError, match="decision"):
        exporter.model_meta("x", "promoted", recipe, "confirm-x1-s1")
    for blank in ("", "   "):  # a blank decision must not inherit the sealed-run text either
        with pytest.raises(ValueError, match="decision"):
            exporter.model_meta("x", "promoted", recipe, "confirm-x1-s1", decision=blank)
        with pytest.raises(ValueError, match="decision"):
            exporter.export("ds", "b", 8, "x", "promoted", run_id="confirm-x1-s1", decision=blank)
    new = exporter.model_meta("x", "promoted", recipe, "run", decision="no sealed run")
    assert new == {"label": "x", "status": "promoted", "backbone": "m", "revision": "r",
                   "run_id": "run", "decision": "no sealed run"}
    assert "decision" not in exporter.model_meta("x", "experimental", recipe, None)
    with pytest.raises(ValueError, match="promoted"):
        exporter.model_meta("x", "experimental", recipe, None, decision="")


def test_export_cli_forwards_decision(monkeypatch):
    import mtgemb.cli as cli
    import mtgemb.export as exporter

    received = {}
    monkeypatch.setattr(exporter, "export", lambda *a, **kw: received.update(kw) or {})
    cli.main(["export", "e5-base-v2", "--max-length", "94", "--label", "fake",
              "--status", "promoted", "--decision", ""])
    assert received["decision"] == ""


def test_card_display_colours_and_back_face():
    from mtgemb.export import card_display

    # single face: top-level colours (a devoid card has none despite a coloured cost)
    assert card_display({"colors": ["R", "W"], "image_uris": {"normal": "f"}}) == {
        "colors": "WR", "image_back": None}
    assert card_display({"colors": [], "mana_cost": "{1}{R}"})["colors"] == ""
    # transform: no top-level colours or image; union of faces, back image from face 2
    dfc = {"card_faces": [{"colors": ["U"], "image_uris": {"normal": "front.jpg"}},
                          {"colors": ["U", "B"], "image_uris": {"normal": "back.jpg"}}]}
    assert card_display(dfc) == {"colors": "UB", "image_back": "back.jpg"}
    # adventure: one printed face, so no back image even though there are two faces
    adv = {"colors": ["W", "R"], "image_uris": {"normal": "f"},
           "card_faces": [{"colors": ["W"]}, {"colors": ["R"]}]}
    assert card_display(adv) == {"colors": "WR", "image_back": None}


def test_display_fields_reads_only_catalogue_cards(tmp_path):
    import gzip
    import hashlib

    from mtgemb.export import display_fields

    rows = [{"oracle_id": "a", "colors": ["G"]}, {"oracle_id": "x", "colors": ["B"]},
            {"oracle_id": "b", "colors": []}]
    gz = tmp_path / "cards.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    sha = hashlib.sha256(gz.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"files": {"oracle_cards": {"path": gz.name, "sha256": sha}}}), "utf-8")
    msha = sha256_file(tmp_path / "manifest.json")
    got = display_fields(tmp_path, ["a", "b"], msha)
    assert got == {"a": {"colors": "G", "image_back": None},
                   "b": {"colors": "", "image_back": None}}
    with pytest.raises(ValueError, match="manifest"):
        display_fields(tmp_path, ["a", "b"], "0" * 64)
    with pytest.raises(ValueError, match="missing"):
        display_fields(tmp_path, ["a", "zz"], msha)
