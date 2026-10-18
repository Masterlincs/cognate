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


def test_export_v3_shards_alignment_mapping_and_artifacts(tmp_path, monkeypatch):
    from contextlib import nullcontext

    import pandas as pd
    import torch

    import mtgemb.export as exporter

    # Textless cards before/between text rows expose accidental text-row bundle indices.
    texts = ["", "A\n//\nB", "", "C", "D"]
    cards = pd.DataFrame([
        {"oracle_id": str(i), "split": "fit", "text_status": "ok" if text else "empty",
         "text": text, "name": str(i), "faces": "[]", "layout": "normal",
         "color_identity": "", "mv": 0, "type_line": "", "is_creature": False,
         "image": "", "scryfall_uri": ""} for i, text in enumerate(texts)])
    monkeypatch.setattr(exporter, "load_dataset", lambda _: {
        "cards": cards, "ident": {"dataset_id": "test", "snapshot": "test"}})
    monkeypatch.setattr(exporter, "DeviceLock", nullcontext)
    monkeypatch.setattr(exporter, "ARTIFACTS", tmp_path / "art")
    monkeypatch.setattr(exporter, "SHARD", 2)
    monkeypatch.setattr(exporter, "log_event", lambda *a, **kw: None)
    monkeypatch.setattr(exporter, "git_identity", lambda: {})
    monkeypatch.setattr(exporter.enc, "load", lambda *a: (None, torch.nn.Identity()))
    monkeypatch.setattr(exporter.enc, "device", lambda: torch.device("cpu"))
    vectors = {"A": [1, 0], "B": [0, 1], "A\n//\nB": [1, 0],
               "C": [0.6, 0.8], "D": [1, 0]}
    monkeypatch.setattr(exporter.enc, "encode", lambda model, tok, lines, *a, **kw: np.array(
        [vectors[line] for line in lines], np.float32))
    result = exporter.export("unused", "e5-base-v2", 94, "fake", "experimental",
                             web_dir=tmp_path / "bundle")
    from pathlib import Path

    bundle, art = Path(result["bundle"]), Path(result["artifacts"])
    release = json.loads((bundle / "release.json").read_text("utf-8"))
    assert release["schema"] == "mtgemb-release-v3" and release["ability_k"] == 50
    assert release["abilities_status"] == "exploratory: not evaluated on the tag proxy"
    expected = [[], [{"text": "A", "neighbours": [[4, 1.0], [3, 0.6]]},
                     {"text": "B", "neighbours": [[3, 0.8], [4, 0.0]]}], [], [], []]
    rows = []
    for i, length in enumerate([2, 2, 1]):
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
                     {"text": "C", "oracle_id": "3"}, {"text": "D", "oracle_id": "4"}]
    assert np.array_equal(np.load(art / "ability_vectors.npy"),
                          np.array([[1, 0], [0, 1], [0.6, 0.8], [1, 0]], np.float32))
    meta = json.loads((art / "release_meta.json").read_text("utf-8"))
    assert meta["n_ability_lines"] == 4 and meta["n_ability_queries"] == 2
    for name in ("ability_vectors", "ability_lines"):
        suffix = ".npy" if name == "ability_vectors" else ".json"
        assert meta[name + "_sha256"] == sha256_file(art / (name + suffix))
    assert meta["ability_parity"]["topk_reference_failures"] == 0


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
