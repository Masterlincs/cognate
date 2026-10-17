import hashlib
import json

import numpy as np
import pytest

from mtgemb.export import (
    finish_interrupted,
    ranking_parity,
    reference_check,
    release_digest,
    topk_neighbours,
)


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
