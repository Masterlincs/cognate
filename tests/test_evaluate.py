import math

import numpy as np
import pandas as pd
import pytest

from mtgemb.evaluate import (
    Benchmark,
    build_benchmark,
    evaluate,
    n_boot_for,
    paired_bootstrap,
    rank_metrics,
)

L2 = math.log2


def test_rank_metrics_hand_example():
    r = rank_metrics([3, 0, 1], [3, 1, 0], n_rel=2)
    assert r["ndcg"] == pytest.approx((3 + 1 / L2(4)) / (3 + 1 / L2(3)))
    assert r["ap"] == pytest.approx((1 + 2 / 3) / 2)
    assert r["mrr"] == 1.0 and r["recall"] == 1.0


def test_rank_metrics_no_positives_and_short_lists():
    r = rank_metrics([0, 0], [], n_rel=0)
    assert math.isnan(r["ndcg"]) and r["ap"] == 0.0 and math.isnan(r["recall"])
    r = rank_metrics([5.0], [5.0], n_rel=1)  # one candidate
    assert r["ndcg"] == 1.0 and r["ap"] == 1.0 and r["mrr"] == 1.0
    r = rank_metrics([0, 2], [2, 0], n_rel=1)
    assert r["mrr"] == 0.5 and r["ap"] == 0.5 and r["ndcg"] == pytest.approx(1 / L2(3))


def test_ap_denominator_caps_at_k_recall_does_not():
    g = [1] * 10
    r = rank_metrics(g, g, n_rel=30)
    assert r["ap"] == 1.0 and r["recall"] == pytest.approx(10 / 30)


def _fixture():
    ids = ["a1", "c1", "c2", "c3", "c4", "d2", "s1"]
    cards = pd.DataFrame({
        "oracle_id": ids,
        "split": ["dev", "fit", "fit", "fit", "fit", "dev", "sealed"],
        "text_status": ["ok"] * 7,
        "family_id": ["a1", "a1", "c2", "c3", "c4", "d2", "s1"],
        "color_identity": ["R", "R", "U", "R", "R", "R", "R"],
        "mv": [1.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0],
        "is_creature": [False, False, False, True, False, False, False],
        "text": ["deal 3 damage", "deal 3 damage to any target", "draw a card",
                 "create a token", "gain life", "scry 2", "deal 3 damage"],
        "n_faces": [1, 1, 1, 1, 1, 2, 1],
    })
    mem = pd.DataFrame({
        "oracle_id": ["a1", "c1", "c2", "a1", "c3", "d2", "c4", "s1"],
        "tag_id": ["t1", "t1", "t1", "t2", "t2", "t3", "t4", "t1"],
    })
    tags = pd.DataFrame({"tag_id": ["t1", "t2", "t3", "t4"], "df_fit": [2, 1, 0, 1],
                         "idf_fit": [2.0, 3.0, None, 1.5]})
    return cards, mem, tags


def _vectors(bm, score_by_id):
    """Rows whose cosine with anchor a1 equals score_by_id (2-D construction)."""
    rows = []
    for oid in bm.ids:
        if oid == "a1":
            rows.append([1.0, 0.0])
        else:
            s = score_by_id[oid]
            rows.append([s, math.sqrt(1 - s * s)])
    return np.array(rows)


def test_benchmark_universe_and_anchors():
    cards, mem, tags = _fixture()
    b = build_benchmark(cards, mem, tags, "ds")
    assert b["universe"] == ["a1", "c1", "c2", "c3", "c4", "d2"]  # sealed excluded, sorted
    assert b["anchors"] == ["a1"]  # d2 only has a tag unseen in fit


def test_views_hand_calculated():
    cards, mem, tags = _fixture()
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    v = _vectors(bm, {"c1": 0.9, "c4": 0.8, "c3": 0.7, "c2": 0.6, "d2": 0.5})
    s = evaluate(bm, v)["per_anchor"].set_index("view")
    # full ranking c1(2) c4(0) c3(3) c2(2) d2(0); ideal 3,2,2
    dcg = 2 + 3 / L2(4) + 2 / L2(5)
    idcg = 3 + 2 / L2(3) + 2 / L2(4)
    assert s.loc["full", "ndcg"] == pytest.approx(dcg / idcg)
    assert s.loc["full", "n_rel"] == 3 and s.loc["full", "n_candidates"] == 5
    assert s.loc["full", "ap"] == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
    # hard drops c1 (same family) from both the ranking and the ideal
    assert s.loc["hard", "ndcg"] == pytest.approx((3 / L2(3) + 2 / L2(4)) / (3 + 2 / L2(3)))
    assert s.loc["hard", "n_rel"] == 2
    # cross-colour: only c2 (U) differs -> one relevant, ranked first
    assert s.loc["cross_colour", "n_candidates"] == 1 and s.loc["cross_colour", "ndcg"] == 1.0
    # cross-cost (|dmv| >= 2): only c2 (mv 3)
    assert s.loc["cross_cost", "n_candidates"] == 1
    # cross-body: only c3 is a creature
    assert s.loc["cross_body", "n_candidates"] == 1 and s.loc["cross_body", "ndcg"] == 1.0
    # strict: same colour R, mv 1, noncreature -> c1, d2
    assert s.loc["strict", "n_candidates"] == 2 and s.loc["strict", "ndcg"] == 1.0
    # dev-candidates-only: just d2, gain 0 -> zero positives, excluded
    assert s.loc["dev_candidates", "n_rel"] == 0 and math.isnan(s.loc["dev_candidates", "ndcg"])


def test_hard_lexical_excludes_high_jaccard():
    cards, mem, tags = _fixture()
    cards.loc[cards.oracle_id == "c3", "text"] = "deal 3 damage"  # Jaccard 1 with a1
    cards.loc[cards.oracle_id == "c3", "family_id"] = "c3"
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    v = _vectors(bm, {"c1": 0.9, "c4": 0.8, "c3": 0.7, "c2": 0.6, "d2": 0.5})
    s = evaluate(bm, v)["per_anchor"].set_index("view")
    assert s.loc["hard", "n_candidates"] == 4 and s.loc["hard_lexical", "n_candidates"] == 3
    assert s.loc["hard_lexical", "n_rel"] == 1  # only c2 remains relevant


def test_ties_break_by_oracle_id_and_row_order_invariance():
    cards, mem, tags = _fixture()
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    v = _vectors(bm, {"c1": 0.5, "c2": 0.5, "c3": 0.5, "c4": 0.5, "d2": 0.5})
    out = evaluate(bm, v, top_n=5)
    assert out["top"]["a1"] == ["c1", "c2", "c3", "c4", "d2"]
    shuffled = cards.sample(frac=1, random_state=3)
    bm2 = Benchmark(build_benchmark(shuffled, mem.sample(frac=1, random_state=4), tags, "ds"),
                    shuffled, mem, tags)
    out2 = evaluate(bm2, v, top_n=5)
    pd.testing.assert_frame_equal(out["per_anchor"], out2["per_anchor"])


def test_duplicate_text_candidates_both_count():
    cards, mem, tags = _fixture()
    cards.loc[cards.oracle_id == "c2", "text"] = "deal 3 damage to any target"  # same as c1
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    v = _vectors(bm, {"c1": 0.9, "c2": 0.9, "c3": 0.7, "c4": 0.8, "d2": 0.5})
    assert evaluate(bm, v, top_n=2)["top"]["a1"] == ["c1", "c2"]


@pytest.mark.parametrize("bad", ["nan", "zero", "rows"])
def test_invalid_vectors_fail(bad):
    cards, mem, tags = _fixture()
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    v = _vectors(bm, {"c1": 0.9, "c4": 0.8, "c3": 0.7, "c2": 0.6, "d2": 0.5})
    if bad == "nan":
        v[2, 0] = np.nan
    elif bad == "zero":
        v[3] = 0
    else:
        v = v[:-1]
    with pytest.raises(ValueError):
        evaluate(bm, v)


def test_lexical_sparse_vectors_accepted():
    import scipy.sparse as sp

    cards, mem, tags = _fixture()
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    v = _vectors(bm, {"c1": 0.9, "c4": 0.8, "c3": 0.7, "c2": 0.6, "d2": 0.5})
    dense = evaluate(bm, v)["per_anchor"]
    sparse = evaluate(bm, sp.csr_matrix(v * 3.0))["per_anchor"]  # scale-invariant
    pd.testing.assert_frame_equal(dense, sparse)


def test_bootstrap_known_sign_and_pairing():
    rng = np.random.default_rng(0)
    n = 400
    base = rng.uniform(0, 1, size=(3, n))
    clusters = np.repeat(np.arange(n // 2), 2)
    up = paired_bootstrap(base + 0.05, base, clusters, n_boot=4000, seed=1, lower_q=0.0125)
    assert up["point"] == pytest.approx(0.05) and up["lower_adjusted"] > 0.04
    down = paired_bootstrap(base - 0.05, base, clusters, n_boot=4000, seed=1)
    assert down["ci95"][1] < 0
    # pairing: huge shared per-anchor noise cancels exactly in the difference
    noisy = base + rng.normal(0, 10, size=(1, n))
    r = paired_bootstrap(noisy + 0.01, noisy, clusters, n_boot=2000, seed=2)
    assert r["ci95"][0] == pytest.approx(0.01) and r["ci95"][1] == pytest.approx(0.01)
    # untrained single-seed baseline broadcasts over seeds
    r = paired_bootstrap(base + 0.02, base[:1], clusters, n_boot=500, seed=3)
    assert r["n_seeds"] == 3


def test_n_boot_tail():
    assert n_boot_for(8) == 16000


def test_untagged_candidate_stays_in_universe_with_zero_gain():
    cards, mem, tags = _fixture()
    extra = cards.iloc[[3]].assign(oracle_id="c0", family_id="c0", text="untagged thing")
    cards = pd.concat([cards, extra], ignore_index=True)
    bm = Benchmark(build_benchmark(cards, mem, tags, "ds"), cards, mem, tags)
    assert "c0" in bm.ids and bm.spec["counts"]["universe_untagged"] == 1
    # c0 outranks every relevant card: it occupies rank 1 with gain 0
    v = _vectors(bm, {"c0": 0.95, "c1": 0.9, "c4": 0.8, "c3": 0.7, "c2": 0.6, "d2": 0.5})
    s = evaluate(bm, v)["per_anchor"].set_index("view")
    dcg = 2 / L2(3) + 3 / L2(5) + 2 / L2(6)
    idcg = 3 + 2 / L2(3) + 2 / L2(4)
    assert s.loc["full", "ndcg"] == pytest.approx(dcg / idcg)


def test_validate_vectors_handles_huge_finite_rows():
    from mtgemb.evaluate import validate_vectors

    x = validate_vectors(np.array([[1e20, 1e20], [1.0, 0.0]], dtype=np.float32), 2)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
    with pytest.raises(ValueError):
        validate_vectors(np.zeros((2, 2)), 2)


def test_benchmark_id_binds_dataset_identity():
    cards, mem, tags = _fixture()
    a = build_benchmark(cards, mem, tags, "ds")
    b = build_benchmark(cards, mem, tags, "ds-other")
    assert a["universe"] == b["universe"] and a["benchmark_id"] != b["benchmark_id"]
