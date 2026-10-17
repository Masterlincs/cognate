from mtgemb.report import advances, promotion


def test_advancement_thresholds_and_guardrail():
    base = {"full": 0.600, "hard_lexical": 0.550, "guardrail_passed": True}
    assert advances({**base, "full": 0.6050}, base)
    assert not advances({**base, "full": 0.6049}, base)
    assert not advances({**base, "full": 0.62, "hard_lexical": 0.5449}, base)
    assert not advances({**base, "full": 0.62, "guardrail_passed": False}, base)


def test_promotion_bounds_are_strict():
    ok_f = {"point": 0.005, "lower_adjusted": 1e-9}
    ok_h = {"point": 0.0, "lower_adjusted": -0.0099}
    assert promotion(ok_f, ok_h)
    assert not promotion({**ok_f, "lower_adjusted": 0.0}, ok_h)
    assert not promotion(ok_f, {**ok_h, "lower_adjusted": -0.010})
    assert not promotion({**ok_f, "point": 0.0049}, ok_h)


def test_contrast_aligns_by_seed_id_not_order(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import pytest

    import mtgemb.report as rp

    monkeypatch.setattr(rp, "RUNS", tmp_path)
    anchors = [f"a{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    base = {s: rng.normal(0.5, 0.2, 30) for s in (1, 2, 3)}

    def put(name, vals):
        d = tmp_path / "eval" / "B" / name
        d.mkdir(parents=True)
        pd.DataFrame({"anchor": anchors, "view": "full", "n_rel": 1, "ndcg": vals}).to_parquet(
            d / "per_anchor.parquet")

    for s in (1, 2, 3):
        put(f"cand{s}", base[s] + 0.01)  # candidate = matched comparator + 0.01 exactly
        put(f"comp{s}", base[s])
    put("pre", np.full(30, 0.5))
    fam = pd.Series(anchors, index=anchors)
    cand = {1: "cand1", 2: "cand2", 3: "cand3"}
    fwd = rp.contrast(cand, {1: "comp1", 2: "comp2", 3: "comp3"}, "full", "B", 4, fam)
    rev = rp.contrast(cand, {3: "comp3", 2: "comp2", 1: "comp1"}, "full", "B", 4, fam)
    assert fwd["point"] == pytest.approx(0.01) and fwd["lower_adjusted"] == pytest.approx(0.01)
    assert rev["lower_adjusted"] == fwd["lower_adjusted"]
    with pytest.raises(ValueError, match="seed sets"):
        rp.contrast(cand, {1: "comp1", 2: "comp2"}, "full", "B", 4, fam)
    assert rp.contrast(cand, "pre", "full", "B", 4, fam)["n_seeds"] == 3
