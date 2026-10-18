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


def test_best_recipe_prefers_lower_lr_scale_unless_advanced():
    from mtgemb.report import best_recipe

    def row(lr, scale, full, hl, ok=True):
        return {"lr": lr, "scale": scale, "full": full, "hard_lexical": hl,
                "finished": True, "guardrail_passed": ok}
    a, b, c = row(1e-5, 20, 0.600, 0.580), row(3e-5, 20, 0.604, 0.590), row(3e-5, 50, 0.610, 0.570)
    assert best_recipe([c, b, a]) is a  # b gains only 0.004; c drops hard-lexical by 0.010
    b2 = row(3e-5, 20, 0.606, 0.576)
    assert best_recipe([a, b2, c]) is b2  # +0.006 full, -0.004 hard-lexical: advances
    assert best_recipe([row(1e-5, 20, 0.7, 0.7, ok=False), b]) is b
    assert best_recipe([row(1e-5, 20, 0.7, 0.7, ok=False)]) is None


def _confirm_fixture(root, cand_hl_shift):
    import json

    import numpy as np
    import pandas as pd

    anchors = [f"a{i}" for i in range(40)]
    rng = np.random.default_rng(1)

    def put(name, full, hl, run=True):
        d = root / "eval" / "B" / name
        d.mkdir(parents=True)
        rows = [pd.DataFrame({"anchor": anchors, "view": v, "n_rel": 1, "ndcg": x,
                              "family": [f"f{i // 2}" for i in range(40)]})
                for v, x in (("full", full), ("hard_lexical", hl))]
        pd.concat(rows).to_parquet(d / "per_anchor.parquet")
        mp = str(root / name / "final") if run else None
        (d / "summary.json").write_text(json.dumps({"meta": {"model_path": mp}}), "utf-8")
        if run:
            (root / name).mkdir()
            (root / name / "result.json").write_text(
                json.dumps({"finished": True, "selection": "final"}), "utf-8")

    put("pretrained-e5-base-v2", np.full(40, 0.4), np.full(40, 0.4), run=False)
    for s in (1, 2, 3):
        b = rng.uniform(0.4, 0.6, 40)
        put(f"confirm-bge-s{s}", b, b)
        put(f"confirm-e5-s{s}", b + 0.03, b + cand_hl_shift)


def test_confirmation_report_decides_and_writes_once(tmp_path, monkeypatch):
    import pytest

    import mtgemb.report as rp

    monkeypatch.setattr(rp, "RUNS", tmp_path)
    _confirm_fixture(tmp_path, 0.01)
    rep = rp.confirmation_report("B")
    assert rep["decision"] == "promote" and rep["m_contrasts"] == 4
    assert rep["contrasts"]["trained_bge"]["full"]["point"] == pytest.approx(0.03)
    assert rep["contrasts"]["trained_bge"]["full"]["lower_q"] == pytest.approx(0.0125)
    assert rep["deployment"]["run_id"] == "confirm-e5-s1"
    with pytest.raises(FileExistsError):
        rp.confirmation_report("B")


def test_confirmation_report_rejects_hard_lexical_loss_and_wrong_weights(tmp_path, monkeypatch):
    import json

    import pytest

    import mtgemb.report as rp

    monkeypatch.setattr(rp, "RUNS", tmp_path / "x")
    _confirm_fixture(tmp_path / "x", -0.001)  # full +0.03, but hard-lexical point < 0 vs BGE
    rep = rp.confirmation_report("B")
    assert rep["decision"] == "not promoted"
    assert rep["contrasts"]["pretrained_e5"]["passes"]
    assert not rep["contrasts"]["trained_bge"]["passes"]
    monkeypatch.setattr(rp, "RUNS", tmp_path / "y")
    _confirm_fixture(tmp_path / "y", 0.01)
    s = tmp_path / "y" / "eval" / "B" / "confirm-e5-s2" / "summary.json"
    s.write_text(json.dumps({"meta": {"model_path": str(tmp_path / "y" / "other")}}), "utf-8")
    with pytest.raises(ValueError, match="confirm-e5-s2"):
        rp.confirmation_report("B")
