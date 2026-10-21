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


def _confirm_fixture(root, cand_hl_shift, fam_size=2, n=240, n_rel=1, x1_full=None):
    """Fake confirmation runs, frozen recipe runs and evals under root (= report.RUNS)."""
    import json

    import numpy as np
    import pandas as pd

    anchors = [f"a{i}" for i in range(n)]
    rng = np.random.default_rng(1)
    recipe = {"e5": {"backbone": "e5-base-v2", "max_length": 94, "lr": 3e-5},
              "bge": {"backbone": "bge-small-en-v1.5", "max_length": 92, "lr": 3e-5}}
    model = {"e5-base-v2": {"model_id": "e5", "revision": "r1"},
             "bge-small-en-v1.5": {"model_id": "bge", "revision": "r2"}}

    def ident(cfg):
        return {"config": str(sorted(cfg.items())), "dataset_id": "d", "pairs_id": "p",
                "recipe": model[cfg["backbone"]], "code": "c", "benchmark_id": "B"}

    def jw(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), "utf-8")

    def put(name, full, hl, cfg=None):
        d = root / "eval" / "B" / name
        d.mkdir(parents=True)
        rows = [pd.DataFrame({"anchor": anchors, "view": v, "n_rel": n_rel, "ndcg": x,
                              "family": [f"f{i // fam_size}" for i in range(n)]})
                for v, x in (("full", full), ("hard_lexical", hl))]
        pd.concat(rows).to_parquet(d / "per_anchor.parquet")
        c = cfg or recipe["e5"]
        mp = str(root / name / "final") if cfg else None
        jw(d / "summary.json", {"benchmark_id": "B", "meta": {
            "model_path": mp, "backbone": c["backbone"], "max_length": c["max_length"],
            "recipe": model[c["backbone"]]}})
        if cfg:
            jw(root / name / "result.json", {"finished": True, "selection": "final"})
            jw(root / name / "manifest.json", {"config": cfg, "identity": ident(cfg)})

    for run, r in (("initial-1-e5-lr3e-05-s20", "e5"), ("initial-4-bge-lr3e-05-s20", "bge")):
        cfg = {**recipe[r], "seed": 0, "data_seed": 0, "selection": "guarded_best"}
        jw(root / run / "manifest.json", {"config": cfg, "identity": ident(cfg)})
    put("pretrained-e5-base-v2", np.full(n, 0.4), np.full(n, 0.4))
    for s in (1, 2, 3):
        b = rng.uniform(0.4, 0.6, n)
        run = {"seed": s, "data_seed": s, "selection": "final"}
        put(f"confirm-bge-s{s}", b, b, {**recipe["bge"], **run})
        put(f"confirm-e5-s{s}", b + 0.03, b + cand_hl_shift, {**recipe["e5"], **run})
        if x1_full is not None:  # 4000-update recipe, x1_full above the incumbent
            put(f"confirm-x1-s{s}", b + 0.03 + x1_full, b + cand_hl_shift + 0.001,
                {**recipe["e5"], "total_updates": 4000, **run})
    if x1_full is not None:
        cfg = {**recipe["e5"], "total_updates": 4000, "seed": 0, "data_seed": 0,
               "selection": "guarded_best"}
        jw(root / "x-1-e5-4000" / "manifest.json", {"config": cfg, "identity": ident(cfg)})


def test_confirmation_report_decides_and_writes_once(tmp_path, monkeypatch):
    import pytest

    import mtgemb.report as rp

    monkeypatch.setattr(rp, "RUNS", tmp_path)
    _confirm_fixture(tmp_path, 0.01)
    rep = rp.confirmation_report("B")
    assert rep["decision"] == "promote" and rep["m_contrasts"] == 4
    full = rep["contrasts"]["trained_bge"]["full"]
    assert full["point"] == pytest.approx(0.03) and full["lower_q"] == pytest.approx(0.0125)
    assert full["n_clusters"] == 120 and rep["excluded_zero_positive"]["full"] == 0
    assert rep["deployment"]["run_id"] == "confirm-e5-s1"
    with pytest.raises(FileExistsError):
        rp.confirmation_report("B")
    assert [p.name for p in (tmp_path / "confirmation" / "B").iterdir()] == ["report.json"]


def test_confirmation_report_reject_and_inconclusive(tmp_path, monkeypatch):
    import mtgemb.report as rp

    monkeypatch.setattr(rp, "RUNS", tmp_path / "r")
    _confirm_fixture(tmp_path / "r", -0.001)  # full +0.03, hard-lexical point < 0 vs BGE
    rep = rp.confirmation_report("B")
    assert rep["decision"] == "reject"
    assert rep["contrasts"]["pretrained_e5"]["passes"]
    assert not rep["contrasts"]["trained_bge"]["passes"]
    monkeypatch.setattr(rp, "RUNS", tmp_path / "i")
    _confirm_fixture(tmp_path / "i", 0.01, fam_size=10)  # 24 families < MIN_CLUSTERS
    assert rp.confirmation_report("B")["decision"] == "inconclusive"
    monkeypatch.setattr(rp, "RUNS", tmp_path / "e")
    _confirm_fixture(tmp_path / "e", 0.01, n_rel=0)  # empty eligible cohort: no bootstrap
    rep = rp.confirmation_report("B")
    assert rep["decision"] == "inconclusive" and rep["contrasts"] == {}


def test_confirmation_report_refuses_wrong_provenance(tmp_path, monkeypatch):
    import json

    import pytest

    import mtgemb.report as rp

    def fresh(tag):
        root = tmp_path / tag
        monkeypatch.setattr(rp, "RUNS", root)
        _confirm_fixture(root, 0.01)
        return root

    def edit(path, key=None, **kw):
        d = json.loads(path.read_text("utf-8"))
        node = d[key] if key else d.get("meta", d.get("config"))
        node.update(kw)
        path.write_text(json.dumps(d), "utf-8")

    cases = [
        ("weights", "eval/B/confirm-e5-s2/summary.json", {"model_path": "elsewhere"}),
        ("maxlen", "eval/B/confirm-bge-s3/summary.json", {"max_length": 94}),
        ("pre", "eval/B/pretrained-e5-base-v2/summary.json", {"max_length": 128}),
        ("seed", "confirm-e5-s3/manifest.json", {"data_seed": 0}),
        ("recipe", "confirm-bge-s1/manifest.json", {"lr": 1e-5}),
        ("select", "confirm-e5-s1/manifest.json", {"selection": "guarded_best"}),
        ("pairs", "confirm-e5-s2/manifest.json", {"key": "identity", "pairs_id": "p2"}),
        ("code", "confirm-bge-s2/manifest.json", {"key": "identity", "code": "c2"}),
        ("bench", "eval/B/confirm-e5-s1/summary.json", {"key": None, "benchmark_id": "X"}),
        ("rev", "eval/B/pretrained-e5-base-v2/summary.json",
         {"recipe": {"model_id": "e5", "revision": "r9"}}),
    ]
    for tag, rel, kw in cases:
        root = fresh(tag)
        if kw.get("key", "") is None:  # top-level field of the eval record
            path = root / rel
            d = json.loads(path.read_text("utf-8"))
            d.update({k: v for k, v in kw.items() if k != "key"})
            path.write_text(json.dumps(d), "utf-8")
        else:
            edit(root / rel, **kw)
        with pytest.raises(ValueError):
            rp.confirmation_report("B")


def test_sealed_report_retests_registered_contrasts_once(tmp_path, monkeypatch):
    import json
    import shutil

    import pytest

    import mtgemb.report as rp

    def sealed_copy(root, bid="S"):
        shutil.copytree(root / "eval" / "B", root / "eval" / "S")
        for p in (root / "eval" / "S").glob("*/summary.json"):
            d = json.loads(p.read_text("utf-8"))
            d["benchmark_id"] = bid
            p.write_text(json.dumps(d), "utf-8")

    monkeypatch.setattr(rp, "RUNS", tmp_path / "ok")
    _confirm_fixture(tmp_path / "ok", 0.01)
    sealed_copy(tmp_path / "ok")
    rep = rp.confirmation_report("S", train_bench="B")
    assert rep["sealed_gate"] == "cleared" and "decision" not in rep
    assert rep["train_benchmark_id"] == "B" and rep["design"]["decisions"][-1] == ""
    assert (tmp_path / "ok" / "sealed" / "S" / "report.json").exists()
    assert not (tmp_path / "ok" / "confirmation").exists()
    with pytest.raises(FileExistsError):
        rp.confirmation_report("S", train_bench="B")
    monkeypatch.setattr(rp, "RUNS", tmp_path / "rej")
    _confirm_fixture(tmp_path / "rej", -0.001)
    sealed_copy(tmp_path / "rej")
    assert rp.confirmation_report("S", train_bench="B")["sealed_gate"] == "not_cleared"
    monkeypatch.setattr(rp, "RUNS", tmp_path / "stale")
    _confirm_fixture(tmp_path / "stale", 0.01)
    sealed_copy(tmp_path / "stale", bid="B")  # records still claim the dev benchmark
    with pytest.raises(ValueError):
        rp.confirmation_report("S", train_bench="B")


def test_x1_design_adds_incumbent_and_writes_under_tag(tmp_path, monkeypatch):
    import pytest

    import mtgemb.report as rp

    monkeypatch.setattr(rp, "RUNS", tmp_path / "p")
    _confirm_fixture(tmp_path / "p", 0.01, x1_full=0.01)
    rep = rp.confirmation_report("B", design=rp.CONFIRM_X1)
    assert rep["decision"] == "promote" and rep["m_contrasts"] == 6
    inc = rep["contrasts"]["incumbent_e5"]["full"]
    assert inc["point"] == pytest.approx(0.01) and inc["comparator"][0] == "confirm-e5-s1"
    assert rep["deployment"]["run_id"] == "confirm-x1-s1"
    assert (tmp_path / "p" / "confirmation" / "B" / "x1" / "report.json").exists()
    assert not (tmp_path / "p" / "confirmation" / "B" / "report.json").exists()
    assert rp.confirmation_report("B")["decision"] == "promote"  # design still separate
    with pytest.raises(ValueError, match="sealed"):
        rp.confirmation_report("S", train_bench="B", design=rp.CONFIRM_X1)

    # +0.003 over the incumbent: passes pretrained and BGE, fails the swap contrast
    monkeypatch.setattr(rp, "RUNS", tmp_path / "r")
    _confirm_fixture(tmp_path / "r", 0.01, x1_full=0.003)
    rep = rp.confirmation_report("B", design=rp.CONFIRM_X1)
    assert rep["decision"] == "reject" and not rep["contrasts"]["incumbent_e5"]["passes"]
    assert rep["contrasts"]["trained_bge"]["passes"]

    # a candidate run not trained from the x-1 recipe is refused
    monkeypatch.setattr(rp, "RUNS", tmp_path / "w")
    _confirm_fixture(tmp_path / "w", 0.01, x1_full=0.01)
    man = tmp_path / "w" / "confirm-x1-s2" / "manifest.json"
    man.write_text(man.read_text("utf-8").replace('"total_updates": 4000', '"total_updates": 2000'))
    with pytest.raises(ValueError, match="frozen recipe x-1-e5-4000"):
        rp.confirmation_report("B", design=rp.CONFIRM_X1)
