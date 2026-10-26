import json
import struct

import numpy as np
import pytest

from mtgemb import sitedata


def test_pack_map_layout_and_sentinel():
    xy = np.array([[-5.0, 0.0], [5.0, 10.0]])
    out = sitedata.pack_map(xy, np.array([0, 2]), 3, ["W", "", "UR"])
    assert len(out) == 15
    recs = [struct.unpack("<hhB", out[i * 5:i * 5 + 5]) for i in range(3)]
    assert recs[1][:2] == (-32768, -32768)
    assert recs[0][2] == 1 and recs[2][2] == 10
    assert recs[0][0] == -32000 and recs[2][0] == 32000
    assert recs[2][1] == 32000


def test_region_labels_pick_highest_lift():
    rng = np.random.default_rng(0)
    xy = np.r_[rng.normal(0, 0.1, (40, 2)), rng.normal(10, 0.1, (40, 2))]
    tags = [frozenset({"a", "common"})] * 40 + [frozenset({"b", "common"})] * 40
    out = sitedata.region_labels(xy, tags, {"a": "Alpha", "b": "Beta", "common": "Common"}, k=2)
    assert sorted(r["label"] for r in out) == ["Alpha", "Beta"]
    assert all(r["n"] == 40 for r in out)


def test_pretrained_neighbours_alignment():
    def encode(texts):
        return np.array([[1.0, 0.0], [0.6, 0.8]])[:len(texts)]

    out = sitedata.pretrained_neighbours(["x", "y"], np.array([0, 2]), 3, k=10, encode=encode)
    assert out[1] == []
    assert out[0][0][0] == 2 and out[2][0][0] == 0
    assert out[0][0][1] == pytest.approx(0.6)


def test_site_metrics_names_missing_record(tmp_path, monkeypatch):
    monkeypatch.setattr(sitedata, "RUNS", tmp_path)
    with pytest.raises(FileNotFoundError, match=str(tmp_path.name)):
        sitedata.site_metrics()


def test_site_metrics_reads_records(tmp_path, monkeypatch):
    runs, data = tmp_path / "runs", tmp_path / "ds"
    bench = sitedata.DEV_BENCH

    def put(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), "utf-8")

    for name in sitedata.DEV_MODELS_FLAT:
        put(runs / "eval" / bench / name / "summary.json",
            {"summary": {"full": {"ndcg@10": 0.5}, "hard_lexical": {"ndcg@10": 0.4}}})
    view = {"point": 0.1, "lower_adjusted": 0.05, "ci95": [0.06, 0.14]}
    rep = {"decision": "promote", "qualifier": "q", "m_contrasts": 4,
           "deployment": {"run_id": "confirm-e5-s1"},
           "contrasts": {"pretrained_e5": {"full": view, "hard_lexical": view, "passes": True}}}
    put(runs / "confirmation" / bench / "report.json", rep)
    put(runs / "sealed" / sitedata.SEALED_BENCH / "report.json", {**rep, "sealed_gate": "cleared"})
    put(runs / "exam" / "confirm-e5-s1" / "2026.json", {"n": 51, "passed": 47, "items": [
        {"id": "T001", "section": "user", "margin": -0.5, "passed": False},
        {"id": "T002", "section": "user", "margin": 0.2, "passed": True}]})
    put(runs / "confirm-e5-s1" / "result.json", {"evals": [{"full": 0.70, "update": 2000}]})
    put(runs / "confirm-x1-s1" / "result.json", {"evals": [{"full": 0.71, "update": 4000}]})
    put(data / "dataset.json", {"report": {"split_counts": {"fit": 8, "dev": 1, "sealed": 1},
                                           "tags_functional": 5, "functional_memberships": 9}})
    monkeypatch.setattr(sitedata, "RUNS", runs)
    monkeypatch.setattr(sitedata, "DATASET", data)
    monkeypatch.setattr(sitedata, "WEB", tmp_path / "web")
    m = sitedata.site_metrics()
    assert m["deployed"] is None
    assert m["dev"]["fine-tuned E5, 2000 updates"]["full"] == [0.5, 0.5, 0.5]
    assert m["dev"]["fine-tuned E5, 2000 updates"]["source"][0].endswith("summary.json")
    assert m["promotion"]["d18"]["contrasts"]["pretrained_e5"]["full"]["point"] == 0.1
    assert m["promotion"]["x1"] is None
    assert m["sealed"]["sealed_gate"] == "cleared"
    assert m["exam"]["confirm-e5-s1"]["passed"] == 47
    assert m["exam"]["confirm-e5-s1"]["failed"] == [{"id": "T001", "section": "user",
                                                     "margin": -0.5}]
    assert m["steps"]["1"] == {"2000": 0.70, "4000": 0.71}
    assert "2" not in m["steps"]
    assert m["data"]["split_counts"]["fit"] == 8


def test_copy_site_data_needs_identical_vectors_and_cards(tmp_path, monkeypatch):
    from mtgemb.util import sha256_file

    monkeypatch.setattr(sitedata, "ARTIFACTS", tmp_path / "art")
    monkeypatch.setattr(sitedata, "WEB", tmp_path / "web")
    monkeypatch.setattr(sitedata, "write_metrics", lambda: tmp_path / "metrics.json")
    data = tmp_path / "web" / "public" / "data"

    def release(rid, vec, ids, ds="ds", recorded=None):
        art = tmp_path / "art" / "releases" / rid
        art.mkdir(parents=True)
        np.save(art / "vectors.npy", np.array(vec, np.float32))
        (art / "release_meta.json").write_text(json.dumps(
            {"vectors_sha256": recorded or sha256_file(art / "vectors.npy"),
             "dataset_id": ds}), "utf-8")
        (data / rid).mkdir(parents=True)
        (data / rid / "ids.json").write_text(json.dumps({"ids": ids}), "utf-8")

    release("old", [[1, 0]], ["a", "b"])
    site = data / "site-old"
    (site / "pretrained").mkdir(parents=True)
    (site / "map.bin").write_bytes(b"")
    (site / "pretrained" / "000.json").write_text("[[], []]", "utf-8")
    (site / "map.json").write_text(json.dumps({"release_id": "old", "regions": []}), "utf-8")
    release("new", [[1, 0]], ["a", "b"])
    sitedata.copy_site_data("new", "old")
    out = data / "site-new"
    assert (out / "map.bin").read_bytes() == b""
    assert (out / "pretrained" / "000.json").read_text("utf-8") == "[[], []]"
    assert json.loads((out / "map.json").read_text("utf-8")) == {
        "release_id": "new", "regions": [], "copied_from": "old"}
    with pytest.raises(FileExistsError):
        sitedata.copy_site_data("new", "old")
    release("other-vectors", [[0, 1]], ["a", "b"])
    with pytest.raises(ValueError, match="vectors"):
        sitedata.copy_site_data("other-vectors", "old")
    release("other-cards", [[1, 0]], ["b", "a"])
    with pytest.raises(ValueError, match="card order"):
        sitedata.copy_site_data("other-cards", "old")
    # same vector bytes and ids from another dataset: the text cards (vector rows) may differ
    release("other-dataset", [[1, 0]], ["a", "b"], ds="ds2")
    with pytest.raises(ValueError, match="dataset"):
        sitedata.copy_site_data("other-dataset", "old")
    # metadata claims the old hash but the file differs
    old_sha = json.loads((tmp_path / "art" / "releases" / "old" / "release_meta.json")
                         .read_text("utf-8"))["vectors_sha256"]
    release("lying-meta", [[0, 1]], ["a", "b"], recorded=old_sha)
    with pytest.raises(ValueError, match="vectors.npy"):
        sitedata.copy_site_data("lying-meta", "old")
    assert [p.name for p in data.glob("site-*")] == ["site-new", "site-old"]


def test_is_keyword_only():
    vocab = sorted(["flying", "vigilance", "protection", "cycling", "haste"], key=len, reverse=True)
    yes = ["Flying", "Flying, vigilance", "Flying (Reminder text.)\nVigilance",
           "Cycling {2} ({2}, Discard this card: Draw a card.)",
           "Protection from white and from blue"]
    no = ["", "(Only reminder text.)", "Flying creatures you control have haste.",
          "Flying\nWhenever this creature attacks, draw a card.", "Haste, you may cast",
          "Flyingfoo"]
    assert all(sitedata.is_keyword_only(t, vocab) for t in yes)
    assert not any(sitedata.is_keyword_only(t, vocab) for t in no)
