import numpy as np

from mtgemb import human


def test_sample_anchors_disjoint_and_stratified(monkeypatch):
    monkeypatch.setattr(human, "N_SCORED", 3)
    monkeypatch.setattr(human, "N_CAL", 1)
    multi = [f"m{i}" for i in range(10)]
    single = [f"s{i}" for i in range(10)]
    a = human.sample_anchors(multi, single, np.random.default_rng(0))
    assert len(a["calibration"]) == 2 and len(a["scored"]) == 6
    assert not set(a["calibration"]) & set(a["scored"])
    for k in a:
        assert sum(x[0] == "m" for x in a[k]) == len(a[k]) // 2
    b = human.sample_anchors(multi, single, np.random.default_rng(0))
    assert a == b  # seeded


def test_top_k_excludes_self_and_breaks_ties_by_row():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert human.top_k(v, 0, 3) == [1, 3, 2]


def test_assemble_blinds_dedupes_and_repeats():
    rng = np.random.default_rng(1)
    anchors = {"calibration": ["c"], "scored": [f"a{i}" for i in range(4)]}
    universe = ["c"] + anchors["scored"] + [f"u{i}" for i in range(30)]
    proposals = {a: {"lexical": ["u1", "u2"], "finetuned-e5-s1": ["u2", "u3"]}
                 for a in ["c"] + anchors["scored"]}
    proposals["a0"]["ability-1"] = ["u3", "u4"]
    items, key = human.assemble(anchors, proposals, universe, rng)
    ids = [it["item_id"] for it in items]
    assert len(ids) == len(set(ids)) == len(key)
    for a in ["c"] + anchors["scored"]:
        mine = [it for it in items if it["anchor"] == a and it["phase"] != "repeat"]
        cands = [it["candidate"] for it in mine]
        assert len(cands) == len(set(cands)) and a not in cands
        want = {c for ids_ in proposals[a].values() for c in ids_}
        rand = [it for it in mine if key[it["item_id"]]["sources"] == [["random", None]]]
        assert len(rand) == human.N_RANDOM and rand[0]["candidate"] not in want
        assert set(cands) == want | {rand[0]["candidate"]}
    u2 = next(it for it in items if it["anchor"] == "a0" and it["candidate"] == "u2")
    assert sorted(key[u2["item_id"]]["sources"]) == [["finetuned-e5-s1", 1], ["lexical", 2]]
    assert all(it["phase"] == "calibration" for it in items if it["anchor"] == "c")
    reps = [it for it in items if it["phase"] == "repeat"]
    assert len(reps) == 2  # one per half (10% of 8-9 scored items rounds to 1)
    for r in reps:
        src = key[key[r["item_id"]]["repeat_of"]]
        assert src["sources"] == key[r["item_id"]]["sources"]
    # the first repeat comes from the first half and is shown after it
    first = reps[0]
    orig = next(it for it in items if it["item_id"] == key[first["item_id"]]["repeat_of"])
    assert orig["anchor"] in ("a0", "a1") and ids.index(orig["item_id"]) < ids.index(
        first["item_id"])
