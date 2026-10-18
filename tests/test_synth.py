import numpy as np
import pandas as pd
import pytest

from mtgemb import augment, synth
from mtgemb.train import TrainData


def _ds():
    rows = [
        ("a", "fit", "ok", "fa", "When this enters, scry 1.", "scry"),
        ("b", "fit", "ok", "fb", "When you cast a spell, scry 1.", "scry"),
        # A valid opener partner needs no positive variant of its own.
        ("c", "fit", "ok", "fc", "When you cast a spell, surveil.", "surveil"),
        ("d", "fit", "ok", "fd", "Surveil 1.", "surveil"),
        ("exam", "dev", "ok", "fx", "When you cast a spell, investigate 1.", "clue"),
        ("twin", "fit", "ok", "fx", "When you cast a spell, investigate 1.", "clue"),
        ("dev", "dev", "ok", "fv", "When you cast a spell, explore 1.", "explore"),
        ("bad", "fit", "empty", "fe", "When you cast a spell, investigate 2.", "clue"),
        ("ineligible", "fit", "ok", "fi", "When you cast a spell, explore 2.", "other"),
        ("untagged", "fit", "ok", "fu", "When you cast a spell, investigate 3.", None),
    ]
    cards = pd.DataFrame(rows, columns=["oracle_id", "split", "text_status", "family_id",
                                       "text", "tag"])
    mem = cards.loc[cards["tag"].notna(), ["oracle_id", "tag"]].rename(columns={"tag": "tag_id"})
    return {"cards": cards.drop(columns="tag"), "memberships": mem}


ELIG = {"scry", "surveil", "clue", "explore"}


def test_build_filters_anchors_and_opener_partners():
    rows = synth.build(_ds(), ELIG, {"exam"})
    assert rows["anchor_ids"] == ["a", "b", "d"]
    assert rows["families"] == ["fa", "fb", "fd"]
    assert rows["negatives"][0] == augment.flip(rows["anchors"][0], 0, key="a")
    assert rows["negatives"][1] == "When you cast a spell, surveil."
    assert rows["negatives"][2] is None
    assert rows["positives"][1:] == ["When you cast a spell, scry 2.", "Surveil 2."]
    assert all(a != p for a, p in zip(rows["anchors"], rows["positives"], strict=True))
    assert len(rows["hash"]) == 16


def test_build_is_seeded_and_input_order_independent():
    ds = _ds()
    first = synth.build(ds, ELIG, {"exam"}, seed=12)
    assert synth.build(ds, ELIG, {"exam"}, seed=12) == first
    shuffled = {k: v.sample(frac=1, random_state=3) for k, v in ds.items()}
    assert synth.build(shuffled, ELIG, {"exam"}, seed=12) == first
    kinds = {synth.build(ds, ELIG, {"exam"}, seed=s)["positive_kinds"][0] for s in range(20)}
    assert kinds == {"numbers", "wording"}  # both valid variants can be chosen
    assert synth.build(ds, set(), set())["anchors"] == []


def test_train_data_uses_exam_exclusions(monkeypatch):
    from mtgemb import exam

    data = TrainData.__new__(TrainData)
    data._ds, data._elig = _ds(), ELIG
    data.ids = sorted(data._ds["cards"]["oracle_id"])
    monkeypatch.setattr(exam, "exam_cards", lambda: {"exam"})
    got = data.synth_rows()
    assert got.pop("card_rows").tolist() == [data.ids.index(o) for o in got["anchor_ids"]]
    assert got == synth.build(data._ds, ELIG, {"exam"})


def test_opener_partners_cannot_share_a_non_pair_eligible_functional_tag():
    ds = _ds()
    extra = pd.DataFrame({"oracle_id": ["b", "c"], "tag_id": "broad"})
    ds["memberships"] = pd.concat([ds["memberships"], extra], ignore_index=True)
    rows = synth.build(ds, ELIG, {"exam"})
    assert rows["negatives"][rows["anchor_ids"].index("b")] is None


def test_schedule_cycles_unique_rows_deterministically():
    rows = {"anchors": list(range(7))}
    batches = synth.schedule(rows, batch=3, updates=9, data_seed=4)
    again = synth.schedule(rows, 3, 9, 4)
    assert len(batches) == 9
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))
    assert all(len(b) == len(set(b)) == 3 for b in batches)
    for i in range(0, 8, 2):
        assert len(set(np.concatenate(batches[i:i + 2]))) == 6
    assert any(not np.array_equal(a, b) for a, b in zip(
        batches, synth.schedule(rows, 3, 9, 5), strict=True))
    assert synth.schedule(rows, 3, 0, 4) == []
    for batch in (0, 8):
        with pytest.raises(ValueError, match="cannot fill"):
            synth.schedule(rows, batch, 1, 0)
