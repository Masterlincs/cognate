import numpy as np
import pandas as pd

from mtgemb import linetag


def test_tag_text_uses_label_and_first_sentence():
    assert linetag.tag_text("mill-self", None) == "mill-self"
    assert linetag.tag_text("lure", "All creatures must block. See also [lure-limited](x).") == (
        "lure: All creatures must block.")
    assert linetag.tag_text("x", "See [a](b) and more") == "x: See a and more"


def _ds():
    cards = pd.DataFrame({
        "oracle_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "split": ["fit", "fit", "fit", "dev", "fit", "fit"],
        "text_status": ["ok"] * 6,
        "text": ["Draw a card.", "Flying\nDraw a card.", "Destroy target creature.",
                 "Draw a card.", "Destroy target creature.", "Gain 3 life."],
        "family_id": ["f1", "f2", "f3", "f4", "f3", "f6"]})  # c5 is c3's same-text twin
    mem = pd.DataFrame({"oracle_id": ["c1", "c1", "c2", "c3", "c4", "c5", "c6"],
                        "tag_id": ["draw", "cantrip", "draw", "removal", "draw", "removal",
                                   "lifegain"]})
    tags = pd.DataFrame({"tag_id": ["draw", "cantrip", "removal", "lifegain"],
                         "label": ["draw", "cantrip", "removal", "lifegain"],
                         "description": [None, "Replaces itself. More.", None, None]})
    return {"cards": cards, "memberships": mem, "tags": tags}


def test_build_uses_fit_one_line_cards_and_excludes_exam_families():
    aux = linetag.build(_ds(), {"draw", "cantrip", "removal", "lifegain"}, exclude={"c3"})
    # c2 has two lines, c4 is dev, c3 is excluded and c5 is its same-text twin
    assert aux["line_ids"] == ["c1", "c6"]
    pairs = {(aux["line_ids"][i], aux["tag_ids"][j])
             for i, j in zip(aux["row_line"], aux["row_tag"], strict=True)}
    assert pairs == {("c1", "draw"), ("c1", "cantrip"), ("c6", "lifegain")}
    assert "cantrip: Replaces itself." in aux["tag_texts"]
    assert linetag.build(_ds(), {"draw", "cantrip", "removal", "lifegain"}, {"c3"})["hash"] == (
        aux["hash"])


def test_schedule_has_no_repeated_line_or_tag_and_masks_known_tags():
    aux = linetag.build(_ds(), {"draw", "cantrip", "removal", "lifegain"}, exclude=set())
    sch = linetag.schedule(aux, batch=2, updates=5, data_seed=0)
    assert len(sch) == 5
    assert all((a == b).all() for a, b in zip(sch, linetag.schedule(aux, 2, 5, 0), strict=True))
    for rows in sch:
        assert len(set(aux["row_line"][rows])) == 2 and len(set(aux["row_tag"][rows])) == 2
    # rows (c1, draw) and (c3, removal): no extra known positives
    r_draw = next(i for i in range(len(aux["row_line"]))
                  if aux["line_ids"][aux["row_line"][i]] == "c1"
                  and aux["tag_ids"][aux["row_tag"][i]] == "draw")
    r_can = next(i for i in range(len(aux["row_line"]))
                 if aux["tag_ids"][aux["row_tag"][i]] == "cantrip")
    r_rem = next(i for i in range(len(aux["row_line"]))
                 if aux["tag_ids"][aux["row_tag"][i]] == "removal")
    assert not linetag.masks(aux, np.array([r_draw, r_rem])).any()
    # rows (c1, cantrip) and (c3, removal) vs (c1, draw): c1 carries draw -> known
    k = linetag.masks(aux, np.array([r_can, r_draw, r_rem]))
    assert k.tolist() == [[False, True, False], [True, False, False], [False, False, False]]
