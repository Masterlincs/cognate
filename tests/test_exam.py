import numpy as np
import pandas as pd
import pytest

from mtgemb import exam

A, M, T = ("00000000-0000-0000-0000-00000000000" + c for c in "abc")


def _cards(texts):
    return pd.DataFrame({"oracle_id": [A, M, T], "name": ["Anchor", "Match", "Trap"],
                         "split": ["dev", "fit", "fit"], "text": texts}).set_index("oracle_id")


def test_opener_and_number_mask():
    assert exam.opener("When this creature dies, draw a card.") == "When this creature dies,"
    assert exam.opener("If you control a commander, you may cast it.") == (
        "If you control a commander,")
    assert exam.opener("Draw a card.") is None
    assert exam.mask_numbers("Draw two cards. Deal 3 damage. X life.") == (
        "Draw # cards. Deal # damage. # life.")


def test_render_parse_round_trip_and_status_rules():
    cards = _cards(["Flying\nDraw a card.", "Draw a card.", "Discard a card."])
    item = {"id": "T001", "section": "user", "status": "proposed", "anchor": A, "line": 2,
            "match": M, "trap": T, "note": "user report"}
    md = exam.HEADER + exam.render(item, cards)
    assert "anchor: " + A + " line 2 | Anchor (dev) | Draw a card." in md
    (p,) = exam.parse(md)
    assert {k: p[k] for k in ("id", "section", "status", "anchor", "line", "match", "trap")} == {
        k: item[k] for k in ("id", "section", "status", "anchor", "line", "match", "trap")}
    with pytest.raises(ValueError, match="unknown section or status"):
        exam.parse(md.replace("· proposed", "· maybe"))
    with pytest.raises(ValueError, match="needs anchor"):
        exam.parse(md.replace("trap: ", "trapp: "))
    with pytest.raises(ValueError, match="malformed item heading"):
        exam.parse(md + "\n## T002 · user · proposed # note\n")
    with pytest.raises(ValueError, match="repeats its anchor"):
        exam.parse(md.replace("match: ", "anchor: "))
    with pytest.raises(ValueError, match="duplicate"):
        exam.parse(md + "\n" + exam.render(item, cards))


def test_exam_cards_lists_every_status(tmp_path):
    cards = _cards(["a", "b", "c"])
    md = exam.HEADER + exam.render({"id": "T001", "section": "rider", "status": "rejected",
                                    "anchor": A, "match": M, "trap": T}, cards)
    (tmp_path / "t.md").write_text(md, "utf-8")
    assert exam.exam_cards(tmp_path / "t.md") == {A, M, T}
    assert exam.exam_cards(tmp_path / "missing.md") == set()


def _u(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


def test_score_whole_card_and_line_anchor():
    angle = {"anchor": 0, "match": 10, "trap": 40, "x": 90, "y": 0, "m1": 80, "m2": 5, "t1": 30}
    encode = lambda texts: np.stack([_u(angle[t]) for t in texts])  # noqa: E731
    whole = _cards(["anchor", "match", "trap"])
    base = {"section": "rider", "status": "accepted", "anchor": A, "match": M, "trap": T}
    r = exam.score([{**base, "id": "T001", "line": None},
                    {**base, "id": "T002", "line": None, "status": "proposed"}], whole, encode)
    assert r["n"] == 1 and r["passed"] == 1  # proposed items are not scored
    assert np.isclose(r["items"][0]["margin"], np.cos(np.deg2rad(10)) - np.cos(np.deg2rad(40)))
    # line anchor "y" (0 deg): match's best line m2 (5 deg) beats trap's t1 (30 deg),
    # although the match's other line m1 is far away
    lines = _cards(["x\ny", "m1\nm2", "t1"])
    r = exam.score([{**base, "id": "T003", "line": 2}], lines, encode)
    assert r["items"][0]["passed"] and np.isclose(r["items"][0]["match"], np.cos(np.deg2rad(5)))
    # swap match and trap: fails
    r = exam.score([{**base, "id": "T004", "line": 2, "match": T, "trap": M}], lines, encode)
    assert r["passed"] == 0 and r["sections"]["rider"] == {
        "n": 1, "passed": 0,
        "mean_margin": pytest.approx(np.cos(np.deg2rad(30)) - np.cos(np.deg2rad(5)))}


def test_guard_compares_identical_items_only():
    def rec(*rows):
        return {"items": [{"id": i, "item": [A, None, M, T if not edited else A], "passed": p}
                          for i, p, edited in rows]}
    old = rec(("T001", True, False), ("T002", False, False), ("T003", True, False),
              ("T004", True, False))
    new = rec(("T001", False, False), ("T002", True, False), ("T003", True, True),
              ("T005", True, False))
    g = exam.guard(old, new)
    assert g == {"n_compared": 2, "regressed": ["T001"], "fixed": ["T002"],
                 "only_old": ["T003", "T004"], "only_new": ["T003", "T005"]}
