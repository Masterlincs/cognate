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
    with pytest.raises(ValueError, match="numbered from 1"):
        exam.parse(md.replace(" line 2 |", " line 0 |"))
    with pytest.raises(ValueError, match="has no line 3"):
        exam.score([{**p, "status": "accepted", "line": 3}], cards, lambda t: None)
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
    assert r["dev_only"] == {"n": 0, "passed": 0}  # match and trap are fit cards
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


@pytest.mark.parametrize("version", ["no_reminder_v1", "no_reminder_v2"])
def test_exam_score_text_version(tmp_path, monkeypatch, selfname_cards, version):
    from mtgemb import data, encoder, selfname

    cards = selfname_cards
    ids = cards["oracle_id"].tolist()
    worksheet = tmp_path / "traps.md"
    items = [{"id": "T001", "section": "user", "status": "accepted", "anchor": ids[0],
              "line": 2, "match": ids[1], "trap": ids[2]},
             {"id": "T002", "section": "user", "status": "accepted", "anchor": ids[0],
              "line": None, "match": ids[1], "trap": ids[2]}]
    worksheet.write_text(exam.HEADER + "\n".join(
        exam.render(it, cards.set_index("oracle_id")) for it in items), "utf-8")
    seen = []

    class Model:
        def to(self, dev):
            assert dev.type == "cpu"

    def encode(model, tok, texts, *args, **kwargs):
        assert kwargs["dev"].type == "cpu"
        seen.append(texts)
        return np.tile([1., 0.], (len(texts), 1))

    monkeypatch.setattr(data, "load_dataset", lambda _: {"cards": cards})
    monkeypatch.setattr(encoder, "load", lambda *_: (None, Model()))
    monkeypatch.setattr(encoder, "encode", encode)
    monkeypatch.setattr(exam, "RUNS", tmp_path / "exam-records")
    kwargs = {} if version == "no_reminder_v1" else {"text_version": version}
    out = exam.exam_score("unused", "fixture", "e5-base-v2", 94, path=worksheet, **kwargs)
    expected = selfname.apply_text_version(cards, version)["text"].tolist()
    assert seen == [[expected[0].split("\n")[1]], [expected[1]], [expected[2]], expected[:3]]
    assert out["text_version"] == version and out["n"] == 2


@pytest.mark.parametrize("args,expected", [([], "no_reminder_v1"),
                                         (["--text-version", "no_reminder_v2"],
                                          "no_reminder_v2")])
def test_exam_cli_passes_text_version(monkeypatch, capsys, args, expected):
    from mtgemb import cli

    def score(*a, **kw):
        assert kw["text_version"] == expected
        return dict(name="fixture", n=0, passed=0, sections={}, dev_only={}, n_proposed=0)

    monkeypatch.setattr(exam, "exam_score", score)
    cli.main(["exam-score", "fixture", "e5-base-v2", "--max-length", "94", *args])
    assert '"fixture"' in capsys.readouterr().out
