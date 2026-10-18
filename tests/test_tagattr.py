import numpy as np
import pandas as pd

from mtgemb import tagattr


def _u(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


def test_attribution_and_twin_check_by_hand():
    # fit one-line cards: 3 "draw" cards near 0 deg, 3 "burn" cards near 90 deg
    ids = ["d1", "d2", "d3", "b1", "b2", "b3", "m"]
    texts = ["Draw 1.", "Draw a card.", "Draw two.", "Burn 2.", "Burn it.", "Burn all.",
             "Burn 3.\nDraw 4."]  # m's lines are number twins of b1 and d1
    split = ["fit"] * 6 + ["dev"]
    ang = [0, 10, -10, 90, 80, 100]
    vectors = np.stack([_u(a) for a in ang] + [_u(45)])
    mem = pd.DataFrame({"oracle_id": ["d1", "d2", "d3", "b1", "b2", "b3", "m", "m"],
                        "tag_id": ["draw"] * 3 + ["burn"] * 3 + ["draw", "burn"]})
    tags = ["burn", "draw"]
    sums, counts, _ = tagattr.centroids(ids, texts, split, vectors, mem, tags)
    assert counts.tolist() == [3, 3]
    m_lines = np.stack([_u(85), _u(5)])  # line 0 burn-like, line 1 draw-like
    assert tagattr.attribute(m_lines, [0, 1], sums, counts) == {0: 0, 1: 1}
    # below MIN_SUPPORT a tag stays card-level
    assert tagattr.attribute(m_lines, [0], sums, np.array([2, 3])) == {0: None}
    line_v = np.stack(list(vectors[:6]) + [_u(85), _u(5)])
    owner = np.array([0, 1, 2, 3, 4, 5, 6, 6])
    # twins of m: "Burn #." -> b1 carries burn, "Draw #." -> d1 carries draw; leaving each
    # twin out leaves 2 supporters, below MIN_SUPPORT, so nothing is checkable...
    r = tagattr.check(ids, texts, split, vectors, line_v, owner, mem, tags)
    assert r["n"] == 0
    # ...with a fourth draw card the draw tag is checkable and lands on its gold line 1
    ids2, texts2 = ids + ["d4"], texts + ["Draw three."]
    vec2 = np.vstack([vectors, _u(5)])
    lv2 = np.vstack([line_v, _u(5)])
    mem2 = pd.concat([mem, pd.DataFrame({"oracle_id": ["d4"], "tag_id": ["draw"]})])
    r = tagattr.check(ids2, texts2, split + ["fit"], vec2, lv2, np.r_[owner, 7], mem2, tags)
    assert r == {"n": 1, "accuracy": 1.0, "uniform_chance": 0.5}


def test_release_tags_shapes():
    ids = ["d1", "d2", "d3", "m"]
    texts = ["Draw.", "Draw a card.", "Draw two.", "Burn.\nDraw."]
    vectors = np.stack([_u(0), _u(5), _u(-5), _u(45)])
    line_v = np.stack([_u(0), _u(5), _u(-5), _u(90), _u(2)])
    mem = pd.DataFrame({"oracle_id": ["d1", "d2", "d3", "m", "m"],
                        "tag_id": ["draw", "draw", "draw", "draw", "burn"]})
    r = tagattr.release_tags(texts, ["fit"] * 3 + ["dev"], ids, vectors, line_v, mem,
                             ["burn", "draw", "unused"])
    assert r["tag_ids"] == ["burn", "draw"]
    assert r["card"] == [[1], [1], [1], [0, 1]]
    # draw goes to line 2; burn has no one-line support, so it stays card-level only
    assert r["lines"] == [[], [], [], [[], [1]]]
