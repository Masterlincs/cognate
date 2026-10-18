import numpy as np

from mtgemb.linescore import LineScorer


def _u(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


def test_line_scores_by_hand():
    # rows: A = free+redirect, R = redirect, H = free+prevent, F, F = free
    texts = ["Free.\nRedirect.", "Redirect.", "Free.\nPrevent.", "Free.", "Free."]
    ang = {"Free.": 0, "Redirect.": 90, "Prevent.": 20}
    v = np.stack([_u(ang[x]) for t in texts for x in t.split("\n")])
    c = lambda d: np.cos(np.deg2rad(d))  # noqa: E731
    s = LineScorer(texts, v, "line_mean")(np.array([0, 1]))
    # A vs R: free->redirect best cos(90)=0, redirect->redirect 1 -> mean 0.5
    # A vs H: free->free 1, redirect->prevent cos(70) -> mean (1 + cos70) / 2
    assert np.allclose(s[0], [1, 0.5, (1 + c(70)) / 2, 0.5, 0.5])
    assert np.allclose(s[1], [1, 1, c(70), 0, 0])  # a one-line anchor is its best line
    # "Free." is on 4 cards (df 4), "Redirect." on 2: weights 1/4 and 1/2
    s = LineScorer(texts, v, "line_invdf")(np.array([0]))
    w1, w2 = 1 / 4, 1 / 2
    f = w1 / (w1 + w2)
    assert np.allclose(s[0], [1, w2 / (w1 + w2), (w1 + w2 * c(70)) / (w1 + w2), f, f])
    # the redirect card now beats the shared-clause card (the T001 failure shape)
    assert s[0, 1] > s[0, 2]
    assert LineScorer(texts, v, "line_mean")(np.array([0]))[0, 1] < (1 + c(70)) / 2
