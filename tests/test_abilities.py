import numpy as np

from mtgemb.abilities import flatten, maxsim_reference, maxsim_topk, split_abilities


def test_split_drops_blank_and_face_boundary():
    assert split_abilities("Flying\n\n  Draw a card. \n//\nScry 2") == [
        "Flying", "Draw a card.", "Scry 2"]
    assert split_abilities("Haste") == ["Haste"]


def test_flatten_offsets():
    lines, owner, starts = flatten(["a\nb", "c", "d\n//\ne\nf"])
    assert lines == ["a", "b", "c", "d", "e", "f"]
    assert owner.tolist() == [0, 0, 1, 2, 2, 2] and starts.tolist() == [0, 2, 3]


def _u(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


def test_maxsim_hand_example():
    # card 0: lines at 0 and 90 deg; card 1: 80 deg; card 2: 10 deg and 180 deg; card 3: 45 deg
    line_v = np.stack([_u(0), _u(90), _u(80), _u(10), _u(180), _u(45)])
    owner = np.array([0, 0, 1, 2, 2, 3])
    starts = np.array([0, 2, 3, 5])
    # query = card 0's 90-degree line: card 1 (10 deg away) beats card 3 (45) beats card 2 (80)
    idx, sco = maxsim_topk(line_v[[1]], np.array([0]), line_v, starts, k=3)
    assert idx.tolist() == [[1, 3, 2]]
    assert np.allclose(sco, [[np.cos(np.deg2rad(10)), np.cos(np.deg2rad(45)),
                              np.cos(np.deg2rad(80))]])
    # query = card 0's 0-degree line: card 2's 10-degree line wins although its other is 180
    idx, _ = maxsim_topk(line_v[[0]], np.array([0]), line_v, starts, k=1)
    assert idx.tolist() == [[2]]
    ref = maxsim_reference(line_v[0], 0, line_v, owner, 4)
    assert np.isneginf(ref[0]) and np.argmax(ref) == 2


def test_maxsim_ties_break_by_lower_row_and_blocks_agree():
    rng = np.random.default_rng(0)
    line_v = rng.normal(size=(40, 8))
    line_v[17] = line_v[25] = line_v[5]  # rows 8 and 12 both hold row 2's line 5
    line_v /= np.linalg.norm(line_v, axis=1, keepdims=True)
    owner = np.repeat(np.arange(20), 2)
    starts = np.arange(0, 40, 2)
    q, q_owner = line_v, owner
    a = maxsim_topk(q, q_owner, line_v, starts, k=5, block=7)
    b = maxsim_topk(q, q_owner, line_v, starts, k=5, block=64)
    assert (a[0] == b[0]).all()
    for i in range(40):
        ref = maxsim_reference(q[i], q_owner[i], line_v, owner, 20)
        assert np.allclose(np.sort(ref)[::-1][:5], a[1][i])
        assert q_owner[i] not in a[0][i]
    # query = line 5 (row 2): rows 8 and 12 tie at cosine 1 -> lower row first; row 2 excluded
    assert a[0][5][:2].tolist() == [8, 12] and np.allclose(a[1][5][:2], 1.0)
