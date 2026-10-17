import numpy as np

from mtgemb.export import reference_check, topk_neighbours


def test_topk_excludes_self_breaks_ties_by_index_and_blocks_agree():
    v = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8], [1, 0]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx, sco = topk_neighbours(v, k=3, block=2)
    assert idx[0].tolist() == [2, 4, 3]  # exact ties 2, 4 by index, then cos 0.6
    assert idx[1].tolist() == [3, 0, 2]  # cos 0.8, then tied zeros by index
    assert np.allclose(sco[0], [1, 1, 0.6])
    assert all(i not in row for i, row in enumerate(idx.tolist()))
    big = np.random.default_rng(0).normal(size=(300, 8)).astype(np.float32)
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    a, _ = topk_neighbours(big, k=10, block=7)
    b, _ = topk_neighbours(big, k=10, block=300)
    assert (a == b).all()
    _, bad, exact = reference_check(big, a, np.arange(20))
    assert bad == [] and exact == 20
