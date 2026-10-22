import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mtgemb.variants import effect_only, face_index, face_max, face_texts


def unit(*xy):
    v = np.array(xy, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_face_texts_split_on_the_face_boundary_and_skip_blank_faces():
    assert face_texts("Bolt text\n//\nCreature text") == ["Bolt text", "Creature text"]
    assert face_texts("one face only") == []
    assert face_texts("Front\n//\n  ") == []  # one real face is not a multi-face card
    texts, owner = face_index(["a", "x\n//\ny", "b", "p\n//\nq\n//\nr"])
    assert texts == ["x", "y", "p", "q", "r"]
    assert owner.tolist() == [1, 1, 3, 3, 3]


def test_face_max_takes_the_best_face_on_either_side():
    # card 0: Bolt; card 1: creature // bolt (whole vector diluted); card 2: unrelated
    W = unit([1, 0], [0.6, 0.8], [0, 1])
    F = unit([0, 1], [1, 0])  # card 1's faces: creature, bolt
    owner = np.array([1, 1])
    s = face_max(W, F, owner, np.array([0, 2]))
    assert np.allclose(s[0], [1.0, 1.0, 0.0])  # Bolt meets card 1's bolt face
    assert np.allclose(s[1], [0.0, 1.0, 1.0])  # unrelated meets card 1's creature face
    # a multi-face query uses its own faces too; single-face pairs keep the whole cosine
    s1 = face_max(W, F, owner, np.array([1]))
    assert np.allclose(s1[0], [1.0, 1.0, 1.0])
    # candidate side only: the query keeps its whole vector, candidates their best face
    c1 = face_max(W, F, owner, np.array([1, 0]), query_faces=False)
    assert np.allclose(c1, [[0.6, 1.0, 0.8], [1.0, 1.0, 0.0]])
    assert np.allclose(face_max(W, F[:0], owner[:0], np.array([0, 2])), W[[0, 2]] @ W.T)


def test_effect_only_drops_modifier_tags_from_relevance_only():
    class Bm:  # the Benchmark fields effect_only touches
        pass
    bm = Bm()
    bm.A = sp.csr_matrix(np.array([[1, 1], [1, 0], [0, 1]], dtype=float))
    bm.Aw = bm.A @ sp.diags([2.0, 3.0])
    bm.card_tags = {"q": ["rider", "fx"], "a": ["rider"], "b": ["fx"]}
    tags = pd.DataFrame({"tag_id": ["rider", "fx", "unseen"], "label": ["synergy-commander",
                         "change target", "x"], "df_fit": [5, 4, 0], "idf_fit": [2.0, 3.0, 9]})
    before = copy.deepcopy(bm.A.toarray())
    eo = effect_only(bm, tags, ["synergy-commander"])
    assert (bm.A.toarray() == before).all()  # the original is untouched
    assert (eo.A.toarray() == [[0, 1], [0, 0], [0, 1]]).all()
    assert (eo.Aw.toarray() == [[0, 3], [0, 0], [0, 3]]).all()
    assert eo.card_tags == {"q": ["fx"], "a": [], "b": ["fx"]}
