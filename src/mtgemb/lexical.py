"""Character 3-5-gram TF-IDF, fitted on fit text only (lexical baseline and negative mining)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

LEXICAL_RECIPE = {"analyzer": "char", "ngram_range": (3, 5), "lowercase": True, "norm": "l2",
                  "dtype": "float32"}


def fit_vectorizer(fit_texts: list[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, norm="l2",
                          dtype=np.float32)
    vec.fit(fit_texts)
    return vec


def topk_sparse(q, c, k: int, exclude=None, block: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine of L2-normalised sparse rows q against c, in row blocks.

    exclude(row_index_block, scores) may set disallowed scores to -inf in place.
    Ties break by lower candidate index (callers order candidates by Oracle ID).
    """
    idx = np.empty((q.shape[0], k), dtype=np.int64)
    val = np.empty((q.shape[0], k), dtype=np.float32)
    ct = c.T.tocsc()
    for s in range(0, q.shape[0], block):
        e = min(s + block, q.shape[0])
        sc = (q[s:e] @ ct).toarray().astype(np.float32)
        if exclude is not None:
            exclude(np.arange(s, e), sc)
        # stable sort on -score keeps the lower index first among equal scores
        order = np.argsort(-sc, axis=1, kind="stable")[:, :k]
        idx[s:e] = order
        val[s:e] = np.take_along_axis(sc, order, axis=1)
    return idx, val
