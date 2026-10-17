import numpy as np
import pytest
import torch

from mtgemb.encoder import RECIPES, add_prefix, choose_max_length, encode, load, pool

FIXTURES = [
    "Flying",
    "Destroy target creature. It can't be regenerated.",
    "When this creature enters, draw a card.\n//\nTarget player mills three cards.",
    # long: well over the 94-token fit length, exercises truncation
    " ".join(["Whenever another creature you control enters, put a +1/+1 counter on it and "
              "you gain 1 life, then scry 1."] * 6),
]


def test_prefix_once():
    assert add_prefix(["a", "b"], "query: ") == ["query: a", "query: b"]
    with pytest.raises(ValueError):
        add_prefix(["query: a"], "query: ")
    assert add_prefix(["x"], "") == ["x"]


def test_choose_max_length():
    lengths = np.arange(1, 101)  # 99% of 100 inputs are <= 99
    assert choose_max_length(lengths) == 99
    assert choose_max_length(np.array([600] * 100)) == 512


def test_mean_pool_ignores_padding():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert torch.allclose(pool(h, m, "mean"), torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(pool(h, m, "cls"), torch.tensor([[1.0, 1.0]]))


@pytest.mark.real_model
@pytest.mark.parametrize("name,max_len", [("e5-base-v2", 512), ("e5-base-v2", 94),
                                          ("bge-small-en-v1.5", 92)])
def test_reference_parity_fp32_cpu(name, max_len):
    from sentence_transformers import SentenceTransformer

    r = RECIPES[name]
    tok, model = load(r)
    ours = encode(model, tok, FIXTURES, r, max_len, dev=torch.device("cpu"))
    st = SentenceTransformer(r["model_id"], revision=r["revision"], device="cpu")
    st.max_seq_length = max_len
    ref = st.encode([r["prefix"] + t for t in FIXTURES], normalize_embeddings=True,
                    convert_to_numpy=True)
    assert ours.shape == (len(FIXTURES), r["dim"])
    cos = (ours * ref).sum(1)
    assert cos.min() >= 0.9999, cos


@pytest.mark.real_model
def test_xpu_matches_cpu():
    if not torch.xpu.is_available():
        pytest.skip("no XPU")
    r = RECIPES["e5-base-v2"]
    tok, model = load(r)
    cpu = encode(model, tok, FIXTURES, r, 94, dev=torch.device("cpu"))
    model.to("xpu")
    xpu = encode(model, tok, FIXTURES, r, 94)
    assert ((cpu * xpu).sum(1)).min() >= 0.9999
