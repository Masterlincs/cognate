import math

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import mtgemb.encoder as enc
import mtgemb.train as tr
from mtgemb.train import (
    TrainConfig,
    build_schedule,
    contrastive_loss,
    schedule_hash,
    select_checkpoint,
)

# ---------------------------------------------------------------- loss hand cases


def _setup():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    known = torch.zeros(2, 3, dtype=torch.bool)
    excl = torch.zeros(2, 3, dtype=torch.bool)
    return q, c, known, excl


def test_masked_infonce_hand_values():
    q, c, known, excl = _setup()
    known[1, 0] = True  # c0 is another known positive of row 1: no negative competition
    loss, st = contrastive_loss(q, c, known, excl, 1.0, "masked_infonce")
    l0 = -1 + math.log(math.e + 1 + math.exp(-1))
    l1 = -1 + math.log(math.e + 1)  # only c2 (cos 0) competes
    assert float(loss) == pytest.approx((l0 + l1) / 2, rel=1e-6)
    assert st["skipped"] == 0 and st["multi_pos_rows"] == 1


def test_multi_positive_hand_values():
    q, c, known, excl = _setup()
    known[1, 0] = True
    loss, _ = contrastive_loss(q, c, known, excl, 1.0, "multi_positive")
    l0 = -1 + math.log(math.e + 1 + math.exp(-1))
    l1 = -math.log((math.e + 1) / (math.e + 1 + 1))
    assert float(loss) == pytest.approx((l0 + l1) / 2, rel=1e-6)


def test_family_duplicate_is_never_a_positive():
    q, c, known, excl = _setup()
    known[1, 0] = True
    excl[1, 0] = True  # same family: excluded from both roles
    loss, _ = contrastive_loss(q, c, known, excl, 1.0, "multi_positive")
    l0 = -1 + math.log(math.e + 1 + math.exp(-1))
    l1 = -1 + math.log(math.e + 1)
    assert float(loss) == pytest.approx((l0 + l1) / 2, rel=1e-6)


def test_rows_without_negatives_are_skipped_and_weights_apply():
    q, c, known, excl = _setup()
    known[1, 0] = True
    excl[1, 2] = True  # row 1 now has no allowed negative
    loss, st = contrastive_loss(q, c, known, excl, 1.0, "masked_infonce")
    assert st["skipped"] == 1
    assert float(loss) == pytest.approx(-1 + math.log(math.e + 1 + math.exp(-1)), rel=1e-6)
    q, c, known, excl = _setup()
    w = torch.tensor([0.0, 1.0])
    loss, _ = contrastive_loss(q, c, known, excl, 1.0, "masked_infonce", w)
    assert float(loss) == pytest.approx(-1 + math.log(math.e + 2), rel=1e-6)


def test_gradient_directions_and_masked_zero_gradient():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(3, 4), dim=1).requires_grad_()
    c = torch.nn.functional.normalize(torch.randn(5, 4), dim=1).requires_grad_()
    known = torch.zeros(3, 5, dtype=torch.bool)
    excl = torch.zeros(3, 5, dtype=torch.bool)
    known[:, 4] = True  # candidate 4 is masked for every row
    loss, _ = contrastive_loss(q, c, known, excl, 20.0, "masked_infonce")
    loss.backward()
    assert torch.all(c.grad[4] == 0)
    # descent moves each designated positive towards its row and an allowed negative away
    for i in range(3):
        assert float(-c.grad[i] @ q[i].detach()) > 0
    assert float(-c.grad[3] @ q.detach().sum(0)) < 0


# ---------------------------------------------------------------- schedule


class FakeData:
    """Minimal TrainData stand-in: 60 cards, 6 tags, families of size 1-2."""

    def __init__(self):
        n = 60
        rng = np.random.default_rng(0)
        self.ids = [f"c{i:03d}" for i in range(n)]
        self.texts = [f"word{i % 7} word{i % 5} card" for i in range(n)]
        self.family = np.array([i // 2 if i < 10 else i for i in range(n)])
        tags = np.arange(n) % 6
        self.E = sp.csr_matrix((np.ones(n), (np.arange(n), tags)), shape=(n, 6))
        a, b, t = [], [], []
        for tag in range(6):
            mem = np.flatnonzero(tags == tag)
            for _ in range(3):
                p = rng.permutation(mem)
                for x, y in zip(p[::2], p[1::2], strict=False):
                    if self.family[x] != self.family[y]:
                        a.append(min(x, y))
                        b.append(max(x, y))
                        t.append(tag)
        self.a, self.b, self.pair_tag = np.array(a), np.array(b), np.array(t)
        self.spec_weight = np.ones(len(a), dtype=np.float32)
        self.neg_pool, self.neg_rank = {}, {}
        for i in range(n):
            pool = np.array([j for j in range(n) if tags[j] != tags[i]
                             and self.family[j] != self.family[i]])
            self.neg_pool[i] = pool[:20]
            self.neg_rank[i] = np.arange(1, len(pool[:20]) + 1)
        self.identity = {"dataset_id": "fake", "pairs_id": "fake"}

    def known_pos(self, rows, cols):
        return (self.E[rows] @ self.E[cols].T).toarray() > 0


def _cfg(**kw):
    base = dict(run_id="t", slot="smoke-0", backbone="tiny", max_length=16, batch_pairs=4,
                hard_negatives=1, neg_band=(3, 10), total_updates=12, eval_every=100,
                dataset_dir="unused", lr=1e-3, warmup_frac=0.25)
    base.update(kw)
    return TrainConfig(**base)


def test_schedule_constraints_and_determinism():
    data = FakeData()
    cfg = _cfg()
    s1, s2 = build_schedule(cfg, data), build_schedule(cfg, data)
    assert len(s1) == 12 and schedule_hash(s1) == schedule_hash(s2)
    assert schedule_hash(build_schedule(_cfg(data_seed=1), data)) != schedule_hash(s1)
    for b in s1:
        cards = np.concatenate([b["anchors"], b["positives"], b["negatives"]])
        assert len(set(cards.tolist())) == len(cards)  # no card twice
        assert len(set(data.family[cards].tolist())) == len(cards)  # no family twice
        assert len(b["anchors"]) == 4
        for p, x, y in zip(b["pairs"], b["anchors"], b["positives"], strict=True):
            assert {int(x), int(y)} == {int(data.a[p]), int(data.b[p])}
    # negatives come from the band of some anchor in the batch
    for b in s1:
        for n in b["negatives"]:
            assert any(n in data.neg_pool[int(x)][2:10] for x in b["anchors"])


def test_tag_grouped_schedule_runs():
    data = FakeData()
    s = build_schedule(_cfg(batch_mode="tag_grouped"), data)
    assert len(s) == 12
    rows_with_extra = 0
    for b in s:
        k = data.known_pos(b["anchors"], b["positives"])
        np.fill_diagonal(k, False)
        rows_with_extra += int(k.any(1).sum())
    assert rows_with_extra > 0


def test_select_checkpoint_rule():
    ev = [{"update": 10, "full": 0.70, "hard_lexical": 0.60},
          {"update": 20, "full": 0.72, "hard_lexical": 0.50},
          {"update": 30, "full": 0.70, "hard_lexical": 0.61}]
    best, ok = select_checkpoint(ev, guardrail=0.60)
    assert ok and best["update"] == 30  # 20 fails guardrail; tie on full -> higher hard-lex
    best, ok = select_checkpoint(ev, guardrail=0.90)
    assert not ok and best["update"] == 30  # none qualifies -> final, failed


# ---------------------------------------------------------------- resume


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "query", ":", "card"]
    words += [f"word{i}" for i in range(7)]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
                                           special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]")
    torch.manual_seed(0)
    model = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32,
                                 max_position_embeddings=32))
    d = tmp_path / "tiny_model"
    model.save_pretrained(d)
    fast.save_pretrained(d)
    monkeypatch.setitem(enc.RECIPES, "tiny", {"model_id": str(d), "revision": None,
                                              "prefix": "query: ", "pooling": "mean",
                                              "normalize": True, "dim": 16})
    monkeypatch.setattr(tr, "RUNS", tmp_path / "runs")
    return tmp_path


def _weights(path):
    from safetensors.torch import load_file

    return load_file(str(path / "model.safetensors"))


def test_resume_matches_uninterrupted_and_rejects_corruption(tiny):
    data = FakeData()
    cpu = torch.device("cpu")
    tr.train(_cfg(run_id="straight"), None, dev=cpu, data=data)
    tr.train(_cfg(run_id="split", stop_after=5), None, dev=cpu, data=data)
    assert (tiny / "runs" / "split" / "last.pt").exists()
    assert not (tiny / "runs" / "split" / "final").exists()
    tr.train(_cfg(run_id="split"), None, dev=cpu, data=data)
    wa = _weights(tiny / "runs" / "straight" / "final")
    wb = _weights(tiny / "runs" / "split" / "final")
    assert wa.keys() == wb.keys()
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k
    lines = (tiny / "runs" / "split" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 12  # no duplicated updates after resume

    # corrupted checkpoint is rejected
    tr.train(_cfg(run_id="bad", stop_after=3), None, dev=cpu, data=data)
    ck = tiny / "runs" / "bad" / "last.pt"
    raw = bytearray(ck.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    ck.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="corrupted"):
        tr.train(_cfg(run_id="bad"), None, dev=cpu, data=data)
    # changed settings must fork, not resume
    with pytest.raises(ValueError, match="different identity"):
        tr.train(_cfg(run_id="split", lr=5e-4), None, dev=cpu, data=data)
