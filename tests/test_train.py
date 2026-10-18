import math
from contextlib import nullcontext

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
    sd = tiny / "runs" / "split"
    assert (sd / "last.json").exists() and not (sd / "final").exists()
    # interruption artefacts: an unpublished newer generation and a torn metrics record
    (sd / "ckpt-0000009.pt").write_bytes(b"partial")
    with open(sd / "metrics.jsonl", "a") as f:
        f.write('{"update": 6, "lo')
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
    ck = tiny / "runs" / "bad" / "ckpt-0000003.pt"
    raw = bytearray(ck.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    ck.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="corrupted"):
        tr.train(_cfg(run_id="bad"), None, dev=cpu, data=data)
    # changed settings must fork, not resume
    with pytest.raises(ValueError, match="different identity"):
        tr.train(_cfg(run_id="split", lr=5e-4), None, dev=cpu, data=data)


def test_checkpoint_carries_skip_counters(tiny):
    data = FakeData()
    tr.train(_cfg(run_id="cnt", stop_after=4), None, dev=torch.device("cpu"), data=data)
    side = tr.read_json(tiny / "runs" / "cnt" / "last.json")
    st = torch.load(tiny / "runs" / "cnt" / side["file"], weights_only=False)
    assert st["rows_total"] == 4 * 2 * 4 and st["skipped_total"] >= 0


def test_selection_modes():
    with pytest.raises(ValueError, match="confirmation"):
        _cfg(slot="confirm-1")
    cfg = _cfg(slot="confirm-1", selection="final", guardrail_hard_lexical=0.60)
    ev = [{"update": 10, "full": 0.90, "hard_lexical": 0.70},
          {"update": 20, "full": 0.70, "hard_lexical": 0.50}]
    sel, ok = tr.finalise_selection(ev, cfg)
    assert sel["update"] == 20 and not ok  # final update regardless of the earlier peak
    sel, ok = tr.finalise_selection(ev, _cfg(guardrail_hard_lexical=0.60))
    assert sel["update"] == 10 and ok


def test_guarded_selection_requires_guardrail(tiny):
    with pytest.raises(ValueError, match="guardrail"):
        tr.train(_cfg(run_id="g"), {"benchmark_id": "x"}, dev=torch.device("cpu"),
                 data=FakeData())


def test_fresh_start_discards_torn_metrics(tiny):
    d = tiny / "runs" / "fresh"
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text('{"update": 1, "lo')
    tr.train(_cfg(run_id="fresh", stop_after=3), None, dev=torch.device("cpu"), data=FakeData())
    tr.train(_cfg(run_id="fresh"), None, dev=torch.device("cpu"), data=FakeData())
    assert len((d / "metrics.jsonl").read_text().splitlines()) == 12


class _Saver:
    def __init__(self, tag):
        self.tag = tag

    def save_pretrained(self, path, **_):
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{self.tag}.txt").write_text(self.tag)


def test_save_weights_recovers_interrupted_swaps(tmp_path):
    p = tmp_path / "best"
    # published new copy, stale backup left behind
    (tmp_path / "best.old").mkdir()
    p.mkdir()
    tr._save_weights(_Saver("m1"), _Saver("t1"), p)
    assert (p / "m1.txt").exists() and not (tmp_path / "best.old").exists()
    # interrupted between renames: only the backup exists
    p.rename(tmp_path / "best.old")
    tr._save_weights(_Saver("m2"), _Saver("t2"), p)
    assert (p / "m2.txt").exists() and not (tmp_path / "best.old").exists()


class AuxData(FakeData):
    """FakeData plus line-tag rows built by linetag.build from a tiny ds."""

    def line_tag_rows(self):
        import pandas as pd

        from mtgemb import linetag

        n = 12
        cards = pd.DataFrame({"oracle_id": [f"o{i:02d}" for i in range(n)], "split": "fit",
                              "text_status": "ok", "family_id": [f"f{i}" for i in range(n)],
                              "text": [f"word{i % 7} card" for i in range(n)]})
        mem = pd.DataFrame({"oracle_id": [f"o{i:02d}" for i in range(n)],
                            "tag_id": [f"t{i % 6}" for i in range(n)]})
        tags = pd.DataFrame({"tag_id": [f"t{j}" for j in range(6)],
                             "label": [f"word{j}" for j in range(6)], "description": None})
        return linetag.build({"cards": cards, "memberships": mem, "tags": tags},
                             {f"t{j}" for j in range(6)}, set())


def test_aux_keeps_card_schedule_and_resumes_exactly(tiny):
    data, cpu = AuxData(), torch.device("cpu")
    base = tr.train(_cfg(run_id="plain"), None, dev=cpu, data=data)
    full = tr.train(_cfg(run_id="aux", aux="line_tag", aux_batch=4), None, dev=cpu, data=data)
    assert full["card_schedule_hash"] == base["schedule_hash"] != full["schedule_hash"]
    tr.train(_cfg(run_id="aux2", aux="line_tag", aux_batch=4, stop_after=5), None, dev=cpu,
             data=data)
    tr.train(_cfg(run_id="aux2", aux="line_tag", aux_batch=4), None, dev=cpu, data=data)
    wa, wb = _weights(tiny / "runs" / "aux" / "final"), _weights(tiny / "runs" / "aux2" / "final")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    wp = _weights(tiny / "runs" / "plain" / "final")
    assert any(not torch.equal(wa[k], wp[k]) for k in wa)  # the aux stream changed training
    rec = tr.json.loads((tiny / "runs" / "aux" / "metrics.jsonl").read_text().splitlines()[0])
    assert rec["aux_loss"] > 0 and rec["aux_skipped"] == 0
    side = tr.read_json(tiny / "runs" / "aux" / "last.json")
    st = torch.load(tiny / "runs" / "aux" / side["file"], weights_only=False)
    assert st["rows_total"] == 12 * (2 * 4 + 2 * 4)  # card rows and aux rows both counted
    man = tr.read_json(tiny / "runs" / "aux" / "manifest.json")
    assert man["identity"]["aux_rows"] == data.line_tag_rows()["hash"]


def test_synth_loss_masks_family_positives_and_keeps_all_negatives(monkeypatch):
    # A0=A1=(1,0); P0=(1,0), P1=(0,1); N0=(-1,0), N1=(0,-1).
    # Same-family P1/P0 are masked. Both negatives compete for each anchor:
    # Row 0: log(exp(1)+exp(-1)+1)-1; row 1: log(1+exp(-1)+1).
    vectors = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]],
                           requires_grad=True)
    rows = {"anchors": ["a0", "a1"], "positives": ["p0", "p1"],
            "negatives": ["n0", "n1"], "families": ["f", "f"]}
    seen = []

    def tokenize(tok, texts, recipe, max_length):
        seen.extend(texts)
        return {}

    monkeypatch.setattr(enc, "tokenize", tokenize)
    monkeypatch.setattr(enc, "embed", lambda *_: vectors)
    loss, stats = tr.synth_loss(None, None, {}, 16, rows, np.arange(2), 1., "cpu", nullcontext())
    expected = (math.log(math.e + math.exp(-1) + 1) - 1
                + math.log(1 + math.exp(-1) + 1)) / 2
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-6)
    assert stats["skipped"] == 0 and stats["rows"] == 2 and stats["eff_neg_mean"] == 2
    assert seen == ["a0", "a1", "p0", "p1", "n0", "n1"]
    loss.backward()
    assert (vectors.grad[4:, 0] > 0).all()  # both negatives repel both anchors


@pytest.mark.parametrize("families,skipped", [(["f", "f"], 2), (["f", "g"], 0)])
def test_synth_loss_without_explicit_negatives(monkeypatch, families, skipped):
    vectors = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
    monkeypatch.setattr(enc, "tokenize", lambda *_: {})
    monkeypatch.setattr(enc, "embed", lambda *_: vectors)
    rows = {"anchors": ["a", "b"], "positives": ["p", "q"],
            "negatives": [None, None], "families": families}
    loss, stats = tr.synth_loss(None, None, {}, 16, rows, np.arange(2), 1., "cpu", nullcontext())
    assert stats["skipped"] == skipped
    assert float(loss.detach()) == pytest.approx(0 if skipped else math.log1p(math.exp(-1)))
    loss.backward()
    assert torch.isfinite(vectors.grad).all()


def test_synth_loss_masks_positives_of_anchors_sharing_a_tag(monkeypatch):
    # different families, but the anchors share an eligible tag: the other positive is masked,
    # so with no explicit negatives both rows have nothing to compete against
    vectors = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
    monkeypatch.setattr(enc, "tokenize", lambda *_: {})
    monkeypatch.setattr(enc, "embed", lambda *_: vectors)
    rows = {"anchors": ["a", "b"], "positives": ["p", "q"],
            "negatives": [None, None], "families": ["f", "g"]}
    shared = np.ones((2, 2), dtype=bool)
    _, stats = tr.synth_loss(None, None, {}, 16, rows, np.arange(2), 1., "cpu", nullcontext(),
                             shared)
    assert stats["skipped"] == 2


def test_synth_loss_single_row_without_negative_is_skipped(monkeypatch):
    vectors = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    monkeypatch.setattr(enc, "tokenize", lambda *_: {})
    monkeypatch.setattr(enc, "embed", lambda *_: vectors)
    rows = {"anchors": ["a"], "positives": ["p"], "negatives": [None], "families": ["f"]}
    loss, stats = tr.synth_loss(None, None, {}, 16, rows, np.array([0]), 1., "cpu", nullcontext())
    assert float(loss.detach()) == 0 and stats["skipped"] == stats["rows"] == 1
    assert stats["masked_frac"] == 0
    loss.backward()
    assert torch.equal(vectors.grad, torch.zeros_like(vectors))


class SynthData(AuxData):
    def synth_rows(self):
        import pandas as pd

        from mtgemb import synth

        n = 12
        cards = pd.DataFrame({"oracle_id": [f"s{i:02d}" for i in range(n)], "split": "fit",
                              "text_status": "ok", "family_id": [f"sf{i}" for i in range(n)],
                              "text": [f"word{i % 7} enters, scry 1." for i in range(n)]})
        mem = pd.DataFrame({"oracle_id": cards["oracle_id"], "tag_id": "scry"})
        rows = synth.build({"cards": cards, "memberships": mem}, {"scry"}, set())
        rows["card_rows"] = np.arange(len(rows["anchors"]))  # stand-ins among FakeData rows
        return rows


@pytest.mark.parametrize("aux", ["none", "line_tag"])
def test_synth_keeps_schedules_and_resumes_exactly(tiny, monkeypatch, aux):
    data, cpu = SynthData(), torch.device("cpu")
    batches_seen, started = {}, []
    original = tr._train_locked

    def capture(cfg, *args):
        batches_seen[cfg.run_id] = args[6]
        return original(cfg, *args)

    monkeypatch.setattr(tr, "_train_locked", capture)
    monkeypatch.setattr(tr, "log_event", lambda *args, **kw: started.append((args, kw)))
    base = tr.train(_cfg(run_id="plain"), None, dev=cpu, data=data)
    compare = tr.train(_cfg(run_id="compare", aux=aux, aux_batch=4), None, dev=cpu, data=data)
    kw = dict(aux=aux, aux_batch=4, synth=True, synth_batch=4)
    full = tr.train(_cfg(run_id="synth", **kw), None, dev=cpu, data=data)
    assert full["card_schedule_hash"] == compare["card_schedule_hash"] == base["schedule_hash"]
    assert full["schedule_hash"] != compare["schedule_hash"]
    for b, c in zip(batches_seen["synth"], batches_seen["compare"], strict=True):
        for key in ("anchors", "positives", "negatives", "pairs", "aux"):
            if key in c:
                assert np.array_equal(b[key], c[key])
    tr.train(_cfg(run_id="split", stop_after=5, **kw), None, dev=cpu, data=data)
    split = tr.train(_cfg(run_id="split", **kw), None, dev=cpu, data=data)
    assert split["schedule_hash"] == full["schedule_hash"]
    wa = _weights(tiny / "runs" / "synth" / "final")
    wb = _weights(tiny / "runs" / "split" / "final")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    wc = _weights(tiny / "runs" / "compare" / "final")
    assert any(not torch.equal(wa[k], wc[k]) for k in wa)
    records = [tr.json.loads(line) for line in
               (tiny / "runs" / "synth" / "metrics.jsonl").read_text().splitlines()]
    resumed = [tr.json.loads(line) for line in
               (tiny / "runs" / "split" / "metrics.jsonl").read_text().splitlines()]
    assert len(resumed) == 12
    for a, b in zip(records, resumed, strict=True):
        assert {k: v for k, v in a.items() if k != "sec"} == {
            k: v for k, v in b.items() if k != "sec"}
    assert records[0]["synth_loss"] > 0 and records[0]["synth_skipped"] == 0
    side = tr.read_json(tiny / "runs" / "synth" / "last.json")
    st = torch.load(tiny / "runs" / "synth" / side["file"], weights_only=False)
    assert st["rows_total"] == 12 * (8 + (8 if aux == "line_tag" else 0) + 4)
    man = tr.read_json(tiny / "runs" / "synth" / "manifest.json")
    assert man["identity"]["synth_rows"] == data.synth_rows()["hash"]
    event = next(kw for args, kw in started if args == ("train", "STARTED")
                 and kw["run_id"] == "synth")
    assert event["synth_rows"] == data.synth_rows()["hash"] and event["synth_count"] == 12


def test_synth_skips_are_checkpointed_and_trip_the_shared_guard(tiny, monkeypatch):
    original = tr.synth_loss

    def skipped(*args):
        loss, stats = original(*args)
        return loss * 0, {**stats, "skipped": stats["rows"]}

    monkeypatch.setattr(tr, "synth_loss", skipped)
    kw = dict(run_id="skip", synth=True, synth_batch=4, total_updates=50)
    data, cpu = SynthData(), torch.device("cpu")
    tr.train(_cfg(stop_after=4, **kw), None, dev=cpu, data=data)
    side = tr.read_json(tiny / "runs" / "skip" / "last.json")
    st = torch.load(tiny / "runs" / "skip" / side["file"], weights_only=False)
    assert st["skipped_total"] == 4 * 4 and st["rows_total"] == 4 * (8 + 4)
    with pytest.raises(RuntimeError, match="skipped-row rate"):
        tr.train(_cfg(**kw), None, dev=cpu, data=data)


def test_text_version_train_data_streams(selfname_cards, monkeypatch):
    import pandas as pd

    ids = selfname_cards["oracle_id"].tolist()
    ds = {"cards": selfname_cards, "ident": {"dataset_id": "fixture"},
          "memberships": pd.DataFrame({"oracle_id": ids, "tag_id": "t"}),
          "tags": pd.DataFrame({"tag_id": ["t"], "label": ["life"], "description": [None]})}
    files = {"pairs.parquet": pd.DataFrame({"a": [ids[1]], "b": [ids[2]],
                                           "tag_id": ["t"], "spec_weight": [1.]}),
             "pair_tags.parquet": pd.DataFrame({"tag_id": ["t"]}),
             "negatives.parquet": pd.DataFrame({"anchor": [ids[1]], "negative": [ids[3]],
                                               "rank": [1]})}
    monkeypatch.setattr(tr, "load_dataset", lambda _: dict(ds))
    monkeypatch.setattr(tr, "read_json", lambda _: {"files": {}, "pairs_id": "fixture"})
    monkeypatch.setattr(pd, "read_parquet", lambda p: files[p.name])
    monkeypatch.setattr(tr.exam, "exam_cards", set)
    v1, v2 = tr.TrainData("unused"), tr.TrainData("unused", "no_reminder_v2")
    assert v1.fit_texts_changed == 0 and v2.fit_texts_changed == 3
    assert v1.texts[0] == "Ally gains 2 life."
    assert v2.texts[v2.a[0]] == "This creature gains 2 life."
    assert v2.texts[v2.b[0]] == "This creature loses 2 life."
    assert "This creature gains 2 life." in v2.line_tag_rows()["lines"]
    rows = v2.synth_rows()
    assert "This creature gains 2 life." in rows["anchors"]
    assert all("Ally" not in t and "Trap" not in t for key in
               ("anchors", "positives", "negatives") for t in rows[key] if t is not None)
    assert ds["cards"].loc[11, "text"] == "Ally gains 2 life."
    for attr in ("ids", "identity"):
        assert getattr(v1, attr) == getattr(v2, attr)
    for attr in ("a", "b", "family", "pair_tag", "spec_weight"):
        np.testing.assert_array_equal(getattr(v1, attr), getattr(v2, attr))


def test_text_version_keeps_schedule_hash():
    cfg, data = _cfg(), FakeData()
    assert schedule_hash(build_schedule(cfg, data)) == (
        "aeebcf111bb4d9dc37f9bf4c743d851ae2c94f0c5b514029d78b3fba0a1726a6")
    assert schedule_hash(build_schedule(_cfg(text_version="no_reminder_v2"), data)) == (
        schedule_hash(build_schedule(cfg, data)))
    assert tr.sha256_json(cfg.model_dump()) != tr.sha256_json(
        _cfg(text_version="no_reminder_v2").model_dump())
    assert "selfname.py" in tr._CODE_FILES


@pytest.mark.parametrize("version", ["no_reminder_v1", "no_reminder_v2"])
def test_training_eval_text_version(tiny, monkeypatch, selfname_cards, version):
    from types import SimpleNamespace

    data = FakeData()
    data.text_version = version
    data.fit_texts_changed = 3 if version == "no_reminder_v2" else 0
    cards = selfname_cards[selfname_cards["text_status"] == "ok"]
    spec = {"benchmark_id": "unchanged"}
    bm = SimpleNamespace(spec=spec, ids=cards["oracle_id"].tolist(), texts=cards["text"].tolist())
    seen, events = [], []

    def encode(model, tok, texts, *args, **kwargs):
        seen.extend(texts)
        return np.tile([1., 0.], (len(texts), 1))

    monkeypatch.setattr(tr, "load_benchmark", lambda _: (bm, {"cards": cards}))
    monkeypatch.setattr(enc, "encode", encode)
    monkeypatch.setattr(tr, "evaluate", lambda *_: {"summary": {
        k: {"ndcg@10": .7} for k in ("full", "hard", "hard_lexical")}})
    monkeypatch.setattr(tr, "log_event", lambda *a, **kw: events.append((a, kw)))
    tr.train(_cfg(total_updates=1, eval_every=1, guardrail_hard_lexical=.5,
                  text_version=version), spec, dev=torch.device("cpu"), data=data)
    expected = tr.selfname.apply_text_version(cards, version)["text"].tolist()
    assert seen == expected and bm.spec == spec
    started = next(kw for a, kw in events if a == ("train", "STARTED"))
    assert started["text_version"] == version
    assert started["fit_texts_changed"] == data.fit_texts_changed
