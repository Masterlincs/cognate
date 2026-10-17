from collections import Counter

import pandas as pd
import pytest

from mtgemb.pairs import build_pairs, candidate_pairs, lexical_negatives


def test_candidate_pairs_all_for_tiny_tag_and_seeded_for_larger_tag():
    tiny = candidate_pairs(["a", "b", "c"], rounds=3, key="tiny")
    assert {(a, b) for a, b, _ in tiny} == {("a", "b"), ("a", "c"), ("b", "c")}
    members = [f"id-{i}" for i in range(20)]
    first = candidate_pairs(members, rounds=3, key="seed-a")
    assert first == candidate_pairs(members, rounds=3, key="seed-a")
    assert first != candidate_pairs(members, rounds=3, key="seed-b")
    assert all(a < b and a != b for a, b, _ in first)


def test_build_pairs_fit_filters_caps_df_and_specificity_weight():
    ids = [f"c{i}" for i in range(8)]
    cards = pd.DataFrame([
        {"oracle_id": oid, "split": "fit", "text_status": "ok", "family_id": f"f{i}",
         "text": f"text {i}"}
        for i, oid in enumerate(ids)
    ] + [
        {"oracle_id": "dev", "split": "dev", "text_status": "ok", "family_id": "fd", "text": "dev"},
        {"oracle_id": "bad-text", "split": "fit", "text_status": "empty", "family_id": "fb",
         "text": ""},
    ])
    rows = [(oid, "small", "very_strong") for oid in ids[:4]]
    rows += [(oid, "broad", "very_strong") for oid in ids[:7]]
    rows += [("dev", "small", "very_strong"), ("bad-text", "small", "very_strong")]
    mem = pd.DataFrame(rows, columns=["oracle_id", "tag_id", "weight"])
    tags = pd.DataFrame([
        {"tag_id": "small", "idf_fit": 3.0, "label": "Small"},
        {"tag_id": "broad", "idf_fit": 1.0, "label": "Broad"},
    ])
    cfg = {"seed": "test", "min_df": 2, "max_df": 6, "rounds": 3,
           "card_cap": 2, "family_cap": 20, "neg_pool": 5, "spec_clip": [0.5, 2.0]}
    pairs, eligible, stats = build_pairs(cards, mem, tags, cfg)
    again, _, _ = build_pairs(cards, mem, tags, cfg)
    assert pairs.equals(again)
    assert eligible == {"small"}
    assert set(pairs["tag_id"]) == {"small"}
    assert set(pairs["split"]) == {"fit"}
    assert not {"dev", "bad-text"} & (set(pairs["a"]) | set(pairs["b"]))
    assert all(pairs["a"] < pairs["b"])
    assert not any(pairs.duplicated(["a", "b"]))
    assert not any(pairs.duplicated(["b", "a"]))
    exposure = Counter(pairs["a"].tolist() + pairs["b"].tolist())
    assert max(exposure.values()) <= 2
    assert pairs["spec_weight"].mean() == pytest.approx(1.0)
    assert pairs["spec_weight"].between(0.5, 2.0).all()
    assert stats["card_exposure_max"] <= 2


def test_lexical_negatives_exclude_shared_tags_family_self_and_untagged():
    ids = [f"c{i}" for i in range(8)]
    cards = pd.DataFrame([
        {"oracle_id": oid, "split": "fit", "text_status": "ok", "family_id": f"family-{i // 2}",
         "text": f"draw cards quickly number {i} with unique phrase"}
        for i, oid in enumerate(ids)
    ] + [
        {"oracle_id": "dev", "split": "dev", "text_status": "ok", "family_id": "dev-family",
         "text": "draw cards"},
    ])
    memberships = pd.DataFrame([
        (oid, f"tag-{i}") for i, oid in enumerate(ids)
    ] + [("dev", "tag-0")], columns=["oracle_id", "tag_id"])
    negatives, _ = lexical_negatives(cards, memberships, k=2)
    assert set(negatives["anchor"]) == set(ids)
    assert set(negatives["negative"]) <= set(ids)
    assert not set(negatives["negative"]) & {"dev", "untagged"}
    tag_by_id = dict(zip(memberships.oracle_id, memberships.tag_id, strict=True))
    family_by_id = dict(zip(cards.oracle_id, cards.family_id, strict=True))
    for row in negatives.itertuples():
        assert row.anchor != row.negative
        assert family_by_id[row.anchor] != family_by_id[row.negative]
        assert tag_by_id[row.anchor] != tag_by_id[row.negative]
    assert set(negatives.groupby("anchor")["rank"].min()) == {1}



