from collections import Counter

import pandas as pd
import pytest

from mtgemb.pairs import build_pairs, candidate_pairs, drop_modifier_tags, lexical_negatives


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



def test_drop_modifier_tags_removes_listed_labels_and_rejects_unknown():
    mem = pd.DataFrame([("a", "t-cost"), ("a", "t-fx"), ("b", "t-cost")],
                       columns=["oracle_id", "tag_id"])
    tags = pd.DataFrame({"tag_id": ["t-cost", "t-fx"],
                         "label": ["manaless value", "change target"]})
    kept, dropped = drop_modifier_tags(mem, tags, ["manaless value"])
    assert kept.to_dict("records") == [{"oracle_id": "a", "tag_id": "t-fx"}]
    assert dropped == {"t-cost"}
    with pytest.raises(ValueError, match="no tag labelled"):
        drop_modifier_tags(mem, tags, ["not a tag"])


def test_effect_only_labels_turn_a_rider_twin_into_a_negative():
    # Deflecting Swat shape: s shares only a cost rider with h and the effect with r.
    texts = {"s": "free if you control a commander. change the target of target spell",
             "h": "free if you control a commander. prevent all combat damage this turn",
             "r": "you may choose new targets for target spell",
             "x": "destroy target artifact"}
    cards = pd.DataFrame([{"oracle_id": o, "split": "fit", "text_status": "ok",
                           "family_id": f"f-{o}", "text": t} for o, t in texts.items()])
    mem = pd.DataFrame([("s", "rider"), ("h", "rider"), ("s", "fx"), ("r", "fx"), ("x", "art"),
                        ("h", "fog")], columns=["oracle_id", "tag_id"]).assign(weight="median")
    tags = pd.DataFrame({"tag_id": ["rider", "fx", "art", "fog"],
                         "label": ["synergy-commander", "change target", "art", "fog"],
                         "idf_fit": 1.0})
    cfg = {"seed": "t", "min_df": 2, "max_df": 9, "rounds": 3, "card_cap": 9,
           "family_cap": 9, "spec_clip": [0.5, 2.0]}
    before, _, _ = build_pairs(cards, mem, tags, cfg)
    assert {tuple(p) for p in before[["a", "b"]].values} == {("h", "s"), ("r", "s")}
    kept, _ = drop_modifier_tags(mem, tags, ["synergy-commander"])
    after, elig, _ = build_pairs(cards, kept, tags, cfg)
    assert {tuple(p) for p in after[["a", "b"]].values} == {("r", "s")}
    assert elig == {"fx"}
    neg_before, _ = lexical_negatives(cards, mem, k=3)
    neg_after, _ = lexical_negatives(cards, kept, k=3)
    assert "h" not in set(neg_before.loc[neg_before.anchor == "s", "negative"])
    assert neg_after.loc[neg_after.anchor == "s"].iloc[0]["negative"] == "h"
