import math

import pandas as pd
import pytest

from mtgemb.data import (
    build_families,
    card_row,
    check_raw_cards,
    corpus_reason,
    face_list,
    idf_weights,
    split_of,
    tag_reason,
)


def card(oid, text, name=None, status="ok"):
    return {
        "oracle_id": oid,
        "name": name or oid,
        "text": text,
        "text_status": status,
        "faces": "[]",
    }


def test_corpus_policy_layout_paper_and_vintage_legality():
    for layout in ("token", "double_faced_token", "emblem", "art_series"):
        policy_card = {"layout": layout, "games": ["paper"], "legalities": {"vintage": "legal"}}
        assert corpus_reason(policy_card) == f"layout:{layout}"
    non_paper = {"layout": "normal", "games": ["arena"], "legalities": {"vintage": "legal"}}
    not_legal = {"layout": "normal", "games": ["paper"], "legalities": {"vintage": "not_legal"}}
    assert corpus_reason(non_paper) == "not_paper"
    assert corpus_reason(not_legal) == "not_tournament_legal"
    for legality in ("legal", "restricted", "banned"):
        included = {"layout": "normal", "games": ["paper"], "legalities": {"vintage": legality}}
        assert corpus_reason(included) is None


def test_check_raw_cards_duplicate_and_core_field_validation():
    original = {"oracle_id": "a", "name": "A", "layout": "normal", "oracle_text": "Draw."}
    check_raw_cards([original, original.copy()])
    with pytest.raises(ValueError, match="conflicting duplicate oracle_id"):
        check_raw_cards([original, {**original, "oracle_text": "Different."}])
    with pytest.raises(ValueError, match="without oracle_id"):
        check_raw_cards([{"name": "A", "layout": "normal"}])
    with pytest.raises(ValueError, match="empty card_faces"):
        check_raw_cards([{**original, "card_faces": []}])


def test_faces_card_row_preserve_order_and_use_ordered_face_contract():
    source = {
        "oracle_id": "multi",
        "name": "Front // Back",
        "layout": "transform",
        "card_faces": [
            {"name": "Front", "type_line": "Creature — Human", "oracle_text": "Front does X."},
            {"name": "Back", "type_line": "Land", "oracle_text": "Back does Y."},
        ],
        "type_line": "Creature // Land",
        "color_identity": ["R", "W", "B"],
    }
    faces = face_list(source)
    assert [f["name"] for f in faces] == ["Front", "Back"]
    row = card_row(source)
    assert row["is_creature"] is True
    assert row["color_identity"] == "WBR"
    assert row["text"] == "Front does X.\n//\nBack does Y."
    source["card_faces"][0]["type_line"] = "Instant"
    source["card_faces"][1]["type_line"] = "Creature — Beast"
    assert card_row(source)["is_creature"] is False


def test_tag_reason_keeps_functional_examples_and_drops_metadata():
    for label in ("removal-creature", "ramp", "name matters"):
        assert tag_reason(label) is None
    for label in (
        "cycle-m19-foo", "alliteration", "type errata human", "unique type line",
        "punny name", "burn with set's mechanic", "un-keyword",
    ):
        assert tag_reason(label) is not None


@pytest.mark.xfail(
    reason="tag_reason currently drops 'tutors by name' via its generic name suffix rule",
    strict=True,
)
def test_tag_reason_keeps_tutors_by_name():
    assert tag_reason("tutors by name") is None


def test_build_families_transitive_edges_empty_text_and_meld():
    cards = pd.DataFrame([
        card("A", "Target CARDNAME draws a card.", "Alpha"),
        card("B", "Shared exact text."),
        card("C", "Target CARDNAME draws a card.", "Gamma"),
        card("D", "", status="empty"),
        card("E", "", status="empty"),
        card("F", "Meld half one."),
        card("G", "Meld half two."),
    ])
    families, edges = build_families(cards, [("G", "F")])
    assert families["A"] == families["C"]  # self-name template
    # Add exact-text bridge for transitive connectivity.
    cards.loc[cards.oracle_id == "B", "text"] = "Target CARDNAME draws a card."
    families, edges = build_families(cards, [("G", "F")])
    assert families["A"] == families["B"] == families["C"] == "A"
    assert families["D"] != families["E"]
    assert families["F"] == families["G"] == "F"
    assert any(reason in {"exact_text", "self_name_template"} for _, _, reason in edges)
    assert ("F", "G", "meld") in edges


def test_split_is_stable_component_scoped_and_balanced():
    assert split_of("family-x") == split_of("family-x")
    assert {split_of("family-x")} <= {"fit", "dev", "sealed"}
    counts = {name: 0 for name in ("fit", "dev", "sealed")}
    n = 5000
    for i in range(n):
        counts[split_of(f"random-family-{i}")] += 1
    assert abs(counts["fit"] / n - 0.8) < 0.02
    assert abs(counts["dev"] / n - 0.1) < 0.02
    assert abs(counts["sealed"] / n - 0.1) < 0.02


def test_idf_weights_use_fit_memberships_and_omit_unknown_tags():
    memberships = pd.DataFrame([
        ("a", "common"), ("b", "common"), ("a", "rare"), ("c", "only-nonfit"),
    ], columns=["oracle_id", "tag_id"])
    weights = idf_weights(memberships, {"a", "b"})
    assert weights["common"] == pytest.approx(1 + math.log(3 / 3))
    assert weights["rare"] == pytest.approx(1 + math.log(3 / 2))
    assert "only-nonfit" not in weights

