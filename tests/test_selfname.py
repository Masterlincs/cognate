import pandas as pd
import pytest

from mtgemb import selfname
from mtgemb.abilities import split_abilities


def test_apply_changes_only_ok_texts_preserving_lines_and_input(selfname_cards):
    before = selfname_cards.copy(deep=True)
    v1 = selfname.apply_text_version(selfname_cards, "no_reminder_v1")
    pd.testing.assert_frame_equal(v1, before)
    assert v1 is not selfname_cards
    v2 = selfname.apply_text_version(selfname_cards, "no_reminder_v2")
    pd.testing.assert_frame_equal(v2.drop(columns="text"), before.drop(columns="text"))
    assert v2.loc[v2["text_status"] == "ok", "text"].tolist() == [
        "Flying\nThis creature draws two cards.", "This creature gains 2 life.",
        "This creature loses 2 life.", 'Create a token named Other.',
        "This creature gains 2 life.\n//\nThis creature draws two cards."]
    not_ok = before["text_status"] != "ok"
    pd.testing.assert_series_equal(v2.loc[not_ok, "text"], before.loc[not_ok, "text"])
    for a, b in zip(before.loc[~not_ok, "text"], v2.loc[~not_ok, "text"], strict=True):
        assert len(split_abilities(a)) == len(split_abilities(b))
    pd.testing.assert_frame_equal(selfname_cards, before)
    with pytest.raises(ValueError, match="unknown text version"):
        selfname.apply_text_version(selfname_cards, "v3")


@pytest.mark.parametrize("bad, message", [("One line.", "ability-line count"),
                                         (None, "v2 text is not ok")])
def test_apply_rejects_changed_line_count_or_status(selfname_cards, monkeypatch, bad, message):
    original = selfname.v2_texts

    def broken(cards):
        texts = original(cards)
        texts.iloc[0] = bad
        return texts

    monkeypatch.setattr(selfname, "v2_texts", broken)
    with pytest.raises(ValueError, match=message):
        selfname.apply_text_version(selfname_cards, "no_reminder_v2")
