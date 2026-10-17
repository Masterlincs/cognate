from mtgemb.text import (
    FACE_BOUNDARY,
    jaccard,
    no_reminder_v1,
    normalise_whitespace,
    strip_parentheticals,
    word_tokens,
)


def test_simple_reminder_removed():
    r = no_reminder_v1(["Flying (This creature can't be blocked except by creatures with flying.)"])
    assert r["text"] == "Flying" and r["status"] == "ok" and r["spans"] == 1


def test_nested_spans_removed_whole():
    s, info = strip_parentheticals("A (b (c) d) e")
    assert s == "A  e" and info["spans"] == 1 and info["nested"] == 1
    assert no_reminder_v1(["A (b (c) d) e"])["text"] == "A e"


def test_unbalanced_quarantined_not_truncated():
    for t in ["Draw a card (then discard", "Draw a card) then discard"]:
        r = no_reminder_v1([t])
        assert r["status"] == "quarantined" and r["text"] is None
        assert r["raw"] == t


def test_keeps_symbols_negation_quantities():
    t = "{T}: Add {G}{G}. Target creature can't block this turn. Draw 3 cards."
    assert no_reminder_v1([t])["text"] == t


def test_whole_line_parenthetical_and_empty_output():
    r = no_reminder_v1(["({T}: Add {G}.)"])
    assert r["status"] == "empty" and r["text"] == ""
    assert r["reason"] == "only parenthetical text" and r["whole_line_spans"] == 1
    r = no_reminder_v1([""])
    assert r["status"] == "empty" and r["reason"] == "no rules text"


def test_faces_ordered_and_empty_face_dropped():
    r = no_reminder_v1(["Front text.", "Back (reminder) text."])
    assert r["text"] == "Front text." + FACE_BOUNDARY + "Back text."
    r = no_reminder_v1(["Front.", "(Only reminder.)"])
    assert r["text"] == "Front."
    assert no_reminder_v1(["B", "A"])["text"] != no_reminder_v1(["A", "B"])["text"]


def test_whitespace_normalisation():
    assert normalise_whitespace("a\r\n\r\n  b   c ,d .\re") == "a\nb c,d.\ne"


def test_jaccard():
    a, b = word_tokens("Draw a card."), word_tokens("draw A CARD")
    assert jaccard(a, b) == 1.0
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert jaccard(word_tokens("a b"), word_tokens("b c")) == 1 / 3
