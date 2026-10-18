from mtgemb import augment


def test_flip_swaps_one_mechanism_word_and_keeps_case():
    out = augment.flip("Tap target creature. It doesn't untap.", seed=0, key="a")
    assert out in {"Untap target creature. It doesn't untap.",
                   "Tap target land. It doesn't untap.",
                   "Tap target creature. It doesn't tap."}
    assert augment.flip("Counter target spell.", 0, "a") is None
    assert augment.flip("Put a +1/+1 counter on it.", 0, "a") == "Put a -1/-1 counter on it."
    # "you" is never flipped (ungrammatical results)
    assert augment.flip("you control", 0, "a") is None
    assert augment.flip("Tap it. Untap it.", 0, "k") == augment.flip("Tap it. Untap it.", 0, "k")


def test_numbers_bumps_free_numbers_only():
    assert augment.numbers("Draw two cards. Deal 3 damage.") == "Draw three cards. Deal 4 damage."
    assert augment.numbers("Put two +1/+1 counters on a 2/2 creature.") == (
        "Put three +1/+1 counters on a 2/2 creature.")
    assert augment.numbers("One target creature gets {2}.") == "Two target creature gets {3}."
    assert augment.numbers("Draw a card.") is None


def test_wording_uses_older_oracle_phrases():
    assert augment.wording("When this creature enters, draw a card. It can't be blocked.") == (
        "When this creature enters the battlefield, draw a card. It is unblockable.")
    assert augment.wording("Whenever a creature dies, its mana value matters.") == (
        "Whenever a creature is put into a graveyard from the battlefield, its converted "
        "mana cost matters.")
    assert augment.wording("When this enters the battlefield, scry 1.") is None


def test_opener_negatives_need_same_opener_other_family_no_shared_tag():
    ids = ["a", "b", "c", "d", "e"]
    texts = ["When this creature enters, draw a card.", "When this creature enters, gain 3 life.",
             "When this creature enters, draw two cards.", "When this creature enters, scry 1.",
             "Flying\nWhen this creature enters, scry 2."]
    fams = ["f1", "f2", "f3", "f1", "f5"]
    tags = {"a": frozenset({"draw"}), "b": frozenset({"life"}), "c": frozenset({"draw"}),
            "d": frozenset({"scry"}), "e": frozenset({"scry"})}
    neg = augment.opener_negatives(ids, texts, fams, tags, seed=0)
    assert neg["a"] == "b"  # c shares "draw", d is a's family, e has two lines
    assert neg["b"] in {"a", "c", "d"} and neg["d"] in {"b", "c"}
    assert "e" not in neg
