from mtgemb.report import advances, promotion


def test_advancement_thresholds_and_guardrail():
    base = {"full": 0.600, "hard_lexical": 0.550, "guardrail_passed": True}
    assert advances({**base, "full": 0.6050}, base)
    assert not advances({**base, "full": 0.6049}, base)
    assert not advances({**base, "full": 0.62, "hard_lexical": 0.5449}, base)
    assert not advances({**base, "full": 0.62, "guardrail_passed": False}, base)


def test_promotion_bounds_are_strict():
    ok_f = {"point": 0.005, "lower_adjusted": 1e-9}
    ok_h = {"point": 0.0, "lower_adjusted": -0.0099}
    assert promotion(ok_f, ok_h)
    assert not promotion({**ok_f, "lower_adjusted": 0.0}, ok_h)
    assert not promotion(ok_f, {**ok_h, "lower_adjusted": -0.010})
    assert not promotion({**ok_f, "point": 0.0049}, ok_h)
