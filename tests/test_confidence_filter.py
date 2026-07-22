from layer3_conscience import confidence_filter
from config.settings import CONFIDENCE_YES_THRESHOLD, CONFIDENCE_NO_THRESHOLD, HIGH_VOLATILITY_CONFIDENCE_THRESHOLD


# 9. Confidence filter blocks trading in the 0.43-0.57 hold zone.
def test_hold_zone_blocks_trading():
    for p_up in (0.43, 0.50, 0.57):
        result = confidence_filter.decide_side(p_up)
        assert result["decision"] == "HOLD"


def test_yes_above_threshold():
    result = confidence_filter.decide_side(CONFIDENCE_YES_THRESHOLD + 0.01)
    assert result["decision"] == "YES"


def test_no_below_threshold():
    result = confidence_filter.decide_side(CONFIDENCE_NO_THRESHOLD - 0.01)
    assert result["decision"] == "NO"


def test_no_prediction_holds():
    result = confidence_filter.decide_side(None)
    assert result["decision"] == "HOLD"
    assert result["reason"] == "no_prediction"


# 13. In HIGH_VOLATILITY, the confidence bar is at least as strict as normal.
# CONFIDENCE_YES_THRESHOLD was raised to 0.60 on 2026-07-22, which happens to
# now equal HIGH_VOLATILITY_CONFIDENCE_THRESHOLD -- so the YES side no longer
# demonstrates a difference. The NO side still does: HIGH_VOLATILITY's NO
# bar is 1-0.60=0.40, stricter than the normal 0.43.
def test_high_volatility_yes_threshold_matches_or_exceeds_normal():
    assert HIGH_VOLATILITY_CONFIDENCE_THRESHOLD >= CONFIDENCE_YES_THRESHOLD
    assert confidence_filter.decide_side(HIGH_VOLATILITY_CONFIDENCE_THRESHOLD + 0.01, regime="HIGH_VOLATILITY")["decision"] == "YES"


def test_high_volatility_no_threshold_is_stricter():
    p_up = 0.415  # below the normal NO bar (0.43), above the high-vol one (0.40)
    normal = confidence_filter.decide_side(p_up, regime=None)
    volatile = confidence_filter.decide_side(p_up, regime="HIGH_VOLATILITY")
    assert normal["decision"] == "NO"
    assert volatile["decision"] == "HOLD"
