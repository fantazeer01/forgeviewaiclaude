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


# 13. In HIGH_VOLATILITY, the threshold rises to 0.60.
def test_high_volatility_raises_threshold():
    p_up = 0.58  # would be YES normally, but not at the raised 0.60 bar
    normal = confidence_filter.decide_side(p_up, regime=None)
    volatile = confidence_filter.decide_side(p_up, regime="HIGH_VOLATILITY")
    assert normal["decision"] == "YES"
    assert volatile["decision"] == "HOLD"

    assert confidence_filter.decide_side(HIGH_VOLATILITY_CONFIDENCE_THRESHOLD + 0.01, regime="HIGH_VOLATILITY")["decision"] == "YES"
