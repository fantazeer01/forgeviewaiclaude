from layer3_conscience import liquidity_filter
from config.settings import MIN_BOOK_DEPTH_USD, MAX_BID_ASK_SPREAD_PCT


# 10. Liquidity filter blocks trading when book depth is too low.
def test_blocks_low_depth():
    snapshot = {"book_depth_yes": 20.0, "book_depth_no": 10.0, "bid_ask_spread": 0.01}
    assert 20.0 + 10.0 < MIN_BOOK_DEPTH_USD  # fixture is deliberately below the configured minimum
    ok, reason = liquidity_filter.passes(snapshot)
    assert ok is False
    assert reason == "low_liquidity"


def test_passes_with_enough_depth():
    snapshot = {"book_depth_yes": MIN_BOOK_DEPTH_USD, "book_depth_no": MIN_BOOK_DEPTH_USD, "bid_ask_spread": 0.01}
    ok, reason = liquidity_filter.passes(snapshot)
    assert ok is True


def test_blocks_wide_spread():
    snapshot = {"book_depth_yes": 500.0, "book_depth_no": 500.0, "bid_ask_spread": MAX_BID_ASK_SPREAD_PCT + 0.01}
    ok, reason = liquidity_filter.passes(snapshot)
    assert ok is False
    assert reason == "wide_spread"
