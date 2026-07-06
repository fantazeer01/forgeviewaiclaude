import pytest

from config.settings import ORDER_BOOK_RATIO_THRESHOLD
from core.signals.order_book_signal import OrderBookSignalGenerator


class FakeFetcher:
    def __init__(self, top=None):
        self._top = top

    def get_order_book_top(self, token_id):
        return self._top


def make_market(market_id="m1", asset="BTC", yes_price=0.5, no_price=0.5,
                 up_token_id="up-tok", minutes_remaining=3.0):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price,
        "no_price": no_price, "up_token_id": up_token_id, "minutes_remaining": minutes_remaining,
    }


def test_fires_when_bid_ask_ratio_exceeds_threshold():
    top = {"total_bid_depth": 300.0, "total_ask_depth": 100.0}  # ratio 3.0
    gen = OrderBookSignalGenerator()
    signal = gen.generate(make_market(), FakeFetcher(top))
    assert signal is not None
    assert signal.direction == "YES"
    assert signal.confidence == pytest.approx(min(0.95, 3.0 / 4.0))


def test_no_signal_when_ratio_at_or_below_threshold():
    top = {"total_bid_depth": ORDER_BOOK_RATIO_THRESHOLD * 100.0, "total_ask_depth": 100.0}  # exactly at threshold, not >
    gen = OrderBookSignalGenerator()
    assert gen.generate(make_market(), FakeFetcher(top)) is None


def test_fires_on_a_realistic_live_ratio_that_the_old_2_0_threshold_would_have_missed():
    # 2026-07-06: live polling found real BTC/ETH/SOL depth ratios in the
    # 0.6-1.1 range, never near the old 2.0 threshold. 1.5 is representative
    # of a real-but-modest imbalance that should now fire under the new
    # ORDER_BOOK_RATIO_THRESHOLD (1.3) but would not have under the old one.
    top = {"total_bid_depth": 150.0, "total_ask_depth": 100.0}  # ratio 1.5
    gen = OrderBookSignalGenerator()
    signal = gen.generate(make_market(), FakeFetcher(top))
    assert signal is not None
    assert ORDER_BOOK_RATIO_THRESHOLD < 1.5 < 2.0


def test_confidence_capped_at_095():
    top = {"total_bid_depth": 10000.0, "total_ask_depth": 10.0}  # huge ratio
    gen = OrderBookSignalGenerator()
    signal = gen.generate(make_market(), FakeFetcher(top))
    assert signal.confidence == 0.95


def test_no_signal_without_order_book():
    gen = OrderBookSignalGenerator()
    assert gen.generate(make_market(), FakeFetcher(top=None)) is None


def test_no_signal_without_up_token_id():
    gen = OrderBookSignalGenerator()
    market = make_market(up_token_id=None)
    assert gen.generate(market, FakeFetcher({"total_bid_depth": 500, "total_ask_depth": 100})) is None


def test_no_signal_when_ask_depth_zero():
    gen = OrderBookSignalGenerator()
    top = {"total_bid_depth": 500.0, "total_ask_depth": 0.0}
    assert gen.generate(make_market(), FakeFetcher(top)) is None
