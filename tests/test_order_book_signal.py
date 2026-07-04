import pytest

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
    top = {"total_bid_depth": 200.0, "total_ask_depth": 100.0}  # ratio 2.0, not > 2.0
    gen = OrderBookSignalGenerator()
    assert gen.generate(make_market(), FakeFetcher(top)) is None


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
