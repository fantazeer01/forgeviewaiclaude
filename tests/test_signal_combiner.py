import pytest

from core.signal_combiner import SignalCombiner
from core.repricing_detector import RepricingSignal


def make_market(market_id="m1", asset="BTC", yes_price=0.5, no_price=0.5,
                 up_token_id="up-tok", volume_24h=1000.0, minutes_remaining=3.0):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price, "no_price": no_price,
        "up_token_id": up_token_id, "volume_24h": volume_24h, "minutes_remaining": minutes_remaining,
    }


def make_repricing_signal(confidence=0.7, asset="BTC", market_id="m1"):
    return RepricingSignal(asset=asset, market_id=market_id, direction="YES",
                            yes_price=0.5, no_price=0.5, confidence=confidence, reason="test repricing")


class FakeFetcher:
    def __init__(self, top=None):
        self._top = top

    def get_order_book_top(self, token_id):
        return self._top


@pytest.fixture
def combiner(tmp_path):
    from core.signals.volume_signal import VolumeSignalGenerator
    return SignalCombiner(
        volume_gen=VolumeSignalGenerator(history_path=str(tmp_path / "volume_history.jsonl")),
        status_path=str(tmp_path / "signal_combiner_status.json"),
    )


def test_returns_none_when_no_signal_fires(combiner):
    result = combiner.combine(make_market(), FakeFetcher(top=None), repricing_signal=None, btc_eth_correlation=None)
    assert result is None


def test_repricing_alone_above_threshold_fires(combiner):
    # repricing alone at 0.7 confidence, weight 0.35 renormalized to 1.0 since
    # it's the only active signal -> combined confidence = 0.7 > 0.60
    signal = make_repricing_signal(confidence=0.7)
    result = combiner.combine(make_market(), FakeFetcher(top=None), repricing_signal=signal, btc_eth_correlation=None)
    assert result is not None
    assert result.confidence == pytest.approx(0.7)
    assert result.direction == "YES"


def test_repricing_alone_below_threshold_does_not_fire(combiner):
    signal = make_repricing_signal(confidence=0.5)
    result = combiner.combine(make_market(), FakeFetcher(top=None), repricing_signal=signal, btc_eth_correlation=None)
    assert result is None


def test_weighted_average_renormalizes_among_active_signals_only(combiner):
    # repricing (weight 0.35, conf=0.9) + order_book (weight 0.25, conf=0.75
    # from a 3.0 bid/ask ratio: min(0.95, 3.0/4.0)=0.75) active; momentum/
    # volume inactive. Renormalized weighted avg over just the active two:
    # (0.35*0.9 + 0.25*0.75) / (0.35+0.25) = 0.8375 -- NOT diluted toward a
    # lower number by the two inactive signals' weight, which a naive
    # zero-padded weighted sum would do.
    top = {"total_bid_depth": 300.0, "total_ask_depth": 100.0}  # ratio 3.0 -> fires at confidence 0.75
    signal = make_repricing_signal(confidence=0.9)
    result = combiner.combine(make_market(), FakeFetcher(top), repricing_signal=signal, btc_eth_correlation=None)
    assert result is not None
    expected = (0.35 * 0.9 + 0.25 * 0.75) / (0.35 + 0.25)
    assert result.confidence == pytest.approx(expected, abs=0.001)


def test_correlation_filter_blocks_eth_regardless_of_other_signals(combiner):
    combiner.correlation_filter.update_btc_price(0.50)
    combiner.correlation_filter.update_btc_price(0.40)  # BTC dropped hard
    signal = make_repricing_signal(confidence=0.95, asset="ETH")
    result = combiner.combine(
        make_market(asset="ETH"), FakeFetcher(top=None),
        repricing_signal=signal, btc_eth_correlation=0.9,  # high correlation
    )
    assert result is None


def test_correlation_filter_does_not_block_btc(combiner):
    combiner.correlation_filter.update_btc_price(0.50)
    combiner.correlation_filter.update_btc_price(0.40)
    signal = make_repricing_signal(confidence=0.9, asset="BTC")
    result = combiner.combine(
        make_market(asset="BTC"), FakeFetcher(top=None),
        repricing_signal=signal, btc_eth_correlation=0.9,
    )
    assert result is not None


def test_status_exported_after_combine(combiner, tmp_path):
    import json
    signal = make_repricing_signal(confidence=0.9)
    combiner.combine(make_market(), FakeFetcher(top=None), repricing_signal=signal, btc_eth_correlation=None)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert "BTC" in data
    assert data["BTC"]["repricing"]["fired"] is True
    assert data["BTC"]["fired"] is True


def test_status_shows_blocked_when_correlation_filter_fires(combiner):
    import json
    combiner.correlation_filter.update_btc_price(0.50)
    combiner.correlation_filter.update_btc_price(0.40)
    signal = make_repricing_signal(confidence=0.95, asset="ETH")
    combiner.combine(make_market(asset="ETH"), FakeFetcher(top=None), repricing_signal=signal, btc_eth_correlation=0.9)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert data["ETH"]["correlation_filter_blocked"] is True
    assert data["ETH"]["fired"] is False
