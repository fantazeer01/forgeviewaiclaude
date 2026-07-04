import pytest

from core.signal_combiner import SignalCombiner


def make_market(market_id="m1", asset="BTC", yes_price=0.5, no_price=0.5,
                 up_token_id="up-tok", volume_24h=1000.0, minutes_remaining=3.0):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price, "no_price": no_price,
        "up_token_id": up_token_id, "volume_24h": volume_24h, "minutes_remaining": minutes_remaining,
    }


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
    result = combiner.combine(make_market(), FakeFetcher(top=None), btc_eth_correlation=None)
    assert result is None


def test_order_book_alone_above_threshold_fires(combiner):
    # ratio 4.0 -> order_book confidence min(0.95, 4.0/4.0) = 0.95 (capped),
    # weight 0.25 renormalized to 1.0 since it's the only active signal ->
    # combined = 0.95 > 0.60
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    result = combiner.combine(make_market(), FakeFetcher(top), btc_eth_correlation=None)
    assert result is not None
    assert result.confidence == pytest.approx(0.95)
    assert result.direction == "YES"


def test_order_book_alone_below_threshold_does_not_fire(combiner):
    # ratio 2.5 -> confidence min(0.95, 2.5/4.0) = 0.625, still above 0.60 --
    # use a ratio that lands the confidence below 0.60 instead
    top = {"total_bid_depth": 210.0, "total_ask_depth": 100.0}  # ratio 2.1 -> confidence 0.525
    result = combiner.combine(make_market(), FakeFetcher(top), btc_eth_correlation=None)
    assert result is None


def test_weighted_average_renormalizes_among_active_signals_only(combiner, mocker):
    # order_book (weight 0.25, conf=0.75 from ratio 3.0) + momentum (weight
    # 0.25, conf=0.9, mocked) active; volume inactive. Renormalized weighted
    # avg over just the active two: (0.25*0.75 + 0.25*0.9) / (0.25+0.25) =
    # 0.825 -- NOT diluted by volume's weight, which a naive zero-padded
    # weighted sum would do.
    top = {"total_bid_depth": 300.0, "total_ask_depth": 100.0}  # ratio 3.0 -> confidence 0.75
    mocker.patch.object(combiner.momentum_gen, "generate", return_value=_fake_signal(0.9))
    result = combiner.combine(make_market(), FakeFetcher(top), btc_eth_correlation=None)
    assert result is not None
    expected = (0.25 * 0.75 + 0.25 * 0.9) / (0.25 + 0.25)
    assert result.confidence == pytest.approx(expected, abs=0.001)


def _fake_signal(confidence):
    from core.repricing_detector import RepricingSignal
    return RepricingSignal(asset="BTC", market_id="m1", direction="YES",
                            yes_price=0.5, no_price=0.5, confidence=confidence, reason="mock")


def test_repricing_is_never_referenced_in_status(combiner, tmp_path):
    import json
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    combiner.combine(make_market(), FakeFetcher(top), btc_eth_correlation=None)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert "repricing" not in data["BTC"]


def test_correlation_filter_blocks_eth_regardless_of_other_signals(combiner):
    combiner.correlation_filter.update_btc_price(0.50)
    combiner.correlation_filter.update_btc_price(0.40)  # BTC dropped hard
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}  # would otherwise fire at 1.0
    result = combiner.combine(
        make_market(asset="ETH"), FakeFetcher(top),
        btc_eth_correlation=0.9,  # high correlation
    )
    assert result is None


def test_correlation_filter_does_not_block_btc(combiner):
    combiner.correlation_filter.update_btc_price(0.50)
    combiner.correlation_filter.update_btc_price(0.40)
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    result = combiner.combine(
        make_market(asset="BTC"), FakeFetcher(top),
        btc_eth_correlation=0.9,
    )
    assert result is not None


def test_status_exported_after_combine(combiner, tmp_path):
    import json
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    combiner.combine(make_market(), FakeFetcher(top), btc_eth_correlation=None)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert "BTC" in data
    assert data["BTC"]["order_book"]["fired"] is True
    assert data["BTC"]["fired"] is True


def test_status_shows_blocked_when_correlation_filter_fires(combiner):
    import json
    combiner.correlation_filter.update_btc_price(0.50)
    combiner.correlation_filter.update_btc_price(0.40)
    combiner.combine(make_market(asset="ETH"), FakeFetcher(top=None), btc_eth_correlation=0.9)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert data["ETH"]["correlation_filter_blocked"] is True
    assert data["ETH"]["fired"] is False
