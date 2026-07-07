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
    assert result.decisive_signal == "order_book"


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
    assert result.decisive_signal == "combined"


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


def test_price_stability_filter_blocks_a_flat_market(combiner):
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}  # would otherwise fire at 0.95
    combiner.combine(make_market(yes_price=0.500), FakeFetcher(top), btc_eth_correlation=None)
    result = combiner.combine(make_market(yes_price=0.501), FakeFetcher(top), btc_eth_correlation=None)  # flat
    assert result is None


def test_price_stability_filter_does_not_block_a_moving_market(combiner):
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    combiner.combine(make_market(yes_price=0.50), FakeFetcher(top), btc_eth_correlation=None)
    result = combiner.combine(make_market(yes_price=0.53), FakeFetcher(top), btc_eth_correlation=None)  # moved 0.03
    assert result is not None


def test_price_stability_blocked_status_is_set(combiner):
    import json
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    combiner.combine(make_market(yes_price=0.500), FakeFetcher(top), btc_eth_correlation=None)
    combiner.combine(make_market(yes_price=0.501), FakeFetcher(top), btc_eth_correlation=None)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert data["BTC"]["price_stability_blocked"] is True
    assert data["BTC"]["fired"] is False
    assert data["BTC"]["order_book"]["fired"] is False  # signal never even evaluated


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


def test_signal_stats_increments_fired_today_and_last_fired_at(combiner):
    import json
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}  # order_book fires
    combiner.combine(make_market(), FakeFetcher(top), btc_eth_correlation=None)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert data["signal_stats"]["order_book"]["fired_today"] == 1
    assert data["signal_stats"]["order_book"]["last_fired_at"] is not None
    assert data["signal_stats"]["momentum"]["fired_today"] == 0
    assert data["signal_stats"]["momentum"]["last_fired_at"] is None
    assert data["signal_stats"]["volume"]["fired_today"] == 0


def test_signal_stats_accumulates_across_multiple_combine_calls(combiner):
    import json
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    # different market_ids -- a repeated identical price on the SAME market_id
    # would now get blocked by the price stability filter, which is tested
    # separately; this test is only about the fired_today counter accumulating.
    combiner.combine(make_market(market_id="m1"), FakeFetcher(top), btc_eth_correlation=None)
    combiner.combine(make_market(market_id="m2"), FakeFetcher(top), btc_eth_correlation=None)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert data["signal_stats"]["order_book"]["fired_today"] == 2


def test_price_below_band_blocks_combine_entirely(combiner):
    import json
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}  # would otherwise fire at 0.95
    result = combiner.combine(make_market(yes_price=0.20, no_price=0.80), FakeFetcher(top), btc_eth_correlation=None)
    assert result is None
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert data["BTC"]["price_out_of_band"] is True
    assert data["BTC"]["fired"] is False
    assert data["BTC"]["order_book"]["fired"] is False  # signal never even evaluated


def test_price_above_band_blocks_combine_entirely(combiner):
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    result = combiner.combine(make_market(yes_price=0.75, no_price=0.25), FakeFetcher(top), btc_eth_correlation=None)
    assert result is None


def test_price_at_band_edges_is_not_filtered(combiner):
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}
    result_low = combiner.combine(make_market(yes_price=0.45, no_price=0.55), FakeFetcher(top), btc_eth_correlation=None)
    assert result_low is not None
    result_high = combiner.combine(make_market(yes_price=0.60, no_price=0.40), FakeFetcher(top), btc_eth_correlation=None)
    assert result_high is not None


def test_extreme_low_price_no_longer_fires_anything(combiner):
    # DISABLED (2026-07-06): extreme mean-reversion (fire YES below 0.20,
    # NO above 0.80) was tried and removed after a 10.5% win rate over 19
    # NO-direction trades. yes_price=0.10 must now be blocked like any other
    # out-of-band price, even with no order book set up (top=None) and even
    # though it would have fired confidently under the old logic.
    result = combiner.combine(make_market(yes_price=0.10, no_price=0.90), FakeFetcher(top=None), btc_eth_correlation=None)
    assert result is None


def test_extreme_high_price_no_longer_fires_anything(combiner):
    result = combiner.combine(make_market(yes_price=0.90, no_price=0.10), FakeFetcher(top=None), btc_eth_correlation=None)
    assert result is None


def test_extreme_low_price_status_shows_blocked_not_fired(combiner):
    import json
    combiner.combine(make_market(yes_price=0.05, no_price=0.95), FakeFetcher(top=None), btc_eth_correlation=None)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert data["BTC"]["price_out_of_band"] is True
    assert data["BTC"]["fired"] is False
    assert "extreme_reversion" not in data["BTC"]


def test_signal_stats_persists_across_new_combiner_instance_same_day(tmp_path):
    import json
    from core.signals.volume_signal import VolumeSignalGenerator

    status_path = str(tmp_path / "signal_combiner_status.json")
    top = {"total_bid_depth": 400.0, "total_ask_depth": 100.0}

    c1 = SignalCombiner(
        volume_gen=VolumeSignalGenerator(history_path=str(tmp_path / "volume_history.jsonl")),
        status_path=status_path,
    )
    c1.combine(make_market(), FakeFetcher(top), btc_eth_correlation=None)

    c2 = SignalCombiner(
        volume_gen=VolumeSignalGenerator(history_path=str(tmp_path / "volume_history_2.jsonl")),
        status_path=status_path,
    )
    assert c2._signal_stats["order_book"]["fired_today"] == 1

    with open(status_path) as f:
        data = json.load(f)
    assert data["signal_stats"]["order_book"]["fired_today"] == 1


# ---------------- NO mean-reversion carve-out (2026-07-07 resurrection) ----------------

def test_no_signal_fires_on_real_reversion_from_extreme_peak(combiner, mocker):
    # NO_TRADING_ENABLED defaults to False (2026-07-07: disabled again after
    # real results confirmed a negative edge, 2/25 = 8.00% win rate). This
    # test exercises the underlying carve-out logic itself, so it explicitly
    # re-enables the flag at BOTH places that check it (defense-in-depth:
    # SignalCombiner's own call-site check, and the generator's internal
    # guard) rather than relying on a default that's deliberately off.
    mocker.patch("core.signal_combiner.NO_TRADING_ENABLED", True)
    mocker.patch("core.signals.mean_reversion_no_signal.NO_TRADING_ENABLED", True)
    combiner.combine(make_market(yes_price=0.97, no_price=0.03), FakeFetcher(top=None), btc_eth_correlation=None)
    result = combiner.combine(make_market(yes_price=0.83, no_price=0.17), FakeFetcher(top=None), btc_eth_correlation=None)
    assert result is not None
    assert result.direction == "NO"
    assert result.confidence > 0.5


def test_no_signal_does_not_fire_without_a_real_peak(combiner):
    # yes_price=0.90 on the very first tick for this market -- no prior
    # history, so there's no real peak-to-current drop to detect yet.
    result = combiner.combine(make_market(yes_price=0.90, no_price=0.10), FakeFetcher(top=None), btc_eth_correlation=None)
    assert result is None


def test_no_signal_disabled_by_kill_switch(combiner, mocker):
    mocker.patch("core.signal_combiner.NO_TRADING_ENABLED", False)
    combiner.combine(make_market(yes_price=0.97, no_price=0.03), FakeFetcher(top=None), btc_eth_correlation=None)
    result = combiner.combine(make_market(yes_price=0.83, no_price=0.17), FakeFetcher(top=None), btc_eth_correlation=None)
    assert result is None


def test_no_signal_status_reflects_fired_not_blocked(combiner, mocker):
    import json
    mocker.patch("core.signal_combiner.NO_TRADING_ENABLED", True)
    mocker.patch("core.signals.mean_reversion_no_signal.NO_TRADING_ENABLED", True)
    combiner.combine(make_market(yes_price=0.97, no_price=0.03), FakeFetcher(top=None), btc_eth_correlation=None)
    combiner.combine(make_market(yes_price=0.83, no_price=0.17), FakeFetcher(top=None), btc_eth_correlation=None)
    with open(combiner.status_path) as f:
        data = json.load(f)
    assert data["BTC"]["price_out_of_band"] is False
    assert data["BTC"]["fired"] is True
    assert data["BTC"]["combined_confidence"] is not None


def test_no_signal_still_blocked_below_the_0_80_gate(combiner):
    # a drop from an extreme peak that lands BELOW NO_REVERSION_MIN_YES_PRICE
    # (0.80) is still just an out-of-band block, not a NO trade.
    combiner.combine(make_market(yes_price=0.97, no_price=0.03), FakeFetcher(top=None), btc_eth_correlation=None)
    result = combiner.combine(make_market(yes_price=0.70, no_price=0.30), FakeFetcher(top=None), btc_eth_correlation=None)
    assert result is None
