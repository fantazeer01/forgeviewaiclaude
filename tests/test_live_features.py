import datetime

import pytest

from core.live_features import LiveFeatureCollector


class FakeFetcher:
    def __init__(self, top=None):
        self._top = top

    def get_order_book_top(self, token_id):
        return self._top


def make_market(market_id="m1", asset="BTC", yes_price=0.4, no_price=0.6,
                 minutes_remaining=3.0, up_token_id="up-tok", volume_24h=1000.0):
    return {
        "market_id": market_id,
        "asset": asset,
        "yes_price": yes_price,
        "no_price": no_price,
        "minutes_remaining": minutes_remaining,
        "up_token_id": up_token_id,
        "volume_24h": volume_24h,
    }


def test_extract_returns_all_nine_features():
    collector = LiveFeatureCollector()
    market = make_market()
    features = collector.extract(market, FakeFetcher())
    expected_keys = {
        "yes_price", "no_price", "bid_ask_spread", "order_book_imbalance",
        "price_momentum_30s", "price_momentum_60s", "volume_24h",
        "time_remaining_pct", "btc_eth_correlation",
    }
    assert set(features.keys()) == expected_keys
    assert features["yes_price"] == 0.4
    assert features["no_price"] == 0.6
    assert features["volume_24h"] == 1000.0
    assert features["time_remaining_pct"] == pytest.approx(3.0 / 5.0)


def test_momentum_none_without_enough_history():
    collector = LiveFeatureCollector()
    market = make_market()
    features = collector.extract(market, FakeFetcher())
    assert features["price_momentum_30s"] is None
    assert features["price_momentum_60s"] is None


def test_momentum_30s_reflects_signed_price_change():
    collector = LiveFeatureCollector()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_ts = now - datetime.timedelta(seconds=35)
    collector._price_history["m1"] = [{"ts": old_ts, "yes": 0.30, "no": 0.70}]
    collector.update("m1", "BTC", 0.40, 0.60)
    features = collector.extract(make_market(yes_price=0.40, no_price=0.60), FakeFetcher())
    assert features["price_momentum_30s"] == pytest.approx(0.10)


def test_spread_and_imbalance_from_order_book_top():
    collector = LiveFeatureCollector()
    top = {"best_bid_price": 0.38, "best_ask_price": 0.42,
           "total_bid_depth": 300.0, "total_ask_depth": 100.0}
    features = collector.extract(make_market(), FakeFetcher(top=top))
    assert features["bid_ask_spread"] == pytest.approx(0.04)
    assert features["order_book_imbalance"] == pytest.approx((300.0 - 100.0) / 400.0)


def test_spread_and_imbalance_none_without_order_book():
    collector = LiveFeatureCollector()
    features = collector.extract(make_market(), FakeFetcher(top=None))
    assert features["bid_ask_spread"] is None
    assert features["order_book_imbalance"] is None


def test_correlation_none_without_enough_samples():
    collector = LiveFeatureCollector()
    for i in range(3):
        collector.update("btc-m", "BTC", 0.4 + i * 0.01, 0.6 - i * 0.01)
        collector.update("eth-m", "ETH", 0.5 + i * 0.01, 0.5 - i * 0.01)
    assert collector.btc_eth_correlation() is None


def test_correlation_positive_when_assets_move_together():
    # varying (non-constant) step sizes so the return series has real
    # variance -- uniform steps would make correlation mathematically
    # undefined (zero-variance returns), which is a different code path
    collector = LiveFeatureCollector()
    btc_deltas = [0.02, 0.01, 0.04, 0.01, 0.04, 0.03]
    eth_deltas = [d * 0.5 for d in btc_deltas]
    btc_prices = [0.30]
    eth_prices = [0.40]
    for db, de in zip(btc_deltas, eth_deltas):
        btc_prices.append(btc_prices[-1] + db)
        eth_prices.append(eth_prices[-1] + de)
    for b, e in zip(btc_prices, eth_prices):
        collector.update("btc-m", "BTC", b, 1 - b)
        collector.update("eth-m", "ETH", e, 1 - e)
    corr = collector.btc_eth_correlation()
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_correlation_resets_when_market_id_rolls_over():
    collector = LiveFeatureCollector()
    deltas = [0.02, 0.01, 0.04, 0.01, 0.04, 0.03]
    btc_price, eth_price = 0.30, 0.40
    for d in deltas:
        btc_price += d
        eth_price += d * 0.5
        collector.update("btc-window-1", "BTC", btc_price, 1 - btc_price)
        collector.update("eth-window-1", "ETH", eth_price, 1 - eth_price)
    assert collector.btc_eth_correlation() is not None
    # a new BTC market_id (window rollover) must not carry the old series forward
    collector.update("btc-window-2", "BTC", 0.90, 0.10)
    assert len(collector._asset_return_history["BTC"]) == 1


def test_update_prunes_history_older_than_retention_window():
    collector = LiveFeatureCollector()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_ts = now - datetime.timedelta(seconds=300)
    collector._price_history["m1"] = [{"ts": old_ts, "yes": 0.5, "no": 0.5}]
    collector.update("m1", "BTC", 0.4, 0.6)
    assert len(collector._price_history["m1"]) == 1
    assert collector._price_history["m1"][0]["yes"] == 0.4
