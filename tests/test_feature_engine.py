import pytest

from core.feature_engine import build_features, CrossMarketState, BASE_FEATURE_NAMES, CROSS_MARKET_FEATURE_NAMES


def _snapshot(asset, yes_price=0.5, seconds_remaining=200, hour_utc=0, weekday=0, **overrides):
    base = {
        "asset": asset, "yes_price": yes_price, "yes_price_change_60s": 0.01, "yes_price_change_120s": 0.02,
        "price_change_1m": 5.0, "price_change_5m": 10.0, "price_change_15m": 15.0,
        "volume_ratio": 1.2, "bid_ask_imbalance": 0.1, "book_imbalance": 0.2, "volume_ratio_window": 0.3,
        "seconds_remaining": seconds_remaining, "hour_utc": hour_utc, "weekday": weekday,
    }
    base.update(overrides)
    return base


# 1. Feature engine generates all 20 features for ETH/SOL, 17 for BTC.
def test_feature_engine_generates_20_features_for_eth_sol():
    btc_snap = _snapshot("BTC")
    features = build_features(_snapshot("ETH"), window_sec=300, btc_snapshot=btc_snap, correlation=0.4)
    expected = set(BASE_FEATURE_NAMES) | set(CROSS_MARKET_FEATURE_NAMES)
    assert set(features.keys()) == expected
    assert len(features) == 20


def test_feature_engine_generates_17_features_for_btc():
    features = build_features(_snapshot("BTC"), window_sec=300)
    assert set(features.keys()) == set(BASE_FEATURE_NAMES)
    assert len(features) == 17


# 12. hour_sin/cos encoding correct (hour=0 -> sin=0, cos=1).
def test_hour_sin_cos_encoding():
    features = build_features(_snapshot("BTC", hour_utc=0), window_sec=300)
    assert features["hour_sin"] == pytest.approx(0.0)
    assert features["hour_cos"] == pytest.approx(1.0)

    features_6 = build_features(_snapshot("BTC", hour_utc=6), window_sec=300)
    assert features_6["hour_sin"] == pytest.approx(1.0)
    assert features_6["hour_cos"] == pytest.approx(0.0)


# 16. btc_momentum influences ETH/SOL features.
def test_btc_momentum_influences_eth_sol_features():
    btc_snap = _snapshot("BTC", price_change_5m=42.0)
    features = build_features(_snapshot("ETH"), window_sec=300, btc_snapshot=btc_snap, correlation=0.0)
    assert features["btc_momentum_5m"] == 42.0
    assert features["btc_yes_price"] == btc_snap["yes_price"]

    features_sol = build_features(_snapshot("SOL"), window_sec=300, btc_snapshot=btc_snap, correlation=0.0)
    assert features_sol["btc_momentum_5m"] == 42.0


# 17. Cross-market correlation updates on every tick (CrossMarketState.update()).
def test_cross_market_correlation_updates_each_tick():
    state = CrossMarketState(window=20)
    assert state.correlation("ETH") is None  # no samples yet

    for i in range(10):
        state.update("ETH", btc_momentum_5m=float(i), asset_momentum_5m=float(i) * 2 + 1)
    corr = state.correlation("ETH")
    assert corr is not None
    assert corr == pytest.approx(1.0, abs=1e-6)  # perfectly linear relationship

    state.update("ETH", btc_momentum_5m=None, asset_momentum_5m=5.0)  # missing sample ignored
    assert len(state._pairs["ETH"]) == 10
