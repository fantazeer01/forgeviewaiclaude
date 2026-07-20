import pytest

from layer2_brain.feature_engine import (
    build_features, CrossMarketState, BASE_FEATURE_NAMES, CROSS_MARKET_FEATURE_NAMES,
)


def _snapshot(asset, yes_price=0.5, seconds_remaining=200, hour_utc=0, weekday=0, **overrides):
    base = {
        "asset": asset, "yes_price": yes_price, "no_price": 1 - yes_price,
        "yes_price_change_60s": 0.01, "yes_price_change_120s": 0.02,
        "price_change_1m": 5.0, "price_change_5m": 10.0, "price_change_15m": 15.0, "price_change_30m": 20.0,
        "volume_ratio": 1.2, "volume_trend": 1.0, "bid_ask_imbalance": 0.1,
        "volatility_5m": 3.0, "volatility_15m": 4.0,
        "book_imbalance": 0.2, "book_depth_ratio": 0.55, "volume_ratio_window": 0.3,
        "seconds_remaining": seconds_remaining, "hour_utc": hour_utc, "weekday": weekday,
    }
    base.update(overrides)
    return base


# 6. Feature engine generates all 33 base features (36 for ETH/SOL with cross-market).
def test_feature_engine_generates_all_base_features_for_btc():
    features = build_features(_snapshot("BTC"), window_sec=300)
    assert set(features.keys()) == set(BASE_FEATURE_NAMES)
    assert len(BASE_FEATURE_NAMES) == 33
    assert len(features) == 33


def test_feature_engine_generates_36_features_for_eth_sol():
    btc_snap = _snapshot("BTC")
    features = build_features(_snapshot("ETH"), window_sec=300, btc_snapshot=btc_snap, correlation=0.4)
    expected = set(BASE_FEATURE_NAMES) | set(CROSS_MARKET_FEATURE_NAMES)
    assert set(features.keys()) == expected
    assert len(features) == 36


# 7. Cross-market features only appear for ETH/SOL, never BTC.
def test_cross_market_features_only_for_eth_sol():
    btc_features = build_features(_snapshot("BTC"), window_sec=300, btc_snapshot=_snapshot("BTC"), correlation=0.9)
    assert not any(k in btc_features for k in CROSS_MARKET_FEATURE_NAMES)

    eth_features = build_features(_snapshot("ETH"), window_sec=300, btc_snapshot=_snapshot("BTC"), correlation=0.9)
    assert all(k in eth_features for k in CROSS_MARKET_FEATURE_NAMES)

    sol_features = build_features(_snapshot("SOL"), window_sec=300, btc_snapshot=_snapshot("BTC"), correlation=0.9)
    assert all(k in sol_features for k in CROSS_MARKET_FEATURE_NAMES)


# 8. hour_sin/cos encoding correct (hour=0 -> sin=0, cos=1).
def test_hour_sin_cos_encoding():
    features = build_features(_snapshot("BTC", hour_utc=0), window_sec=300)
    assert features["hour_sin"] == pytest.approx(0.0)
    assert features["hour_cos"] == pytest.approx(1.0)

    features_6 = build_features(_snapshot("BTC", hour_utc=6), window_sec=300)
    assert features_6["hour_sin"] == pytest.approx(1.0)
    assert features_6["hour_cos"] == pytest.approx(0.0)


def test_news_fear_greed_whale_memory_regime_wired_in():
    features = build_features(
        _snapshot("BTC"), window_sec=300,
        news={"sentiment_1h": 0.4, "count_1h": 5, "has_major": True},
        fear_greed={"normalized": 0.8, "change_24h": -3.0},
        whale={"imbalance": 0.3, "volume_total": 2000.0, "activity": 4},
        memory={"win_rate_1h": 0.6, "win_rate_6h": 0.55},
        regime="TRENDING_UP",
    )
    assert features["news_sentiment_1h"] == 0.4
    assert features["news_count_1h"] == 5
    assert features["news_has_major"] == 1
    assert features["fear_greed_normalized"] == 0.8
    assert features["fear_greed_change"] == -3.0
    assert features["whale_imbalance"] == 0.3
    assert features["whale_volume_total"] == 2000.0
    assert features["whale_activity"] == 4
    assert features["rolling_win_rate_1h"] == 0.6
    assert features["rolling_win_rate_6h"] == 0.55
    assert features["regime_score"] == 1.0


def test_cross_market_correlation_updates_each_tick():
    state = CrossMarketState(window=20)
    assert state.correlation("ETH") is None
    for i in range(10):
        state.update("ETH", btc_momentum_5m=float(i), asset_momentum_5m=float(i) * 2 + 1)
    corr = state.correlation("ETH")
    assert corr == pytest.approx(1.0, abs=1e-6)
