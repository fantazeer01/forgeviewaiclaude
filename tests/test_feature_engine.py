from core.feature_engine import build_features, FEATURE_NAMES


def test_build_features_generates_all_eight_keys():
    context = {
        "momentum_1m_bps": 5.0, "momentum_5m_bps": 10.0, "volume_ratio": 1.2,
        "seconds_remaining": 120.0, "hour_utc": 14, "fear_greed": 60,
        "yes_price": 0.5, "bid_ask_imbalance": 0.1,
    }
    features = build_features(context)
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 8
    assert features["price_momentum_1m"] == 5.0


def test_build_features_defaults_when_missing():
    features = build_features({})
    assert features["fear_greed"] == 50
    assert features["yes_price"] == 0.5
    assert features["price_momentum_1m"] == 0.0
    assert features["bid_ask_imbalance"] == 0.0
