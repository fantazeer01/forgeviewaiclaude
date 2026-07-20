from layer2_brain.model import OnlineModel
from config.settings import KELLY_MIN_EXAMPLES

FEATURES_BTC = {
    "spot_momentum_1m": 0.0, "spot_momentum_5m": 0.0, "spot_momentum_15m": 0.0, "spot_momentum_30m": 0.0,
    "volume_ratio_5m": 0.0, "volume_trend": 0.0, "bid_ask_imbalance_binance": 0.0,
    "volatility_5m": 0.0, "volatility_15m": 0.0, "price_acceleration": 0.0,
    "yes_price": 0.5, "yes_momentum_60s": 0.0, "yes_momentum_120s": 0.0, "distance_from_half": 0.0,
    "book_imbalance_polymarket": 0.0, "book_depth_ratio": 0.5, "volume_ratio_window": 0.0,
    "seconds_remaining_pct": 0.5,
    "news_sentiment_1h": 0.0, "news_count_1h": 0, "news_has_major": 0,
    "fear_greed_normalized": 0.5, "fear_greed_change": 0.0,
    "whale_imbalance": 0.0, "whale_volume_total": 0.0, "whale_activity": 0,
    "hour_sin": 0.0, "hour_cos": 1.0, "day_sin": 0.0, "day_cos": 1.0,
    "rolling_win_rate_1h": 0.5, "rolling_win_rate_6h": 0.5, "regime_score": 0.0,
}


def _model(tmp_path, asset="BTC", timeframe="5m"):
    return OnlineModel(weights_file=str(tmp_path / f"model_{asset}_{timeframe}.pkl"), asset=asset, timeframe=timeframe)


def test_kelly_min_examples_is_200():
    assert KELLY_MIN_EXAMPLES == 200


def test_model_not_warmed_up_before_200_examples(tmp_path):
    model = _model(tmp_path)
    for _ in range(KELLY_MIN_EXAMPLES - 1):
        model.learn(FEATURES_BTC, True)
    assert model.is_warmed_up() is False


def test_model_warmed_up_at_200_examples(tmp_path):
    model = _model(tmp_path)
    for _ in range(KELLY_MIN_EXAMPLES):
        model.learn(FEATURES_BTC, True)
    assert model.is_warmed_up() is True


# 27. Model NEVER auto-resets.
def test_model_never_auto_resets(tmp_path):
    model = _model(tmp_path)
    for i in range(400):
        model.learn(FEATURES_BTC, i % 2 == 0)
    assert model.n_examples == 400
    assert not hasattr(model, "_reset_to_fresh")
    assert not hasattr(model, "_run_health_check")
    assert not hasattr(model, "_run_stability_monitor")


def test_model_weights_persist_across_restart(tmp_path):
    weights_file = str(tmp_path / "model_btc_5m.pkl")
    model = OnlineModel(weights_file=weights_file, asset="BTC", timeframe="5m")
    for i in range(37):
        model.learn(FEATURES_BTC, i % 2 == 0)
    reloaded = OnlineModel(weights_file=weights_file, asset="BTC", timeframe="5m")
    assert reloaded.n_examples == 37


def test_top_feature_names_falls_back_before_training(tmp_path):
    model = _model(tmp_path)
    names = model.top_feature_names(10)
    assert len(names) == 10
    assert all(n in FEATURES_BTC for n in names)


def test_top_feature_names_after_training(tmp_path):
    model = _model(tmp_path, asset="ETH")
    for i in range(50):
        model.learn(FEATURES_BTC, i % 2 == 0)
    names = model.top_feature_names(10)
    assert len(names) == 10
