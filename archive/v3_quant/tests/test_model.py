from core.model import OnlineModel
from config.settings import KELLY_MIN_EXAMPLES, MIN_SECONDS_REMAINING_5M, MIN_SECONDS_REMAINING_15M

FEATURES_BTC = {
    "yes_price": 0.5, "yes_price_momentum_60s": 0.0, "yes_price_momentum_120s": 0.0,
    "distance_from_half": 0.0, "is_above_half": 0,
    "spot_momentum_1m": 0.0, "spot_momentum_5m": 0.0, "spot_momentum_15m": 0.0,
    "volume_ratio": 0.0, "bid_ask_imbalance": 0.0,
    "book_imbalance": 0.0, "volume_ratio_window": 0.0,
    "seconds_remaining_pct": 0.5, "hour_sin": 0.0, "hour_cos": 1.0,
    "day_of_week_sin": 0.0, "day_of_week_cos": 1.0,
}


def _model(tmp_path, asset="BTC", timeframe="5m"):
    return OnlineModel(weights_file=str(tmp_path / f"model_{asset}_{timeframe}.pkl"), asset=asset, timeframe=timeframe)


# 2. Model doesn't trade until 100 examples.
def test_model_no_trade_before_100_examples(tmp_path):
    model = _model(tmp_path)
    for _ in range(KELLY_MIN_EXAMPLES - 1):
        model.learn(FEATURES_BTC, True)
    result = model.decide(FEATURES_BTC, seconds_remaining=250)
    assert result["decision"] == "HOLD"
    assert "warmup" in result["reason"]


def test_model_trades_once_100_examples_reached(tmp_path):
    model = _model(tmp_path)
    for _ in range(KELLY_MIN_EXAMPLES):
        model.learn(FEATURES_BTC, True)
    result = model.decide(FEATURES_BTC, seconds_remaining=250)
    assert result["decision"] != "HOLD" or result["reason"] != f"warmup {KELLY_MIN_EXAMPLES}/{KELLY_MIN_EXAMPLES}"
    assert result["p_up"] is not None


# 3. Model NEVER auto-resets.
def test_model_never_auto_resets(tmp_path):
    model = _model(tmp_path)
    for i in range(300):
        model.learn(FEATURES_BTC, i % 2 == 0)
    assert model.n_examples == 300
    # no reset/health-check method exists at all on the class
    assert not hasattr(model, "_reset_to_fresh")
    assert not hasattr(model, "_run_health_check")
    assert not hasattr(model, "_run_stability_monitor")


# 11. seconds_remaining filter works for both timeframes.
def test_seconds_remaining_filter_5m(tmp_path):
    model = _model(tmp_path, timeframe="5m")
    for _ in range(KELLY_MIN_EXAMPLES):
        model.learn(FEATURES_BTC, True)
    too_late = model.decide(FEATURES_BTC, seconds_remaining=MIN_SECONDS_REMAINING_5M - 1)
    assert too_late["decision"] == "HOLD"
    assert too_late["reason"] == "too_late"

    still_ok = model.decide(FEATURES_BTC, seconds_remaining=MIN_SECONDS_REMAINING_5M + 1)
    assert still_ok["reason"] != "too_late"


def test_seconds_remaining_filter_15m(tmp_path):
    model = _model(tmp_path, timeframe="15m")
    for _ in range(KELLY_MIN_EXAMPLES):
        model.learn(FEATURES_BTC, True)
    too_late = model.decide(FEATURES_BTC, seconds_remaining=MIN_SECONDS_REMAINING_15M - 1)
    assert too_late["decision"] == "HOLD"
    assert too_late["reason"] == "too_late"

    still_ok = model.decide(FEATURES_BTC, seconds_remaining=MIN_SECONDS_REMAINING_15M + 1)
    assert still_ok["reason"] != "too_late"


# Persistence: save and load model weights.
def test_model_weights_persist_across_restart(tmp_path):
    weights_file = str(tmp_path / "model_btc_5m.pkl")
    model = OnlineModel(weights_file=weights_file, asset="BTC", timeframe="5m")
    for i in range(37):
        model.learn(FEATURES_BTC, i % 2 == 0)
    assert model.n_examples == 37

    reloaded = OnlineModel(weights_file=weights_file, asset="BTC", timeframe="5m")
    assert reloaded.n_examples == 37
