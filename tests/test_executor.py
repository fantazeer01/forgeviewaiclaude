import requests

from layer5_hands.executor import Executor
from layer4_wallet.risk_manager import RiskManager
from layer2_brain.model import OnlineModel

FEATURES = {
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


def _executor(tmp_path, monkeypatch):
    monkeypatch.setattr("layer5_hands.executor.PAPER_TRADES_LOG", str(tmp_path / "trades.jsonl"))
    model = OnlineModel(weights_file=str(tmp_path / "model.pkl"), asset="BTC", timeframe="5m")
    risk = RiskManager(state_file=str(tmp_path / "risk_state.json"))
    return Executor(model, risk)


# 22. PnL formula with fee is correct at TAKER_FEE_RATE=0.
def test_pnl_formula_with_zero_fee(tmp_path, monkeypatch):
    executor = _executor(tmp_path, monkeypatch)
    win_id = executor.open_position("BTC", "5m", "YES", 0.5, 10.0, FEATURES, "m1", model_prob=0.6, regime="RANGE")
    pnl_win = executor.close_position(win_id, True)
    assert pnl_win == 10.0  # shares=20; pnl = 20*(1-0.5) - 0 = 10.0

    lose_id = executor.open_position("BTC", "5m", "YES", 0.5, 10.0, FEATURES, "m2", model_prob=0.6, regime="RANGE")
    pnl_loss = executor.close_position(lose_id, False)
    assert pnl_loss == -10.0


def test_pnl_formula_no_win_at_different_entry_price(tmp_path, monkeypatch):
    executor = _executor(tmp_path, monkeypatch)
    position_id = executor.open_position("ETH", "15m", "NO", 0.4, 5.0, FEATURES, "m3", model_prob=0.3, regime="RANGE")
    pnl = executor.close_position(position_id, False)  # outcome_up=False -> NO side wins
    assert pnl == 7.5  # shares=12.5; pnl = 12.5*(1-0.4) - 0 = 7.5


# 21. Paper mode never calls a real API.
def _blocked_request(*args, **kwargs):
    raise AssertionError("network call attempted in paper mode")


def test_paper_mode_never_calls_real_api(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.Session, "get", _blocked_request)
    monkeypatch.setattr(requests, "get", _blocked_request)
    executor = _executor(tmp_path, monkeypatch)
    position_id = executor.open_position("BTC", "5m", "YES", 0.5, 2.0, FEATURES, "m4", model_prob=0.6, regime="RANGE")
    pnl = executor.close_position(position_id, True)
    assert isinstance(pnl, float)


# 23. features_snapshot is saved with every trade record.
def test_features_snapshot_saved_in_trade_log(tmp_path, monkeypatch):
    import json

    log_path = tmp_path / "trades.jsonl"
    monkeypatch.setattr("layer5_hands.executor.PAPER_TRADES_LOG", str(log_path))
    model = OnlineModel(weights_file=str(tmp_path / "model.pkl"), asset="BTC", timeframe="5m")
    risk = RiskManager(state_file=str(tmp_path / "risk_state.json"))
    executor = Executor(model, risk)

    position_id = executor.open_position(
        "BTC", "5m", "YES", 0.5, 2.0, FEATURES, "m5", model_prob=0.61, regime="TRENDING_UP",
    )
    executor.close_position(position_id, True)

    with open(log_path, encoding="utf-8") as f:
        record = json.loads(f.readline())

    assert "features_snapshot" in record
    assert len(record["features_snapshot"]) == 10
    assert set(record["features_snapshot"].keys()).issubset(FEATURES.keys())
    assert record["model_prob"] == 0.61
    assert record["regime"] == "TRENDING_UP"
    assert record["trade_id"] == position_id
    assert record["pnl_usd"] == 2.0  # shares=4; pnl = 4*(1-0.5) - 0 = 2.0
