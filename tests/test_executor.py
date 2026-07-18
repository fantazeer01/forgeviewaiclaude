import requests

from core.executor import Executor
from core.risk_manager import RiskManager
from core.model import OnlineModel

FEATURES = {
    "yes_price": 0.5, "yes_price_momentum_60s": 0.0, "yes_price_momentum_120s": 0.0,
    "distance_from_half": 0.0, "is_above_half": 0,
    "spot_momentum_1m": 0.0, "spot_momentum_5m": 0.0, "spot_momentum_15m": 0.0,
    "volume_ratio": 0.0, "bid_ask_imbalance": 0.0,
    "book_imbalance": 0.0, "volume_ratio_window": 0.0,
    "seconds_remaining_pct": 0.5, "hour_sin": 0.0, "hour_cos": 1.0,
    "day_of_week_sin": 0.0, "day_of_week_cos": 1.0,
}


def _executor(tmp_path, monkeypatch):
    monkeypatch.setattr("core.executor.PAPER_TRADES_LOG", str(tmp_path / "trades.jsonl"))
    model = OnlineModel(weights_file=str(tmp_path / "model.pkl"), asset="BTC", timeframe="5m")
    risk = RiskManager(state_file=str(tmp_path / "risk_state.json"))
    return Executor(model, risk)


# 14. PnL formula with fee is correct at TAKER_FEE_RATE=0.
def test_pnl_formula_with_zero_fee(tmp_path, monkeypatch):
    executor = _executor(tmp_path, monkeypatch)
    win_id = executor.open_position("BTC", "5m", "YES", 0.5, 10.0, FEATURES, "m1")
    pnl_win = executor.close_position(win_id, True)
    # shares = 10/0.5 = 20; fee = 20*0*0.5*0.5 = 0; pnl = 20*(1-0.5) - 0 = 10.0
    assert pnl_win == 10.0

    lose_id = executor.open_position("BTC", "5m", "YES", 0.5, 10.0, FEATURES, "m2")
    pnl_loss = executor.close_position(lose_id, False)
    # pnl_loss = -size - fee = -10 - 0 = -10.0
    assert pnl_loss == -10.0


def test_pnl_formula_no_win_at_different_entry_price(tmp_path, monkeypatch):
    executor = _executor(tmp_path, monkeypatch)
    position_id = executor.open_position("ETH", "15m", "NO", 0.4, 5.0, FEATURES, "m3")
    pnl = executor.close_position(position_id, False)  # outcome_up=False -> NO side wins
    # shares = 5/0.4 = 12.5; pnl = 12.5*(1-0.4) - 0 = 7.5
    assert pnl == 7.5


def _blocked_request(*args, **kwargs):
    raise AssertionError("network call attempted in paper mode")


def test_paper_mode_never_calls_real_api(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.Session, "get", _blocked_request)
    monkeypatch.setattr(requests, "get", _blocked_request)
    executor = _executor(tmp_path, monkeypatch)
    position_id = executor.open_position("BTC", "5m", "YES", 0.5, 2.0, FEATURES, "m4")
    pnl = executor.close_position(position_id, True)
    assert isinstance(pnl, float)
