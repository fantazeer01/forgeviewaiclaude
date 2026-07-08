import json

import requests

from core.executor import Executor
from core.risk_manager import RiskManager
from models.momentum_model import MomentumModel
from models.volume_model import VolumeModel

FEATURES = {"price_momentum_1m": 1.0, "price_momentum_5m": 1.0, "volume_ratio": 0.0, "bid_ask_imbalance": 0.0}


def _blocked_request(*args, **kwargs):
    raise AssertionError("network call attempted in paper mode")


def _make_executor(tmp_path, monkeypatch, log_name="trades.jsonl"):
    monkeypatch.setattr("core.executor.PAPER_TRADES_LOG", str(tmp_path / log_name))
    momentum = MomentumModel(weights_file=str(tmp_path / "m.pkl"))
    volume = VolumeModel(weights_file=str(tmp_path / "v.pkl"))
    risk = RiskManager()
    return Executor(momentum, volume, risk)


def test_paper_mode_never_calls_real_api(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.Session, "get", _blocked_request)
    monkeypatch.setattr(requests, "get", _blocked_request)
    executor = _make_executor(tmp_path, monkeypatch)

    position_id = executor.open_position("BTC", "YES", 0.5, 2.0, FEATURES, "market123")
    pnl = executor.close_position(position_id, True)
    assert isinstance(pnl, float)


def test_close_position_settles_win_and_loss_pnl(tmp_path, monkeypatch):
    executor = _make_executor(tmp_path, monkeypatch)

    win_id = executor.open_position("BTC", "YES", 0.5, 10.0, FEATURES, "m1")
    pnl_win = executor.close_position(win_id, True)
    assert pnl_win > 0

    lose_id = executor.open_position("BTC", "YES", 0.5, 10.0, FEATURES, "m2")
    pnl_loss = executor.close_position(lose_id, False)
    assert pnl_loss == -10.0


def test_close_position_logs_trade_to_file(tmp_path, monkeypatch):
    log_path = tmp_path / "trades.jsonl"
    executor = _make_executor(tmp_path, monkeypatch)

    position_id = executor.open_position("ETH", "NO", 0.4, 5.0, FEATURES, "m3")
    executor.close_position(position_id, False)  # outcome_up=False -> NO side wins

    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["asset"] == "ETH"
    assert record["won"] is True
