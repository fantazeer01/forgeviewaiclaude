import json

import pytest

from config.settings import MAX_DAILY_LOSS_USD, MAX_LOSS_STREAK, MAX_OPEN_POSITIONS, PAPER_TRADE_SIZE_USD
from core.dedup_guard import DedupGuard
from core.paper_trading_engine import PaperTradingEngine
from core.repricing_detector import RepricingSignal
from core.state_manager import StateManager


@pytest.fixture
def engine(tmp_path, monkeypatch):
    trades_log = tmp_path / "trades.jsonl"
    monkeypatch.setattr("core.paper_trading_engine.TRADES_LOG", str(trades_log))
    state = StateManager(state_file=str(tmp_path / "state.json"))
    dedup = DedupGuard(state_file=str(tmp_path / "dedup.json"))
    return PaperTradingEngine(state, dedup)


def make_signal(market_id="m1", asset="BTC", direction="YES", yes_price=0.6, no_price=0.4, confidence=0.7):
    return RepricingSignal(
        asset=asset, market_id=market_id, direction=direction,
        yes_price=yes_price, no_price=no_price, confidence=confidence, reason="test",
    )


def test_can_open_true_by_default(engine):
    ok, reason = engine.can_open()
    assert ok is True


def test_can_open_false_when_stopped(engine):
    engine.state.stop_system("manual stop")
    ok, reason = engine.can_open()
    assert ok is False
    assert "manual stop" in reason


def test_can_open_false_when_max_positions_reached(engine):
    for i in range(MAX_OPEN_POSITIONS):
        engine.open_trade(make_signal(market_id=f"m{i}"))
    ok, reason = engine.can_open()
    assert ok is False
    assert "Max open positions" in reason


def test_can_open_false_and_stops_on_daily_loss_limit(engine):
    engine.state.set("daily_loss_usd", MAX_DAILY_LOSS_USD)
    ok, reason = engine.can_open()
    assert ok is False
    assert engine.state.is_stopped() is True


def test_can_open_false_and_stops_on_loss_streak(engine):
    engine.state.set("loss_streak", MAX_LOSS_STREAK)
    ok, reason = engine.can_open()
    assert ok is False
    assert engine.state.is_stopped() is True


def test_open_trade_creates_trade_and_marks_dedup(engine):
    trade = engine.open_trade(make_signal())
    assert trade is not None
    assert trade.entry_price == 0.6
    assert trade.size_tokens == pytest.approx(PAPER_TRADE_SIZE_USD / 0.6)
    assert engine.dedup.is_duplicate("m1") is True
    assert len(engine.get_open_trades()) == 1


def test_open_trade_no_op_when_duplicate(engine):
    engine.open_trade(make_signal())
    result = engine.open_trade(make_signal())
    assert result is None
    assert len(engine.get_open_trades()) == 1


def test_open_trade_returns_none_for_nonpositive_entry_price(engine):
    signal = make_signal(direction="NO", no_price=0.0)
    assert engine.open_trade(signal) is None


def test_open_trade_returns_none_when_cannot_open(engine):
    engine.state.stop_system("stopped")
    assert engine.open_trade(make_signal()) is None


def test_close_trade_win_updates_state(engine):
    engine.open_trade(make_signal(direction="YES", yes_price=0.6))
    trade = engine.close_trade("m1", outcome="YES")
    assert trade.result == "WIN"
    assert trade.pnl_usd == round((PAPER_TRADE_SIZE_USD / 0.6) * 0.4, 4)
    assert engine.state.get("wins") == 1
    assert engine.state.get("loss_streak") == 0
    assert "m1" not in [t.market_id for t in engine.get_open_trades()]
    assert engine.dedup.is_duplicate("m1") is False


def test_close_trade_loss_updates_state_and_daily_loss(engine):
    engine.open_trade(make_signal(direction="YES", yes_price=0.6))
    trade = engine.close_trade("m1", outcome="NO")
    assert trade.result == "LOSS"
    assert trade.pnl_usd == pytest.approx(-PAPER_TRADE_SIZE_USD)
    assert engine.state.get("losses") == 1
    assert engine.state.get("loss_streak") == 1
    assert engine.state.get("daily_loss_usd") == pytest.approx(abs(trade.pnl_usd))


def test_close_trade_returns_none_for_unknown_market(engine):
    assert engine.close_trade("nope", outcome="YES") is None


def test_close_trade_stops_system_on_loss_streak(engine):
    for i in range(MAX_LOSS_STREAK):
        engine.open_trade(make_signal(market_id=f"m{i}", direction="YES", yes_price=0.6))
        engine.close_trade(f"m{i}", outcome="NO")
    assert engine.state.is_stopped() is True
    assert "Loss streak" in engine.state.get("stop_reason")


def test_restore_open_trades_from_log(tmp_path, monkeypatch):
    trades_log = tmp_path / "trades.jsonl"
    monkeypatch.setattr("core.paper_trading_engine.TRADES_LOG", str(trades_log))
    open_entry = {
        "trade_id": "abc12345", "market_id": "m1", "asset": "BTC", "direction": "YES",
        "entry_price": 0.6, "size_usd": 10.0, "size_tokens": 16.6667,
        "signal_confidence": 0.7, "signal_reason": "test", "signal_source": "repricing",
        "open_ts": "2024-01-01T00:00:00", "minutes_at_open": 3.0, "status": "open",
        "close_ts": None, "close_price": None, "pnl_usd": None, "result": None,
    }
    trades_log.write_text(json.dumps(open_entry) + "\n")
    state = StateManager(state_file=str(tmp_path / "state.json"))
    dedup = DedupGuard(state_file=str(tmp_path / "dedup.json"))
    engine = PaperTradingEngine(state, dedup)
    open_trades = engine.get_open_trades()
    assert len(open_trades) == 1
    assert open_trades[0].market_id == "m1"


def test_restore_skips_closed_trades(tmp_path, monkeypatch):
    trades_log = tmp_path / "trades.jsonl"
    monkeypatch.setattr("core.paper_trading_engine.TRADES_LOG", str(trades_log))
    lines = [
        {"trade_id": "t1", "market_id": "m1", "status": "open"},
        {"trade_id": "t1", "market_id": "m1", "status": "win"},
    ]
    trades_log.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    state = StateManager(state_file=str(tmp_path / "state.json"))
    dedup = DedupGuard(state_file=str(tmp_path / "dedup.json"))
    engine = PaperTradingEngine(state, dedup)
    assert engine.get_open_trades() == []
