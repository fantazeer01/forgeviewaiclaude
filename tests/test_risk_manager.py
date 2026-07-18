import time

from core.risk_manager import RiskManager
from config.settings import (
    BANKROLL_USD, FIXED_POSITION_USD, KELLY_MIN_EXAMPLES, KELLY_MAX_FRACTION,
    MAX_DAILY_LOSS_USD, MAX_OPEN_POSITIONS_5M, MAX_OPEN_POSITIONS_15M,
    MAX_CONSECUTIVE_LOSSES, CONSECUTIVE_LOSS_PAUSE_MINUTES,
)


def _rm(tmp_path, **kwargs):
    return RiskManager(state_file=str(tmp_path / "risk_state.json"), **kwargs)


# 4. Kelly not active until 100 examples.
def test_kelly_inactive_before_100_examples(tmp_path):
    rm = _rm(tmp_path)
    size = rm.position_size(win_probability=0.9, entry_price=0.5)
    assert size == round(FIXED_POSITION_USD, 2)


def test_kelly_active_after_100_examples(tmp_path):
    rm = _rm(tmp_path)
    rm.trades_closed = KELLY_MIN_EXAMPLES
    size = rm.position_size(win_probability=0.95, entry_price=0.5)
    max_size = round(BANKROLL_USD * KELLY_MAX_FRACTION, 2)
    assert 0 < size <= max_size
    assert size != FIXED_POSITION_USD


# 5. Daily $100 limit blocks trading.
def test_daily_loss_limit_blocks_trading(tmp_path):
    rm = _rm(tmp_path)
    rm.daily_pnl = -MAX_DAILY_LOSS_USD
    ok, reason = rm.can_open_trade("5m")
    assert ok is False
    assert "daily loss" in reason


# 6. After 10 consecutive losses -> 20 min pause (not permanent).
def test_pause_after_10_consecutive_losses_is_temporary(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(MAX_CONSECUTIVE_LOSSES):
        rm.register_close("5m", -1.0)
    assert rm.loss_streak == MAX_CONSECUTIVE_LOSSES
    ok, reason = rm.can_open_trade("5m")
    assert ok is False
    assert "pause" in reason
    # temporary: paused_until is a bounded future timestamp, not infinite
    assert rm.paused_until <= time.time() + CONSECUTIVE_LOSS_PAUSE_MINUTES * 60 + 1


# 7. Trading resumes once the 20-minute pause expires.
def test_trading_resumes_after_pause_expires(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(MAX_CONSECUTIVE_LOSSES):
        rm.register_close("5m", -1.0)
    future = rm.paused_until + 1
    ok, reason = rm.can_open_trade("5m", now=future)
    assert ok is True


# 8/9. MAX_OPEN_POSITIONS_5M / _15M enforced independently.
def test_max_open_positions_5m_enforced(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(MAX_OPEN_POSITIONS_5M):
        rm.register_open("5m")
    ok, reason = rm.can_open_trade("5m")
    assert ok is False
    assert "5m" in reason


def test_max_open_positions_15m_enforced(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(MAX_OPEN_POSITIONS_15M):
        rm.register_open("15m")
    ok, reason = rm.can_open_trade("15m")
    assert ok is False
    assert "15m" in reason


# 10. 5-min and 15-min markets are independent -- 5m being full doesn't block 15m.
def test_5m_and_15m_position_caps_are_independent(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(MAX_OPEN_POSITIONS_5M):
        rm.register_open("5m")
    ok_5m, _ = rm.can_open_trade("5m")
    ok_15m, _ = rm.can_open_trade("15m")
    assert ok_5m is False
    assert ok_15m is True


# 15. Persistence: save and load risk_manager state.
def test_risk_manager_state_persists_across_restart(tmp_path):
    state_file = str(tmp_path / "risk_state.json")
    rm = RiskManager(state_file=state_file)
    for _ in range(5):
        rm.register_close("5m", 2.0)
    rm.register_close("5m", -1.0)
    assert rm.trades_closed == 6

    restarted = RiskManager(state_file=state_file)
    assert restarted.trades_closed == 6
    assert restarted.bankroll == rm.bankroll
    assert restarted.loss_streak == rm.loss_streak
