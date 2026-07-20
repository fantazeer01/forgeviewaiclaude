import time

from layer4_wallet.risk_manager import RiskManager
from config.settings import (
    BANKROLL_USD, WARMUP_POSITION_SIZE, KELLY_MIN_EXAMPLES, KELLY_MAX_FRACTION, KELLY_MAX_POSITION_USD,
    MAX_DAILY_LOSS_USD, MAX_OPEN_POSITIONS_5M, MAX_OPEN_POSITIONS_15M,
    MAX_CONSECUTIVE_LOSSES, CONSECUTIVE_LOSS_PAUSE_MINUTES,
    TIMEFRAME_MAX_CONSECUTIVE_LOSSES, TIMEFRAME_LOSS_PAUSE_MINUTES,
)


def _rm(tmp_path, **kwargs):
    return RiskManager(state_file=str(tmp_path / "risk_state.json"), **kwargs)


# 14. Kelly not active until 200 examples.
def test_kelly_inactive_before_200_examples(tmp_path):
    rm = _rm(tmp_path)
    assert KELLY_MIN_EXAMPLES == 200
    size = rm.position_size(win_probability=0.9, entry_price=0.5)
    assert size == round(WARMUP_POSITION_SIZE, 2)


def test_kelly_active_after_200_examples(tmp_path):
    rm = _rm(tmp_path)
    rm.trades_closed = KELLY_MIN_EXAMPLES
    size = rm.position_size(win_probability=0.95, entry_price=0.5)
    assert 0 < size <= KELLY_MAX_POSITION_USD
    assert size != WARMUP_POSITION_SIZE


# 15. Kelly is capped at $5 even when the bankroll fraction would exceed it.
def test_kelly_capped_at_5_dollars(tmp_path):
    rm = _rm(tmp_path)
    rm.trades_closed = KELLY_MIN_EXAMPLES
    rm.bankroll = 10_000.0  # 3% of this is $300, way above the $5 cap
    size = rm.position_size(win_probability=0.99, entry_price=0.5)
    assert size == KELLY_MAX_POSITION_USD


# 16. Daily $100 limit blocks trading.
def test_daily_loss_limit_blocks_trading(tmp_path):
    rm = _rm(tmp_path)
    rm.daily_pnl = -MAX_DAILY_LOSS_USD
    ok, reason = rm.can_open_trade("5m")
    assert ok is False
    assert "daily loss" in reason


# 17. 10 consecutive losses -> 20 minute pause.
def test_pause_after_10_consecutive_losses(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(MAX_CONSECUTIVE_LOSSES):
        rm.register_close("5m", -1.0)
    assert rm.loss_streak == MAX_CONSECUTIVE_LOSSES
    ok, reason = rm.can_open_trade("5m")
    assert ok is False
    assert "pause" in reason
    assert rm.paused_until <= time.time() + CONSECUTIVE_LOSS_PAUSE_MINUTES * 60 + 1


# 18. Trading resumes after the pause expires.
def test_trading_resumes_after_pause_expires(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(MAX_CONSECUTIVE_LOSSES):
        rm.register_close("5m", -1.0)
    # both the bot-wide and the (stricter, longer) per-timeframe breaker
    # fired on this streak -- resuming needs whichever clears last
    future = max(rm.paused_until, rm.timeframe_paused_until["5m"]) + 1
    ok, _ = rm.can_open_trade("5m", now=future)
    assert ok is True


# Per-timeframe breaker: 5 losses in a row on ONE timeframe pauses only that
# timeframe for 30 minutes, leaving the other timeframe (and the bot as a
# whole) free to keep trading.
def test_timeframe_breaker_pauses_only_that_timeframe(tmp_path):
    rm = _rm(tmp_path)
    assert TIMEFRAME_MAX_CONSECUTIVE_LOSSES == 5
    assert TIMEFRAME_LOSS_PAUSE_MINUTES == 30
    for _ in range(TIMEFRAME_MAX_CONSECUTIVE_LOSSES):
        rm.register_close("5m", -1.0)
    assert rm.timeframe_loss_streak["5m"] == 5
    assert rm.loss_streak == 5  # bot-wide streak also moved, but MAX_CONSECUTIVE_LOSSES(10) not reached

    ok_5m, reason = rm.can_open_trade("5m")
    assert ok_5m is False
    assert "5m" in reason

    ok_15m, _ = rm.can_open_trade("15m")
    assert ok_15m is True  # the other timeframe is untouched


def test_timeframe_breaker_resumes_after_its_own_pause(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(TIMEFRAME_MAX_CONSECUTIVE_LOSSES):
        rm.register_close("15m", -1.0)
    future = rm.timeframe_paused_until["15m"] + 1
    ok, _ = rm.can_open_trade("15m", now=future)
    assert ok is True


def test_timeframe_streak_resets_on_win(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(TIMEFRAME_MAX_CONSECUTIVE_LOSSES - 1):
        rm.register_close("5m", -1.0)
    rm.register_close("5m", 2.0)  # a win resets the streak before the breaker fires
    assert rm.timeframe_loss_streak["5m"] == 0
    ok, _ = rm.can_open_trade("5m")
    assert ok is True


def test_timeframe_breaker_state_persists_across_restart(tmp_path):
    state_file = str(tmp_path / "risk_state.json")
    rm = RiskManager(state_file=state_file)
    for _ in range(TIMEFRAME_MAX_CONSECUTIVE_LOSSES):
        rm.register_close("5m", -1.0)
    restarted = RiskManager(state_file=state_file)
    assert restarted.timeframe_loss_streak["5m"] == 5
    assert restarted.timeframe_paused_until["5m"] == rm.timeframe_paused_until["5m"]


def test_max_open_positions_enforced_independently(tmp_path):
    rm = _rm(tmp_path)
    for _ in range(MAX_OPEN_POSITIONS_5M):
        rm.register_open("5m")
    ok_5m, _ = rm.can_open_trade("5m")
    ok_15m, _ = rm.can_open_trade("15m")
    assert ok_5m is False
    assert ok_15m is True
    assert MAX_OPEN_POSITIONS_15M == 5


def test_state_persists_across_restart(tmp_path):
    state_file = str(tmp_path / "risk_state.json")
    rm = RiskManager(state_file=state_file)
    for _ in range(5):
        rm.register_close("5m", 2.0)
    restarted = RiskManager(state_file=state_file)
    assert restarted.trades_closed == 5
    assert restarted.bankroll == rm.bankroll
