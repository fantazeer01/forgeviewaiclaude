from config.settings import BANKROLL_USD, FIXED_POSITION_PCT, KELLY_WARMUP_TRADES, KELLY_FRACTION_CAP
from core.risk_manager import RiskManager


def test_daily_loss_limit_blocks_trading():
    rm = RiskManager()
    rm.daily_pnl = -10.0
    ok, reason = rm.can_open_trade()
    assert ok is False
    assert "daily loss" in reason


def test_loss_streak_pause_blocks_trading():
    rm = RiskManager()
    for _ in range(5):
        rm.register_close(-1.0)
    ok, reason = rm.can_open_trade()
    assert ok is False
    assert "pause" in reason


def test_max_open_positions_blocks_trading():
    rm = RiskManager()
    for _ in range(3):
        rm.register_open()
    ok, reason = rm.can_open_trade()
    assert ok is False
    assert "positions" in reason


def test_loss_streak_resets_on_win():
    rm = RiskManager()
    rm.register_close(-1.0)
    rm.register_close(-1.0)
    assert rm.loss_streak == 2
    rm.register_close(5.0)
    assert rm.loss_streak == 0


def test_fixed_sizing_before_kelly_warmup():
    rm = RiskManager()
    size = rm.position_size(win_probability=0.9, entry_price=0.5)
    assert size == round(BANKROLL_USD * FIXED_POSITION_PCT, 2)


def test_kelly_sizing_after_warmup_capped():
    rm = RiskManager()
    rm.trades_closed = KELLY_WARMUP_TRADES
    size = rm.position_size(win_probability=0.95, entry_price=0.5)
    max_size = round(BANKROLL_USD * KELLY_FRACTION_CAP, 2)
    assert 0 < size <= max_size
