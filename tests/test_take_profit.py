import datetime

from layer4_wallet.take_profit import TakeProfitManager
from config.settings import DAILY_TAKE_PROFIT_USD, DRAWDOWN_FROM_PEAK_PCT


# 19. Take profit $50 stops trading for the day.
def test_take_profit_hit_at_50_dollars():
    tp = TakeProfitManager()
    assert DAILY_TAKE_PROFIT_USD == 50.0
    assert tp.take_profit_hit(49.99) is False
    assert tp.take_profit_hit(50.0) is True
    assert tp.take_profit_hit(60.0) is True


def test_take_profit_remaining():
    tp = TakeProfitManager()
    assert tp.take_profit_remaining(20.0) == 30.0
    assert tp.take_profit_remaining(100.0) == 0.0  # never negative


# 20. A 30% drawdown from the daily peak halves position size.
def test_drawdown_halves_position_size():
    tp = TakeProfitManager()
    assert DRAWDOWN_FROM_PEAK_PCT == 0.30
    now = datetime.datetime(2026, 7, 20, 12, 0, tzinfo=datetime.timezone.utc)
    tp.update(bankroll=150.0, now=now)  # today's peak
    assert tp.drawdown_size_multiplier(150.0) == 1.0
    assert tp.drawdown_size_multiplier(120.0) == 1.0  # 20% down -- not enough yet
    assert tp.drawdown_size_multiplier(104.0) == 0.5  # ~30.7% down -- triggers the halving
    assert tp.drawdown_size_multiplier(105.0) == 0.5  # exactly 30% down -- boundary is inclusive
    assert tp.drawdown_size_multiplier(106.0) == 1.0  # ~29.3% down -- just under the boundary


def test_drawdown_resets_on_new_day():
    tp = TakeProfitManager()
    day1 = datetime.datetime(2026, 7, 20, 23, 0, tzinfo=datetime.timezone.utc)
    tp.update(bankroll=200.0, now=day1)
    assert tp.drawdown_size_multiplier(140.0) == 0.5  # 30% down from 200

    day2 = datetime.datetime(2026, 7, 21, 0, 5, tzinfo=datetime.timezone.utc)
    tp.update(bankroll=140.0, now=day2)  # new day -- peak resets to today's starting bankroll
    assert tp.drawdown_size_multiplier(140.0) == 1.0
