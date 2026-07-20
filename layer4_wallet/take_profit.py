"""Layer 4 (wallet): lock in daily gains and defend the bankroll after a
drawdown -- separate from risk_manager's loss-side guardrails."""

import datetime

from config.settings import DAILY_TAKE_PROFIT_USD, DRAWDOWN_FROM_PEAK_PCT


class TakeProfitManager:
    def __init__(self, daily_take_profit_usd: float = DAILY_TAKE_PROFIT_USD,
                 drawdown_pct: float = DRAWDOWN_FROM_PEAK_PCT):
        self.daily_take_profit_usd = daily_take_profit_usd
        self.drawdown_pct = drawdown_pct
        self._daily_peak_bankroll = None
        self._daily_date = None

    def update(self, bankroll: float, now: datetime.datetime = None):
        """Call once per tick (or at least once per closed trade) with the
        current bankroll -- tracks the intraday peak, resetting at UTC
        midnight."""
        now = now or datetime.datetime.now(datetime.timezone.utc)
        today = now.date()
        if self._daily_date != today:
            self._daily_date = today
            self._daily_peak_bankroll = bankroll
        else:
            self._daily_peak_bankroll = max(self._daily_peak_bankroll, bankroll)

    def take_profit_hit(self, daily_pnl: float) -> bool:
        return daily_pnl >= self.daily_take_profit_usd

    def take_profit_remaining(self, daily_pnl: float) -> float:
        return max(0.0, self.daily_take_profit_usd - daily_pnl)

    def drawdown_size_multiplier(self, bankroll: float) -> float:
        """1.0 normally; 0.5 once the bankroll has fallen
        DRAWDOWN_FROM_PEAK_PCT off today's peak."""
        if not self._daily_peak_bankroll or self._daily_peak_bankroll <= 0:
            return 1.0
        drawdown = (self._daily_peak_bankroll - bankroll) / self._daily_peak_bankroll
        return 0.5 if drawdown >= self.drawdown_pct else 1.0
