"""Bankroll and trade-frequency guardrails, independent of signal quality."""

import datetime
import json
import logging
import os
import time

from config.settings import (
    BANKROLL_USD, FIXED_POSITION_PCT, KELLY_WARMUP_TRADES, KELLY_FRACTION_CAP,
    DAILY_LOSS_LIMIT_USD, MAX_OPEN_POSITIONS, LOSS_STREAK_LIMIT, LOSS_STREAK_PAUSE_SEC,
    RISK_STATE_FILE,
)

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, bankroll: float = BANKROLL_USD, state_file: str = RISK_STATE_FILE):
        self.bankroll = bankroll
        self.state_file = state_file
        self.trades_closed = self._load_trades_closed()
        self.daily_pnl = 0.0
        self.daily_date = datetime.datetime.now(datetime.timezone.utc).date()
        self.loss_streak = 0
        self.paused_until = 0.0
        self.open_positions = 0

    def _load_trades_closed(self) -> int:
        """Kelly sizing (position_size()) only activates once trades_closed
        reaches KELLY_WARMUP_TRADES -- without persisting this, every
        restart re-warms from 0 and Kelly effectively never turns on for a
        bot that gets restarted regularly."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                return int(data.get("trades_closed", 0))
            except Exception as e:
                logger.warning(f"RiskManager state load error, starting from 0: {e}")
        return 0

    def _save_trades_closed(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"trades_closed": self.trades_closed}, f)
            os.replace(tmp, self.state_file)
        except Exception as e:
            logger.error(f"RiskManager state save error: {e}")

    def _roll_day(self):
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if today != self.daily_date:
            self.daily_date = today
            self.daily_pnl = 0.0

    def can_open_trade(self, now: float = None) -> tuple:
        now = time.time() if now is None else now
        self._roll_day()
        if now < self.paused_until:
            return False, f"loss-streak pause until {self.paused_until}"
        if self.daily_pnl <= -DAILY_LOSS_LIMIT_USD:
            return False, "daily loss limit reached"
        if self.open_positions >= MAX_OPEN_POSITIONS:
            return False, "max open positions reached"
        return True, ""

    def position_size(self, win_probability: float, entry_price: float) -> float:
        """entry_price is the price of the side actually being bought (yes_price
        for a YES trade, 1-yes_price for a NO trade)."""
        if self.trades_closed < KELLY_WARMUP_TRADES:
            return round(self.bankroll * FIXED_POSITION_PCT, 2)
        if not entry_price or entry_price <= 0:
            return 0.0
        b = (1 - entry_price) / entry_price
        if b <= 0:
            return 0.0
        edge = (win_probability * b - (1 - win_probability)) / b
        fraction = max(0.0, min(edge, KELLY_FRACTION_CAP))
        return round(self.bankroll * fraction, 2)

    def register_open(self):
        self.open_positions += 1

    def register_close(self, pnl: float):
        self._roll_day()
        self.open_positions = max(0, self.open_positions - 1)
        self.trades_closed += 1
        self._save_trades_closed()
        self.daily_pnl += pnl
        self.bankroll += pnl
        if pnl < 0:
            self.loss_streak += 1
            if self.loss_streak >= LOSS_STREAK_LIMIT:
                self.paused_until = time.time() + LOSS_STREAK_PAUSE_SEC
                logger.warning(
                    f"RiskManager: {self.loss_streak} consecutive losses, pausing {LOSS_STREAK_PAUSE_SEC}s"
                )
        else:
            self.loss_streak = 0
