"""Bankroll and trade-frequency guardrails for v3 -- independent open-position
caps per timeframe, Kelly sizing after warmup, a temporary (not permanent)
pause on a losing streak, and full state persistence across restarts."""

import datetime
import json
import logging
import os
import time

from config.settings import (
    BANKROLL_USD, FIXED_POSITION_USD, KELLY_MIN_EXAMPLES, KELLY_MAX_FRACTION,
    MAX_DAILY_LOSS_USD, MAX_OPEN_POSITIONS_5M, MAX_OPEN_POSITIONS_15M,
    MAX_CONSECUTIVE_LOSSES, CONSECUTIVE_LOSS_PAUSE_MINUTES, RISK_STATE_FILE,
)

logger = logging.getLogger(__name__)

MAX_OPEN_POSITIONS = {"5m": MAX_OPEN_POSITIONS_5M, "15m": MAX_OPEN_POSITIONS_15M}


class RiskManager:
    def __init__(self, bankroll: float = BANKROLL_USD, state_file: str = RISK_STATE_FILE):
        self.state_file = state_file
        state = self._load_state()
        self.bankroll = state.get("bankroll", bankroll)
        self.trades_closed = state.get("trades_closed", 0)
        self.loss_streak = state.get("loss_streak", 0)
        self.paused_until = state.get("paused_until", 0.0)
        self.daily_pnl = 0.0
        self.daily_date = datetime.datetime.now(datetime.timezone.utc).date()
        self.open_positions = {"5m": 0, "15m": 0}

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                logger.info(
                    f"RiskManager: loaded state from {self.state_file} "
                    f"(trades_closed={data.get('trades_closed', 0)})"
                )
                return data
            except Exception as e:
                logger.warning(f"RiskManager state load error, starting fresh: {e}")
        else:
            logger.info(f"RiskManager: no state file at {self.state_file}, starting fresh")
        return {}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "bankroll": self.bankroll,
                    "trades_closed": self.trades_closed,
                    "loss_streak": self.loss_streak,
                    "paused_until": self.paused_until,
                }, f)
            os.replace(tmp, self.state_file)
        except Exception as e:
            logger.error(f"RiskManager state save error: {e}")

    def _roll_day(self):
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if today != self.daily_date:
            self.daily_date = today
            self.daily_pnl = 0.0

    def can_open_trade(self, timeframe: str, now: float = None) -> tuple:
        now = time.time() if now is None else now
        self._roll_day()
        if now < self.paused_until:
            return False, f"loss-streak pause until {self.paused_until}"
        if self.daily_pnl <= -MAX_DAILY_LOSS_USD:
            return False, "daily loss limit reached"
        cap = MAX_OPEN_POSITIONS.get(timeframe, 0)
        if self.open_positions.get(timeframe, 0) >= cap:
            return False, f"max open {timeframe} positions reached"
        return True, ""

    def position_size(self, win_probability: float, entry_price: float) -> float:
        """entry_price is the price of the side actually being bought (yes_price
        for a YES trade, 1-yes_price for a NO trade)."""
        if self.trades_closed < KELLY_MIN_EXAMPLES:
            return round(FIXED_POSITION_USD, 2)
        if not entry_price or entry_price <= 0:
            return 0.0
        b = (1 - entry_price) / entry_price
        if b <= 0:
            return 0.0
        edge = (win_probability * b - (1 - win_probability)) / b
        fraction = max(0.0, min(edge, KELLY_MAX_FRACTION))
        return round(self.bankroll * fraction, 2)

    def register_open(self, timeframe: str):
        self.open_positions[timeframe] = self.open_positions.get(timeframe, 0) + 1

    def register_close(self, timeframe: str, pnl: float):
        self._roll_day()
        self.open_positions[timeframe] = max(0, self.open_positions.get(timeframe, 0) - 1)
        self.trades_closed += 1
        self.daily_pnl += pnl
        self.bankroll += pnl
        if pnl < 0:
            self.loss_streak += 1
            if self.loss_streak >= MAX_CONSECUTIVE_LOSSES:
                self.paused_until = time.time() + CONSECUTIVE_LOSS_PAUSE_MINUTES * 60
                logger.warning(
                    f"RiskManager: {self.loss_streak} consecutive losses, "
                    f"pausing {CONSECUTIVE_LOSS_PAUSE_MINUTES}min"
                )
        else:
            self.loss_streak = 0
        self._save_state()
