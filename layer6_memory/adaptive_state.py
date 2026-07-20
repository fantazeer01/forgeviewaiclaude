"""Layer 6 (memory): tracks the bot's current market regime and a
per-timeframe "model temperature" -- hot/cold/neutral based on the last N
outcomes *for that timeframe specifically*, not a single blended number.
5m and 15m markets can perform very differently at the same moment (see the
2026-07-20 post-peak analysis, where 5m win rate collapsed to 9% while 15m
held at 50%), so a single shared temperature would hide exactly the signal
this is meant to catch. Exports to data/market/adaptive_state.json."""

import collections
import datetime
import json
import logging
import os

from config.settings import (
    ADAPTIVE_STATE_FILE, ADAPTIVE_LOOKBACK_TRADES, ADAPTIVE_HOT_WIN_RATE,
    ADAPTIVE_COLD_WIN_RATE, ADAPTIVE_COLD_SIZE_MULTIPLIER, TIMEFRAMES,
)

logger = logging.getLogger(__name__)

HOT = "hot"
COLD = "cold"
NEUTRAL = "neutral"


class AdaptiveState:
    def __init__(self, path: str = ADAPTIVE_STATE_FILE, lookback: int = ADAPTIVE_LOOKBACK_TRADES,
                 timeframes=None):
        self.path = path
        self.lookback = lookback
        self.timeframes = list(timeframes) if timeframes is not None else list(TIMEFRAMES)
        self._recent_outcomes = {tf: collections.deque(maxlen=lookback) for tf in self.timeframes}
        self._temperatures = {tf: NEUTRAL for tf in self.timeframes}
        self.regime = None

    def record_close(self, timeframe: str, won: bool):
        outcomes = self._recent_outcomes.setdefault(timeframe, collections.deque(maxlen=self.lookback))
        outcomes.append(bool(won))
        self._recompute_temperature(timeframe)

    def _recompute_temperature(self, timeframe: str):
        outcomes = self._recent_outcomes.get(timeframe, ())
        if len(outcomes) < self.lookback:
            self._temperatures[timeframe] = NEUTRAL
            return
        win_rate = sum(outcomes) / len(outcomes)
        if win_rate > ADAPTIVE_HOT_WIN_RATE:
            self._temperatures[timeframe] = HOT
        elif win_rate < ADAPTIVE_COLD_WIN_RATE:
            self._temperatures[timeframe] = COLD
        else:
            self._temperatures[timeframe] = NEUTRAL

    def temperature(self, timeframe: str) -> str:
        return self._temperatures.get(timeframe, NEUTRAL)

    def is_cold(self, timeframe: str) -> bool:
        return self.temperature(timeframe) == COLD

    def win_rate(self, timeframe: str):
        outcomes = self._recent_outcomes.get(timeframe)
        if not outcomes:
            return None
        return sum(outcomes) / len(outcomes)

    def set_regime(self, regime: str):
        self.regime = regime

    def size_multiplier(self, timeframe: str) -> float:
        return ADAPTIVE_COLD_SIZE_MULTIPLIER if self.is_cold(timeframe) else 1.0

    def export(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "regime": self.regime,
                    "temperatures": dict(self._temperatures),
                    "recent_win_rates": {tf: self.win_rate(tf) for tf in self.timeframes},
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }, f)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.error(f"AdaptiveState export error: {e}")
