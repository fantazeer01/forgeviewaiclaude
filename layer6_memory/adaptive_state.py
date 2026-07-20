"""Layer 6 (memory): tracks the bot's current market regime and "model
temperature" -- hot/cold/neutral based on the last N trades' win rate --
and exports both to data/market/adaptive_state.json for the dashboard."""

import collections
import datetime
import json
import logging
import os

from config.settings import (
    ADAPTIVE_STATE_FILE, ADAPTIVE_LOOKBACK_TRADES, ADAPTIVE_HOT_WIN_RATE,
    ADAPTIVE_COLD_WIN_RATE, ADAPTIVE_COLD_SIZE_MULTIPLIER,
)

logger = logging.getLogger(__name__)

HOT = "hot"
COLD = "cold"
NEUTRAL = "neutral"


class AdaptiveState:
    def __init__(self, path: str = ADAPTIVE_STATE_FILE, lookback: int = ADAPTIVE_LOOKBACK_TRADES):
        self.path = path
        self.lookback = lookback
        self._recent_outcomes = collections.deque(maxlen=lookback)
        self.regime = None
        self.temperature = NEUTRAL

    def record_close(self, won: bool):
        self._recent_outcomes.append(bool(won))
        self._recompute_temperature()

    def _recompute_temperature(self):
        if len(self._recent_outcomes) < self.lookback:
            self.temperature = NEUTRAL
            return
        win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
        if win_rate > ADAPTIVE_HOT_WIN_RATE:
            self.temperature = HOT
        elif win_rate < ADAPTIVE_COLD_WIN_RATE:
            self.temperature = COLD
        else:
            self.temperature = NEUTRAL

    def set_regime(self, regime: str):
        self.regime = regime

    def size_multiplier(self) -> float:
        return ADAPTIVE_COLD_SIZE_MULTIPLIER if self.temperature == COLD else 1.0

    def export(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "regime": self.regime,
                    "temperature": self.temperature,
                    "recent_win_rate": (
                        sum(self._recent_outcomes) / len(self._recent_outcomes)
                        if self._recent_outcomes else None
                    ),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }, f)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.error(f"AdaptiveState export error: {e}")
