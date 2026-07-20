"""Layer 6 (memory): remembers what conditions past trades were opened
under, and reports historical performance for a given set of conditions --
used by the filters to skip setups that have historically lost money."""

import datetime
import json
import logging
import os

from config.settings import PATTERN_MEMORY_FILE, PATTERN_MIN_TRADES_FOR_SIGNAL, PATTERN_BREAKEVEN_AVG_PNL

logger = logging.getLogger(__name__)


def price_bucket(yes_price) -> str:
    if yes_price is None:
        return "unknown"
    if yes_price < 0.45:
        return "low"
    if yes_price > 0.55:
        return "high"
    return "mid"


def hour_bucket(hour_utc) -> str:
    if hour_utc is None:
        return "unknown"
    block = (hour_utc // 6) * 6
    return f"{block:02d}-{block + 6:02d}"


class PatternMemory:
    def __init__(self, path: str = PATTERN_MEMORY_FILE):
        self.path = path
        self._records = []  # [{conditions, won, pnl, timestamp}]
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._records = data.get("records", [])
        except Exception as e:
            logger.warning(f"PatternMemory load error: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"records": self._records}, f)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.error(f"PatternMemory save error: {e}")

    def record(self, conditions: dict, won: bool, pnl: float, timestamp: datetime.datetime = None):
        timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)
        self._records.append({
            "conditions": conditions,
            "won": won,
            "pnl": pnl,
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        })
        self._save()

    def get_historical_performance(self, conditions: dict) -> dict:
        matches = [r for r in self._records if self._matches(r["conditions"], conditions)]
        n = len(matches)
        if n == 0:
            return {"n_trades": 0, "win_rate": None, "avg_pnl": None}
        wins = sum(1 for r in matches if r["won"])
        avg_pnl = sum(r["pnl"] for r in matches) / n
        return {"n_trades": n, "win_rate": wins / n, "avg_pnl": avg_pnl}

    def should_avoid(self, conditions: dict) -> bool:
        """True once there's enough history for these exact conditions to
        trust the stat, and that history's average pnl is below breakeven."""
        perf = self.get_historical_performance(conditions)
        if perf["n_trades"] < PATTERN_MIN_TRADES_FOR_SIGNAL:
            return False
        return perf["avg_pnl"] < PATTERN_BREAKEVEN_AVG_PNL

    def _matches(self, stored: dict, query: dict) -> bool:
        return all(stored.get(k) == v for k, v in query.items())
