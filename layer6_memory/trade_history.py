"""Layer 6 (memory): rolling record of closed-trade outcomes across all
assets/timeframes, used for the rolling_win_rate_1h / rolling_win_rate_6h
features. Loads data/trades/paper_trades_v4.jsonl at startup and updates
live on every position close."""

import collections
import datetime
import json
import os

ROLLING_WIN_RATE_NEUTRAL = 0.5


class TradeHistory:
    def __init__(self, log_path: str = None):
        self._log_path = log_path
        self._closes = collections.deque()  # (closed_at: datetime, won: bool)
        if log_path and os.path.exists(log_path):
            self._load(log_path)

    def _load(self, log_path: str):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                closed_at = record.get("closed_at")
                if closed_at is None or "won" not in record:
                    continue
                self._closes.append((_parse_iso(closed_at), bool(record["won"])))

    def record_close(self, closed_at, won: bool):
        if isinstance(closed_at, str):
            closed_at = _parse_iso(closed_at)
        self._closes.append((closed_at, bool(won)))

    def win_rate(self, hours: float, now: datetime.datetime = None) -> float:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        cutoff = now - datetime.timedelta(hours=hours)
        recent = [won for closed_at, won in self._closes if closed_at >= cutoff]
        if not recent:
            return ROLLING_WIN_RATE_NEUTRAL
        return sum(recent) / len(recent)

    def __len__(self):
        return len(self._closes)


def _parse_iso(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
