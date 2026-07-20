"""Layer 1 (eyes): watches recent Polymarket trades for large ($500+) fills
and tracks buy/sell pressure over a rolling window. The exact public
trade-history endpoint on Polymarket's CLOB isn't guaranteed stable, so
every call is defensive: any fetch/parse failure just yields zero pressure
for that tick rather than raising."""

import collections
import datetime
import logging
import time
from typing import Optional

import requests

from config.settings import POLYMARKET_API_BASE, WHALE_TRADE_MIN_USD, WHALE_WINDOW_MINUTES

logger = logging.getLogger(__name__)


class WhaleTracker:
    def __init__(self, session: requests.Session = None, min_trade_usd: float = WHALE_TRADE_MIN_USD,
                 window_minutes: float = WHALE_WINDOW_MINUTES):
        self.session = session or requests.Session()
        self.min_trade_usd = min_trade_usd
        self.window_minutes = window_minutes
        self._trades = collections.deque()  # (ts: float epoch, side: "YES"/"NO", usd_size: float)

    def poll(self, token_id: str, now: float = None):
        now = time.time() if now is None else now
        for trade in self._fetch_trades(token_id):
            if trade["usd_size"] >= self.min_trade_usd:
                self._trades.append((trade["ts"], trade["side"], trade["usd_size"]))
        self._prune(now)

    def _fetch_trades(self, token_id: str) -> list:
        try:
            resp = self.session.get(
                f"{POLYMARKET_API_BASE}/trades", params={"market": token_id}, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"WhaleTracker fetch error token={token_id}: {e}")
            return []

        raw_trades = data if isinstance(data, list) else data.get("trades", [])
        trades = []
        for item in raw_trades:
            trade = self._parse_trade(item)
            if trade is not None:
                trades.append(trade)
        return trades

    def _parse_trade(self, item: dict) -> Optional[dict]:
        try:
            price = float(item.get("price"))
            size = float(item.get("size"))
            side_raw = str(item.get("side", "")).strip().upper()
            side = "YES" if side_raw in ("BUY", "YES") else "NO"
            ts = self._parse_ts(item.get("timestamp") or item.get("match_time"))
            if ts is None:
                return None
            return {"ts": ts, "side": side, "usd_size": price * size}
        except (TypeError, ValueError, AttributeError):
            return None

    def _parse_ts(self, value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
        try:
            return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def _prune(self, now: float):
        cutoff = now - self.window_minutes * 60
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

    def whale_buy_pressure(self) -> float:
        return sum(usd for _, side, usd in self._trades if side == "YES")

    def whale_sell_pressure(self) -> float:
        return sum(usd for _, side, usd in self._trades if side == "NO")

    def whale_imbalance(self) -> float:
        buy = self.whale_buy_pressure()
        sell = self.whale_sell_pressure()
        total = buy + sell
        if total <= 0:
            return 0.0
        return (buy - sell) / total

    def whale_activity(self) -> int:
        return len(self._trades)

    def whale_buy_count(self) -> int:
        return sum(1 for _, side, _ in self._trades if side == "YES")
