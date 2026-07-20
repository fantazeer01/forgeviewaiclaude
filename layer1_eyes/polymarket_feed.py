"""Layer 1 (eyes): Polymarket REST feed for BTC/ETH/SOL 5-min and 15-min
up/down markets. yes_price and no_price are read independently from their
own order books (not derived as 1-yes) since the two sides can trade at a
combined price away from exactly 1.0."""

import collections
import datetime
import json
import logging
import statistics
import time
from typing import Optional

import requests

from config.settings import (
    ASSETS, TIMEFRAMES, CONTEXT_POLL_INTERVAL_SEC, POLYMARKET_GAMMA_BASE, POLYMARKET_API_BASE,
)

logger = logging.getLogger(__name__)

# enough 3s-spaced samples to look back 180s+ with margin
PRICE_HISTORY_MAXLEN = 100


def compute_book_imbalance(depth_yes: float, depth_no: float) -> Optional[float]:
    """(depth_yes - depth_no)/(depth_yes + depth_no), always in [-1, 1];
    None when both sides are empty."""
    total = depth_yes + depth_no
    if total <= 0:
        return None
    return (depth_yes - depth_no) / total


class PolymarketFeed:
    def __init__(self, session: requests.Session = None):
        self.session = session or requests.Session()
        self._market_cache = {}  # timeframe -> {"markets": {...}, "fetched_at": ts}
        self._price_history = {}  # (asset, timeframe) -> deque[(ts, yes_price)]
        self._volume_history = {}  # (asset, timeframe) -> deque[volume_usd samples]

    def _current_boundary(self, now: float, window_sec: int) -> int:
        now_int = int(now)
        return now_int - (now_int % window_sec)

    def get_markets(self, timeframe: str) -> dict:
        window_sec = TIMEFRAMES[timeframe]
        now = time.time()
        cached = self._market_cache.get(timeframe, {"markets": {}, "fetched_at": 0.0})
        if cached["markets"] and now - cached["fetched_at"] < CONTEXT_POLL_INTERVAL_SEC:
            return cached["markets"]

        boundary = self._current_boundary(now, window_sec)
        slugs = [f"{a.lower()}-updown-{timeframe}-{boundary}" for a in ASSETS]
        try:
            resp = self.session.get(
                f"{POLYMARKET_GAMMA_BASE}/markets",
                params=[("slug", s) for s in slugs] + [("limit", len(slugs))],
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_markets = data if isinstance(data, list) else data.get("markets", [])
        except Exception as e:
            logger.warning(f"PolymarketFeed fetch error ({timeframe}): {e}")
            return cached["markets"]

        markets = {}
        for m in raw_markets:
            if m.get("closed"):
                continue
            asset = self._asset_from_slug(m.get("slug", ""), timeframe)
            if asset is None:
                continue
            parsed = self._parse_market(m, window_sec)
            if parsed:
                markets[asset] = parsed
                self._record_price(asset, timeframe, now, parsed["yes_price"])
                self._record_volume(asset, timeframe, parsed["volume_usd"])

        self._market_cache[timeframe] = {"markets": markets, "fetched_at": now}
        return markets

    def _asset_from_slug(self, slug: str, timeframe: str) -> Optional[str]:
        for asset in ASSETS:
            if slug.startswith(f"{asset.lower()}-updown-{timeframe}-"):
                return asset
        return None

    def _parse_market(self, m: dict, window_sec: int) -> Optional[dict]:
        try:
            outcomes = json.loads(m.get("outcomes", "[]"))
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            token_by_label = {str(l).strip().lower(): str(t) for l, t in zip(outcomes, token_ids)}
            up_token = token_by_label.get("up")
            down_token = token_by_label.get("down")
            if not up_token or not down_token:
                return None
            yes_book = self._fetch_book(up_token)
            no_book = self._fetch_book(down_token)
            yes_price = self._mid_price(yes_book)
            no_price = self._mid_price(no_book)
            if yes_price is None:
                return None

            bid_size_yes = self._total_size(yes_book.get("bids") if yes_book else None)
            ask_size_yes = self._total_size(yes_book.get("asks") if yes_book else None)
            bid_size_no = self._total_size(no_book.get("bids") if no_book else None)
            ask_size_no = self._total_size(no_book.get("asks") if no_book else None)
            # "depth" = buy-side interest on each token, per spec
            book_depth_yes = bid_size_yes
            book_depth_no = bid_size_no
            book_imbalance = compute_book_imbalance(book_depth_yes, book_depth_no)
            book_depth_ratio = (
                book_depth_yes / (book_depth_yes + book_depth_no)
                if (book_depth_yes + book_depth_no) > 0 else None
            )
            bid_ask_spread = self._relative_spread(yes_book)

            end_date = m.get("endDate") or ""
            seconds_remaining = float(window_sec)
            if end_date:
                end_dt = datetime.datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                seconds_remaining = max(0.0, (end_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds())

            return {
                "market_id": m.get("conditionId") or m.get("id", ""),
                "yes_price": yes_price,
                "no_price": no_price if no_price is not None else (1 - yes_price),
                "up_token_id": up_token,
                "down_token_id": down_token,
                "seconds_remaining": seconds_remaining,
                "window_sec": window_sec,
                "volume_usd": float(m.get("volumeNum") or m.get("volume") or 0),
                "bid_size_yes": bid_size_yes,
                "ask_size_yes": ask_size_yes,
                "bid_size_no": bid_size_no,
                "ask_size_no": ask_size_no,
                "book_depth_yes": book_depth_yes,
                "book_depth_no": book_depth_no,
                "book_imbalance": book_imbalance,
                "book_depth_ratio": book_depth_ratio,
                "bid_ask_spread": bid_ask_spread,
            }
        except Exception as e:
            logger.warning(f"_parse_market error: {e}")
            return None

    def _fetch_book(self, token_id: str) -> Optional[dict]:
        try:
            resp = self.session.get(f"{POLYMARKET_API_BASE}/book", params={"token_id": token_id}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"_fetch_book error token={token_id}: {e}")
            return None

    def _mid_price(self, book: Optional[dict]) -> Optional[float]:
        if not book:
            return None
        bid = self._best_price(book.get("bids"), highest=True)
        ask = self._best_price(book.get("asks"), highest=False)
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return ask if ask is not None else bid

    def _relative_spread(self, book: Optional[dict]) -> Optional[float]:
        if not book:
            return None
        bid = self._best_price(book.get("bids"), highest=True)
        ask = self._best_price(book.get("asks"), highest=False)
        if bid is None or ask is None:
            return None
        mid = (bid + ask) / 2
        if not mid:
            return None
        return (ask - bid) / mid

    def _best_price(self, levels: Optional[list], highest: bool) -> Optional[float]:
        prices = []
        for level in levels or []:
            try:
                prices.append(float(level.get("price")))
            except (TypeError, ValueError, AttributeError):
                continue
        if not prices:
            return None
        return max(prices) if highest else min(prices)

    def _total_size(self, levels: Optional[list]) -> float:
        total = 0.0
        for level in levels or []:
            try:
                total += float(level.get("size", 0))
            except (TypeError, ValueError, AttributeError):
                continue
        return total

    def _record_price(self, asset: str, timeframe: str, ts: float, yes_price: float):
        key = (asset, timeframe)
        history = self._price_history.setdefault(key, collections.deque(maxlen=PRICE_HISTORY_MAXLEN))
        history.append((ts, yes_price))

    def get_yes_price_change(self, asset: str, timeframe: str, lookback_sec: int) -> Optional[float]:
        history = self._price_history.get((asset, timeframe))
        if not history or len(history) < 2:
            return None
        now = history[-1][0]
        current = history[-1][1]
        target_ts = now - lookback_sec
        past = None
        for ts, price in history:
            if ts <= target_ts:
                past = price
            else:
                break
        if past is None:
            return None
        return current - past

    def _record_volume(self, asset: str, timeframe: str, volume_usd: float):
        key = (asset, timeframe)
        history = self._volume_history.setdefault(key, collections.deque(maxlen=PRICE_HISTORY_MAXLEN))
        history.append(volume_usd)

    def get_volume_ratio_window(self, asset: str, timeframe: str) -> Optional[float]:
        history = self._volume_history.get((asset, timeframe))
        if not history or len(history) < 10:
            return None
        values = list(history)
        recent = values[-5:]
        baseline = values[:-5]
        recent_avg = sum(recent) / len(recent)
        if len(baseline) < 2:
            return None
        mean = statistics.mean(baseline)
        stdev = statistics.pstdev(baseline)
        if stdev == 0:
            return 0.0
        return (recent_avg - mean) / stdev

    def get_resolution(self, market_id: str) -> Optional[str]:
        """UP / DOWN once the market has resolved -- either Polymarket has
        marked it closed with a winning token, or a token's live price
        already shows a near-certain winner (>0.95)."""
        try:
            resp = self.session.get(f"{POLYMARKET_API_BASE}/markets/{market_id}", timeout=10)
            resp.raise_for_status()
            resolution = resp.json()
        except Exception as e:
            logger.warning(f"get_resolution error market_id={market_id}: {e}")
            return None

        tokens = resolution.get("tokens", [])
        if resolution.get("closed"):
            for token in tokens:
                if not token.get("winner"):
                    continue
                outcome = str(token.get("outcome", "")).strip().lower()
                if outcome == "up":
                    return "UP"
                if outcome == "down":
                    return "DOWN"
            return None
        return self._resolution_from_price(tokens)

    RESOLUTION_PRICE_THRESHOLD = 0.95

    def _resolution_from_price(self, tokens: list) -> Optional[str]:
        best_outcome, best_price = None, -1.0
        for token in tokens:
            try:
                price = float(token.get("price"))
            except (TypeError, ValueError):
                continue
            if price > best_price:
                best_price = price
                best_outcome = str(token.get("outcome", "")).strip().lower()
        if best_price <= self.RESOLUTION_PRICE_THRESHOLD:
            return None
        if best_outcome == "up":
            return "UP"
        if best_outcome == "down":
            return "DOWN"
        return None
