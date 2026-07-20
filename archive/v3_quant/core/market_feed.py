"""Live market data for v3: a Binance WebSocket feed (spot price, momentum,
volume, book) plus a Polymarket REST feed for both 5-min and 15-min
up/down markets (yes_price, momentum, book depth), combined into one
snapshot per (asset, timeframe) every poll tick."""

import collections
import datetime
import json
import logging
import statistics
import threading
import time
from typing import Optional

import requests
import websocket

from config.settings import (
    ASSETS, BINANCE_SYMBOLS, BINANCE_WS_BASE, BINANCE_KLINE_HISTORY_MIN,
    BINANCE_RECONNECT_BACKOFF_SEC, TIMEFRAMES, CONTEXT_POLL_INTERVAL_SEC,
    POLYMARKET_GAMMA_BASE, POLYMARKET_API_BASE,
)

logger = logging.getLogger(__name__)

# enough 3s-spaced samples to look back 120s+ with margin
YES_PRICE_HISTORY_MAXLEN = 80


def compute_book_imbalance(bid_size: float, ask_size: float) -> Optional[float]:
    """(bid-ask)/(bid+ask), always in [-1, 1]; None when the book is empty."""
    total = bid_size + ask_size
    if total <= 0:
        return None
    return (bid_size - ask_size) / total


class BinanceFeed:
    """Spot price / volume / book state for every symbol in `symbols`, via
    one combined Binance WebSocket stream (1m klines + top-5 depth). Runs
    its own reconnect loop with backoff so a dropped connection doesn't
    kill the bot."""

    def __init__(self, symbols: dict = None):
        self.symbols = symbols or BINANCE_SYMBOLS
        self._sym_to_asset = {v: k for k, v in self.symbols.items()}
        self._lock = threading.Lock()
        self._klines = {asset: collections.deque(maxlen=BINANCE_KLINE_HISTORY_MIN) for asset in self.symbols}
        self._depth = {asset: {} for asset in self.symbols}
        self._last_price = {}
        self._stop = threading.Event()
        self._ws = None
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_with_reconnect, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    def _stream_url(self) -> str:
        streams = []
        for sym in self.symbols.values():
            streams.append(f"{sym}@kline_1m")
            streams.append(f"{sym}@depth5@100ms")
        return f"{BINANCE_WS_BASE}?streams=" + "/".join(streams)

    def _run_with_reconnect(self):
        attempt = 0
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self._stream_url(),
                    on_message=self._on_message,
                    on_error=self._on_error,
                )
                attempt = 0
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"BinanceFeed connection error: {e}")
            if self._stop.is_set():
                return
            attempt += 1
            backoff = min(BINANCE_RECONNECT_BACKOFF_SEC * attempt, 60)
            logger.warning(f"BinanceFeed reconnecting in {backoff}s (attempt {attempt})")
            time.sleep(backoff)

    def _on_error(self, ws, error):
        logger.error(f"BinanceFeed ws error: {error}")

    def _on_message(self, ws, message):
        try:
            payload = json.loads(message)
            stream = payload.get("stream", "")
            data = payload.get("data", {})
            if "@kline" in stream:
                self._handle_kline(data)
            elif "@depth" in stream:
                self._handle_depth(stream, data)
        except Exception as e:
            logger.error(f"BinanceFeed message parse error: {e}")

    def _handle_kline(self, data: dict):
        k = data.get("k", {})
        sym = data.get("s", "").lower()
        asset = self._sym_to_asset.get(sym)
        if asset is None:
            return
        close = float(k.get("c", 0))
        volume = float(k.get("v", 0))
        with self._lock:
            self._last_price[asset] = close
            if k.get("x"):
                self._klines[asset].append({"close": close, "volume": volume})

    def _handle_depth(self, stream: str, data: dict):
        sym = stream.split("@")[0]
        asset = self._sym_to_asset.get(sym)
        if asset is None:
            return
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        bid_prices = [float(b[0]) for b in bids] if bids else []
        ask_prices = [float(a[0]) for a in asks] if asks else []
        bid_size = sum(float(b[1]) for b in bids)
        ask_size = sum(float(a[1]) for a in asks)
        with self._lock:
            self._depth[asset] = {
                "bid_size": bid_size,
                "ask_size": ask_size,
                "best_bid": max(bid_prices) if bid_prices else None,
                "best_ask": min(ask_prices) if ask_prices else None,
            }

    def get_price(self, asset: str) -> Optional[float]:
        with self._lock:
            return self._last_price.get(asset)

    def get_momentum_bps(self, asset: str, minutes: int) -> Optional[float]:
        with self._lock:
            price = self._last_price.get(asset)
            bars = list(self._klines.get(asset, ()))
        if price is None or len(bars) < minutes:
            return None
        past_close = bars[-minutes]["close"]
        if not past_close:
            return None
        return (price - past_close) / past_close * 10000

    def get_volume(self, asset: str, minutes: int) -> Optional[float]:
        """Raw traded volume (base asset units) summed over the last
        `minutes` finalized 1m bars."""
        with self._lock:
            bars = list(self._klines.get(asset, ()))
        if len(bars) < minutes:
            return None
        return sum(b["volume"] for b in bars[-minutes:])

    def get_volume_ratio(self, asset: str) -> Optional[float]:
        with self._lock:
            bars = list(self._klines.get(asset, ()))
        if len(bars) < 10:
            return None
        recent = bars[-5:]
        baseline = bars[:-5]
        recent_vol = sum(b["volume"] for b in recent) / len(recent)
        baseline_vols = [b["volume"] for b in baseline]
        if len(baseline_vols) < 2:
            return None
        mean = statistics.mean(baseline_vols)
        stdev = statistics.pstdev(baseline_vols)
        if stdev == 0:
            return 0.0
        return (recent_vol - mean) / stdev

    def get_bid_ask_imbalance(self, asset: str) -> Optional[float]:
        with self._lock:
            depth = dict(self._depth.get(asset, {}))
        bid = depth.get("bid_size")
        ask = depth.get("ask_size")
        if bid is None or ask is None or (bid + ask) == 0:
            return None
        return (bid - ask) / (bid + ask)

    def get_bid_ask_spread(self, asset: str) -> Optional[float]:
        """Relative spread (ask-bid)/mid -- dimensionless, so it's
        comparable across BTC/ETH/SOL's very different price scales."""
        with self._lock:
            depth = dict(self._depth.get(asset, {}))
        bid = depth.get("best_bid")
        ask = depth.get("best_ask")
        if bid is None or ask is None:
            return None
        mid = (bid + ask) / 2
        if not mid:
            return None
        return (ask - bid) / mid


class PolymarketFeed:
    """Yes-price / book / timing state for the BTC/ETH/SOL up/down markets,
    for every timeframe in TIMEFRAMES, fetched over REST (Polymarket has no
    public WebSocket for market data)."""

    def __init__(self, session: requests.Session = None):
        self.session = session or requests.Session()
        self._market_cache = {}  # timeframe -> {"markets": {...}, "fetched_at": ts}
        self._yes_price_history = {}  # (asset, timeframe) -> deque[(ts, price)]
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
                self._record_yes_price(asset, timeframe, now, parsed["yes_price"])
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
            if yes_price is None:
                return None
            bid_size_yes = self._total_size(yes_book.get("bids") if yes_book else None)
            ask_size_yes = self._total_size(yes_book.get("asks") if yes_book else None)
            bid_size_no = self._total_size(no_book.get("bids") if no_book else None)
            ask_size_no = self._total_size(no_book.get("asks") if no_book else None)
            book_imbalance = compute_book_imbalance(bid_size_yes, ask_size_yes)

            end_date = m.get("endDate") or ""
            seconds_remaining = float(window_sec)
            if end_date:
                end_dt = datetime.datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                seconds_remaining = max(0.0, (end_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds())

            return {
                "market_id": m.get("conditionId") or m.get("id", ""),
                "yes_price": yes_price,
                "up_token_id": up_token,
                "down_token_id": down_token,
                "seconds_remaining": seconds_remaining,
                "volume_usd": float(m.get("volumeNum") or m.get("volume") or 0),
                "bid_size_yes": bid_size_yes,
                "ask_size_yes": ask_size_yes,
                "bid_size_no": bid_size_no,
                "ask_size_no": ask_size_no,
                "book_imbalance": book_imbalance,
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

    def _record_yes_price(self, asset: str, timeframe: str, ts: float, yes_price: float):
        key = (asset, timeframe)
        history = self._yes_price_history.setdefault(key, collections.deque(maxlen=YES_PRICE_HISTORY_MAXLEN))
        history.append((ts, yes_price))

    def get_yes_price_change(self, asset: str, timeframe: str, lookback_sec: int) -> Optional[float]:
        history = self._yes_price_history.get((asset, timeframe))
        if not history or len(history) < 2:
            return None
        now = history[-1][0]
        current = history[-1][1]
        target_ts = now - lookback_sec
        # nearest sample at or before target_ts
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
        history = self._volume_history.setdefault(key, collections.deque(maxlen=YES_PRICE_HISTORY_MAXLEN))
        history.append(volume_usd)

    def get_volume_ratio_window(self, asset: str, timeframe: str) -> Optional[float]:
        """z-score of this window's traded volume vs the recent baseline --
        same shape as BinanceFeed.get_volume_ratio(), just for the
        Polymarket market's own volume series."""
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
        already shows a near-certain winner (>0.95), same oracle-lag
        shortcut used in v2."""
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


class MarketFeed:
    """Combines BinanceFeed + PolymarketFeed into one snapshot per
    (asset, timeframe)."""

    def __init__(self, binance: BinanceFeed = None, polymarket: PolymarketFeed = None):
        self.binance = binance or BinanceFeed()
        self.polymarket = polymarket or PolymarketFeed()

    def start(self):
        self.binance.start()

    def stop(self):
        self.binance.stop()

    def snapshot(self, asset: str, timeframe: str) -> dict:
        now = datetime.datetime.now(datetime.timezone.utc)
        markets = self.polymarket.get_markets(timeframe)
        market = markets.get(asset, {})
        return {
            "asset": asset,
            "timeframe": timeframe,
            "timestamp": now.isoformat(),
            "hour_utc": now.hour,
            "weekday": now.weekday(),
            # Binance
            "spot_price": self.binance.get_price(asset),
            "price_change_1m": self.binance.get_momentum_bps(asset, 1),
            "price_change_5m": self.binance.get_momentum_bps(asset, 5),
            "price_change_15m": self.binance.get_momentum_bps(asset, 15),
            "price_change_30m": self.binance.get_momentum_bps(asset, 30),
            "volume_1m": self.binance.get_volume(asset, 1),
            "volume_5m": self.binance.get_volume(asset, 5),
            "volume_ratio": self.binance.get_volume_ratio(asset),
            "bid_ask_spread": self.binance.get_bid_ask_spread(asset),
            "bid_ask_imbalance": self.binance.get_bid_ask_imbalance(asset),
            # Polymarket
            "yes_price": market.get("yes_price"),
            "yes_price_change_60s": self.polymarket.get_yes_price_change(asset, timeframe, 60),
            "yes_price_change_120s": self.polymarket.get_yes_price_change(asset, timeframe, 120),
            "volume_usd": market.get("volume_usd"),
            "volume_ratio_window": self.polymarket.get_volume_ratio_window(asset, timeframe),
            "seconds_remaining": market.get("seconds_remaining"),
            "market_id": market.get("market_id"),
            "up_token_id": market.get("up_token_id"),
            "bid_size_yes": market.get("bid_size_yes"),
            "ask_size_yes": market.get("ask_size_yes"),
            "bid_size_no": market.get("bid_size_no"),
            "ask_size_no": market.get("ask_size_no"),
            "book_imbalance": market.get("book_imbalance"),
        }
