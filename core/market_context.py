"""Live market context: BTC/ETH spot state from a single Binance WebSocket
feed (never REST-polled), plus Polymarket yes_price and Fear&Greed, combined
into one snapshot per asset every poll tick."""

import collections
import datetime
import json
import logging
import os
import statistics
import threading
import time
from typing import Optional

import requests
import websocket

from config.settings import (
    ASSETS, BINANCE_SYMBOLS, BINANCE_WS_BASE, BINANCE_KLINE_HISTORY_MIN,
    BINANCE_RECONNECT_BACKOFF_SEC, WINDOW_SEC, CONTEXT_POLL_INTERVAL_SEC,
    POLYMARKET_GAMMA_BASE, POLYMARKET_API_BASE,
    FEAR_GREED_API_BASE, FEAR_GREED_REFRESH_SEC, FEAR_GREED_LOG,
)

logger = logging.getLogger(__name__)


class BinanceFeed:
    """Maintains live BTC/ETH price + depth state via one combined Binance
    WebSocket stream (1m klines + top-5 depth). Runs its own reconnect loop
    with backoff so a dropped connection doesn't kill the bot."""

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
        """Reconnect loop: run_forever() blocks until the socket drops (or
        raises), then this backs off and reconnects -- a dropped connection
        never leaves the feed permanently dead."""
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
        bid_size = sum(float(b[1]) for b in bids)
        ask_size = sum(float(a[1]) for a in asks)
        with self._lock:
            self._depth[asset] = {"bid_size": bid_size, "ask_size": ask_size}

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


class MarketContext:
    """Combines the live BinanceFeed with Polymarket yes_price and
    Fear&Greed into a single per-asset snapshot every poll tick."""

    def __init__(self, feed: BinanceFeed = None, session: requests.Session = None):
        self.feed = feed or BinanceFeed()
        self.session = session or requests.Session()
        self._fg_cache = {"value": None, "fetched_at": 0.0}
        self._market_cache = {"markets": {}, "fetched_at": 0.0}

    def start(self):
        self.feed.start()

    def stop(self):
        self.feed.stop()

    def get_fear_greed(self) -> Optional[int]:
        now = time.time()
        if self._fg_cache["value"] is not None and now - self._fg_cache["fetched_at"] < FEAR_GREED_REFRESH_SEC:
            return self._fg_cache["value"]
        try:
            resp = self.session.get(FEAR_GREED_API_BASE, timeout=10)
            resp.raise_for_status()
            value = int(resp.json()["data"][0]["value"])
            self._fg_cache = {"value": value, "fetched_at": now}
            self._export_fear_greed(value)
            return value
        except Exception as e:
            logger.warning(f"fear_greed fetch error: {e}")
            return self._fg_cache["value"]

    def _export_fear_greed(self, value: int):
        try:
            os.makedirs(os.path.dirname(FEAR_GREED_LOG), exist_ok=True)
            with open(FEAR_GREED_LOG, "w") as f:
                json.dump({
                    "value": value,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }, f)
        except Exception as e:
            logger.error(f"fear_greed export error: {e}")

    def _current_boundary(self, now: float) -> int:
        now_int = int(now)
        return now_int - (now_int % WINDOW_SEC)

    def get_polymarket_snapshot(self) -> dict:
        now = time.time()
        if self._market_cache["markets"] and now - self._market_cache["fetched_at"] < CONTEXT_POLL_INTERVAL_SEC:
            return self._market_cache["markets"]
        boundary = self._current_boundary(now)
        slugs = [f"{a.lower()}-updown-5m-{boundary}" for a in ASSETS]
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
            logger.warning(f"Polymarket fetch error: {e}")
            return self._market_cache["markets"]

        markets = {}
        for m in raw_markets:
            if m.get("closed"):
                continue
            asset = self._asset_from_slug(m.get("slug", ""))
            if asset is None:
                continue
            parsed = self._parse_market(m)
            if parsed:
                markets[asset] = parsed
        self._market_cache = {"markets": markets, "fetched_at": now}
        return markets

    def _asset_from_slug(self, slug: str) -> Optional[str]:
        for asset in ASSETS:
            if slug.startswith(f"{asset.lower()}-updown-5m-"):
                return asset
        return None

    def _parse_market(self, m: dict) -> Optional[dict]:
        try:
            outcomes = json.loads(m.get("outcomes", "[]"))
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            token_by_label = {str(l).strip().lower(): str(t) for l, t in zip(outcomes, token_ids)}
            up_token = token_by_label.get("up")
            if not up_token:
                return None
            yes_price = self._token_mid_price(up_token)
            if yes_price is None:
                return None
            end_date = m.get("endDate") or ""
            seconds_remaining = float(WINDOW_SEC)
            if end_date:
                end_dt = datetime.datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                seconds_remaining = max(0.0, (end_dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
            return {
                "market_id": m.get("conditionId") or m.get("id", ""),
                "yes_price": yes_price,
                "up_token_id": up_token,
                "seconds_remaining": seconds_remaining,
            }
        except Exception as e:
            logger.warning(f"_parse_market error: {e}")
            return None

    def _token_mid_price(self, token_id: str) -> Optional[float]:
        try:
            resp = self.session.get(f"{POLYMARKET_API_BASE}/book", params={"token_id": token_id}, timeout=10)
            resp.raise_for_status()
            book = resp.json()
            bid = self._best_price(book.get("bids"), highest=True)
            ask = self._best_price(book.get("asks"), highest=False)
            if bid is not None and ask is not None:
                return (bid + ask) / 2
            return ask if ask is not None else bid
        except Exception as e:
            logger.warning(f"_token_mid_price error token={token_id}: {e}")
            return None

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

    def get_resolution(self, market_id: str) -> Optional[str]:
        """UP / DOWN once the market has resolved, else None."""
        try:
            resp = self.session.get(f"{POLYMARKET_API_BASE}/markets/{market_id}", timeout=10)
            resp.raise_for_status()
            resolution = resp.json()
        except Exception as e:
            logger.warning(f"get_resolution error market_id={market_id}: {e}")
            return None
        if not resolution.get("closed"):
            return None
        for token in resolution.get("tokens", []):
            if not token.get("winner"):
                continue
            outcome = str(token.get("outcome", "")).strip().lower()
            if outcome == "up":
                return "UP"
            if outcome == "down":
                return "DOWN"
        return None

    def snapshot(self, asset: str) -> dict:
        now = datetime.datetime.now(datetime.timezone.utc)
        markets = self.get_polymarket_snapshot()
        market = markets.get(asset, {})
        return {
            "asset": asset,
            "timestamp": now.isoformat(),
            "spot_price": self.feed.get_price(asset),
            "momentum_1m_bps": self.feed.get_momentum_bps(asset, 1),
            "momentum_5m_bps": self.feed.get_momentum_bps(asset, 5),
            "momentum_15m_bps": self.feed.get_momentum_bps(asset, 15),
            "momentum_60m_bps": self.feed.get_momentum_bps(asset, 60),
            "volume_ratio": self.feed.get_volume_ratio(asset),
            "bid_ask_imbalance": self.feed.get_bid_ask_imbalance(asset),
            "fear_greed": self.get_fear_greed(),
            "hour_utc": now.hour,
            "weekday": now.weekday(),
            "yes_price": market.get("yes_price"),
            "market_id": market.get("market_id"),
            "up_token_id": market.get("up_token_id"),
            "seconds_remaining": market.get("seconds_remaining"),
        }
