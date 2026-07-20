"""Layer 1 (eyes): Binance spot price/volume/book state via one combined
WebSocket stream (1m klines + top-5 depth), with automatic reconnect."""

import collections
import json
import logging
import statistics
import threading
import time
from typing import Optional

import websocket

from config.settings import (
    BINANCE_SYMBOLS, BINANCE_WS_BASE, BINANCE_KLINE_HISTORY_MIN, BINANCE_RECONNECT_BACKOFF_SEC,
)

logger = logging.getLogger(__name__)


class BinanceFeed:
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
                    self._stream_url(), on_message=self._on_message, on_error=self._on_error,
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

    def get_volume_trend(self, asset: str) -> Optional[float]:
        """+1 growing, -1 shrinking, 0 flat -- compares the last 5 bars'
        average volume against the 5 bars before that."""
        with self._lock:
            bars = list(self._klines.get(asset, ()))
        if len(bars) < 10:
            return None
        recent = [b["volume"] for b in bars[-5:]]
        prior = [b["volume"] for b in bars[-10:-5]]
        recent_avg = sum(recent) / len(recent)
        prior_avg = sum(prior) / len(prior)
        if prior_avg == 0:
            return 0.0
        change = (recent_avg - prior_avg) / prior_avg
        if change > 0.05:
            return 1.0
        if change < -0.05:
            return -1.0
        return 0.0

    def get_bid_ask_imbalance(self, asset: str) -> Optional[float]:
        with self._lock:
            depth = dict(self._depth.get(asset, {}))
        bid = depth.get("bid_size")
        ask = depth.get("ask_size")
        if bid is None or ask is None or (bid + ask) == 0:
            return None
        return (bid - ask) / (bid + ask)

    def get_bid_ask_spread(self, asset: str) -> Optional[float]:
        """Relative spread (ask-bid)/mid -- dimensionless, comparable across
        BTC/ETH/SOL's very different price scales."""
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

    def get_volatility(self, asset: str, minutes: int) -> Optional[float]:
        """Std-dev (population) of minute-over-minute % returns over the
        last `minutes` finalized bars, in bps."""
        with self._lock:
            bars = list(self._klines.get(asset, ()))
        if len(bars) < minutes + 1:
            return None
        closes = [b["close"] for b in bars[-(minutes + 1):]]
        returns = []
        for prev, cur in zip(closes, closes[1:]):
            if prev:
                returns.append((cur - prev) / prev * 10000)
        if len(returns) < 2:
            return None
        return statistics.pstdev(returns)
