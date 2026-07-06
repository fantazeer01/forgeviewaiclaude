import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MOMENTUM_30S_LOOKBACK = 30
MOMENTUM_60S_LOOKBACK = 60
HISTORY_RETENTION_SEC = 240
CORRELATION_WINDOW_SEC = 180
CORRELATION_MIN_RETURNS = 5

FEATURE_NAMES = [
    "yes_price", "no_price", "bid_ask_spread", "order_book_imbalance",
    "price_momentum_30s", "price_momentum_60s", "volume_24h",
    "time_remaining_pct", "btc_eth_correlation",
    # Appended 2026-07-06 -- MUST stay appended, never inserted/reordered:
    # OnlineQuantModel._migrate_new_features() only supports a clean append,
    # since the classifier's learned coefficients are positional. Reordering
    # these (or the 9 above) would silently mismap every existing weight to
    # the wrong feature.
    "ema_5", "ema_20", "price_vs_ema",
    "ohlc_open", "ohlc_high", "ohlc_low", "ohlc_close",
    "order_book_depth",
]

EMA_SHORT_SPAN = 5
EMA_LONG_SPAN = 20


class LiveFeatureCollector:
    """Collects the live features the online model trains on, sampled every
    poll tick (update() then extract()). Independent of QuantFeatureExtractor
    (core/quant_features.py, the older shadow-model feature set) so this
    online-learning feature set can evolve without touching that one.

    price_momentum_30s / 60s: signed yes_price change over the lookback
    window (positive = price rose), per-market so a new 5-minute window's
    own history never mixes with the previous window's.

    btc_eth_correlation: rolling Pearson correlation of BTC and ETH yes_price
    returns over the last CORRELATION_WINDOW_SEC seconds. Each asset's return
    series is reset when its active market_id rolls over to a new 5-minute
    window, since the new window's opening price is not a continuation of the
    previous window's price level and would otherwise inject a spurious
    return at the boundary.

    ema_5 / ema_20 (2026-07-06): standard EMA (alpha=2/(n+1)) over the last
    5 / 20 polls of this market_id -- fewer if the window hasn't run that
    long yet. price_vs_ema: yes_price/ema_20 - 1, how far current price sits
    from its own recent average.

    ohlc_open/high/low/close (2026-07-06): tracked in a dedicated
    per-market_id accumulator (_window_ohlc), NOT derived from
    _price_history -- that list is pruned to HISTORY_RETENTION_SEC (240s),
    shorter than a full 5-minute window, so it would lose the true opening
    tick once a window has been running more than ~4 minutes. _window_ohlc
    is never time-pruned; it's naturally bounded by market_id's own ~5-minute
    lifetime and reset the moment a new market_id is first seen.
    """

    def __init__(self):
        self._price_history: dict[str, list[dict]] = {}
        self._asset_return_history: dict[str, list[dict]] = {"BTC": [], "ETH": []}
        self._asset_current_market: dict[str, str] = {}
        self._window_ohlc: dict[str, dict] = {}

    def update(self, market_id: str, asset: str, yes_price: float, no_price: float):
        ts = datetime.datetime.now(datetime.timezone.utc)
        history = self._price_history.setdefault(market_id, [])
        history.append({"ts": ts, "yes": yes_price, "no": no_price})
        cutoff = ts - datetime.timedelta(seconds=HISTORY_RETENTION_SEC)
        self._price_history[market_id] = [p for p in history if p["ts"] > cutoff]

        ohlc = self._window_ohlc.setdefault(market_id, {
            "open": yes_price, "high": yes_price, "low": yes_price, "close": yes_price,
        })
        ohlc["high"] = max(ohlc["high"], yes_price)
        ohlc["low"] = min(ohlc["low"], yes_price)
        ohlc["close"] = yes_price

        if asset in self._asset_return_history:
            if self._asset_current_market.get(asset) != market_id:
                self._asset_return_history[asset] = []
                self._asset_current_market[asset] = market_id
            series = self._asset_return_history[asset]
            series.append({"ts": ts, "yes": yes_price})
            cutoff2 = ts - datetime.timedelta(seconds=CORRELATION_WINDOW_SEC)
            self._asset_return_history[asset] = [p for p in series if p["ts"] > cutoff2]

    def _price_before(self, market_id: str, seconds_ago: float,
                       now: datetime.datetime) -> Optional[dict]:
        history = self._price_history.get(market_id, [])
        target = now - datetime.timedelta(seconds=seconds_ago)
        candidates = [p for p in history if p["ts"] <= target]
        return candidates[-1] if candidates else None

    def _momentum(self, market_id: str, yes_price: float, seconds_ago: float,
                  now: datetime.datetime) -> Optional[float]:
        ref = self._price_before(market_id, seconds_ago, now)
        if ref is None:
            return None
        return yes_price - ref["yes"]

    def _ema(self, market_id: str, span: int) -> Optional[float]:
        """Standard EMA (alpha=2/(n+1)) over the last `span` polls of this
        market_id, seeded with the earliest of those prices as the initial
        value -- n falls back to however many ticks are actually available
        if the window hasn't run that long yet (never returns None once
        there's at least 1 tick, same as a live price would show)."""
        history = self._price_history.get(market_id, [])
        if not history:
            return None
        prices = [p["yes"] for p in history[-span:]]
        alpha = 2.0 / (len(prices) + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    @staticmethod
    def _imbalance_from_top(top: Optional[dict]) -> Optional[float]:
        if not top:
            return None
        bid_depth = top.get("total_bid_depth")
        ask_depth = top.get("total_ask_depth")
        if bid_depth is None or ask_depth is None:
            return None
        denom = bid_depth + ask_depth
        if denom == 0:
            return None
        return (bid_depth - ask_depth) / denom

    @staticmethod
    def _spread_from_top(top: Optional[dict]) -> Optional[float]:
        if not top:
            return None
        bid = top.get("best_bid_price")
        ask = top.get("best_ask_price")
        if bid is None or ask is None:
            return None
        return ask - bid

    @staticmethod
    def _depth_ratio_from_top(top: Optional[dict]) -> Optional[float]:
        """sum of the top-5 bid sizes / sum of the top-5 ask sizes -- unlike
        order_book_imbalance (which uses ALL levels' total depth and is
        already a normalized (bid-ask)/(bid+ask) difference), this is a
        raw ratio over just the nearest 5 levels on each side, a shallower
        and more execution-relevant slice of the book."""
        if not top:
            return None
        bid5 = top.get("bid_depth_top5")
        ask5 = top.get("ask_depth_top5")
        if bid5 is None or ask5 is None or ask5 == 0:
            return None
        return bid5 / ask5

    def btc_eth_correlation(self) -> Optional[float]:
        btc = self._asset_return_history.get("BTC", [])
        eth = self._asset_return_history.get("ETH", [])
        n = min(len(btc), len(eth))
        if n < CORRELATION_MIN_RETURNS + 1:
            return None
        btc_prices = [p["yes"] for p in btc[-n:]]
        eth_prices = [p["yes"] for p in eth[-n:]]
        btc_returns = [btc_prices[i] - btc_prices[i - 1] for i in range(1, n)]
        eth_returns = [eth_prices[i] - eth_prices[i - 1] for i in range(1, n)]
        if len(btc_returns) < CORRELATION_MIN_RETURNS:
            return None
        mean_b = sum(btc_returns) / len(btc_returns)
        mean_e = sum(eth_returns) / len(eth_returns)
        cov = sum((b - mean_b) * (e - mean_e) for b, e in zip(btc_returns, eth_returns))
        var_b = sum((b - mean_b) ** 2 for b in btc_returns)
        var_e = sum((e - mean_e) ** 2 for e in eth_returns)
        if var_b == 0 or var_e == 0:
            return None
        return cov / (var_b * var_e) ** 0.5

    def extract(self, market: dict, fetcher) -> dict:
        """fetcher only needs to expose get_order_book_top(token_id) -> dict|None."""
        yes_price = market["yes_price"]
        no_price = market["no_price"]
        market_id = market["market_id"]
        minutes_remaining = market.get("minutes_remaining", 5.0)
        up_token_id = market.get("up_token_id")
        top = fetcher.get_order_book_top(up_token_id) if up_token_id else None
        now = datetime.datetime.now(datetime.timezone.utc)

        ema_5 = self._ema(market_id, EMA_SHORT_SPAN)
        ema_20 = self._ema(market_id, EMA_LONG_SPAN)
        ohlc = self._window_ohlc.get(market_id, {})

        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "bid_ask_spread": self._spread_from_top(top),
            "order_book_imbalance": self._imbalance_from_top(top),
            "price_momentum_30s": self._momentum(market_id, yes_price, MOMENTUM_30S_LOOKBACK, now),
            "price_momentum_60s": self._momentum(market_id, yes_price, MOMENTUM_60S_LOOKBACK, now),
            "volume_24h": market.get("volume_24h", 0.0),
            "time_remaining_pct": minutes_remaining / 5.0,
            "btc_eth_correlation": self.btc_eth_correlation(),
            "ema_5": ema_5,
            "ema_20": ema_20,
            "price_vs_ema": (yes_price / ema_20 - 1) if ema_20 else None,
            "ohlc_open": ohlc.get("open"),
            "ohlc_high": ohlc.get("high"),
            "ohlc_low": ohlc.get("low"),
            "ohlc_close": ohlc.get("close"),
            "order_book_depth": self._depth_ratio_from_top(top),
        }
