import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

PRICE_HISTORY_WINDOW_SEC = 120
VELOCITY_LOOKBACK_SEC = 60
ACCEL_LOOKBACK_SEC = 30


class QuantFeatureExtractor:
    """Rolling per-market price history plus a stateless feature-extraction pass
    over a market snapshot and the live order book. Kept independent of
    RepricingDetector's own price history so feature engineering can evolve
    without touching the trading detector's behavior.

    price_velocity: average rate of yes_price change per second over the last
    VELOCITY_LOOKBACK_SEC (60s) seconds, i.e. (price_now - price_60s_ago) / 60.

    price_acceleration: change in that velocity over the last ACCEL_LOOKBACK_SEC
    (30s) seconds, i.e. velocity_now - velocity_as_of_30s_ago (velocity_as_of_30s_ago
    is itself the same 60s-lookback velocity, evaluated at t-30 using price at t-90).
    Requires ~90s of history; returns None while warming up.
    """

    def __init__(self):
        self._price_history: dict[str, list[dict]] = {}

    def update(self, market_id: str, yes_price: float, no_price: float):
        ts = datetime.datetime.now(datetime.timezone.utc)
        history = self._price_history.setdefault(market_id, [])
        history.append({"ts": ts, "yes": yes_price, "no": no_price})
        cutoff = ts - datetime.timedelta(seconds=PRICE_HISTORY_WINDOW_SEC)
        self._price_history[market_id] = [p for p in history if p["ts"] > cutoff]

    def _price_before(self, market_id: str, seconds_ago: float,
                       now: datetime.datetime) -> Optional[dict]:
        history = self._price_history.get(market_id, [])
        target = now - datetime.timedelta(seconds=seconds_ago)
        candidates = [p for p in history if p["ts"] <= target]
        return candidates[-1] if candidates else None

    def price_velocity(self, market_id: str, yes_price: float,
                        now: Optional[datetime.datetime] = None) -> Optional[float]:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        ref = self._price_before(market_id, VELOCITY_LOOKBACK_SEC, now)
        if ref is None:
            return None
        elapsed = (now - ref["ts"]).total_seconds()
        if elapsed <= 0:
            return None
        return (yes_price - ref["yes"]) / elapsed

    def price_acceleration(self, market_id: str, yes_price: float,
                            now: Optional[datetime.datetime] = None) -> Optional[float]:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        velocity_now = self.price_velocity(market_id, yes_price, now)
        if velocity_now is None:
            return None
        ref_30 = self._price_before(market_id, ACCEL_LOOKBACK_SEC, now)
        ref_90 = self._price_before(market_id, ACCEL_LOOKBACK_SEC + VELOCITY_LOOKBACK_SEC, now)
        if ref_30 is None or ref_90 is None:
            return None
        elapsed = (ref_30["ts"] - ref_90["ts"]).total_seconds()
        if elapsed <= 0:
            return None
        velocity_30s_ago = (ref_30["yes"] - ref_90["yes"]) / elapsed
        return velocity_now - velocity_30s_ago

    @staticmethod
    def _imbalance_from_top(top: Optional[dict]) -> Optional[float]:
        if not top:
            return None
        bid_size = top.get("best_bid_size")
        ask_size = top.get("best_ask_size")
        if bid_size is None or ask_size is None:
            return None
        denom = bid_size + ask_size
        if denom == 0:
            return None
        return (bid_size - ask_size) / denom

    @staticmethod
    def _spread_from_top(top: Optional[dict]) -> Optional[float]:
        if not top:
            return None
        bid = top.get("best_bid_price")
        ask = top.get("best_ask_price")
        if bid is None or ask is None:
            return None
        return ask - bid

    def extract(self, market: dict, fetcher) -> dict:
        """fetcher only needs to expose get_order_book_top(token_id) -> dict|None."""
        yes_price = market["yes_price"]
        no_price = market["no_price"]
        market_id = market["market_id"]
        minutes_remaining = market.get("minutes_remaining", 5.0)
        up_token_id = market.get("up_token_id")
        top = fetcher.get_order_book_top(up_token_id) if up_token_id else None

        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "price_velocity": self.price_velocity(market_id, yes_price),
            "price_acceleration": self.price_acceleration(market_id, yes_price),
            "order_book_imbalance": self._imbalance_from_top(top),
            "volume_24h": market.get("volume_24h", 0.0),
            "time_remaining_pct": minutes_remaining / 5.0,
            "spread": self._spread_from_top(top),
        }
