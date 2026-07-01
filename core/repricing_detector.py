import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional
from config.settings import REPRICING_FROZEN

logger = logging.getLogger(__name__)

@dataclass
class RepricingSignal:
    asset: str
    market_id: str
    direction: str
    yes_price: float
    no_price: float
    confidence: float
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    minutes_remaining: float = 5.0

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "market_id": self.market_id,
            "direction": self.direction,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "confidence": self.confidence,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "minutes_remaining": self.minutes_remaining,
        }

class RepricingDetector:
    def __init__(self):
        self.params = REPRICING_FROZEN
        self._price_history: dict[str, list[dict]] = {}

    def update_prices(self, market_id: str, yes_price: float, no_price: float):
        ts = datetime.datetime.utcnow()
        if market_id not in self._price_history:
            self._price_history[market_id] = []
        self._price_history[market_id].append({"ts": ts, "yes": yes_price, "no": no_price})
        cutoff = ts - datetime.timedelta(seconds=self.params["max_time_window_sec"])
        self._price_history[market_id] = [
            p for p in self._price_history[market_id] if p["ts"] > cutoff
        ]

    def detect(self, market: dict) -> Optional[RepricingSignal]:
        market_id = market["market_id"]
        yes_price = market["yes_price"]
        no_price = market["no_price"]
        minutes_remaining = market.get("minutes_remaining", 5.0)
        if minutes_remaining < 1.0 or minutes_remaining > 4.5:
            return None
        history = self._price_history.get(market_id, [])
        if len(history) < 2:
            return None
        now = datetime.datetime.utcnow()
        min_cutoff = now - datetime.timedelta(seconds=self.params["min_time_window_sec"])
        old_obs = next((p for p in history if p["ts"] <= min_cutoff), None)
        if old_obs is None:
            return None
        yes_drop = old_obs["yes"] - yes_price
        no_drop = old_obs["no"] - no_price
        threshold = self.params["min_price_move"]
        conf_threshold = self.params["confidence_threshold"]
        if yes_drop >= threshold:
            confidence = min(0.95, conf_threshold + (yes_drop - threshold) * 2)
            if confidence >= conf_threshold:
                return RepricingSignal(
                    asset=market["asset"], market_id=market_id, direction="YES",
                    yes_price=yes_price, no_price=no_price,
                    confidence=round(confidence, 3),
                    reason=f"YES dropped {yes_drop:.3f} in {self.params['min_time_window_sec']}s",
                    minutes_remaining=minutes_remaining,
                )
        elif no_drop >= threshold:
            confidence = min(0.95, conf_threshold + (no_drop - threshold) * 2)
            if confidence >= conf_threshold:
                return RepricingSignal(
                    asset=market["asset"], market_id=market_id, direction="NO",
                    yes_price=yes_price, no_price=no_price,
                    confidence=round(confidence, 3),
                    reason=f"NO dropped {no_drop:.3f} in {self.params['min_time_window_sec']}s",
                    minutes_remaining=minutes_remaining,
                )
        return None

    def reset_market(self, market_id: str):
        self._price_history.pop(market_id, None)
