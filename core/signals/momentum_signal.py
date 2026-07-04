import datetime
from typing import Optional

from config.settings import MOMENTUM_WINDOW_SEC, MOMENTUM_MIN_SAMPLES
from core.repricing_detector import RepricingSignal


class MomentumSignalGenerator:
    """Tracks each market's YES price over a rolling MOMENTUM_WINDOW_SEC
    (45s) window and fires a YES signal on an early bounce: price was
    falling, then the most recent sample reverses upward. This is a
    3-point pattern (fall, then rise) meant to catch a reversal before it's
    obvious, distinct from RepricingDetector's own drop-only trigger.
    """

    def __init__(self):
        self._history: dict[str, list[dict]] = {}

    def update(self, market_id: str, yes_price: float):
        ts = datetime.datetime.now(datetime.timezone.utc)
        history = self._history.setdefault(market_id, [])
        history.append({"ts": ts, "yes": yes_price})
        cutoff = ts - datetime.timedelta(seconds=MOMENTUM_WINDOW_SEC)
        self._history[market_id] = [p for p in history if p["ts"] > cutoff]

    def generate(self, market: dict) -> Optional[RepricingSignal]:
        history = self._history.get(market["market_id"], [])
        if len(history) < MOMENTUM_MIN_SAMPLES:
            return None
        p0, p1, p2 = history[-3]["yes"], history[-2]["yes"], history[-1]["yes"]
        was_falling = p1 < p0
        now_rising = p2 > p1
        if not (was_falling and now_rising):
            return None
        drop = p0 - p1
        bounce = p2 - p1
        if drop <= 0:
            return None
        reversal_strength = bounce / drop
        confidence = min(0.95, 0.5 + reversal_strength * 0.4)
        return RepricingSignal(
            asset=market["asset"], market_id=market["market_id"], direction="YES",
            yes_price=market["yes_price"], no_price=market["no_price"],
            confidence=round(confidence, 3),
            reason=f"bounce reversal: dropped {drop:.3f} then rose {bounce:.3f}",
            minutes_remaining=market.get("minutes_remaining", 5.0),
        )
