import datetime
import json
import logging
import os
from typing import Optional

from config.settings import (
    VOLUME_HISTORY_LOG, VOLUME_RATIO_THRESHOLD, VOLUME_LOOKBACK_DAYS,
    VOLUME_RECORD_INTERVAL_SEC, VOLUME_SKIP_MINUTES_REMAINING_THRESHOLD,
)
from core.repricing_detector import RepricingSignal

logger = logging.getLogger(__name__)


class VolumeSignalGenerator:
    """Fires a YES signal when a market's current 24h volume is unusually
    high relative to its own trailing 7-day average -- the idea being that
    a volume spike suggests more informed/active participation, which
    should make whatever direction the price is already leaning firmer.

    Volume history is appended to data/volume_history.jsonl, throttled to
    at most once per VOLUME_RECORD_INTERVAL_SEC (1 hour) per asset --
    volume_24h itself only changes gradually, so recording it every 3s poll
    tick would just bloat the log with near-duplicate rows for no benefit.

    Skips firing entirely in the first 60s of a fresh 5-min window
    (minutes_remaining > VOLUME_SKIP_MINUTES_REMAINING_THRESHOLD) -- prices
    and the order book are less settled right after a market opens, so a
    volume-based confirmation there is less reliable (2026-07-06 signal
    quality pass).
    """

    def __init__(self, history_path: str = VOLUME_HISTORY_LOG):
        self.history_path = history_path
        self._last_recorded_ts: dict[str, datetime.datetime] = {}

    def record_volume(self, asset: str, volume_24h: float):
        now = datetime.datetime.now(datetime.timezone.utc)
        last = self._last_recorded_ts.get(asset)
        if last and (now - last).total_seconds() < VOLUME_RECORD_INTERVAL_SEC:
            return
        self._last_recorded_ts[asset] = now
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        entry = {"asset": asset, "volume_24h": volume_24h, "timestamp": now.isoformat()}
        try:
            with open(self.history_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"VolumeSignalGenerator record error: {e}")

    def _seven_day_average(self, asset: str) -> Optional[float]:
        if not os.path.exists(self.history_path):
            return None
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=VOLUME_LOOKBACK_DAYS)
        values = []
        try:
            with open(self.history_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("asset") != asset:
                        continue
                    try:
                        ts = datetime.datetime.fromisoformat(row["timestamp"])
                    except (KeyError, ValueError):
                        continue
                    if ts > cutoff:
                        values.append(row["volume_24h"])
        except Exception as e:
            logger.error(f"VolumeSignalGenerator average read error: {e}")
            return None
        if not values:
            return None
        return sum(values) / len(values)

    def generate(self, market: dict) -> Optional[RepricingSignal]:
        minutes_remaining = market.get("minutes_remaining", 5.0)
        if minutes_remaining > VOLUME_SKIP_MINUTES_REMAINING_THRESHOLD:
            return None
        current = market.get("volume_24h")
        if current is None:
            return None
        avg = self._seven_day_average(market["asset"])
        if avg is None or avg <= 0:
            return None
        ratio = current / avg
        if ratio <= VOLUME_RATIO_THRESHOLD:
            return None
        confidence = min(0.95, 0.5 + (ratio - VOLUME_RATIO_THRESHOLD) * 0.3)
        return RepricingSignal(
            asset=market["asset"], market_id=market["market_id"], direction="YES",
            yes_price=market["yes_price"], no_price=market["no_price"],
            confidence=round(confidence, 3),
            reason=f"volume {ratio:.2f}x 7-day avg ({current:.0f} vs {avg:.0f})",
            minutes_remaining=market.get("minutes_remaining", 5.0),
        )
