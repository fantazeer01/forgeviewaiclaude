import logging
import datetime
from typing import Optional
from core.repricing_detector import RepricingDetector, RepricingSignal
from core.state_manager import StateManager
from config.settings import SIGNAL_COOLDOWN_SEC

logger = logging.getLogger(__name__)

class RepricingSignalGenerator:
    def __init__(self, detector: RepricingDetector, state: StateManager):
        self.detector = detector
        self.state = state

    def process_market(self, market: dict) -> Optional[RepricingSignal]:
        self.detector.update_prices(market["market_id"], market["yes_price"], market["no_price"])
        if self._is_in_cooldown(market["asset"]):
            return None
        signal = self.detector.detect(market)
        if signal:
            self._set_cooldown(market["asset"])
        return signal

    def _is_in_cooldown(self, asset: str) -> bool:
        last_ts_map = self.state.get("last_signal_ts", {}) or {}
        last_ts_str = last_ts_map.get(asset)
        if not last_ts_str:
            return False
        try:
            last_ts = datetime.datetime.fromisoformat(last_ts_str)
            return (datetime.datetime.utcnow() - last_ts).total_seconds() < SIGNAL_COOLDOWN_SEC
        except Exception:
            return False

    def _set_cooldown(self, asset: str):
        last_ts_map = self.state.get("last_signal_ts", {}) or {}
        last_ts_map[asset] = datetime.datetime.utcnow().isoformat()
        self.state.set("last_signal_ts", last_ts_map)
