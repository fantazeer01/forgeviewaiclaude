import logging
import datetime
from typing import Callable, Optional
from core.repricing_detector import RepricingDetector, RepricingSignal
from core.state_manager import StateManager
from config.settings import SIGNAL_COOLDOWN_SEC

logger = logging.getLogger(__name__)

class RepricingSignalGenerator:
    def __init__(self, detector: RepricingDetector, state: StateManager,
                 market_bias_provider: Optional[Callable[[], Optional[str]]] = None):
        self.detector = detector
        self.state = state
        # a zero-arg callable so this always reads the LATEST bias at
        # decision time rather than a value that could go stale between
        # construction and the signal actually firing. None (the default)
        # means "no real bias data available" -- and per the no-fake-data
        # rule, that must never be treated as BEARISH; it just disables the
        # filter, exactly like the pre-market-bias behavior.
        self.market_bias_provider = market_bias_provider

    def process_market(self, market: dict) -> Optional[RepricingSignal]:
        self.detector.update_prices(market["market_id"], market["yes_price"], market["no_price"])
        if self._is_in_cooldown(market["asset"]):
            return None
        signal = self.detector.detect(market)
        if signal is None:
            return None
        self._set_cooldown(market["asset"])
        if self._blocked_by_market_bias(signal):
            logger.info(f"Signal blocked by BEARISH market bias: {signal.asset} {signal.direction} reason={signal.reason}")
            return None
        return signal

    def _blocked_by_market_bias(self, signal: RepricingSignal) -> bool:
        """BEARISH blocks YES signals (a falling market makes a YES/Up
        contract less likely to resolve favorably); BULLISH and NEUTRAL both
        allow YES signals through unchanged, matching the spec exactly."""
        if self.market_bias_provider is None:
            return False
        bias = self.market_bias_provider()
        return bias == "BEARISH" and signal.direction == "YES"

    def _is_in_cooldown(self, asset: str) -> bool:
        last_ts_map = self.state.get("last_signal_ts", {}) or {}
        last_ts_str = last_ts_map.get(asset)
        if not last_ts_str:
            return False
        try:
            last_ts = datetime.datetime.fromisoformat(last_ts_str)
            return (datetime.datetime.now(datetime.timezone.utc) - last_ts).total_seconds() < SIGNAL_COOLDOWN_SEC
        except Exception:
            return False

    def _set_cooldown(self, asset: str):
        last_ts_map = self.state.get("last_signal_ts", {}) or {}
        last_ts_map[asset] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.state.set("last_signal_ts", last_ts_map)
