import logging
import datetime
from typing import Optional

from core.repricing_detector import RepricingDetector, RepricingSignal
from core.state_manager import StateManager
from core.quant_features import QuantFeatureExtractor
from core.quant_dataset import QuantDataset
from signals.repricing_signal import RepricingSignalGenerator

logger = logging.getLogger(__name__)


class QuantSignalGenerator:
    """Drop-in replacement for RepricingSignalGenerator: trading decisions are
    identical (delegated straight to the existing repricing logic), but every
    fired signal also gets a quantitative feature snapshot logged, and the
    real outcome is logged once that market resolves -- regardless of whether
    a trade was actually opened for it (e.g. blocked by max open positions),
    since even an untaken signal is a labeled training example.

    This is a data-collection placeholder for a future model-driven signal
    generator; it changes no trading behavior on its own.
    """

    def __init__(self, detector: RepricingDetector, state: StateManager, fetcher,
                 features: Optional[QuantFeatureExtractor] = None,
                 dataset: Optional[QuantDataset] = None):
        self._repricing = RepricingSignalGenerator(detector, state)
        self.fetcher = fetcher
        self.features = features or QuantFeatureExtractor()
        self.dataset = dataset or QuantDataset()
        self._pending: dict[str, dict] = {}

    def process_market(self, market: dict) -> Optional[RepricingSignal]:
        self.features.update(market["market_id"], market["yes_price"], market["no_price"])
        signal = self._repricing.process_market(market)
        if signal:
            self._log_signal(market, signal)
        return signal

    def resolve_pending(self):
        for market_id in list(self._pending.keys()):
            resolution = self.fetcher.get_market_resolution(market_id)
            if not resolution:
                continue
            outcome_dir = self.fetcher.resolve_outcome(resolution)
            if outcome_dir is None:
                continue
            entry = self._pending.pop(market_id)
            outcome = 1 if outcome_dir == entry["direction"] else 0
            self.dataset.log_resolution(
                sample_id=market_id,
                market_id=market_id,
                asset=entry["asset"],
                features=entry["features"],
                entry_price=entry["entry_price"],
                direction=entry["direction"],
                outcome=outcome,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )

    def _log_signal(self, market: dict, signal: RepricingSignal):
        snapshot = self.features.extract(market, self.fetcher)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry_price = signal.yes_price if signal.direction == "YES" else signal.no_price
        self.dataset.log_signal(
            sample_id=market["market_id"],
            market_id=market["market_id"],
            asset=signal.asset,
            features=snapshot,
            entry_price=entry_price,
            direction=signal.direction,
            timestamp=timestamp,
        )
        self._pending[market["market_id"]] = {
            "asset": signal.asset,
            "features": snapshot,
            "entry_price": entry_price,
            "direction": signal.direction,
        }
