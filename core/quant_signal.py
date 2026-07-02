import logging
import datetime
from typing import Optional

from config.settings import QUANT_MODEL_PATH
from core.repricing_detector import RepricingDetector, RepricingSignal
from core.state_manager import StateManager
from core.quant_features import QuantFeatureExtractor
from core.quant_dataset import QuantDataset
from core.quant_model import QuantModel
from signals.repricing_signal import RepricingSignalGenerator

logger = logging.getLogger(__name__)


class QuantSignalGenerator:
    """Drop-in replacement for RepricingSignalGenerator: trading decisions are
    identical (delegated straight to the existing repricing logic), but every
    fired signal also gets a quantitative feature snapshot logged, and the
    real outcome is logged once that market resolves -- regardless of whether
    a trade was actually opened for it (e.g. blocked by max open positions),
    since even an untaken signal is a labeled training example.

    SHADOW MODE ONLY: if a trained QuantModel is available
    (data/quant_model.pkl), its predicted win probability is logged alongside
    each signal as model_probability. It has no effect on which signal is
    returned or which trade gets opened -- the repricing detector is the only
    thing deciding trades.

    This is a deliberate revert. A prior version of this class briefly used
    model.predict_proba() to drive trading decisions directly. A from-scratch
    reproduction of forgeview-ai's own logistic-regression experiment on this
    project's historical data (data/historical/README.md) did not beat the
    naive "trust the market's own YES price" baseline (log loss 0.598 vs.
    0.591, Brier 0.207 vs. 0.205 on held-out data), so that model was not a
    demonstrated improvement over the repricing rule it had replaced. The
    model is logged here for ongoing evaluation against live data, not
    trusted with real decisions.
    """

    def __init__(self, detector: RepricingDetector, state: StateManager, fetcher,
                 features: Optional[QuantFeatureExtractor] = None,
                 dataset: Optional[QuantDataset] = None,
                 model: Optional[QuantModel] = None):
        self._repricing = RepricingSignalGenerator(detector, state)
        self.fetcher = fetcher
        self.features = features or QuantFeatureExtractor()
        self.dataset = dataset or QuantDataset()
        self.model = model if model is not None else QuantModel.load(QUANT_MODEL_PATH)
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
                model_probability=entry["model_probability"],
            )

    def _log_signal(self, market: dict, signal: RepricingSignal):
        snapshot = self.features.extract(market, self.fetcher)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry_price = signal.yes_price if signal.direction == "YES" else signal.no_price
        model_probability = self.model.predict_proba_one(snapshot) if self.model else None
        self.dataset.log_signal(
            sample_id=market["market_id"],
            market_id=market["market_id"],
            asset=signal.asset,
            features=snapshot,
            entry_price=entry_price,
            direction=signal.direction,
            timestamp=timestamp,
            model_probability=model_probability,
        )
        self._pending[market["market_id"]] = {
            "asset": signal.asset,
            "features": snapshot,
            "entry_price": entry_price,
            "direction": signal.direction,
            "model_probability": model_probability,
        }
