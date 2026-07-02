import logging
import datetime
from typing import Optional

from config.settings import QUANT_MODEL_PATH, SIGNAL_COOLDOWN_SEC
from core.repricing_detector import RepricingSignal
from core.state_manager import StateManager
from core.quant_features import QuantFeatureExtractor
from core.quant_dataset import QuantDataset
from core.quant_model import QuantModel

logger = logging.getLogger(__name__)

MODEL_CONFIDENCE_THRESHOLD = 0.55
MIN_MINUTES_REMAINING = 1.0
MAX_MINUTES_REMAINING = 4.5


class QuantSignalGenerator:
    """Model-driven signal generator: fires a YES signal when the trained
    QuantModel's predict_proba() crosses confidence_threshold. This REPLACES
    the previous repricing-rule (price-drop threshold) decision logic, per
    explicit instruction.

    CAVEAT this project's own evidence raised: a from-scratch reproduction of
    forgeview-ai's own logistic-regression experiment on this project's copy
    of its historical data (see data/historical/README.md) did NOT beat the
    naive "trust the market's own YES price" baseline -- log loss 0.598 vs.
    0.591, Brier 0.207 vs. 0.205 on a held-out split -- matching what that
    source repo's own evidence-gated research found repeatedly
    (NO_EDGE_FOUND_YET; decisions D-031/D-033 for the microstructure-augmented
    version). This generator is wired live despite that finding because it
    was explicitly requested three times; it is not a demonstrated
    improvement over the repricing rule it replaces, which had an established
    positive live paper-trading track record. Compare live results against
    the pre-swap baseline in docs/polymarket/PAPER_TRADING_REPORT.md.

    If no trained model is available (data/quant_model.pkl missing),
    process_market() always returns None -- there is no rule-based fallback.
    """

    def __init__(self, state: StateManager, fetcher,
                 features: Optional[QuantFeatureExtractor] = None,
                 dataset: Optional[QuantDataset] = None,
                 model: Optional[QuantModel] = None,
                 confidence_threshold: float = MODEL_CONFIDENCE_THRESHOLD):
        self.state = state
        self.fetcher = fetcher
        self.features = features or QuantFeatureExtractor()
        self.dataset = dataset or QuantDataset()
        self.model = model if model is not None else QuantModel.load(QUANT_MODEL_PATH)
        self.confidence_threshold = confidence_threshold
        self._pending: dict[str, dict] = {}

    def process_market(self, market: dict) -> Optional[RepricingSignal]:
        market_id = market["market_id"]
        self.features.update(market_id, market["yes_price"], market["no_price"])

        if self.model is None:
            return None
        minutes_remaining = market.get("minutes_remaining", 5.0)
        if minutes_remaining < MIN_MINUTES_REMAINING or minutes_remaining > MAX_MINUTES_REMAINING:
            return None
        if self._is_in_cooldown(market["asset"]):
            return None

        snapshot = self.features.extract(market, self.fetcher)
        model_probability = self.model.predict_proba_one(snapshot)
        if model_probability is None or model_probability < self.confidence_threshold:
            return None

        signal = RepricingSignal(
            asset=market["asset"],
            market_id=market_id,
            direction="YES",
            yes_price=market["yes_price"],
            no_price=market["no_price"],
            confidence=round(model_probability, 3),
            reason=f"model predict_proba={model_probability:.3f}",
            minutes_remaining=minutes_remaining,
        )
        self._set_cooldown(market["asset"])
        self._log_signal(market, signal, snapshot, model_probability)
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

    def _log_signal(self, market: dict, signal: RepricingSignal, snapshot: dict, model_probability: float):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.dataset.log_signal(
            sample_id=market["market_id"],
            market_id=market["market_id"],
            asset=signal.asset,
            features=snapshot,
            entry_price=signal.yes_price,
            direction=signal.direction,
            timestamp=timestamp,
            model_probability=model_probability,
        )
        self._pending[market["market_id"]] = {
            "asset": signal.asset,
            "features": snapshot,
            "entry_price": signal.yes_price,
            "direction": signal.direction,
            "model_probability": model_probability,
        }
