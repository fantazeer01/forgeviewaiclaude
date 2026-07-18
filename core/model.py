"""Per (asset, timeframe) online model: River StandardScaler|LogisticRegression,
trained on every resolved window (not just real trades), persisted to disk,
and never auto-reset. decide() applies the entry filters/thresholds from
config/settings.py on top of the model's own P(UP)."""

import logging
import os
import pickle

from river import linear_model, preprocessing

from config.settings import (
    KELLY_MIN_EXAMPLES, MODEL_TRADE_THRESHOLD_YES, MODEL_TRADE_THRESHOLD_NO,
    ENTRY_YES_PRICE_MIN, ENTRY_YES_PRICE_MAX,
    MIN_SECONDS_REMAINING_5M, MIN_SECONDS_REMAINING_15M,
)
from core.feature_engine import BASE_FEATURE_NAMES, CROSS_MARKET_FEATURE_NAMES

logger = logging.getLogger(__name__)

MIN_SECONDS_REMAINING = {"5m": MIN_SECONDS_REMAINING_5M, "15m": MIN_SECONDS_REMAINING_15M}


class OnlineModel:
    def __init__(self, weights_file: str, asset: str, timeframe: str):
        self.weights_file = weights_file
        self.asset = asset
        self.timeframe = timeframe
        self.n_examples = 0
        self.model = self._load()

    def _feature_keys(self) -> list:
        keys = list(BASE_FEATURE_NAMES)
        if self.asset != "BTC":
            keys = keys + CROSS_MARKET_FEATURE_NAMES
        return keys

    def _fresh_model(self):
        return preprocessing.StandardScaler() | linear_model.LogisticRegression()

    def _load(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "rb") as f:
                    state = pickle.load(f)
                self.n_examples = state.get("n_examples", 0)
                logger.info(
                    f"OnlineModel[{self.asset}-{self.timeframe}]: loaded {self.n_examples} "
                    f"examples from {self.weights_file}"
                )
                return state["model"]
            except Exception as e:
                logger.warning(f"OnlineModel[{self.asset}-{self.timeframe}] load error, starting fresh: {e}")
        return self._fresh_model()

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
            tmp = self.weights_file + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"model": self.model, "n_examples": self.n_examples}, f)
            os.replace(tmp, self.weights_file)
        except Exception as e:
            logger.error(f"OnlineModel[{self.asset}-{self.timeframe}] save error: {e}")

    def _x(self, features: dict) -> dict:
        return {k: features.get(k, 0.0) for k in self._feature_keys()}

    def predict_proba(self, features: dict) -> float:
        try:
            return self.model.predict_proba_one(self._x(features)).get(True, 0.5)
        except Exception:
            return 0.5

    def learn(self, features: dict, outcome_up: bool):
        """Called on every resolved window, not just real trades -- and,
        by design, this is the ONLY place model state changes. There is no
        stability/health-check auto-reset anywhere: once trained, always
        trained, forever."""
        self.model.learn_one(self._x(features), outcome_up)
        self.n_examples += 1
        self.save()

    def decide(self, features: dict, seconds_remaining) -> dict:
        min_remaining = MIN_SECONDS_REMAINING.get(self.timeframe, 0)

        if self.n_examples < KELLY_MIN_EXAMPLES:
            return {
                "p_up": None, "decision": "HOLD",
                "reason": f"warmup {self.n_examples}/{KELLY_MIN_EXAMPLES}",
            }
        if seconds_remaining is None or seconds_remaining < min_remaining:
            return {"p_up": None, "decision": "HOLD", "reason": "too_late"}

        p_up = self.predict_proba(features)
        yes_price = features.get("yes_price", 0.5)
        in_band = ENTRY_YES_PRICE_MIN <= yes_price <= ENTRY_YES_PRICE_MAX

        if p_up > MODEL_TRADE_THRESHOLD_YES and in_band:
            return {"p_up": p_up, "decision": "YES", "reason": None}
        if p_up < MODEL_TRADE_THRESHOLD_NO and in_band:
            return {"p_up": p_up, "decision": "NO", "reason": None}
        return {"p_up": p_up, "decision": "HOLD", "reason": "uncertainty_or_price_band"}
