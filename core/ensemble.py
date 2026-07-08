"""Combines momentum, volume, and macro signals into a single trade
decision per 5-min market window."""

import logging

from config.settings import (
    ENSEMBLE_WEIGHTS, ENSEMBLE_YES_SCORE_THRESHOLD, ENSEMBLE_YES_PRICE_BAND,
    ENSEMBLE_NO_SCORE_THRESHOLD, ENSEMBLE_NO_PRICE_BAND,
    ENSEMBLE_MIN_TRAINING_EXAMPLES,
)
from models.macro_model import macro_bias

logger = logging.getLogger(__name__)


class Ensemble:
    def __init__(self, momentum_model, volume_model):
        self.momentum_model = momentum_model
        self.volume_model = volume_model

    def _training_examples(self) -> int:
        return min(self.momentum_model.n_examples, self.volume_model.n_examples)

    def score(self, features: dict, fear_greed: int, hour_utc: int) -> dict:
        momentum_p = self.momentum_model.predict_up(features)
        volume_p = self.volume_model.predict_up(features)
        macro_p = 0.5 + macro_bias(fear_greed, hour_utc)
        final_score = (
            ENSEMBLE_WEIGHTS["momentum"] * momentum_p
            + ENSEMBLE_WEIGHTS["volume"] * volume_p
            + ENSEMBLE_WEIGHTS["macro"] * macro_p
        )
        return {
            "momentum_p": momentum_p,
            "volume_p": volume_p,
            "macro_p": macro_p,
            "final_score": final_score,
        }

    def decide(self, features: dict, fear_greed: int, hour_utc: int) -> dict:
        result = self.score(features, fear_greed, hour_utc)
        final_score = result["final_score"]
        yes_price = features.get("yes_price")
        decision = None

        if self._training_examples() < ENSEMBLE_MIN_TRAINING_EXAMPLES:
            result["decision"] = None
            result["reason"] = f"warmup: {self._training_examples()}/{ENSEMBLE_MIN_TRAINING_EXAMPLES} examples"
            return result

        yes_lo, yes_hi = ENSEMBLE_YES_PRICE_BAND
        no_lo, no_hi = ENSEMBLE_NO_PRICE_BAND

        if final_score > ENSEMBLE_YES_SCORE_THRESHOLD and yes_price is not None and yes_lo <= yes_price <= yes_hi:
            decision = "YES"
        elif final_score < ENSEMBLE_NO_SCORE_THRESHOLD and yes_price is not None and no_lo <= yes_price <= no_hi:
            decision = "NO"

        result["decision"] = decision
        result["reason"] = None if decision else "uncertainty zone or price out of band"
        return result
