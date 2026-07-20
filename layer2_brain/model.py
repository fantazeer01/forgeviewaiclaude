"""Layer 2 (brain): per (asset, timeframe) River online model. Only
predicts and learns -- confidence/liquidity/timing/regime gating all live in
layer3_conscience, composed on top of predict_proba() in bot.py.

Trained on every resolved window (shadow learning), not just real trades,
and never auto-reset: once trained, always trained. Same
data/models/model_{asset}_{tf}.pkl path as v3 on purpose, so the examples
already learned carry over across rebuilds."""

import logging
import os
import pickle

from river import linear_model, preprocessing

from config.settings import KELLY_MIN_EXAMPLES
from layer2_brain.feature_engine import BASE_FEATURE_NAMES, CROSS_MARKET_FEATURE_NAMES

logger = logging.getLogger(__name__)


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

    def is_warmed_up(self) -> bool:
        return self.n_examples >= KELLY_MIN_EXAMPLES

    def predict_proba(self, features: dict) -> float:
        try:
            return self.model.predict_proba_one(self._x(features)).get(True, 0.5)
        except Exception:
            return 0.5

    def learn(self, features: dict, outcome_up: bool):
        """Called on every resolved window, not just real trades. The ONLY
        place model state changes -- no stability/health-check auto-reset
        anywhere: once trained, always trained, forever."""
        self.model.learn_one(self._x(features), outcome_up)
        self.n_examples += 1
        self.save()

    def top_feature_names(self, n: int = 10) -> list:
        """The n feature names with the largest |learned weight| in the
        LogisticRegression step -- used for executor's features_snapshot.
        Falls back to the first n feature keys if weights aren't available
        yet (e.g. before any learn_one() call)."""
        try:
            classifier = list(self.model.steps.values())[-1]
            weights = classifier.weights
            ranked = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
            names = [k for k, _ in ranked if k in self._feature_keys()]
            if names:
                return names[:n]
        except Exception:
            pass
        return self._feature_keys()[:n]
