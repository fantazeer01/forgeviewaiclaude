"""Online-learning volume signal: River LogisticRegression over volume_ratio
and bid_ask_imbalance, trained on every resolved outcome, never reset.
Weights persist to disk so learning survives process restarts."""

import logging
import os
import pickle

from river import linear_model, preprocessing

from config.settings import VOLUME_WEIGHTS_FILE

logger = logging.getLogger(__name__)

FEATURE_KEYS = ["volume_ratio", "bid_ask_imbalance"]


class VolumeModel:
    def __init__(self, weights_file: str = VOLUME_WEIGHTS_FILE):
        self.weights_file = weights_file
        self.n_examples = 0
        self.model = self._load()

    def _fresh_model(self):
        return preprocessing.StandardScaler() | linear_model.LogisticRegression()

    def _load(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "rb") as f:
                    state = pickle.load(f)
                self.n_examples = state.get("n_examples", 0)
                return state["model"]
            except Exception as e:
                logger.warning(f"VolumeModel load error, starting fresh: {e}")
        return self._fresh_model()

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
            tmp = self.weights_file + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"model": self.model, "n_examples": self.n_examples}, f)
            os.replace(tmp, self.weights_file)
        except Exception as e:
            logger.error(f"VolumeModel save error: {e}")

    def _x(self, features: dict) -> dict:
        return {k: features.get(k, 0.0) for k in FEATURE_KEYS}

    def predict_up(self, features: dict) -> float:
        try:
            return self.model.predict_proba_one(self._x(features)).get(True, 0.5)
        except Exception:
            return 0.5

    def learn(self, features: dict, outcome_up: bool):
        self.model.learn_one(self._x(features), outcome_up)
        self.n_examples += 1
        self.save()
