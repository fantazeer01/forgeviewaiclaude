import datetime
import json
import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.linear_model import SGDClassifier

from config.settings import (
    KELLY_FRACTION_CAP, ONLINE_MODEL_STATE_FILE, ONLINE_MODEL_WARMUP_TRADES,
    ONLINE_MODEL_CONFIDENCE_THRESHOLD, ONLINE_MODEL_MAX_TRADE_USD,
    ONLINE_MODEL_STATUS_FILE,
)
from core.kelly_criterion import net_odds_from_price, kelly_fraction
from core.live_features import FEATURE_NAMES

logger = logging.getLogger(__name__)

Z_SCORE_CLIP = 8.0


class OnlineQuantModel:
    """Online-learning trading model: an SGDClassifier that updates itself
    with partial_fit() after every resolved trade, instead of being trained
    once on a static historical batch.

    This is a deliberately different approach from the two prior batch-model
    sprints (docs/polymarket/DECISIONS.md D-001, data/historical/README.md):
    those trained on forgeview-ai's historical research data and found every
    model scored at or below chance on genuinely unseen live data, because
    historical and live data come from different market regimes a static
    model can't adapt to. An online model only ever trains on this project's
    own live outcomes, so there is no cross-regime mismatch -- but it starts
    from zero evidence, so it needs a warm-up period before its predictions
    can be trusted with real trading decisions.

    For the first ONLINE_MODEL_WARMUP_TRADES (200) resolved trades, decide()
    always defers to the repricing signal passed in -- trading behavior is
    unchanged during warm-up, but every one of those resolutions is still fed
    into the model as a training example. After warm-up, decide() ignores the
    repricing signal's fire/no-fire decision and instead fires whenever the
    model's own predicted win probability crosses the confidence threshold,
    sized via Kelly criterion and hard-capped at ONLINE_MODEL_MAX_TRADE_USD
    ($10) regardless of predicted confidence -- a raised warm-up floor alone
    doesn't guarantee a well-calibrated model (a from-scratch SGDClassifier
    can still saturate to p=1.0 on limited data and demand a huge Kelly size;
    the hard dollar cap is what actually bounds the damage from that).
    """

    def __init__(self, feature_names: list[str] = FEATURE_NAMES,
                 state_path: str = ONLINE_MODEL_STATE_FILE,
                 warmup_trades: int = ONLINE_MODEL_WARMUP_TRADES):
        self.feature_names = list(feature_names)
        self.state_path = state_path
        self.warmup_trades = warmup_trades
        self.clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                                  learning_rate="optimal", random_state=20260704)
        self._fitted = False
        self._n_updates = 0
        n = len(self.feature_names)
        self._mean = np.zeros(n)
        self._m2 = np.zeros(n)
        self._count_vec = np.zeros(n)
        self._pending: dict[str, dict] = {}
        self._load()

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def is_warmed_up(self) -> bool:
        return self._n_updates >= self.warmup_trades

    def _vectorize(self, features: dict) -> np.ndarray:
        return np.array([
            features.get(name) if features.get(name) is not None else np.nan
            for name in self.feature_names
        ], dtype=float)

    def _update_standardizer(self, x: np.ndarray):
        """Welford's online per-feature mean/variance update; NaN entries
        (a feature that wasn't available for this sample) are skipped rather
        than treated as zero, so they don't distort the running estimate."""
        for i, v in enumerate(x):
            if v != v:
                continue
            self._count_vec[i] += 1
            delta = v - self._mean[i]
            self._mean[i] += delta / self._count_vec[i]
            self._m2[i] += delta * (v - self._mean[i])

    def _std(self) -> np.ndarray:
        var = np.where(self._count_vec > 1, self._m2 / np.maximum(self._count_vec - 1, 1), 1.0)
        return np.sqrt(np.where(var > 1e-12, var, 1.0))

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        z = np.where(np.isnan(x), 0.0, (x - self._mean) / self._std())
        return np.clip(z, -Z_SCORE_CLIP, Z_SCORE_CLIP)

    def update(self, features: dict, outcome: int):
        """One resolved-trade training example: outcome=1 for a win, 0 for a
        loss. Called for every resolution, whether or not decide() itself
        was the one that opened that trade -- warm-up trades are training
        data too."""
        x = self._vectorize(features)
        self._update_standardizer(x)
        x_std = self._standardize(x).reshape(1, -1)
        if not self._fitted:
            self.clf.partial_fit(x_std, [outcome], classes=np.array([0, 1]))
            self._fitted = True
        else:
            self.clf.partial_fit(x_std, [outcome])
        self._n_updates += 1
        self.save()

    def predict_proba_one(self, features: dict) -> Optional[float]:
        if not self._fitted:
            return None
        x = self._vectorize(features)
        x_std = self._standardize(x).reshape(1, -1)
        return float(self.clf.predict_proba(x_std)[0][1])

    def decide(self, features: dict, repricing_signal,
               confidence_threshold: float = ONLINE_MODEL_CONFIDENCE_THRESHOLD):
        """Returns (should_trade, direction, win_probability, reason).

        Before warm-up: mirrors the repricing signal exactly (fires iff a
        repricing signal fired, using its own direction/confidence).
        After warm-up: fires iff the model's predicted YES win probability
        clears confidence_threshold, independent of the repricing signal.
        """
        if not self.is_warmed_up():
            if repricing_signal is None:
                return False, None, None, "warmup: no repricing signal"
            return True, repricing_signal.direction, repricing_signal.confidence, "warmup: repricing rule"

        p = self.predict_proba_one(features)
        if p is None:
            return False, None, None, "model not fitted"
        if p >= confidence_threshold:
            return True, "YES", p, f"online model p={p:.3f}"
        return False, None, p, f"online model p={p:.3f} below threshold"

    def kelly_size(self, win_probability: float, entry_price: float, bankroll: float) -> float:
        """f = (p*b - (1-p)) / b, capped at KELLY_FRACTION_CAP (0.25) -- the
        literal formula requested, not the more conservative quarter-Kelly
        (f * 0.25) variant core/kelly_criterion.py also offers. The resulting
        dollar size is then hard-capped at ONLINE_MODEL_MAX_TRADE_USD ($10)
        regardless of win_probability or the fractional-Kelly result -- a
        saturated (near-0 or near-1) probability from an undertrained model
        would otherwise still demand a maximal-fraction position."""
        net_odds = net_odds_from_price(entry_price)
        f = min(kelly_fraction(win_probability, net_odds), KELLY_FRACTION_CAP)
        size = f * bankroll
        return round(min(size, ONLINE_MODEL_MAX_TRADE_USD), 4)

    def record_features(self, market_id: str, features: dict):
        """Stash the feature snapshot a decision was made from, keyed by
        market_id, so resolve() can pair it with the eventual outcome."""
        self._pending[market_id] = features

    def resolve(self, market_id: str, outcome: int) -> bool:
        """Feed the resolved outcome (1=win, 0=loss) for a previously
        record_features()'d market back into the model. Returns False if no
        matching pending record exists (nothing to learn from)."""
        features = self._pending.pop(market_id, None)
        if features is None:
            return False
        self.update(features, outcome)
        return True

    def save(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp_path = self.state_path + ".tmp"
        state = {
            "feature_names": self.feature_names,
            "warmup_trades": self.warmup_trades,
            "clf": self.clf,
            "fitted": self._fitted,
            "n_updates": self._n_updates,
            "mean": self._mean,
            "m2": self._m2,
            "count_vec": self._count_vec,
            "pending": self._pending,
        }
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(state, f)
            os.replace(tmp_path, self.state_path)
        except Exception as e:
            logger.error(f"OnlineQuantModel save error: {e}")
        self._export_status()

    def _export_status(self):
        """Writes a small JSON mirror of the warmup/fit status a browser can
        actually read -- the .pkl file above is a Python pickle and can't be
        parsed client-side. Called after every save() and on load() so the
        dashboard always reflects the model's real last-known state, not a
        guess."""
        status = {
            "n_updates": self._n_updates,
            "warmup_trades": self.warmup_trades,
            "fitted": self._fitted,
            "is_warmed_up": self.is_warmed_up(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            os.makedirs(os.path.dirname(ONLINE_MODEL_STATUS_FILE), exist_ok=True)
            tmp_path = ONLINE_MODEL_STATUS_FILE + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(status, f)
            os.replace(tmp_path, ONLINE_MODEL_STATUS_FILE)
        except Exception as e:
            logger.error(f"OnlineQuantModel status export error: {e}")

    def _load(self):
        if not os.path.exists(self.state_path):
            self._export_status()
            return
        try:
            with open(self.state_path, "rb") as f:
                state = pickle.load(f)
            self.feature_names = state["feature_names"]
            # warmup_trades is deliberately NOT restored from the persisted
            # state: it's a live policy knob (config), not trained model
            # state. Restoring it here would mean a raised
            # ONLINE_MODEL_WARMUP_TRADES never actually takes effect for an
            # existing state file, silently keeping the old (lower) bar.
            self.clf = state["clf"]
            self._fitted = state["fitted"]
            self._n_updates = state["n_updates"]
            self._mean = state["mean"]
            self._m2 = state["m2"]
            self._count_vec = state["count_vec"]
            self._pending = state.get("pending", {})
        except Exception as e:
            logger.error(f"OnlineQuantModel load error: {e}")
        self._export_status()
