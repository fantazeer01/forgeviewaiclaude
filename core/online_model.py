import datetime
import json
import logging
import math
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.linear_model import SGDClassifier

from config.settings import (
    ONLINE_MODEL_STATE_FILE, ONLINE_MODEL_WARMUP_TRADES,
    ONLINE_MODEL_CONFIDENCE_THRESHOLD,
    ONLINE_MODEL_STATUS_FILE, ONLINE_MODEL_PRIOR_YES_PRICE_WEIGHT,
    ONLINE_MODEL_PRIOR_INTERCEPT, ONLINE_MODEL_OWN_THRESHOLD,
    ONLINE_MODEL_COMBINER_THRESHOLD, ONLINE_MODEL_CALIBRATION_UPPER,
    ONLINE_MODEL_CALIBRATION_STEEPNESS, BET_SIZES,
)
# ONLINE_MODEL_CALIBRATION_LOWER (0.20) is not imported separately: the
# tanh-based transform below is symmetric around 0.5 by construction
# (tanh is an odd function), so the lower asymptote always equals
# 1 - ONLINE_MODEL_CALIBRATION_UPPER without needing to be passed in.
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
    always defers to the repricing signal passed in, and (in isolation) every
    one of those resolutions would be fed into the model as a training
    example. CAUTION: under run.py's current QUANT_ONLY_MODE=True wiring,
    this warm-up fallback branch is explicitly skipped (no trade opens, so
    record_features()/resolve() are never called), so a model reset back to
    n_updates=0 would never re-warm itself live -- it would need a non-empty
    warm-up trading path (QUANT_ONLY_MODE=False, or a manual backfill) to
    ever accumulate training examples again. Prefer temporarily adjusting
    ONLINE_MODEL_OWN_THRESHOLD over resetting the model for this reason.

    After warm-up, a trade requires BOTH signals to agree: the model's own
    (calibrated) prediction must exceed ONLINE_MODEL_OWN_THRESHOLD AND
    the signal combiner's independent output (passed in as `repricing_signal`
    -- see run.py, which now feeds SignalCombiner.combine()'s result here
    instead of the raw repricing detector) must be non-None, i.e. already
    cleared its own ONLINE_MODEL_COMBINER_THRESHOLD (0.60). Sized via a flat
    BET_SIZES lookup table keyed off the signal combiner's confidence (see
    kelly_size()), not a Kelly-criterion formula.

    predict_proba_one() runs the raw SGDClassifier output through a
    tanh-based calibration (_calibrate_proba) before returning it: this
    project observed the raw sigmoid saturating to ~0.0/1.0 well before 200
    updates (coefficients growing past +/-50 by update 234), so decide()'s
    threshold check should not trust the raw value at face value.
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
        if not self._fitted:
            self._seed_yes_price_prior()

    def _seed_yes_price_prior(self):
        """Manually initializes the linear model's weights with an informed
        prior BEFORE any real partial_fit() call, instead of starting from
        an uninformative all-zero coefficient vector. This runs only on a
        genuinely fresh model (self._fitted was False after _load(), i.e.
        no persisted real training progress exists) -- it never overwrites
        real learned state.

        This is NOT synthetic training data: no fake example is fed through
        update(), and n_updates/warmup progress are completely untouched
        (still 0, still gated on real resolved trades). It's a documented
        initial condition grounded in a real, statistically significant
        finding (docs/polymarket/DECISIONS.md D-002). Every real resolved
        trade from here on still calls partial_fit() and performs a normal
        SGD gradient step starting from this prior, exactly as it would
        from an all-zero start -- only the starting point changes, not the
        learning mechanism.

        Caveat: the calibration below (P~0.40 at yes_price=0.45, P~0.55 at
        yes_price=0.60) holds only at the instant this runs, when the
        online standardizer is still at its (mean=0, std=1) default and so
        transforms yes_price ~unchanged. As real data accumulates, both the
        standardizer and the coefficient will evolve away from this exact
        snapshot -- what's durable is the direction (higher yes_price biases
        toward YES), not the precise probabilities quoted here.
        """
        n = len(self.feature_names)
        self.clf.coef_ = np.zeros((1, n))
        if "yes_price" in self.feature_names:
            idx = self.feature_names.index("yes_price")
            self.clf.coef_[0][idx] = ONLINE_MODEL_PRIOR_YES_PRICE_WEIGHT
        self.clf.intercept_ = np.array([ONLINE_MODEL_PRIOR_INTERCEPT])
        self.clf.classes_ = np.array([0, 1])
        self.clf.t_ = 1.0
        # sklearn only sets n_features_in_ itself on a partial_fit() call
        # that passes classes= (a genuine "first fit") -- since seeding
        # bypasses partial_fit entirely, it's otherwise left unset, which
        # would silently skip sklearn's own feature-count validation on
        # every later call (masking exactly the kind of shape mismatch
        # _migrate_new_features() has to guard against by hand).
        self.clf.n_features_in_ = n
        self._fitted = True
        logger.info(
            f"OnlineQuantModel: seeded yes_price prior (weight="
            f"{ONLINE_MODEL_PRIOR_YES_PRICE_WEIGHT}, intercept="
            f"{ONLINE_MODEL_PRIOR_INTERCEPT}) per docs/polymarket/DECISIONS.md "
            f"D-002 -- not learned from real data yet"
        )

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
        p_raw = float(self.clf.predict_proba(x_std)[0][1])
        return self._calibrate_proba(p_raw)

    @staticmethod
    def _calibrate_proba(p_raw: float) -> float:
        """Compresses the raw sigmoid output through a tanh-based transform
        into an asymptotic ~(0.20, 0.80) band instead of trusting it at face
        value. tanh(x) = 2*sigmoid(2x) - 1, so this is itself built from a
        (steeper) sigmoid -- "sigmoid scaling" in the literal sense, not a
        hard clip: p_raw=0.5 stays exactly 0.5, and even a fully saturated
        p_raw=1.0 only approaches (never reaches) the upper bound, so the
        transform is smooth and monotonic (ordering between predictions is
        preserved) everywhere.
        """
        centered = 2.0 * (p_raw - 0.5)  # (0,1) -> (-1,1)
        squashed = math.tanh(ONLINE_MODEL_CALIBRATION_STEEPNESS * centered)
        span = ONLINE_MODEL_CALIBRATION_UPPER - 0.5
        return 0.5 + span * squashed

    def decide(self, features: dict, repricing_signal,
               confidence_threshold: float = ONLINE_MODEL_CONFIDENCE_THRESHOLD):
        """Returns (should_trade, direction, win_probability, reason).

        Before warm-up: mirrors the repricing signal exactly (fires iff a
        repricing signal fired, using its own direction/confidence).

        After warm-up: requires BOTH the model's own (calibrated) prediction
        AND the signal combiner's independent agreement before firing --
        `repricing_signal` here is expected to actually be the signal
        combiner's output once live (see run.py), which is non-None only
        when its own weighted confidence already exceeds
        ONLINE_MODEL_COMBINER_THRESHOLD. Either signal alone is no longer
        sufficient; this replaces the old single-model-confidence gate
        entirely. confidence_threshold is accepted for backward
        compatibility but no longer used in the live branch -- the model's
        own bar is ONLINE_MODEL_OWN_THRESHOLD (see config/settings.py for the
        current value and whether it's been temporarily lowered from its 0.5
        spec value).

        predict_proba_one() always returns the calibrated P(YES wins |
        features) -- the model has no direction feature, so it cannot ask
        "does this specific proposed bet win," only "does YES win." For a
        YES-direction repricing_signal that's directly usable; for a
        NO-direction one (see core/signal_combiner.py's extreme-reversion
        zone, added 2026-07-06) the model's own bar must be checked against
        P(NO wins) = 1 - p instead, or a NO signal would be gated by the
        wrong side of the model's belief.
        """
        if not self.is_warmed_up():
            if repricing_signal is None:
                return False, None, None, "warmup: no repricing signal"
            return True, repricing_signal.direction, repricing_signal.confidence, "warmup: repricing rule"

        p = self.predict_proba_one(features)
        if p is None:
            return False, None, None, "model not fitted"
        if repricing_signal is None or repricing_signal.confidence <= ONLINE_MODEL_COMBINER_THRESHOLD:
            return False, None, p, (
                f"signal combiner did not agree (confidence <= {ONLINE_MODEL_COMBINER_THRESHOLD})"
            )
        direction = repricing_signal.direction
        own_side_p = p if direction == "YES" else (1.0 - p)
        if own_side_p <= ONLINE_MODEL_OWN_THRESHOLD:
            return False, None, p, (
                f"online model P({direction})={own_side_p:.3f} <= {ONLINE_MODEL_OWN_THRESHOLD}"
            )
        return True, direction, p, (
            f"online model P({direction})={own_side_p:.3f} AND combiner "
            f"confidence={repricing_signal.confidence:.3f} agree"
        )

    def kelly_size(self, combiner_confidence: float) -> float:
        """Simple step-function bet sizing keyed directly off the signal
        combiner's confidence value (NOT the online model's own calibrated
        probability) -- replaces the previous Kelly-criterion formula
        entirely per explicit request. See config.settings.BET_SIZES for
        the table: the largest threshold the confidence clears determines
        the flat dollar size. Below the lowest bucket (0.60) returns $0 --
        should never happen in practice since the signal combiner itself
        never returns a signal below that same 0.60 threshold."""
        for threshold in sorted(BET_SIZES, reverse=True):
            if combiner_confidence >= threshold:
                return float(BET_SIZES[threshold])
        return 0.0

    def record_features(self, market_id: str, features: dict):
        """Stash the feature snapshot a decision was made from, keyed by
        market_id, so resolve() can pair it with the eventual outcome.
        Calls save() immediately (2026-07-06 fix): previously this only
        updated the in-memory dict, and _pending was only ever persisted to
        disk from within update()'s save() call -- so a trade opened, then a
        bot restart before any OTHER trade resolved, would silently lose
        that trade's pending features (resolve() would find nothing there
        and return False, with no error, no visible symptom). Saving here
        closes that window."""
        self._pending[market_id] = features
        self.save()

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
            persisted_features = state["feature_names"]
            target_features = self.feature_names  # set in __init__ before _load()
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
            if persisted_features == target_features:
                self.feature_names = persisted_features
            elif (len(target_features) > len(persisted_features)
                  and target_features[:len(persisted_features)] == persisted_features):
                self._migrate_new_features(persisted_features, target_features)
            else:
                # Not a clean append (a feature was removed, renamed, or
                # reordered) -- no safe automatic migration exists, since
                # the learned coefficients are positional. Keep training on
                # whatever was actually persisted rather than silently
                # mismapping weights to the wrong features.
                logger.error(
                    f"OnlineQuantModel feature_names changed incompatibly "
                    f"(persisted={persisted_features}, target={target_features}) "
                    f"-- keeping the persisted feature set; a real migration or "
                    f"reset is needed to pick up the new one."
                )
                self.feature_names = persisted_features
        except Exception as e:
            logger.error(f"OnlineQuantModel load error: {e}")
        self._export_status()

    def _migrate_new_features(self, persisted_features: list[str], target_features: list[str]):
        """New features were purely APPENDED to FEATURE_NAMES (the only
        migration this supports -- inserting/reordering/removing would
        silently mismap the existing positional coefficients to the wrong
        features). Preserves every real learned weight and all training
        progress: pads the standardizer accumulators and the classifier's
        own coef_ with zeros for the new dimensions, so the new features
        start at exactly zero influence -- neutral, not a guess -- and get
        trained up from there by ordinary partial_fit() calls same as any
        other feature. n_updates/fitted are untouched; nothing is reset."""
        n_new = len(target_features) - len(persisted_features)
        self._mean = np.concatenate([self._mean, np.zeros(n_new)])
        self._m2 = np.concatenate([self._m2, np.zeros(n_new)])
        self._count_vec = np.concatenate([self._count_vec, np.zeros(n_new)])
        if self._fitted and hasattr(self.clf, "coef_"):
            self.clf.coef_ = np.hstack([self.clf.coef_, np.zeros((1, n_new))])
            # sklearn validates future predict()/partial_fit() calls against
            # n_features_in_, tracked separately from coef_'s own shape --
            # padding coef_ alone still raises "X has N features, but
            # SGDClassifier is expecting <old N> features" without this.
            if hasattr(self.clf, "n_features_in_"):
                self.clf.n_features_in_ = len(target_features)
        self.feature_names = target_features
        logger.info(
            f"OnlineQuantModel: extended feature set with {n_new} new feature(s) "
            f"{target_features[len(persisted_features):]}, zero-initialized -- "
            f"n_updates={self._n_updates} and all existing weights preserved."
        )
