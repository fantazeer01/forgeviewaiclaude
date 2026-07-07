import datetime
import json
import logging
import math
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from config.settings import (
    ONLINE_MODEL_STATE_FILE, ONLINE_MODEL_WARMUP_TRADES,
    ONLINE_MODEL_CONFIDENCE_THRESHOLD,
    ONLINE_MODEL_STATUS_FILE, ONLINE_MODEL_PRIOR_YES_PRICE_WEIGHT,
    ONLINE_MODEL_PRIOR_INTERCEPT, ONLINE_MODEL_OWN_THRESHOLD,
    ONLINE_MODEL_COMBINER_THRESHOLD, ONLINE_MODEL_CALIBRATION_UPPER,
    ONLINE_MODEL_CALIBRATION_STEEPNESS, ONLINE_MODEL_C,
    ONLINE_MODEL_HEALTH_CHECK_INTERVAL, ONLINE_MODEL_KELLY_BANKROLL_USD,
    ONLINE_MODEL_KELLY_MIN_SIZE_USD, ONLINE_MODEL_KELLY_MAX_SIZE_USD,
    STABILITY_CHECK_INTERVAL, STABILITY_WIN_RATE_WINDOW, STABILITY_MIN_WIN_RATE,
    STABILITY_PREDICTIONS_WINDOW, STABILITY_MIN_PREDICTION_STD, STABILITY_COEF_BOUND,
    MODEL_HEALTH_LOG, RETRAIN_INTERVAL, RETRAIN_WINDOW, MODEL_RETRAIN_LOG,
)
from core.kelly_criterion import kelly_fraction, net_odds_from_price
# ONLINE_MODEL_CALIBRATION_LOWER (0.20) is not imported separately: the
# tanh-based transform below is symmetric around 0.5 by construction
# (tanh is an odd function), so the lower asymptote always equals
# 1 - ONLINE_MODEL_CALIBRATION_UPPER without needing to be passed in.
from core.live_features import FEATURE_NAMES

logger = logging.getLogger(__name__)

Z_SCORE_CLIP = 8.0
# Bounds how much of the accumulated resolution history a single fit() call
# re-trains on -- fit cost grows with history size, so this keeps it bounded
# over a long-running session instead of growing forever. Same shape as the
# dashboard's own Math.min(portfolio.closed.length, 5000) cap for its Monte
# Carlo inputs.
HISTORY_MAX = 5000
# Saturation health check (2026-07-07, replaces the old COEF_CLIP hard clip
# entirely -- see the class docstring's "Why LogisticRegression" section for
# why clipping individual coefficients didn't actually prevent the model
# collapsing to a near-constant output). Probes predict_proba_one() across a
# spread of yes_price values with everything else held at the running mean;
# a healthy model should show at least some spread in its output across that
# sweep. SATURATION_EPSILON is deliberately tight -- 0.01 on a probability
# scale is "these 5 genuinely different inputs produced the same answer,"
# not normal model behavior.
SATURATION_PROBE_YES_PRICES = [0.20, 0.35, 0.50, 0.65, 0.80]
SATURATION_EPSILON = 0.01


class OnlineQuantModel:
    """Online-learning trading model: a LogisticRegression that re-fits on
    its full accumulated resolution history after every resolved trade,
    instead of being trained once on a static historical batch.

    This is a deliberately different approach from the two prior batch-model
    sprints (docs/polymarket/DECISIONS.md D-001, data/historical/README.md):
    those trained on forgeview-ai's historical research data and found every
    model scored at or below chance on genuinely unseen live data, because
    historical and live data come from different market regimes a static
    model can't adapt to. An online model only ever trains on this project's
    own live outcomes, so there is no cross-regime mismatch -- but it starts
    from zero evidence, so it needs a warm-up period before its predictions
    can be trusted with real trading decisions.

    For the first ONLINE_MODEL_WARMUP_TRADES (50) resolved trades, decide()
    always defers to the repricing signal passed in (see run.py's warm-up
    branch, which now also sizes those trades at a flat WARMUP_FLAT_SIZE_USD
    regardless of confidence -- real Kelly sizing is reserved for after
    warm-up, once the model itself is actually driving decisions). Every
    one of those warm-up resolutions is still fed into the model as a
    training example, via update().

    After warm-up, a trade requires BOTH signals to agree: the model's own
    (calibrated) prediction must exceed ONLINE_MODEL_OWN_THRESHOLD AND
    the signal combiner's independent output (passed in as `repricing_signal`
    -- see run.py, which feeds SignalCombiner.combine()'s result here instead
    of the raw repricing detector) must be non-None, i.e. already cleared its
    own ONLINE_MODEL_COMBINER_THRESHOLD (0.60). Sized via real Kelly-criterion
    sizing keyed off the model's own win_probability AND the market's actual
    payout ratio (see kelly_size()) -- a flat BET_SIZES lookup table keyed
    only off signal_combiner confidence was used from 2026-07-06 to
    2026-07-07, but it ignored entry_price entirely, so it couldn't tell a
    favorable payout (low yes_price, high b) from an unfavorable one (high
    yes_price, low b) at the same confidence level.

    Why LogisticRegression(solver="liblinear") instead of SGDClassifier
    (2026-07-07, second divergence): the SGD model's coefficients diverged
    TWICE -- first to +/-80 (intercept) after 509 updates, then again to the
    COEF_CLIP=10 hard bound itself (multiple correlated features -- yes_price,
    no_price, ohlc_open/high/low/close all move together -- pinned right at
    +/-10 simultaneously) after 328 more updates post-reset. Root cause,
    confirmed by inspecting the live pickle before this rewrite: (1) clipping
    each coefficient independently to [-10, 10] does nothing to bound their
    COMBINED linear contribution to the logit when several correlated
    features are all pinned at the boundary at once -- the dot product can
    still be enormous even though no single weight exceeds the clip: (2) the
    alpha=1e-2 regularization strengthening from the same 2026-07-06 fix
    turned out to have NEVER actually taken effect on the running model --
    _load() unconditionally replaces self.clf with whatever classifier
    object was pickled, and a classifier's hyperparameters (alpha, in SGD's
    case) are baked into that object at construction time, not re-read from
    config on every load. Only a full reset (a fresh __init__ with no
    persisted pkl) would have picked up the new alpha; simply restarting the
    bot after that code change did not. (Both issues are fixed here: no more
    per-coefficient clip to reason about, and _load() now explicitly
    re-applies C/solver onto the loaded classifier every time, so a
    future tuning change takes effect on the next restart without requiring
    another manual reset.)

    LogisticRegression + liblinear structurally cannot repeat either failure:
    each fit() call solves a single well-posed, strongly-regularized (C=0.1)
    convex optimization over the FULL accumulated history (capped at
    HISTORY_MAX) from scratch -- there is no sequence of small per-sample
    gradient steps that can compound into a runaway trajectory the way SGD's
    partial_fit stream could. liblinear requires at least 2 distinct outcome
    classes to have been observed before its first fit() call (a single-class
    fit raises), so update() only actually re-fits once both a win and a loss
    have been seen; before that (and always, on a fresh model) predict_proba_one()
    serves the seeded yes_price prior instead of an unfit classifier. Note
    `warm_start` was deliberately NOT set: sklearn's liblinear code path
    returns before ever consulting warm_start (verified against sklearn
    1.9's LogisticRegression.fit() source), so it would have been a silent
    no-op -- refitting the full (bounded) history each time is cheap enough
    at this data scale to not need it.

    As a second, independent safety net (not a replacement for the above,
    which prevents the specific past failure mode -- this catches ANY future
    one): every ONLINE_MODEL_HEALTH_CHECK_INTERVAL (10) resolved trades,
    _run_health_check() probes predict_proba_one() across a deliberate
    spread of yes_price values (SATURATION_PROBE_YES_PRICES) with every
    other feature held at its running mean. A healthy model shows some
    spread in its output across that sweep; if all 5 probes land within
    SATURATION_EPSILON of each other, that's the exact signature of the
    binary-step-function collapse observed twice now, and the model is
    auto-reset with a logged warning rather than left to keep serving a
    degenerate prediction. The result ("healthy"/"saturated"/"reset") is
    exported to data/online_model_status.json as model_health so the
    dashboard can surface it.

    predict_proba_one() runs the raw classifier output through a tanh-based
    calibration (_calibrate_proba) before returning it, compressing it into
    an asymptotic ~(0.20, 0.80) band -- kept as an extra safety margin
    against overconfident predictions even with the new, better-regularized
    model; not itself a fix for either divergence (that was the raw
    coefficients, not the calibration).
    """

    def __init__(self, feature_names: list[str] = FEATURE_NAMES,
                 state_path: str = ONLINE_MODEL_STATE_FILE,
                 warmup_trades: int = ONLINE_MODEL_WARMUP_TRADES):
        self.feature_names = list(feature_names)
        self.state_path = state_path
        self.warmup_trades = warmup_trades
        self.clf = self._fresh_classifier()
        self._fitted = False
        self._n_updates = 0
        n = len(self.feature_names)
        self._mean = np.zeros(n)
        self._m2 = np.zeros(n)
        self._count_vec = np.zeros(n)
        self._pending: dict[str, dict] = {}
        self._history_X: list[np.ndarray] = []
        self._history_y: list[int] = []
        self._model_health = "healthy"
        # Rolling buffer of this model's own predict_proba_one() output at
        # the moment of each resolved trade (see update()) -- used by
        # _run_stability_monitor()'s diversity check. Real predictions on
        # real examples, not a synthetic sweep.
        self._recent_predictions: list[float] = []
        self._load()
        if not self._fitted:
            self._seed_yes_price_prior()

    @staticmethod
    def _fresh_classifier() -> LogisticRegression:
        # penalty is deliberately NOT passed -- 'l2' is already
        # LogisticRegression's default, and sklearn 1.8+ emits a
        # FutureWarning ("removed in 1.10") whenever it's set explicitly.
        return LogisticRegression(C=ONLINE_MODEL_C, solver="liblinear", random_state=20260704)

    def _seed_yes_price_prior(self):
        """Manually initializes the linear model's weights with an informed
        prior BEFORE any real fit() call, instead of starting from an
        uninformative all-zero coefficient vector. This runs only on a
        genuinely fresh model (self._fitted was False after _load(), i.e.
        no persisted real training progress exists) -- it never overwrites
        real learned state.

        This is NOT synthetic training data: no fake example is fed through
        update(), and n_updates/warmup progress are completely untouched
        (still 0, still gated on real resolved trades). It's a documented
        initial condition grounded in a real, statistically significant
        finding (docs/polymarket/DECISIONS.md D-002). Every real resolved
        trade from here on still calls update() and performs a normal
        refit starting from this prior in the training history, exactly as
        it would from an all-zero start -- only the starting point changes,
        not the learning mechanism.

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
        # sklearn only sets n_features_in_ itself on a real fit() call --
        # since seeding bypasses fit() entirely, it's otherwise left unset,
        # which would silently skip sklearn's own feature-count validation on
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

    def _reset_to_fresh(self):
        """Full reset back to a brand-new model: fresh classifier (current
        C/solver), zeroed standardizer, cleared history/pending,
        n_updates back to 0, then re-seeded with the same yes_price prior a
        genuinely new instance would get. Used by the saturation health
        check (_run_health_check) -- this is the automatic equivalent of the
        manual "delete the pkl and restart" reset performed twice by hand for
        the two prior divergences."""
        n = len(self.feature_names)
        self.clf = self._fresh_classifier()
        self._fitted = False
        self._n_updates = 0
        self._mean = np.zeros(n)
        self._m2 = np.zeros(n)
        self._count_vec = np.zeros(n)
        self._pending = {}
        self._history_X = []
        self._history_y = []
        self._recent_predictions = []
        self._seed_yes_price_prior()

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def model_health(self) -> str:
        return self._model_health

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
        data too.

        Appends to the accumulated history and re-fits on the whole thing
        (bounded at HISTORY_MAX) rather than taking a single incremental
        gradient step -- see the class docstring for why. liblinear needs
        both outcome classes present at least once; before that, this still
        records the history and advances n_updates, it just skips the fit()
        call itself (predict_proba_one() keeps serving the seeded prior or
        whatever was last actually fitted)."""
        x = self._vectorize(features)
        self._update_standardizer(x)
        x_std = self._standardize(x)
        self._history_X.append(x_std)
        self._history_y.append(int(outcome))
        if len(self._history_y) > HISTORY_MAX:
            self._history_X = self._history_X[-HISTORY_MAX:]
            self._history_y = self._history_y[-HISTORY_MAX:]
        if len(set(self._history_y)) >= 2:
            self.clf.fit(np.array(self._history_X), np.array(self._history_y))
            self._fitted = True
        self._n_updates += 1
        # Real prediction on this real, just-resolved example's own features
        # (post-fit, so it reflects the just-updated model) -- feeds the
        # stability monitor's diversity check below.
        p_now = self.predict_proba_one(features)
        if p_now is not None:
            self._recent_predictions.append(p_now)
            if len(self._recent_predictions) > STABILITY_PREDICTIONS_WINDOW:
                self._recent_predictions = self._recent_predictions[-STABILITY_PREDICTIONS_WINDOW:]
        if self._n_updates % ONLINE_MODEL_HEALTH_CHECK_INTERVAL == 0:
            self._run_health_check()
        if self._n_updates % STABILITY_CHECK_INTERVAL == 0:
            self._run_stability_monitor()
        if self._n_updates % RETRAIN_INTERVAL == 0:
            self._run_scheduled_retrain()
        self.save()

    def _run_health_check(self) -> str:
        """Probes predict_proba_one() across SATURATION_PROBE_YES_PRICES
        (yes_price swept from 0.20 to 0.80, every other feature held at its
        running mean) and compares the spread of the 5 outputs against
        SATURATION_EPSILON. Sets and returns self._model_health
        ("healthy" or "saturated", or "reset" once a reset actually fires).
        This is a real behavioral check (does the model's output actually
        vary across genuinely different inputs), not a proxy on coefficient
        magnitude -- that's what let the previous COEF_CLIP-based approach
        miss the second divergence even though it kept every individual
        coefficient within its bound."""
        if not self._fitted:
            self._model_health = "healthy"
            return self._model_health
        base = {name: self._mean[i] for i, name in enumerate(self.feature_names)}
        outputs = []
        for yp in SATURATION_PROBE_YES_PRICES:
            feats = dict(base)
            feats["yes_price"] = yp
            if "no_price" in feats:
                feats["no_price"] = 1.0 - yp
            p = self.predict_proba_one(feats)
            if p is None:
                self._model_health = "healthy"
                return self._model_health
            outputs.append(p)
        spread = max(outputs) - min(outputs)
        if spread < SATURATION_EPSILON:
            logger.warning(
                f"OnlineQuantModel: saturation detected (predict_proba spread "
                f"{spread:.4f} < {SATURATION_EPSILON} across yes_price sweep "
                f"{SATURATION_PROBE_YES_PRICES}, outputs={[round(o, 4) for o in outputs]}) "
                f"after {self._n_updates} updates -- auto-resetting to a fresh model."
            )
            self._reset_to_fresh()
            self._model_health = "reset"
        else:
            self._model_health = "healthy"
        return self._model_health

    def _run_stability_monitor(self):
        """Every STABILITY_CHECK_INTERVAL (50) updates: a broader check than
        _run_health_check()'s saturation probe above. Three independent
        signals:
          - real recent win rate over the trailing STABILITY_WIN_RATE_WINDOW
            resolved examples -- WARNS only, does not reset (a losing
            streak alone isn't proof the model itself broke, could just be
            a bad market regime; auto-resetting on that would be reacting
            to noise).
          - real prediction diversity over the trailing
            STABILITY_PREDICTIONS_WINDOW actual predict_proba_one() outputs
            recorded at real resolutions (see update()) -- auto-resets,
            same "collapsed to a near-constant output" signature as the
            saturation check, just measured on real examples instead of a
            synthetic sweep.
          - raw |coefficient| magnitude against STABILITY_COEF_BOUND --
            auto-resets; at this model's normal scale (observed max ~0.41
            post-fix) exceeding 5.0 is a red flag on its own, not just a
            proxy.
        Always appends a report to MODEL_HEALTH_LOG regardless of outcome."""
        recent_y = self._history_y[-STABILITY_WIN_RATE_WINDOW:]
        win_rate = (sum(recent_y) / len(recent_y)) if recent_y else None
        win_rate_ok = win_rate is None or win_rate > STABILITY_MIN_WIN_RATE

        preds = self._recent_predictions[-STABILITY_PREDICTIONS_WINDOW:]
        pred_std = float(np.std(preds)) if len(preds) >= 2 else None
        diversity_ok = pred_std is None or pred_std > STABILITY_MIN_PREDICTION_STD

        if self._fitted and hasattr(self.clf, "coef_"):
            max_abs_coef = max(float(np.max(np.abs(self.clf.coef_))), float(np.max(np.abs(self.clf.intercept_))))
        else:
            max_abs_coef = None
        coef_ok = max_abs_coef is None or max_abs_coef <= STABILITY_COEF_BOUND

        warnings = []
        if not win_rate_ok:
            msg = f"win rate {win_rate:.3f} <= {STABILITY_MIN_WIN_RATE} over last {len(recent_y)} resolved examples"
            warnings.append(msg)
            logger.warning(f"OnlineQuantModel stability monitor: {msg}")

        action = "none"
        if not diversity_ok or not coef_ok:
            reasons = []
            if not diversity_ok:
                reasons.append(f"prediction std {pred_std:.4f} <= {STABILITY_MIN_PREDICTION_STD}")
            if not coef_ok:
                reasons.append(f"max abs coef {max_abs_coef:.4f} > {STABILITY_COEF_BOUND}")
            logger.warning(
                f"OnlineQuantModel stability monitor: auto-resetting at {self._n_updates} "
                f"updates ({'; '.join(reasons)})"
            )
            self._reset_to_fresh()
            self._model_health = "reset"
            action = "reset"

        self._append_jsonl(MODEL_HEALTH_LOG, {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_updates": self._n_updates,
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "win_rate_window": len(recent_y),
            "win_rate_ok": win_rate_ok,
            "prediction_std": round(pred_std, 4) if pred_std is not None else None,
            "prediction_window": len(preds),
            "diversity_ok": diversity_ok,
            "max_abs_coef": round(max_abs_coef, 4) if max_abs_coef is not None else None,
            "coef_ok": coef_ok,
            "warnings": warnings,
            "action": action,
        })

    def _predict_proba_with(self, clf, features: dict) -> float:
        """Same as predict_proba_one(), but against an arbitrary candidate
        classifier instead of self.clf -- used to evaluate a retrain
        candidate's health BEFORE it's swapped in (see
        _run_scheduled_retrain()), without mutating any live state."""
        x = self._vectorize(features)
        x_std = self._standardize(x).reshape(1, -1)
        p_raw = float(clf.predict_proba(x_std)[0][1])
        return self._calibrate_proba(p_raw)

    def _check_candidate_diversity(self, clf) -> tuple[bool, float]:
        base = {name: self._mean[i] for i, name in enumerate(self.feature_names)}
        outputs = []
        for yp in SATURATION_PROBE_YES_PRICES:
            feats = dict(base)
            feats["yes_price"] = yp
            if "no_price" in feats:
                feats["no_price"] = 1.0 - yp
            outputs.append(self._predict_proba_with(clf, feats))
        spread = max(outputs) - min(outputs)
        return spread >= SATURATION_EPSILON, spread

    def _feature_importance(self, clf) -> list[dict]:
        """Feature importance for a linear model = |coefficient| magnitude,
        ranked descending -- the direct, standard reading for a logistic
        regression fit on z-scored features (already on a comparable
        scale, so no further normalization is needed)."""
        ranked = sorted(zip(self.feature_names, clf.coef_[0]), key=lambda p: abs(p[1]), reverse=True)
        return [{"feature": name, "coef": round(float(c), 4)} for name, c in ranked]

    def _run_scheduled_retrain(self):
        """Every RETRAIN_INTERVAL (500) updates: trains a FRESH classifier
        from scratch on only the trailing RETRAIN_WINDOW (500) examples --
        NOT the full up-to-HISTORY_MAX history the continuous per-update
        refit in update() uses -- so the model can adapt to a regime shift
        instead of an ever-growing majority of old training data dominating
        forever. The candidate is verified against the same coefficient-
        bound and prediction-diversity checks _run_stability_monitor() uses
        BEFORE being swapped in: self.clf is only ever reassigned after the
        candidate has already passed every check (atomic switch -- there is
        no window where a partially-verified model is live). A candidate
        that fails stays rejected, logged, and the live model keeps running
        exactly as it was. Feature importance is logged either way."""
        window_X = self._history_X[-RETRAIN_WINDOW:]
        window_y = self._history_y[-RETRAIN_WINDOW:]
        if len(set(window_y)) < 2:
            logger.info(
                f"OnlineQuantModel: scheduled retrain skipped at {self._n_updates} updates -- "
                f"trailing {len(window_y)}-example window has only "
                f"{len(set(window_y))} outcome class(es) present."
            )
            return

        candidate = self._fresh_classifier()
        candidate.fit(np.array(window_X), np.array(window_y))
        max_abs_coef = max(float(np.max(np.abs(candidate.coef_))), float(np.max(np.abs(candidate.intercept_))))
        coef_ok = max_abs_coef <= STABILITY_COEF_BOUND
        diversity_ok, spread = self._check_candidate_diversity(candidate)
        accepted = coef_ok and diversity_ok
        importances = self._feature_importance(candidate)

        self._append_jsonl(MODEL_RETRAIN_LOG, {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_updates": self._n_updates,
            "window_size": len(window_y),
            "max_abs_coef": round(max_abs_coef, 4),
            "coef_ok": coef_ok,
            "diversity_spread": round(spread, 4),
            "diversity_ok": diversity_ok,
            "accepted": accepted,
            "feature_importance": importances,
        })

        if accepted:
            self.clf = candidate
            self._fitted = True
            top5 = ", ".join(f"{f['feature']}={f['coef']:+.3f}" for f in importances[:5])
            logger.info(
                f"OnlineQuantModel: scheduled retrain ACCEPTED at {self._n_updates} updates "
                f"(window={len(window_y)}, max_abs_coef={max_abs_coef:.4f}). Top features: {top5}"
            )
        else:
            logger.warning(
                f"OnlineQuantModel: scheduled retrain REJECTED at {self._n_updates} updates "
                f"(coef_ok={coef_ok}, diversity_ok={diversity_ok}, spread={spread:.4f}) -- "
                f"keeping the live model unchanged."
            )

    @staticmethod
    def _append_jsonl(path: str, entry: dict):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"OnlineQuantModel jsonl append error ({path}): {e}")

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
               confidence_threshold: float = ONLINE_MODEL_CONFIDENCE_THRESHOLD,
               asset: Optional[str] = None):
        """Returns (should_trade, direction, win_probability, reason).

        `asset` is optional and used only for the SKIP log lines below
        (2026-07-07 diagnostics) -- defaults to repricing_signal.asset when
        a signal is present (the authoritative source), falling back to
        this explicit kwarg only for the one case where repricing_signal is
        None and there'd otherwise be no way to know which asset a call was
        even for. Kept optional (not a required positional param) so the
        many existing decide() call sites in tests didn't need touching.

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
        current value).

        predict_proba_one() always returns the calibrated P(YES wins |
        features) -- the model has no direction feature, so it cannot ask
        "does this specific proposed bet win," only "does YES win." For a
        YES-direction repricing_signal that's directly usable; for a
        NO-direction one (see core/signal_combiner.py's extreme-reversion
        zone, added 2026-07-06) the model's own bar must be checked against
        P(NO wins) = 1 - p instead, or a NO signal would be gated by the
        wrong side of the model's belief.

        SKIP reasons are logged at INFO here (2026-07-07 diagnostics --
        previously only returned as a string, never logged anywhere, which
        made "why isn't anything trading" impossible to answer from the log
        alone). The SUCCESS ("SIGNAL [...]") line is deliberately NOT logged
        here, even though this method can return should_trade=True -- see
        run.py._decide_and_open, which is where kelly_size()'s REAL Kelly
        fraction is computed and can still be <= 0 (no edge) even when
        own_side_p already cleared ONLINE_MODEL_OWN_THRESHOLD here, since
        the two use different math (a simple probability bar vs. the real
        payout-ratio-aware Kelly formula). Logging a SIGNAL line from this
        method alone would be premature -- it doesn't yet know whether a
        trade will actually open.
        """
        resolved_asset = repricing_signal.asset if repricing_signal is not None else asset
        yes_price = features.get("yes_price")

        if not self.is_warmed_up():
            if repricing_signal is None:
                logger.info(f"SKIP [{resolved_asset}] reason=warmup: no repricing signal yes_price={yes_price}")
                return False, None, None, "warmup: no repricing signal"
            return True, repricing_signal.direction, repricing_signal.confidence, "warmup: repricing rule"

        p = self.predict_proba_one(features)
        if p is None:
            logger.info(f"SKIP [{resolved_asset}] reason=model not fitted yes_price={yes_price}")
            return False, None, None, "model not fitted"
        if repricing_signal is None or repricing_signal.confidence <= ONLINE_MODEL_COMBINER_THRESHOLD:
            reason = f"signal combiner did not agree (confidence <= {ONLINE_MODEL_COMBINER_THRESHOLD})"
            logger.info(f"SKIP [{resolved_asset}] reason={reason} yes_price={yes_price}")
            return False, None, p, reason
        direction = repricing_signal.direction
        own_side_p = p if direction == "YES" else (1.0 - p)
        if own_side_p <= ONLINE_MODEL_OWN_THRESHOLD:
            reason = f"online model P({direction})={own_side_p:.3f} <= {ONLINE_MODEL_OWN_THRESHOLD}"
            logger.info(f"SKIP [{resolved_asset}] reason={reason} yes_price={yes_price}")
            return False, None, p, reason
        return True, direction, p, (
            f"online model P({direction})={own_side_p:.3f} AND combiner "
            f"confidence={repricing_signal.confidence:.3f} agree"
        )

    def kelly_size(self, win_probability: float, yes_price: float) -> float:
        """Real Kelly-criterion sizing (2026-07-07 CRITICAL FIX -- replaces
        the flat BET_SIZES lookup table used 2026-07-06 to 2026-07-07, which
        keyed size off signal_combiner confidence and ignored entry_price
        entirely: it couldn't tell a favorable payout (low yes_price, high
        b) from an unfavorable one (high yes_price, low b) at the same
        confidence level).

        b = (1-yes_price)/yes_price is this market's real payout ratio (a $1
        YES win pays b dollars profit; a loss costs the full stake) --
        computed via core/kelly_criterion.py's net_odds_from_price(), the
        same already-tested helper core/kelly_criterion.py's own
        kelly_position_size() uses. f = (p*b - (1-p))/b via that module's
        kelly_fraction(), using FULL Kelly (not quarter-Kelly) since the
        [ONLINE_MODEL_KELLY_MIN_SIZE_USD, ONLINE_MODEL_KELLY_MAX_SIZE_USD]
        clamp below already bounds the position size independently.

        kelly_fraction() already clamps a non-positive edge to exactly 0.0
        -- callers MUST treat a 0.0 return as "do not open this trade," not
        "open at $0" (see run.py._decide_and_open, which returns None
        without opening when this is 0.0). A positive fraction is applied
        to ONLINE_MODEL_KELLY_BANKROLL_USD and clamped to
        [ONLINE_MODEL_KELLY_MIN_SIZE_USD, ONLINE_MODEL_KELLY_MAX_SIZE_USD] --
        so even a razor-thin positive edge still opens at the $5 floor,
        it's only edges <= 0 that are skipped entirely."""
        b = net_odds_from_price(yes_price)
        f = kelly_fraction(win_probability, b)
        if f <= 0:
            return 0.0
        size = f * ONLINE_MODEL_KELLY_BANKROLL_USD
        return max(ONLINE_MODEL_KELLY_MIN_SIZE_USD, min(ONLINE_MODEL_KELLY_MAX_SIZE_USD, size))

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
            "history_X": self._history_X,
            "history_y": self._history_y,
            "model_health": self._model_health,
            "recent_predictions": self._recent_predictions,
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
            "model_health": self._model_health,
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
            # Re-apply current-code hyperparameters onto the loaded classifier
            # (2026-07-07) instead of trusting whatever was pickled -- this is
            # the actual root cause behind the previous alpha fix silently
            # never taking effect: a classifier's hyperparameters are baked in
            # at construction time and persist through pickling, so bumping
            # ONLINE_MODEL_C in config alone would otherwise do nothing for an
            # existing state file until another full reset. Now a tuning
            # change takes effect on the very next restart.
            self.clf.C = ONLINE_MODEL_C
            self.clf.solver = "liblinear"
            self._fitted = state["fitted"]
            self._n_updates = state["n_updates"]
            self._mean = state["mean"]
            self._m2 = state["m2"]
            self._count_vec = state["count_vec"]
            self._pending = state.get("pending", {})
            self._history_X = state.get("history_X", [])
            self._history_y = state.get("history_y", [])
            self._model_health = state.get("model_health", "healthy")
            self._recent_predictions = state.get("recent_predictions", [])
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
        progress: pads the standardizer accumulators, the classifier's own
        coef_, and every row of the accumulated history with zeros for the
        new dimensions, so the new features start at exactly zero influence
        -- neutral, not a guess -- and get trained up from there by ordinary
        fit() calls same as any other feature. n_updates/fitted are
        untouched; nothing is reset."""
        n_new = len(target_features) - len(persisted_features)
        self._mean = np.concatenate([self._mean, np.zeros(n_new)])
        self._m2 = np.concatenate([self._m2, np.zeros(n_new)])
        self._count_vec = np.concatenate([self._count_vec, np.zeros(n_new)])
        if self._history_X:
            self._history_X = [np.concatenate([row, np.zeros(n_new)]) for row in self._history_X]
        if self._fitted and hasattr(self.clf, "coef_"):
            self.clf.coef_ = np.hstack([self.clf.coef_, np.zeros((1, n_new))])
            # sklearn validates future predict()/fit() calls against
            # n_features_in_, tracked separately from coef_'s own shape --
            # padding coef_ alone still raises "X has N features, but
            # LogisticRegression is expecting <old N> features" without this.
            if hasattr(self.clf, "n_features_in_"):
                self.clf.n_features_in_ = len(target_features)
        self.feature_names = target_features
        logger.info(
            f"OnlineQuantModel: extended feature set with {n_new} new feature(s) "
            f"{target_features[len(persisted_features):]}, zero-initialized -- "
            f"n_updates={self._n_updates} and all existing weights preserved."
        )
