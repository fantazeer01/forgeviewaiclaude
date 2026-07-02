import csv
import glob
import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

logger = logging.getLogger(__name__)

# Canonical feature order the model is trained and served on. Maps 1:1 onto
# QuantFeatureExtractor.extract()'s live output (see core/quant_features.py)
# except volume_24h, which the historical dataset never captured and is
# therefore excluded here -- a model can only be evaluated on features that
# exist in both the historical training data and the live feed.
FEATURE_NAMES = [
    "yes_price", "no_price", "price_velocity", "price_acceleration",
    "order_book_imbalance", "spread", "spread_compression", "time_remaining_pct",
]

# Historical dataset column -> canonical feature name. The historical capture
# (forgeview-ai's Market Microstructure Feature Capture v1) used these exact
# names; time_remaining_pct is derived from seconds_to_expiry / 300.
HISTORICAL_COLUMN_MAP = {
    "yes_price": "yes_price",
    "no_price": "no_price",
    "price_velocity": "repricing_velocity",
    "price_acceleration": "repricing_acceleration",
    "order_book_imbalance": "book_imbalance",
    "spread": "bid_ask_spread",
    "spread_compression": "spread_compression",
}

MARKET_WINDOW_SEC = 300.0

# Standardized feature values are clipped to this many std-devs before the
# linear layer. Needed because some historical features (notably
# time_remaining_pct, which forgeview-ai's capture always anchored at ~60s
# into the 5-minute window) have near-zero training variance; a live value
# far from that narrow historical range would otherwise standardize to a
# huge z-score and saturate the sigmoid to 0.0 or 1.0 rather than degrading
# gracefully.
Z_SCORE_CLIP = 8.0


def _parse_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def load_historical_dataset(paths: list[str]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Reads one or more microstructure-schema CSVs (outcome-labeled) and
    returns (X, y, yes_price_baseline) using the canonical FEATURE_NAMES
    order. Rows without a usable outcome are skipped. yes_price_baseline is
    returned alongside so the model can be compared against the naive
    "trust the market's own YES price" baseline, per forgeview-ai's own
    evidence standard (a model that can't beat this isn't an edge)."""
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    yes_price_baseline: list[float] = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                outcome_raw = row.get("outcome")
                if outcome_raw in (None, ""):
                    continue
                try:
                    outcome = float(outcome_raw)
                except ValueError:
                    continue
                features = []
                for name in FEATURE_NAMES:
                    if name == "time_remaining_pct":
                        seconds = _parse_float(row.get("seconds_to_expiry"))
                        features.append(seconds / MARKET_WINDOW_SEC if seconds == seconds else float("nan"))
                    else:
                        features.append(_parse_float(row.get(HISTORICAL_COLUMN_MAP[name])))
                X_rows.append(features)
                y_rows.append(outcome)
                yes_price_baseline.append(_parse_float(row.get("yes_price")))
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float), yes_price_baseline


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((y_pred >= threshold).astype(float) == y_true))


def _clip_z(X):
    return np.clip(X, -Z_SCORE_CLIP, Z_SCORE_CLIP)


class QuantModel:
    """scikit-learn L2-regularized logistic regression (median imputation ->
    standardization -> z-score clipping -> LogisticRegression), the same
    modelling approach forgeview-ai used for Baseline Probability Model v1.
    That model did not beat raw Polymarket YES price on log loss or Brier
    score in the source repo's own evidence-gated evaluation (verdict:
    NO_EDGE_FOUND_YET), and a YES-plus-microstructure-features version lost
    twice more (D-031, D-033) after that feature set was built specifically
    to try to fix it. A from-scratch reproduction of the same experiment on
    this project's copy of the historical data (see data/historical/README.md)
    got the same result: log loss 0.598 vs. 0.591 for YES price, Brier 0.207
    vs. 0.205 on a held-out split.

    predict_proba() output should be treated as an experimental signal, not
    a demonstrated edge, until it's shown to beat the yes_price baseline on
    live-collected data.
    """

    def __init__(self, feature_names: list[str] = FEATURE_NAMES, C: float = 1.0):
        self.feature_names = list(feature_names)
        self.C = C
        self.pipeline: Optional[Pipeline] = None

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clip", FunctionTransformer(_clip_z)),
            ("logreg", LogisticRegression(C=self.C, max_iter=2000)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantModel":
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict_proba_one(self, features: dict) -> Optional[float]:
        """Convenience for live use: takes a QuantFeatureExtractor.extract()-style
        dict (by feature name) and returns a single win-probability, or None if
        the model isn't trained yet."""
        if self.pipeline is None:
            return None
        row = [features.get(name) if features.get(name) is not None else float("nan")
               for name in self.feature_names]
        return float(self.predict_proba(np.array([row], dtype=float))[0])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"feature_names": self.feature_names, "C": self.C, "pipeline": self.pipeline}, f)

    @classmethod
    def load(cls, path: str) -> Optional["QuantModel"]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            model = cls(d["feature_names"], C=d.get("C", 1.0))
            model.pipeline = d["pipeline"]
            return model
        except Exception as e:
            logger.error(f"QuantModel load error: {e}")
            return None


def train_test_split_indices(n: int, test_fraction: float = 0.3, seed: int = 20260703):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    return idx[n_test:], idx[:n_test]


def train_and_evaluate(data_paths: list[str], seed: int = 20260703) -> dict:
    """Trains a QuantModel on the given historical CSVs and reports both its
    own metrics and the yes-price-as-probability baseline on the same held-out
    split, mirroring forgeview-ai's evaluation discipline: a model is only
    interesting if it beats that baseline. Returns {"results": ..., "model": ...}."""
    X, y, yes_price = load_historical_dataset(data_paths)
    yes_price = np.array(yes_price, dtype=float)
    train_idx, test_idx = train_test_split_indices(len(y), seed=seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    yes_price_test = yes_price[test_idx]

    model = QuantModel().fit(X_train, y_train)
    model_pred = model.predict_proba(X_test)

    results = {
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": float(np.mean(y)),
        "model": {
            "log_loss": log_loss(y_test, model_pred),
            "brier": brier_score(y_test, model_pred),
            "accuracy": accuracy(y_test, model_pred),
        },
        "yes_price_baseline": {
            "log_loss": log_loss(y_test, yes_price_test),
            "brier": brier_score(y_test, yes_price_test),
            "accuracy": accuracy(y_test, yes_price_test),
        },
    }
    results["model_beats_yes_price"] = (
        results["model"]["log_loss"] < results["yes_price_baseline"]["log_loss"]
        and results["model"]["brier"] < results["yes_price_baseline"]["brier"]
    )
    return {"results": results, "model": model}


def default_historical_paths(data_dir: str = "data/historical") -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "microstructure_dataset_batch_*.csv")))
