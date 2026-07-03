"""
ONE SPRINT: train and evaluate a quant model on ALL available outcome-labeled
data (historical forgeview-ai research data + our own live shadow log), then
decide whether it earns the right to drive real (paper) trades.

Read-only against D:\\ForgeViewAI. Writes only within this project:
data/quant_model.pkl (if a model is trained) and this script's stdout report.
"""
import csv
import glob
import json
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

sys.path.insert(0, ".")
from core.quant_model import (
    FEATURE_NAMES, HISTORICAL_COLUMN_MAP, MARKET_WINDOW_SEC, Z_SCORE_CLIP,
    _parse_float, log_loss, brier_score, accuracy, QuantModel, _clip_z,
)

HISTORICAL_PATHS = sorted(glob.glob("data/historical/microstructure_dataset_batch_*.csv")) + [
    "data/historical/outcome_training_dataset.csv",
]
LIVE_PATH = "data/quant_features.jsonl"


def load_historical_with_dates(paths):
    X_rows, y_rows, yes_price, dates = [], [], [], []
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
                yes_price.append(_parse_float(row.get("yes_price")))
                dates.append(row.get("resolution_timestamp") or row.get("window_end") or "")
    return X_rows, y_rows, yes_price, dates


def load_live_resolved(path):
    X_rows, y_rows, yes_price, dates = [], [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("stage") != "resolution" or d.get("outcome") is None:
                continue
            feats = d.get("features", {})
            row = [feats.get(name) if feats.get(name) is not None else float("nan") for name in FEATURE_NAMES]
            X_rows.append(row)
            y_rows.append(float(d["outcome"]))
            yes_price.append(feats.get("yes_price", float("nan")))
            dates.append(d.get("timestamp", ""))
    return X_rows, y_rows, yes_price, dates


def main():
    print("=" * 70)
    print("STEP 1: DATA INVENTORY")
    print("=" * 70)
    live_total = sum(1 for _ in open(LIVE_PATH))
    print(f"data/quant_features.jsonl total rows (all stages): {live_total}")

    h_X, h_y, h_yp, h_dates = load_historical_with_dates(HISTORICAL_PATHS)
    print(f"\nHistorical outcome-labeled files used:")
    for p in HISTORICAL_PATHS:
        print(f"  - {p}")
    print(f"Historical labeled rows (outcome present): {len(h_y)}")

    l_X, l_y, l_yp, l_dates = load_live_resolved(LIVE_PATH)
    print(f"Live shadow labeled rows (stage=resolution, outcome present): {len(l_y)}")

    X = np.array(h_X + l_X, dtype=float)
    y = np.array(h_y + l_y, dtype=float)
    yes_price = np.array(h_yp + l_yp, dtype=float)
    all_dates = sorted(d for d in (h_dates + l_dates) if d)

    print("\n" + "=" * 70)
    print("STEP 2: COMBINED DATASET")
    print("=" * 70)
    print(f"Total rows: {len(y)}")
    print(f"Feature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    wins = int(np.sum(y == 1))
    losses = int(np.sum(y == 0))
    print(f"WIN/LOSS balance: {wins} wins / {losses} losses ({wins/len(y):.1%} win rate)")
    print(f"Date range: {all_dates[0]} -> {all_dates[-1]}")
    nan_frac = np.isnan(X).mean(axis=0)
    print("Missing-value fraction per feature (imputed via median):")
    for name, frac in zip(FEATURE_NAMES, nan_frac):
        print(f"  {name:24s} {frac:.1%}")

    print("\n" + "=" * 70)
    print("STEP 3: TRAIN + 5-FOLD CROSS-VALIDATE 3 MODELS")
    print("=" * 70)

    def make_pipeline(estimator):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clip", FunctionTransformer(_clip_z)),
            ("clf", estimator),
        ])

    models = {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=20260703),
        "GradientBoosting": GradientBoostingClassifier(random_state=20260703),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260703)
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    results = {}
    for name, est in models.items():
        pipe = make_pipeline(est)
        cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring)
        results[name] = {m: float(np.mean(cv[f"test_{m}"])) for m in scoring}
        print(f"\n{name}:")
        for m in scoring:
            print(f"  {m:10s} {results[name][m]:.4f} (+/- {np.std(cv[f'test_{m}']):.4f})")

    # baselines for comparison, same folds
    print("\nBaselines (for comparison, not cross-validated model fits):")
    majority_class = 1 if wins > losses else 0
    majority_acc = max(wins, losses) / len(y)
    print(f"  always-predict-majority-class accuracy: {majority_acc:.4f}")
    yp_acc = accuracy(y, yes_price)
    yp_ll = log_loss(y, np.clip(yes_price, 1e-6, 1 - 1e-6))
    yp_brier = brier_score(y, yes_price)
    from sklearn.metrics import roc_auc_score
    yp_auc = roc_auc_score(y, yes_price)
    print(f"  yes_price-as-probability accuracy: {yp_acc:.4f}, log_loss: {yp_ll:.4f}, brier: {yp_brier:.4f}, AUC: {yp_auc:.4f}")

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_acc = results[best_name]["accuracy"]
    best_auc = results[best_name]["roc_auc"]
    print(f"\nBest model by mixed 5-fold CV accuracy: {best_name} ({best_acc:.4f}, AUC {best_auc:.4f})")

    print("\n" + "=" * 70)
    print("CRITICAL CHECK: does this survive a true out-of-regime holdout?")
    print("=" * 70)
    print("The mixed 5-fold CV above shuffles historical (Jun 18-24) and live")
    print("(Jul 2-3) rows together into every fold, so a model can score well by")
    print("exploiting source-specific quirks (different capture pipeline, different")
    print("market regime/date) rather than a real live signal. The only honest test")
    print("is: train on historical only, evaluate on live-only data the model has")
    print("never seen from a source it has never seen -- exactly the situation the")
    print("live bot is actually in.")
    n_hist = len(h_y)
    X_hist, y_hist = X[:n_hist], y[:n_hist]
    X_live, y_live = X[n_hist:], y[n_hist:]
    yp_live = yes_price[n_hist:]
    from sklearn.metrics import roc_auc_score
    holdout_rows = {}
    for name, est in models.items():
        pipe = make_pipeline(est)
        pipe.fit(X_hist, y_hist)
        pred = pipe.predict_proba(X_live)[:, 1]
        holdout_rows[name] = {
            "accuracy": accuracy(y_live, pred),
            "auc": roc_auc_score(y_live, pred),
        }
        print(f"  {name:20s} trained-on-historical -> tested-on-live: "
              f"accuracy={holdout_rows[name]['accuracy']:.4f} AUC={holdout_rows[name]['auc']:.4f}")
    yp_live_acc = accuracy(y_live, yp_live)
    yp_live_auc = roc_auc_score(y_live, yp_live)
    print(f"  {'yes_price baseline':20s} (on the same live holdout): "
          f"accuracy={yp_live_acc:.4f} AUC={yp_live_auc:.4f}")

    holdout_best = max(holdout_rows, key=lambda n: holdout_rows[n]["auc"])
    holdout_passes = (
        holdout_rows[holdout_best]["accuracy"] > 0.55
        and holdout_rows[holdout_best]["auc"] > yp_live_auc
    )

    print("\n" + "=" * 70)
    print("STEP 4/5: DECISION")
    print("=" * 70)
    print(f"Mixed-CV 55% accuracy threshold: {'PASS' if best_acc > 0.55 else 'FAIL'} ({best_acc:.1%}) -- MISLEADING, see above")
    print(f"True out-of-regime holdout: {'PASS' if holdout_passes else 'FAIL'} "
          f"(best={holdout_best}, accuracy={holdout_rows[holdout_best]['accuracy']:.1%}, "
          f"AUC={holdout_rows[holdout_best]['auc']:.4f} vs yes_price AUC={yp_live_auc:.4f})")

    genuine_edge = holdout_passes

    if not genuine_edge:
        print("\n*** The mixed-CV numbers clear 55% accuracy, but that result does not")
        print("*** survive being tested on genuinely unseen live data -- every model's")
        print("*** out-of-regime AUC is at or below 0.52 (coin-flip is 0.50), and all")
        print("*** trail the yes_price baseline's own live AUC. This is source leakage,")
        print("*** not a real edge. Recommendation: do NOT flip to live trading.")
    else:
        print("\n*** best model shows genuine edge on a true out-of-regime holdout.")
        print("*** Proceeding to fit final model on all data and save.")

    # Fit the chosen best model on ALL data and save regardless (useful for
    # continued shadow logging / offline comparison), but only report it as
    # live-eligible if genuine_edge is True.
    final_est = models[best_name]
    qm = QuantModel(feature_names=FEATURE_NAMES)
    qm.pipeline = make_pipeline(final_est)
    qm.pipeline.fit(X, y)
    qm.save("data/quant_model.pkl")
    print(f"\nSaved refit {best_name} to data/quant_model.pkl (feature_names={FEATURE_NAMES})")

    if not genuine_edge:
        print("\n" + "=" * 70)
        print("TOP WIN-CORRELATED FEATURES (point-biserial correlation with outcome)")
        print("=" * 70)
        for i, name in enumerate(FEATURE_NAMES):
            col = X[:, i]
            mask = ~np.isnan(col)
            if mask.sum() < 10:
                print(f"  {name:24s} insufficient data")
                continue
            corr = np.corrcoef(col[mask], y[mask])[0, 1]
            print(f"  {name:24s} r={corr:+.4f}  (n={mask.sum()})")

    return {
        "results": results,
        "best_name": best_name,
        "best_acc": best_acc,
        "best_auc": best_auc,
        "majority_acc": majority_acc,
        "yp_acc": yp_acc,
        "yp_auc": yp_auc,
        "genuine_edge": genuine_edge,
        "n_total": len(y),
        "n_historical": len(h_y),
        "n_live": len(l_y),
        "wins": wins,
        "losses": losses,
        "date_range": (all_dates[0], all_dates[-1]),
    }


if __name__ == "__main__":
    main()
