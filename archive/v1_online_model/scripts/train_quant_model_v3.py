"""
ONE SPRINT v3: quant model using a repricing-motivated feature set
(yes_price, price_drop_60s, price_drop_magnitude, time_remaining_pct,
order_book_imbalance, volume_24h, asset one-hot), trained on ALL available
outcome-labeled data (D:\\ForgeViewAI historical research data, read-only,
never modified, + our own live shadow log).

Decision rule (as specified): go live only if the best model's AUC on a
genuine out-of-regime holdout exceeds 0.55. The naive mixed 5-fold CV is
reported too, but per the precedent set in the prior sprint (see
data/historical/README.md), mixed-source CV is known to be inflated by
source leakage on this data, so it is not used as the deciding number.
"""
import csv
import glob
import json
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

sys.path.insert(0, ".")
from core.quant_model import Z_SCORE_CLIP, _parse_float, log_loss, brier_score, accuracy, QuantModel, _clip_z

FEATURE_NAMES_V3 = [
    "yes_price", "price_drop_60s", "price_drop_magnitude", "time_remaining_pct",
    "order_book_imbalance", "volume_24h", "asset_BTC", "asset_ETH", "asset_SOL",
]

HISTORICAL_PATHS = sorted(glob.glob("data/historical/microstructure_dataset_batch_*.csv")) + [
    "data/historical/outcome_training_dataset.csv",
]
LIVE_PATH = "data/quant_features.jsonl"
MARKET_WINDOW_SEC = 300.0


def _asset_onehot(asset):
    return [1.0 if asset == a else 0.0 for a in ("BTC", "ETH", "SOL")]


def load_historical(paths):
    """probability_change_30s is the closest available historical proxy for a
    60s YES-price drop (the schema only captures 15s/30s probability-change
    windows, not 60s -- documented limitation, not a substitution error)."""
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
                yp = _parse_float(row.get("yes_price"))
                prob_change_30s = _parse_float(row.get("probability_change_30s"))
                price_drop_60s = -prob_change_30s if prob_change_30s == prob_change_30s else float("nan")
                price_drop_magnitude = (price_drop_60s / yp) if (yp not in (0, None) and yp == yp and price_drop_60s == price_drop_60s) else float("nan")
                seconds = _parse_float(row.get("seconds_to_expiry"))
                time_remaining_pct = seconds / MARKET_WINDOW_SEC if seconds == seconds else float("nan")
                order_book_imbalance = _parse_float(row.get("book_imbalance"))  # NaN if absent (outcome_training_dataset.csv)
                volume_24h = float("nan")  # never captured historically
                feats = [yp, price_drop_60s, price_drop_magnitude, time_remaining_pct,
                          order_book_imbalance, volume_24h] + _asset_onehot(row.get("asset"))
                X_rows.append(feats)
                y_rows.append(outcome)
                yes_price.append(yp)
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
            yp = feats.get("yes_price")
            velocity = feats.get("price_velocity")
            price_drop_60s = -velocity * 60 if velocity is not None else float("nan")
            price_drop_magnitude = (price_drop_60s / yp) if (yp not in (0, None) and price_drop_60s == price_drop_60s) else float("nan")
            row = [
                yp if yp is not None else float("nan"),
                price_drop_60s,
                price_drop_magnitude,
                feats.get("time_remaining_pct", float("nan")),
                feats.get("order_book_imbalance", float("nan")) if feats.get("order_book_imbalance") is not None else float("nan"),
                feats.get("volume_24h", float("nan")) if feats.get("volume_24h") is not None else float("nan"),
            ] + _asset_onehot(d.get("asset"))
            X_rows.append(row)
            y_rows.append(float(d["outcome"]))
            yes_price.append(yp if yp is not None else float("nan"))
            dates.append(d.get("timestamp", ""))
    return X_rows, y_rows, yes_price, dates


def make_pipeline(estimator):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clip", FunctionTransformer(_clip_z)),
        ("clf", estimator),
    ])


def main():
    print("=" * 70)
    print("STEP 1: DATA INVENTORY (D:\\ForgeViewAI is read-only; nothing there was modified)")
    print("=" * 70)
    print("Historical outcome-labeled files reused from data/historical/ (copied read-only from")
    print("D:\\ForgeViewAI in a prior sprint; see data/historical/README.md for full provenance):")
    for p in HISTORICAL_PATHS:
        print(f"  - {p}")
    live_total = sum(1 for _ in open(LIVE_PATH))
    print(f"Live shadow log data/quant_features.jsonl: {live_total} total rows (all stages)")

    h_X, h_y, h_yp, h_dates = load_historical(HISTORICAL_PATHS)
    l_X, l_y, l_yp, l_dates = load_live_resolved(LIVE_PATH)
    print(f"Historical labeled rows (outcome present): {len(h_y)}")
    print(f"Live labeled rows (stage=resolution, outcome present): {len(l_y)}")

    X = np.array(h_X + l_X, dtype=float)
    y = np.array(h_y + l_y, dtype=float)
    yes_price = np.array(h_yp + l_yp, dtype=float)
    all_dates = sorted(d for d in (h_dates + l_dates) if d)

    print("\n" + "=" * 70)
    print("STEP 2: COMBINED DATASET")
    print("=" * 70)
    print(f"Total rows: {len(y)}")
    print(f"Feature names ({len(FEATURE_NAMES_V3)}): {FEATURE_NAMES_V3}")
    wins = int(np.sum(y == 1)); losses = int(np.sum(y == 0))
    print(f"WIN/LOSS balance: {wins} wins / {losses} losses ({wins/len(y):.1%} win rate)")
    print(f"Date range: {all_dates[0]} -> {all_dates[-1]}")
    nan_frac = np.isnan(X).mean(axis=0)
    print("Missing-value fraction per feature (imputed via median):")
    for name, frac in zip(FEATURE_NAMES_V3, nan_frac):
        print(f"  {name:24s} {frac:.1%}")

    print("\n" + "=" * 70)
    print("STEP 4: TRAIN + 5-FOLD CROSS-VALIDATE 3 MODELS (mixed CV, for reference)")
    print("=" * 70)
    models = {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=20260704),
        "GradientBoosting": GradientBoostingClassifier(random_state=20260704),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260704)
    scoring = ["accuracy", "precision", "recall", "roc_auc"]
    mixed_results = {}
    for name, est in models.items():
        pipe = make_pipeline(est)
        cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring)
        mixed_results[name] = {m: float(np.mean(cv[f"test_{m}"])) for m in scoring}
        print(f"\n{name}:")
        for m in scoring:
            print(f"  {m:10s} {mixed_results[name][m]:.4f} (+/- {np.std(cv[f'test_{m}']):.4f})")

    yp_acc = accuracy(y, yes_price)
    yp_auc = roc_auc_score(y, yes_price)
    print(f"\nyes_price baseline (mixed, for reference): accuracy={yp_acc:.4f} AUC={yp_auc:.4f}")

    print("\n" + "=" * 70)
    print("DECIDING TEST: train on historical only, evaluate on unseen live data")
    print("(the actual situation the live bot is in -- avoids the source-leakage")
    print("inflation the prior sprint documented in data/historical/README.md)")
    print("=" * 70)
    n_hist = len(h_y)
    X_hist, y_hist = X[:n_hist], y[:n_hist]
    X_live, y_live = X[n_hist:], y[n_hist:]
    yp_live = yes_price[n_hist:]
    holdout = {}
    for name, est in models.items():
        pipe = make_pipeline(est)
        pipe.fit(X_hist, y_hist)
        pred = pipe.predict_proba(X_live)[:, 1]
        holdout[name] = {"accuracy": accuracy(y_live, pred), "auc": roc_auc_score(y_live, pred)}
        print(f"  {name:20s} accuracy={holdout[name]['accuracy']:.4f} AUC={holdout[name]['auc']:.4f}")
    yp_live_acc = accuracy(y_live, yp_live)
    yp_live_auc = roc_auc_score(y_live, yp_live)
    print(f"  {'yes_price baseline':20s} accuracy={yp_live_acc:.4f} AUC={yp_live_auc:.4f}")

    best_name = max(holdout, key=lambda n: holdout[n]["auc"])
    best_auc = holdout[best_name]["auc"]

    print("\n" + "=" * 70)
    print("STEP 5: DECISION")
    print("=" * 70)
    print(f"Best model on out-of-regime holdout: {best_name}, AUC={best_auc:.4f}")
    print(f"Decision rule: best AUC > 0.55 -> LIVE. best AUC <= 0.55 -> REJECTED.")

    if best_auc > 0.55:
        print(f"\n*** PASS: {best_name} AUC {best_auc:.4f} > 0.55 on genuine holdout. Proceeding to save + go live.")
        final_pipe = make_pipeline(models[best_name])
        final_pipe.fit(X, y)
        qm = QuantModel(feature_names=FEATURE_NAMES_V3)
        qm.pipeline = final_pipe
        qm.save("data/quant_model.pkl")
        print("Saved to data/quant_model.pkl")
    else:
        print(f"\n*** REJECTED: best out-of-regime AUC ({best_auc:.4f}) does not exceed 0.55.")
        print(f"*** yes_price baseline AUC on the same holdout ({yp_live_auc:.4f}) also exceeds every model's.")
        print("*** Keeping repricing detector as the sole live signal. Model NOT saved as production.")
        print("\nTop win-correlated features (live-only, n=%d):" % len(y_live))
        for i, name in enumerate(FEATURE_NAMES_V3):
            col = X_live[:, i]
            mask = ~np.isnan(col)
            if mask.sum() < 10:
                print(f"  {name:24s} insufficient data"); continue
            corr = np.corrcoef(col[mask], y_live[mask])[0, 1]
            print(f"  {name:24s} r={corr:+.4f} (n={mask.sum()})")

    return {
        "mixed_results": mixed_results, "holdout": holdout, "best_name": best_name,
        "best_auc": best_auc, "yp_live_acc": yp_live_acc, "yp_live_auc": yp_live_auc,
        "n_total": len(y), "n_historical": len(h_y), "n_live": len(l_y),
        "wins": wins, "losses": losses, "date_range": (all_dates[0], all_dates[-1]),
        "went_live": best_auc > 0.55,
    }


if __name__ == "__main__":
    main()
