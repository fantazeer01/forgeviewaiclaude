import csv

import numpy as np
import pytest

from core.quant_model import (
    FEATURE_NAMES,
    QuantModel,
    accuracy,
    brier_score,
    load_historical_dataset,
    log_loss,
    train_and_evaluate,
    train_test_split_indices,
)


def write_csv(path, rows):
    fieldnames = ["outcome", "yes_price", "no_price", "repricing_velocity",
                  "repricing_acceleration", "book_imbalance", "bid_ask_spread",
                  "spread_compression", "seconds_to_expiry"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_row(outcome=1, yes_price=0.5, seconds_to_expiry=200.0, **overrides):
    row = {
        "outcome": outcome, "yes_price": yes_price, "no_price": 1 - yes_price,
        "repricing_velocity": 0.001, "repricing_acceleration": 0.0001,
        "book_imbalance": 0.1, "bid_ask_spread": 0.02, "spread_compression": 0.0,
        "seconds_to_expiry": seconds_to_expiry,
    }
    row.update(overrides)
    return row


def test_log_loss_perfect_predictions_near_zero():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.999, 0.001, 0.999])
    assert log_loss(y, p) < 0.01


def test_log_loss_penalizes_confident_wrong_predictions():
    y = np.array([1.0])
    confident_right = log_loss(y, np.array([0.9]))
    confident_wrong = log_loss(y, np.array([0.1]))
    assert confident_wrong > confident_right


def test_brier_score_zero_for_perfect_predictions():
    y = np.array([1.0, 0.0])
    p = np.array([1.0, 0.0])
    assert brier_score(y, p) == 0.0


def test_brier_score_matches_squared_error():
    y = np.array([1.0])
    p = np.array([0.7])
    assert brier_score(y, p) == pytest.approx(0.09)


def test_accuracy_counts_correct_threshold_predictions():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.6, 0.4, 0.3, 0.9])
    assert accuracy(y, p) == 0.5


def test_train_test_split_indices_deterministic_for_same_seed():
    train1, test1 = train_test_split_indices(100, seed=42)
    train2, test2 = train_test_split_indices(100, seed=42)
    assert list(train1) == list(train2)
    assert list(test1) == list(test2)


def test_train_test_split_indices_no_overlap():
    train, test = train_test_split_indices(100, test_fraction=0.3)
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == 100


def test_load_historical_dataset_maps_columns_to_canonical_names(tmp_path):
    path = tmp_path / "data.csv"
    write_csv(path, [make_row(outcome=1, yes_price=0.6, seconds_to_expiry=150.0)])
    X, y, yes_price = load_historical_dataset([str(path)])
    assert X.shape == (1, len(FEATURE_NAMES))
    assert y[0] == 1.0
    assert yes_price[0] == 0.6
    idx = FEATURE_NAMES.index("time_remaining_pct")
    assert X[0, idx] == pytest.approx(150.0 / 300.0)


def test_load_historical_dataset_skips_rows_without_outcome(tmp_path):
    path = tmp_path / "data.csv"
    write_csv(path, [make_row(outcome=""), make_row(outcome=1)])
    X, y, _ = load_historical_dataset([str(path)])
    assert len(y) == 1


def test_load_historical_dataset_handles_blank_feature_values(tmp_path):
    path = tmp_path / "data.csv"
    write_csv(path, [make_row(outcome=1, repricing_velocity="")])
    X, y, _ = load_historical_dataset([str(path)])
    idx = FEATURE_NAMES.index("price_velocity")
    assert np.isnan(X[0, idx])


def test_load_historical_dataset_combines_multiple_files(tmp_path):
    path1 = tmp_path / "a.csv"
    path2 = tmp_path / "b.csv"
    write_csv(path1, [make_row(outcome=1)])
    write_csv(path2, [make_row(outcome=0), make_row(outcome=1)])
    X, y, _ = load_historical_dataset([str(path1), str(path2)])
    assert len(y) == 3


def _make_separable_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    yes_price = rng.uniform(0.1, 0.9, n)
    y = (yes_price + rng.normal(0, 0.05, n) > 0.5).astype(float)
    X = np.column_stack([
        yes_price, 1 - yes_price,
        rng.normal(0, 0.01, n), rng.normal(0, 0.005, n),
        rng.normal(0, 0.1, n), rng.uniform(0.01, 0.05, n),
        rng.normal(0, 0.01, n), rng.uniform(0.2, 0.9, n),
    ])
    return X, y


def test_quant_model_fit_and_predict_proba_shape():
    X, y = _make_separable_dataset()
    model = QuantModel().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_quant_model_learns_a_separable_signal():
    X, y = _make_separable_dataset(n=400)
    split = len(y) // 2
    model = QuantModel().fit(X[:split], y[:split])
    proba = model.predict_proba(X[split:])
    assert accuracy(y[split:], proba) > 0.7


def test_quant_model_predict_proba_one_maps_feature_dict():
    X, y = _make_separable_dataset()
    model = QuantModel().fit(X, y)
    features = dict(zip(FEATURE_NAMES, X[0]))
    proba = model.predict_proba_one(features)
    assert proba is not None
    assert 0.0 <= proba <= 1.0


def test_quant_model_predict_proba_one_returns_none_when_untrained():
    model = QuantModel()
    assert model.predict_proba_one({name: 0.5 for name in FEATURE_NAMES}) is None


def test_quant_model_predict_proba_one_handles_missing_features_gracefully():
    X, y = _make_separable_dataset()
    model = QuantModel().fit(X, y)
    partial_features = {"yes_price": 0.5}
    proba = model.predict_proba_one(partial_features)
    assert proba is not None
    assert 0.0 <= proba <= 1.0


def test_quant_model_handles_extreme_out_of_distribution_input_without_blowing_up():
    # a feature with near-zero training variance should not saturate predict_proba
    # to exactly 0.0 or 1.0 when a live value falls far outside that narrow range
    X, y = _make_separable_dataset(n=200)
    X[:, -1] = 0.8 + np.random.default_rng(1).normal(0, 0.0002, len(y))  # tiny variance
    model = QuantModel().fit(X, y)
    features = dict(zip(FEATURE_NAMES, X[0]))
    features["time_remaining_pct"] = 0.1  # far outside the ~0.8 training range
    proba = model.predict_proba_one(features)
    assert proba is not None
    assert 0.0 < proba < 1.0


def test_quant_model_save_and_load_roundtrip(tmp_path):
    X, y = _make_separable_dataset()
    model = QuantModel().fit(X, y)
    original_proba = model.predict_proba(X)
    path = str(tmp_path / "model.pkl")
    model.save(path)
    loaded = QuantModel.load(path)
    assert loaded is not None
    loaded_proba = loaded.predict_proba(X)
    assert np.allclose(original_proba, loaded_proba)


def test_quant_model_load_returns_none_for_missing_file(tmp_path):
    assert QuantModel.load(str(tmp_path / "nonexistent.pkl")) is None


def test_train_and_evaluate_returns_model_and_baseline_comparison(tmp_path):
    path = tmp_path / "data.csv"
    rows = []
    rng = np.random.default_rng(7)
    for _ in range(120):
        yes_price = float(rng.uniform(0.1, 0.9))
        outcome = int(yes_price + rng.normal(0, 0.05) > 0.5)
        rows.append(make_row(outcome=outcome, yes_price=yes_price))
    write_csv(path, rows)

    out = train_and_evaluate([str(path)])
    results = out["results"]
    assert results["n_total"] == 120
    assert results["n_train"] + results["n_test"] == 120
    assert "log_loss" in results["model"]
    assert "log_loss" in results["yes_price_baseline"]
    assert isinstance(results["model_beats_yes_price"], bool)
    assert out["model"].predict_proba_one(dict(zip(FEATURE_NAMES, [0.5] * len(FEATURE_NAMES)))) is not None
