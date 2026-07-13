from core.ensemble import Ensemble


class StubModel:
    def __init__(self, p, n_examples):
        self.p = p
        self.n_examples = n_examples

    def predict_up(self, features):
        return self.p


class ExplodingModel:
    """Raises if ever consulted -- used to prove decide()/_fair_value_decide()
    never touch the momentum/volume models at all."""

    n_examples = 999

    def predict_up(self, features):
        raise AssertionError("model should not be consulted by the fair-value strategy")


def _ensemble(momentum_p, volume_p, n_examples=30):
    return Ensemble(StubModel(momentum_p, n_examples), StubModel(volume_p, n_examples))


def _asset_ensemble(momentum_p, volume_p, asset, n_examples=30):
    return Ensemble(StubModel(momentum_p, n_examples), StubModel(volume_p, n_examples), asset=asset)


# ---- _live_decide() / _warmup_decide() are dormant (decide() no longer
# routes to them as of the 2026-07-13 fair-value strategy) but are kept in
# the codebase for possible future re-enabling -- these tests call them
# directly to confirm that logic itself still works correctly.

def test_dormant_warmup_reports_disabled():
    ensemble = _ensemble(0.9, 0.9, n_examples=5)
    result = ensemble._warmup_decide({"yes_price": 0.5, "seconds_remaining": 150}, training_examples=5)
    assert result["decision"] is None
    assert result["mode"] == "warmup"
    assert "disabled" in result["reason"]


def test_dormant_live_no_trade_in_uncertainty_zone():
    ensemble = _ensemble(0.5, 0.5)
    result = ensemble._live_decide({"yes_price": 0.5}, fear_greed=50, hour_utc=12)
    assert result["final_score"] == 0.5
    assert result["decision"] is None


def test_dormant_live_yes_trade_when_score_high_and_price_in_band():
    ensemble = _ensemble(0.9, 0.9)
    result = ensemble._live_decide({"yes_price": 0.5}, fear_greed=50, hour_utc=12)
    assert result["final_score"] > 0.55
    assert result["decision"] == "YES"


def test_dormant_live_no_side_trade_when_score_low_and_price_in_band():
    ensemble = _asset_ensemble(0.1, 0.1, asset="SOL")
    result = ensemble._live_decide({"yes_price": 0.45}, fear_greed=50, hour_utc=12)
    assert result["final_score"] < 0.45
    assert result["decision"] == "NO"


def test_dormant_live_no_side_blocked_for_btc_and_eth():
    for asset in ("BTC", "ETH"):
        ensemble = _asset_ensemble(0.1, 0.1, asset=asset)
        result = ensemble._live_decide({"yes_price": 0.45}, fear_greed=50, hour_utc=12)
        assert result["decision"] is None, f"{asset} should never open a NO trade"


def test_dormant_warmup_never_trades_regardless_of_signal():
    cases = [
        {"yes_price": 0.52, "price_momentum_5m": 3.0, "seconds_remaining": 150},
        {"yes_price": 0.52, "price_momentum_5m": -3.0, "seconds_remaining": 150},
        {"yes_price": 0.60, "price_momentum_5m": 3.0, "seconds_remaining": 150},
    ]
    for asset in ("BTC", "ETH", "SOL"):
        ensemble = _asset_ensemble(0.5, 0.5, asset=asset, n_examples=0)
        for features in cases:
            result = ensemble._warmup_decide(features, training_examples=0)
            assert result["mode"] == "warmup"
            assert result["decision"] is None, f"{asset} {features} should never trade in warmup"


# ---- _fair_value_decide() / decide() -- the live, active strategy as of
# 2026-07-13. decide() is the real entry point bot.py calls, so these
# exercise it directly (not the private method), proving the wiring works.

def _fv_ensemble(asset="BTC"):
    # ExplodingModel proves the models are never consulted by this strategy.
    return Ensemble(ExplodingModel(), ExplodingModel(), asset=asset)


def test_fair_value_yes_entry_when_price_cheap_and_early():
    ensemble = _fv_ensemble()
    result = ensemble.decide({"yes_price": 0.45, "seconds_remaining": 250}, fear_greed=50, hour_utc=12)
    assert result["mode"] == "fair_value"
    assert result["decision"] == "YES"


def test_fair_value_no_entry_when_price_cheap_and_early():
    ensemble = _fv_ensemble()
    result = ensemble.decide({"yes_price": 0.55, "seconds_remaining": 250}, fear_greed=50, hour_utc=12)
    assert result["mode"] == "fair_value"
    assert result["decision"] == "NO"


def test_fair_value_no_entry_in_neutral_zone():
    ensemble = _fv_ensemble()
    result = ensemble.decide({"yes_price": 0.50, "seconds_remaining": 250}, fear_greed=50, hour_utc=12)
    assert result["decision"] is None
    assert result["reason"] == "price_near_fair_value"


def test_fair_value_no_entry_when_too_late():
    ensemble = _fv_ensemble()
    result = ensemble.decide({"yes_price": 0.45, "seconds_remaining": 100}, fear_greed=50, hour_utc=12)
    assert result["decision"] is None
    assert result["reason"] == "too_late_for_fv"


def test_fair_value_no_entry_when_price_too_far_from_fair_value():
    ensemble = _fv_ensemble()
    result = ensemble.decide({"yes_price": 0.40, "seconds_remaining": 250}, fear_greed=50, hour_utc=12)
    assert result["decision"] is None


def test_fair_value_never_consults_models():
    # ExplodingModel would raise AssertionError if predict_up() were called --
    # decide() completing without error across all branches proves the models
    # are genuinely bypassed, not just coincidentally unused in one branch.
    ensemble = _fv_ensemble()
    for features in [
        {"yes_price": 0.45, "seconds_remaining": 250},
        {"yes_price": 0.55, "seconds_remaining": 250},
        {"yes_price": 0.50, "seconds_remaining": 250},
        {"yes_price": 0.45, "seconds_remaining": 100},
    ]:
        ensemble.decide(features, fear_greed=50, hour_utc=12)
