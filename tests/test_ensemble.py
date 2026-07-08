from core.ensemble import Ensemble


class StubModel:
    def __init__(self, p, n_examples):
        self.p = p
        self.n_examples = n_examples

    def predict_up(self, features):
        return self.p


def _ensemble(momentum_p, volume_p, n_examples=30):
    return Ensemble(StubModel(momentum_p, n_examples), StubModel(volume_p, n_examples))


def test_no_trade_below_min_training_examples():
    ensemble = _ensemble(0.9, 0.9, n_examples=5)
    result = ensemble.decide({"yes_price": 0.5, "seconds_remaining": 100}, fear_greed=50, hour_utc=12)
    assert result["decision"] is None
    assert "warmup" in result["reason"]


def test_no_trade_in_uncertainty_zone():
    # score lands exactly at 0.5 -- neither > 0.55 nor < 0.45
    ensemble = _ensemble(0.5, 0.5)
    result = ensemble.decide({"yes_price": 0.5}, fear_greed=50, hour_utc=12)
    assert result["final_score"] == 0.5
    assert result["decision"] is None


def test_yes_trade_when_score_high_and_price_in_band():
    ensemble = _ensemble(0.9, 0.9)
    result = ensemble.decide({"yes_price": 0.5}, fear_greed=50, hour_utc=12)
    assert result["final_score"] > 0.55
    assert result["decision"] == "YES"


def test_no_side_trade_when_score_low_and_price_in_band():
    ensemble = _ensemble(0.1, 0.1)
    result = ensemble.decide({"yes_price": 0.45}, fear_greed=50, hour_utc=12)
    assert result["final_score"] < 0.45
    assert result["decision"] == "NO"


def test_no_side_blocked_when_price_out_of_band():
    ensemble = _ensemble(0.1, 0.1)
    result = ensemble.decide({"yes_price": 0.60}, fear_greed=50, hour_utc=12)
    assert result["final_score"] < 0.45
    assert result["decision"] is None


def test_warmup_yes_trade_on_positive_momentum_and_price_in_band():
    ensemble = _ensemble(0.5, 0.5, n_examples=0)
    result = ensemble.decide(
        {"yes_price": 0.5, "price_momentum_5m": 3.0, "seconds_remaining": 100}, fear_greed=50, hour_utc=12
    )
    assert result["mode"] == "warmup"
    assert result["decision"] == "YES"
    assert result["final_score"] is None


def test_warmup_no_trade_on_negative_momentum_and_price_in_band():
    ensemble = _ensemble(0.5, 0.5, n_examples=0)
    result = ensemble.decide(
        {"yes_price": 0.45, "price_momentum_5m": -3.0, "seconds_remaining": 100}, fear_greed=50, hour_utc=12
    )
    assert result["mode"] == "warmup"
    assert result["decision"] == "NO"


def test_warmup_no_decision_when_momentum_flat_or_price_out_of_band():
    ensemble = _ensemble(0.5, 0.5, n_examples=0)
    flat = ensemble.decide(
        {"yes_price": 0.5, "price_momentum_5m": 0.0, "seconds_remaining": 100}, fear_greed=50, hour_utc=12
    )
    assert flat["decision"] is None
    out_of_band = ensemble.decide(
        {"yes_price": 0.9, "price_momentum_5m": 3.0, "seconds_remaining": 100}, fear_greed=50, hour_utc=12
    )
    assert out_of_band["decision"] is None


def test_warmup_blocked_when_seconds_remaining_below_60():
    ensemble = _ensemble(0.5, 0.5, n_examples=0)
    result = ensemble.decide(
        {"yes_price": 0.5, "price_momentum_5m": 3.0, "seconds_remaining": 30}, fear_greed=50, hour_utc=12
    )
    assert result["mode"] == "warmup"
    assert result["decision"] is None
    assert result["reason"] == "too_late"
