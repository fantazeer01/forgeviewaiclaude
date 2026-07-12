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
    result = ensemble.decide({"yes_price": 0.5, "seconds_remaining": 150}, fear_greed=50, hour_utc=12)
    assert result["decision"] is None
    assert "warmup" in result["reason"]
    assert result["mode"] == "warmup"


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


def _asset_ensemble(momentum_p, volume_p, asset, n_examples=30):
    return Ensemble(StubModel(momentum_p, n_examples), StubModel(volume_p, n_examples), asset=asset)


def test_no_side_trade_when_score_low_and_price_in_band():
    # NO is only allowed for SOL (2026-07-12: BTC/ETH NO ran 33.3% win rate).
    ensemble = _asset_ensemble(0.1, 0.1, asset="SOL")
    result = ensemble.decide({"yes_price": 0.45}, fear_greed=50, hour_utc=12)
    assert result["final_score"] < 0.45
    assert result["decision"] == "NO"


def test_no_side_blocked_when_price_out_of_band():
    ensemble = _asset_ensemble(0.1, 0.1, asset="SOL")
    result = ensemble.decide({"yes_price": 0.60}, fear_greed=50, hour_utc=12)
    assert result["final_score"] < 0.45
    assert result["decision"] is None


def test_no_side_blocked_for_btc_and_eth():
    # Same score/price that fires NO for SOL must never fire for BTC/ETH.
    for asset in ("BTC", "ETH"):
        ensemble = _asset_ensemble(0.1, 0.1, asset=asset)
        result = ensemble.decide({"yes_price": 0.45}, fear_greed=50, hour_utc=12)
        assert result["final_score"] < 0.45
        assert result["decision"] is None, f"{asset} should never open a NO trade"


def test_warmup_never_trades_regardless_of_signal():
    # Warmup trading is fully disabled (2026-07-12): shadow learning in
    # bot.py is now the only path to accumulate examples -- no combination
    # of momentum/price/timing/asset should ever produce a decision here.
    cases = [
        {"yes_price": 0.52, "price_momentum_5m": 3.0, "seconds_remaining": 150},   # would have fired pre-fix
        {"yes_price": 0.52, "price_momentum_5m": -3.0, "seconds_remaining": 150},
        {"yes_price": 0.52, "price_momentum_5m": 0.0, "seconds_remaining": 150},
        {"yes_price": 0.60, "price_momentum_5m": 3.0, "seconds_remaining": 150},
        {"yes_price": 0.52, "price_momentum_5m": 3.0, "seconds_remaining": 5},
        {"yes_price": 0.52, "price_momentum_5m": 3.0, "seconds_remaining": 299},
    ]
    for asset in ("BTC", "ETH", "SOL"):
        ensemble = _asset_ensemble(0.5, 0.5, asset=asset, n_examples=0)
        for features in cases:
            result = ensemble.decide(features, fear_greed=50, hour_utc=12)
            assert result["mode"] == "warmup"
            assert result["decision"] is None, f"{asset} {features} should never trade in warmup"
            assert result["final_score"] is None
