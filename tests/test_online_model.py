import pytest

from core.online_model import OnlineQuantModel
from core.repricing_detector import RepricingSignal


def make_features(**overrides):
    base = {
        "yes_price": 0.4, "no_price": 0.6, "bid_ask_spread": 0.02,
        "order_book_imbalance": 0.1, "price_momentum_30s": -0.02,
        "price_momentum_60s": -0.05, "volume_24h": 500.0,
        "time_remaining_pct": 0.6, "btc_eth_correlation": 0.5,
    }
    base.update(overrides)
    return base


def make_repricing_signal(direction="YES", confidence=0.7):
    return RepricingSignal(
        asset="BTC", market_id="m1", direction=direction,
        yes_price=0.4, no_price=0.6, confidence=confidence, reason="test",
    )


@pytest.fixture
def model(tmp_path):
    return OnlineQuantModel(state_path=str(tmp_path / "online_model.pkl"), warmup_trades=5)


def test_starts_not_warmed_up(model):
    assert model.is_warmed_up() is False
    assert model.n_updates == 0


def test_decide_mirrors_repricing_signal_during_warmup(model):
    signal = make_repricing_signal()
    should_trade, direction, prob, reason = model.decide(make_features(), signal)
    assert should_trade is True
    assert direction == "YES"
    assert prob == 0.7
    assert "warmup" in reason


def test_decide_false_during_warmup_when_no_repricing_signal(model):
    should_trade, direction, prob, reason = model.decide(make_features(), None)
    assert should_trade is False
    assert direction is None


def test_update_increments_n_updates_and_persists(tmp_path):
    state_path = str(tmp_path / "online_model.pkl")
    model = OnlineQuantModel(state_path=state_path, warmup_trades=5)
    model.update(make_features(), 1)
    assert model.n_updates == 1

    reloaded = OnlineQuantModel(state_path=state_path, warmup_trades=5)
    assert reloaded.n_updates == 1


def test_switches_to_model_after_warmup_threshold(model):
    for i in range(5):
        model.update(make_features(yes_price=0.3 + i * 0.05), i % 2)
    assert model.is_warmed_up() is True
    should_trade, direction, prob, reason = model.decide(make_features(), make_repricing_signal())
    # after warmup, decision no longer echoes the repricing signal's fixed 0.7 confidence
    assert "online model" in reason
    assert prob is not None


def test_predict_proba_one_is_none_before_any_update(model):
    assert model.predict_proba_one(make_features()) is None


def test_predict_proba_one_returns_probability_after_update(model):
    model.update(make_features(), 1)
    p = model.predict_proba_one(make_features())
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_record_and_resolve_round_trip(model):
    features = make_features()
    model.record_features("m1", features)
    resolved = model.resolve("m1", 1)
    assert resolved is True
    assert model.n_updates == 1


def test_resolve_returns_false_for_unknown_market(model):
    assert model.resolve("unknown", 1) is False


def test_kelly_size_zero_when_no_edge(model):
    # win probability equal to break-even implied by entry price -> ~0 edge
    size = model.kelly_size(win_probability=0.4, entry_price=0.6, bankroll=1000.0)
    assert size == pytest.approx(0.0, abs=1e-6)


def test_kelly_size_capped_at_quarter_of_bankroll(model):
    # near-certain win at a cheap price -> full Kelly wants nearly all of bankroll,
    # but sizing must never exceed the 0.25 cap
    size = model.kelly_size(win_probability=0.99, entry_price=0.1, bankroll=1000.0)
    assert size == pytest.approx(250.0, abs=0.01)


def test_kelly_size_positive_with_real_edge(model):
    size = model.kelly_size(win_probability=0.7, entry_price=0.4, bankroll=1000.0)
    assert size > 0.0
    assert size <= 250.0


def test_standardizer_handles_missing_features_without_crashing(model):
    features = make_features(btc_eth_correlation=None, price_momentum_30s=None)
    model.update(features, 1)
    p = model.predict_proba_one(make_features())
    assert p is not None
