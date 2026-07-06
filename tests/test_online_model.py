import pytest

from config.settings import ONLINE_MODEL_OWN_THRESHOLD
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


def test_fresh_model_is_seeded_with_yes_price_prior_not_unfitted(model):
    # a genuinely fresh model (no persisted state) is seeded with an
    # informed yes_price prior immediately -- it is NOT unfitted/None
    p = model.predict_proba_one(make_features())
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_yes_price_prior_is_monotonic_increasing(model):
    # per D-002: higher yes_price should predict a higher win probability
    p_low = model.predict_proba_one(make_features(yes_price=0.30))
    p_mid = model.predict_proba_one(make_features(yes_price=0.45))
    p_high = model.predict_proba_one(make_features(yes_price=0.60))
    assert p_low < p_mid < p_high


def test_seeded_prior_does_not_affect_warmup_progress(model):
    # seeding the prior must never touch real training/warmup state
    assert model.n_updates == 0
    assert model.is_warmed_up() is False


def test_seeded_prior_does_not_override_decide_during_warmup(model):
    # even though predict_proba_one() now returns a real number immediately,
    # decide() must still mirror the repricing signal during warmup, exactly
    # as before the prior was added
    signal = make_repricing_signal(confidence=0.7)
    should_trade, direction, prob, reason = model.decide(make_features(), signal)
    assert prob == 0.7
    assert "warmup" in reason


def test_predict_proba_one_returns_probability_after_update(model):
    model.update(make_features(), 1)
    p = model.predict_proba_one(make_features())
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_real_progress_is_not_overwritten_by_prior_on_reload(tmp_path):
    # a model that already has real training progress must not be
    # re-seeded with the prior when reloaded -- real learned state wins
    state_path = str(tmp_path / "online_model.pkl")
    trained = OnlineQuantModel(state_path=state_path, warmup_trades=5)
    trained.update(make_features(yes_price=0.5), 1)
    coef_after_training = trained.clf.coef_.copy()

    reloaded = OnlineQuantModel(state_path=state_path, warmup_trades=5)
    assert reloaded.n_updates == 1
    assert (reloaded.clf.coef_ == coef_after_training).all()


def test_record_and_resolve_round_trip(model):
    features = make_features()
    model.record_features("m1", features)
    resolved = model.resolve("m1", 1)
    assert resolved is True
    assert model.n_updates == 1


def test_resolve_returns_false_for_unknown_market(model):
    assert model.resolve("unknown", 1) is False


def test_record_features_persists_immediately_to_disk(tmp_path):
    # 2026-07-06 fix: record_features() used to only update the in-memory
    # dict -- _pending was only ever written to disk from within update()'s
    # save() call. A trade opened, then a bot restart before any OTHER
    # trade resolved, silently lost that trade's pending features. Now
    # record_features() saves immediately, so a fresh instance pointed at
    # the same state file (simulating a restart) must see the pending entry
    # without needing any other trade to resolve first.
    state_path = str(tmp_path / "online_model.pkl")
    model = OnlineQuantModel(state_path=state_path, warmup_trades=5)
    model.record_features("m1", make_features())

    restarted = OnlineQuantModel(state_path=state_path, warmup_trades=5)
    assert restarted.resolve("m1", 1) is True
    assert restarted.n_updates == 1


def test_kelly_size_below_lowest_bucket_returns_zero(model):
    size = model.kelly_size(0.59)
    assert size == pytest.approx(0.0, abs=1e-6)


def test_kelly_size_first_bucket(model):
    size = model.kelly_size(0.60)
    assert size == pytest.approx(5.0)
    size = model.kelly_size(0.65)
    assert size == pytest.approx(5.0)


def test_kelly_size_second_bucket(model):
    size = model.kelly_size(0.70)
    assert size == pytest.approx(10.0)
    size = model.kelly_size(0.75)
    assert size == pytest.approx(10.0)


def test_kelly_size_third_bucket(model):
    size = model.kelly_size(0.80)
    assert size == pytest.approx(15.0)
    size = model.kelly_size(0.85)
    assert size == pytest.approx(15.0)


def test_kelly_size_top_bucket(model):
    size = model.kelly_size(0.90)
    assert size == pytest.approx(25.0)
    size = model.kelly_size(1.0)
    assert size == pytest.approx(25.0)


def test_warmup_trades_not_restored_from_persisted_state(tmp_path):
    # warmup_trades is a live config knob, not trained model state -- raising
    # ONLINE_MODEL_WARMUP_TRADES between runs must actually take effect for
    # an existing state file, not get silently overridden by whatever value
    # was saved the last time the file was written
    state_path = str(tmp_path / "online_model.pkl")
    low_bar_model = OnlineQuantModel(state_path=state_path, warmup_trades=5)
    for i in range(5):
        low_bar_model.update(make_features(yes_price=0.3 + i * 0.05), i % 2)
    assert low_bar_model.is_warmed_up() is True

    raised_bar_model = OnlineQuantModel(state_path=state_path, warmup_trades=200)
    assert raised_bar_model.n_updates == 5
    assert raised_bar_model.warmup_trades == 200
    assert raised_bar_model.is_warmed_up() is False


def test_standardizer_handles_missing_features_without_crashing(model):
    features = make_features(btc_eth_correlation=None, price_momentum_30s=None)
    model.update(features, 1)
    p = model.predict_proba_one(make_features())
    assert p is not None


# ---------------- calibration ----------------

def test_calibration_preserves_midpoint():
    assert OnlineQuantModel._calibrate_proba(0.5) == pytest.approx(0.5)


def test_calibration_compresses_saturated_probability_into_target_band():
    assert OnlineQuantModel._calibrate_proba(1.0) == pytest.approx(0.789, abs=0.01)
    assert OnlineQuantModel._calibrate_proba(0.0) == pytest.approx(0.211, abs=0.01)
    # never actually reaches the 0.20/0.80 bounds, only approaches them
    assert 0.20 < OnlineQuantModel._calibrate_proba(1.0) < 0.80
    assert 0.20 < OnlineQuantModel._calibrate_proba(0.0) < 0.80


def test_calibration_is_monotonic():
    values = [OnlineQuantModel._calibrate_proba(p) for p in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]]
    assert values == sorted(values)


def test_calibration_never_produces_exactly_0_or_1():
    for p_raw in [0.0, 1.0, 0.0000001, 0.9999999]:
        calibrated = OnlineQuantModel._calibrate_proba(p_raw)
        assert calibrated != 0.0
        assert calibrated != 1.0


# ---------------- post-warmup dual-gate: model AND combiner must agree ----------------

def test_live_mode_fires_when_model_and_combiner_both_agree(model, mocker):
    model._n_updates = model.warmup_trades  # force warmed-up without needing real trades
    mocker.patch.object(model, "predict_proba_one", return_value=0.7)
    combiner_signal = make_repricing_signal(confidence=0.75)
    should_trade, direction, prob, reason = model.decide(make_features(), combiner_signal)
    assert should_trade is True
    assert direction == "YES"
    assert prob == 0.7
    assert "agree" in reason


def test_live_mode_skips_when_combiner_signal_is_none(model, mocker):
    model._n_updates = model.warmup_trades
    mocker.patch.object(model, "predict_proba_one", return_value=0.7)
    should_trade, direction, prob, reason = model.decide(make_features(), None)
    assert should_trade is False
    assert "combiner" in reason


def test_live_mode_skips_when_combiner_confidence_too_low(model, mocker):
    model._n_updates = model.warmup_trades
    mocker.patch.object(model, "predict_proba_one", return_value=0.7)
    weak_combiner_signal = make_repricing_signal(confidence=0.55)  # <= 0.60 threshold
    should_trade, direction, prob, reason = model.decide(make_features(), weak_combiner_signal)
    assert should_trade is False
    assert "combiner" in reason


def test_live_mode_skips_when_model_disagrees_even_if_combiner_fires(model, mocker):
    model._n_updates = model.warmup_trades
    below_threshold = ONLINE_MODEL_OWN_THRESHOLD - 0.1
    mocker.patch.object(model, "predict_proba_one", return_value=below_threshold)
    strong_combiner_signal = make_repricing_signal(confidence=0.9)
    should_trade, direction, prob, reason = model.decide(make_features(), strong_combiner_signal)
    assert should_trade is False
    assert prob == below_threshold


def test_live_mode_boundary_model_exactly_at_threshold_does_not_fire(model, mocker):
    model._n_updates = model.warmup_trades
    mocker.patch.object(model, "predict_proba_one", return_value=ONLINE_MODEL_OWN_THRESHOLD)  # exactly at threshold, not >
    combiner_signal = make_repricing_signal(confidence=0.9)
    should_trade, _, _, _ = model.decide(make_features(), combiner_signal)
    assert should_trade is False


def test_live_mode_boundary_combiner_exactly_at_threshold_does_not_fire(model, mocker):
    model._n_updates = model.warmup_trades
    mocker.patch.object(model, "predict_proba_one", return_value=0.7)
    combiner_signal = make_repricing_signal(confidence=0.60)  # exactly at threshold, not >
    should_trade, _, _, _ = model.decide(make_features(), combiner_signal)
    assert should_trade is False


# ---------------- NO-direction signals (extreme mean-reversion, 2026-07-06) ----------------
# predict_proba_one() always returns P(YES wins); a NO-direction combiner
# signal must be gated on P(NO wins) = 1-p instead, or it'd be checked
# against the wrong side of the model's belief.

def test_live_mode_fires_for_no_direction_when_model_agrees(model, mocker):
    model._n_updates = model.warmup_trades
    # p=0.1 -> P(YES)=0.1, P(NO)=0.9, comfortably above threshold
    mocker.patch.object(model, "predict_proba_one", return_value=0.1)
    no_signal = make_repricing_signal(direction="NO", confidence=0.9)
    should_trade, direction, prob, reason = model.decide(make_features(), no_signal)
    assert should_trade is True
    assert direction == "NO"
    assert prob == 0.1  # raw P(YES) is still what's returned, not P(NO)
    assert "NO" in reason


def test_live_mode_skips_for_no_direction_when_model_disagrees(model, mocker):
    model._n_updates = model.warmup_trades
    # p=0.9 -> P(YES)=0.9, P(NO)=0.1 -- model thinks YES wins, should block a NO bet
    mocker.patch.object(model, "predict_proba_one", return_value=0.9)
    no_signal = make_repricing_signal(direction="NO", confidence=0.9)
    should_trade, direction, prob, reason = model.decide(make_features(), no_signal)
    assert should_trade is False


def test_live_mode_no_direction_fires_just_above_threshold(model, mocker):
    model._n_updates = model.warmup_trades
    p = 1.0 - ONLINE_MODEL_OWN_THRESHOLD - 0.01  # P(NO) = threshold + 0.01
    mocker.patch.object(model, "predict_proba_one", return_value=p)
    no_signal = make_repricing_signal(direction="NO", confidence=0.9)
    should_trade, _, _, _ = model.decide(make_features(), no_signal)
    assert should_trade is True


def test_live_mode_no_direction_does_not_fire_just_below_threshold(model, mocker):
    model._n_updates = model.warmup_trades
    p = 1.0 - ONLINE_MODEL_OWN_THRESHOLD + 0.01  # P(NO) = threshold - 0.01
    mocker.patch.object(model, "predict_proba_one", return_value=p)
    no_signal = make_repricing_signal(direction="NO", confidence=0.9)
    should_trade, _, _, _ = model.decide(make_features(), no_signal)
    assert should_trade is False
