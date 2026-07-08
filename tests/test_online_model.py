import json

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


def test_migrates_appended_features_preserving_weights_and_progress(tmp_path):
    # 2026-07-06: new features get appended to FEATURE_NAMES over time.
    # Loading an older, shorter persisted feature set must extend it
    # in-place -- not reset the model -- since a reset can never re-warm
    # itself live under QUANT_ONLY_MODE (see core/online_model.py docstring).
    state_path = str(tmp_path / "online_model.pkl")
    old_features = ["yes_price", "no_price", "volume_24h"]
    old_model = OnlineQuantModel(state_path=state_path, warmup_trades=5, feature_names=old_features)
    old_model.update({"yes_price": 0.5, "no_price": 0.5, "volume_24h": 100.0}, 1)
    old_model.update({"yes_price": 0.45, "no_price": 0.55, "volume_24h": 200.0}, 0)
    coef_before = old_model.clf.coef_.copy()
    n_updates_before = old_model.n_updates

    new_features = old_features + ["ema_5", "order_book_depth"]
    migrated = OnlineQuantModel(state_path=state_path, warmup_trades=5, feature_names=new_features)

    assert migrated.feature_names == new_features
    assert migrated.n_updates == n_updates_before
    assert migrated._fitted is True
    assert migrated.clf.coef_.shape == (1, len(new_features))
    assert (migrated.clf.coef_[0][:len(old_features)] == coef_before[0]).all()  # preserved exactly
    assert (migrated.clf.coef_[0][len(old_features):] == 0.0).all()  # new dims start neutral
    assert migrated._mean.shape == (len(new_features),)
    assert migrated._m2.shape == (len(new_features),)
    assert migrated._count_vec.shape == (len(new_features),)

    p = migrated.predict_proba_one({
        "yes_price": 0.5, "no_price": 0.5, "volume_24h": 100.0,
        "ema_5": 0.5, "order_book_depth": 1.2,
    })
    assert p is not None


def test_migration_is_a_noop_when_feature_names_unchanged(tmp_path):
    state_path = str(tmp_path / "online_model.pkl")
    features = ["yes_price", "no_price"]
    m1 = OnlineQuantModel(state_path=state_path, warmup_trades=5, feature_names=features)
    m1.update({"yes_price": 0.5, "no_price": 0.5}, 1)

    m2 = OnlineQuantModel(state_path=state_path, warmup_trades=5, feature_names=features)
    assert m2.feature_names == features
    assert m2.n_updates == 1


def test_incompatible_feature_change_keeps_persisted_feature_set(tmp_path):
    # Not a clean append (a feature was removed/reordered) -- no safe
    # migration exists since coefficients are positional. Must keep
    # training on the persisted set rather than silently mismap weights.
    state_path = str(tmp_path / "online_model.pkl")
    old_features = ["yes_price", "no_price", "volume_24h"]
    m1 = OnlineQuantModel(state_path=state_path, warmup_trades=5, feature_names=old_features)
    m1.update({"yes_price": 0.5, "no_price": 0.5, "volume_24h": 100.0}, 1)

    incompatible = ["yes_price", "no_price", "ema_5"]  # "volume_24h" replaced, not appended
    m2 = OnlineQuantModel(state_path=state_path, warmup_trades=5, feature_names=incompatible)
    assert m2.feature_names == old_features
    assert m2.n_updates == 1


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


def test_kelly_size_negative_edge_returns_zero(model):
    # the reported example: yes_price=0.55, p=0.54 -> raw f = (0.54*b-0.46)/b
    # with b=(1-0.55)/0.55=0.818 is slightly negative (~-0.022) -- no edge.
    size = model.kelly_size(0.54, 0.55)
    assert size == pytest.approx(0.0, abs=1e-6)


def test_kelly_size_zero_edge_returns_zero_not_a_zero_dollar_trade(model):
    # b=1 (yes_price=0.5), p=0.5 -> f=(0.5-0.5)/1=0 exactly. Callers must
    # treat this 0.0 as "skip the trade," not "open at $0" -- see
    # run.py._decide_and_open.
    size = model.kelly_size(0.5, 0.5)
    assert size == pytest.approx(0.0, abs=1e-6)


def test_kelly_size_small_positive_edge_below_min_edge_returns_zero(model):
    # b=1 (yes_price=0.5), p=0.51 -> f=0.02, below KELLY_MIN_EDGE (0.05) -- an
    # unconfirmed edge, must not open a position (2026-07-08 min-edge filter,
    # applied inside kelly_fraction() itself). Previously this floored to the
    # $5 minimum; now it's filtered out before reaching the dollar clamp.
    size = model.kelly_size(0.51, 0.5)
    assert size == pytest.approx(0.0, abs=1e-6)


def test_kelly_size_mid_range_edge_scales_with_real_fraction(model):
    # b=1 (yes_price=0.5), p=0.6 -> f=0.2 -> $20, between the $5/$25 clamps
    size = model.kelly_size(0.6, 0.5)
    assert size == pytest.approx(20.0)


def test_kelly_size_large_edge_capped_at_maximum(model):
    # b=1 (yes_price=0.5), p=0.9 -> f=0.8 -> $80 raw, capped at $25
    size = model.kelly_size(0.9, 0.5)
    assert size == pytest.approx(25.0)


def test_kelly_size_accounts_for_actual_payout_ratio(model):
    # same win_probability (0.6), different yes_price -> different size,
    # since the payout ratio b=(1-yes_price)/yes_price differs -- exactly
    # what the old flat BET_SIZES table (keyed only on combiner confidence,
    # blind to entry_price) could never distinguish.
    favorable = model.kelly_size(0.6, 0.45)    # b=1.222 -> real edge -> sized
    unfavorable = model.kelly_size(0.6, 0.64)  # b=0.5625 -> f<0 -> no edge -> $0
    assert favorable > 0
    assert unfavorable == pytest.approx(0.0, abs=1e-6)
    assert favorable > unfavorable


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


# ---------------- saturation health check (2026-07-07) ----------------

def test_health_check_flags_saturated_model_and_resets(model, mocker):
    # a model whose predict_proba_one() returns the same value regardless of
    # input is exactly the failure mode observed twice in production (SGD
    # coefficients diverging into a near-binary step function) -- the health
    # check must catch this from the OUTPUT behavior itself, not coefficient
    # magnitude, and reset rather than leave it running degenerate.
    mocker.patch.object(model, "predict_proba_one", return_value=0.79)
    health = model._run_health_check()
    assert health == "reset"
    assert model.model_health == "reset"
    assert model.n_updates == 0
    assert model.is_warmed_up() is False


def test_health_check_leaves_healthy_model_alone(model):
    # the seeded yes_price prior is monotonic in yes_price (see
    # test_yes_price_prior_is_monotonic_increasing), so a genuine spread
    # across the saturation probe's yes_price sweep is expected -- no reset.
    n_updates_before = model.n_updates
    health = model._run_health_check()
    assert health == "healthy"
    assert model.model_health == "healthy"
    assert model.n_updates == n_updates_before


def test_health_check_runs_every_health_check_interval_updates(model, mocker):
    spy = mocker.spy(model, "_run_health_check")
    mocker.patch("core.online_model.ONLINE_MODEL_HEALTH_CHECK_INTERVAL", 3)
    for i in range(3):
        model.update(make_features(yes_price=0.3 + i * 0.05), i % 2)
    assert spy.call_count == 1  # only on the 3rd update, not every update


def test_saturation_reset_clears_history_and_pending(model, mocker, tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_model.MODEL_CHECKPOINT_FILE", str(tmp_path / "checkpoint.pkl"))
    model.update(make_features(yes_price=0.5), 1)
    model.record_features("still-open-market", make_features())
    mocker.patch.object(model, "predict_proba_one", return_value=0.21)
    model._run_health_check()
    assert model._history_X == []
    assert model._history_y == []
    assert model._pending == {}


# ---------------- soft reset / checkpoint (2026-07-08) ----------------

def test_reset_saves_checkpoint_when_there_is_real_progress(model, tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pkl"
    monkeypatch.setattr("core.online_model.MODEL_CHECKPOINT_FILE", str(checkpoint_path))
    model.update(make_features(yes_price=0.5), 1)
    model.update(make_features(yes_price=0.4), 0)
    saved = model._reset_to_fresh()
    assert saved is True
    assert checkpoint_path.exists()


def test_reset_skips_checkpoint_on_a_genuinely_fresh_model(model, tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pkl"
    monkeypatch.setattr("core.online_model.MODEL_CHECKPOINT_FILE", str(checkpoint_path))
    # no update() calls at all -- n_updates is still 0, nothing real to save
    saved = model._reset_to_fresh()
    assert saved is False
    assert not checkpoint_path.exists()


def test_checkpoint_contains_the_pre_reset_state(model, tmp_path, monkeypatch):
    import pickle
    checkpoint_path = tmp_path / "checkpoint.pkl"
    monkeypatch.setattr("core.online_model.MODEL_CHECKPOINT_FILE", str(checkpoint_path))
    model.update(make_features(yes_price=0.5), 1)
    model.update(make_features(yes_price=0.4), 0)
    n_updates_before = model.n_updates
    coef_before = model.clf.coef_.copy()
    model._reset_to_fresh()
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    assert checkpoint["n_updates"] == n_updates_before
    assert (checkpoint["clf"].coef_ == coef_before).all()
    assert checkpoint["feature_names"] == model.feature_names


def test_reset_does_not_warm_start_the_new_model_from_the_checkpoint(model, tmp_path, monkeypatch):
    # explicit non-goal, per the 2026-07-08 request: the new model must be
    # the ordinary seeded-prior fresh start, NOT initialized from the
    # checkpoint's (just-diverged/collapsed) weights.
    monkeypatch.setattr("core.online_model.MODEL_CHECKPOINT_FILE", str(tmp_path / "checkpoint.pkl"))
    model.update(make_features(yes_price=0.5), 1)
    model.update(make_features(yes_price=0.4), 0)
    coef_before_reset = model.clf.coef_.copy()
    model._reset_to_fresh()
    # the fresh model is the documented seeded prior (zeros except yes_price),
    # not a copy of the pre-reset coefficients
    assert not (model.clf.coef_ == coef_before_reset).all()
    assert model.n_updates == 0


# ---------------- stability monitor (2026-07-07) ----------------

def test_stability_monitor_runs_every_stability_check_interval_updates(model, mocker, tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_model.MODEL_HEALTH_LOG", str(tmp_path / "health.jsonl"))
    spy = mocker.spy(model, "_run_stability_monitor")
    mocker.patch("core.online_model.STABILITY_CHECK_INTERVAL", 3)
    for i in range(3):
        model.update(make_features(yes_price=0.3 + i * 0.05), i % 2)
    assert spy.call_count == 1  # only on the 3rd update, not every update


def test_stability_monitor_logs_report_to_jsonl(model, tmp_path, monkeypatch):
    log_path = tmp_path / "health.jsonl"
    monkeypatch.setattr("core.online_model.MODEL_HEALTH_LOG", str(log_path))
    model.update(make_features(yes_price=0.5), 1)
    model.update(make_features(yes_price=0.4), 0)
    model._run_stability_monitor()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["n_updates"] == model.n_updates
    assert "win_rate" in entry
    assert "action" in entry


def test_stability_monitor_warns_but_does_not_reset_on_low_win_rate(model, tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_model.MODEL_HEALTH_LOG", str(tmp_path / "health.jsonl"))
    # 9 losses then 1 win -- win rate 0.1, well under STABILITY_MIN_WIN_RATE (0.45)
    for i in range(10):
        model.update(make_features(yes_price=0.3 + i * 0.02), 0 if i < 9 else 1)
    n_updates_before = model.n_updates
    model._run_stability_monitor()
    # low win rate warns, but does not reset -- n_updates/history untouched
    assert model.n_updates == n_updates_before
    assert model._history_y != []


def test_stability_monitor_resets_on_low_prediction_diversity(model, tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_model.MODEL_HEALTH_LOG", str(tmp_path / "health.jsonl"))
    monkeypatch.setattr("core.online_model.MODEL_CHECKPOINT_FILE", str(tmp_path / "checkpoint.pkl"))
    model._recent_predictions = [0.5] * 20  # zero variance -- collapsed predictions
    model._run_stability_monitor()
    assert model.model_health == "reset"
    assert model.n_updates == 0


def test_stability_monitor_resets_on_coef_out_of_bound(model, tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_model.MODEL_HEALTH_LOG", str(tmp_path / "health.jsonl"))
    monkeypatch.setattr("core.online_model.MODEL_CHECKPOINT_FILE", str(tmp_path / "checkpoint.pkl"))
    model.update(make_features(yes_price=0.5), 1)
    model.update(make_features(yes_price=0.4), 0)
    model.clf.coef_[0][0] = 6.0  # exceeds STABILITY_COEF_BOUND (5.0)
    model._run_stability_monitor()
    assert model.model_health == "reset"
    assert model.n_updates == 0


# ---------------- scheduled full retrain (2026-07-07) ----------------

def test_scheduled_retrain_runs_every_retrain_interval_updates(model, mocker, tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_model.MODEL_RETRAIN_LOG", str(tmp_path / "retrain.jsonl"))
    mocker.patch("core.online_model.STABILITY_CHECK_INTERVAL", 1000)  # avoid interfering
    mocker.patch("core.online_model.ONLINE_MODEL_HEALTH_CHECK_INTERVAL", 1000)
    spy = mocker.spy(model, "_run_scheduled_retrain")
    mocker.patch("core.online_model.RETRAIN_INTERVAL", 4)
    for i in range(4):
        model.update(make_features(yes_price=0.3 + i * 0.05), i % 2)
    assert spy.call_count == 1


def test_scheduled_retrain_skips_when_only_one_class_in_window(model, tmp_path, monkeypatch):
    monkeypatch.setattr("core.online_model.MODEL_RETRAIN_LOG", str(tmp_path / "retrain.jsonl"))
    monkeypatch.setattr("core.online_model.RETRAIN_WINDOW", 3)
    clf_before = model.clf
    model.update(make_features(yes_price=0.5), 1)
    model.update(make_features(yes_price=0.45), 1)
    model._run_scheduled_retrain()
    assert model.clf is clf_before  # untouched
    assert not (tmp_path / "retrain.jsonl").exists()  # nothing logged either


def test_scheduled_retrain_accepts_healthy_candidate_and_swaps(model, tmp_path, monkeypatch):
    log_path = tmp_path / "retrain.jsonl"
    monkeypatch.setattr("core.online_model.MODEL_RETRAIN_LOG", str(log_path))
    for i in range(10):
        model.update(make_features(yes_price=0.3 + (i % 5) * 0.05), i % 2)
    old_clf = model.clf
    model._run_scheduled_retrain()
    assert model.clf is not old_clf  # swapped
    entry = json.loads(log_path.read_text().strip().split("\n")[-1])
    assert entry["accepted"] is True
    assert len(entry["feature_importance"]) == len(model.feature_names)


def test_scheduled_retrain_rejects_unhealthy_candidate_and_keeps_live_model(model, mocker, tmp_path, monkeypatch):
    log_path = tmp_path / "retrain.jsonl"
    monkeypatch.setattr("core.online_model.MODEL_RETRAIN_LOG", str(log_path))
    for i in range(10):
        model.update(make_features(yes_price=0.3 + (i % 5) * 0.05), i % 2)
    old_clf = model.clf
    mocker.patch.object(model, "_check_candidate_diversity", return_value=(False, 0.001))
    model._run_scheduled_retrain()
    assert model.clf is old_clf  # rejected, unchanged
    entry = json.loads(log_path.read_text().strip().split("\n")[-1])
    assert entry["accepted"] is False


def test_feature_importance_ranks_by_absolute_coefficient(model):
    model.clf.coef_[0] = 0.0
    idx_a = model.feature_names.index("yes_price")
    idx_b = model.feature_names.index("no_price")
    model.clf.coef_[0][idx_a] = 0.9
    model.clf.coef_[0][idx_b] = -0.1
    importances = model._feature_importance(model.clf)
    assert importances[0]["feature"] == "yes_price"
    assert importances[0]["coef"] == pytest.approx(0.9)
