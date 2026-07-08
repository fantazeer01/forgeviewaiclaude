import datetime

from config.settings import SIGNAL_COOLDOWN_SEC
from core.repricing_detector import RepricingDetector
from core.state_manager import StateManager
from signals.repricing_signal import RepricingSignalGenerator


def make_market(market_id, asset="BTC", yes_price=0.5, no_price=0.5, minutes_remaining=3.0):
    return {
        "market_id": market_id,
        "asset": asset,
        "yes_price": yes_price,
        "no_price": no_price,
        "minutes_remaining": minutes_remaining,
    }


def test_process_market_returns_none_without_signal(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    generator = RepricingSignalGenerator(RepricingDetector(), state)
    assert generator.process_market(make_market("m1")) is None


def test_process_market_detects_and_sets_cooldown(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    detector = RepricingDetector()
    generator = RepricingSignalGenerator(detector, state)
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]
    signal = generator.process_market(make_market("m1", yes_price=0.5, no_price=0.5))
    assert signal is not None
    assert signal.direction == "YES"
    assert "BTC" in state.get("last_signal_ts")


def test_process_market_suppressed_during_cooldown(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    state.set("last_signal_ts", {"BTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    detector = RepricingDetector()
    generator = RepricingSignalGenerator(detector, state)
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m2"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]
    signal = generator.process_market(make_market("m2", asset="BTC", yes_price=0.5, no_price=0.5))
    assert signal is None


def test_cooldown_expired_allows_detection(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    expired_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=SIGNAL_COOLDOWN_SEC + 5)
    state.set("last_signal_ts", {"BTC": expired_ts.isoformat()})
    detector = RepricingDetector()
    generator = RepricingSignalGenerator(detector, state)
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m3"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]
    signal = generator.process_market(make_market("m3", asset="BTC", yes_price=0.5, no_price=0.5))
    assert signal is not None


def test_is_in_cooldown_false_without_prior_signal(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    generator = RepricingSignalGenerator(RepricingDetector(), state)
    assert generator._is_in_cooldown("BTC") is False


def test_is_in_cooldown_handles_malformed_timestamp(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    state.set("last_signal_ts", {"BTC": "not-a-timestamp"})
    generator = RepricingSignalGenerator(RepricingDetector(), state)
    assert generator._is_in_cooldown("BTC") is False


def test_process_market_updates_detector_history_even_in_cooldown(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    state.set("last_signal_ts", {"BTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    detector = RepricingDetector()
    generator = RepricingSignalGenerator(detector, state)
    generator.process_market(make_market("m4", asset="BTC"))
    assert "m4" in detector._price_history


def _make_signal_ready_generator(tmp_path, market_bias_provider):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    detector = RepricingDetector()
    generator = RepricingSignalGenerator(detector, state, market_bias_provider=market_bias_provider)
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["mbias"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]
    return generator, state


def test_bearish_bias_blocks_yes_signal(tmp_path):
    generator, _ = _make_signal_ready_generator(tmp_path, market_bias_provider=lambda: "BEARISH")
    signal = generator.process_market(make_market("mbias", yes_price=0.5, no_price=0.5))
    assert signal is None


def test_bearish_bias_still_sets_cooldown_even_though_blocked(tmp_path):
    generator, state = _make_signal_ready_generator(tmp_path, market_bias_provider=lambda: "BEARISH")
    generator.process_market(make_market("mbias", yes_price=0.5, no_price=0.5))
    assert "BTC" in state.get("last_signal_ts")


def test_bullish_bias_allows_yes_signal(tmp_path):
    generator, _ = _make_signal_ready_generator(tmp_path, market_bias_provider=lambda: "BULLISH")
    signal = generator.process_market(make_market("mbias", yes_price=0.5, no_price=0.5))
    assert signal is not None
    assert signal.direction == "YES"


def test_neutral_bias_allows_yes_signal(tmp_path):
    generator, _ = _make_signal_ready_generator(tmp_path, market_bias_provider=lambda: "NEUTRAL")
    signal = generator.process_market(make_market("mbias", yes_price=0.5, no_price=0.5))
    assert signal is not None


def test_no_bias_provider_allows_signal_unchanged(tmp_path):
    generator, _ = _make_signal_ready_generator(tmp_path, market_bias_provider=None)
    signal = generator.process_market(make_market("mbias", yes_price=0.5, no_price=0.5))
    assert signal is not None


def test_bias_provider_returning_none_does_not_block(tmp_path):
    # provider exists but has no real data yet (e.g. first CoinGecko fetch
    # hasn't completed) -- must never be treated as BEARISH
    generator, _ = _make_signal_ready_generator(tmp_path, market_bias_provider=lambda: None)
    signal = generator.process_market(make_market("mbias", yes_price=0.5, no_price=0.5))
    assert signal is not None
