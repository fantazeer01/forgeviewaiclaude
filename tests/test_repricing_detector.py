import datetime

import pytest

from config.settings import REPRICING_FROZEN
from core.repricing_detector import RepricingDetector, RepricingSignal


def make_market(market_id="m1", asset="BTC", yes_price=0.5, no_price=0.5, minutes_remaining=3.0):
    return {
        "market_id": market_id,
        "asset": asset,
        "yes_price": yes_price,
        "no_price": no_price,
        "minutes_remaining": minutes_remaining,
    }


def test_signal_to_dict_roundtrip():
    signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="YES",
        yes_price=0.6, no_price=0.4, confidence=0.8, reason="test",
        timestamp="2024-01-01T00:00:00", minutes_remaining=3.0,
    )
    assert signal.to_dict() == {
        "asset": "BTC", "market_id": "m1", "direction": "YES",
        "yes_price": 0.6, "no_price": 0.4, "confidence": 0.8, "reason": "test",
        "timestamp": "2024-01-01T00:00:00", "minutes_remaining": 3.0,
    }


def test_detect_returns_none_with_insufficient_history():
    detector = RepricingDetector()
    detector.update_prices("m1", 0.5, 0.5)
    assert detector.detect(make_market()) is None


@pytest.mark.parametrize("minutes_remaining", [0.5, 4.6])
def test_detect_returns_none_outside_time_window(minutes_remaining):
    detector = RepricingDetector()
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [
        {"ts": old_ts, "yes": 0.9, "no": 0.1},
        {"ts": datetime.datetime.now(datetime.timezone.utc), "yes": 0.5, "no": 0.5},
    ]
    market = make_market(minutes_remaining=minutes_remaining)
    assert detector.detect(market) is None


def test_detect_returns_none_when_no_old_observation():
    detector = RepricingDetector()
    detector.update_prices("m1", 0.9, 0.1)
    detector.update_prices("m1", 0.5, 0.5)
    assert detector.detect(make_market(yes_price=0.5, no_price=0.5)) is None


def test_detect_yes_drop_triggers_signal():
    detector = RepricingDetector()
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]
    detector.update_prices("m1", 0.5, 0.5)
    signal = detector.detect(make_market(yes_price=0.5, no_price=0.5))
    assert signal is not None
    assert signal.direction == "YES"
    assert signal.confidence == 0.95
    assert "YES dropped" in signal.reason


def test_detect_no_drop_does_not_trigger_signal():
    detector = RepricingDetector()
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.1, "no": 0.9}]
    detector.update_prices("m1", 0.5, 0.5)
    assert detector.detect(make_market(yes_price=0.5, no_price=0.5)) is None


def test_detect_only_ever_returns_yes_direction():
    detector = RepricingDetector()
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]
    detector.update_prices("m1", 0.5, 0.5)
    signal = detector.detect(make_market(yes_price=0.5, no_price=0.5))
    assert signal is not None
    assert signal.direction == "YES"


def test_detect_skips_signal_when_yes_price_below_floor():
    detector = RepricingDetector()
    floor = REPRICING_FROZEN["min_yes_price"]
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": floor + 0.10, "no": 1 - (floor + 0.10)}]
    below = floor - 0.01
    detector.update_prices("m1", below, 1 - below)
    assert detector.detect(make_market(yes_price=below, no_price=1 - below)) is None


def test_detect_allows_signal_when_yes_price_at_floor():
    detector = RepricingDetector()
    floor = REPRICING_FROZEN["min_yes_price"]
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": floor + 0.10, "no": 1 - (floor + 0.10)}]
    detector.update_prices("m1", floor, 1 - floor)
    signal = detector.detect(make_market(yes_price=floor, no_price=1 - floor))
    assert signal is not None
    assert signal.direction == "YES"


def test_detect_allows_signal_just_below_ceiling():
    detector = RepricingDetector()
    ceiling = REPRICING_FROZEN["max_yes_price"]
    just_below = ceiling - 0.01
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": just_below + 0.10, "no": 1 - (just_below + 0.10)}]
    detector.update_prices("m1", just_below, 1 - just_below)
    signal = detector.detect(make_market(yes_price=just_below, no_price=1 - just_below))
    assert signal is not None
    assert signal.direction == "YES"


def test_detect_skips_signal_when_yes_price_at_or_above_ceiling():
    detector = RepricingDetector()
    ceiling = REPRICING_FROZEN["max_yes_price"]
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": ceiling + 0.10, "no": 1 - (ceiling + 0.10)}]
    detector.update_prices("m1", ceiling, 1 - ceiling)
    assert detector.detect(make_market(yes_price=ceiling, no_price=1 - ceiling)) is None


def test_detect_below_threshold_returns_none():
    detector = RepricingDetector()
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.51, "no": 0.49}]
    detector.update_prices("m1", 0.50, 0.50)
    assert detector.detect(make_market(yes_price=0.50, no_price=0.50)) is None


def test_detect_confidence_scales_with_drop_size():
    detector = RepricingDetector()
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.60, "no": 0.40}]
    detector.update_prices("m1", 0.55, 0.45)
    signal = detector.detect(make_market(yes_price=0.55, no_price=0.45))
    threshold = REPRICING_FROZEN["min_price_move"]
    conf_threshold = REPRICING_FROZEN["confidence_threshold"]
    expected = round(conf_threshold + (0.05 - threshold) * 2, 3)
    assert signal.confidence == expected


def test_update_prices_prunes_old_entries():
    detector = RepricingDetector()
    max_window = REPRICING_FROZEN["max_time_window_sec"]
    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=max_window + 10)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]
    detector.update_prices("m1", 0.5, 0.5)
    assert len(detector._price_history["m1"]) == 1
    assert detector._price_history["m1"][0]["yes"] == 0.5


def test_reset_market_clears_history():
    detector = RepricingDetector()
    detector.update_prices("m1", 0.5, 0.5)
    detector.reset_market("m1")
    assert "m1" not in detector._price_history
