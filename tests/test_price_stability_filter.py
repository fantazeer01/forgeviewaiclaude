import datetime

from core.signals.price_stability_filter import PriceStabilityFilter


def test_not_stable_with_fewer_than_2_samples():
    f = PriceStabilityFilter()
    f.update("m1", 0.50)
    assert f.is_stable("m1") is False


def test_not_stable_for_unknown_market():
    f = PriceStabilityFilter()
    assert f.is_stable("unknown") is False


def test_stable_when_price_barely_moves():
    f = PriceStabilityFilter()
    f.update("m1", 0.500)
    f.update("m1", 0.505)
    f.update("m1", 0.501)
    assert f.is_stable("m1") is True  # range 0.005 < 0.02


def test_not_stable_when_price_moves_enough():
    f = PriceStabilityFilter()
    f.update("m1", 0.50)
    f.update("m1", 0.53)  # range 0.03 >= 0.02
    assert f.is_stable("m1") is False


def test_range_based_not_endpoint_based():
    # price swings out and back to its starting point -- endpoint diff is
    # ~0, but the market clearly moved, so this must NOT be called stable.
    f = PriceStabilityFilter()
    f.update("m1", 0.50)
    f.update("m1", 0.60)
    f.update("m1", 0.50)
    assert f.is_stable("m1") is False


def test_history_pruned_outside_window(mocker):
    f = PriceStabilityFilter()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_ts = now - datetime.timedelta(seconds=100)
    f._history["m1"] = [{"ts": old_ts, "yes": 0.9}]
    f.update("m1", 0.50)
    assert len(f._history["m1"]) == 1
    assert f._history["m1"][0]["yes"] == 0.50


def test_different_markets_tracked_independently():
    f = PriceStabilityFilter()
    f.update("m1", 0.50)
    f.update("m1", 0.501)  # flat
    f.update("m2", 0.50)
    f.update("m2", 0.60)  # moved
    assert f.is_stable("m1") is True
    assert f.is_stable("m2") is False
