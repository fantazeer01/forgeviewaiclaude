import datetime

import pytest

from core.signals.mean_reversion_no_signal import MeanReversionNoSignalGenerator


def make_market(market_id="m1", asset="BTC", yes_price=0.85, no_price=0.15, minutes_remaining=3.0):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price,
        "no_price": no_price, "minutes_remaining": minutes_remaining,
    }


def test_no_signal_below_min_yes_price_gate():
    gen = MeanReversionNoSignalGenerator()
    gen.update("m1", 0.95)
    gen.update("m1", 0.79)  # dropped a lot, but current price is below the 0.80 gate
    assert gen.generate(make_market(yes_price=0.79)) is None


def test_no_signal_when_peak_never_reached_0_90():
    gen = MeanReversionNoSignalGenerator()
    gen.update("m1", 0.85)  # peak only 0.85, below NO_REVERSION_PEAK_MIN_YES_PRICE (0.90)
    gen.update("m1", 0.81)
    assert gen.generate(make_market(yes_price=0.81)) is None


def test_fires_on_real_reversion_from_extreme_peak(mocker):
    # NO_TRADING_ENABLED defaults to False (2026-07-07: disabled again after
    # real results confirmed a negative edge, 2/25 = 8.00% win rate) -- this
    # test exercises the underlying reversion-detection logic itself, so it
    # explicitly re-enables the flag rather than relying on a default that's
    # deliberately off. See test_disabled_by_default_kill_switch below for
    # the off-by-default behavior itself.
    mocker.patch("core.signals.mean_reversion_no_signal.NO_TRADING_ENABLED", True)
    gen = MeanReversionNoSignalGenerator()
    gen.update("m1", 0.95)  # extreme peak
    gen.update("m1", 0.83)  # dropped 0.12, comfortably above NO_REVERSION_MIN_DROP (0.05)
    signal = gen.generate(make_market(yes_price=0.83))
    assert signal is not None
    assert signal.direction == "NO"
    assert signal.yes_price == 0.83
    assert signal.no_price == pytest.approx(0.17)
    assert 0.0 < signal.confidence <= 0.95


def test_disabled_by_default_kill_switch():
    # 2026-07-07: NO_TRADING_ENABLED = False is the real, current default --
    # confirms generate() respects it directly (defense-in-depth on top of
    # SignalCombiner's own check at the call site) without needing any mock.
    gen = MeanReversionNoSignalGenerator()
    gen.update("m1", 0.95)
    gen.update("m1", 0.83)  # would otherwise fire -- same setup as the test above
    assert gen.generate(make_market(yes_price=0.83)) is None


def test_no_signal_when_drop_too_small():
    gen = MeanReversionNoSignalGenerator()
    gen.update("m1", 0.95)
    gen.update("m1", 0.93)  # only dropped 0.02, below NO_REVERSION_MIN_DROP (0.05)
    assert gen.generate(make_market(yes_price=0.93)) is None


def test_larger_drop_gives_higher_confidence(mocker):
    mocker.patch("core.signals.mean_reversion_no_signal.NO_TRADING_ENABLED", True)
    gen_small = MeanReversionNoSignalGenerator()
    gen_small.update("m1", 0.95)
    gen_small.update("m1", 0.88)  # modest drop
    small_signal = gen_small.generate(make_market(yes_price=0.88))

    gen_large = MeanReversionNoSignalGenerator()
    gen_large.update("m2", 0.99)
    gen_large.update("m2", 0.81)  # deep drop, nearly all the way back to the 0.80 floor
    large_signal = gen_large.generate(make_market(market_id="m2", yes_price=0.81))

    assert small_signal is not None
    assert large_signal is not None
    assert large_signal.confidence > small_signal.confidence


def test_history_pruned_outside_window(mocker):
    gen = MeanReversionNoSignalGenerator()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_ts = now - datetime.timedelta(seconds=100)
    gen._history["m1"] = [{"ts": old_ts, "yes": 0.95}]
    gen.update("m1", 0.83)
    assert len(gen._history["m1"]) == 1
    assert gen._history["m1"][0]["yes"] == 0.83


def test_no_signal_with_empty_history():
    gen = MeanReversionNoSignalGenerator()
    assert gen.generate(make_market(yes_price=0.85)) is None
