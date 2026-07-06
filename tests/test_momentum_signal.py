import datetime

from core.signals.momentum_signal import MomentumSignalGenerator


def make_market(market_id="m1", asset="BTC", yes_price=0.5, no_price=0.5, minutes_remaining=3.0):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price,
        "no_price": no_price, "minutes_remaining": minutes_remaining,
    }


def test_no_signal_with_fewer_than_3_samples():
    gen = MomentumSignalGenerator()
    gen.update("m1", 0.50)
    gen.update("m1", 0.45)
    assert gen.generate(make_market(yes_price=0.45)) is None


def test_fires_on_fall_then_rise_pattern():
    gen = MomentumSignalGenerator()
    gen.update("m1", 0.50)  # p0
    gen.update("m1", 0.44)  # p1 -- fell (drop 0.06)
    gen.update("m1", 0.472)  # p2 -- rose (bounce 0.032, comfortably >50% of the drop)
    signal = gen.generate(make_market(yes_price=0.472))
    assert signal is not None
    assert signal.direction == "YES"
    assert 0.0 < signal.confidence <= 0.95


def test_no_signal_when_still_falling():
    gen = MomentumSignalGenerator()
    gen.update("m1", 0.50)
    gen.update("m1", 0.45)
    gen.update("m1", 0.40)  # kept falling, no reversal
    assert gen.generate(make_market(yes_price=0.40)) is None


def test_no_signal_when_rising_the_whole_time():
    gen = MomentumSignalGenerator()
    gen.update("m1", 0.40)
    gen.update("m1", 0.45)  # was NOT falling before this
    gen.update("m1", 0.50)
    assert gen.generate(make_market(yes_price=0.50)) is None


def test_stronger_bounce_gives_higher_confidence():
    gen_weak = MomentumSignalGenerator()
    gen_weak.update("m1", 0.50)
    gen_weak.update("m1", 0.45)
    gen_weak.update("m1", 0.48)  # bounce 0.03 -- 60% of the 0.05 drop, clears the filter but modest

    weak_signal = gen_weak.generate(make_market(yes_price=0.48))

    gen_strong = MomentumSignalGenerator()
    gen_strong.update("m2", 0.50)
    gen_strong.update("m2", 0.45)
    gen_strong.update("m2", 0.495)  # bounce 0.045 -- 90% of the drop, nearly recovering it all
    strong_signal = gen_strong.generate(make_market(market_id="m2", yes_price=0.495))

    assert weak_signal is not None
    assert strong_signal is not None
    assert strong_signal.confidence > weak_signal.confidence


def test_no_signal_when_bounce_is_less_than_half_the_drop():
    gen = MomentumSignalGenerator()
    gen.update("m1", 0.50)
    gen.update("m1", 0.45)
    gen.update("m1", 0.47)  # bounce 0.02 -- only 40% of the 0.05 drop
    assert gen.generate(make_market(yes_price=0.47)) is None


def test_history_pruned_outside_window(mocker):
    gen = MomentumSignalGenerator()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_ts = now - datetime.timedelta(seconds=100)
    gen._history["m1"] = [{"ts": old_ts, "yes": 0.5}]
    gen.update("m1", 0.45)
    assert len(gen._history["m1"]) == 1
    assert gen._history["m1"][0]["yes"] == 0.45
