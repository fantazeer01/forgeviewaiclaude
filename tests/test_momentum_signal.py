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
    gen.update("m1", 0.44)  # p1 -- fell
    gen.update("m1", 0.47)  # p2 -- rose (bounce)
    signal = gen.generate(make_market(yes_price=0.47))
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
    gen_weak.update("m1", 0.451)  # tiny bounce
    weak_signal = gen_weak.generate(make_market(yes_price=0.451))

    gen_strong = MomentumSignalGenerator()
    gen_strong.update("m2", 0.50)
    gen_strong.update("m2", 0.45)
    gen_strong.update("m2", 0.49)  # strong bounce, nearly recovering the whole drop
    strong_signal = gen_strong.generate(make_market(market_id="m2", yes_price=0.49))

    assert strong_signal.confidence > weak_signal.confidence


def test_history_pruned_outside_window(mocker):
    gen = MomentumSignalGenerator()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_ts = now - datetime.timedelta(seconds=100)
    gen._history["m1"] = [{"ts": old_ts, "yes": 0.5}]
    gen.update("m1", 0.45)
    assert len(gen._history["m1"]) == 1
    assert gen._history["m1"][0]["yes"] == 0.45
