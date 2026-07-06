import datetime
import json

from config.settings import VOLUME_SKIP_MINUTES_REMAINING_THRESHOLD
from core.signals.volume_signal import VolumeSignalGenerator


def make_market(market_id="m1", asset="BTC", yes_price=0.5, no_price=0.5,
                 volume_24h=1000.0, minutes_remaining=3.0):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price,
        "no_price": no_price, "volume_24h": volume_24h, "minutes_remaining": minutes_remaining,
    }


def write_history(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_no_signal_without_history(tmp_path):
    gen = VolumeSignalGenerator(history_path=str(tmp_path / "volume_history.jsonl"))
    assert gen.generate(make_market()) is None


def test_fires_when_volume_exceeds_threshold(tmp_path):
    path = str(tmp_path / "volume_history.jsonl")
    now = datetime.datetime.now(datetime.timezone.utc)
    write_history(path, [
        {"asset": "BTC", "volume_24h": 1000.0, "timestamp": (now - datetime.timedelta(days=1)).isoformat()},
        {"asset": "BTC", "volume_24h": 1000.0, "timestamp": (now - datetime.timedelta(days=2)).isoformat()},
    ])
    gen = VolumeSignalGenerator(history_path=path)
    signal = gen.generate(make_market(volume_24h=2000.0))  # 2.0x avg of 1000
    assert signal is not None
    assert signal.direction == "YES"


def test_skips_in_first_60s_of_a_fresh_window(tmp_path):
    # 2026-07-06 signal quality pass: minutes_remaining > 4.0 means less
    # than 60s has elapsed since the 5-min window opened -- skip even with
    # a qualifying volume spike.
    path = str(tmp_path / "volume_history.jsonl")
    now = datetime.datetime.now(datetime.timezone.utc)
    write_history(path, [
        {"asset": "BTC", "volume_24h": 1000.0, "timestamp": (now - datetime.timedelta(days=1)).isoformat()},
    ])
    gen = VolumeSignalGenerator(history_path=path)
    assert gen.generate(make_market(volume_24h=2000.0, minutes_remaining=4.5)) is None


def test_fires_right_at_the_60s_boundary(tmp_path):
    path = str(tmp_path / "volume_history.jsonl")
    now = datetime.datetime.now(datetime.timezone.utc)
    write_history(path, [
        {"asset": "BTC", "volume_24h": 1000.0, "timestamp": (now - datetime.timedelta(days=1)).isoformat()},
    ])
    gen = VolumeSignalGenerator(history_path=path)
    signal = gen.generate(make_market(volume_24h=2000.0, minutes_remaining=VOLUME_SKIP_MINUTES_REMAINING_THRESHOLD))
    assert signal is not None


def test_no_signal_when_ratio_at_or_below_threshold(tmp_path):
    path = str(tmp_path / "volume_history.jsonl")
    now = datetime.datetime.now(datetime.timezone.utc)
    write_history(path, [
        {"asset": "BTC", "volume_24h": 1000.0, "timestamp": (now - datetime.timedelta(days=1)).isoformat()},
    ])
    gen = VolumeSignalGenerator(history_path=path)
    assert gen.generate(make_market(volume_24h=1500.0)) is None  # exactly 1.5x, not > 1.5x


def test_ignores_other_assets_average(tmp_path):
    path = str(tmp_path / "volume_history.jsonl")
    now = datetime.datetime.now(datetime.timezone.utc)
    write_history(path, [
        {"asset": "ETH", "volume_24h": 50.0, "timestamp": (now - datetime.timedelta(days=1)).isoformat()},
    ])
    gen = VolumeSignalGenerator(history_path=path)
    # no BTC history at all -- must not use ETH's average
    assert gen.generate(make_market(asset="BTC", volume_24h=2000.0)) is None


def test_ignores_history_older_than_lookback_window(tmp_path):
    path = str(tmp_path / "volume_history.jsonl")
    now = datetime.datetime.now(datetime.timezone.utc)
    write_history(path, [
        {"asset": "BTC", "volume_24h": 10.0, "timestamp": (now - datetime.timedelta(days=10)).isoformat()},
    ])
    gen = VolumeSignalGenerator(history_path=path)
    assert gen.generate(make_market(volume_24h=2000.0)) is None


def test_record_volume_throttles_within_interval(tmp_path):
    path = str(tmp_path / "volume_history.jsonl")
    gen = VolumeSignalGenerator(history_path=path)
    gen.record_volume("BTC", 100.0)
    gen.record_volume("BTC", 200.0)  # should be throttled, not appended
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 1


def test_record_volume_appends_for_different_assets(tmp_path):
    path = str(tmp_path / "volume_history.jsonl")
    gen = VolumeSignalGenerator(history_path=path)
    gen.record_volume("BTC", 100.0)
    gen.record_volume("ETH", 200.0)
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 2


def test_skips_malformed_history_rows(tmp_path):
    path = str(tmp_path / "volume_history.jsonl")
    now = datetime.datetime.now(datetime.timezone.utc)
    with open(path, "w") as f:
        f.write("not json\n")
        f.write(json.dumps({"asset": "BTC", "volume_24h": 1000.0, "timestamp": now.isoformat()}) + "\n")
    gen = VolumeSignalGenerator(history_path=path)
    signal = gen.generate(make_market(volume_24h=2000.0))
    assert signal is not None
