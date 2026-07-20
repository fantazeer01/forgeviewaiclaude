import datetime
import json
import logging

import pytest

from core.latency_probe import LatencyProbe

T0 = datetime.datetime(2026, 7, 20, 10, 15, 20, tzinfo=datetime.timezone.utc)


def _probe(tmp_path, **kwargs):
    return LatencyProbe(log_path=str(tmp_path / "latency_log.jsonl"), **kwargs)


def _seconds(n):
    return datetime.timedelta(seconds=n)


# 1. A move over the 5bps threshold is detected.
def test_movement_detected_above_threshold(tmp_path):
    probe = _probe(tmp_path, movement_threshold_bps=5.0)
    probe.update(spot_price=64000.0, yes_price=0.50, now=T0)
    probe.update(spot_price=64000.0 * 1.0006, yes_price=0.50, now=T0 + _seconds(3))  # +6bps
    assert probe.movements_detected == 1
    assert len(probe._pending_moves) == 1
    assert probe._pending_moves[0]["direction"] == "UP"
    assert probe._pending_moves[0]["magnitude_bps"] == pytest.approx(6.0, abs=0.01)


# 2. A move under the threshold is ignored.
def test_small_movement_ignored(tmp_path):
    probe = _probe(tmp_path, movement_threshold_bps=5.0)
    probe.update(spot_price=64000.0, yes_price=0.50, now=T0)
    probe.update(spot_price=64000.0 * 1.0004, yes_price=0.50, now=T0 + _seconds(3))  # +4bps
    assert probe.movements_detected == 0
    assert probe._pending_moves == []


# 3. Lag is computed correctly once yes_price confirms in the same direction.
def test_lag_computed_correctly(tmp_path):
    probe = _probe(tmp_path, movement_threshold_bps=5.0, confirm_threshold_pct=2.0)
    probe.update(spot_price=64000.0, yes_price=0.50, now=T0)
    probe.update(spot_price=64000.0 * 1.0006, yes_price=0.50, now=T0 + _seconds(3))  # move detected
    probe.update(spot_price=64000.0 * 1.0006, yes_price=0.55, now=T0 + _seconds(3) + _seconds(4.5))  # +5pp
    assert probe.lags == [pytest.approx(4.5)]


def test_lag_for_down_move(tmp_path):
    probe = _probe(tmp_path, movement_threshold_bps=5.0, confirm_threshold_pct=2.0)
    probe.update(spot_price=64000.0, yes_price=0.50, now=T0)
    probe.update(spot_price=64000.0 * 0.9994, yes_price=0.50, now=T0 + _seconds(3))  # -6bps
    probe.update(spot_price=64000.0 * 0.9994, yes_price=0.44, now=T0 + _seconds(9))  # -6pp
    assert probe.lags == [pytest.approx(6.0)]


# unconfirmed movements never produce a log entry.
def test_unconfirmed_movement_stays_pending(tmp_path):
    probe = _probe(tmp_path, movement_threshold_bps=5.0, confirm_threshold_pct=2.0)
    probe.update(spot_price=64000.0, yes_price=0.50, now=T0)
    probe.update(spot_price=64000.0 * 1.0006, yes_price=0.50, now=T0 + _seconds(3))
    probe.update(spot_price=64000.0 * 1.0006, yes_price=0.505, now=T0 + _seconds(6))  # only +0.5pp
    assert probe.lags == []
    assert len(probe._pending_moves) == 1


def test_stale_pending_move_expires(tmp_path):
    probe = _probe(tmp_path, movement_threshold_bps=5.0, confirm_threshold_pct=2.0, pending_expiry_sec=10.0)
    probe.update(spot_price=64000.0, yes_price=0.50, now=T0)
    probe.update(spot_price=64000.0 * 1.0006, yes_price=0.50, now=T0 + _seconds(3))
    # 20s later -- past the 10s expiry -- yes_price finally moves, but it's too late
    probe.update(spot_price=64000.0 * 1.0006, yes_price=0.60, now=T0 + _seconds(23))
    assert probe.lags == []
    assert probe._pending_moves == []


# 4. The log file is created and each line matches the exact spec'd schema.
def test_log_file_written_correctly(tmp_path):
    probe = _probe(tmp_path, movement_threshold_bps=5.0, confirm_threshold_pct=2.0, market="btc-5m")
    probe.update(spot_price=64000.0, yes_price=0.52, now=T0)
    probe.update(spot_price=64000.0 * 1.0008, yes_price=0.52, now=T0 + _seconds(3))
    probe.update(spot_price=64000.0 * 1.0008, yes_price=0.58, now=T0 + _seconds(3) + _seconds(3.333))

    with open(probe.log_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    record = lines[0]
    assert record["direction"] == "UP"
    assert record["market"] == "btc-5m"
    assert record["yes_price_before"] == 0.52
    assert record["yes_price_after"] == 0.58
    assert record["lag_seconds"] == pytest.approx(3.333, abs=0.01)
    assert record["magnitude_bps"] == pytest.approx(8.0, abs=0.01)
    assert record["ts_binance"].endswith("Z")
    assert record["ts_polymarket"].endswith("Z")


# 5. The periodic stats block reports correct median/mean/min/max and hit rate.
def test_stats_computed_correctly(tmp_path, caplog):
    probe = _probe(tmp_path, movement_threshold_bps=5.0, confirm_threshold_pct=2.0)
    now = T0
    spot = 64000.0
    yes = 0.50
    lags_wanted = [2.0, 4.0, 6.0]

    probe.update(spot_price=spot, yes_price=yes, now=now)  # establishes the baseline, no move yet
    for lag in lags_wanted:
        now += _seconds(1)
        spot *= 1.0006  # a fresh +6bps move each time -- price only ever steps up
        probe.update(spot_price=spot, yes_price=yes, now=now)
        now += _seconds(lag)
        yes += 0.05  # +5pp confirms the move just made
        probe.update(spot_price=spot, yes_price=yes, now=now)
        now += _seconds(5)
    # one extra big, unconfirmed movement so the hit-rate isn't 100%
    now += _seconds(1)
    spot *= 1.05
    probe.update(spot_price=spot, yes_price=yes, now=now)

    with caplog.at_level(logging.INFO, logger="core.latency_probe"):
        probe.log_stats()

    assert probe.lags == pytest.approx(lags_wanted)
    assert probe.movements_detected == 4
    message = caplog.text
    assert "N=3 measurements" in message
    assert "Median lag: 4.0s" in message
    assert "Mean: 4.0s" in message
    assert "Min: 2.0s" in message
    assert "Max: 6.0s" in message
    assert "Movements detected: 4" in message
    assert "Polymarket updated: 3 (75.0%)" in message


def test_stats_with_no_measurements_does_not_crash(tmp_path, caplog):
    probe = _probe(tmp_path)
    with caplog.at_level(logging.INFO, logger="core.latency_probe"):
        probe.log_stats()
    assert "N=0 measurements" in caplog.text
