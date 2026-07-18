import json

from core.stats_tracker import StatsTracker, bucket_key, FAIR_VALUE_DEPLOY_CUTOFF


def _tracker(tmp_path, trades_log_path=None):
    return StatsTracker(
        state_file=str(tmp_path / "stats_tracker.json"),
        trades_log_path=trades_log_path or str(tmp_path / "nonexistent_trades.jsonl"),
    )


def test_should_trade_true_when_insufficient_samples(tmp_path):
    # n < 10 (STATS_EARLY_BLOCK_MIN_SAMPLES) -- no data at all yet, trade
    # regardless of win rate to start accumulating.
    tracker = _tracker(tmp_path)
    for _ in range(5):
        tracker.record(0.46, 12, False)  # 0/5 win rate, but n < 10
    assert tracker.should_trade(0.46, 12) is True


def test_should_trade_false_early_block_when_win_rate_below_045_at_n10(tmp_path):
    # n >= 10 with win_rate < 0.45 -- blocked immediately, doesn't wait for n=50.
    tracker = _tracker(tmp_path)
    for _ in range(4):
        tracker.record(0.46, 12, True)
    for _ in range(6):
        tracker.record(0.46, 12, False)
    # 4/10 = 40% < 0.45
    assert tracker.should_trade(0.46, 12) is False


def test_should_trade_true_when_win_rate_above_045_but_below_50_samples(tmp_path):
    # Not early-blocked (win_rate >= 0.45) and not yet at n=50 -- keeps
    # accumulating normally, same as the old "insufficient data" behavior.
    tracker = _tracker(tmp_path)
    for _ in range(10):
        tracker.record(0.46, 12, True)
    for _ in range(10):
        tracker.record(0.46, 12, False)
    # 10/20 = 50% -- above the 0.45 early-block bar, below n=50
    assert tracker.should_trade(0.46, 12) is True


def test_should_trade_true_when_win_rate_above_threshold(tmp_path):
    tracker = _tracker(tmp_path)
    for _ in range(30):
        tracker.record(0.46, 12, True)
    for _ in range(20):
        tracker.record(0.46, 12, False)
    # 30/50 = 60% >= 0.52
    assert tracker.should_trade(0.46, 12) is True


def test_should_trade_false_when_win_rate_below_threshold_and_enough_samples(tmp_path):
    # Above the 0.45 early-block bar (so this exercises the n>=50 final
    # check specifically, not the early block) but below the 0.52 bar.
    tracker = _tracker(tmp_path)
    for _ in range(24):
        tracker.record(0.46, 12, True)
    for _ in range(26):
        tracker.record(0.46, 12, False)
    # 24/50 = 48% -- >= 0.45 but < 0.52
    assert tracker.should_trade(0.46, 12) is False


def test_record_updates_stats_correctly(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record(0.46, 12, True)
    tracker.record(0.46, 12, True)
    tracker.record(0.46, 12, False)

    stats = tracker.get_stats()
    row = next(r for r in stats["buckets"] if r["price_bucket"] == 0.45 and r["hour_bucket"] == 12)
    assert row["trades"] == 3
    assert row["win_rate"] == 2 / 3


def test_backfill_loads_historical_fair_value_trades_from_paper_trades_log(tmp_path):
    trades_log = tmp_path / "paper_trades.jsonl"
    before_cutoff = {
        "side": "YES", "entry_price": 0.46, "won": True,
        "opened_at": "2026-07-10T00:00:00+00:00", "closed_at": "2026-07-10T00:10:00+00:00",
    }
    after_cutoff_1 = {
        "side": "YES", "entry_price": 0.46, "won": True,
        "opened_at": "2026-07-14T00:00:00+00:00", "closed_at": "2026-07-14T09:10:00+00:00",
    }
    after_cutoff_2 = {
        "side": "NO", "entry_price": 0.46, "won": False,  # yes_price = 1-0.46 = 0.54
        "opened_at": "2026-07-14T00:05:00+00:00", "closed_at": "2026-07-14T09:20:00+00:00",
    }
    with open(trades_log, "w") as f:
        for t in (before_cutoff, after_cutoff_1, after_cutoff_2):
            f.write(json.dumps(t) + "\n")

    assert before_cutoff["opened_at"] < FAIR_VALUE_DEPLOY_CUTOFF
    assert after_cutoff_1["opened_at"] > FAIR_VALUE_DEPLOY_CUTOFF

    tracker = _tracker(tmp_path, trades_log_path=str(trades_log))
    stats = tracker.get_stats()
    total_loaded = sum(r["trades"] for r in stats["buckets"])
    assert total_loaded == 2  # only the two after-cutoff trades

    row_046 = next(r for r in stats["buckets"] if r["price_bucket"] == 0.45 and r["hour_bucket"] == 6)
    assert row_046["trades"] == 1
    assert row_046["win_rate"] == 1.0

    row_054 = next(r for r in stats["buckets"] if r["price_bucket"] == 0.55 and r["hour_bucket"] == 6)
    assert row_054["trades"] == 1
    assert row_054["win_rate"] == 0.0


def test_hour_bucket_boundaries():
    assert bucket_key(0.50, 0)[1] == 0
    assert bucket_key(0.50, 5)[1] == 0
    assert bucket_key(0.50, 6)[1] == 6
    assert bucket_key(0.50, 11)[1] == 6
    assert bucket_key(0.50, 12)[1] == 12
    assert bucket_key(0.50, 17)[1] == 12
    assert bucket_key(0.50, 18)[1] == 18
    assert bucket_key(0.50, 23)[1] == 18


def test_price_bucket_step_rounding():
    assert bucket_key(0.46, 0)[0] == 0.45
    assert bucket_key(0.49, 0)[0] == 0.50
    assert bucket_key(0.51, 0)[0] == 0.50
    assert bucket_key(0.54, 0)[0] == 0.55


def test_persistence_save_and_load(tmp_path):
    state_file = str(tmp_path / "stats_tracker.json")
    empty_log = str(tmp_path / "no_trades.jsonl")

    tracker = StatsTracker(state_file=state_file, trades_log_path=empty_log)
    tracker.record(0.46, 12, True)
    tracker.record(0.46, 12, False)
    tracker.record(0.54, 18, True)

    reloaded = StatsTracker(state_file=state_file, trades_log_path=empty_log)
    stats = reloaded.get_stats()
    total_trades = sum(r["trades"] for r in stats["buckets"])
    assert total_trades == 3  # loaded from state_file, not re-backfilled/doubled

    row = next(r for r in stats["buckets"] if r["price_bucket"] == 0.45 and r["hour_bucket"] == 12)
    assert row["trades"] == 2
    assert row["win_rate"] == 0.5
