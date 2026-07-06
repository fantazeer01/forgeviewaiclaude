import datetime
import json

import pytest

from core.stats_calculator import StatsCalculator, MIN_DAYS_RUNNING


def write_trades(path, trades):
    with open(path, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def make_trade(trade_id, open_ts, close_ts=None, status="open", pnl_usd=None):
    return {
        "trade_id": trade_id, "open_ts": open_ts, "close_ts": close_ts,
        "status": status, "pnl_usd": pnl_usd,
    }


def iso(days_ago=0, hours_ago=0):
    now = datetime.datetime.now(datetime.timezone.utc)
    return (now - datetime.timedelta(days=days_ago, hours=hours_ago)).isoformat()


def test_empty_file_returns_zeroed_stats(tmp_path):
    calc = StatsCalculator(trades_log=str(tmp_path / "trades.jsonl"), stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    assert stats["n_trades"] == 0
    assert stats["total_pnl_usd"] == 0.0
    assert stats["sharpe_ratio"] is None
    assert stats["avg_rr"] is None
    assert stats["apy_pct"] is None
    assert stats["max_win_streak"] == 0
    assert stats["current_win_streak"] == 0


def test_dedupes_by_trade_id_keeping_last_line(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [
        make_trade("t1", iso(days_ago=1), status="open"),
        make_trade("t1", iso(days_ago=1), close_ts=iso(hours_ago=1), status="win", pnl_usd=10.0),
    ])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    assert stats["n_trades"] == 1
    assert stats["total_pnl_usd"] == 10.0


def test_open_trades_excluded_from_resolved_stats_but_count_toward_days_running(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [
        make_trade("t1", iso(days_ago=2), status="open"),
        make_trade("t2", iso(days_ago=1), close_ts=iso(hours_ago=1), status="win", pnl_usd=10.0),
    ])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    assert stats["n_trades"] == 1  # only the resolved one
    assert stats["days_running"] == pytest.approx(2.0, abs=0.01)  # from the OPEN trade's open_ts


def test_sharpe_ratio_formula(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    pnls = [10.0, -5.0, 8.0, -3.0, 6.0]
    write_trades(path, [
        make_trade(f"t{i}", iso(days_ago=5), close_ts=iso(days_ago=5 - i), status="win" if p > 0 else "loss", pnl_usd=p)
        for i, p in enumerate(pnls)
    ])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    mean_pnl = sum(pnls) / len(pnls)
    variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
    std_pnl = variance ** 0.5
    expected_sharpe = (mean_pnl / std_pnl) * (len(pnls) ** 0.5)
    assert stats["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=0.001)


def test_sharpe_ratio_none_with_fewer_than_2_trades(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [make_trade("t1", iso(days_ago=1), close_ts=iso(), status="win", pnl_usd=10.0)])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    assert calc.compute()["sharpe_ratio"] is None


def test_avg_rr_formula(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [
        make_trade("t1", iso(days_ago=1), close_ts=iso(), status="win", pnl_usd=20.0),
        make_trade("t2", iso(days_ago=1), close_ts=iso(), status="win", pnl_usd=10.0),
        make_trade("t3", iso(days_ago=1), close_ts=iso(), status="loss", pnl_usd=-5.0),
        make_trade("t4", iso(days_ago=1), close_ts=iso(), status="loss", pnl_usd=-15.0),
    ])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    avg_win = (20.0 + 10.0) / 2  # 15.0
    avg_loss = (-5.0 + -15.0) / 2  # -10.0
    assert stats["avg_rr"] == pytest.approx(avg_win / abs(avg_loss))  # 1.5


def test_avg_rr_none_without_both_wins_and_losses(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [make_trade("t1", iso(days_ago=1), close_ts=iso(), status="win", pnl_usd=10.0)])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    assert calc.compute()["avg_rr"] is None


def test_apy_formula(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [make_trade("t1", iso(days_ago=10), close_ts=iso(), status="win", pnl_usd=50.0)])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    # (total_pnl / 100) * 365 / days_running * 100, days_running ~= 10
    expected = (50.0 / 100.0) * 365.0 / stats["days_running"] * 100.0
    assert stats["apy_pct"] == pytest.approx(expected, abs=0.01)


def test_days_running_floors_at_1_hour(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [make_trade("t1", iso(), close_ts=iso(), status="win", pnl_usd=5.0)])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    assert stats["days_running"] >= MIN_DAYS_RUNNING


def test_best_day_pnl_groups_by_close_date(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [
        make_trade("t1", iso(days_ago=2), close_ts="2026-07-01T10:00:00+00:00", status="win", pnl_usd=10.0),
        make_trade("t2", iso(days_ago=2), close_ts="2026-07-01T14:00:00+00:00", status="win", pnl_usd=15.0),
        make_trade("t3", iso(days_ago=1), close_ts="2026-07-02T10:00:00+00:00", status="loss", pnl_usd=-3.0),
    ])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    assert stats["best_day_pnl"] == pytest.approx(25.0)  # Jul 1: 10+15=25, Jul 2: -3


def test_win_streaks(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    results = ["win", "win", "loss", "win", "win", "win", "loss", "win"]
    write_trades(path, [
        make_trade(f"t{i}", iso(days_ago=1), close_ts=f"2026-07-0{i+1}T00:00:00+00:00", status=r,
                   pnl_usd=10.0 if r == "win" else -5.0)
        for i, r in enumerate(results)
    ])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    stats = calc.compute()
    assert stats["max_win_streak"] == 3  # the win,win,win run
    assert stats["current_win_streak"] == 1  # last trade was a win, but preceded by a loss


def test_current_win_streak_zero_when_last_trade_is_a_loss(tmp_path):
    path = str(tmp_path / "trades.jsonl")
    write_trades(path, [
        make_trade("t1", iso(days_ago=1), close_ts="2026-07-01T00:00:00+00:00", status="win", pnl_usd=10.0),
        make_trade("t2", iso(days_ago=1), close_ts="2026-07-02T00:00:00+00:00", status="loss", pnl_usd=-5.0),
    ])
    calc = StatsCalculator(trades_log=path, stats_file=str(tmp_path / "stats.json"))
    assert calc.compute()["current_win_streak"] == 0


def test_export_writes_stats_file(tmp_path):
    trades_path = str(tmp_path / "trades.jsonl")
    stats_path = str(tmp_path / "stats.json")
    write_trades(trades_path, [make_trade("t1", iso(days_ago=1), close_ts=iso(), status="win", pnl_usd=10.0)])
    calc = StatsCalculator(trades_log=trades_path, stats_file=stats_path)
    returned = calc.export()
    with open(stats_path) as f:
        written = json.load(f)
    assert written["n_trades"] == 1
    assert returned["n_trades"] == 1
