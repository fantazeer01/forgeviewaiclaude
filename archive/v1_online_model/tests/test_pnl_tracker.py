import json

import pytest

from core.pnl_tracker import PnLTracker


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    trades_log = tmp_path / "trades.jsonl"
    monkeypatch.setattr("core.pnl_tracker.TRADES_LOG", str(trades_log))
    return PnLTracker(), trades_log


def write_trades(path, trades):
    with open(path, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def test_empty_stats_when_no_log(tracker):
    pnl_tracker, _ = tracker
    stats = pnl_tracker.compute_stats()
    assert stats == {
        "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "expectancy_usd": 0.0, "total_pnl_usd": 0.0, "max_drawdown_usd": 0.0, "by_asset": {},
    }


def test_compute_stats_basic_metrics(tracker):
    pnl_tracker, trades_log = tracker
    trades = [
        {"trade_id": "t1", "asset": "BTC", "status": "win", "pnl_usd": 10},
        {"trade_id": "t2", "asset": "BTC", "status": "loss", "pnl_usd": -5},
        {"trade_id": "t3", "asset": "ETH", "status": "loss", "pnl_usd": -20},
        {"trade_id": "t4", "asset": "ETH", "status": "win", "pnl_usd": 15},
    ]
    write_trades(trades_log, trades)
    stats = pnl_tracker.compute_stats()
    assert stats["total_trades"] == 4
    assert stats["wins"] == 2
    assert stats["losses"] == 2
    assert stats["win_rate"] == 0.5
    assert stats["total_pnl_usd"] == 0.0
    assert stats["expectancy_usd"] == 0.0
    assert stats["max_drawdown_usd"] == 25.0
    assert stats["by_asset"]["BTC"] == {"trades": 2, "wins": 1, "pnl": 5.0}
    assert stats["by_asset"]["ETH"] == {"trades": 2, "wins": 1, "pnl": -5.0}


def test_compute_stats_deduplicates_by_trade_id_keeping_latest(tracker):
    pnl_tracker, trades_log = tracker
    trades = [
        {"trade_id": "t1", "asset": "BTC", "status": "open", "pnl_usd": None},
        {"trade_id": "t1", "asset": "BTC", "status": "win", "pnl_usd": 10},
    ]
    write_trades(trades_log, trades)
    stats = pnl_tracker.compute_stats()
    assert stats["total_trades"] == 1
    assert stats["total_pnl_usd"] == 10.0


def test_compute_stats_excludes_still_open_trades(tracker):
    pnl_tracker, trades_log = tracker
    trades = [{"trade_id": "t1", "asset": "BTC", "status": "open", "pnl_usd": None}]
    write_trades(trades_log, trades)
    stats = pnl_tracker.compute_stats()
    assert stats["total_trades"] == 0


def test_compute_stats_ignores_blank_lines(tracker):
    pnl_tracker, trades_log = tracker
    trades_log.write_text(
        json.dumps({"trade_id": "t1", "asset": "BTC", "status": "win", "pnl_usd": 5}) + "\n\n"
    )
    stats = pnl_tracker.compute_stats()
    assert stats["total_trades"] == 1
