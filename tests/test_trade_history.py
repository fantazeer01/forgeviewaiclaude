import datetime
import json

from layer6_memory.trade_history import TradeHistory, ROLLING_WIN_RATE_NEUTRAL


# 24. Trade history loads existing history at startup.
def test_loads_existing_log_at_startup(tmp_path):
    log_path = tmp_path / "paper_trades_v4.jsonl"
    now = datetime.datetime.now(datetime.timezone.utc)
    lines = [
        json.dumps({"won": True, "closed_at": (now - datetime.timedelta(minutes=10)).isoformat()}),
        json.dumps({"won": False, "closed_at": (now - datetime.timedelta(minutes=20)).isoformat()}),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    history = TradeHistory(log_path=str(log_path))
    assert len(history) == 2
    assert history.win_rate(1, now=now) == 0.5


def test_neutral_when_no_data():
    history = TradeHistory()
    assert history.win_rate(1) == ROLLING_WIN_RATE_NEUTRAL


def test_record_close_updates_live():
    history = TradeHistory()
    now = datetime.datetime.now(datetime.timezone.utc)
    history.record_close(now, True)
    history.record_close(now, True)
    history.record_close(now, False)
    assert history.win_rate(1, now=now) == 2 / 3
