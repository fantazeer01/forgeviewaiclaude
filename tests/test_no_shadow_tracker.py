import json

import pytest

from core.no_shadow_tracker import NoShadowTracker
from core.online_model import OnlineQuantModel


def make_market(market_id="m1", asset="BTC", yes_price=0.85, no_price=0.15):
    return {"market_id": market_id, "asset": asset, "yes_price": yes_price, "no_price": no_price}


class FakeFetcher:
    def __init__(self, winning_outcome=None):
        self.winning_outcome = winning_outcome

    def get_market_resolution(self, market_id):
        if self.winning_outcome is None:
            return None
        return {"closed": True}

    def resolve_outcome(self, resolution):
        return self.winning_outcome


@pytest.fixture
def tracker(tmp_path):
    return NoShadowTracker(log_path=str(tmp_path / "no_shadow.jsonl"))


@pytest.fixture
def model(tmp_path):
    return OnlineQuantModel(state_path=str(tmp_path / "online_model.pkl"), warmup_trades=5)


def test_maybe_record_above_threshold_records_and_logs(tracker):
    tracker.maybe_record(make_market(yes_price=0.85), {"yes_price": 0.85}, 0.9)
    assert "m1" in tracker._pending
    with open(tracker.log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 1
    assert lines[0]["status"] == "open"
    assert lines[0]["yes_price"] == 0.85


def test_maybe_record_at_or_below_threshold_does_not_record(tracker):
    tracker.maybe_record(make_market(yes_price=0.80), {"yes_price": 0.80}, 0.5)
    tracker.maybe_record(make_market(yes_price=0.50), {"yes_price": 0.50}, 0.5)
    assert tracker._pending == {}
    assert not __import__("os").path.exists(tracker.log_path)


def test_maybe_record_is_idempotent_for_same_market(tracker):
    tracker.maybe_record(make_market(yes_price=0.85), {"yes_price": 0.85}, 0.9)
    tracker.maybe_record(make_market(yes_price=0.90), {"yes_price": 0.90}, 0.9)  # same market_id, later tick
    with open(tracker.log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 1  # not re-recorded


def test_resolve_pending_no_won_feeds_label_0(tracker, model, mocker):
    features = {"yes_price": 0.85}
    tracker.maybe_record(make_market(yes_price=0.85), features, 0.9)
    update_spy = mocker.spy(model, "update")

    tracker.resolve_pending(FakeFetcher(winning_outcome="NO"), model)

    update_spy.assert_called_once_with(features, 0)
    assert "m1" not in tracker._pending
    with open(tracker.log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert lines[-1]["status"] == "resolved"
    assert lines[-1]["outcome"] == "NO"
    assert lines[-1]["no_won"] is True


def test_resolve_pending_yes_won_feeds_label_1(tracker, model, mocker):
    features = {"yes_price": 0.85}
    tracker.maybe_record(make_market(yes_price=0.85), features, 0.9)
    update_spy = mocker.spy(model, "update")

    tracker.resolve_pending(FakeFetcher(winning_outcome="YES"), model)

    update_spy.assert_called_once_with(features, 1)
    with open(tracker.log_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert lines[-1]["no_won"] is False


def test_resolve_pending_noop_when_market_not_yet_resolved(tracker, model, mocker):
    tracker.maybe_record(make_market(yes_price=0.85), {"yes_price": 0.85}, 0.9)
    update_spy = mocker.spy(model, "update")

    tracker.resolve_pending(FakeFetcher(winning_outcome=None), model)

    update_spy.assert_not_called()
    assert "m1" in tracker._pending


def test_restore_pending_reloads_open_entries_after_restart(tmp_path):
    log_path = str(tmp_path / "no_shadow.jsonl")
    t1 = NoShadowTracker(log_path=log_path)
    t1.maybe_record(make_market(yes_price=0.85), {"yes_price": 0.85}, 0.9)

    t2 = NoShadowTracker(log_path=log_path)  # simulates a restart
    assert "m1" in t2._pending
    assert t2._pending["m1"]["features"] == {"yes_price": 0.85}


def test_restore_pending_skips_already_resolved_entries(tmp_path):
    log_path = str(tmp_path / "no_shadow.jsonl")
    t1 = NoShadowTracker(log_path=log_path)
    t1.maybe_record(make_market(yes_price=0.85), {"yes_price": 0.85}, 0.9)
    model = OnlineQuantModel(state_path=str(tmp_path / "online_model.pkl"), warmup_trades=5)
    t1.resolve_pending(FakeFetcher(winning_outcome="YES"), model)

    t2 = NoShadowTracker(log_path=log_path)
    assert t2._pending == {}
