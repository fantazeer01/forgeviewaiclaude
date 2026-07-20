from layer1_eyes.whale_tracker import WhaleTracker


def _tracker(trades):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"trades": trades}

    class FakeSession:
        def get(self, *a, **k):
            return FakeResponse()

    return WhaleTracker(session=FakeSession())


# 5. Whale tracker filters trades > $500.
def test_filters_out_small_trades():
    trades = [
        {"price": "0.5", "size": "100", "side": "BUY", "timestamp": "1000"},   # $50 -- too small
        {"price": "0.5", "size": "2000", "side": "BUY", "timestamp": "1000"},  # $1000 -- whale
    ]
    tracker = _tracker(trades)
    tracker.poll("token123", now=1000.0)
    assert tracker.whale_activity() == 1
    assert tracker.whale_buy_pressure() == 1000.0


def test_exactly_at_threshold_counts():
    trades = [{"price": "1.0", "size": "500", "side": "BUY", "timestamp": "1000"}]  # exactly $500
    tracker = _tracker(trades)
    tracker.poll("token123", now=1000.0)
    assert tracker.whale_activity() == 1


def test_sell_side_counted_separately():
    trades = [
        {"price": "1.0", "size": "600", "side": "BUY", "timestamp": "1000"},
        {"price": "1.0", "size": "1000", "side": "SELL", "timestamp": "1000"},
    ]
    tracker = _tracker(trades)
    tracker.poll("token123", now=1000.0)
    assert tracker.whale_buy_pressure() == 600.0
    assert tracker.whale_sell_pressure() == 1000.0
    assert tracker.whale_imbalance() < 0  # more selling pressure


def test_old_trades_pruned_outside_window():
    tracker = _tracker([])
    tracker._trades.append((0.0, "YES", 1000.0))
    tracker._prune(now=10 * 60)  # 10 minutes later, default window is 5 minutes
    assert tracker.whale_activity() == 0


def test_fetch_failure_returns_zero_pressure():
    class BrokenSession:
        def get(self, *a, **k):
            raise ConnectionError("boom")

    tracker = WhaleTracker(session=BrokenSession())
    tracker.poll("token123", now=1000.0)
    assert tracker.whale_activity() == 0
    assert tracker.whale_imbalance() == 0.0
