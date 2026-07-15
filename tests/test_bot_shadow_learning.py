import bot as bot_module


def _make_bot(monkeypatch, tmp_path):
    monkeypatch.setattr("core.executor.PAPER_TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr("bot.momentum_weights_path", lambda a: str(tmp_path / f"momentum_{a}.pkl"))
    monkeypatch.setattr("bot.volume_weights_path", lambda a: str(tmp_path / f"volume_{a}.pkl"))
    monkeypatch.setattr("bot.RISK_STATE_FILE", str(tmp_path / "risk_state.json"))
    monkeypatch.setattr("bot.STATS_TRACKER_FILE", str(tmp_path / "stats_tracker.json"))
    monkeypatch.setattr("bot.PAPER_TRADES_LOG", str(tmp_path / "no_history.jsonl"))
    return bot_module.Bot()


def test_shadow_capture_at_midpoint(monkeypatch, tmp_path):
    b = _make_bot(monkeypatch, tmp_path)
    features = {"yes_price": 0.5, "price_momentum_5m": 1.0}

    b._maybe_capture_shadow("BTC", {"market_id": "m1", "seconds_remaining": 300}, features)
    assert "m1" not in b.shadow_pending  # too early, well before the midpoint

    b._maybe_capture_shadow("BTC", {"market_id": "m1", "seconds_remaining": 150}, features)
    assert "m1" in b.shadow_pending
    assert b.shadow_pending["m1"]["asset"] == "BTC"
    assert b.shadow_pending["m1"]["features"] == features


def test_shadow_capture_only_happens_once_per_market(monkeypatch, tmp_path):
    b = _make_bot(monkeypatch, tmp_path)
    features_first = {"yes_price": 0.5, "price_momentum_5m": 1.0}
    features_second = {"yes_price": 0.9, "price_momentum_5m": -9.0}

    b._maybe_capture_shadow("BTC", {"market_id": "m1", "seconds_remaining": 150}, features_first)
    b._maybe_capture_shadow("BTC", {"market_id": "m1", "seconds_remaining": 100}, features_second)

    assert b.shadow_pending["m1"]["features"] == features_first


def test_shadow_resolution_learns_without_opening_position(monkeypatch, tmp_path):
    b = _make_bot(monkeypatch, tmp_path)
    features = {"yes_price": 0.5, "price_momentum_5m": 1.0}
    b.shadow_pending["m1"] = {
        "asset": "BTC",
        "features": features,
        "captured_at": 0.0,
        "seconds_remaining": 0.0,  # already "elapsed" so the resolution check fires immediately
    }
    monkeypatch.setattr(b.context, "get_resolution", lambda market_id: "UP")

    assert b._examples("BTC") == 0
    b._check_shadow_resolutions()

    assert b._examples("BTC") == 1
    assert "m1" not in b.shadow_pending
    assert b.executors["BTC"].open_positions == {}
    assert b.pending == {}


def test_shadow_resolution_waits_for_unresolved_market(monkeypatch, tmp_path):
    b = _make_bot(monkeypatch, tmp_path)
    b.shadow_pending["m1"] = {
        "asset": "BTC",
        "features": {"yes_price": 0.5, "price_momentum_5m": 1.0},
        "captured_at": 0.0,
        "seconds_remaining": 0.0,
    }
    monkeypatch.setattr(b.context, "get_resolution", lambda market_id: None)

    b._check_shadow_resolutions()

    assert "m1" in b.shadow_pending
    assert b._examples("BTC") == 0
