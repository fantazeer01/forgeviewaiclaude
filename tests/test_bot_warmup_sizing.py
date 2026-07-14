import bot as bot_module


def _make_bot(monkeypatch, tmp_path):
    monkeypatch.setattr("core.executor.PAPER_TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr("bot.momentum_weights_path", lambda a: str(tmp_path / f"momentum_{a}.pkl"))
    monkeypatch.setattr("bot.volume_weights_path", lambda a: str(tmp_path / f"volume_{a}.pkl"))
    monkeypatch.setattr("bot.RISK_STATE_FILE", str(tmp_path / "risk_state.json"))
    return bot_module.Bot()


def test_bot_uses_fixed_warmup_size_for_warmup_trades(monkeypatch, tmp_path):
    b = _make_bot(monkeypatch, tmp_path)
    snapshot = {"market_id": "m1", "seconds_remaining": 300}
    features = {"yes_price": 0.5, "price_momentum_5m": 5.0}
    result = {"mode": "warmup", "decision": "YES", "final_score": None}

    b._maybe_trade("BTC", snapshot, features, result)

    assert "m1" in b.pending
    position_id = b.pending["m1"]["position_id"]
    position = b.executors["BTC"].open_positions[position_id]
    assert position["size_usd"] == 2.0


def test_bot_uses_fixed_size_for_fair_value_trades(monkeypatch, tmp_path):
    # Without the fair_value branch in _maybe_trade's sizing, this would
    # KeyError on result["final_score"] (fair_value never sets it).
    b = _make_bot(monkeypatch, tmp_path)
    snapshot = {"market_id": "m2", "seconds_remaining": 250}
    features = {"yes_price": 0.45}
    result = {"mode": "fair_value", "decision": "YES", "final_score": None}

    b._maybe_trade("BTC", snapshot, features, result)

    assert "m2" in b.pending
    position_id = b.pending["m2"]["position_id"]
    position = b.executors["BTC"].open_positions[position_id]
    assert position["size_usd"] == 2.0
