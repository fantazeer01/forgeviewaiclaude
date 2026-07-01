from core.state_manager import StateManager


def test_default_state_created_when_missing(tmp_path):
    state_file = tmp_path / "state.json"
    state = StateManager(state_file=str(state_file))
    assert state_file.exists()
    assert state.get("daily_loss_usd") == 0.0
    assert state.get("system_stopped") is False


def test_get_set_and_persistence(tmp_path):
    state_file = tmp_path / "state.json"
    state = StateManager(state_file=str(state_file))
    state.set("wins", 5)
    reloaded = StateManager(state_file=str(state_file))
    assert reloaded.get("wins") == 5


def test_get_returns_default_for_missing_key(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    assert state.get("nonexistent", "fallback") == "fallback"


def test_update_multiple_keys(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    state.update({"wins": 3, "losses": 1})
    assert state.get("wins") == 3
    assert state.get("losses") == 1


def test_stop_system_sets_flags(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    assert state.is_stopped() is False
    state.stop_system("test reason")
    assert state.is_stopped() is True
    assert state.get("stop_reason") == "test reason"


def test_reset_daily_clears_loss_counters(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    state.update({"daily_loss_usd": 25.0, "loss_streak": 3})
    state.reset_daily()
    assert state.get("daily_loss_usd") == 0.0
    assert state.get("loss_streak") == 0


def test_load_merges_saved_state_with_defaults(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text('{"wins": 10}')
    state = StateManager(state_file=str(state_file))
    assert state.get("wins") == 10
    assert state.get("losses") == 0


def test_load_recovers_from_corrupt_file(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("not valid json")
    state = StateManager(state_file=str(state_file))
    assert state.get("wins") == 0
    assert state.get("daily_loss_usd") == 0.0
