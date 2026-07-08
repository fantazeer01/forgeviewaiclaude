from core.state_manager import StateManager
from reporting.telegram_reporter import TelegramReporter
from run import _auto_reset_on_stop


def test_auto_reset_on_stop_clears_stop_flags_and_counters(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    state.update({"daily_loss_usd": 40.0, "loss_streak": 5})
    state.stop_system("Loss streak limit hit")
    assert state.is_stopped() is True

    _auto_reset_on_stop(state, TelegramReporter())

    assert state.is_stopped() is False
    assert state.get("stop_reason") == ""
    assert state.get("daily_loss_usd") == 0.0
    assert state.get("loss_streak") == 0


def test_auto_reset_on_stop_is_noop_safe_when_reason_missing(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    state.stop_system("")

    _auto_reset_on_stop(state, TelegramReporter())

    assert state.is_stopped() is False
    assert state.get("stop_reason") == ""
