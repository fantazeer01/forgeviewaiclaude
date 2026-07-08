import datetime
import json
import os

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


def test_save_does_not_leave_a_stray_tmp_file(tmp_path):
    state_file = tmp_path / "state.json"
    state = StateManager(state_file=str(state_file))
    state.set("wins", 1)
    assert not os.path.exists(str(state_file) + ".tmp")


def test_save_uses_atomic_replace_and_leaves_original_intact_on_failure(tmp_path, mocker):
    state_file = tmp_path / "state.json"
    state = StateManager(state_file=str(state_file))
    state.set("wins", 5)
    original_content = state_file.read_text()

    mocker.patch("core.state_manager.os.replace", side_effect=OSError("simulated failure"))
    state.set("wins", 999)

    # the failed atomic replace must not have corrupted or truncated the
    # original file -- it should still hold the last successfully-saved state
    assert state_file.read_text() == original_content


# ---------------- session_start_ts (2026-07-08) ----------------

def test_session_start_ts_set_on_genuinely_first_start(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    assert state.get("session_start_ts")  # non-empty


def test_session_start_ts_preserved_across_restart(tmp_path):
    state_file = tmp_path / "state.json"
    first = StateManager(state_file=str(state_file))
    first_ts = first.get("session_start_ts")

    # simulate a bot restart: a brand-new StateManager instance loading the
    # same on-disk file -- session_start_ts must NOT change just because
    # the process restarted (only an explicit Reset Session click should
    # move it).
    second = StateManager(state_file=str(state_file))
    assert second.get("session_start_ts") == first_ts


def test_session_start_ts_migrates_from_old_session_start_clean_field(tmp_path):
    # one-time migration: an existing state.json from before this field
    # existed, but with the old session_start_clean, must carry that value
    # over rather than silently resetting the session on upgrade.
    state_file = tmp_path / "state.json"
    old_ts = "2026-01-01T00:00:00+00:00"
    state_file.write_text(f'{{"session_start_clean": "{old_ts}"}}')
    state = StateManager(state_file=str(state_file))
    assert state.get("session_start_ts") == old_ts


def test_session_start_ts_migration_is_written_to_disk_immediately(tmp_path):
    state_file = tmp_path / "state.json"
    old_ts = "2026-01-01T00:00:00+00:00"
    state_file.write_text(f'{{"session_start_clean": "{old_ts}"}}')
    StateManager(state_file=str(state_file))
    with open(state_file) as f:
        on_disk = json.load(f)
    assert on_disk["session_start_ts"] == old_ts


def test_session_start_still_resets_every_restart_unlike_session_start_ts(tmp_path, mocker):
    # session_start (no _ts/_clean suffix, pre-existing field) DOES reset on
    # every restart by design -- confirms session_start_ts's preserved
    # behavior above is a deliberate difference, not an accident of both
    # fields happening to behave the same way.
    state_file = tmp_path / "state.json"
    fixed_now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    mocker.patch("core.state_manager.datetime.datetime", **{
        "now.return_value": fixed_now, "fromisoformat": datetime.datetime.fromisoformat,
    })
    first = StateManager(state_file=str(state_file))
    session_start_ts_before = first.get("session_start_ts")

    later_now = datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)
    mocker.patch("core.state_manager.datetime.datetime", **{
        "now.return_value": later_now, "fromisoformat": datetime.datetime.fromisoformat,
    })
    second = StateManager(state_file=str(state_file))
    assert second.get("session_start") == later_now.isoformat()
    assert second.get("session_start_ts") == session_start_ts_before
