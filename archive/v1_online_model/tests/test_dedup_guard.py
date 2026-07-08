import json
import os

from core.dedup_guard import DedupGuard


def test_is_duplicate_false_initially(tmp_path):
    guard = DedupGuard(state_file=str(tmp_path / "state.json"))
    assert guard.is_duplicate("m1") is False


def test_mark_open_and_closed(tmp_path):
    guard = DedupGuard(state_file=str(tmp_path / "state.json"))
    guard.mark_open("m1")
    assert guard.is_duplicate("m1") is True
    guard.mark_closed("m1")
    assert guard.is_duplicate("m1") is False


def test_mark_closed_on_unknown_id_is_noop(tmp_path):
    guard = DedupGuard(state_file=str(tmp_path / "state.json"))
    guard.mark_closed("never-opened")
    assert guard.is_duplicate("never-opened") is False


def test_persistence_across_instances(tmp_path):
    state_file = tmp_path / "state.json"
    guard = DedupGuard(state_file=str(state_file))
    guard.mark_open("m1")
    guard2 = DedupGuard(state_file=str(state_file))
    assert guard2.is_duplicate("m1") is True


def test_save_preserves_other_keys_in_state_file(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"wins": 5}))
    guard = DedupGuard(state_file=str(state_file))
    guard.mark_open("m1")
    data = json.loads(state_file.read_text())
    assert data["wins"] == 5
    assert data["open_market_ids"] == ["m1"]


def test_load_ignores_missing_file(tmp_path):
    guard = DedupGuard(state_file=str(tmp_path / "nonexistent.json"))
    assert guard.is_duplicate("anything") is False


def test_load_recovers_from_corrupt_file(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("not valid json")
    guard = DedupGuard(state_file=str(state_file))
    assert guard.is_duplicate("anything") is False


def test_save_does_not_leave_a_stray_tmp_file(tmp_path):
    state_file = tmp_path / "state.json"
    guard = DedupGuard(state_file=str(state_file))
    guard.mark_open("m1")
    assert not os.path.exists(str(state_file) + ".tmp")


def test_save_uses_atomic_replace_and_leaves_original_intact_on_failure(tmp_path, mocker):
    state_file = tmp_path / "state.json"
    guard = DedupGuard(state_file=str(state_file))
    guard.mark_open("m1")
    original_content = state_file.read_text()

    mocker.patch("core.dedup_guard.os.replace", side_effect=OSError("simulated failure"))
    guard.mark_open("m2")

    assert state_file.read_text() == original_content
