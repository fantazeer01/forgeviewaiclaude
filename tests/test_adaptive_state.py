from layer6_memory.adaptive_state import AdaptiveState, HOT, COLD, NEUTRAL
from config.settings import ADAPTIVE_HOT_WIN_RATE, ADAPTIVE_COLD_WIN_RATE, ADAPTIVE_COLD_SIZE_MULTIPLIER


def _state(tmp_path, lookback=20):
    return AdaptiveState(path=str(tmp_path / "adaptive_state.json"), lookback=lookback)


# 26. Adaptive state switches between hot/cold/neutral.
def test_stays_neutral_before_enough_trades(tmp_path):
    state = _state(tmp_path, lookback=20)
    for _ in range(19):
        state.record_close(True)
    assert state.temperature == NEUTRAL


def test_switches_to_hot_on_high_win_rate(tmp_path):
    state = _state(tmp_path, lookback=20)
    assert ADAPTIVE_HOT_WIN_RATE == 0.55
    for i in range(20):
        state.record_close(i < 12)  # 12/20 = 60% win rate
    assert state.temperature == HOT


def test_switches_to_cold_on_low_win_rate(tmp_path):
    state = _state(tmp_path, lookback=20)
    assert ADAPTIVE_COLD_WIN_RATE == 0.45
    for i in range(20):
        state.record_close(i < 6)  # 6/20 = 30% win rate
    assert state.temperature == COLD
    assert state.size_multiplier() == ADAPTIVE_COLD_SIZE_MULTIPLIER


def test_neutral_band_between_hot_and_cold(tmp_path):
    state = _state(tmp_path, lookback=20)
    for i in range(20):
        state.record_close(i < 10)  # exactly 50%
    assert state.temperature == NEUTRAL
    assert state.size_multiplier() == 1.0


def test_rolling_window_drops_old_outcomes(tmp_path):
    state = _state(tmp_path, lookback=20)
    for _ in range(20):
        state.record_close(True)  # 100% -> hot
    assert state.temperature == HOT
    for _ in range(20):
        state.record_close(False)  # rolls the window to 0% -> cold
    assert state.temperature == COLD


def test_export_writes_json(tmp_path):
    path = tmp_path / "adaptive_state.json"
    state = AdaptiveState(path=str(path), lookback=5)
    state.set_regime("TRENDING_UP")
    for _ in range(5):
        state.record_close(True)
    state.export()
    assert path.exists()
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["regime"] == "TRENDING_UP"
    assert data["temperature"] == HOT
