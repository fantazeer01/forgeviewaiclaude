from layer3_conscience import timing_filter
from config.settings import MIN_SECONDS_REMAINING_5M, MIN_SECONDS_REMAINING_15M, MIN_WINDOW_AGE_SEC, TIMEFRAMES


# 11. Timing filter blocks the last 120s (5m) and 360s (15m).
def test_blocks_last_120s_of_5m_window():
    ok, reason = timing_filter.passes("5m", MIN_SECONDS_REMAINING_5M - 1, window_sec=TIMEFRAMES["5m"])
    assert ok is False
    assert reason == "too_late"

    ok, _ = timing_filter.passes("5m", MIN_SECONDS_REMAINING_5M + MIN_WINDOW_AGE_SEC + 1, window_sec=TIMEFRAMES["5m"])
    assert ok is True


def test_blocks_last_360s_of_15m_window():
    ok, reason = timing_filter.passes("15m", MIN_SECONDS_REMAINING_15M - 1, window_sec=TIMEFRAMES["15m"])
    assert ok is False
    assert reason == "too_late"


def test_blocks_first_30s_of_window():
    window_sec = TIMEFRAMES["5m"]
    # 5 seconds into the window -> seconds_remaining is window_sec - 5
    seconds_remaining = window_sec - (MIN_WINDOW_AGE_SEC - 1)
    ok, reason = timing_filter.passes("5m", seconds_remaining, window_sec=window_sec)
    assert ok is False
    assert reason == "too_early"


def test_unknown_timing_blocks():
    ok, reason = timing_filter.passes("5m", None, window_sec=300)
    assert ok is False
    assert reason == "unknown_timing"
