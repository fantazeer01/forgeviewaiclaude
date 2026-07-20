"""Layer 3 (conscience): don't enter too close to a window's close, and
don't enter in the first 30s of a window while prices are still settling."""

from config.settings import (
    MIN_SECONDS_REMAINING_5M, MIN_SECONDS_REMAINING_15M, MIN_WINDOW_AGE_SEC, TIMEFRAMES,
)

MIN_SECONDS_REMAINING = {"5m": MIN_SECONDS_REMAINING_5M, "15m": MIN_SECONDS_REMAINING_15M}


def passes(timeframe: str, seconds_remaining, window_sec: int = None) -> tuple:
    if seconds_remaining is None:
        return False, "unknown_timing"

    min_remaining = MIN_SECONDS_REMAINING.get(timeframe, 0)
    if seconds_remaining < min_remaining:
        return False, "too_late"

    window_sec = window_sec if window_sec is not None else TIMEFRAMES.get(timeframe)
    if window_sec is not None:
        window_age = window_sec - seconds_remaining
        if window_age < MIN_WINDOW_AGE_SEC:
            return False, "too_early"

    return True, None
