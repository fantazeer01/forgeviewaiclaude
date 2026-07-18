"""Rule-based macro bias -- not ML, just a few thresholds on Fear&Greed and
time-of-day. Returns a float in [-MACRO_BIAS_CLAMP, +MACRO_BIAS_CLAMP]."""

from config.settings import (
    MACRO_FEAR_GREED_BEARISH, MACRO_FEAR_GREED_BULLISH,
    MACRO_BEARISH_BIAS, MACRO_BULLISH_BIAS,
    MACRO_ASIA_CLOSE_UTC, MACRO_NYSE_OPEN_UTC,
    MACRO_VOLATILITY_BIAS, MACRO_BIAS_CLAMP,
)


def macro_bias(fear_greed: int, hour_utc: int) -> float:
    bias = 0.0
    if fear_greed is not None:
        if fear_greed < MACRO_FEAR_GREED_BEARISH:
            bias += MACRO_BEARISH_BIAS
        elif fear_greed > MACRO_FEAR_GREED_BULLISH:
            bias += MACRO_BULLISH_BIAS

    asia_start, asia_end = MACRO_ASIA_CLOSE_UTC
    if hour_utc is not None and asia_start <= hour_utc < asia_end:
        bias = 0.0

    nyse_start, nyse_end = MACRO_NYSE_OPEN_UTC
    if hour_utc is not None and nyse_start <= hour_utc < nyse_end:
        bias += MACRO_VOLATILITY_BIAS if bias >= 0 else -MACRO_VOLATILITY_BIAS

    return max(-MACRO_BIAS_CLAMP, min(MACRO_BIAS_CLAMP, bias))
