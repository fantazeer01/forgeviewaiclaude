"""Layer 3 (conscience): classifies the current market regime from BTC's own
15m momentum and 5m volatility. Feeds confidence_filter's threshold bump and
bot.py's position-size cut in RANGE."""

import collections
import statistics

from config.settings import REGIME_TREND_MOMENTUM_BPS, REGIME_HIGH_VOL_MULTIPLIER, REGIME_HISTORY_WINDOW

TRENDING_UP = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
RANGE = "RANGE"


class RegimeDetector:
    def __init__(self, window: int = REGIME_HISTORY_WINDOW):
        self._vol_history = collections.deque(maxlen=window)

    def detect(self, spot_momentum_15m, volatility_5m) -> str:
        # median is computed from history BEFORE this sample is added, so a
        # single new spike can't drag its own baseline up and mask itself.
        median_vol = statistics.median(self._vol_history) if len(self._vol_history) >= 3 else None

        regime = RANGE
        if volatility_5m is not None and median_vol is not None and median_vol > 0 \
                and volatility_5m > REGIME_HIGH_VOL_MULTIPLIER * median_vol:
            regime = HIGH_VOLATILITY
        elif spot_momentum_15m is not None:
            low_vol = median_vol is None or volatility_5m is None or volatility_5m <= median_vol
            if spot_momentum_15m > REGIME_TREND_MOMENTUM_BPS and low_vol:
                regime = TRENDING_UP
            elif spot_momentum_15m < -REGIME_TREND_MOMENTUM_BPS and low_vol:
                regime = TRENDING_DOWN

        if volatility_5m is not None:
            self._vol_history.append(volatility_5m)
        return regime
