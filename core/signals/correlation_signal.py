import datetime
from typing import Optional

from config.settings import (
    CORRELATION_HIGH_THRESHOLD, CORRELATION_BTC_DROP_THRESHOLD, CORRELATION_BTC_WINDOW_SEC,
)
# CORRELATION_LOW_THRESHOLD (0.3) isn't used as a separate branch: "independent
# -> normal trading" for correlation < 0.3 is already the natural fallthrough
# of should_block() below (it only ever blocks above CORRELATION_HIGH_THRESHOLD),
# not a distinct code path.


class CorrelationFilter:
    """A FILTER, not a signal generator: blocks ETH trades when BTC and ETH
    are moving in lockstep (correlation > 0.8) and BTC's own YES price just
    dropped -- ETH is likely to follow BTC down shortly, making a YES bet on
    ETH riskier right now. When correlation < 0.3, the assets are moving
    independently and this filter never blocks (explicitly "normal
    trading" per spec). BTC signals are never blocked by this filter --
    BTC is the leading indicator here, not the follower.
    """

    def __init__(self):
        self._btc_history: list[dict] = []

    def update_btc_price(self, yes_price: float):
        ts = datetime.datetime.now(datetime.timezone.utc)
        self._btc_history.append({"ts": ts, "yes": yes_price})
        cutoff = ts - datetime.timedelta(seconds=CORRELATION_BTC_WINDOW_SEC)
        self._btc_history = [p for p in self._btc_history if p["ts"] > cutoff]

    def _btc_just_dropped(self) -> bool:
        if len(self._btc_history) < 2:
            return False
        drop = self._btc_history[0]["yes"] - self._btc_history[-1]["yes"]
        return drop >= CORRELATION_BTC_DROP_THRESHOLD

    def should_block(self, asset: str, btc_eth_correlation: Optional[float]) -> bool:
        if asset != "ETH":
            return False
        if btc_eth_correlation is None:
            return False  # no real correlation data yet -- never block on missing data
        if btc_eth_correlation <= CORRELATION_HIGH_THRESHOLD:
            return False
        return self._btc_just_dropped()
