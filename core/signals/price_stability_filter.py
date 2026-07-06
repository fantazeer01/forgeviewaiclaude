import datetime

from config.settings import PRICE_STABILITY_WINDOW_SEC, PRICE_STABILITY_MIN_MOVE


class PriceStabilityFilter:
    """A FILTER, not a signal generator (same category as CorrelationFilter):
    blocks combine() entirely when yes_price has moved less than
    PRICE_STABILITY_MIN_MOVE (0.02) over the trailing
    PRICE_STABILITY_WINDOW_SEC (90s) -- order_book/momentum/volume are all
    continuation signals that only mean something if the market is actually
    moving; a flat market gives them nothing real to detect, so we want to
    trade momentum, not flat markets (2026-07-06 signal quality pass).

    Movement is measured as max-min range over the window, not the
    endpoint-to-endpoint difference -- a price that swings out and back to
    where it started is not "stable," it's just as untradeable-flat by
    round-trip but the range measure still (correctly) doesn't call it
    stable if it moved a lot in between.
    """

    def __init__(self):
        self._history: dict[str, list[dict]] = {}

    def update(self, market_id: str, yes_price: float):
        ts = datetime.datetime.now(datetime.timezone.utc)
        history = self._history.setdefault(market_id, [])
        history.append({"ts": ts, "yes": yes_price})
        cutoff = ts - datetime.timedelta(seconds=PRICE_STABILITY_WINDOW_SEC)
        self._history[market_id] = [p for p in history if p["ts"] > cutoff]

    def is_stable(self, market_id: str) -> bool:
        history = self._history.get(market_id, [])
        if len(history) < 2:
            return False  # not enough data yet -- never block on missing data
        prices = [p["yes"] for p in history]
        return (max(prices) - min(prices)) < PRICE_STABILITY_MIN_MOVE
