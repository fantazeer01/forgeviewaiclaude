import logging
from typing import Optional

from config.settings import ORDER_BOOK_RATIO_THRESHOLD
from core.repricing_detector import RepricingSignal

logger = logging.getLogger(__name__)


class OrderBookSignalGenerator:
    """Fires a YES signal when the live CLOB order book for a market's YES
    token shows much more resting bid depth than ask depth -- more buyers
    than sellers at current prices. Reuses MarketFetcher.get_order_book_top()
    (already used by core/live_features.py), not a separate raw fetch.
    """

    def generate(self, market: dict, fetcher) -> Optional[RepricingSignal]:
        up_token_id = market.get("up_token_id")
        if not up_token_id:
            return None
        top = fetcher.get_order_book_top(up_token_id)
        if not top:
            return None
        bid = top.get("total_bid_depth")
        ask = top.get("total_ask_depth")
        if not bid or not ask or ask <= 0:
            return None
        ratio = bid / ask
        if ratio <= ORDER_BOOK_RATIO_THRESHOLD:
            return None
        confidence = min(0.95, ratio / 4.0)
        return RepricingSignal(
            asset=market["asset"], market_id=market["market_id"], direction="YES",
            yes_price=market["yes_price"], no_price=market["no_price"],
            confidence=round(confidence, 3),
            reason=f"bid/ask depth ratio {ratio:.2f} (bid={bid:.0f}, ask={ask:.0f})",
            minutes_remaining=market.get("minutes_remaining", 5.0),
        )
