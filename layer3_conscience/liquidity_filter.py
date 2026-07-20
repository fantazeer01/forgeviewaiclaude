"""Layer 3 (conscience): refuse to trade a market that's too thin to fill
or too wide to fill well."""

from config.settings import MIN_BOOK_DEPTH_USD, MAX_BID_ASK_SPREAD_PCT


def passes(snapshot: dict) -> tuple:
    depth_yes = snapshot.get("book_depth_yes") or 0.0
    depth_no = snapshot.get("book_depth_no") or 0.0
    if (depth_yes + depth_no) < MIN_BOOK_DEPTH_USD:
        return False, "low_liquidity"

    spread = snapshot.get("bid_ask_spread")
    if spread is not None and spread > MAX_BID_ASK_SPREAD_PCT:
        return False, "wide_spread"

    return True, None
