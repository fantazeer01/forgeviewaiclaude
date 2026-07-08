"""Turns a MarketContext snapshot into the flat feature dict the models
consume."""

FEATURE_NAMES = [
    "price_momentum_1m",
    "price_momentum_5m",
    "volume_ratio",
    "seconds_remaining",
    "time_of_day",
    "fear_greed",
    "yes_price",
    "bid_ask_imbalance",
]


def build_features(context: dict) -> dict:
    return {
        "price_momentum_1m": context.get("momentum_1m_bps") if context.get("momentum_1m_bps") is not None else 0.0,
        "price_momentum_5m": context.get("momentum_5m_bps") if context.get("momentum_5m_bps") is not None else 0.0,
        "volume_ratio": context.get("volume_ratio") if context.get("volume_ratio") is not None else 0.0,
        "seconds_remaining": context.get("seconds_remaining") if context.get("seconds_remaining") is not None else 0.0,
        "time_of_day": context.get("hour_utc") if context.get("hour_utc") is not None else 0,
        "fear_greed": context.get("fear_greed") if context.get("fear_greed") is not None else 50,
        "yes_price": context.get("yes_price") if context.get("yes_price") is not None else 0.5,
        "bid_ask_imbalance": context.get("bid_ask_imbalance") if context.get("bid_ask_imbalance") is not None else 0.0,
    }
