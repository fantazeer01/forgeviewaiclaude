"""Layer 2 (brain): turns Layer 1's raw feeds + Layer 6's memory into the
model's feature vector.

The spec's category breakdown (Binance 10 + Polymarket 8 + news 3 + macro 2
+ whale 3 + context 4 + memory 3 = 33) is what's actually itemized; its own
"~36 features (BTC)" total line is off by the same margin as the cross-market
count would suggest -- so, as with v3's feature count, the itemized list is
treated as authoritative: BTC gets these 33 base features, ETH/SOL get 33 +
3 cross-market = 36."""

import collections
import math

BASE_FEATURE_NAMES = [
    # Binance (10)
    "spot_momentum_1m", "spot_momentum_5m", "spot_momentum_15m", "spot_momentum_30m",
    "volume_ratio_5m", "volume_trend", "bid_ask_imbalance_binance",
    "volatility_5m", "volatility_15m", "price_acceleration",
    # Polymarket (8)
    "yes_price", "yes_momentum_60s", "yes_momentum_120s", "distance_from_half",
    "book_imbalance_polymarket", "book_depth_ratio", "volume_ratio_window", "seconds_remaining_pct",
    # News (3)
    "news_sentiment_1h", "news_count_1h", "news_has_major",
    # Macro (2)
    "fear_greed_normalized", "fear_greed_change",
    # Whale tracker (3)
    "whale_imbalance", "whale_volume_total", "whale_activity",
    # Context (4)
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    # Memory (3)
    "rolling_win_rate_1h", "rolling_win_rate_6h", "regime_score",
]
CROSS_MARKET_FEATURE_NAMES = ["btc_momentum_5m", "btc_yes_price", "correlation_with_btc"]

CORRELATION_WINDOW = 20

REGIME_SCORES = {
    "TRENDING_UP": 1.0,
    "TRENDING_DOWN": -1.0,
    "HIGH_VOLATILITY": 0.0,
    "RANGE": 0.0,
}


class CrossMarketState:
    """Rolling window of (btc_momentum, asset_momentum) samples per non-BTC
    asset; reports their Pearson correlation as correlation_with_btc."""

    def __init__(self, window: int = CORRELATION_WINDOW):
        self.window = window
        self._pairs = {}

    def update(self, asset: str, btc_momentum_5m, asset_momentum_5m):
        if btc_momentum_5m is None or asset_momentum_5m is None:
            return
        history = self._pairs.setdefault(asset, collections.deque(maxlen=self.window))
        history.append((btc_momentum_5m, asset_momentum_5m))

    def correlation(self, asset: str):
        history = self._pairs.get(asset)
        if not history or len(history) < 3:
            return None
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return 0.0
        return cov / denom


def build_features(snapshot: dict, window_sec: int, *,
                    news: dict = None, fear_greed: dict = None, whale: dict = None,
                    memory: dict = None, regime: str = None,
                    btc_snapshot: dict = None, correlation: float = None) -> dict:
    """snapshot: merged Binance+Polymarket fields for one (asset, timeframe),
    same shape as layer1_eyes' combined feed snapshot. news/fear_greed/whale/
    memory are small dicts of already-aggregated values from their own
    layer1/layer6 objects -- see bot.py for how they're assembled."""
    news = news or {}
    fear_greed = fear_greed or {}
    whale = whale or {}
    memory = memory or {}

    yes_price = snapshot.get("yes_price")
    yes_price = yes_price if yes_price is not None else 0.5

    seconds_remaining = snapshot.get("seconds_remaining")
    seconds_remaining_pct = (seconds_remaining / window_sec) if seconds_remaining is not None else 0.0

    hour = snapshot.get("hour_utc") if snapshot.get("hour_utc") is not None else 0
    weekday = snapshot.get("weekday") if snapshot.get("weekday") is not None else 0

    momentum_1m = snapshot.get("price_change_1m") or 0.0
    momentum_5m = snapshot.get("price_change_5m") or 0.0

    features = {
        # Binance
        "spot_momentum_1m": momentum_1m,
        "spot_momentum_5m": momentum_5m,
        "spot_momentum_15m": snapshot.get("price_change_15m") or 0.0,
        "spot_momentum_30m": snapshot.get("price_change_30m") or 0.0,
        "volume_ratio_5m": snapshot.get("volume_ratio") or 0.0,
        "volume_trend": snapshot.get("volume_trend") or 0.0,
        "bid_ask_imbalance_binance": snapshot.get("bid_ask_imbalance") or 0.0,
        "volatility_5m": snapshot.get("volatility_5m") or 0.0,
        "volatility_15m": snapshot.get("volatility_15m") or 0.0,
        "price_acceleration": momentum_1m - momentum_5m,
        # Polymarket
        "yes_price": yes_price,
        "yes_momentum_60s": snapshot.get("yes_price_change_60s") or 0.0,
        "yes_momentum_120s": snapshot.get("yes_price_change_120s") or 0.0,
        "distance_from_half": abs(yes_price - 0.5),
        "book_imbalance_polymarket": snapshot.get("book_imbalance") or 0.0,
        "book_depth_ratio": snapshot.get("book_depth_ratio") if snapshot.get("book_depth_ratio") is not None else 0.5,
        "volume_ratio_window": snapshot.get("volume_ratio_window") or 0.0,
        "seconds_remaining_pct": seconds_remaining_pct,
        # News
        "news_sentiment_1h": news.get("sentiment_1h", 0.0),
        "news_count_1h": news.get("count_1h", 0),
        "news_has_major": 1 if news.get("has_major") else 0,
        # Macro
        "fear_greed_normalized": fear_greed.get("normalized", 0.5),
        "fear_greed_change": fear_greed.get("change_24h", 0.0),
        # Whale tracker
        "whale_imbalance": whale.get("imbalance", 0.0),
        "whale_volume_total": whale.get("volume_total", 0.0),
        "whale_activity": whale.get("activity", 0),
        # Context
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "day_sin": math.sin(2 * math.pi * weekday / 7),
        "day_cos": math.cos(2 * math.pi * weekday / 7),
        # Memory
        "rolling_win_rate_1h": memory.get("win_rate_1h", 0.5),
        "rolling_win_rate_6h": memory.get("win_rate_6h", 0.5),
        "regime_score": REGIME_SCORES.get(regime, 0.0),
    }

    if snapshot.get("asset") != "BTC" and btc_snapshot is not None:
        btc_yes_price = btc_snapshot.get("yes_price")
        features["btc_momentum_5m"] = btc_snapshot.get("price_change_5m") or 0.0
        features["btc_yes_price"] = btc_yes_price if btc_yes_price is not None else 0.5
        features["correlation_with_btc"] = correlation if correlation is not None else 0.0

    return features
