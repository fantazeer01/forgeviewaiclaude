"""Turns a MarketFeed snapshot into the model's feature vector. BTC gets 19
features; ETH and SOL get 22 -- the 3 cross-market features (btc_momentum_5m,
btc_yes_price, correlation_btc_eth) only make sense relative to another
asset, so BTC's own vector never carries them."""

import collections
import datetime
import json
import math
import os

BASE_FEATURE_NAMES = [
    "yes_price", "yes_price_momentum_60s", "yes_price_momentum_120s",
    "distance_from_half", "is_above_half",
    "spot_momentum_1m", "spot_momentum_5m", "spot_momentum_15m",
    "volume_ratio", "bid_ask_imbalance",
    "book_imbalance", "volume_ratio_window",
    "seconds_remaining_pct", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "rolling_win_rate_1h", "rolling_win_rate_6h",
]
CROSS_MARKET_FEATURE_NAMES = ["btc_momentum_5m", "btc_yes_price", "correlation_btc_eth"]

CORRELATION_WINDOW = 20
ROLLING_WIN_RATE_NEUTRAL = 0.5


class TradeHistory:
    """Rolling record of closed-trade outcomes (across all assets/timeframes),
    used to derive the rolling_win_rate_1h / rolling_win_rate_6h features so
    the model can learn which regimes are good to trade in, rather than that
    being hard-coded as a fixed trading-hours window."""

    def __init__(self, log_path: str = None):
        self._log_path = log_path
        self._closes = collections.deque()  # (closed_at: datetime, won: bool)
        if log_path and os.path.exists(log_path):
            self._load(log_path)

    def _load(self, log_path: str):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                closed_at = record.get("closed_at")
                if closed_at is None or "won" not in record:
                    continue
                self._closes.append((_parse_iso(closed_at), bool(record["won"])))

    def record_close(self, closed_at, won: bool):
        if isinstance(closed_at, str):
            closed_at = _parse_iso(closed_at)
        self._closes.append((closed_at, bool(won)))

    def win_rate(self, hours: float, now: datetime.datetime = None) -> float:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        cutoff = now - datetime.timedelta(hours=hours)
        recent = [won for closed_at, won in self._closes if closed_at >= cutoff]
        if not recent:
            return ROLLING_WIN_RATE_NEUTRAL
        return sum(recent) / len(recent)


def _parse_iso(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


class CrossMarketState:
    """Tracks a rolling window of (btc_momentum, asset_momentum) samples per
    non-BTC asset and reports their Pearson correlation -- the
    correlation_btc_eth feature (same key name for both ETH and SOL, per
    spec)."""

    def __init__(self, window: int = CORRELATION_WINDOW):
        self.window = window
        self._pairs = {}  # asset -> deque[(btc_mom, asset_mom)]

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


def build_features(snapshot: dict, window_sec: int, btc_snapshot: dict = None, correlation: float = None,
                    trade_history: TradeHistory = None) -> dict:
    yes_price = snapshot.get("yes_price")
    yes_price = yes_price if yes_price is not None else 0.5

    seconds_remaining = snapshot.get("seconds_remaining")
    seconds_remaining_pct = (seconds_remaining / window_sec) if seconds_remaining is not None else 0.0

    hour = snapshot.get("hour_utc") if snapshot.get("hour_utc") is not None else 0
    weekday = snapshot.get("weekday") if snapshot.get("weekday") is not None else 0

    if trade_history is not None:
        rolling_win_rate_1h = trade_history.win_rate(1)
        rolling_win_rate_6h = trade_history.win_rate(6)
    else:
        rolling_win_rate_1h = ROLLING_WIN_RATE_NEUTRAL
        rolling_win_rate_6h = ROLLING_WIN_RATE_NEUTRAL

    features = {
        "yes_price": yes_price,
        "yes_price_momentum_60s": snapshot.get("yes_price_change_60s") or 0.0,
        "yes_price_momentum_120s": snapshot.get("yes_price_change_120s") or 0.0,
        "distance_from_half": abs(yes_price - 0.5),
        "is_above_half": 1 if yes_price > 0.5 else 0,
        "spot_momentum_1m": snapshot.get("price_change_1m") or 0.0,
        "spot_momentum_5m": snapshot.get("price_change_5m") or 0.0,
        "spot_momentum_15m": snapshot.get("price_change_15m") or 0.0,
        "volume_ratio": snapshot.get("volume_ratio") or 0.0,
        "bid_ask_imbalance": snapshot.get("bid_ask_imbalance") or 0.0,
        "book_imbalance": snapshot.get("book_imbalance") or 0.0,
        "volume_ratio_window": snapshot.get("volume_ratio_window") or 0.0,
        "seconds_remaining_pct": seconds_remaining_pct,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "day_of_week_sin": math.sin(2 * math.pi * weekday / 7),
        "day_of_week_cos": math.cos(2 * math.pi * weekday / 7),
        "rolling_win_rate_1h": rolling_win_rate_1h,
        "rolling_win_rate_6h": rolling_win_rate_6h,
    }

    if snapshot.get("asset") != "BTC" and btc_snapshot is not None:
        btc_yes_price = btc_snapshot.get("yes_price")
        features["btc_momentum_5m"] = btc_snapshot.get("price_change_5m") or 0.0
        features["btc_yes_price"] = btc_yes_price if btc_yes_price is not None else 0.5
        features["correlation_btc_eth"] = correlation if correlation is not None else 0.0

    return features
