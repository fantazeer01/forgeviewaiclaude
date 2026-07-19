import datetime
import json

import pytest

from core.feature_engine import (
    build_features, CrossMarketState, TradeHistory,
    BASE_FEATURE_NAMES, CROSS_MARKET_FEATURE_NAMES, ROLLING_WIN_RATE_NEUTRAL,
)


def _snapshot(asset, yes_price=0.5, seconds_remaining=200, hour_utc=0, weekday=0, **overrides):
    base = {
        "asset": asset, "yes_price": yes_price, "yes_price_change_60s": 0.01, "yes_price_change_120s": 0.02,
        "price_change_1m": 5.0, "price_change_5m": 10.0, "price_change_15m": 15.0,
        "volume_ratio": 1.2, "bid_ask_imbalance": 0.1, "book_imbalance": 0.2, "volume_ratio_window": 0.3,
        "seconds_remaining": seconds_remaining, "hour_utc": hour_utc, "weekday": weekday,
    }
    base.update(overrides)
    return base


# 1. Feature engine generates all 22 features for ETH/SOL, 19 for BTC
# (17 base + rolling_win_rate_1h/6h = 19; + 3 cross-market = 22).
def test_feature_engine_generates_22_features_for_eth_sol():
    btc_snap = _snapshot("BTC")
    features = build_features(_snapshot("ETH"), window_sec=300, btc_snapshot=btc_snap, correlation=0.4)
    expected = set(BASE_FEATURE_NAMES) | set(CROSS_MARKET_FEATURE_NAMES)
    assert set(features.keys()) == expected
    assert len(features) == 22


def test_feature_engine_generates_19_features_for_btc():
    features = build_features(_snapshot("BTC"), window_sec=300)
    assert set(features.keys()) == set(BASE_FEATURE_NAMES)
    assert len(features) == 19


# 12. hour_sin/cos encoding correct (hour=0 -> sin=0, cos=1).
def test_hour_sin_cos_encoding():
    features = build_features(_snapshot("BTC", hour_utc=0), window_sec=300)
    assert features["hour_sin"] == pytest.approx(0.0)
    assert features["hour_cos"] == pytest.approx(1.0)

    features_6 = build_features(_snapshot("BTC", hour_utc=6), window_sec=300)
    assert features_6["hour_sin"] == pytest.approx(1.0)
    assert features_6["hour_cos"] == pytest.approx(0.0)


# 16. btc_momentum influences ETH/SOL features.
def test_btc_momentum_influences_eth_sol_features():
    btc_snap = _snapshot("BTC", price_change_5m=42.0)
    features = build_features(_snapshot("ETH"), window_sec=300, btc_snapshot=btc_snap, correlation=0.0)
    assert features["btc_momentum_5m"] == 42.0
    assert features["btc_yes_price"] == btc_snap["yes_price"]

    features_sol = build_features(_snapshot("SOL"), window_sec=300, btc_snapshot=btc_snap, correlation=0.0)
    assert features_sol["btc_momentum_5m"] == 42.0


# 17. Cross-market correlation updates on every tick (CrossMarketState.update()).
def test_cross_market_correlation_updates_each_tick():
    state = CrossMarketState(window=20)
    assert state.correlation("ETH") is None  # no samples yet

    for i in range(10):
        state.update("ETH", btc_momentum_5m=float(i), asset_momentum_5m=float(i) * 2 + 1)
    corr = state.correlation("ETH")
    assert corr is not None
    assert corr == pytest.approx(1.0, abs=1e-6)  # perfectly linear relationship

    state.update("ETH", btc_momentum_5m=None, asset_momentum_5m=5.0)  # missing sample ignored
    assert len(state._pairs["ETH"]) == 10


# rolling_win_rate_1h / rolling_win_rate_6h: neutral 0.5 when there's no data.
def test_rolling_win_rate_neutral_when_no_trade_history_passed():
    features = build_features(_snapshot("BTC"), window_sec=300)
    assert features["rolling_win_rate_1h"] == ROLLING_WIN_RATE_NEUTRAL
    assert features["rolling_win_rate_6h"] == ROLLING_WIN_RATE_NEUTRAL


def test_rolling_win_rate_neutral_when_trade_history_empty():
    history = TradeHistory()
    features = build_features(_snapshot("BTC"), window_sec=300, trade_history=history)
    assert features["rolling_win_rate_1h"] == 0.5
    assert features["rolling_win_rate_6h"] == 0.5


def test_rolling_win_rate_computed_from_recent_closes():
    now = datetime.datetime(2026, 7, 20, 12, 0, 0, tzinfo=datetime.timezone.utc)
    history = TradeHistory()
    # 3 wins, 1 loss inside the last hour -> 1h win rate 0.75
    for won in (True, True, True, False):
        history.record_close(now - datetime.timedelta(minutes=30), won)
    # 4 more losses between 2h and 5h ago -> only visible in the 6h window
    for _ in range(4):
        history.record_close(now - datetime.timedelta(hours=3), False)

    assert history.win_rate(1, now=now) == pytest.approx(0.75)
    assert history.win_rate(6, now=now) == pytest.approx(3 / 8)


def test_rolling_win_rate_excludes_trades_outside_window():
    now = datetime.datetime(2026, 7, 20, 12, 0, 0, tzinfo=datetime.timezone.utc)
    history = TradeHistory()
    history.record_close(now - datetime.timedelta(hours=10), True)  # outside both windows
    assert history.win_rate(1, now=now) == ROLLING_WIN_RATE_NEUTRAL
    assert history.win_rate(6, now=now) == ROLLING_WIN_RATE_NEUTRAL


def test_trade_history_loads_existing_log_at_startup(tmp_path):
    log_path = tmp_path / "paper_trades_v3.jsonl"
    now = datetime.datetime.now(datetime.timezone.utc)
    lines = [
        json.dumps({"won": True, "closed_at": (now - datetime.timedelta(minutes=10)).isoformat()}),
        json.dumps({"won": False, "closed_at": (now - datetime.timedelta(minutes=20)).isoformat()}),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    history = TradeHistory(log_path=str(log_path))
    assert history.win_rate(1, now=now) == pytest.approx(0.5)


def test_trade_history_record_close_updates_live():
    history = TradeHistory()
    now = datetime.datetime.now(datetime.timezone.utc)
    assert history.win_rate(1, now=now) == ROLLING_WIN_RATE_NEUTRAL
    history.record_close(now, True)
    history.record_close(now, True)
    assert history.win_rate(1, now=now) == pytest.approx(1.0)
