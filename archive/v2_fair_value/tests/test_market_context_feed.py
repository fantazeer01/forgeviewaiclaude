import time

import pytest

import core.market_context as mc


class FakeWebSocketApp:
    calls = []

    def __init__(self, url, on_message=None, on_error=None):
        FakeWebSocketApp.calls.append(url)
        self.on_message = on_message
        self.on_error = on_error

    def run_forever(self, **kwargs):
        if len(FakeWebSocketApp.calls) < 2:
            raise RuntimeError("simulated connection drop")
        time.sleep(0.05)

    def close(self):
        pass


def test_websocket_reconnects_after_drop(monkeypatch):
    FakeWebSocketApp.calls = []
    monkeypatch.setattr(mc.websocket, "WebSocketApp", FakeWebSocketApp)
    monkeypatch.setattr(mc, "BINANCE_RECONNECT_BACKOFF_SEC", 0.01, raising=False)

    feed = mc.BinanceFeed(symbols={"BTC": "btcusdt"})
    feed.start()
    time.sleep(0.3)
    feed.stop()

    assert len(FakeWebSocketApp.calls) >= 2


def test_volume_ratio_zscore_computation():
    feed = mc.BinanceFeed(symbols={"BTC": "btcusdt"})
    baseline = [{"close": 100, "volume": v} for v in [10, 11, 9, 10, 12]]
    recent = [{"close": 100, "volume": v} for v in [20, 22, 19, 21, 20]]
    for bar in baseline + recent:
        feed._klines["BTC"].append(bar)

    ratio = feed.get_volume_ratio("BTC")
    assert ratio is not None
    assert ratio > 0


def test_bid_ask_imbalance_computation():
    feed = mc.BinanceFeed(symbols={"BTC": "btcusdt"})
    feed._depth["BTC"] = {"bid_size": 30, "ask_size": 10}
    imbalance = feed.get_bid_ask_imbalance("BTC")
    assert imbalance == pytest.approx((30 - 10) / (30 + 10))
