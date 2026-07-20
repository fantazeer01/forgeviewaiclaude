from layer1_eyes.binance_feed import BinanceFeed


def _feed():
    return BinanceFeed(symbols={"BTC": "btcusdt"})


# 1. Binance feed returns spot_price.
def test_binance_feed_returns_spot_price():
    feed = _feed()
    assert feed.get_price("BTC") is None
    feed._handle_kline({"s": "BTCUSDT", "k": {"c": "64000.5", "v": "10", "x": False}})
    assert feed.get_price("BTC") == 64000.5


def test_binance_feed_momentum_needs_finalized_bars():
    feed = _feed()
    for i in range(5):
        feed._handle_kline({"s": "BTCUSDT", "k": {"c": str(64000 + i), "v": "1", "x": True}})
    feed._handle_kline({"s": "BTCUSDT", "k": {"c": "64100", "v": "1", "x": False}})
    momentum = feed.get_momentum_bps("BTC", 5)
    assert momentum is not None


def test_binance_feed_volatility_none_without_enough_bars():
    feed = _feed()
    feed._handle_kline({"s": "BTCUSDT", "k": {"c": "64000", "v": "1", "x": True}})
    assert feed.get_volatility("BTC", 5) is None
