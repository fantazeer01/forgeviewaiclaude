import datetime

from core.signals.correlation_signal import CorrelationFilter


def test_never_blocks_btc():
    f = CorrelationFilter()
    f.update_btc_price(0.50)
    f.update_btc_price(0.40)  # big drop
    assert f.should_block("BTC", btc_eth_correlation=0.95) is False


def test_blocks_eth_when_high_correlation_and_btc_dropped():
    f = CorrelationFilter()
    f.update_btc_price(0.50)
    f.update_btc_price(0.45)
    f.update_btc_price(0.40)  # dropped 0.10, well above the 0.02 threshold
    assert f.should_block("ETH", btc_eth_correlation=0.85) is True


def test_does_not_block_eth_when_correlation_at_or_below_high_threshold():
    f = CorrelationFilter()
    f.update_btc_price(0.50)
    f.update_btc_price(0.40)
    assert f.should_block("ETH", btc_eth_correlation=0.8) is False


def test_does_not_block_eth_when_correlation_low_even_if_btc_dropped():
    f = CorrelationFilter()
    f.update_btc_price(0.50)
    f.update_btc_price(0.40)
    assert f.should_block("ETH", btc_eth_correlation=0.2) is False


def test_does_not_block_eth_when_btc_did_not_drop():
    f = CorrelationFilter()
    f.update_btc_price(0.40)
    f.update_btc_price(0.50)  # rose, not dropped
    assert f.should_block("ETH", btc_eth_correlation=0.9) is False


def test_does_not_block_when_correlation_is_none():
    f = CorrelationFilter()
    f.update_btc_price(0.50)
    f.update_btc_price(0.40)
    assert f.should_block("ETH", btc_eth_correlation=None) is False


def test_does_not_block_with_insufficient_btc_history():
    f = CorrelationFilter()
    f.update_btc_price(0.40)  # only one sample
    assert f.should_block("ETH", btc_eth_correlation=0.9) is False


def test_btc_history_pruned_outside_window():
    f = CorrelationFilter()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_ts = now - datetime.timedelta(seconds=100)
    f._btc_history = [{"ts": old_ts, "yes": 0.9}]
    f.update_btc_price(0.5)
    assert len(f._btc_history) == 1
    assert f._btc_history[0]["yes"] == 0.5
