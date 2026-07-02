import datetime

import pytest

from core.quant_features import QuantFeatureExtractor


class FakeFetcher:
    def __init__(self, top=None):
        self._top = top
        self.calls = []

    def get_order_book_top(self, token_id):
        self.calls.append(token_id)
        return self._top


def make_market(market_id="m1", asset="BTC", yes_price=0.4, no_price=0.6,
                 minutes_remaining=3.0, up_token_id="up-tok", volume_24h=1000.0):
    return {
        "market_id": market_id,
        "asset": asset,
        "yes_price": yes_price,
        "no_price": no_price,
        "minutes_remaining": minutes_remaining,
        "up_token_id": up_token_id,
        "volume_24h": volume_24h,
    }


def test_update_prunes_history_older_than_window():
    extractor = QuantFeatureExtractor()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_ts = now - datetime.timedelta(seconds=200)
    extractor._price_history["m1"] = [{"ts": old_ts, "yes": 0.5, "no": 0.5}]
    extractor.update("m1", 0.4, 0.6)
    assert len(extractor._price_history["m1"]) == 1
    assert extractor._price_history["m1"][0]["yes"] == 0.4


def test_price_velocity_none_without_enough_history():
    extractor = QuantFeatureExtractor()
    extractor.update("m1", 0.5, 0.5)
    assert extractor.price_velocity("m1", 0.5) is None


def test_price_velocity_computed_from_60s_ago_reference():
    extractor = QuantFeatureExtractor()
    now = datetime.datetime.now(datetime.timezone.utc)
    ref_ts = now - datetime.timedelta(seconds=60)
    extractor._price_history["m1"] = [{"ts": ref_ts, "yes": 0.40, "no": 0.60}]
    velocity = extractor.price_velocity("m1", 0.46, now=now)
    assert velocity == pytest.approx((0.46 - 0.40) / 60, rel=1e-3)


def test_price_acceleration_none_without_90s_of_history():
    extractor = QuantFeatureExtractor()
    now = datetime.datetime.now(datetime.timezone.utc)
    extractor._price_history["m1"] = [{"ts": now - datetime.timedelta(seconds=60), "yes": 0.40, "no": 0.60}]
    assert extractor.price_acceleration("m1", 0.46, now=now) is None


def test_price_acceleration_positive_when_velocity_increasing():
    extractor = QuantFeatureExtractor()
    now = datetime.datetime.now(datetime.timezone.utc)
    # prices at t-90, t-30 give an earlier velocity of (0.40-0.30)/60 = +0.00167/s
    # price now vs t-60 gives velocity_now = (0.60-0.40)/60 = +0.00333/s -> accelerating
    extractor._price_history["m1"] = [
        {"ts": now - datetime.timedelta(seconds=90), "yes": 0.30, "no": 0.70},
        {"ts": now - datetime.timedelta(seconds=60), "yes": 0.40, "no": 0.60},
        {"ts": now - datetime.timedelta(seconds=30), "yes": 0.40, "no": 0.60},
    ]
    accel = extractor.price_acceleration("m1", 0.60, now=now)
    assert accel is not None
    assert accel > 0


def test_imbalance_from_top_positive_when_more_bid_depth():
    top = {"total_bid_depth": 1000.0, "total_ask_depth": 200.0}
    assert QuantFeatureExtractor._imbalance_from_top(top) == pytest.approx((1000 - 200) / 1200)


def test_imbalance_from_top_none_when_missing():
    assert QuantFeatureExtractor._imbalance_from_top(None) is None
    assert QuantFeatureExtractor._imbalance_from_top({"total_bid_depth": None, "total_ask_depth": 5}) is None


def test_spread_from_top_computes_ask_minus_bid():
    top = {"best_bid_price": 0.40, "best_ask_price": 0.44}
    assert QuantFeatureExtractor._spread_from_top(top) == pytest.approx(0.04)


def test_spread_from_top_none_when_missing():
    assert QuantFeatureExtractor._spread_from_top(None) is None


def test_extract_returns_all_expected_keys():
    extractor = QuantFeatureExtractor()
    fetcher = FakeFetcher(top={"best_bid_price": 0.39, "best_bid_size": 50.0,
                                "best_ask_price": 0.41, "best_ask_size": 30.0,
                                "total_bid_depth": 1500.0, "total_ask_depth": 500.0})
    market = make_market(yes_price=0.40, no_price=0.60, minutes_remaining=2.5)
    snapshot = extractor.extract(market, fetcher)
    assert set(snapshot.keys()) == {
        "yes_price", "no_price", "price_velocity", "price_acceleration",
        "order_book_imbalance", "volume_24h", "time_remaining_pct", "spread",
        "spread_compression",
    }
    assert snapshot["yes_price"] == 0.40
    assert snapshot["no_price"] == 0.60
    assert snapshot["time_remaining_pct"] == pytest.approx(2.5 / 5.0)
    assert snapshot["volume_24h"] == 1000.0
    assert snapshot["order_book_imbalance"] == pytest.approx((1500 - 500) / 2000)
    assert snapshot["spread"] == pytest.approx(0.02)
    assert snapshot["spread_compression"] is None  # no prior spread observed yet
    assert fetcher.calls == ["up-tok"]


def test_extract_handles_missing_up_token_id_gracefully():
    extractor = QuantFeatureExtractor()
    fetcher = FakeFetcher(top=None)
    market = make_market(up_token_id=None)
    snapshot = extractor.extract(market, fetcher)
    assert snapshot["order_book_imbalance"] is None
    assert snapshot["spread"] is None
    assert snapshot["spread_compression"] is None
    assert fetcher.calls == []


def test_spread_compression_none_on_first_observation():
    extractor = QuantFeatureExtractor()
    assert extractor._spread_compression("m1", 0.05) is None


def test_spread_compression_positive_when_spread_narrows():
    extractor = QuantFeatureExtractor()
    extractor._spread_compression("m1", 0.05)
    compression = extractor._spread_compression("m1", 0.03)
    assert compression == pytest.approx(0.02)


def test_spread_compression_negative_when_spread_widens():
    extractor = QuantFeatureExtractor()
    extractor._spread_compression("m1", 0.03)
    compression = extractor._spread_compression("m1", 0.05)
    assert compression == pytest.approx(-0.02)


def test_spread_compression_none_when_spread_is_none():
    extractor = QuantFeatureExtractor()
    extractor._spread_compression("m1", 0.05)
    assert extractor._spread_compression("m1", None) is None
