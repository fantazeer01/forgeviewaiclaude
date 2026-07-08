import datetime
import json

import pytest

from core.market_fetcher import MarketFetcher


def make_market(asset_prefix, boundary, outcomes=("Up", "Down"), token_ids=("up-token", "down-token"),
                 minutes_from_now=3.0, closed=False, condition_id="cond-1", volume_24h=5678.9):
    end_date = (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=minutes_from_now)
    ).isoformat()
    return {
        "slug": f"{asset_prefix}-updown-5m-{boundary}",
        "conditionId": condition_id,
        "question": f"{asset_prefix.upper()} Up or Down",
        "outcomes": json.dumps(list(outcomes)),
        "clobTokenIds": json.dumps(list(token_ids)),
        "endDate": end_date,
        "closed": closed,
        "volumeNum": 1234.5,
        "volume24hr": volume_24h,
    }


def make_book(bids=(), asks=()):
    return {
        "bids": [{"price": str(p)} for p in bids],
        "asks": [{"price": str(p)} for p in asks],
    }


def time_module_now():
    import time
    return time.time()


def test_current_boundary_rounds_down_to_5min_mark():
    fetcher = MarketFetcher()
    now = 1782944953  # not aligned to 300s, between 1782944700 and 1782945000
    assert fetcher._current_boundary(now) == 1782944700


def test_current_boundary_returns_same_when_already_aligned():
    fetcher = MarketFetcher()
    now = 1782945000  # exact multiple of 300
    assert fetcher._current_boundary(now) == 1782945000


def test_asset_from_slug_matches_known_prefixes():
    fetcher = MarketFetcher()
    assert fetcher._asset_from_slug("btc-updown-5m-1782945000") == "BTC"
    assert fetcher._asset_from_slug("eth-updown-5m-1782945000") == "ETH"
    assert fetcher._asset_from_slug("sol-updown-5m-1782945000") == "SOL"
    assert fetcher._asset_from_slug("some-unrelated-market") is None


def test_asset_from_slug_does_not_match_other_unlisted_assets():
    fetcher = MarketFetcher()
    assert fetcher._asset_from_slug("xrp-updown-5m-1782945000") is None
    assert fetcher._asset_from_slug("doge-updown-5m-1782945000") is None


def test_fetch_markets_by_slug_sends_slug_params_and_handles_list_response(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    resp.json.return_value = [{"slug": "btc-updown-5m-1"}]
    get_mock = mocker.patch.object(fetcher.session, "get", return_value=resp)
    result = fetcher._fetch_markets_by_slug(["btc-updown-5m-1", "eth-updown-5m-1"])
    assert result == [{"slug": "btc-updown-5m-1"}]
    called_params = get_mock.call_args.kwargs["params"]
    assert called_params == [("slug", "btc-updown-5m-1"), ("slug", "eth-updown-5m-1"), ("limit", 2)]


def test_fetch_markets_by_slug_handles_dict_response(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    resp.json.return_value = {"markets": [{"slug": "btc-updown-5m-1"}]}
    mocker.patch.object(fetcher.session, "get", return_value=resp)
    assert fetcher._fetch_markets_by_slug(["btc-updown-5m-1"]) == [{"slug": "btc-updown-5m-1"}]


def test_best_price_picks_highest_bid_and_lowest_ask():
    fetcher = MarketFetcher()
    levels = [{"price": "0.40"}, {"price": "0.55"}, {"price": "0.48"}]
    assert fetcher._best_price(levels, highest=True) == 0.55
    assert fetcher._best_price(levels, highest=False) == 0.40


def test_best_price_returns_none_for_empty_levels():
    fetcher = MarketFetcher()
    assert fetcher._best_price([], highest=True) is None
    assert fetcher._best_price(None, highest=True) is None


def test_best_price_skips_malformed_levels():
    fetcher = MarketFetcher()
    levels = [{"price": "not-a-number"}, {"price": "0.5"}]
    assert fetcher._best_price(levels, highest=True) == 0.5


def test_token_mid_price_averages_best_bid_and_ask(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book(bids=[0.40, 0.44], asks=[0.46, 0.50]))
    assert fetcher._token_mid_price("token-1") == 0.45


def test_token_mid_price_falls_back_to_ask_only(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book(bids=[], asks=[0.30]))
    assert fetcher._token_mid_price("token-1") == 0.30


def test_token_mid_price_falls_back_to_bid_only(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book(bids=[0.30], asks=[]))
    assert fetcher._token_mid_price("token-1") == 0.30


def test_token_mid_price_returns_none_when_book_empty(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book())
    assert fetcher._token_mid_price("token-1") is None


def test_token_mid_price_returns_none_on_fetch_error(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book", side_effect=RuntimeError("boom"))
    assert fetcher._token_mid_price("token-1") is None


def test_fetch_order_book_sends_token_id_param(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    resp.json.return_value = {"bids": [], "asks": []}
    get_mock = mocker.patch.object(fetcher.session, "get", return_value=resp)
    fetcher._fetch_order_book("token-xyz")
    assert get_mock.call_args.kwargs["params"] == {"token_id": "token-xyz"}


def test_parse_market_uses_clob_mid_prices_for_up_down(mocker):
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000, outcomes=("Up", "Down"), token_ids=("up-tok", "down-tok"))

    def fake_book(token_id):
        if token_id == "up-tok":
            return make_book(bids=[0.60], asks=[0.64])
        return make_book(bids=[0.36], asks=[0.40])

    mocker.patch.object(fetcher, "_fetch_order_book", side_effect=fake_book)
    parsed = fetcher._parse_market(market, "BTC", time_module_now())
    assert parsed["market_id"] == "cond-1"
    assert parsed["asset"] == "BTC"
    assert parsed["yes_price"] == 0.62
    assert parsed["no_price"] == 0.38
    assert 2.9 <= parsed["minutes_remaining"] <= 3.1
    assert parsed["up_token_id"] == "up-tok"
    assert parsed["down_token_id"] == "down-tok"
    assert parsed["volume_24h"] == 5678.9


def test_parse_market_returns_none_when_closed(mocker):
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000, closed=True)
    fetch_mock = mocker.patch.object(fetcher, "_fetch_order_book")
    assert fetcher._parse_market(market, "BTC", time_module_now()) is None
    fetch_mock.assert_not_called()


def test_parse_market_returns_none_when_labels_unrecognized(mocker):
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000, outcomes=("Yes", "No"), token_ids=("yes-tok", "no-tok"))
    fetch_mock = mocker.patch.object(fetcher, "_fetch_order_book")
    assert fetcher._parse_market(market, "BTC", time_module_now()) is None
    fetch_mock.assert_not_called()


def test_parse_market_returns_none_when_order_book_unavailable(mocker):
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000)
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book())
    assert fetcher._parse_market(market, "BTC", time_module_now()) is None


def test_parse_market_falls_back_to_default_minutes_on_bad_date(mocker):
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000)
    market["endDate"] = "not-a-date"
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book(bids=[0.5], asks=[0.5]))
    parsed = fetcher._parse_market(market, "BTC", time_module_now())
    assert parsed["minutes_remaining"] == 5.0


def test_get_active_5min_markets_filters_and_parses(mocker):
    fetcher = MarketFetcher()
    boundary = fetcher._current_boundary(time_module_now())
    markets = [
        make_market("btc", boundary),
        make_market("eth", boundary),
        make_market("sol", boundary),
        {"slug": "xrp-updown-5m-" + str(boundary), "closed": False},
    ]
    mocker.patch.object(fetcher, "_fetch_markets_by_slug", return_value=markets)
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book(bids=[0.5], asks=[0.52]))
    result = fetcher.get_active_5min_markets()
    assert len(result) == 3
    assets = {m["asset"] for m in result}
    assert assets == {"BTC", "ETH", "SOL"}


def test_get_active_5min_markets_requests_all_three_asset_slugs(mocker):
    fetcher = MarketFetcher()
    spy = mocker.patch.object(fetcher, "_fetch_markets_by_slug", return_value=[])
    fetcher.get_active_5min_markets()
    requested_slugs = spy.call_args[0][0]
    prefixes = {slug.split("-")[0] for slug in requested_slugs}
    assert prefixes == {"btc", "eth", "sol"}


def test_get_active_5min_markets_returns_empty_on_error(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_markets_by_slug", side_effect=RuntimeError("boom"))
    assert fetcher.get_active_5min_markets() == []


def test_get_market_resolution_sends_condition_id_in_url(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    resp.json.return_value = {"closed": True}
    get_mock = mocker.patch.object(fetcher.session, "get", return_value=resp)
    result = fetcher.get_market_resolution("0xabc123")
    assert result == {"closed": True}
    called_url = get_mock.call_args.args[0]
    assert called_url.endswith("/markets/0xabc123")


def test_get_market_resolution_returns_none_on_error(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher.session, "get", side_effect=RuntimeError("boom"))
    assert fetcher.get_market_resolution("0xabc123") is None


def test_resolve_outcome_returns_yes_when_up_token_wins():
    fetcher = MarketFetcher()
    resolution = {
        "closed": True,
        "tokens": [
            {"outcome": "Up", "price": 1, "winner": True},
            {"outcome": "Down", "price": 0, "winner": False},
        ],
    }
    assert fetcher.resolve_outcome(resolution) == "YES"


def test_resolve_outcome_returns_no_when_down_token_wins():
    fetcher = MarketFetcher()
    resolution = {
        "closed": True,
        "tokens": [
            {"outcome": "Up", "price": 0, "winner": False},
            {"outcome": "Down", "price": 1, "winner": True},
        ],
    }
    assert fetcher.resolve_outcome(resolution) == "NO"


def test_resolve_outcome_returns_none_when_not_closed():
    fetcher = MarketFetcher()
    resolution = {
        "closed": False,
        "tokens": [
            {"outcome": "Up", "price": 0.5, "winner": False},
            {"outcome": "Down", "price": 0.5, "winner": False},
        ],
    }
    assert fetcher.resolve_outcome(resolution) is None


def test_resolve_outcome_returns_none_when_no_winner_flagged():
    fetcher = MarketFetcher()
    resolution = {
        "closed": True,
        "tokens": [
            {"outcome": "Up", "price": 0.5, "winner": False},
            {"outcome": "Down", "price": 0.5, "winner": False},
        ],
    }
    assert fetcher.resolve_outcome(resolution) is None


def make_book_with_sizes(bids=(), asks=()):
    return {
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    }


def test_best_level_picks_highest_bid_and_lowest_ask_with_size():
    fetcher = MarketFetcher()
    levels = [{"price": "0.40", "size": "10"}, {"price": "0.55", "size": "20"}, {"price": "0.48", "size": "5"}]
    assert fetcher._best_level(levels, highest=True) == (0.55, 20.0)
    assert fetcher._best_level(levels, highest=False) == (0.40, 10.0)


def test_best_level_returns_none_for_empty_levels():
    fetcher = MarketFetcher()
    assert fetcher._best_level([], highest=True) is None
    assert fetcher._best_level(None, highest=True) is None


def test_best_level_defaults_size_to_zero_when_missing():
    fetcher = MarketFetcher()
    assert fetcher._best_level([{"price": "0.5"}], highest=True) == (0.5, 0.0)


def test_get_order_book_top_returns_best_bid_and_ask(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book",
                         return_value=make_book_with_sizes(bids=[(0.40, 100), (0.44, 50)],
                                                            asks=[(0.46, 30), (0.50, 80)]))
    top = fetcher.get_order_book_top("token-1")
    assert top == {"best_bid_price": 0.44, "best_bid_size": 50.0,
                    "best_ask_price": 0.46, "best_ask_size": 30.0,
                    "total_bid_depth": 150.0, "total_ask_depth": 110.0,
                    "bid_depth_top5": 150.0, "ask_depth_top5": 110.0}


def test_get_order_book_top_handles_empty_book(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book_with_sizes())
    top = fetcher.get_order_book_top("token-1")
    assert top == {"best_bid_price": None, "best_bid_size": None,
                    "best_ask_price": None, "best_ask_size": None,
                    "total_bid_depth": 0.0, "total_ask_depth": 0.0,
                    "bid_depth_top5": 0.0, "ask_depth_top5": 0.0}


def test_get_order_book_top_total_depth_sums_all_levels(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book",
                         return_value=make_book_with_sizes(bids=[(0.40, 100), (0.44, 50), (0.30, 25)],
                                                            asks=[(0.46, 30), (0.50, 80), (0.60, 10)]))
    top = fetcher.get_order_book_top("token-1")
    assert top["total_bid_depth"] == 175.0
    assert top["total_ask_depth"] == 120.0


def test_get_order_book_top_depth5_only_sums_the_nearest_5_levels(mocker):
    # 7 bid levels, 6 ask levels -- top5 must exclude the 2 farthest bids
    # and the 1 farthest ask, and must sort by price first (best_level-style
    # generators aren't guaranteed to hand back pre-sorted levels).
    fetcher = MarketFetcher()
    bids = [(0.30, 1), (0.44, 50), (0.40, 100), (0.20, 2), (0.42, 10), (0.10, 3), (0.05, 4)]
    asks = [(0.60, 6), (0.46, 30), (0.50, 80), (0.48, 20), (0.55, 5), (0.70, 7)]
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book_with_sizes(bids=bids, asks=asks))
    top = fetcher.get_order_book_top("token-1")
    # top 5 bids by price desc: 0.44,0.42,0.40,0.30,0.20 -> sizes 50+10+100+1+2 = 163
    assert top["bid_depth_top5"] == 163.0
    # top 5 asks by price asc: 0.46,0.48,0.50,0.55,0.60 -> sizes 30+20+80+5+6 = 141
    assert top["ask_depth_top5"] == 141.0
    assert top["total_bid_depth"] == 170.0  # sanity: all 7 levels
    assert top["total_ask_depth"] == 148.0  # sanity: all 6 levels


def test_get_order_book_top_returns_none_on_fetch_error(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book", side_effect=RuntimeError("boom"))
    assert fetcher.get_order_book_top("token-1") is None


def test_ping_returns_true_on_success(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    mocker.patch.object(fetcher.session, "get", return_value=resp)
    assert fetcher.ping() is True


def test_ping_returns_false_on_error(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher.session, "get", side_effect=RuntimeError("network down"))
    assert fetcher.ping() is False


def test_record_latency_writes_avg_p99_last_to_disk(tmp_path, mocker):
    mocker.patch("core.market_fetcher.LATENCY_LOG", str(tmp_path / "latency.json"))
    fetcher = MarketFetcher()
    for v in [100, 200, 150, 300, 120]:
        fetcher._record_latency(v)
    data = json.loads((tmp_path / "latency.json").read_text())
    assert data["last_ms"] == 120
    assert data["avg_ms"] == pytest.approx((100 + 200 + 150 + 300 + 120) / 5, abs=0.1)
    assert data["sample_count"] == 5
    assert data["p99_ms"] == 300  # with 5 samples, p99 index falls on the max


def test_record_latency_evicts_oldest_beyond_window(mocker):
    mocker.patch("core.market_fetcher.LATENCY_LOG", "unused")
    mocker.patch("core.market_fetcher.LATENCY_WINDOW", 3)
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_export_latency_status")
    for v in [10, 20, 30, 40]:
        fetcher._record_latency(v)
    assert fetcher._latencies_ms == [20, 30, 40]


def test_timed_get_records_a_sample(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    mocker.patch.object(fetcher.session, "get", return_value=resp)
    mocker.patch.object(fetcher, "_export_latency_status")
    result = fetcher._timed_get("http://example.test", timeout=5)
    assert result is resp
    assert len(fetcher._latencies_ms) == 1
    assert fetcher._latencies_ms[0] >= 0


def test_timed_get_records_a_sample_even_when_request_raises(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher.session, "get", side_effect=RuntimeError("boom"))
    mocker.patch.object(fetcher, "_export_latency_status")
    try:
        fetcher._timed_get("http://example.test", timeout=5)
    except RuntimeError:
        pass
    assert len(fetcher._latencies_ms) == 1


def test_timed_get_increments_api_call_count(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    mocker.patch.object(fetcher.session, "get", return_value=resp)
    mocker.patch.object(fetcher, "_export_latency_status")
    mocker.patch.object(fetcher, "_export_api_stats")
    fetcher._timed_get("http://example.test", timeout=5)
    fetcher._timed_get("http://example.test", timeout=5)
    assert fetcher.api_call_count == 2
    assert len(fetcher._api_call_ts) == 2


def test_export_api_stats_writes_calls_last_minute_and_total(tmp_path, mocker):
    mocker.patch("core.market_fetcher.API_STATS_LOG", str(tmp_path / "api_stats.json"))
    fetcher = MarketFetcher()
    now = 1000.0
    # 3 calls within the last 60s, 2 calls older than 60s
    fetcher._api_call_ts = [now - 90, now - 70, now - 50, now - 10, now - 1]
    fetcher.api_call_count = 5
    fetcher._export_api_stats(now)
    data = json.loads((tmp_path / "api_stats.json").read_text())
    assert data["calls_last_minute"] == 3
    assert data["calls_total"] == 5
    assert "last_updated" in data


def test_export_api_stats_prunes_stale_timestamps(mocker):
    mocker.patch("core.market_fetcher.API_STATS_LOG", "unused")
    fetcher = MarketFetcher()
    now = 1000.0
    fetcher._api_call_ts = [now - 90, now - 70, now - 50]
    fetcher._export_api_stats(now)
    assert fetcher._api_call_ts == [now - 50]


def test_maybe_export_api_stats_skips_within_interval(mocker):
    mocker.patch("core.market_fetcher.API_STATS_EXPORT_INTERVAL_SEC", 60)
    fetcher = MarketFetcher()
    export_mock = mocker.patch.object(fetcher, "_export_api_stats")
    fetcher._last_api_stats_export = 1000.0
    fetcher._maybe_export_api_stats(1030.0)  # only 30s elapsed
    export_mock.assert_not_called()


def test_maybe_export_api_stats_fires_after_interval(mocker):
    mocker.patch("core.market_fetcher.API_STATS_EXPORT_INTERVAL_SEC", 60)
    fetcher = MarketFetcher()
    export_mock = mocker.patch.object(fetcher, "_export_api_stats")
    fetcher._last_api_stats_export = 1000.0
    fetcher._maybe_export_api_stats(1065.0)  # 65s elapsed
    export_mock.assert_called_once_with(1065.0)
    assert fetcher._last_api_stats_export == 1065.0
