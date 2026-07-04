import datetime
import json

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
    assert fetcher._asset_from_slug("some-unrelated-market") is None


def test_asset_from_slug_does_not_match_sol_or_other_unlisted_assets():
    fetcher = MarketFetcher()
    assert fetcher._asset_from_slug("sol-updown-5m-1782945000") is None
    assert fetcher._asset_from_slug("xrp-updown-5m-1782945000") is None


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
        {"slug": "xrp-updown-5m-" + str(boundary), "closed": False},
    ]
    mocker.patch.object(fetcher, "_fetch_markets_by_slug", return_value=markets)
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book(bids=[0.5], asks=[0.52]))
    result = fetcher.get_active_5min_markets()
    assert len(result) == 2
    assets = {m["asset"] for m in result}
    assert assets == {"BTC", "ETH"}


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
                    "total_bid_depth": 150.0, "total_ask_depth": 110.0}


def test_get_order_book_top_handles_empty_book(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book", return_value=make_book_with_sizes())
    top = fetcher.get_order_book_top("token-1")
    assert top == {"best_bid_price": None, "best_bid_size": None,
                    "best_ask_price": None, "best_ask_size": None,
                    "total_bid_depth": 0.0, "total_ask_depth": 0.0}


def test_get_order_book_top_total_depth_sums_all_levels(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_order_book",
                         return_value=make_book_with_sizes(bids=[(0.40, 100), (0.44, 50), (0.30, 25)],
                                                            asks=[(0.46, 30), (0.50, 80), (0.60, 10)]))
    top = fetcher.get_order_book_top("token-1")
    assert top["total_bid_depth"] == 175.0
    assert top["total_ask_depth"] == 120.0


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
