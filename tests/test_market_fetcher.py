import datetime
import json

from core.market_fetcher import MarketFetcher


def make_market(asset_prefix, boundary, outcomes=("Up", "Down"), prices=("0.55", "0.45"),
                 minutes_from_now=3.0, closed=False, condition_id="cond-1"):
    end_date = (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=minutes_from_now)
    ).isoformat()
    return {
        "slug": f"{asset_prefix}-updown-5m-{boundary}",
        "conditionId": condition_id,
        "question": f"{asset_prefix.upper()} Up or Down",
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps(list(prices)),
        "endDate": end_date,
        "closed": closed,
        "volumeNum": 1234.5,
    }


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
    assert fetcher._asset_from_slug("xrp-updown-5m-1782945000") is None
    assert fetcher._asset_from_slug("some-unrelated-market") is None


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


def test_parse_market_maps_up_down_labels_to_yes_no_prices():
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000, outcomes=("Up", "Down"), prices=("0.62", "0.38"))
    parsed = fetcher._parse_market(market, "BTC", time_module_now())
    assert parsed["market_id"] == "cond-1"
    assert parsed["asset"] == "BTC"
    assert parsed["yes_price"] == 0.62
    assert parsed["no_price"] == 0.38
    assert 2.9 <= parsed["minutes_remaining"] <= 3.1


def test_parse_market_returns_none_when_closed():
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000, closed=True)
    assert fetcher._parse_market(market, "BTC", time_module_now()) is None


def test_parse_market_defaults_prices_when_labels_unrecognized():
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000, outcomes=("Yes", "No"), prices=("0.7", "0.3"))
    parsed = fetcher._parse_market(market, "BTC", time_module_now())
    assert parsed["yes_price"] == 0.5
    assert parsed["no_price"] == 0.5


def test_parse_market_falls_back_to_default_minutes_on_bad_date():
    fetcher = MarketFetcher()
    market = make_market("btc", 1782945000)
    market["endDate"] = "not-a-date"
    parsed = fetcher._parse_market(market, "BTC", time_module_now())
    assert parsed["minutes_remaining"] == 5.0


def test_get_active_5min_markets_filters_and_parses(mocker):
    fetcher = MarketFetcher()
    boundary = fetcher._current_boundary(time_module_now())
    markets = [
        make_market("btc", boundary, prices=("0.55", "0.45")),
        make_market("eth", boundary, prices=("0.40", "0.60")),
        {"slug": "xrp-updown-5m-" + str(boundary), "closed": False},
    ]
    mocker.patch.object(fetcher, "_fetch_markets_by_slug", return_value=markets)
    result = fetcher.get_active_5min_markets()
    assert len(result) == 2
    assets = {m["asset"] for m in result}
    assert assets == {"BTC", "ETH"}


def test_get_active_5min_markets_returns_empty_on_error(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_markets_by_slug", side_effect=RuntimeError("boom"))
    assert fetcher.get_active_5min_markets() == []


def time_module_now():
    import time
    return time.time()
