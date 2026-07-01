import datetime

from core.market_fetcher import MarketFetcher


def test_detect_asset_matches_keywords():
    fetcher = MarketFetcher()
    assert fetcher._detect_asset("Will Bitcoin be up in 5 minutes?") == "BTC"
    assert fetcher._detect_asset("ETH price above $3000?") == "ETH"
    assert fetcher._detect_asset("Solana moon soon") == "SOL"
    assert fetcher._detect_asset("Will it rain tomorrow?") is None


def test_is_5min_updown_with_duration_field():
    fetcher = MarketFetcher()
    market = {"question": "Will BTC be up?", "duration": 300}
    assert fetcher._is_5min_updown(market) is True


def test_is_5min_updown_false_without_direction_words():
    fetcher = MarketFetcher()
    market = {"question": "Will BTC reach $100k?", "duration": 300}
    assert fetcher._is_5min_updown(market) is False


def test_is_5min_updown_duration_out_of_range():
    fetcher = MarketFetcher()
    market = {"question": "Will BTC be up?", "duration": 3600}
    assert fetcher._is_5min_updown(market) is False


def test_is_5min_updown_fallback_without_duration():
    fetcher = MarketFetcher()
    market = {"question": "Will BTC be up in 5 minutes?", "duration": None}
    assert fetcher._is_5min_updown(market) is True


def test_is_5min_updown_missing_duration_defaults_to_zero():
    fetcher = MarketFetcher()
    market = {"question": "Will BTC be up in 5 minutes?"}
    assert fetcher._is_5min_updown(market) is False


def test_parse_market_extracts_prices_and_id():
    fetcher = MarketFetcher()
    m = {
        "conditionId": "abc123",
        "question": "Will BTC be up?",
        "tokens": [{"outcome": "Yes", "price": "0.62"}, {"outcome": "No", "price": "0.38"}],
        "volume": "1000.5",
        "endDate": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=3)).isoformat(),
    }
    parsed = fetcher._parse_market(m, "BTC")
    assert parsed["market_id"] == "abc123"
    assert parsed["yes_price"] == 0.62
    assert parsed["no_price"] == 0.38
    assert parsed["volume"] == 1000.5
    assert 2.9 <= parsed["minutes_remaining"] <= 3.1


def test_parse_market_defaults_when_tokens_missing():
    fetcher = MarketFetcher()
    m = {"id": "xyz", "question": "Will ETH be up?"}
    parsed = fetcher._parse_market(m, "ETH")
    assert parsed["market_id"] == "xyz"
    assert parsed["yes_price"] == 0.5
    assert parsed["no_price"] == 0.5
    assert parsed["minutes_remaining"] == 5.0


def test_parse_market_falls_back_to_default_minutes_on_bad_date():
    fetcher = MarketFetcher()
    m = {"id": "xyz", "question": "Will ETH be up?", "endDate": "not-a-date"}
    parsed = fetcher._parse_market(m, "ETH")
    assert parsed["minutes_remaining"] == 5.0


def test_get_active_5min_markets_filters_and_parses(mocker):
    fetcher = MarketFetcher()
    raw_markets = [
        {
            "conditionId": "m1", "question": "Will BTC be up in 5 minutes?",
            "duration": 300,
            "tokens": [{"outcome": "Yes", "price": "0.55"}, {"outcome": "No", "price": "0.45"}],
            "volume": "500",
        },
        {
            "conditionId": "m2", "question": "Will it rain tomorrow?",
            "duration": 300,
        },
        {
            "conditionId": "m3", "question": "Will ETH be up?",
            "duration": 3600,
        },
    ]
    mocker.patch.object(fetcher, "_fetch_gamma_markets", return_value=raw_markets)
    result = fetcher.get_active_5min_markets()
    assert len(result) == 1
    assert result[0]["market_id"] == "m1"
    assert result[0]["asset"] == "BTC"


def test_get_active_5min_markets_returns_empty_on_error(mocker):
    fetcher = MarketFetcher()
    mocker.patch.object(fetcher, "_fetch_gamma_markets", side_effect=RuntimeError("boom"))
    assert fetcher.get_active_5min_markets() == []


def test_fetch_gamma_markets_handles_list_response(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    resp.json.return_value = [{"a": 1}]
    mocker.patch.object(fetcher.session, "get", return_value=resp)
    assert fetcher._fetch_gamma_markets() == [{"a": 1}]


def test_fetch_gamma_markets_handles_dict_response(mocker):
    fetcher = MarketFetcher()
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    resp.json.return_value = {"markets": [{"a": 1}]}
    mocker.patch.object(fetcher.session, "get", return_value=resp)
    assert fetcher._fetch_gamma_markets() == [{"a": 1}]
