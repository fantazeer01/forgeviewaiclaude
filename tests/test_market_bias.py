from core.market_bias import MarketBiasFetcher, FearGreedFetcher


def make_response(mocker, payload):
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    resp.json.return_value = payload
    return resp


def test_fetch_returns_parsed_fields_and_bullish_bias(mocker):
    mocker.patch(
        "core.market_bias.requests.get",
        return_value=make_response(mocker, {
            "bitcoin": {"usd": 62000.0, "usd_24h_change": 2.5},
            "ethereum": {"usd": 1750.0, "usd_24h_change": 1.1},
            "solana": {"usd": 150.0, "usd_24h_change": 3.3},
        }),
    )
    result = MarketBiasFetcher().fetch()
    assert result == {
        "btc_price": 62000.0, "btc_24h_change": 2.5,
        "eth_price": 1750.0, "eth_24h_change": 1.1,
        "sol_price": 150.0, "sol_24h_change": 3.3,
        "market_bias": "BULLISH",
    }


def test_fetch_still_works_when_solana_missing_from_response(mocker):
    # SOL is best-effort and must never block the BTC-driven bias result
    mocker.patch(
        "core.market_bias.requests.get",
        return_value=make_response(mocker, {
            "bitcoin": {"usd": 62000.0, "usd_24h_change": 2.5},
            "ethereum": {"usd": 1750.0, "usd_24h_change": 1.1},
        }),
    )
    result = MarketBiasFetcher().fetch()
    assert result["sol_price"] is None
    assert result["sol_24h_change"] is None
    assert result["market_bias"] == "BULLISH"


def test_fetch_returns_bearish_bias_below_threshold(mocker):
    mocker.patch(
        "core.market_bias.requests.get",
        return_value=make_response(mocker, {
            "bitcoin": {"usd": 60000.0, "usd_24h_change": -1.5},
            "ethereum": {"usd": 1700.0, "usd_24h_change": -0.9},
        }),
    )
    result = MarketBiasFetcher().fetch()
    assert result["market_bias"] == "BEARISH"


def test_fetch_returns_neutral_bias_within_threshold(mocker):
    mocker.patch(
        "core.market_bias.requests.get",
        return_value=make_response(mocker, {
            "bitcoin": {"usd": 61000.0, "usd_24h_change": 0.3},
            "ethereum": {"usd": 1720.0, "usd_24h_change": -0.2},
        }),
    )
    result = MarketBiasFetcher().fetch()
    assert result["market_bias"] == "NEUTRAL"


def test_bias_from_change_boundary_values():
    assert MarketBiasFetcher.bias_from_change(1.0) == "NEUTRAL"
    assert MarketBiasFetcher.bias_from_change(1.01) == "BULLISH"
    assert MarketBiasFetcher.bias_from_change(-1.0) == "NEUTRAL"
    assert MarketBiasFetcher.bias_from_change(-1.01) == "BEARISH"


def test_fetch_returns_none_on_request_exception(mocker):
    mocker.patch("core.market_bias.requests.get", side_effect=RuntimeError("network down"))
    assert MarketBiasFetcher().fetch() is None


def test_fetch_returns_none_on_incomplete_response(mocker):
    mocker.patch(
        "core.market_bias.requests.get",
        return_value=make_response(mocker, {"bitcoin": {"usd": 62000.0}, "ethereum": {"usd": 1750.0, "usd_24h_change": 1.1}}),
    )
    assert MarketBiasFetcher().fetch() is None


def test_fetch_returns_none_on_missing_coin_key(mocker):
    mocker.patch("core.market_bias.requests.get", return_value=make_response(mocker, {}))
    assert MarketBiasFetcher().fetch() is None


def test_fear_greed_fetch_returns_value_and_classification(mocker):
    mocker.patch(
        "core.market_bias.requests.get",
        return_value=make_response(mocker, {
            "name": "Fear and Greed Index",
            "data": [{"value": "67", "value_classification": "Greed", "timestamp": "1", "time_until_update": "1"}],
            "metadata": {"error": None},
        }),
    )
    result = FearGreedFetcher().fetch()
    assert result == {"value": 67, "classification": "Greed"}


def test_fear_greed_fetch_returns_none_on_empty_data(mocker):
    mocker.patch("core.market_bias.requests.get", return_value=make_response(mocker, {"data": []}))
    assert FearGreedFetcher().fetch() is None


def test_fear_greed_fetch_returns_none_on_request_exception(mocker):
    mocker.patch("core.market_bias.requests.get", side_effect=RuntimeError("network down"))
    assert FearGreedFetcher().fetch() is None
