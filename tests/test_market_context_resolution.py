from core.market_context import MarketContext


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, **kwargs):
        return FakeResponse(self._payload)


def _context(payload):
    return MarketContext(session=FakeSession(payload))


def test_closed_market_returns_winner_outcome():
    ctx = _context({
        "closed": True,
        "tokens": [
            {"outcome": "Up", "price": 1.0, "winner": True},
            {"outcome": "Down", "price": 0.0, "winner": False},
        ],
    })
    assert ctx.get_resolution("m1") == "UP"


def test_closed_market_no_winner_flag_returns_none():
    ctx = _context({
        "closed": True,
        "tokens": [
            {"outcome": "Up", "price": 0.5, "winner": False},
            {"outcome": "Down", "price": 0.5, "winner": False},
        ],
    })
    assert ctx.get_resolution("m1") is None


def test_unclosed_market_resolves_when_price_above_threshold():
    ctx = _context({
        "closed": False,
        "tokens": [
            {"outcome": "Up", "price": 0.97, "winner": False},
            {"outcome": "Down", "price": 0.03, "winner": False},
        ],
    })
    assert ctx.get_resolution("m1") == "UP"


def test_unclosed_market_below_threshold_still_waits():
    ctx = _context({
        "closed": False,
        "tokens": [
            {"outcome": "Up", "price": 0.80, "winner": False},
            {"outcome": "Down", "price": 0.20, "winner": False},
        ],
    })
    assert ctx.get_resolution("m1") is None


def test_unclosed_market_resolves_by_outcome_label_not_index():
    # "Down" listed first -- an index-based check (winner_idx == 0 -> UP)
    # would wrongly return UP here. Resolution must key off the outcome
    # label, not list position.
    ctx = _context({
        "closed": False,
        "tokens": [
            {"outcome": "Down", "price": 0.03, "winner": False},
            {"outcome": "Up", "price": 0.97, "winner": False},
        ],
    })
    assert ctx.get_resolution("m1") == "UP"


def test_get_resolution_network_error_returns_none():
    class RaisingSession:
        def get(self, url, **kwargs):
            raise ConnectionError("boom")

    ctx = MarketContext(session=RaisingSession())
    assert ctx.get_resolution("m1") is None
