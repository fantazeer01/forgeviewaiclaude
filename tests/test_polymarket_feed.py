from layer1_eyes.polymarket_feed import PolymarketFeed, compute_book_imbalance


def _market_payload():
    return {
        "conditionId": "0xabc",
        "outcomes": '["Up", "Down"]',
        "clobTokenIds": '["up-token", "down-token"]',
        "endDate": "",
    }


# 2. Polymarket feed reads YES and NO prices independently (not 1-yes).
def test_yes_and_no_prices_are_independent():
    feed = PolymarketFeed()

    def fake_fetch_book(token_id):
        if token_id == "up-token":
            return {"bids": [{"price": "0.60", "size": "100"}], "asks": [{"price": "0.62", "size": "50"}]}
        return {"bids": [{"price": "0.35", "size": "80"}], "asks": [{"price": "0.37", "size": "40"}]}

    feed._fetch_book = fake_fetch_book
    parsed = feed._parse_market(_market_payload(), window_sec=300)

    assert parsed["yes_price"] == 0.61
    assert parsed["no_price"] == 0.36
    # a book that were derived as 1-yes would give 0.39, not 0.36 -- confirms independence
    assert parsed["no_price"] != round(1 - parsed["yes_price"], 2)


def test_no_price_falls_back_to_1_minus_yes_when_no_book_empty():
    feed = PolymarketFeed()

    def fake_fetch_book(token_id):
        if token_id == "up-token":
            return {"bids": [{"price": "0.60", "size": "100"}], "asks": [{"price": "0.62", "size": "50"}]}
        return None

    feed._fetch_book = fake_fetch_book
    parsed = feed._parse_market(_market_payload(), window_sec=300)
    assert parsed["no_price"] == 1 - parsed["yes_price"]


def test_book_imbalance_bounded():
    assert compute_book_imbalance(100, 0) == 1.0
    assert compute_book_imbalance(0, 100) == -1.0
    assert compute_book_imbalance(0, 0) is None
    assert -1.0 <= compute_book_imbalance(30, 70) <= 1.0
