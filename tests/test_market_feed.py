import pytest

from core.market_feed import compute_book_imbalance


# 13. book_imbalance is always within [-1, 1].
@pytest.mark.parametrize(
    "bid_size, ask_size",
    [
        (0, 0),
        (100, 0),
        (0, 100),
        (50, 50),
        (1, 1000),
        (1000, 1),
        (0.0001, 0.0002),
        (12345.678, 1.23),
    ],
)
def test_book_imbalance_bounded(bid_size, ask_size):
    imbalance = compute_book_imbalance(bid_size, ask_size)
    if bid_size + ask_size == 0:
        assert imbalance is None
    else:
        assert -1.0 <= imbalance <= 1.0


def test_book_imbalance_extremes():
    assert compute_book_imbalance(100, 0) == pytest.approx(1.0)
    assert compute_book_imbalance(0, 100) == pytest.approx(-1.0)
    assert compute_book_imbalance(50, 50) == pytest.approx(0.0)


def test_book_imbalance_empty_book_is_none():
    assert compute_book_imbalance(0, 0) is None
