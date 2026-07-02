import pytest

from core.kelly_criterion import (
    kelly_fraction,
    kelly_position_size,
    net_odds_from_price,
    quarter_kelly_fraction,
)


def test_net_odds_from_price_matches_binary_payout_formula():
    assert net_odds_from_price(0.30) == pytest.approx((1 - 0.30) / 0.30)
    assert net_odds_from_price(0.50) == pytest.approx(1.0)


def test_net_odds_from_price_returns_zero_for_nonpositive_price():
    assert net_odds_from_price(0.0) == 0.0
    assert net_odds_from_price(-0.1) == 0.0


def test_kelly_fraction_positive_edge():
    # p=0.6, b=1 (even money) -> f = (0.6*1 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)


def test_kelly_fraction_no_edge_is_zero():
    # p=0.5, b=1 -> f = (0.5 - 0.5)/1 = 0
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)


def test_kelly_fraction_negative_edge_clamped_to_zero():
    # p=0.3, b=1 -> raw f = (0.3 - 0.7)/1 = -0.4, clamped to 0
    assert kelly_fraction(0.3, 1.0) == 0.0


def test_kelly_fraction_clamped_to_one_for_out_of_range_probability():
    # for valid p in [0,1] and b>0 the raw formula never exceeds 1 (max is p=1 -> f=1),
    # so exercise the clamp defensively with a malformed p>1 input
    assert kelly_fraction(1.5, 100.0) == 1.0


def test_kelly_fraction_zero_odds_returns_zero():
    assert kelly_fraction(0.9, 0.0) == 0.0
    assert kelly_fraction(0.9, -1.0) == 0.0


def test_quarter_kelly_fraction_applies_default_quarter_multiplier():
    full = kelly_fraction(0.6, 1.0)  # 0.2
    assert quarter_kelly_fraction(0.6, 1.0) == pytest.approx(full * 0.25)


def test_quarter_kelly_fraction_hard_capped_at_kelly_fraction_cap():
    # full kelly = 1.0 (extreme edge), multiplier=1.0 would give 1.0 uncapped,
    # but the hard cap must still clamp the result to KELLY_FRACTION_CAP (0.25)
    assert quarter_kelly_fraction(0.999, 1000.0, multiplier=1.0) <= 0.25


def test_kelly_position_size_scales_with_bankroll():
    size_1000 = kelly_position_size(0.6, 1.0, bankroll=1000.0)
    size_2000 = kelly_position_size(0.6, 1.0, bankroll=2000.0)
    assert size_2000 == pytest.approx(size_1000 * 2)


def test_kelly_position_size_zero_for_nonpositive_bankroll():
    assert kelly_position_size(0.6, 1.0, bankroll=0.0) == 0.0
    assert kelly_position_size(0.6, 1.0, bankroll=-100.0) == 0.0


def test_kelly_position_size_zero_when_no_edge():
    assert kelly_position_size(0.5, 1.0, bankroll=1000.0) == 0.0
