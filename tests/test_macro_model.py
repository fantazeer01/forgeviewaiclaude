import pytest

from models.macro_model import macro_bias


def test_bearish_fear_greed_bias():
    assert macro_bias(fear_greed=10, hour_utc=12) == pytest.approx(-0.1)


def test_bullish_fear_greed_bias():
    assert macro_bias(fear_greed=90, hour_utc=12) == pytest.approx(0.1)


def test_asia_close_neutral():
    # Would be bullish (+0.1) outside the Asia-close window; inside it, forced neutral.
    assert macro_bias(fear_greed=90, hour_utc=7) == 0.0


def test_bias_clamped_to_range():
    for fg in [0, 25, 50, 75, 100]:
        for hour in range(24):
            bias = macro_bias(fg, hour)
            assert -0.2 <= bias <= 0.2
