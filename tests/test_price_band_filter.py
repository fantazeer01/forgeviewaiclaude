from layer3_conscience import price_band_filter
from config.settings import ENTRY_YES_PRICE_MIN, ENTRY_YES_PRICE_MAX


# Regression test for the dropped-during-v4-rebuild entry price band: a
# market at yes_price=0.835 (like the real ETH trade that exposed this bug)
# must never be allowed to open.
def test_blocks_the_real_bug_case():
    ok, reason = price_band_filter.passes(0.835)
    assert ok is False
    assert reason == "price_out_of_band"


def test_blocks_below_min():
    ok, reason = price_band_filter.passes(ENTRY_YES_PRICE_MIN - 0.01)
    assert ok is False
    assert reason == "price_out_of_band"


def test_blocks_above_max():
    ok, reason = price_band_filter.passes(ENTRY_YES_PRICE_MAX + 0.01)
    assert ok is False
    assert reason == "price_out_of_band"


def test_passes_within_band():
    ok, _ = price_band_filter.passes(0.5)
    assert ok is True
    ok, _ = price_band_filter.passes(ENTRY_YES_PRICE_MIN)
    assert ok is True
    ok, _ = price_band_filter.passes(ENTRY_YES_PRICE_MAX)
    assert ok is True


def test_blocks_unknown_price():
    ok, reason = price_band_filter.passes(None)
    assert ok is False
    assert reason == "unknown_price"
