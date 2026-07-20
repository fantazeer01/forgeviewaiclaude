"""Layer 3 (conscience): only trade a market whose yes_price sits in the
moderate-uncertainty band. A market already priced at 0.835 (or 0.165) has
almost no room to profit if right and a near-total loss if wrong -- entering
there is a bad risk/reward trade regardless of how confident the model is."""

from config.settings import ENTRY_YES_PRICE_MIN, ENTRY_YES_PRICE_MAX


def passes(yes_price) -> tuple:
    if yes_price is None:
        return False, "unknown_price"
    if yes_price < ENTRY_YES_PRICE_MIN or yes_price > ENTRY_YES_PRICE_MAX:
        return False, "price_out_of_band"
    return True, None
