import logging
from config.settings import KELLY_FRACTION_CAP

logger = logging.getLogger(__name__)


def net_odds_from_price(entry_price: float) -> float:
    """Net odds b for a binary token bought at entry_price and settling to $0/$1:
    profit per $1 staked on a win is (1 - entry_price) / entry_price."""
    if entry_price <= 0:
        return 0.0
    return (1.0 - entry_price) / entry_price


def kelly_fraction(win_probability: float, net_odds: float) -> float:
    """Full Kelly fraction f = (p*b - (1-p)) / b, clamped to [0, 1].
    A non-positive result means the edge doesn't justify a position."""
    if net_odds <= 0:
        return 0.0
    p = win_probability
    f = (p * net_odds - (1 - p)) / net_odds
    return max(0.0, min(1.0, f))


def quarter_kelly_fraction(win_probability: float, net_odds: float,
                            multiplier: float = KELLY_FRACTION_CAP) -> float:
    """Fractional Kelly (quarter-Kelly by default) for risk reduction, additionally
    hard-capped at KELLY_FRACTION_CAP regardless of multiplier so a single signal
    can never be sized above that share of bankroll."""
    f = kelly_fraction(win_probability, net_odds) * multiplier
    return min(f, KELLY_FRACTION_CAP)


def kelly_position_size(win_probability: float, net_odds: float, bankroll: float,
                         multiplier: float = KELLY_FRACTION_CAP) -> float:
    """Position size in dollars using quarter-Kelly (capped) sizing."""
    if bankroll <= 0:
        return 0.0
    fraction = quarter_kelly_fraction(win_probability, net_odds, multiplier)
    return fraction * bankroll
