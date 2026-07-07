import datetime
from typing import Optional

from config.settings import (
    NO_TRADING_ENABLED, NO_REVERSION_WINDOW_SEC, NO_REVERSION_PEAK_MIN_YES_PRICE,
    NO_REVERSION_MIN_YES_PRICE, NO_REVERSION_MIN_DROP,
)
from core.repricing_detector import RepricingSignal


class MeanReversionNoSignalGenerator:
    """Fires a NO signal when yes_price has fallen back from a recent
    extreme peak (>= NO_REVERSION_PEAK_MIN_YES_PRICE, 0.90) toward the
    NO_REVERSION_MIN_YES_PRICE (0.80) floor within a rolling
    NO_REVERSION_WINDOW_SEC window -- betting that the market's own
    near-certainty in YES was overdone and is mean-reverting back down.

    This is the "extreme mean-reversion" strategy removed 2026-07-06 (10.5%
    win rate over 19 trades, net -$139.67 -- see config.settings.NO_TRADING_ENABLED
    for the full history), resurrected 2026-07-07 once the online model's
    divergence and kelly_size()'s entry-price-blind sizing were both fixed,
    then DISABLED AGAIN the same day once real results in the corrected form
    came in: 2/25 = 8.00% win rate -- confirming a genuine negative edge,
    not a symptom of the old bugs. generate() below checks NO_TRADING_ENABLED
    as its very first line, before any other condition, so this class can
    never produce a signal while the flag is off regardless of caller --
    defense-in-depth on top of core/signal_combiner.py's own check at the
    call site.

    Structurally the same 3-point shape as MomentumSignalGenerator (track a
    rolling window, look for a real reversal, require a minimum drop so
    noise doesn't fire it) but tuned for the opposite, much more extreme
    price regime and direction.
    """

    def __init__(self):
        self._history: dict[str, list[dict]] = {}

    def update(self, market_id: str, yes_price: float):
        ts = datetime.datetime.now(datetime.timezone.utc)
        history = self._history.setdefault(market_id, [])
        history.append({"ts": ts, "yes": yes_price})
        cutoff = ts - datetime.timedelta(seconds=NO_REVERSION_WINDOW_SEC)
        self._history[market_id] = [p for p in history if p["ts"] > cutoff]

    def generate(self, market: dict) -> Optional[RepricingSignal]:
        if not NO_TRADING_ENABLED:
            return None
        yes_price = market["yes_price"]
        if yes_price < NO_REVERSION_MIN_YES_PRICE:
            return None
        history = self._history.get(market["market_id"], [])
        if not history:
            return None
        peak = max(p["yes"] for p in history)
        if peak < NO_REVERSION_PEAK_MIN_YES_PRICE:
            return None
        drop = peak - yes_price
        if drop < NO_REVERSION_MIN_DROP:
            return None
        span = max(peak - NO_REVERSION_MIN_YES_PRICE, 1e-6)
        reversal_strength = min(1.0, drop / span)
        confidence = min(0.95, 0.5 + reversal_strength * 0.4)
        no_price = 1.0 - yes_price
        return RepricingSignal(
            asset=market["asset"], market_id=market["market_id"], direction="NO",
            yes_price=yes_price, no_price=no_price,
            confidence=round(confidence, 3),
            reason=f"mean reversion: peak {peak:.3f} dropped {drop:.3f} to {yes_price:.3f}",
            minutes_remaining=market.get("minutes_remaining", 5.0),
        )
