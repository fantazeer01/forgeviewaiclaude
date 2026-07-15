"""Combines momentum, volume, and macro signals into a single trade
decision per 5-min market window."""

import logging

from config.settings import (
    ENSEMBLE_WEIGHTS, ENSEMBLE_YES_SCORE_THRESHOLD, ENSEMBLE_YES_PRICE_BAND,
    ENSEMBLE_NO_SCORE_THRESHOLD, ENSEMBLE_NO_PRICE_BAND,
    ENSEMBLE_MIN_TRAINING_EXAMPLES,
)
from models.macro_model import macro_bias

logger = logging.getLogger(__name__)


class Ensemble:
    def __init__(self, momentum_model, volume_model, asset: str = None, stats_tracker=None):
        self.momentum_model = momentum_model
        self.volume_model = volume_model
        self.asset = asset
        self.stats_tracker = stats_tracker

    def _training_examples(self) -> int:
        return min(self.momentum_model.n_examples, self.volume_model.n_examples)

    def score(self, features: dict, fear_greed: int, hour_utc: int) -> dict:
        momentum_p = self.momentum_model.predict_up(features)
        volume_p = self.volume_model.predict_up(features)
        macro_p = 0.5 + macro_bias(fear_greed, hour_utc)
        final_score = (
            ENSEMBLE_WEIGHTS["momentum"] * momentum_p
            + ENSEMBLE_WEIGHTS["volume"] * volume_p
            + ENSEMBLE_WEIGHTS["macro"] * macro_p
        )
        return {
            "momentum_p": momentum_p,
            "volume_p": volume_p,
            "macro_p": macro_p,
            "final_score": final_score,
        }

    def decide(self, features: dict, fear_greed: int, hour_utc: int) -> dict:
        """Fair-value entry strategy (2026-07-13) replaces model-driven
        trading entirely -- see _fair_value_decide(). _live_decide() and
        _warmup_decide() are kept below, dormant/uncalled, in case
        model-driven trading is re-enabled later; the momentum/volume
        models themselves keep learning regardless (Executor.close_position()
        and Bot's shadow learning both still call .learn() on them), they're
        just not consulted for this decision."""
        return self._fair_value_decide(features, self.asset, hour_utc)

    def _fair_value_decide(self, features: dict, asset: str, hour_utc: int) -> dict:
        """Doesn't predict direction at all: in the first 60s of a window
        (seconds_remaining >= 240), buys whichever side (YES or NO) is
        priced below its 0.50 fair value, betting on the ~50/50 UP/DOWN
        base rate rather than any model signal. No trade in the 0.47-0.53
        dead zone (edge too thin after slippage).

        Bands narrowed 2026-07-15: a 204-trade breakdown found the outer
        edges of the original 0.43-0.47/0.53-0.57 band (0.43-0.45 and
        0.55-0.57) ran 38.6%/41.7% win rate -- a real negative edge vs.
        breakeven (n=80, n=36), not noise -- while the inner 0.45-0.47/
        0.53-0.55 sub-bands ran 51.6% (n=124, n=62), +5.7pp over breakeven.
        The outer edges are dropped entirely rather than kept as "still
        cheap enough": beyond 0.45/0.55 a real move is likely already
        underway, not just noise around fair value."""
        yes_price = features.get("yes_price", 0.5)
        seconds_remaining = features.get("seconds_remaining", 0)

        base = {
            "mode": "fair_value",
            "momentum_p": None,
            "volume_p": None,
            "macro_p": None,
            "final_score": None,
        }

        if seconds_remaining < 240:
            return {**base, "decision": None, "reason": "too_late_for_fv"}

        if 0.45 <= yes_price < 0.47:
            candidate = "YES"
        elif 0.53 < yes_price <= 0.55:
            candidate = "NO"
        else:
            return {**base, "decision": None, "reason": "price_near_fair_value"}

        # Stats filter (2026-07-15): a learned win-rate veto layered on top
        # of the price/timing checks above -- it can only block a trade this
        # logic already selected, never pick a direction of its own.
        if self.stats_tracker is not None and not self.stats_tracker.should_trade(yes_price, hour_utc):
            return {**base, "decision": None, "reason": "stats_filter_blocked"}

        reason = f"yes_price={yes_price:.3f} < 0.47" if candidate == "YES" else f"yes_price={yes_price:.3f} > 0.53"
        return {**base, "decision": candidate, "reason": reason}

    def _warmup_decide(self, features: dict, training_examples: int) -> dict:
        """Warmup trading is fully disabled (2026-07-12): no capital is risked
        below ENSEMBLE_MIN_TRAINING_EXAMPLES at all anymore. bot.py's shadow
        learning accumulates (features, outcome) pairs every window instead,
        with no position ever opened -- see Bot._maybe_capture_shadow() /
        Bot._check_shadow_resolutions()."""
        return {
            "mode": "warmup",
            "momentum_p": None,
            "volume_p": None,
            "macro_p": None,
            "final_score": None,
            "decision": None,
            "reason": f"warmup trading disabled: {training_examples}/{ENSEMBLE_MIN_TRAINING_EXAMPLES} examples, shadow learning only",
        }

    def _live_decide(self, features: dict, fear_greed: int, hour_utc: int) -> dict:
        result = self.score(features, fear_greed, hour_utc)
        result["mode"] = "live"
        final_score = result["final_score"]
        yes_price = features.get("yes_price")
        decision = None

        yes_lo, yes_hi = ENSEMBLE_YES_PRICE_BAND
        no_lo, no_hi = ENSEMBLE_NO_PRICE_BAND

        if final_score > ENSEMBLE_YES_SCORE_THRESHOLD and yes_price is not None and yes_lo <= yes_price <= yes_hi:
            decision = "YES"
        elif final_score < ENSEMBLE_NO_SCORE_THRESHOLD and yes_price is not None and no_lo <= yes_price <= no_hi:
            # 2026-07-12 trade-breakdown: BTC/NO and ETH/NO both ran 33.3%
            # win rate today (-$7.43, -$4.40) while SOL/NO ran 66.7%
            # (+$1.67) -- NO is only allowed for SOL until that changes.
            if self.asset is not None and self.asset.upper() == "SOL":
                decision = "NO"

        result["decision"] = decision
        result["reason"] = None if decision else "uncertainty zone or price out of band"
        return result
