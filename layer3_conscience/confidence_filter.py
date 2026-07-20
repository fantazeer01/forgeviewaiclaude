"""Layer 3 (conscience): turns a raw P(UP) into YES/NO/HOLD. Wider hold
zone than v3 (0.43-0.57 vs 0.45-0.55) -- fewer, higher-conviction trades.
In the HIGH_VOLATILITY regime the bar is raised further, to 0.60/0.40."""

from config.settings import CONFIDENCE_YES_THRESHOLD, CONFIDENCE_NO_THRESHOLD, HIGH_VOLATILITY_CONFIDENCE_THRESHOLD


def decide_side(p_up, regime: str = None) -> dict:
    if p_up is None:
        return {"decision": "HOLD", "reason": "no_prediction"}

    yes_threshold = CONFIDENCE_YES_THRESHOLD
    no_threshold = CONFIDENCE_NO_THRESHOLD
    if regime == "HIGH_VOLATILITY":
        yes_threshold = HIGH_VOLATILITY_CONFIDENCE_THRESHOLD
        no_threshold = 1 - HIGH_VOLATILITY_CONFIDENCE_THRESHOLD

    if p_up > yes_threshold:
        return {"decision": "YES", "reason": None}
    if p_up < no_threshold:
        return {"decision": "NO", "reason": None}
    return {"decision": "HOLD", "reason": "low_confidence"}
