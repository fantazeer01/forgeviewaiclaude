import datetime
import json
import logging
import os

from config.settings import NO_SHADOW_LOG, NO_SHADOW_YES_PRICE_THRESHOLD

logger = logging.getLogger(__name__)


class NoShadowTracker:
    """Shadow-learns from what a NO-direction bet would have seen above
    NO_SHADOW_YES_PRICE_THRESHOLD, with zero money at risk -- real
    NO-direction trading was tried and disabled (see
    core/signal_combiner.py's docstring: 10.5% win rate over 19 trades, net
    -$139.67). No trade ever opens here; this only records a feature
    snapshot at "would-have-opened" time and, once the market resolves,
    feeds it into the same online model real trades train on.

    Mirrors PaperTradingEngine's open/close/restore-on-startup pattern:
    appends one row per event to NO_SHADOW_LOG (status="open" then a
    separate status="resolved" row later, same market_id) instead of
    keeping any state that a restart could lose.
    """

    def __init__(self, log_path: str = NO_SHADOW_LOG):
        self.log_path = log_path
        self._pending: dict[str, dict] = {}
        self._restore_pending()

    def _restore_pending(self):
        if not os.path.exists(self.log_path):
            return
        open_by_market: dict[str, dict] = {}
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    mid = entry.get("market_id")
                    if not mid:
                        continue
                    if entry.get("status") == "open":
                        open_by_market[mid] = entry
                    else:
                        open_by_market.pop(mid, None)
            for mid, entry in open_by_market.items():
                self._pending[mid] = entry
        except Exception as e:
            logger.error(f"NoShadowTracker restore error: {e}")

    def maybe_record(self, market: dict, features: dict, model_p_at_open):
        """Records a shadow "would-have-bet-NO" observation if yes_price is
        past the threshold and this market_id isn't already pending --
        no-op otherwise (including: still below threshold, or already
        recorded and awaiting resolution)."""
        market_id = market["market_id"]
        if market_id in self._pending:
            return
        yes_price = market["yes_price"]
        if yes_price <= NO_SHADOW_YES_PRICE_THRESHOLD:
            return
        entry = {
            "market_id": market_id,
            "asset": market["asset"],
            "yes_price": yes_price,
            "no_price": market["no_price"],
            "features": features,
            "model_p_at_open": model_p_at_open,
            "opened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "open",
        }
        self._pending[market_id] = entry
        self._append(entry)
        logger.info(f"NO shadow recorded (no real trade): {market['asset']} {market_id} yes_price={yes_price:.3f}")

    def resolve_pending(self, fetcher, online_model):
        """Checks every pending shadow observation for resolution; on
        resolution, feeds (features, 1 if outcome=='YES' else 0) into
        online_model.update() -- the same direction-independent "did YES
        win" label real trades train on -- and appends the resolution row."""
        for market_id, entry in list(self._pending.items()):
            resolution = fetcher.get_market_resolution(market_id)
            if not resolution:
                continue
            outcome = fetcher.resolve_outcome(resolution)
            if outcome is None:
                continue
            yes_won = 1 if outcome == "YES" else 0
            online_model.update(entry["features"], yes_won)
            resolved_entry = {
                **entry,
                "status": "resolved",
                "outcome": outcome,
                "no_won": outcome == "NO",
                "resolved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            self._append(resolved_entry)
            del self._pending[market_id]
            logger.info(
                f"NO shadow resolved: {entry['asset']} {market_id} outcome={outcome} "
                f"(would-have-been-NO {'WIN' if outcome == 'NO' else 'LOSS'})"
            )

    def _append(self, entry: dict):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
