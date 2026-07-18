"""Learns a win-rate table keyed by (yes_price bucket, hour-of-day bucket)
from resolved trades, and gates new entries on it. This is a filter layered
ON TOP of _fair_value_decide()'s existing price/timing checks in
core/ensemble.py -- it never predicts direction itself, it only vetoes a
trade already selected by that logic if the same conditions have
historically lost money."""

import datetime
import json
import logging
import os

from config.settings import (
    PAPER_TRADES_LOG, STATS_TRACKER_FILE, STATS_MIN_SAMPLES, STATS_MIN_WIN_RATE,
    STATS_EARLY_BLOCK_MIN_SAMPLES, STATS_EARLY_BLOCK_MAX_WIN_RATE,
)

logger = logging.getLogger(__name__)

# fair_value became the only strategy that ever opens a real trade at this
# commit (2026-07-13T16:58:04 UTC) -- trades before it are from the retired
# warmup/live-ensemble strategies (different entry logic entirely) and would
# skew these buckets if included in the historical backfill.
FAIR_VALUE_DEPLOY_CUTOFF = "2026-07-13T16:58:04"


def bucket_key(yes_price: float, hour_utc: int) -> tuple:
    price_bucket = round(round(yes_price * 20) / 20, 2)
    hour_bucket = (hour_utc // 6) * 6
    return price_bucket, hour_bucket


class StatsTracker:
    def __init__(self, state_file: str = STATS_TRACKER_FILE, trades_log_path: str = PAPER_TRADES_LOG):
        self.state_file = state_file
        self.trades_log_path = trades_log_path
        self.buckets = self._load_or_backfill()

    def _load_or_backfill(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    buckets = json.load(f)
                logger.info(f"StatsTracker: loaded {len(buckets)} buckets from {self.state_file}")
                return buckets
            except Exception as e:
                logger.warning(f"StatsTracker state load error, rebuilding from trade history: {e}")
        buckets = self._backfill_from_history()
        self._save(buckets)
        return buckets

    def _backfill_from_history(self) -> dict:
        buckets = {}
        if not os.path.exists(self.trades_log_path):
            return buckets
        n_loaded = 0
        try:
            with open(self.trades_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        trade = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if trade.get("opened_at", "") < FAIR_VALUE_DEPLOY_CUTOFF:
                        continue
                    try:
                        yes_price = (
                            trade["entry_price"] if trade["side"] == "YES"
                            else round(1 - trade["entry_price"], 10)
                        )
                        hour_utc = datetime.datetime.fromisoformat(trade["closed_at"]).hour
                        self._apply(buckets, yes_price, hour_utc, trade["won"])
                        n_loaded += 1
                    except (KeyError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"StatsTracker backfill error: {e}")
        logger.info(f"StatsTracker: backfilled {n_loaded} historical fair_value trades into {len(buckets)} buckets")
        return buckets

    def _apply(self, buckets: dict, yes_price: float, hour_utc: int, won: bool):
        price_bucket, hour_bucket = bucket_key(yes_price, hour_utc)
        key = f"{price_bucket}_{hour_bucket}"
        entry = buckets.setdefault(
            key, {"price_bucket": price_bucket, "hour_bucket": hour_bucket, "wins": 0, "total": 0}
        )
        entry["total"] += 1
        if won:
            entry["wins"] += 1

    def record(self, yes_price: float, hour_utc: int, won: bool):
        self._apply(self.buckets, yes_price, hour_utc, won)
        self._save(self.buckets)

    def should_trade(self, yes_price: float, hour_utc: int) -> bool:
        price_bucket, hour_bucket = bucket_key(yes_price, hour_utc)
        key = f"{price_bucket}_{hour_bucket}"
        entry = self.buckets.get(key)
        if entry is None or entry["total"] < STATS_EARLY_BLOCK_MIN_SAMPLES:
            return True  # no data at all yet
        win_rate = entry["wins"] / entry["total"]
        if win_rate < STATS_EARLY_BLOCK_MAX_WIN_RATE:
            return False  # already a clear losing signal, don't wait for n=50
        if entry["total"] < STATS_MIN_SAMPLES:
            return True  # not yet a clear loser -- keep accumulating toward n=50
        return win_rate >= STATS_MIN_WIN_RATE

    def get_stats(self) -> dict:
        rows = []
        for entry in self.buckets.values():
            win_rate = entry["wins"] / entry["total"] if entry["total"] else None
            rows.append({
                "price_bucket": entry["price_bucket"],
                "hour_bucket": entry["hour_bucket"],
                "trades": entry["total"],
                "win_rate": win_rate,
                "trading": self.should_trade(entry["price_bucket"], entry["hour_bucket"]),
            })
        rows.sort(key=lambda r: (r["price_bucket"], r["hour_bucket"]))
        return {"buckets": rows}

    def _save(self, buckets: dict):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(buckets, f)
            os.replace(tmp, self.state_file)
        except Exception as e:
            logger.error(f"StatsTracker save error: {e}")
