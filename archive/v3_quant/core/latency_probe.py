"""Measurement-only module: how long does Polymarket's BTC-5m yes_price take
to react after a real move in Binance BTC spot price? Feeds no signal back
into the trading loop -- it only observes and logs to data/latency_log.jsonl.

yes_price is itself a 0-1 probability, so its "2%" confirmation threshold is
treated as 2 percentage points (e.g. 0.52 -> 0.54), not a 2% relative change --
that matches how prediction-market price moves are normally talked about."""

import datetime
import json
import logging
import os
import statistics

from config.settings import (
    LATENCY_LOG_FILE, LATENCY_MOVEMENT_THRESHOLD_BPS, LATENCY_CONFIRM_THRESHOLD_PCT,
    LATENCY_PENDING_EXPIRY_SEC, LATENCY_STATS_INTERVAL_SEC,
)

logger = logging.getLogger(__name__)


class LatencyProbe:
    def __init__(self, log_path: str = LATENCY_LOG_FILE, market: str = "btc-5m",
                 movement_threshold_bps: float = LATENCY_MOVEMENT_THRESHOLD_BPS,
                 confirm_threshold_pct: float = LATENCY_CONFIRM_THRESHOLD_PCT,
                 pending_expiry_sec: float = LATENCY_PENDING_EXPIRY_SEC,
                 stats_interval_sec: float = LATENCY_STATS_INTERVAL_SEC):
        self.log_path = log_path
        self.market = market
        self.movement_threshold_bps = movement_threshold_bps
        self.confirm_threshold_pct = confirm_threshold_pct
        self.pending_expiry_sec = pending_expiry_sec
        self.stats_interval_sec = stats_interval_sec

        self._last_spot_price = None
        self._pending_moves = []  # list of dicts awaiting a Polymarket reaction
        self.movements_detected = 0
        self.lags = []  # seconds, one per confirmed measurement
        self._last_stats_at = None

    def update(self, spot_price, yes_price, now: datetime.datetime = None):
        """Call once per tick with the latest BTC spot price and BTC-5m
        yes_price. Pure observation -- return value and internal state are
        never read by the trading loop."""
        now = now or datetime.datetime.now(datetime.timezone.utc)
        if self._last_stats_at is None:
            self._last_stats_at = now

        self._detect_movement(spot_price, yes_price, now)
        self._expire_stale_pending(now)
        self._check_confirmations(yes_price, now)

        if spot_price is not None:
            self._last_spot_price = spot_price

        self._maybe_log_stats(now)

    def _detect_movement(self, spot_price, yes_price, now):
        if spot_price is None or self._last_spot_price is None or self._last_spot_price == 0:
            return
        change_bps = (spot_price - self._last_spot_price) / self._last_spot_price * 10000
        if abs(change_bps) <= self.movement_threshold_bps:
            return
        self.movements_detected += 1
        self._pending_moves.append({
            "ts_binance": now,
            "direction": "UP" if change_bps > 0 else "DOWN",
            "magnitude_bps": abs(change_bps),
            "yes_price_at_binance_move": yes_price,
        })

    def _check_confirmations(self, yes_price, now):
        if yes_price is None or not self._pending_moves:
            return
        still_pending = []
        for move in self._pending_moves:
            base = move["yes_price_at_binance_move"]
            if base is None:
                still_pending.append(move)
                continue
            change_pct_points = (yes_price - base) * 100
            confirmed = (
                (move["direction"] == "UP" and change_pct_points > self.confirm_threshold_pct)
                or (move["direction"] == "DOWN" and change_pct_points < -self.confirm_threshold_pct)
            )
            if confirmed:
                self._record_measurement(move, yes_price, now)
            else:
                still_pending.append(move)
        self._pending_moves = still_pending

    def _expire_stale_pending(self, now):
        self._pending_moves = [
            m for m in self._pending_moves
            if (now - m["ts_binance"]).total_seconds() <= self.pending_expiry_sec
        ]

    def _record_measurement(self, move, yes_price_after, ts_polymarket):
        lag_seconds = round((ts_polymarket - move["ts_binance"]).total_seconds(), 3)
        self.lags.append(lag_seconds)
        record = {
            "ts_binance": _iso(move["ts_binance"]),
            "ts_polymarket": _iso(ts_polymarket),
            "lag_seconds": lag_seconds,
            "direction": move["direction"],
            "magnitude_bps": round(move["magnitude_bps"], 3),
            "yes_price_before": move["yes_price_at_binance_move"],
            "yes_price_after": yes_price_after,
            "market": self.market,
        }
        self._write_log(record)

    def _write_log(self, record: dict):
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"LatencyProbe log write error: {e}")

    def _maybe_log_stats(self, now):
        if (now - self._last_stats_at).total_seconds() < self.stats_interval_sec:
            return
        self._last_stats_at = now
        self.log_stats()

    def log_stats(self):
        n = len(self.lags)
        pct = (n / self.movements_detected * 100) if self.movements_detected else 0.0
        if n == 0:
            logger.info(
                f"LATENCY PROBE: N=0 measurements\n"
                f"Movements detected: {self.movements_detected} | Polymarket updated: 0 (0.0%)"
            )
            return
        logger.info(
            f"LATENCY PROBE: N={n} measurements\n"
            f"Median lag: {statistics.median(self.lags):.1f}s | Mean: {statistics.mean(self.lags):.1f}s | "
            f"Min: {min(self.lags):.1f}s | Max: {max(self.lags):.1f}s\n"
            f"Movements detected: {self.movements_detected} | Polymarket updated: {n} ({pct:.1f}%)"
        )


def _iso(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
