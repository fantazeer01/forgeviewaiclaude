import datetime
import json
import logging
import os
from collections import defaultdict
from typing import Optional

from config.settings import TRADES_LOG, STATS_FILE, STATS_APY_NOTIONAL_CAPITAL_USD

logger = logging.getLogger(__name__)

# Floors days_running so apy_pct doesn't blow up toward infinity in the
# first minutes after the very first trade (a $5 win 10 minutes in would
# otherwise "annualize" to an absurd number).
MIN_DAYS_RUNNING = 1.0 / 24  # 1 hour


class StatsCalculator:
    """Computes portfolio-level statistics from data/paper_trades.jsonl and
    exports them to data/stats.json -- meant to be called on a throttle
    (see run.py, STATS_EXPORT_INTERVAL_SEC=60) since this re-reads and
    re-aggregates every resolved trade each time, cheap at current volume
    but no reason to redo it every 3s poll tick.

    days_running is measured from the earliest open_ts across ALL trades
    ever (open or resolved) to now -- the same "time since first trade
    ever" basis dashboard_pro.html's Pace panel uses, not the span between
    the first and last *resolved* trade, which would understate real
    elapsed time.

    2026-07-08: also exports an "alltime" breakdown block mirroring the
    top-level fields (all resolved trades ever, same numbers, just also
    grouped under this key for a client that wants both shapes).

    2026-07-08 (later same day): a "session" block scoped to
    data/state.json's session_start_ts briefly lived here too, but was
    removed -- dashboard_pro.html never read it (it has its own client-side
    scopeTradesToSession(), which was and remains the single source of
    truth for the Portfolio panel's session view), so this was a second,
    unused implementation of the same scoping logic. POST /api/reset-session
    in scripts/dashboard_server.py still updates state.json's
    session_start_ts -- the client-side code picks that up directly.
    """

    def __init__(self, trades_log: str = TRADES_LOG, stats_file: str = STATS_FILE):
        self.trades_log = trades_log
        self.stats_file = stats_file

    def _load_trades(self) -> list[dict]:
        if not os.path.exists(self.trades_log):
            return []
        by_id: dict[str, dict] = {}
        try:
            with open(self.trades_log) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    trade_id = entry.get("trade_id")
                    if trade_id:
                        by_id[trade_id] = entry
        except Exception as e:
            logger.error(f"StatsCalculator read error: {e}")
            return []
        return list(by_id.values())

    def compute(self) -> dict:
        trades = self._load_trades()
        days_running = self._days_running(trades)
        resolved = [
            t for t in trades
            if t.get("status") in ("win", "loss") and t.get("pnl_usd") is not None and t.get("close_ts")
        ]
        resolved.sort(key=lambda t: t["close_ts"])
        n = len(resolved)

        if n == 0:
            stats = self._stats(n=0, days_running=days_running)
        else:
            pnls = [t["pnl_usd"] for t in resolved]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            total_pnl = sum(pnls)
            mean_pnl = total_pnl / n
            std_pnl = self._stdev(pnls, mean_pnl)
            sharpe_ratio = (mean_pnl / std_pnl) * (n ** 0.5) if std_pnl > 0 else None

            avg_win = sum(wins) / len(wins) if wins else None
            avg_loss = sum(losses) / len(losses) if losses else None
            avg_rr = (avg_win / abs(avg_loss)) if avg_win is not None and avg_loss not in (None, 0) else None

            apy_pct = None
            if days_running is not None and days_running > 0:
                apy_pct = (total_pnl / STATS_APY_NOTIONAL_CAPITAL_USD) * (365.0 / days_running) * 100.0

            max_streak, current_streak = self._win_streaks(resolved)

            stats = self._stats(
                n=n, days_running=days_running, total_pnl=total_pnl,
                sharpe_ratio=sharpe_ratio, avg_rr=avg_rr, apy_pct=apy_pct,
                best_day_pnl=self._best_day_pnl(resolved),
                max_win_streak=max_streak, current_win_streak=current_streak,
            )

        stats["alltime"] = self._resolved_breakdown(resolved)
        return stats

    @staticmethod
    def _resolved_breakdown(resolved: list[dict]) -> dict:
        """n_trades/win_rate/avg_pnl_per_trade computed over RESOLVED trades
        only -- a caller that also wants open trades counted in n_trades
        (see "session" above) adds those on top afterward; win_rate and
        avg_pnl_per_trade stay resolved-only since an open trade has no
        realized outcome to average in yet."""
        n = len(resolved)
        if n == 0:
            return {"n_trades": 0, "total_pnl_usd": 0.0, "win_rate": None, "avg_pnl_per_trade": None}
        pnls = [t["pnl_usd"] for t in resolved]
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        return {
            "n_trades": n,
            "total_pnl_usd": round(total_pnl, 4),
            "win_rate": round(wins / n, 4),
            "avg_pnl_per_trade": round(total_pnl / n, 4),
        }

    @staticmethod
    def _stdev(values: list[float], mean: float) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return variance ** 0.5

    @staticmethod
    def _days_running(trades: list[dict]) -> Optional[float]:
        open_timestamps = []
        for t in trades:
            open_ts = t.get("open_ts")
            if not open_ts:
                continue
            try:
                open_timestamps.append(datetime.datetime.fromisoformat(open_ts))
            except ValueError:
                continue
        if not open_timestamps:
            return None
        earliest = min(open_timestamps)
        now = datetime.datetime.now(datetime.timezone.utc)
        elapsed_days = (now - earliest).total_seconds() / 86400.0
        return max(elapsed_days, MIN_DAYS_RUNNING)

    @staticmethod
    def _best_day_pnl(resolved: list[dict]) -> Optional[float]:
        by_date: dict[str, float] = defaultdict(float)
        for t in resolved:
            date = t["close_ts"][:10]  # YYYY-MM-DD
            by_date[date] += t["pnl_usd"]
        if not by_date:
            return None
        return max(by_date.values())

    @staticmethod
    def _win_streaks(resolved: list[dict]) -> tuple[int, int]:
        """resolved must already be sorted chronologically by close_ts.
        current_win_streak is the streak ending at the most recently
        resolved trade -- 0 if that trade was a loss."""
        max_streak = 0
        running = 0
        for t in resolved:
            if t["status"] == "win":
                running += 1
                max_streak = max(max_streak, running)
            else:
                running = 0
        return max_streak, running

    @staticmethod
    def _stats(n: int, days_running: Optional[float], total_pnl: float = 0.0,
               sharpe_ratio: Optional[float] = None, avg_rr: Optional[float] = None,
               apy_pct: Optional[float] = None, best_day_pnl: Optional[float] = None,
               max_win_streak: int = 0, current_win_streak: int = 0) -> dict:
        return {
            "n_trades": n,
            "total_pnl_usd": round(total_pnl, 4),
            "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            "avg_rr": round(avg_rr, 4) if avg_rr is not None else None,
            "apy_pct": round(apy_pct, 2) if apy_pct is not None else None,
            "days_running": round(days_running, 4) if days_running is not None else None,
            "best_day_pnl": round(best_day_pnl, 4) if best_day_pnl is not None else None,
            "max_win_streak": max_win_streak,
            "current_win_streak": current_win_streak,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def export(self) -> dict:
        stats = self.compute()
        try:
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            tmp_path = self.stats_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(stats, f, indent=2)
            os.replace(tmp_path, self.stats_file)
        except Exception as e:
            logger.error(f"StatsCalculator export error: {e}")
        return stats
