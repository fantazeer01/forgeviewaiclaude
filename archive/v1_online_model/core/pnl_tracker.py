import json
import os
import logging
from config.settings import TRADES_LOG

logger = logging.getLogger(__name__)

class PnLTracker:
    def compute_stats(self) -> dict:
        trades = self._load_closed_trades()
        if not trades:
            return self._empty_stats()
        pnls = [t["pnl_usd"] for t in trades if t.get("pnl_usd") is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = len(pnls)
        win_rate = len(wins) / total if total else 0.0
        expectancy = sum(pnls) / total if total else 0.0
        cumulative, running, peak, max_dd = [], 0.0, 0.0, 0.0
        for p in pnls:
            running += p
            cumulative.append(running)
        for v in cumulative:
            if v > peak:
                peak = v
            if peak - v > max_dd:
                max_dd = peak - v
        by_asset = {}
        for t in trades:
            a = t.get("asset", "?")
            if a not in by_asset:
                by_asset[a] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_asset[a]["trades"] += 1
            if (t.get("pnl_usd") or 0) > 0:
                by_asset[a]["wins"] += 1
            by_asset[a]["pnl"] += t.get("pnl_usd") or 0
        return {
            "total_trades": total, "wins": len(wins), "losses": len(losses),
            "win_rate": round(win_rate, 4), "expectancy_usd": round(expectancy, 4),
            "total_pnl_usd": round(sum(pnls), 4), "max_drawdown_usd": round(max_dd, 4),
            "by_asset": by_asset,
        }

    def _load_closed_trades(self) -> list[dict]:
        if not os.path.exists(TRADES_LOG):
            return []
        by_id: dict[str, dict] = {}
        try:
            with open(TRADES_LOG) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    if t.get("trade_id"):
                        by_id[t["trade_id"]] = t
        except Exception as e:
            logger.error(f"load error: {e}")
            return []
        return [t for t in by_id.values() if t.get("status") != "open"]

    def _empty_stats(self) -> dict:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "expectancy_usd": 0.0, "total_pnl_usd": 0.0, "max_drawdown_usd": 0.0, "by_asset": {}}
