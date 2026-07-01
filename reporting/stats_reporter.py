from core.pnl_tracker import PnLTracker
from core.state_manager import StateManager

class StatsReporter:
    def __init__(self, tracker: PnLTracker, state: StateManager):
        self.tracker = tracker
        self.state = state

    def generate_report(self) -> str:
        s = self.tracker.compute_stats()
        lines = [
            "===================================",
            "  ForgeViewAI - Paper Trading Report",
            "===================================",
            f"  Total trades:   {s['total_trades']}",
            f"  Wins/Losses:    {s['wins']} / {s['losses']}",
            f"  Win rate:       {s['win_rate']*100:.1f}%",
            f"  Expectancy:     ${s['expectancy_usd']:+.4f}/trade",
            f"  Total PnL:      ${s['total_pnl_usd']:+.4f}",
            f"  Max Drawdown:   ${s['max_drawdown_usd']:.4f}",
            "", "  By Asset:",
        ]
        for asset, d in s.get("by_asset", {}).items():
            wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
            lines.append(f"    {asset}: {d['trades']} trades  {wr:.0f}% WR  PnL ${d['pnl']:+.3f}")
        if self.state.get("system_stopped"):
            lines += ["", f"  STOPPED: {self.state.get('stop_reason')}"]
        lines.append("===================================")
        return "\n".join(lines)
