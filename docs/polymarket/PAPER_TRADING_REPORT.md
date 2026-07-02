# ForgeViewAI Paper Trading Report

## Current run: overnight autonomous monitoring

Config as of this run: BTC/ETH only, YES-direction signals only, entry price
range 0.30–0.60, MAX_OPEN_POSITIONS=5, MARKET_POLL_INTERVAL_SEC=3,
SIGNAL_COOLDOWN_SEC=120, MAX_DAILY_LOSS_USD=10000, MAX_LOSS_STREAK=50 (both
raised high enough that the bot's own circuit breakers are effectively
disabled for this run — see intervention log below). `core/quant_signal.py`
runs the repricing detector as the sole trading decision; a trained
QuantModel logs shadow predictions to `data/quant_features.jsonl` alongside
each signal but does not influence trades.

**Run started:** 2026-07-02T22:53:04Z
**Target end:** 2026-07-03T05:00:00Z (8:00 AM Moscow time, UTC+3)
**Cadence:** bot-health check hourly; full stats snapshot + push every 2 hours

## Latest snapshot — 2026-07-02T22:53:04Z (run start)

**All-time bot totals:**
- Total trades: 60
- Wins/Losses: 25 / 35
- Win rate: 41.7%
- Total PnL: +$129.4684
- Max drawdown: $110.00

**By asset:**
- ETH: 30 trades, 13 wins, PnL +$104.7362
- SOL: 9 trades, 2 wins, PnL -$36.6199
- BTC: 21 trades, 10 wins, PnL +$61.3521

## Auto-reset / intervention log

| Time (UTC) | Reason | Action |
|---|---|---|
| 2026-07-02T22:53:04Z | (run start) | Overnight monitoring loop started; no intervention yet |
| 2026-07-03T02:38:57Z | Process killed externally (not a crash — clean log up to the last line, `system_stopped: false`; likely session/environment reclaiming the background task) | Restarted immediately (new task id `bictkhtbv`, was `b5gamxp7k`). Note for the next scheduled checkpoint: verify liveness via data/state.json recency, not just the originally-scheduled task id, to avoid spawning a duplicate bot process. |

## Hourly checkpoints

_(populated as the overnight run progresses)_

---

## Previous run: 8-hour monitoring (superseded)

Started 2026-07-02T21:53:25Z. Superseded mid-run by several config/architecture
changes (quant model sprint, revert to repricing-primary + shadow model, poll
interval 15s→3s, signal cooldown 300s→120s, risk limits raised). Last recorded
snapshot before supersession: 47 total trades, 40.4% win rate, +$104.7430 PnL.
