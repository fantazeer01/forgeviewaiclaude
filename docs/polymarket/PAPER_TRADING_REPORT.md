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

## Latest snapshot — 2026-07-03T02:58:05Z (checkpoint 4, 2-hour mark)

**All-time bot totals:**
- Total trades: 124
- Wins/Losses: 52 / 72
- Win rate: 41.9%
- Total PnL: +$158.7749
- Max drawdown: $110.00

**By asset:**
- ETH: 61 trades, 26 wins, PnL +$139.9791
- SOL: 9 trades, 2 wins, PnL -$36.6199
- BTC: 54 trades, 24 wins, PnL +$55.4157

## Auto-reset / intervention log

| Time (UTC) | Reason | Action |
|---|---|---|
| 2026-07-02T22:53:04Z | (run start) | Overnight monitoring loop started; no intervention yet |
| 2026-07-03T02:38:57Z | Process killed externally (not a crash — clean log up to the last line, `system_stopped: false`; likely session/environment reclaiming the background task) | Restarted immediately (new task id `bictkhtbv`, was `b5gamxp7k`). Verify liveness via data/state.json recency at each checkpoint, not just the last-known task id, to avoid spawning a duplicate bot process. |
| 2026-07-03T03:17:41Z | Process killed externally a second time (same pattern — clean log up to the last line, `system_stopped: false`) | Restarted immediately (new task id `b30eiw0p8`, was `bictkhtbv`). Second occurrence this run; likely environment/session periodically reclaiming long-running background tasks rather than a bug in run.py. |

## Hourly checkpoints

| Checkpoint | Time (UTC) | Total trades | Win rate | Total PnL |
|---|---|---|---|---|
| 1 (light) | 2026-07-02T23:55:07Z | 73 | — (not recomputed, light check only) | — |
| 2 (full) | 2026-07-03T00:56:07Z | 92 | 42.4% | +$157.0128 |
| 3 (light) | 2026-07-03T01:57:06Z | 106 | — (not recomputed, light check only) | — |
| 4 (full) | 2026-07-03T02:58:05Z | 124 | 41.9% | +$158.7749 |

---

## Previous run: 8-hour monitoring (superseded)

Started 2026-07-02T21:53:25Z. Superseded mid-run by several config/architecture
changes (quant model sprint, revert to repricing-primary + shadow model, poll
interval 15s→3s, signal cooldown 300s→120s, risk limits raised). Last recorded
snapshot before supersession: 47 total trades, 40.4% win rate, +$104.7430 PnL.
