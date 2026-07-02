# ForgeViewAI Paper Trading Report

## Current run: 8-hour autonomous monitoring

As of this run, `run.py` has an in-process auto-reset fix: whenever the daily loss
limit or loss streak limit trips, the bot resets those counters itself and keeps
running (see commit `0bbf681`) — it no longer requires external intervention to
resume. Hourly checkpoints below verify the bot process is alive and, as a safety
net, still auto-reset/restart it externally if it's ever found stopped or dead.
Config: BTC/ETH only, YES-direction signals only, entry price range 0.30–0.60,
MAX_OPEN_POSITIONS=5.

**Run started:** 2026-07-02T21:53:25Z
**Target end:** 2026-07-03T05:53:25Z (8 hours)
**Cohort start (0.30–0.60 filter went live):** 2026-07-02T13:26:54Z

## Latest snapshot — 2026-07-02T21:53:25Z (run start)

**BTC/ETH 0.30–0.60 cohort (since 2026-07-02T13:26:54Z):**
- Closed trades: 18
- Win rate: 27.8%
- Total PnL: -$55.9960

**All-time bot totals (includes pre-filter and SOL history):**
- Total trades: 47
- Wins/Losses: 19 / 28
- Win rate: 40.4%
- Total PnL: +$104.7430

## Auto-reset / intervention log

| Time (UTC) | Reason | Action |
|---|---|---|
| 2026-07-02T21:28:34Z | Loss streak limit hit | Reset daily loss + loss streak, restarted bot (pre-fix, manual) |
| 2026-07-02T21:34:32Z | (manual restart) | Restarted to deploy in-process auto-reset fix (commit 0bbf681) |
| 2026-07-02T21:41:03Z | (manual restart) | Restarted to deploy MAX_OPEN_POSITIONS=5 (commit 05066f9) |

## Hourly checkpoints

_(populated as the 8-hour run progresses)_

---

## Previous run: 3-hour monitoring (superseded)

Started 2026-07-02T21:29:08Z targeting 3h / 30 cohort trades. Superseded by the
8-hour run above before completion; last recorded snapshot at start was 16 closed
cohort trades, 31.2% win rate, -$35.9960 PnL.
