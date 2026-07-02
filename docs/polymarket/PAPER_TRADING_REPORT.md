# ForgeViewAI Paper Trading Report

Autonomous monitoring run. Bot auto-resets (daily loss limit + loss streak limit)
whenever it stops, so it keeps trading continuously through this window rather than
halting on its own risk limits. Config: BTC/ETH only, YES-direction signals only,
entry price range 0.30–0.60.

**Run started:** 2026-07-02T21:29:08Z (restart after last auto-reset)
**Target:** minimum 3 hours runtime AND 30+ closed trades in the BTC/ETH 0.30–0.60 cohort
**Cohort start (filter went live):** 2026-07-02T13:26:54Z

## Latest snapshot — 2026-07-02T21:29:08Z (run start)

**BTC/ETH 0.30–0.60 cohort (since 2026-07-02T13:26:54Z):**
- Closed trades: 16 / 30 target
- Win rate: 31.2%
- Total PnL: -$35.9960

**All-time bot totals (includes pre-filter and SOL history):**
- Total trades: 45
- Wins/Losses: 19 / 26
- Win rate: 42.2%
- Total PnL: +$124.7430

## Auto-reset log

| Time (UTC) | Reason | Action |
|---|---|---|
| 2026-07-02T21:28:34Z | Loss streak limit hit | Reset daily loss + loss streak, restarted bot |

## Hourly checkpoints

_(populated as the run progresses)_
