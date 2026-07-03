# ForgeViewAI Paper Trading Report

## Run COMPLETE: overnight autonomous monitoring (2026-07-02T22:53:04Z → 2026-07-03T20:40:31Z)

Config during this run: BTC/ETH only, YES-direction signals only, entry price
range 0.30–0.60, MAX_OPEN_POSITIONS=5, MARKET_POLL_INTERVAL_SEC=3,
SIGNAL_COOLDOWN_SEC=120, MAX_DAILY_LOSS_USD=10000, MAX_LOSS_STREAK=50 (both
raised high enough that the bot's own circuit breakers were effectively
disabled for this run). `core/quant_signal.py` ran the repricing detector as
the sole trading decision; a trained QuantModel logged shadow predictions to
`data/quant_features.jsonl` alongside each signal but never influenced trades.

**Run started:** 2026-07-02T22:53:04Z
**Requested target:** 2026-07-03T05:00:00Z (8:00 AM Moscow time, UTC+3)
**Actual final report delivered:** 2026-07-03T20:40:31Z (23:40 Moscow) — **~15.5
hours later than requested.** The checkpoint chain landed correctly on schedule
through 05:01 UTC (see checkpoint history below), but the wake scheduled to
fire at that point was not processed until many hours later — most likely the
environment/session was suspended and only resumed at 20:40 UTC. This was not
a repeat of an earlier scheduling arithmetic error (that one was caught and
corrected before this delay). The bot process itself survived the gap and was
confirmed healthy at delivery time.

## FINAL snapshot — 2026-07-03T20:40:31Z

**All-time bot totals:**
- Total trades: 384
- Wins/Losses: 154 / 230
- Win rate: 40.1%
- Total PnL: +$170.2112
- Max drawdown: $186.8809

**By asset:**
- ETH: 185 trades, 75 wins, PnL +$156.5044
- SOL: 9 trades, 2 wins, PnL -$36.6199
- BTC: 190 trades, 77 wins, PnL +$50.3267

**Best trade (all-time):** ETH YES @ entry 0.105, PnL +$85.2381 (opened
2026-07-02T11:57:07Z, closed 2026-07-02T12:06:54Z — from before this
monitoring run started; the 0.30–0.60 entry-price band adopted afterward was
specifically meant to filter out this kind of low-entry outlier).

**Worst trade:** tied at exactly -$10.00 (flat stake loss) across 230 trades —
every loss is capped at the $10 stake since a losing YES token resolves to
$0. First such loss: SOL NO @ entry 0.225 (2026-07-02T11:41:48Z).

**Overnight-window-only** (trades opened after 2026-07-02T22:53:04Z, i.e.
actually attributable to this run): 321 closed, 128 wins, 39.9% win rate,
+$50.1242 PnL. Best overnight trade: ETH YES @ entry 0.305, +$22.7869
(2026-07-03T02:06:03Z). Worst: BTC YES @ entry 0.395, -$10.00
(2026-07-02T22:57:05Z, the very first trade after run start).

**Comparison to run-start baseline** (60 trades, 25W/35L, 41.7% WR, +$129.4684):
+324 trades, win rate essentially flat (41.7% → 40.1%, within normal noise
given the small overnight-only sample), total PnL +$40.74 net over the whole
run (+$170.21 − $129.47), though the overnight-only slice (+$50.12 on 321
trades) shows a thinner per-trade edge than the historical baseline —
consistent with win rate holding roughly flat while volume increased sharply
under the 3s poll / 120s cooldown change.

## Correction: the "state.json drift" finding above was a false alarm

The previous version of this report claimed `data/state.json`'s counters had
drifted out of sync with the authoritative trade log. That claim was wrong.
It came from comparing a state.json read taken *before* the ~15.5-hour
suspension gap against a `PnLTracker` recomputation taken *after* it, without
accounting for how many real trades happened in between — the two numbers
looked inconsistent only because they were snapshots from very different
points in time, not because anything had actually drifted. A direct,
simultaneous re-check this session (fresh `state.json` read +
fresh `PnLTracker.compute_stats()` call in the same script, same instant)
showed the two sources match **exactly**: 154W / 230L / 384 total /
+$170.2112. There is no counter-drift bug.

That said, one legitimate latent risk was found and fixed while
investigating: both `StateManager._save()` and `DedupGuard._save()` wrote
their JSON state files non-atomically (`open(path, "w")` truncates before
the new content is fully written), so a crash or kill mid-write — and this
run saw two external kills — could in principle corrupt the file. Both now
write to a `.tmp` file and `os.replace()` it into place atomically. Covered
by new regression tests in `tests/test_state_manager.py` and
`tests/test_dedup_guard.py` (full suite: 163 passed). This is a real
hardening fix, just not a fix for the drift that was originally reported.

## Shadow quant model: does it beat the naive baseline?

Analyzed `data/quant_features.jsonl` (767 rows: 434 `signal`-stage,
333 `resolution`-stage with a known `outcome`). 329 of the 333 resolved rows
have a logged `model_probability` from the shadow `QuantModel`.

**Overall resolved win rate this run:** 134/333 = 40.2% (matches the
384-trade all-time figure closely; consistent, not a new number).

| Metric (n=329, resolved + model ran) | Shadow model | Baseline (yes_price as P(win)) |
|---|---|---|
| Accuracy @ 0.5 threshold | 59.9% | 62.9% |
| Brier score (lower better) | 0.3244 | 0.2344 |
| Log loss (lower better) | 0.9716 | 0.6616 |
| AUC (win vs loss ranking) | 0.550 | 0.579 |

**Diagnosis: the model's 59.9% "accuracy" is an illusion.** Its predicted
`model_probability` never once crossed 0.5 across all 329 samples (min
0.053, max 0.494, mean 0.110) — it always predicts "loss." Since 197/329
(59.9%) of resolved trades actually were losses, always predicting "loss"
scores 59.9% accuracy for free, with zero real discrimination. Its AUC
(0.550, barely above the 0.500 coin-flip line) confirms this — and it's
*below* the baseline's AUC (0.579). The raw `yes_price` alone remains a
better-calibrated, better-discriminating signal than the trained model on
this data. This reproduces, on fresh live data, the same conclusion
`forgeview-ai`'s own research reached: no model tried so far beats trusting
the market's own YES price. **Answer to "does it predict better than 40% win
rate": no** — as a usable go/no-go signal it would fire on zero trades
(nothing ever exceeds the 0.5 threshold), and its ranking power is weaker
than the naive baseline it's meant to improve on. Shadow mode (logging only,
no live influence) remains the right call.

## Auto-reset / intervention log

| Time (UTC) | Reason | Action |
|---|---|---|
| 2026-07-02T22:53:04Z | (run start) | Overnight monitoring loop started; no intervention yet |
| 2026-07-03T02:38:57Z | Process killed externally (not a crash — clean log up to the last line, `system_stopped: false`; likely session/environment reclaiming the background task) | Restarted immediately (new task id `bictkhtbv`, was `b5gamxp7k`). |
| 2026-07-03T03:17:41Z | Process killed externally a second time (same pattern) | Restarted immediately (new task id `b30eiw0p8`, was `bictkhtbv`). Survived the rest of the run, including the long suspension gap below. |
| ~2026-07-03T05:01Z–20:40Z | Environment/session appears to have been suspended for ~15.5 hours between the last processed checkpoint and this final report | No action taken/needed — task `b30eiw0p8` survived the gap and was confirmed healthy on resume; bot kept trading throughout (321 overnight trades accumulated). |
| 2026-07-03T20:40:31Z | (report delivery) | Reported a suspected state.json counter drift; later disproven (see Correction section above) — was a snapshot-timing artifact, not a real bug. |
| 2026-07-03 (follow-up) | Re-verified state.json vs PnLTracker directly — confirmed exact match, no drift. Hardened `StateManager`/`DedupGuard` writes to be atomic regardless (real but separate risk, given the two external kills this run). Analyzed shadow quant model accuracy (see section above): no edge found, model never predicts >0.5, AUC below baseline. | Corrected report, ran full pytest (163 passed), pushed to GitHub. |

## Overall assessment

**Bot reliability:** good. Survived two external kills (both auto-recovered
within minutes, no duplicate processes spawned) and a ~15.5-hour environment
suspension without crashing or requiring manual intervention. **Data
integrity:** no real bug — the originally reported state.json drift was a
false alarm from comparing snapshots across the suspension gap; a direct
re-check found state.json and the JSONL-derived stats match exactly.
Non-atomic writes were hardened as a precaution regardless, given the two
external kills. **Quant model:** shadow-mode analysis of 329 resolved
predictions found no edge — the model never predicts a win probability
above 0.5 and its AUC (0.550) trails the naive yes_price baseline (0.579).
Repricing remains the sole live signal. **Performance:** win rate held roughly flat (~40-42%) across the whole run;
total PnL grew modestly (+$40.74 net), but the overnight-only slice shows a
thinner edge (+$50.12 over 321 trades, avg +$0.16/trade) than the historical
baseline — not a red flag on its own given normal variance, but not evidence
that the faster 3s-poll/120s-cooldown change (made right before this run)
meaningfully improved results either.

## Hourly checkpoints

| Checkpoint | Time (UTC) | Total trades | Win rate | Total PnL |
|---|---|---|---|---|
| 1 (light) | 2026-07-02T23:55:07Z | 73 | — (not recomputed, light check only) | — |
| 2 (full) | 2026-07-03T00:56:07Z | 92 | 42.4% | +$157.0128 |
| 3 (light) | 2026-07-03T01:57:06Z | 106 | — (not recomputed, light check only) | — |
| 4 (full) | 2026-07-03T02:58:05Z | 124 | 41.9% | +$158.7749 |
| 5 (light) | 2026-07-03T03:59:07Z | 140 | — (not recomputed, light check only) | — |
| FINAL | 2026-07-03T20:40:31Z | 384 | 40.1% | +$170.2112 |

---

## Previous run: 8-hour monitoring (superseded)

Started 2026-07-02T21:53:25Z. Superseded mid-run by several config/architecture
changes (quant model sprint, revert to repricing-primary + shadow model, poll
interval 15s→3s, signal cooldown 300s→120s, risk limits raised). Last recorded
snapshot before supersession: 47 total trades, 40.4% win rate, +$104.7430 PnL.
