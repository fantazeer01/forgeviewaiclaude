# ForgeViewAI Paper Trading Bot — Decisions Log

Durable architecture and policy decisions for this project (`D:\forgeviewaiclaude`,
`github.com/fantazeer01/forgeviewaiclaude`). Not to be confused with the
separate, much larger `docs/polymarket/DECISIONS.md` in the read-only
research repo `D:\ForgeViewAI` — that log belongs to a different project.

## D-001 (2026-07-04): REJECTED — repricing-feature quant model does not clear the AUC bar; stays in shadow mode

**Question:** Trained on a repricing-motivated feature set (`yes_price`,
`price_drop_60s`, `price_drop_magnitude`, `time_remaining_pct`,
`order_book_imbalance`, `volume_24h`, `asset` one-hot), does any of
LogisticRegression / RandomForest / GradientBoosting predict Polymarket
5-minute BTC/ETH/SOL Up/Down outcomes well enough (AUC > 0.55 on genuinely
unseen data) to replace the repricing detector as the live trading signal?

**Data:** 1827 rows combined — 1490 historical rows from `D:\ForgeViewAI`
(read-only; not modified) copied into `data/historical/` in a prior sprint,
plus 337 resolved rows from this project's own live shadow log
(`data/quant_features.jsonl`). 47.8% win rate. Date range
2026-06-18T23:20:52Z → 2026-07-03T21:07:10Z. Full inventory and feature
derivation in `scripts/train_quant_model_v3.py`.

**Method:** Two evaluations were run, and only the second was used to
decide:

1. Naive mixed 5-fold cross-validation (historical + live shuffled
   together). All three models looked strong here — GradientBoosting hit
   68.1% accuracy, AUC 0.734, comfortably clearing an AUC-0.55 or
   accuracy-0.55 bar.
2. The deciding test: train on historical data only, evaluate on live data
   the model never saw — the actual situation the deployed bot is in.
   Historical (Jun 18-24, 49.7% win rate) and live (Jul 2-3, 40.1% win
   rate) come from different capture pipelines and different market
   regimes; mixing them into CV folds lets a model exploit which source a
   row came from rather than learn anything transferable. This is the same
   source-leakage pattern documented for a different feature set in
   `data/historical/README.md`'s 2026-07-04 sprint entry, and it reproduced
   here with an independent feature set.

**Result (deciding test, out-of-regime holdout):**

| Model | Accuracy | AUC |
|---|---|---|
| LogisticRegression (best) | 52.8% | 0.5209 |
| RandomForest | 52.2% | 0.5094 |
| GradientBoosting | 44.5% | 0.4381 (worse than a coin flip) |
| yes_price baseline (same holdout) | 63.2% | 0.5879 |

**Verdict: REJECTED.** Best out-of-regime AUC (0.5209) does not exceed the
0.55 threshold, and even that near-chance result trails the plain
`yes_price` baseline (0.5879) on the same holdout. The model was not saved
as production and `quant_signal.py` was not modified — the repricing
detector remains the sole live trading signal, and the bot was not
restarted. Kelly criterion sizing was not implemented for this model since
it did not qualify to go live.

**Top win-correlated features, live-only (n=337, the distribution the
deployed bot actually sees):** `yes_price` (r=+0.175), `time_remaining_pct`
(r=+0.101); `price_drop_60s`, `price_drop_magnitude`, `order_book_imbalance`,
and `volume_24h` are all near-zero (|r|<0.04). This is consistent with every
prior quant-model attempt in this project and with forgeview-ai's own
research (`MASTER_OBJECTIVE.md` §14, `NO_EDGE_FOUND_YET` /
`FEATURE_SET_INCOMPLETE`): no feature set tried so far — original
microstructure features, this repricing-motivated set, or the source
research repo's own baseline/microstructure candidates — beats trusting the
market's own YES price on genuinely unseen data.

**What would change this:** a much larger same-regime dataset (weeks of
live-only capture; mixing regimes with different base rates actively
misleads rather than merely adding noise), and feature families not yet
tried in this project (cross-asset correlation, wallet/flow-based signals —
see forgeview-ai's `wallet_intelligence_v1` research).

## D-002 (2026-07-04): Deep factor analysis of all 468 resolved trades — only entry price predicts WIN; everything else is noise or untestable

**Question:** Across every trade in `data/paper_trades.jsonl`, what actually
predicts a win — hour of day, entry price, asset, Fear & Greed, or day of
week? Rank the top 3 most predictive factors.

**Data:** 472 unique trades (deduped by `trade_id`), 468 resolved
(open_ts range 2026-07-02T11:21Z → 2026-07-04T15:16Z, ~2.5 days). Overall
win rate 189/468 = 40.4%. Fear & Greed values are real historical daily
data pulled live from alternative.me (`limit=30`), joined to each trade's
`open_ts` date — not the single current-value snapshot
`data/fear_greed.json` normally serves, since that has no history and
can't answer a per-trade question. All significance checks below are
point-biserial correlation with a t-test, a chi-square goodness-of-fit
against the pooled rate, or a two-proportion z-test, as noted.

**1. Entry price bucket — the one real, statistically significant factor:**

| Bucket | n | Win rate | 95% CI |
|---|---|---|---|
| 0.30–0.35 | 126 | 37.3% | [29.4%, 46.0%] |
| 0.35–0.40 | 118 | 35.6% | [27.5%, 44.6%] |
| 0.40–0.45 | 132 | 43.2% | [35.0%, 51.7%] |
| 0.45–0.50 | 55 | 45.5% | [33.0%, 58.5%] |
| 0.50–0.55 | 18 | 61.1% | [38.6%, 79.7%] |
| 0.55–0.60 | 6 | 66.7% | [30.0%, 90.3%] |

Clean monotonic trend. Point-biserial correlation of `entry_price` vs. win
across all 468 trades: **r = +0.151, t = 3.305 (p < 0.001)** — real and not
explained by the small top-bucket samples alone: restricted to only the
currently-traded 0.30–0.60 band (n=455), it's still significant
(r = +0.122, t = 2.61, p < 0.01).

**This is not a newly discovered edge.** It's the same fact this project
has hit repeatedly (`D-001` above, `data/historical/README.md`, every
batch-model sprint): Polymarket's own YES price is already a reasonably
calibrated probability estimate, so of course a higher entry price
correlates with a higher realized win rate — that's what calibration
means, not an inefficiency to exploit. The actionable implication is
narrower but real: **within the repricing detector's qualifying signals,
the higher-priced end of the current 0.30–0.60 band realizes a
meaningfully better win rate than the lower end.** Worth a follow-up test
of raising `min_yes_price` (e.g. toward 0.40–0.45) once more data
accumulates in the thin top buckets (n=18 and n=6 are too small to commit
to a specific new cutoff yet).

**2 & 3. No other factor tested is confidently distinguishable from noise:**

- **Hour of day (UTC):** win rate ranges 20.0%–58.8% across 24 hours (n=11–43
  each), but a chi-square goodness-of-fit against the pooled 40.4% rate
  gives **χ² = 14.39 on 23 df** — nowhere near the p=0.05 critical value of
  ~35.2. The apparent "good hours" (02:00, 20:00, 23:00) and "bad hours"
  (03:00, 09:00) are consistent with random scatter at this sample size,
  not a real intraday pattern.
- **Asset (BTC vs. ETH):** 40.6% (n=229) vs. 40.9% (n=230) — two-proportion
  z-test **z = −0.056**, indistinguishable from zero difference. SOL shows
  22.2% but n=9 and SOL isn't currently traded (`REPRICING_FROZEN.assets`
  excludes it) — too sparse and stale to act on.
- **Day of week:** only Thursday, Friday, and Saturday appear at all,
  because the entire trade history spans one Thursday-to-Saturday window
  (2.5 days). This isn't a day-of-week effect — it's every trade the bot has
  ever made, sliced by weekday label. Cannot be evaluated until trading
  spans multiple full weeks.
- **Fear & Greed:** **untestable, not merely "no effect."** Every single day
  in the trade history (Jul 2 = 19, Jul 3 = 21, Jul 4 = 22) was "Extreme
  Fear." Zero trades occurred with F&G > 50, so the requested `<30` vs
  `>50` comparison has an empty second group — there is no real-world
  variation to compare within this sample, regardless of how much trade
  data accumulates, until market sentiment actually shifts out of Extreme
  Fear.

**Verdict:** of the 5 factors requested, only entry price shows a real,
statistically significant relationship with outcome — and it's a
restatement of known market calibration, not a new edge. Hour-of-day and
asset show no signal. Day-of-week and Fear & Greed cannot be evaluated yet
for lack of real-world variation in the observation window (more calendar
days, and an actual shift in market sentiment, respectively) — not for lack
of trade volume.
