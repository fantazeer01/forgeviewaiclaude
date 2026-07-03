# Historical research data (from forgeview-ai)

Copied read-only from a local checkout of `https://github.com/fantazeer01/forgeview-ai`
(`D:\ForgeViewAI`) on 2026-07-03. That repo is not modified, committed to, or
pushed to by this project.

## Files

| File | Rows | Label | Source |
|---|---|---|---|
| `microstructure_dataset_batch_001.csv` | 213 | `outcome` (final UP/DOWN resolution) | `polymarket/data/microstructure_dataset_batch_001/export/dataset.csv` |
| `microstructure_dataset_batch_002.csv` | 213 | `outcome` | `polymarket/data/microstructure_dataset_batch_002/export/dataset.csv` |
| `outcome_training_dataset.csv` | 1064 (public+mock) | `outcome` | `polymarket/data/training/dataset.csv` (Feature Engine v1 canonical dataset) |
| `repricing_labels_batch_001.csv` | 130 | `repriced_favorably` (early-exit target, NOT final resolution) | `polymarket/data/repricing_research_balanced_batch_001/repricing_labels.csv` |
| `repricing_labels_batch_002.csv` | 42 | `repriced_favorably` | `polymarket/data/repricing_research_balanced_batch_002/repricing_labels.csv` |
| `repricing_labels_short_replay.csv` | 28 | `repriced_favorably` | `polymarket/models/repricing_research_v1/short_replay/repricing_labels.csv` |
| `repricing_labels_soak_v1.csv` | 73 | `repriced_favorably` | `polymarket/runs/repricing_paper_soak_v1/.../repricing_labels.csv` |
| `repricing_labels_soak_v2_recovery.csv` | 84 | `repriced_favorably` | `polymarket/runs/repricing_paper_soak_v2/.../repricing_labels.csv` |
| `repricing_labels_soak_v3.csv` | 175 | `repriced_favorably` | `polymarket/runs/repricing_paper_soak_v3/.../repricing_labels.csv` |
| `repricing_labels_soak_v4.csv` | 166 | `repriced_favorably` | `polymarket/runs/repricing_paper_soak_v4/.../repricing_labels.csv` |
| `baseline_v1_predictions_validation.csv` | 153 | reference only | `polymarket/models/baseline_v1/predictions_validation.csv` |
| `baseline_v1_validation_report.json` | - | reference only | `polymarket/models/baseline_v1/validation_report.json` |

**Deliberately not copied**: raw `session.jsonl` capture logs (100-350MB each,
unprocessed tick-by-tick, duplicated across many pipeline-input folders),
wallet-intelligence trade history (a different, unrelated research branch),
and `proxy_reference_dataset` (labels sourced from an unvalidated external
reference feed the source docs say "must not be used unless explicitly
enabled").

## Two different label semantics — do not mix them

- **`outcome`-labeled files** (`microstructure_dataset_*`, `outcome_training_dataset`):
  label = the market's actual final UP/DOWN resolution. This is what our live
  bot needs, since it holds to expiry.
- **`repriced_favorably`-labeled files** (`repricing_labels_*`): label = whether
  a *different*, early-exit strategy (buy the lagging side, exit at a 0.03
  target / 0.03 stop / 180s timeout) would have won *before* the market
  resolved. It is not a proxy for final outcome and should not be trained
  against as if it were one.

`core/quant_model.py` trains only on the `outcome`-labeled microstructure
datasets for this reason.

## 2026-07-04 sprint: combined historical+live model, and why it doesn't ship

`scripts/train_quant_model_v2.py` combined all outcome-labeled data available
to this project (1490 historical rows from this directory + 335 resolved
rows from this project's own `data/quant_features.jsonl` shadow log = 1825
total, 47.9% win rate) and 5-fold cross-validated LogisticRegression,
RandomForest, and GradientBoosting. Naive mixed-CV results looked strong
(GradientBoosting: 68.8% accuracy, AUC 0.751, clearing the 55% bar easily).

That result does not survive scrutiny. Historical data (Jun 18-24) and live
data (Jul 2-3) have different win rates (49.7% vs 40.1%) and come from
different capture pipelines, so a model trained and evaluated on a random
mix of both can score well by exploiting source-specific quirks rather than
a real live signal. The decisive test is training **only** on historical
data and evaluating **only** on live data the model never saw — i.e. the
actual situation the live bot is in. Under that honest holdout, every model
scored at or below chance (AUC 0.46-0.52) and all trailed the plain
`yes_price` baseline evaluated on that same live holdout (accuracy 63.0%,
AUC 0.586). This is source leakage, not an edge, and matches every prior
finding in this project and in forgeview-ai's own research: no model tried
so far beats trusting the market's own YES price. The GradientBoosting
model was still refit on all 1825 rows and saved to `data/quant_model.pkl`
for continued shadow-mode logging (harmless, no live effect) — it is not
validated for driving real trading decisions.

Point-biserial correlation with outcome, live-only (n=335, the distribution
that actually matters for the deployed bot): `yes_price` r=+0.173,
`time_remaining_pct` r=+0.106, everything else (`price_velocity`,
`order_book_imbalance`, `spread`, `spread_compression`) at or near zero.
What's still needed to find a genuine edge, if one exists: a much larger
same-regime dataset (weeks, not days, of live-only capture, since mixing
regimes with different base rates is actively misleading) and features not
yet tried here (e.g. cross-asset correlation, wallet/flow-based signals —
see forgeview-ai's `wallet_intelligence_v1` research, not yet incorporated
into this project).

## What the source research already found (see docs/polymarket in forgeview-ai)

- `BASELINE_PROBABILITY_MODEL_V1.md`: an L2-regularized logistic regression on
  the outcome-labeled dataset did **not** beat raw Polymarket YES price on log
  loss or Brier score (`NO_EDGE_FOUND_YET`).
- `BASELINE_FAILURE_DIAGNOSTICS_V1.md`: none of 8 feature-group combinations
  beat YES price either (`FEATURE_SET_INCOMPLETE`).
- Decisions D-031 / D-033: after building the microstructure feature set
  specifically to fix this, the microstructure-only and YES-plus-microstructure
  models *still* lost to YES price, on two independent batches.
- `REPRICING_RESEARCH_V1.md`: a separate early-exit repricing strategy showed
  real signal in paper simulation (78.9% win rate best batch, beats a
  precommitted random-timing baseline at p<0.001), but the edge **weakens to
  negative** once realistic ~2-second execution latency and fees are modeled
  — and their poll cadence was already faster than our live bot's 15s.
- No Kelly-criterion implementation exists in the source repo; it appears only
  as a backlog item gated behind "proven calibrated probabilities and positive
  net expectancy," a bar that was never cleared.

We independently re-ran the same kind of experiment (`core/quant_model.py`) on
`microstructure_dataset_batch_001.csv` + `batch_002.csv` and got the same
result: log loss 0.598 vs. 0.591 for YES price, Brier 0.207 vs. 0.205 — the
model does not beat the naive baseline. See the sprint commit message for the
full run.
