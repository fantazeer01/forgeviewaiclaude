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
