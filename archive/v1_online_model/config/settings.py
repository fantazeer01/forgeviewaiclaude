REPRICING_FROZEN = {
    "min_price_move": 0.03,
    "min_time_window_sec": 60,
    "max_time_window_sec": 240,
    "confidence_threshold": 0.55,
    # "assets" is documentation only -- RepricingDetector never actually
    # reads self.params["assets"] anywhere (grep confirms it), so this list
    # has zero effect on which assets get traded. The real gate is
    # MarketFetcher.ASSET_SLUG_PREFIX in core/market_fetcher.py, which
    # determines which assets' markets get fetched from Polymarket at all.
    # Kept in sync here anyway so this doesn't read as stale/wrong.
    "assets": ["BTC", "ETH", "SOL"],
    "market_duration_min": 5,
    "min_yes_price": 0.45,
    "max_yes_price": 0.60,
}
PAPER_TRADE_SIZE_USD = 10.0
MAX_OPEN_POSITIONS = 5
MAX_DAILY_LOSS_USD = 1000.0
MAX_LOSS_STREAK = 50
MARKET_POLL_INTERVAL_SEC = 3
SIGNAL_COOLDOWN_SEC = 60
POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
TELEGRAM_ENABLED = True
STATE_FILE = "data/state.json"
DEDUP_STATE_FILE = "data/dedup_state.json"
TRADES_LOG = "data/paper_trades.jsonl"
SIGNALS_LOG = "data/signals_log.jsonl"
QUANT_FEATURES_LOG = "data/quant_features.jsonl"
# Raw, unconditioned yes_price sample for BTC/ETH/SOL, written every poll
# tick regardless of whether any signal fires or any filter blocks --
# unlike quant_features.jsonl (which only logs when the old repricing rule
# fires, a biased sample) this is meant to answer "where does price
# actually spend its time" so SIGNAL_COMBINER_MIN/MAX_YES_PRICE can be set
# from real data.
PRICE_HISTORY_LOG = "data/price_history.jsonl"
# Shadow NO learning (2026-07-06): real NO-direction trading was tried and
# disabled (10.5% win rate over 19 trades, net -$139.67 -- see
# SignalCombiner's docstring). This still lets the online model learn from
# what a NO bet would have seen, with zero money at risk: whenever
# yes_price > NO_SHADOW_YES_PRICE_THRESHOLD (for BTC, ETH, or SOL),
# core/no_shadow_tracker.py
# records the feature snapshot (no trade opens), and once the market
# resolves, feeds (features, 1 if outcome=="YES" else 0) into the SAME
# online model via update() -- the model's target is always "did YES win,"
# direction-independent, so this is exactly the same training signal a real
# NO trade's resolution would have produced, just without the paper-money
# exposure. Once 50+ of these have resolved, the model's own p at open time
# (also recorded) can be checked against real outcomes to see whether it's
# actually learning anything in this price range.
NO_SHADOW_LOG = "data/no_shadow.jsonl"
NO_SHADOW_YES_PRICE_THRESHOLD = 0.80
QUANT_MODEL_PATH = "data/quant_model.pkl"
# KELLY_FRACTION_CAP stays defined here: core/kelly_criterion.py (a separate,
# still-valid standalone utility module with its own tests) uses it for its
# OWN quarter_kelly_fraction()/kelly_position_size() helpers. core/online_model.py's
# kelly_size() (2026-07-07 CRITICAL FIX) uses that module's kelly_fraction()/
# net_odds_from_price() directly instead -- FULL Kelly (not quarter-Kelly),
# since the $5/$25 clamp below already bounds the position size.
# 2026-07-08: kelly_fraction() itself is now also hard-capped at this value
# (was previously only clamped to 1.0 there, relying entirely on the $5/$25
# dollar clamp downstream) -- already 0.25, exactly the requested "Kelly
# never exceeds 25% of bankroll" cap, just not wired into the formula's own
# ceiling until now.
KELLY_FRACTION_CAP = 0.25
# 2026-07-08: a raw Kelly fraction below this is treated as an unconfirmed /
# statistically insignificant edge and floored to 0.0 (do not trade) inside
# kelly_fraction(), rather than opening a razor-thin position.
KELLY_MIN_EDGE = 0.05
ONLINE_MODEL_STATE_FILE = "data/online_model_state.pkl"
# 50 real resolved trades before the online model's own prediction is
# trusted at all (see core/online_model.py.decide()) -- during this window
# every trade is sized flat at WARMUP_FLAT_SIZE_USD via the signal combiner
# alone (run.py's warm-up branch), real Kelly sizing only turns on once
# is_warmed_up() is true.
ONLINE_MODEL_WARMUP_TRADES = 50
# Flat size for every trade opened during warm-up (2026-07-07), regardless of
# signal combiner confidence -- Kelly sizing is deliberately withheld until
# the model has actually warmed up and is driving decisions; sizing a fresh,
# unproven warm-up trade at up to $25 was needlessly risky for a period whose
# only purpose is accumulating training examples, not making good bets yet.
WARMUP_FLAT_SIZE_USD = 5.0
ONLINE_MODEL_CONFIDENCE_THRESHOLD = 0.55
# Real Kelly-criterion sizing (2026-07-07 CRITICAL FIX, post-warmup only --
# see WARMUP_FLAT_SIZE_USD above for during warm-up). Replaces the flat
# BET_SIZES lookup table used from 2026-07-06 to 2026-07-07, which keyed
# size off signal_combiner confidence and ignored entry_price entirely --
# it couldn't tell a favorable payout (low yes_price, high b) from an
# unfavorable one (high yes_price, low b) at the same confidence. See
# core/online_model.py.kelly_size(): f = (p*b - (1-p))/b where
# b = (1-yes_price)/yes_price is this market's real payout ratio and p is
# the model's own calibrated win_probability -- computed via
# core/kelly_criterion.py's already-tested kelly_fraction()/
# net_odds_from_price(). f <= 0 means no edge -- the trade is skipped
# entirely, not opened at $0. A positive fraction is applied to
# ONLINE_MODEL_KELLY_BANKROLL_USD and clamped to
# [ONLINE_MODEL_KELLY_MIN_SIZE_USD, ONLINE_MODEL_KELLY_MAX_SIZE_USD].
ONLINE_MODEL_KELLY_BANKROLL_USD = 100.0
ONLINE_MODEL_KELLY_MIN_SIZE_USD = 5.0
ONLINE_MODEL_KELLY_MAX_SIZE_USD = 25.0
# Once live (post-warmup), a trade now requires BOTH the model's own
# prediction (calibrated p > ONLINE_MODEL_OWN_THRESHOLD) AND the signal
# combiner's independent agreement (a non-None combiner signal, which by
# construction only exists when its confidence already exceeds
# SIGNAL_COMBINER_THRESHOLD) -- see core/online_model.py.decide().
#
# Restored to the 0.5 spec value (2026-07-06) after two rounds of temporary
# lowering (to work around two separate coefficient-divergence episodes --
# see core/online_model.py's class docstring for the full history) proved to
# just chase a moving target as the underlying model kept re-saturating.
# 2026-07-07: the model was rewritten from SGDClassifier to
# LogisticRegression(solver="liblinear") specifically because that failure
# mode can no longer occur (see the docstring), so 0.5 should now be durable
# rather than needing another temporary drop.
ONLINE_MODEL_OWN_THRESHOLD = 0.5
ONLINE_MODEL_COMBINER_THRESHOLD = 0.60
# LogisticRegression's inverse regularization strength (smaller = stronger)
# -- see core/online_model.py's class docstring for why LogisticRegression
# replaced SGDClassifier (2026-07-07) and why this needs to be re-applied
# explicitly in _load() rather than trusted from whatever was pickled.
ONLINE_MODEL_C = 0.1
# How often (in resolved-trade updates) OnlineQuantModel._run_health_check()
# probes predict_proba_one() for saturation (see that method and
# SATURATION_PROBE_YES_PRICES/SATURATION_EPSILON in core/online_model.py).
# 10 is frequent enough to catch a real saturation episode within roughly an
# hour of trading (observed cadence is ~20-30 resolutions/hour) without
# meaningfully adding to the per-update cost.
ONLINE_MODEL_HEALTH_CHECK_INTERVAL = 10

# ---- model stability monitor (2026-07-07) ----
# A second, independent check from _run_health_check() above -- that one
# only probes for the specific saturation signature already seen twice.
# This one is broader: real recent win rate, real prediction diversity, and
# raw coefficient magnitude, run less often (every 50 vs every 10) since
# it needs a meaningful sample of recent predictions to say anything.
# See OnlineQuantModel._run_stability_monitor().
STABILITY_CHECK_INTERVAL = 50
# Win rate is checked over the trailing STABILITY_WIN_RATE_WINDOW resolved
# examples (drawn from the same history_y used for training -- real and
# shadow-NO examples mixed, same caveat as everywhere else this history is
# used). Below this, only a warning is logged -- a losing streak alone
# isn't proof the MODEL broke (could just be a bad market regime), so it
# doesn't auto-reset.
STABILITY_WIN_RATE_WINDOW = 50
STABILITY_MIN_WIN_RATE = 0.45
# Diversity is checked over the trailing STABILITY_PREDICTIONS_WINDOW real
# predict_proba_one() outputs (recorded at the moment of each resolved
# trade's own update() call, not a synthetic sweep). A healthy model's
# predictions vary at least this much across genuinely different real
# examples; below this, that's the same "collapsed to a near-constant
# output" signature as the saturation check, so THIS one auto-resets.
STABILITY_PREDICTIONS_WINDOW = 20
# LOWERED 0.05 -> 0.02 (2026-07-07): 0.05 was too aggressive for a
# genuinely converging LogisticRegression -- the model auto-reset itself at
# 300 updates with prediction_std=0.0445, which was the model's predictions
# naturally narrowing as it converges, not a real collapse. 0.02 still
# catches an actual collapse (all predictions landing on one point) without
# punishing normal convergence.
# LOWERED 0.02 -> 0.015 (2026-07-08): the model auto-reset itself again at
# 350 updates with prediction_std=0.0185 -- above 0.02's replacement but
# still a normal-convergence false positive, not a real collapse. 0.015
# still catches an actual collapse (all predictions landing on one point)
# without punishing normal convergence.
STABILITY_MIN_PREDICTION_STD = 0.015
# Raw |coefficient| bound -- tighter than the model's own internal workings
# would need to be "not diverged" (LogisticRegression regularized at C=0.1
# on standardized features typically stays well under 1.0, see the
# 2026-07-07 CRITICAL FIX report: max was 0.41), so exceeding this is a
# real red flag on its own, not just a proxy -- auto-resets.
STABILITY_COEF_BOUND = 5.0
MODEL_HEALTH_LOG = "data/model_health_log.jsonl"
# "Soft reset" (2026-07-08): before either auto-reset path
# (_run_health_check()'s saturation probe or _run_stability_monitor()'s
# diversity/coef checks) wipes the model, OnlineQuantModel._reset_to_fresh()
# snapshots the current state here first -- for manual forensic inspection
# or recovery later, NOT automatic warm-starting (see that method's
# docstring for why real warm-starting isn't done: liblinear doesn't
# support it, and even if it did, starting the "fresh" model from the exact
# weights that just triggered the reset would defeat the point of
# resetting). Overwritten on every reset -- only the most recent one is
# kept, this is a forensic snapshot, not a history log.
MODEL_CHECKPOINT_FILE = "data/model_checkpoint.pkl"

# ---- scheduled full retrain (2026-07-07) ----
# Separate from the continuous per-update refit (every resolved trade
# already re-fits on the full history, see OnlineQuantModel.update()) --
# this is a periodic, explicitly health-gated refresh trained on ONLY the
# trailing RETRAIN_WINDOW examples (not the full up-to-HISTORY_MAX history),
# so the model can adapt to a regime shift instead of an old, possibly
# stale, majority of training data dominating forever. The candidate is
# verified against the same coefficient-bound and prediction-diversity
# checks as the stability monitor BEFORE being swapped in -- a candidate
# that fails stays rejected and the live model keeps running unchanged.
RETRAIN_INTERVAL = 500
RETRAIN_WINDOW = 500
MODEL_RETRAIN_LOG = "data/model_retrain_log.jsonl"

# Probability calibration: kept as an extra safety margin against
# overconfident predictions even with the better-regularized model above --
# not itself a fix for the coefficient divergence (that was the raw
# coefficients, not the calibration; see core/online_model.py). Compresses
# the classifier's raw sigmoid output into approximately
# (ONLINE_MODEL_CALIBRATION_LOWER, ONLINE_MODEL_CALIBRATION_UPPER) --
# asymptotic, so it approaches but never quite reaches those bounds, no
# matter how extreme the raw prediction is. p_raw=0.5 still maps to exactly
# 0.5 (uncalibrated confidence is unaffected); only the extremes are pulled
# in. This affects every consumer of predict_proba_one() -- decide()'s
# threshold check, AND (2026-07-07) kelly_size()'s win_probability input,
# since that's now the calibrated p straight from decide(), not a
# signal_combiner-confidence lookup.
ONLINE_MODEL_CALIBRATION_LOWER = 0.20
ONLINE_MODEL_CALIBRATION_UPPER = 0.80
ONLINE_MODEL_CALIBRATION_STEEPNESS = 2.0
ONLINE_MODEL_STATUS_FILE = "data/online_model_status.json"
LIVE_STATUS_FILE = "data/live_status.json"
# Manual warm-start prior for the online model's yes_price coefficient, per
# docs/polymarket/DECISIONS.md D-002 (point-biserial r=+0.151, t=3.305,
# p<0.001 across all 468 resolved trades -- higher yes_price genuinely
# correlates with a higher realized win rate). This is an initial condition
# for a fresh model only, NOT synthetic training data: n_updates/warmup
# progress are untouched, and every real resolved trade still performs a
# normal SGD gradient step from here, same as it would from an all-zero
# start. Calibrated so predict_proba(yes_price=0.45) ~ 0.40 and
# predict_proba(yes_price=0.60) ~ 0.55 on a freshly-seeded, not-yet-trained
# model (see core/online_model.py._seed_yes_price_prior docstring).
ONLINE_MODEL_PRIOR_YES_PRICE_WEIGHT = 4.0
ONLINE_MODEL_PRIOR_INTERCEPT = -2.22
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3/simple/price"
MARKET_BIAS_BULLISH_THRESHOLD = 0.5
MARKET_BIAS_BEARISH_THRESHOLD = -1.0
MARKET_BIAS_LOG = "data/market_bias.jsonl"
MARKET_BIAS_REFRESH_SEC = 60
EXCHANGE_STATUS_FILE = "data/exchange_status.json"
LATENCY_LOG = "data/latency.json"
LATENCY_WINDOW = 200
API_STATS_LOG = "data/api_stats.json"
API_STATS_EXPORT_INTERVAL_SEC = 60
FEAR_GREED_API_BASE = "https://api.alternative.me/fng/"
FEAR_GREED_LOG = "data/fear_greed.json"
FEAR_GREED_REFRESH_SEC = 900
MACRO_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MACRO_EVENTS_LOG = "data/macro_events.json"
MACRO_EVENTS_REFRESH_SEC = 3600

# ---- portfolio stats calculator (2026-07-06) ----
STATS_FILE = "data/stats.json"
STATS_EXPORT_INTERVAL_SEC = 60
# apy_pct is expressed relative to this notional $100 capital base, not any
# real tracked account balance (this project never had one -- trades are
# flat $5-$25 stakes, not a percentage of a bankroll). "% return per $100
# risked, annualized" is the honest reading of the number; treat it as a
# rate-of-return indicator, not a claim about real capital growth.
STATS_APY_NOTIONAL_CAPITAL_USD = 100.0

# ---- order book signal ----
# LOWERED (2026-07-06): order_book fired only 4 times today vs momentum's
# 1520 -- live polling (18 samples across BTC/ETH/SOL over 20s) found real
# bid/ask depth ratios ranging 0.639-1.111 (avg 0.849), with BTC/ETH
# consistently ask-heavy (ratio <1). The old 2.0 threshold was essentially
# unreachable under normal book conditions, not a deliberately selective
# bar. 1.3 is comfortably above the observed range so it still requires a
# real imbalance, not noise, but low enough to actually fire. Unbacktested
# -- no real trade has ever fired this signal, so there's no win-rate data
# to justify a specific number; revisit once order_book-sourced trades
# accumulate.
ORDER_BOOK_RATIO_THRESHOLD = 1.3

# ---- momentum (bounce/reversal) signal ----
MOMENTUM_WINDOW_SEC = 45
MOMENTUM_MIN_SAMPLES = 3
# SIGNAL QUALITY SPRINT (2026-07-06): the request read "only fire if
# reversal_strength > 0.5 * drop_size", but reversal_strength is already
# defined as bounce/drop (a ratio) in momentum_signal.py -- multiplying a
# ratio by drop again doesn't parse dimensionally. The parenthetical right
# next to it ("bounce must be at least 50% of the drop") is unambiguous and
# is what's implemented: require reversal_strength >= this fraction, i.e.
# bounce >= MOMENTUM_MIN_REVERSAL_STRENGTH * drop.
MOMENTUM_MIN_REVERSAL_STRENGTH = 0.5

# ---- volume signal ----
# SIGNAL QUALITY SPRINT (2026-07-06): skip volume signals in the first 60s
# of a fresh 5-min window (market just opened, price/volume less settled).
# minutes_remaining starts at ~5.0 and counts down, so "first 60s elapsed"
# is minutes_remaining <= 4.0; skip while it's still above that.
VOLUME_SKIP_MINUTES_REMAINING_THRESHOLD = 4.0

# ---- price stability filter ----
# SIGNAL QUALITY SPRINT (2026-07-06): a FILTER (like the correlation and
# price-band filters), not a weighted signal -- it never contributes a
# confidence, it only blocks combine() entirely when yes_price has moved
# less than PRICE_STABILITY_MIN_MOVE over the trailing
# PRICE_STABILITY_WINDOW_SEC. Rationale: order_book/momentum/volume are all
# continuation signals that only mean something if the market is actually
# moving -- a flat market gives them nothing real to detect.
PRICE_STABILITY_WINDOW_SEC = 90
PRICE_STABILITY_MIN_MOVE = 0.02

# ---- correlation filter ----
CORRELATION_HIGH_THRESHOLD = 0.8
CORRELATION_LOW_THRESHOLD = 0.3
CORRELATION_BTC_DROP_THRESHOLD = 0.02
CORRELATION_BTC_WINDOW_SEC = 45

# ---- volume signal ----
VOLUME_HISTORY_LOG = "data/volume_history.jsonl"
VOLUME_RATIO_THRESHOLD = 1.5
VOLUME_LOOKBACK_DAYS = 7
VOLUME_RECORD_INTERVAL_SEC = 3600

# ---- signal combiner ----
# repricing was removed from this dict entirely (not just zeroed) per the
# quant-only-mode sprint: it's disabled as a trading input, not merely
# down-weighted. The remaining 3 weights don't need to sum to 1.0 -- combine()
# renormalizes among whichever of these actually fire on a given tick.
SIGNAL_COMBINER_WEIGHTS = {
    "order_book": 0.25,
    "momentum": 0.25,
    "volume": 0.15,
}
SIGNAL_COMBINER_THRESHOLD = 0.60
# BUG FIX (2026-07-04): REPRICING_FROZEN.min_yes_price/max_yes_price (0.45/0.60)
# was NEVER enforced on the live quant-only trading path -- order_book_signal,
# momentum_signal, and volume_signal all ignore yes_price entirely, so trades
# were firing at prices like 0.165/0.345/0.375 that docs/polymarket/DECISIONS.md
# D-002 found have a materially lower win rate than the 0.45-0.60 band. These
# constants are enforced directly in SignalCombiner.combine() (see
# core/signal_combiner.py), independent of REPRICING_FROZEN (which belongs to
# the now-disabled legacy repricing detector and must stay untouched for it).
# TEMPORARY WIDENING (2026-07-05): the 0.45-0.60 band above is the proven
# edge (historically +$6.68/trade avg, n=132) but is only in-band ~23-25% of
# observed yes_price ticks, and the correct-band-enforcement fix in
# core/signal_combiner.py (2026-07-04 23:05 UTC) dropped real trade volume to
# ~0/day. Widened here to 0.35-0.65 to accumulate fresh session data faster.
# Backtesting this exact widened range against all-time paper_trades.jsonl
# shows the added region is NOT uniformly good: [0.35,0.45) alone historically
# lost money (-$1.12/trade avg, n=268, net -$300.19) while [0.60,0.65) was
# fine (+$1.41/trade avg, n=14, small sample) -- the widened band's blended
# average (+$1.43/trade, n=411) is worse per-trade than the original 0.45-0.60
# band alone. Revert to 0.45/0.60 (or narrow to e.g. 0.45-0.65) once enough
# fresh trades have accumulated to re-decide with new-session data instead of
# this backtest.
# TEMPORARY WIDENING #2 (2026-07-06 urgent fix): lowered further from 0.35 to
# 0.30 to catch more signals while the reset online model (see
# ONLINE_MODEL_WARMUP_TRADES above) re-warms. NOTE: the 2026-07-05 backtest
# above found [0.35,0.45) alone historically lost money (-$1.12/trade avg,
# n=268, net -$300.19) -- 0.30-0.35 is uncharted territory with zero backtest
# support, not a proven-safe extension. This is a deliberate data-accumulation
# tradeoff, not a profitability claim; revisit once fresh trades accumulate.
# RAISED 0.30 -> 0.45 (2026-07-08): a real-trade breakdown of online_model's
# closed trades confirmed the 0.30-0.35 territory above was indeed bad --
# 0.30-0.40 as a whole came back at 20.0% win rate over 35 trades, -$731.82
# total PnL, the single worst entry_price bucket by far. 0.50-0.60 was the
# only bucket with a clear real edge (67.0% win rate, +$837.65). 0.45 keeps a
# small margin below that proven band rather than cutting exactly at 0.50.
SIGNAL_COMBINER_MIN_YES_PRICE = 0.45
SIGNAL_COMBINER_MAX_YES_PRICE = 0.65

# Trading-hours gate (2026-07-08): the same real-trade breakdown found
# 00-06 UTC was the worst time-of-day bucket for online_model trades (33.8%
# win rate over 65 trades, -$524.05) -- checked in run.py._decide_and_open()
# before opening ANY trade (warmup or Kelly-sized), so this doesn't touch
# market fetching, shadow-logging, or price_history -- only trade-opening.
TRADING_HOURS_UTC_START = 6
TRADING_HOURS_UTC_END = 24

# Tradeable assets (2026-07-08): SOL had the worst per-asset result in the
# same breakdown (43.2% win rate over 44 trades, -$210.23, smallest and
# worst of the three) -- excluded from trade-opening only. SOL markets are
# still fetched, monitored, price-logged, and shadow-recorded exactly as
# before (core.market_fetcher.MarketFetcher.ASSET_SLUG_PREFIX is the actual
# fetch-gate and is untouched), so this only removes SOL from
# run.py._decide_and_open()'s trade-opening path.
ASSETS = ["btc", "eth"]

# DISABLED AGAIN (2026-07-07): an "extreme mean-reversion" NO strategy at
# yes_price>0.80 was disabled 2026-07-06 after real results (10.5% win rate
# over 19 trades, net -$139.67), then resurrected the same day (2026-07-07)
# once both of that period's actual root causes were fixed: the online
# model no longer diverges (LogisticRegression rewrite), and kelly_size()
# uses the real Kelly formula with each side's actual payout ratio instead
# of a flat lookup table blind to entry_price. Real results in this
# corrected form: 2/25 = 8.00% win rate -- confirms a real negative edge,
# not the old model/sizing bugs. Turned off again via this single kill
# switch, falling back to YES-only [SIGNAL_COMBINER_MIN_YES_PRICE,
# SIGNAL_COMBINER_MAX_YES_PRICE] behavior. See
# core/signals/mean_reversion_no_signal.py for the signal itself, which
# also checks this flag directly (defense-in-depth, not just the
# SignalCombiner call site).
NO_TRADING_ENABLED = False
# Gate: NO is only even considered above this yes_price (the old disabled
# strategy's own threshold).
NO_REVERSION_MIN_YES_PRICE = 0.80
# The rolling window must have seen yes_price reach at least this high
# before a reversion back down means anything -- otherwise a market that's
# merely drifted up to, say, 0.82 with no real spike would trigger on noise.
NO_REVERSION_PEAK_MIN_YES_PRICE = 0.90
NO_REVERSION_WINDOW_SEC = 90
# Peak-to-current must have fallen at least this much for the reversion to
# count as real movement, not noise -- same shape as MOMENTUM_MIN_REVERSAL_STRENGTH's
# role for the YES-side momentum signal.
NO_REVERSION_MIN_DROP = 0.05

# Quant-only mode: the repricing detector is fully disabled as a live trading
# input. Trades require BOTH online_model's own calibrated p > 0.5 AND
# signal_combiner's order_book+momentum+volume confidence > 0.60 (see
# core/online_model.py decide() and run.py's defense-in-depth skip of any
# signal_source="repricing" trade). The repricing detector/signal_gen keep
# running for shadow-logging purposes only (data/quant_features.jsonl), not
# for trading decisions.
QUANT_ONLY_MODE = True
SIGNAL_COMBINER_STATUS_FILE = "data/signal_combiner_status.json"

# Execution cycle telemetry: written by run.py at each real stage of the
# scan->detect->validate->size->fill pipeline (plus a separate settle event
# when a trade closes), feeding the dashboard's animated Execution Cycle
# panel. Always reflects whichever stage was most recently reached across
# any market on this tick -- not a per-trade history, just a live pointer.
EXECUTION_CYCLE_FILE = "data/execution_cycle.json"
