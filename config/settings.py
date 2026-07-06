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
SIGNAL_COOLDOWN_SEC = 120
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
# still-valid standalone utility module with its own tests) uses it
# independently of the online model's own sizing. It's no longer used by
# core/online_model.py's kelly_size(), which now uses the flat BET_SIZES
# table below instead of the Kelly formula.
KELLY_FRACTION_CAP = 0.25
ONLINE_MODEL_STATE_FILE = "data/online_model_state.pkl"
ONLINE_MODEL_WARMUP_TRADES = 200
ONLINE_MODEL_CONFIDENCE_THRESHOLD = 0.55
# Simple step-function bet sizing, replacing the old Kelly-criterion formula
# (which used to need ONLINE_MODEL_BANKROLL_USD, ONLINE_MODEL_MIN_TRADE_USD,
# and ONLINE_MODEL_MAX_TRADE_USD -- all removed, since a flat lookup table
# needs none of them: the table's own values ARE the floor/ceiling now).
# Keyed by signal_combiner confidence (NOT the online model's own calibrated
# probability) -- core/online_model.py.kelly_size() finds the largest
# threshold the confidence clears and returns that flat dollar amount.
BET_SIZES = {0.60: 5, 0.70: 10, 0.80: 15, 0.90: 25}
# Once live (post-warmup), a trade now requires BOTH the model's own
# prediction (calibrated p > ONLINE_MODEL_OWN_THRESHOLD) AND the signal
# combiner's independent agreement (a non-None combiner signal, which by
# construction only exists when its confidence already exceeds
# SIGNAL_COMBINER_THRESHOLD) -- see core/online_model.py.decide().
#
# TEMPORARY LOWERING (2026-07-06): the model (297/200 warm-up trades, already
# past warm-up) was trained almost entirely on 0.45-0.60 yes_price data, so
# now that SIGNAL_COMBINER_MIN/MAX_YES_PRICE is temporarily 0.35-0.65 (see
# above), it's extrapolating p<=0.5 for the new price wings just from lack of
# training data there, blocking every combiner-agreed signal in that range.
# Lowered to 0.3 here (not reset to 0.5 threshold + wiped model weights) --
# a model *reset* was considered and rejected as the riskier option: with
# QUANT_ONLY_MODE=True, run.py's warm-up fallback branch (the ONLY path that
# would normally open trades and call record_features()/resolve() to
# re-accumulate n_updates during warm-up) is explicitly skipped, so a reset
# model would never open a trade and could never re-warm itself -- a
# permanent trading halt, not just slower learning. Lowering this threshold
# is a single reversible constant with no such failure mode. Revert to 0.5
# once the model has enough resolved trades in the new 0.35-0.65 range to
# have relearned it (or once the price band itself reverts to 0.45-0.60).
ONLINE_MODEL_OWN_THRESHOLD = 0.3
ONLINE_MODEL_COMBINER_THRESHOLD = 0.60
# Probability calibration: the raw SGDClassifier sigmoid output saturates to
# ~0.0/1.0 on this project's limited real training data (observed
# coefficients growing past +/-50 by 234 real updates). Rather than trust
# that raw output, predict_proba_one() runs it through a tanh-based
# transform that compresses the full (0,1) range into approximately
# (ONLINE_MODEL_CALIBRATION_LOWER, ONLINE_MODEL_CALIBRATION_UPPER) --
# asymptotic, so it approaches but never quite reaches those bounds, no
# matter how extreme the raw prediction is. p_raw=0.5 still maps to exactly
# 0.5 (uncalibrated confidence is unaffected); only the extremes are pulled
# in. This affects every consumer of predict_proba_one() -- decide()'s
# threshold check -- not just display. (kelly_size() no longer consumes this
# value at all: it's keyed off signal_combiner confidence instead, since the
# BET_SIZES sprint.)
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
SIGNAL_COMBINER_MIN_YES_PRICE = 0.35
SIGNAL_COMBINER_MAX_YES_PRICE = 0.65

# DISABLED (2026-07-06): an "extreme mean-reversion" strategy briefly traded
# NO above yes_price=0.80 and YES below 0.20 (SIGNAL_COMBINER_EXTREME_LOW/
# HIGH_YES_PRICE), with a graduated size cap (SIZE_CAP_*) layered on top.
# Removed after real results: 10.5% win rate over 19 resolved NO-direction
# trades, net -$139.67 -- not a viable strategy. Everything outside
# [SIGNAL_COMBINER_MIN_YES_PRICE, SIGNAL_COMBINER_MAX_YES_PRICE] is simply
# not traded again, same as before this was ever added.

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
