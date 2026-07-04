REPRICING_FROZEN = {
    "min_price_move": 0.03,
    "min_time_window_sec": 60,
    "max_time_window_sec": 240,
    "confidence_threshold": 0.55,
    "assets": ["BTC", "ETH"],
    "market_duration_min": 5,
    "min_yes_price": 0.45,
    "max_yes_price": 0.60,
}
PAPER_TRADE_SIZE_USD = 10.0
MAX_OPEN_POSITIONS = 5
MAX_DAILY_LOSS_USD = 10000.0
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
QUANT_MODEL_PATH = "data/quant_model.pkl"
KELLY_FRACTION_CAP = 0.25
ONLINE_MODEL_STATE_FILE = "data/online_model_state.pkl"
ONLINE_MODEL_WARMUP_TRADES = 200
ONLINE_MODEL_CONFIDENCE_THRESHOLD = 0.55
ONLINE_MODEL_BANKROLL_USD = 1000.0
ONLINE_MODEL_MIN_TRADE_USD = 1.0
ONLINE_MODEL_MAX_TRADE_USD = 10.0
# Once live (post-warmup), a trade now requires BOTH the model's own
# prediction (calibrated p > ONLINE_MODEL_OWN_THRESHOLD) AND the signal
# combiner's independent agreement (a non-None combiner signal, which by
# construction only exists when its confidence already exceeds
# SIGNAL_COMBINER_THRESHOLD) -- see core/online_model.py.decide().
ONLINE_MODEL_OWN_THRESHOLD = 0.5
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
# in. This affects every consumer of predict_proba_one() (decide() and
# kelly_size()'s win_probability input alike), not just display.
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
MARKET_BIAS_BULLISH_THRESHOLD = 1.0
MARKET_BIAS_BEARISH_THRESHOLD = -1.0
MARKET_BIAS_LOG = "data/market_bias.jsonl"
MARKET_BIAS_REFRESH_SEC = 60
EXCHANGE_STATUS_FILE = "data/exchange_status.json"
LATENCY_LOG = "data/latency.json"
LATENCY_WINDOW = 200
FEAR_GREED_API_BASE = "https://api.alternative.me/fng/"
FEAR_GREED_LOG = "data/fear_greed.json"
FEAR_GREED_REFRESH_SEC = 900
MACRO_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MACRO_EVENTS_LOG = "data/macro_events.json"
MACRO_EVENTS_REFRESH_SEC = 3600

# ---- order book signal ----
ORDER_BOOK_RATIO_THRESHOLD = 2.0

# ---- momentum (bounce/reversal) signal ----
MOMENTUM_WINDOW_SEC = 45
MOMENTUM_MIN_SAMPLES = 3

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

# Quant-only mode: the repricing detector is fully disabled as a live trading
# input. Trades require BOTH online_model's own calibrated p > 0.5 AND
# signal_combiner's order_book+momentum+volume confidence > 0.60 (see
# core/online_model.py decide() and run.py's defense-in-depth skip of any
# signal_source="repricing" trade). The repricing detector/signal_gen keep
# running for shadow-logging purposes only (data/quant_features.jsonl), not
# for trading decisions.
QUANT_ONLY_MODE = True
SIGNAL_COMBINER_STATUS_FILE = "data/signal_combiner_status.json"
