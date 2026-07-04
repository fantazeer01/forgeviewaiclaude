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
