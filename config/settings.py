"""All tunable parameters for the v2 quant bot."""

# ---- mode ----
PAPER_MODE = True

# ---- bankroll / risk ----
BANKROLL_USD = 100.0
FIXED_POSITION_PCT = 0.02          # 2% flat sizing before Kelly kicks in
KELLY_WARMUP_TRADES = 50           # trades needed before Kelly sizing activates
KELLY_FRACTION_CAP = 0.05          # Kelly capped at 5% of bankroll post-warmup
DAILY_LOSS_LIMIT_USD = 10.0        # 10% of bankroll
MAX_OPEN_POSITIONS = 3
LOSS_STREAK_LIMIT = 5              # consecutive losses before pausing
LOSS_STREAK_PAUSE_SEC = 30 * 60

# ---- ensemble ----
ENSEMBLE_WEIGHTS = {
    "momentum": 0.5,
    "volume": 0.3,
    "macro": 0.2,
}
ENSEMBLE_YES_SCORE_THRESHOLD = 0.55
ENSEMBLE_YES_PRICE_BAND = (0.45, 0.65)
ENSEMBLE_NO_SCORE_THRESHOLD = 0.45
ENSEMBLE_NO_PRICE_BAND = (0.35, 0.55)
ENSEMBLE_MIN_TRAINING_EXAMPLES = 30
WARMUP_TRADE_SIZE_USD = 2.0        # fixed size while accumulating the first training examples
FAIR_VALUE_TRADE_SIZE_USD = 2.0    # fixed size for the fair-value strategy (2026-07-13, no stats yet)

# ---- macro model ----
MACRO_FEAR_GREED_BEARISH = 25
MACRO_FEAR_GREED_BULLISH = 75
MACRO_BEARISH_BIAS = -0.1
MACRO_BULLISH_BIAS = 0.1
MACRO_ASIA_CLOSE_UTC = (6, 10)
MACRO_NYSE_OPEN_UTC = (13, 17)
MACRO_VOLATILITY_BIAS = 0.05
MACRO_BIAS_CLAMP = 0.2

# ---- market context ----
CONTEXT_POLL_INTERVAL_SEC = 3
WINDOW_SEC = 300
ASSETS = ["BTC", "ETH", "SOL"]
BINANCE_SYMBOLS = {"BTC": "btcusdt", "ETH": "ethusdt", "SOL": "solusdt"}
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
BINANCE_KLINE_HISTORY_MIN = 65     # keep enough 1m bars for the 60m momentum window
BINANCE_RECONNECT_BACKOFF_SEC = 5

POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"

FEAR_GREED_API_BASE = "https://api.alternative.me/fng/"
FEAR_GREED_REFRESH_SEC = 900
FEAR_GREED_LOG = "data/market/fear_greed.json"

# ---- models ----
# Each asset trains its own momentum/volume model independently -- one
# resolved BTC trade never updates SOL's weights and vice versa. These
# generic paths stay as the class-level default (used only if a caller
# doesn't pass an explicit weights_file); real per-asset instances always
# get an explicit path from momentum_weights_path()/volume_weights_path().
MOMENTUM_WEIGHTS_FILE = "data/models/momentum_weights.pkl"
VOLUME_WEIGHTS_FILE = "data/models/volume_weights.pkl"


def momentum_weights_path(asset: str) -> str:
    return f"data/models/momentum_weights_{asset.lower()}.pkl"


def volume_weights_path(asset: str) -> str:
    return f"data/models/volume_weights_{asset.lower()}.pkl"

# ---- trades / logs ----
PAPER_TRADES_LOG = "data/trades/paper_trades.jsonl"
MARKET_CONTEXT_LOG = "data/market/context.jsonl"
BOT_STATUS_FILE = "data/market/bot_status.json"
# Just trades_closed -- the one RiskManager field Kelly sizing needs to
# survive a restart. daily_pnl/loss_streak/paused_until stay in-memory only
# (daily_pnl in particular must keep resetting fresh each UTC day).
RISK_STATE_FILE = "data/trades/risk_state.json"

# ---- stats-based trading filter ----
# Persisted (price_bucket, hour_bucket) win-rate table -- StatsTracker's own
# state file. Rebuilt once from PAPER_TRADES_LOG the first time it's missing,
# then updated incrementally on every real trade close.
STATS_TRACKER_FILE = "data/trades/stats_tracker.json"
# Dashboard-facing snapshot (mirrors BOT_STATUS_FILE's role for bot_status.json).
STATS_EXPORT_FILE = "data/market/stats.json"
# RAISED 20 -> 50 (2026-07-15): buckets like n=15/win_rate=20% were being
# waved through as "not enough data" when 20% over 15 trades is already a
# real signal the bucket is bad. 50 trades before the filter trusts a
# bucket's win rate is a more conservative, reliable bar.
STATS_MIN_SAMPLES = 50
STATS_MIN_WIN_RATE = 0.52
# Early block (2026-07-16): a bucket with win_rate < 0.45 at n >= 10 is
# already a real losing signal -- don't wait for n=50 to veto it. Buckets
# that clear this bar keep accumulating normally (should_trade() still
# returns True) until they hit STATS_MIN_SAMPLES and get judged by the
# full STATS_MIN_WIN_RATE bar above.
STATS_EARLY_BLOCK_MIN_SAMPLES = 10
STATS_EARLY_BLOCK_MAX_WIN_RATE = 0.45

# ---- dashboard ----
DASHBOARD_PORT = 8080

# ---- console summary ----
CONSOLE_SUMMARY_INTERVAL_SEC = 60
