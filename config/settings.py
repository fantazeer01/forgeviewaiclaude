"""All tunable parameters for the v4 six-layer quant bot."""

# ---- mode ----
PAPER_MODE = True

# ---- assets / timeframes ----
ASSETS = ["BTC", "ETH", "SOL"]
BINANCE_SYMBOLS = {"BTC": "btcusdt", "ETH": "ethusdt", "SOL": "solusdt"}
TIMEFRAMES = {"5m": 300, "15m": 900}
# 5m real-money trading disabled (2026-07-22: post-peak analysis showed 45%
# win rate on 5m vs 60.6% on 15m) -- shadow learning still runs on ALL of
# TIMEFRAMES so 5m models keep training; only real trade entry is gated.
ENABLED_TIMEFRAMES = ["15m"]

# ---- layer4_wallet: risk ----
BANKROLL_USD = 100.0
MAX_DAILY_LOSS_USD = 100.0
MAX_OPEN_POSITIONS_5M = 5
MAX_OPEN_POSITIONS_15M = 5
MAX_CONSECUTIVE_LOSSES = 10
CONSECUTIVE_LOSS_PAUSE_MINUTES = 20
TIMEFRAME_MAX_CONSECUTIVE_LOSSES = 5   # stricter, faster breaker scoped to one timeframe only
TIMEFRAME_LOSS_PAUSE_MINUTES = 30
KELLY_MIN_EXAMPLES = 200               # raised from v3's 100 -- fewer, better-trained trades
KELLY_MAX_FRACTION = 0.03              # 3% of bankroll, more conservative than v3's 5%
KELLY_MAX_POSITION_USD = 5.0
WARMUP_POSITION_SIZE = 2.0             # flat size before KELLY_MIN_EXAMPLES real trades close

# ---- layer4_wallet: take profit / drawdown ----
DAILY_TAKE_PROFIT_USD = 50.0           # stop trading for the day once hit
DRAWDOWN_FROM_PEAK_PCT = 0.30          # halve position size once bankroll falls 30% off its daily peak

# ---- layer3_conscience: confidence_filter ----
# YES raised 0.57->0.60 (2026-07-22: YES win rate 45.3% vs NO's 56.2% in the
# post-peak sample -- YES needs more conviction to justify a trade at all).
CONFIDENCE_YES_THRESHOLD = 0.60
CONFIDENCE_NO_THRESHOLD = 0.43
HIGH_VOLATILITY_CONFIDENCE_THRESHOLD = 0.60  # overrides CONFIDENCE_YES_THRESHOLD in that regime

# ---- layer3_conscience: price_band_filter ----
# Present in v3 (inside model.decide()) but dropped by accident during the
# v4 rebuild -- restored as its own layer3 filter. Without it, a market
# already sitting at yes_price=0.835 (or 0.165) can still open a trade with
# almost no room to profit if right and a near-certain loss if wrong.
ENTRY_YES_PRICE_MIN = 0.35
ENTRY_YES_PRICE_MAX = 0.65

# ---- layer3_conscience: liquidity_filter ----
MIN_BOOK_DEPTH_USD = 100.0
MAX_BID_ASK_SPREAD_PCT = 0.05

# ---- layer3_conscience: timing_filter ----
MIN_SECONDS_REMAINING_5M = 120
MIN_SECONDS_REMAINING_15M = 360
MIN_WINDOW_AGE_SEC = 30                # skip the first 30s of a window -- prices are still settling

# ---- layer3_conscience: regime_detector ----
REGIME_TREND_MOMENTUM_BPS = 10.0       # |spot_momentum_15m| above this + low vol => trending
REGIME_HIGH_VOL_MULTIPLIER = 2.0       # volatility_5m > this * median => HIGH_VOLATILITY
REGIME_RANGE_SIZE_MULTIPLIER = 0.5     # halve position size while regime == RANGE
REGIME_HISTORY_WINDOW = 50             # samples of volatility_5m kept for the median

# ---- layer6_memory: adaptive_state ----
ADAPTIVE_LOOKBACK_TRADES = 20
ADAPTIVE_HOT_WIN_RATE = 0.55
ADAPTIVE_COLD_WIN_RATE = 0.45
ADAPTIVE_COLD_SIZE_MULTIPLIER = 0.5

# ---- layer6_memory: pattern_memory ----
PATTERN_MIN_TRADES_FOR_SIGNAL = 10     # need at least this many historical matches to trust the stat
PATTERN_BREAKEVEN_AVG_PNL = 0.0        # below this -> skip the trade

# ---- execution ----
TAKER_FEE_RATE = 0.0

# ---- layer1_eyes: market feed ----
CONTEXT_POLL_INTERVAL_SEC = 3
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
BINANCE_KLINE_HISTORY_MIN = 65         # enough finalized 1m bars for the 60m momentum feature
BINANCE_RECONNECT_BACKOFF_SEC = 5

POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"

CORRELATION_WINDOW = 20

# ---- layer1_eyes: news_feed ----
import os  # noqa: E402

CRYPTOPANIC_API_BASE = "https://cryptopanic.com/api/v1/posts/"
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")
NEWS_POLL_INTERVAL_SEC = 60
NEWS_SENTIMENT_WINDOW_HOURS = 1
NEWS_CACHE_FILE = "data/market/news_cache.json"
NEWS_MAJOR_SOURCES = {"coindesk", "cointelegraph", "reuters", "bloomberg", "theblock"}

# ---- layer1_eyes: fear_greed ----
FEAR_GREED_API_URL = "https://api.alternative.me/fng/"
FEAR_GREED_POLL_INTERVAL_SEC = 900     # 15 minutes
FEAR_GREED_CACHE_FILE = "data/market/fear_greed.json"

# ---- layer1_eyes: whale_tracker ----
WHALE_POLL_INTERVAL_SEC = 10
WHALE_TRADE_MIN_USD = 500.0
WHALE_WINDOW_MINUTES = 5

# ---- layer2_brain: models ----
# Same path/convention as v3 on purpose -- data/models/ was preserved across
# the archive move, so shadow-learned examples carry over instead of
# starting from zero.
def model_weights_path(asset: str, timeframe: str) -> str:
    return f"data/models/model_{asset.lower()}_{timeframe}.pkl"


# ---- trades / logs ----
# New filenames on purpose -- v4's risk state and trade log start fresh
# under the new rules (KELLY_MIN_EXAMPLES=200, take-profit, drawdown) even
# though the models themselves keep learning continuously.
PAPER_TRADES_LOG = "data/trades/paper_trades_v4.jsonl"
RISK_STATE_FILE = "data/trades/risk_state_v4.json"
BOT_STATUS_FILE = "data/market/bot_status.json"
ADAPTIVE_STATE_FILE = "data/market/adaptive_state.json"
PATTERN_MEMORY_FILE = "data/memory/patterns.json"

# ---- dashboard ----
DASHBOARD_PORT = 8080

# ---- console summary ----
CONSOLE_SUMMARY_INTERVAL_SEC = 60
