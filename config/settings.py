"""All tunable parameters for the v3 quant bot."""

# ---- mode ----
PAPER_MODE = True

# ---- assets / timeframes ----
ASSETS = ["BTC", "ETH", "SOL"]
BINANCE_SYMBOLS = {"BTC": "btcusdt", "ETH": "ethusdt", "SOL": "solusdt"}
# window size in seconds per timeframe label
TIMEFRAMES = {"5m": 300, "15m": 900}

# ---- risk (exact values, 2026-07-18 spec) ----
BANKROLL_USD = 100.0                    # paper bankroll
MAX_DAILY_LOSS_USD = 100.0              # daily stop-loss
MAX_OPEN_POSITIONS_5M = 5               # max concurrent on 5-min markets
MAX_OPEN_POSITIONS_15M = 5              # max concurrent on 15-min markets
MAX_CONSECUTIVE_LOSSES = 10             # stop after this many losses in a row
CONSECUTIVE_LOSS_PAUSE_MINUTES = 20     # temporary pause, not a permanent block
# Shared by both the model's own trade gate (core/model.py) and Kelly sizing
# (core/risk_manager.py) -- the spec gives one number for both.
KELLY_MIN_EXAMPLES = 100
KELLY_MAX_FRACTION = 0.05               # Kelly capped at 5% of bankroll
KELLY_MAX_POSITION_USD = 5.0            # absolute cap, regardless of bankroll growth
FIXED_POSITION_USD = 2.0                # flat size before KELLY_MIN_EXAMPLES is reached

# ---- entry filters ----
MIN_SECONDS_REMAINING_5M = 120          # don't enter in the last 2 min of a 5-min window
MIN_SECONDS_REMAINING_15M = 360         # don't enter in the last 6 min of a 15-min window
ENTRY_YES_PRICE_MIN = 0.35
ENTRY_YES_PRICE_MAX = 0.65
MODEL_TRADE_THRESHOLD_YES = 0.55        # P(UP) > this -> YES
MODEL_TRADE_THRESHOLD_NO = 0.45         # P(UP) < this -> NO

# ---- trading hours ----
# 2026-07-19 trade log: 06-12 UTC was the only profitable 6h block (60.8% win
# rate, +$1.48 avg pnl); the other three blocks were flat-to-negative.
TRADING_HOURS_START_UTC = 6
TRADING_HOURS_END_UTC = 12

# ---- execution ----
TAKER_FEE_RATE = 0.0                    # fee formula wired in now, rate is 0 for the moment

# ---- market feed ----
CONTEXT_POLL_INTERVAL_SEC = 3
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
# Enough finalized 1m bars for the 30m momentum feature plus headroom.
BINANCE_KLINE_HISTORY_MIN = 35
BINANCE_RECONNECT_BACKOFF_SEC = 5

POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"

# rolling window (in resolved windows) for the BTC<->ETH/SOL correlation feature
CORRELATION_WINDOW = 20

# ---- models ----
def model_weights_path(asset: str, timeframe: str) -> str:
    return f"data/models/model_{asset.lower()}_{timeframe}.pkl"


# ---- trades / logs ----
# A new filename on purpose -- keeps v3's trade history from mixing with the
# v2 paper_trades.jsonl history preserved during the archive move.
PAPER_TRADES_LOG = "data/trades/paper_trades_v3.jsonl"
RISK_STATE_FILE = "data/trades/risk_state_v3.json"
BOT_STATUS_FILE = "data/market/bot_status.json"

# ---- dashboard ----
DASHBOARD_PORT = 8080

# ---- console summary ----
CONSOLE_SUMMARY_INTERVAL_SEC = 60
