REPRICING_FROZEN = {
    "min_price_move": 0.03,
    "min_time_window_sec": 60,
    "max_time_window_sec": 240,
    "confidence_threshold": 0.55,
    "assets": ["BTC", "ETH"],
    "market_duration_min": 5,
    "min_yes_price": 0.30,
    "max_yes_price": 0.60,
}
PAPER_TRADE_SIZE_USD = 10.0
MAX_OPEN_POSITIONS = 5
MAX_DAILY_LOSS_USD = 50.0
MAX_LOSS_STREAK = 5
MARKET_POLL_INTERVAL_SEC = 15
SIGNAL_COOLDOWN_SEC = 300
POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
TELEGRAM_ENABLED = True
STATE_FILE = "data/state.json"
DEDUP_STATE_FILE = "data/dedup_state.json"
TRADES_LOG = "data/paper_trades.jsonl"
SIGNALS_LOG = "data/signals_log.jsonl"
