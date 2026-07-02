import logging
import time
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run")

from config.settings import MARKET_POLL_INTERVAL_SEC
from core.market_fetcher import MarketFetcher
from core.repricing_detector import RepricingDetector
from core.state_manager import StateManager
from core.dedup_guard import DedupGuard
from core.paper_trading_engine import PaperTradingEngine
from core.pnl_tracker import PnLTracker
from signals.repricing_signal import RepricingSignalGenerator
from reporting.stats_reporter import StatsReporter
from reporting.telegram_reporter import TelegramReporter

def main():
    logger.info("=== ForgeViewAI starting ===")
    state = StateManager()
    dedup = DedupGuard()
    fetcher = MarketFetcher()
    detector = RepricingDetector()
    engine = PaperTradingEngine(state, dedup)
    tracker = PnLTracker()
    signal_gen = RepricingSignalGenerator(detector, state)
    stats_rep = StatsReporter(tracker, state)
    tg = TelegramReporter()

    if state.is_stopped():
        reason = state.get("stop_reason", "unknown")
        logger.warning(f"Previously stopped: {reason}. Exiting.")
        return

    tg.send_text("ForgeViewAI paper trading started")
    last_report_ts = datetime.datetime.now(datetime.timezone.utc)

    while True:
        try:
            if state.is_stopped():
                tg.send_stop(state.get("stop_reason", ""))
                break
            _maybe_reset_daily(state)
            _close_resolved_trades(engine, fetcher, tg, tracker, stats_rep)
            markets = fetcher.get_active_5min_markets()
            for market in markets:
                signal = signal_gen.process_market(market)
                if signal:
                    trade = engine.open_trade(signal)
                    if trade:
                        tg.send_signal(signal, trade)
                        _log_signal(signal)
            if (datetime.datetime.now(datetime.timezone.utc) - last_report_ts).total_seconds() > 3600:
                report = stats_rep.generate_report()
                logger.info(report)
                tg.send_text(report)
                last_report_ts = datetime.datetime.now(datetime.timezone.utc)
            time.sleep(MARKET_POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Shutting down")
            logger.info(stats_rep.generate_report())
            break
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)
            time.sleep(30)

def _close_resolved_trades(engine, fetcher, tg, tracker, stats_rep):
    for trade in engine.get_open_trades():
        resolution = fetcher.get_market_resolution(trade.market_id)
        if not resolution:
            continue
        outcome = fetcher.resolve_outcome(resolution)
        if outcome is None:
            continue
        closed = engine.close_trade(trade.market_id, outcome)
        if closed:
            tg.send_close(closed, tracker.compute_stats())
            logger.info(stats_rep.generate_report())

def _maybe_reset_daily(state: StateManager):
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_daily_reset", "") != today:
        state.reset_daily()
        state.set("last_daily_reset", today)

def _log_signal(signal):
    import json
    from config.settings import SIGNALS_LOG
    os.makedirs(os.path.dirname(SIGNALS_LOG), exist_ok=True)
    with open(SIGNALS_LOG, "a") as f:
        f.write(json.dumps(signal.to_dict()) + "\n")

if __name__ == "__main__":
    main()
