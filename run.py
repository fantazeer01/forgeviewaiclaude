import logging
import time
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run")

from config.settings import MARKET_POLL_INTERVAL_SEC, ONLINE_MODEL_BANKROLL_USD, ONLINE_MODEL_MIN_TRADE_USD
from core.market_fetcher import MarketFetcher
from core.repricing_detector import RepricingDetector, RepricingSignal
from core.state_manager import StateManager
from core.dedup_guard import DedupGuard
from core.paper_trading_engine import PaperTradingEngine
from core.pnl_tracker import PnLTracker
from core.quant_signal import QuantSignalGenerator
from core.live_features import LiveFeatureCollector
from core.online_model import OnlineQuantModel
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
    signal_gen = QuantSignalGenerator(detector, state, fetcher)
    live_features = LiveFeatureCollector()
    online_model = OnlineQuantModel()
    stats_rep = StatsReporter(tracker, state)
    tg = TelegramReporter()
    logger.info(f"Online model: {online_model.n_updates}/{online_model.warmup_trades} warm-up trades "
                f"({'LIVE (model-driven)' if online_model.is_warmed_up() else 'WARMUP (repricing rule)'})")

    if state.is_stopped():
        _auto_reset_on_stop(state, tg)

    tg.send_text("ForgeViewAI paper trading started")
    last_report_ts = datetime.datetime.now(datetime.timezone.utc)

    while True:
        try:
            if state.is_stopped():
                _auto_reset_on_stop(state, tg)
            _maybe_reset_daily(state)
            _close_resolved_trades(engine, fetcher, tg, tracker, stats_rep, online_model)
            signal_gen.resolve_pending()
            markets = fetcher.get_active_5min_markets()
            for market in markets:
                live_features.update(market["market_id"], market["asset"], market["yes_price"], market["no_price"])
                repricing_signal = signal_gen.process_market(market)
                snapshot = live_features.extract(market, fetcher)
                _decide_and_open(engine, online_model, market, repricing_signal, snapshot, tg)
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

def _decide_and_open(engine, online_model, market, repricing_signal, snapshot, tg):
    should_trade, direction, win_probability, reason = online_model.decide(snapshot, repricing_signal)
    if not should_trade:
        return None
    if online_model.is_warmed_up():
        signal = RepricingSignal(
            asset=market["asset"], market_id=market["market_id"], direction=direction,
            yes_price=market["yes_price"], no_price=market["no_price"],
            confidence=round(win_probability, 3), reason=reason,
            minutes_remaining=market.get("minutes_remaining", 5.0),
        )
        entry_price = market["yes_price"] if direction == "YES" else market["no_price"]
        size_usd = online_model.kelly_size(win_probability, entry_price, ONLINE_MODEL_BANKROLL_USD)
        if size_usd < ONLINE_MODEL_MIN_TRADE_USD:
            return None
        trade = engine.open_trade(signal, source="online_model", size_usd=size_usd)
    else:
        signal = repricing_signal
        trade = engine.open_trade(signal, source="repricing")
    if trade:
        online_model.record_features(trade.market_id, snapshot)
        tg.send_signal(signal, trade)
        _log_signal(signal)
    return trade

def _auto_reset_on_stop(state: StateManager, tg: TelegramReporter):
    reason = state.get("stop_reason", "unknown")
    logger.warning(f"Auto-reset: system was stopped ({reason}). Resetting daily loss/loss streak and resuming.")
    state.reset_daily()
    state.update({"system_stopped": False, "stop_reason": ""})
    tg.send_text(f"Auto-reset after stop: {reason}. Resuming trading.")

def _close_resolved_trades(engine, fetcher, tg, tracker, stats_rep, online_model):
    for trade in engine.get_open_trades():
        resolution = fetcher.get_market_resolution(trade.market_id)
        if not resolution:
            continue
        outcome = fetcher.resolve_outcome(resolution)
        if outcome is None:
            continue
        closed = engine.close_trade(trade.market_id, outcome)
        if closed:
            online_model.resolve(closed.market_id, 1 if closed.result == "WIN" else 0)
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
