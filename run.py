import logging
import time
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run")

from config.settings import (
    MARKET_POLL_INTERVAL_SEC,
    LIVE_STATUS_FILE, MARKET_BIAS_REFRESH_SEC, MARKET_BIAS_LOG, EXCHANGE_STATUS_FILE,
    FEAR_GREED_LOG, FEAR_GREED_REFRESH_SEC, MACRO_EVENTS_LOG, MACRO_EVENTS_REFRESH_SEC,
    QUANT_ONLY_MODE, EXECUTION_CYCLE_FILE, PRICE_HISTORY_LOG, SIGNALS_LOG,
    STATS_EXPORT_INTERVAL_SEC, WARMUP_FLAT_SIZE_USD,
)
from core.market_fetcher import MarketFetcher
from core.market_bias import MarketBiasFetcher, FearGreedFetcher
from core.macro_events import MacroEventsFetcher
from core.repricing_detector import RepricingDetector, RepricingSignal
from core.state_manager import StateManager
from core.dedup_guard import DedupGuard
from core.paper_trading_engine import PaperTradingEngine
from core.pnl_tracker import PnLTracker
from core.quant_signal import QuantSignalGenerator
from core.live_features import LiveFeatureCollector
from core.online_model import OnlineQuantModel
from core.signal_combiner import SignalCombiner
from core.no_shadow_tracker import NoShadowTracker
from core.stats_calculator import StatsCalculator
from reporting.stats_reporter import StatsReporter
from reporting.telegram_reporter import TelegramReporter

# module-level so the market_bias_provider lambda passed into QuantSignalGenerator
# always reads the latest fetched value, not a snapshot taken at startup
_external_status = {"bias": None, "last_check_ts": None}
_fear_greed_status = {"last_check_ts": None}
_macro_events_status = {"last_check_ts": None}
_stats_status = {"last_export_ts": None}

def main():
    logger.info("=== ForgeViewAI starting ===")
    state = StateManager()
    dedup = DedupGuard()
    fetcher = MarketFetcher()
    bias_fetcher = MarketBiasFetcher()
    fear_greed_fetcher = FearGreedFetcher()
    macro_events_fetcher = MacroEventsFetcher()
    detector = RepricingDetector()
    engine = PaperTradingEngine(state, dedup)
    tracker = PnLTracker()
    signal_gen = QuantSignalGenerator(detector, state, fetcher,
                                      market_bias_provider=lambda: _external_status["bias"])
    live_features = LiveFeatureCollector()
    online_model = OnlineQuantModel()
    signal_combiner = SignalCombiner()
    no_shadow = NoShadowTracker()
    stats_rep = StatsReporter(tracker, state)
    stats_calc = StatsCalculator()
    tg = TelegramReporter()
    logger.info(f"Online model: {online_model.n_updates}/{online_model.warmup_trades} warm-up trades "
                f"({'LIVE (model-driven)' if online_model.is_warmed_up() else 'WARMUP (repricing rule)'})")
    logger.info(f"QUANT_ONLY_MODE={QUANT_ONLY_MODE}: repricing signal is "
                f"{'DISABLED for trading (shadow-logging only)' if QUANT_ONLY_MODE else 'active'}")

    if state.is_stopped():
        _auto_reset_on_stop(state, tg)

    tg.send_text("ForgeViewAI paper trading started")
    last_report_ts = datetime.datetime.now(datetime.timezone.utc)

    while True:
        try:
            if state.is_stopped():
                _auto_reset_on_stop(state, tg)
            _maybe_reset_daily(state)
            _maybe_refresh_external_status(bias_fetcher, fetcher)
            _maybe_refresh_fear_greed(fear_greed_fetcher)
            _maybe_refresh_macro_events(macro_events_fetcher)
            _maybe_export_stats(stats_calc)
            _close_resolved_trades(engine, fetcher, tg, tracker, stats_rep, online_model)
            no_shadow.resolve_pending(fetcher, online_model)
            signal_gen.resolve_pending()
            markets = fetcher.get_active_5min_markets()
            for market in markets:
                _export_execution_cycle("scan", asset=market["asset"], market_id=market["market_id"],
                                         detail="scanning market for signals")
                if market["asset"] in ("BTC", "ETH", "SOL"):
                    _log_price_history(market["asset"], market["yes_price"])
                live_features.update(market["market_id"], market["asset"], market["yes_price"], market["no_price"])
                # kept only for its shadow-logging side effect (data/quant_features.jsonl via the
                # old static QuantModel) -- its return value is NOT used for trading decisions in
                # quant-only mode; the signal combiner no longer accepts or weights it at all.
                signal_gen.process_market(market)
                snapshot = live_features.extract(market, fetcher)
                _export_live_status(snapshot)
                if market["asset"] in ("BTC", "ETH", "SOL"):
                    no_shadow.maybe_record(market, snapshot, online_model.predict_proba_one(snapshot))
                combined_signal = signal_combiner.combine(
                    market, fetcher, snapshot.get("btc_eth_correlation"),
                )
                if combined_signal is not None:
                    _export_execution_cycle("detect", asset=combined_signal.asset,
                                             market_id=combined_signal.market_id, detail=combined_signal.reason)
                _decide_and_open(engine, online_model, market, combined_signal, snapshot, tg)
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

def _decide_and_open(engine, online_model, market, combined_signal, snapshot, tg):
    should_trade, direction, win_probability, reason = online_model.decide(snapshot, combined_signal)
    if not should_trade:
        return None
    _export_execution_cycle("validate", asset=market["asset"], market_id=market["market_id"], detail=reason)
    if online_model.is_warmed_up():
        # Real Kelly fraction (2026-07-07 CRITICAL FIX, direction-aware for
        # NO trades added the same day) computed from the model's own
        # win_probability AND the ACTUAL side being entered's own price --
        # win_probability from decide() is always raw P(YES wins) regardless
        # of direction, so a NO trade needs P(NO)=1-win_probability and
        # NO's own entry price (no_price = 1-yes_price), or kelly_size()
        # would be handed the wrong side's price/probability pair entirely.
        # A non-positive fraction (no edge) returns exactly 0.0, meaning
        # skip the trade entirely, not open it at $0. Checked BEFORE
        # constructing the RepricingSignal so a no-edge tick doesn't build
        # an object it'll never use.
        if direction == "YES":
            kelly_p, kelly_price = win_probability, market["yes_price"]
        else:
            kelly_p, kelly_price = 1.0 - win_probability, market["no_price"]
        size_usd = online_model.kelly_size(kelly_p, kelly_price)
        if size_usd <= 0:
            logger.debug(
                f"Kelly fraction non-positive for {market['asset']} {direction} "
                f"(p={kelly_p:.3f}, entry_price={kelly_price:.3f}) -- no edge, skipping"
            )
            return None
        signal = RepricingSignal(
            asset=market["asset"], market_id=market["market_id"], direction=direction,
            yes_price=market["yes_price"], no_price=market["no_price"],
            confidence=round(win_probability, 3), reason=reason,
            minutes_remaining=market.get("minutes_remaining", 5.0),
            decisive_signal=combined_signal.decisive_signal,
        )
        _export_execution_cycle("size", asset=market["asset"], market_id=market["market_id"],
                                 size_usd=size_usd, detail=f"sized at ${size_usd:.2f}")
        trade = engine.open_trade(signal, source="online_model", size_usd=size_usd)
    else:
        # warm-up bootstrap fallback: the model has no training data of its own
        # yet, so it needs *some* real trades to learn from. UNGATED FROM
        # QUANT_ONLY_MODE (2026-07-06 urgent fix): this used to return None
        # here unconditionally, meaning a reset model could never re-warm
        # itself (no trade ever opened to call record_features()/resolve()).
        # `combined_signal` at this point is already the signal combiner's own
        # output (decide()'s warmup branch only returns True when it's
        # non-None), so this still isn't the legacy repricing detector making
        # the call -- it's the combiner alone, model gate bypassed, exactly
        # per decide()'s warmup docstring.
        #
        # FLAT WARMUP_FLAT_SIZE_USD, NOT KELLY (2026-07-07 reversal): briefly
        # sized warm-up trades via kelly_size() too (2026-07-06), but that let
        # a totally unproven, freshly-reset model open trades up to
        # ONLINE_MODEL_KELLY_MAX_SIZE_USD purely off combiner confidence.
        # Warm-up's only job is accumulating training examples safely --
        # real Kelly sizing is reserved for after the model has actually
        # warmed up and is driving the decision itself (the branch above).
        signal = combined_signal
        size_usd = WARMUP_FLAT_SIZE_USD
        _export_execution_cycle("size", asset=market["asset"], market_id=market["market_id"],
                                 size_usd=size_usd, detail=f"sized at ${size_usd:.2f} (warmup flat)")
        trade = engine.open_trade(signal, source="repricing", size_usd=size_usd)
    if trade:
        _export_execution_cycle("fill", asset=trade.asset, market_id=trade.market_id, trade_id=trade.trade_id,
                                 size_usd=trade.size_usd, detail=f"{trade.direction} @ {trade.entry_price:.3f}")
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
            _export_execution_cycle("settle", asset=closed.asset, market_id=closed.market_id,
                                     trade_id=closed.trade_id, pnl_usd=closed.pnl_usd,
                                     detail=f"{closed.result} pnl={closed.pnl_usd:+.2f}")
            # online_model.predict_proba_one() always estimates P(YES wins),
            # never "did our specific bet win" -- with NO-direction trades now
            # possible (see core/signal_combiner.py's extreme-reversion zone),
            # the training label must stay "did YES actually win" regardless
            # of which direction we happened to bet, or the model's target
            # would silently flip meaning depending on trade.direction.
            online_model.resolve(closed.market_id, 1 if outcome == "YES" else 0)
            tg.send_close(closed, tracker.compute_stats())
            logger.info(stats_rep.generate_report())

def _maybe_reset_daily(state: StateManager):
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_daily_reset", "") != today:
        state.reset_daily()
        state.set("last_daily_reset", today)

def _log_signal(signal):
    import json
    os.makedirs(os.path.dirname(SIGNALS_LOG), exist_ok=True)
    with open(SIGNALS_LOG, "a") as f:
        f.write(json.dumps(signal.to_dict()) + "\n")

def _log_price_history(asset: str, yes_price: float):
    """Appends one row per poll tick, unconditioned by any filter or signal --
    unlike quant_features.jsonl's 'signal' stage (only written when the old
    repricing rule fires), this is meant to answer where yes_price actually
    spends its time so SIGNAL_COMBINER_MIN/MAX_YES_PRICE can be set from real
    data instead of a biased sample."""
    import json
    os.makedirs(os.path.dirname(PRICE_HISTORY_LOG), exist_ok=True)
    entry = {
        "asset": asset,
        "yes_price": yes_price,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(PRICE_HISTORY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def _maybe_refresh_external_status(bias_fetcher: MarketBiasFetcher, fetcher: MarketFetcher):
    """Runs at most once every MARKET_BIAS_REFRESH_SEC (60s), independent of
    the 3s market-poll cadence: a CoinGecko spot-price fetch and a Polymarket
    reachability ping, both real external checks. Updates the module-level
    _external_status dict the repricing signal filter reads from, appends a
    row to data/market_bias.jsonl, and mirrors exchange reachability to
    data/exchange_status.json for the dashboard."""
    now = datetime.datetime.now(datetime.timezone.utc)
    last = _external_status["last_check_ts"]
    if last and (now - last).total_seconds() < MARKET_BIAS_REFRESH_SEC:
        return
    _external_status["last_check_ts"] = now

    bias_result = bias_fetcher.fetch()
    if bias_result:
        _external_status["bias"] = bias_result["market_bias"]
        _log_market_bias(bias_result, now)
    else:
        logger.warning("Market bias fetch failed this cycle; leaving prior bias/filter state unchanged")

    exchange_ok = fetcher.ping()
    _export_exchange_status(exchange_ok, now)

def _log_market_bias(result: dict, now: datetime.datetime):
    import json
    os.makedirs(os.path.dirname(MARKET_BIAS_LOG), exist_ok=True)
    entry = dict(result)
    entry["timestamp"] = now.isoformat()
    with open(MARKET_BIAS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def _export_exchange_status(ok: bool, now: datetime.datetime):
    import json
    os.makedirs(os.path.dirname(EXCHANGE_STATUS_FILE), exist_ok=True)
    tmp_path = EXCHANGE_STATUS_FILE + ".tmp"
    data = {"status": "OK" if ok else "ERROR", "checked_at": now.isoformat()}
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, EXCHANGE_STATUS_FILE)
    except Exception as e:
        logger.error(f"exchange status export error: {e}")

def _maybe_refresh_fear_greed(fear_greed_fetcher: FearGreedFetcher):
    """Runs at most once every FEAR_GREED_REFRESH_SEC (15 min) -- the index
    itself only updates roughly daily in reality, so polling every 60s like
    the price checks would be pointless API load for no new information."""
    now = datetime.datetime.now(datetime.timezone.utc)
    last = _fear_greed_status["last_check_ts"]
    if last and (now - last).total_seconds() < FEAR_GREED_REFRESH_SEC:
        return
    _fear_greed_status["last_check_ts"] = now
    result = fear_greed_fetcher.fetch()
    if not result:
        logger.warning("Fear & Greed fetch failed this cycle")
        return
    import json
    os.makedirs(os.path.dirname(FEAR_GREED_LOG), exist_ok=True)
    data = dict(result)
    data["updated_at"] = now.isoformat()
    tmp_path = FEAR_GREED_LOG + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, FEAR_GREED_LOG)
    except Exception as e:
        logger.error(f"fear & greed export error: {e}")

def _maybe_refresh_macro_events(macro_events_fetcher: MacroEventsFetcher):
    """Runs at most once every MACRO_EVENTS_REFRESH_SEC (1 hour) -- the
    economic calendar changes at most a few times a day, so this is by far
    the least frequently refreshed external check."""
    now = datetime.datetime.now(datetime.timezone.utc)
    last = _macro_events_status["last_check_ts"]
    if last and (now - last).total_seconds() < MACRO_EVENTS_REFRESH_SEC:
        return
    _macro_events_status["last_check_ts"] = now
    events = macro_events_fetcher.fetch_next_events(3)
    if events is None:
        logger.warning("Macro events fetch failed this cycle")
        return
    import json
    os.makedirs(os.path.dirname(MACRO_EVENTS_LOG), exist_ok=True)
    data = {"events": events, "updated_at": now.isoformat()}
    tmp_path = MACRO_EVENTS_LOG + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, MACRO_EVENTS_LOG)
    except Exception as e:
        logger.error(f"macro events export error: {e}")

def _maybe_export_stats(stats_calc: StatsCalculator):
    """Runs at most once every STATS_EXPORT_INTERVAL_SEC (60s) -- re-reads
    and re-aggregates every resolved trade each time, cheap at current
    volume but no reason to redo it every 3s poll tick."""
    now = datetime.datetime.now(datetime.timezone.utc)
    last = _stats_status["last_export_ts"]
    if last and (now - last).total_seconds() < STATS_EXPORT_INTERVAL_SEC:
        return
    _stats_status["last_export_ts"] = now
    stats_calc.export()

def _export_live_status(snapshot):
    """Mirrors the live BTC/ETH correlation (and a fresh timestamp the
    dashboard can use as a poll-liveness signal) to a small JSON file --
    LiveFeatureCollector's own state lives only in this process's memory and
    is otherwise unreadable from a browser."""
    import json
    os.makedirs(os.path.dirname(LIVE_STATUS_FILE), exist_ok=True)
    data = {
        "btc_eth_correlation": snapshot.get("btc_eth_correlation"),
        "quant_only_mode": QUANT_ONLY_MODE,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    tmp_path = LIVE_STATUS_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, LIVE_STATUS_FILE)
    except Exception as e:
        logger.error(f"live status export error: {e}")

def _export_execution_cycle(stage: str, asset=None, market_id=None, trade_id=None,
                             size_usd=None, pnl_usd=None, detail=""):
    """Writes the single most-recently-reached real pipeline stage (scan ->
    detect -> validate -> size -> fill, plus a separate settle event when a
    trade closes) for the dashboard's animated Execution Cycle panel. This is
    a live pointer, not a per-trade history -- most ticks only ever reach
    "scan" since most markets don't produce a signal, which is the honest
    common case, not a bug."""
    import json
    data = {
        "stage": stage,
        "asset": asset,
        "market_id": market_id,
        "trade_id": trade_id,
        "size_usd": size_usd,
        "pnl_usd": pnl_usd,
        "detail": detail,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        os.makedirs(os.path.dirname(EXECUTION_CYCLE_FILE), exist_ok=True)
        tmp_path = EXECUTION_CYCLE_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, EXECUTION_CYCLE_FILE)
    except Exception as e:
        logger.error(f"execution cycle export error: {e}")

if __name__ == "__main__":
    main()
