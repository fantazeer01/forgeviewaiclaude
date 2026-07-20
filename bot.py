"""v4 main trading loop -- six layers per tick (every 3s):
  1. layer1_eyes    - refresh all data sources
  2. layer2_brain   - build features, get P(UP), train on resolved windows
  3. layer3_conscience - confidence / liquidity / timing / regime filters
  4. layer4_wallet  - risk + take-profit/drawdown sizing
  5. layer5_hands   - paper execution
  6. layer6_memory  - trade history, pattern memory, adaptive state
"""

import datetime
import json
import logging
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from config.settings import (
    ASSETS, TIMEFRAMES, CONTEXT_POLL_INTERVAL_SEC, CONSOLE_SUMMARY_INTERVAL_SEC,
    BOT_STATUS_FILE, RISK_STATE_FILE, PAPER_TRADES_LOG, model_weights_path,
    REGIME_RANGE_SIZE_MULTIPLIER,
)
from layer1_eyes.binance_feed import BinanceFeed
from layer1_eyes.polymarket_feed import PolymarketFeed
from layer1_eyes.news_feed import NewsFeed
from layer1_eyes.fear_greed import FearGreedIndex
from layer1_eyes.whale_tracker import WhaleTracker
from layer2_brain.feature_engine import build_features, CrossMarketState
from layer2_brain.model import OnlineModel
from layer3_conscience import confidence_filter, liquidity_filter, timing_filter
from layer3_conscience.regime_detector import RegimeDetector, RANGE
from layer4_wallet.risk_manager import RiskManager
from layer4_wallet.take_profit import TakeProfitManager
from layer5_hands.executor import Executor
from layer6_memory.trade_history import TradeHistory
from layer6_memory.pattern_memory import PatternMemory, price_bucket, hour_bucket
from layer6_memory.adaptive_state import AdaptiveState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

WINDOW_LEARN_FRACTION = 0.5  # shadow-learning capture point, midway through each window


class Bot:
    def __init__(self):
        # layer1_eyes
        self.binance = BinanceFeed()
        self.polymarket = PolymarketFeed()
        self.news = NewsFeed()
        self.fear_greed = FearGreedIndex()
        self.whales = {asset: WhaleTracker() for asset in ASSETS}

        # layer2_brain
        self.cross_market = CrossMarketState()
        self.regime_detector = RegimeDetector()

        # layer4_wallet
        self.risk_manager = RiskManager(state_file=RISK_STATE_FILE)
        self.take_profit = TakeProfitManager()

        # layer6_memory
        self.trade_history = TradeHistory(log_path=PAPER_TRADES_LOG)
        self.pattern_memory = PatternMemory()
        self.adaptive_state = AdaptiveState()

        # layer2_brain models + layer5_hands executors, one pair per (asset, timeframe)
        self.models = {}
        self.executors = {}
        for asset in ASSETS:
            for timeframe in TIMEFRAMES:
                key = (asset, timeframe)
                model = OnlineModel(
                    weights_file=model_weights_path(asset, timeframe), asset=asset, timeframe=timeframe
                )
                self.models[key] = model
                self.executors[key] = Executor(
                    model, self.risk_manager, trade_history=self.trade_history, pattern_memory=self.pattern_memory,
                )

        self.pending = {}           # market_id -> {trade_id, asset, timeframe, seconds_remaining, opened_at}
        self.pending_learning = {}  # market_id -> {asset, timeframe, features, captured_at, seconds_remaining}
        self.day_wins = 0
        self.day_losses = 0
        self.last_summary = 0.0
        self.last_scores = {}       # (asset, timeframe) -> {p_up, decision, snapshot, features}
        self.current_regime = RANGE

    def start(self):
        self.binance.start()

    def run_forever(self):
        self.start()
        try:
            while True:
                self.tick()
                time.sleep(CONTEXT_POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self.binance.stop()

    # ---- layer1_eyes: snapshot assembly ----

    def _snapshot(self, asset: str, timeframe: str) -> dict:
        now = datetime.datetime.now(datetime.timezone.utc)
        markets = self.polymarket.get_markets(timeframe)
        market = markets.get(asset, {})
        return {
            "asset": asset,
            "timeframe": timeframe,
            "hour_utc": now.hour,
            "weekday": now.weekday(),
            # Binance
            "spot_price": self.binance.get_price(asset),
            "price_change_1m": self.binance.get_momentum_bps(asset, 1),
            "price_change_5m": self.binance.get_momentum_bps(asset, 5),
            "price_change_15m": self.binance.get_momentum_bps(asset, 15),
            "price_change_30m": self.binance.get_momentum_bps(asset, 30),
            "price_change_60m": self.binance.get_momentum_bps(asset, 60),
            "volume_1m": self.binance.get_volume(asset, 1),
            "volume_5m": self.binance.get_volume(asset, 5),
            "volume_15m": self.binance.get_volume(asset, 15),
            "volume_ratio": self.binance.get_volume_ratio(asset),
            "volume_trend": self.binance.get_volume_trend(asset),
            "bid_ask_imbalance": self.binance.get_bid_ask_imbalance(asset),
            "volatility_5m": self.binance.get_volatility(asset, 5),
            "volatility_15m": self.binance.get_volatility(asset, 15),
            # Polymarket
            "yes_price": market.get("yes_price"),
            "no_price": market.get("no_price"),
            "yes_price_change_60s": self.polymarket.get_yes_price_change(asset, timeframe, 60),
            "yes_price_change_120s": self.polymarket.get_yes_price_change(asset, timeframe, 120),
            "volume_usd": market.get("volume_usd"),
            "volume_ratio_window": self.polymarket.get_volume_ratio_window(asset, timeframe),
            "seconds_remaining": market.get("seconds_remaining"),
            "window_sec": market.get("window_sec"),
            "market_id": market.get("market_id"),
            "up_token_id": market.get("up_token_id"),
            "book_depth_yes": market.get("book_depth_yes"),
            "book_depth_no": market.get("book_depth_no"),
            "book_depth_ratio": market.get("book_depth_ratio"),
            "book_imbalance": market.get("book_imbalance"),
            "bid_ask_spread": market.get("bid_ask_spread"),
        }

    # ---- main tick ----

    def tick(self):
        btc_snapshot_5m = self._snapshot("BTC", "5m")

        # layer1_eyes: slow-poll sources (each throttles itself internally)
        self.news.poll()
        self.fear_greed.poll()
        for asset in ASSETS:
            snap = btc_snapshot_5m if asset == "BTC" else self._snapshot(asset, "5m")
            token_id = snap.get("up_token_id")
            if token_id:
                self.whales[asset].poll(token_id)

        # layer3_conscience: regime, from BTC's own momentum/volatility
        self.current_regime = self.regime_detector.detect(
            btc_snapshot_5m.get("price_change_15m"), btc_snapshot_5m.get("volatility_5m"),
        )
        self.adaptive_state.set_regime(self.current_regime)

        # layer4_wallet: track today's peak bankroll for the drawdown rule
        self.take_profit.update(self.risk_manager.bankroll)

        btc_mom_5m = self.binance.get_momentum_bps("BTC", 5)
        for asset in ASSETS:
            if asset != "BTC":
                self.cross_market.update(asset, btc_mom_5m, self.binance.get_momentum_bps(asset, 5))

        for timeframe, window_sec in TIMEFRAMES.items():
            btc_snapshot = btc_snapshot_5m if timeframe == "5m" else self._snapshot("BTC", timeframe)
            self._process(("BTC", timeframe), btc_snapshot, window_sec, None, None)
            for asset in ASSETS:
                if asset == "BTC":
                    continue
                snapshot = self._snapshot(asset, timeframe)
                correlation = self.cross_market.correlation(asset)
                self._process((asset, timeframe), snapshot, window_sec, btc_snapshot, correlation)

        self._check_resolutions()
        self._check_window_resolutions()
        self._maybe_print_summary()
        self._export_status()

    def _news_dict(self) -> dict:
        return {
            "sentiment_1h": self.news.sentiment_1h(),
            "count_1h": self.news.count_1h(),
            "has_major": self.news.has_major_1h(),
        }

    def _fear_greed_dict(self) -> dict:
        return {"normalized": self.fear_greed.normalized(), "change_24h": self.fear_greed.change_24h()}

    def _whale_dict(self, asset: str) -> dict:
        tracker = self.whales[asset]
        return {
            "imbalance": tracker.whale_imbalance(),
            "volume_total": tracker.whale_buy_pressure() + tracker.whale_sell_pressure(),
            "activity": tracker.whale_activity(),
        }

    def _memory_dict(self) -> dict:
        return {
            "win_rate_1h": self.trade_history.win_rate(1),
            "win_rate_6h": self.trade_history.win_rate(6),
        }

    def _process(self, key, snapshot, window_sec, btc_snapshot, correlation):
        asset, timeframe = key
        features = build_features(
            snapshot, window_sec,
            news=self._news_dict(), fear_greed=self._fear_greed_dict(), whale=self._whale_dict(asset),
            memory=self._memory_dict(), regime=self.current_regime,
            btc_snapshot=btc_snapshot, correlation=correlation,
        )
        model = self.models[key]
        p_up = model.predict_proba(features) if model.is_warmed_up() else None
        decision_result = confidence_filter.decide_side(p_up, self.current_regime)
        decision = decision_result["decision"]
        self.last_scores[key] = {"p_up": p_up, "decision": decision, "snapshot": snapshot, "features": features}
        self._maybe_trade(key, snapshot, features, p_up, decision)
        self._maybe_capture_window(key, snapshot, features, window_sec)

    def _maybe_trade(self, key, snapshot, features, p_up, decision):
        asset, timeframe = key
        market_id = snapshot.get("market_id")
        if decision not in ("YES", "NO") or market_id is None or market_id in self.pending:
            return

        ok, reason = liquidity_filter.passes(snapshot)
        if not ok:
            return
        ok, reason = timing_filter.passes(timeframe, snapshot.get("seconds_remaining"), snapshot.get("window_sec"))
        if not ok:
            return
        ok, reason = self.risk_manager.can_open_trade(timeframe)
        if not ok:
            logger.info(f"SKIP [{asset}-{timeframe}] risk blocked: {reason}")
            return
        if self.take_profit.take_profit_hit(self.risk_manager.daily_pnl):
            return

        conditions = {
            "regime": self.current_regime,
            "hour_bucket": hour_bucket(snapshot.get("hour_utc")),
            "price_bucket": price_bucket(snapshot.get("yes_price")),
            "asset": asset,
        }
        if self.pattern_memory.should_avoid(conditions):
            return

        yes_price = features["yes_price"]
        entry_price = yes_price if decision == "YES" else 1 - yes_price
        win_probability = p_up if decision == "YES" else 1 - p_up
        size = self.risk_manager.position_size(win_probability, entry_price)
        if self.current_regime == RANGE:
            size *= REGIME_RANGE_SIZE_MULTIPLIER
        size *= self.take_profit.drawdown_size_multiplier(self.risk_manager.bankroll)
        size *= self.adaptive_state.size_multiplier()
        size = round(size, 2)
        if size <= 0:
            return

        trade_id = self.executors[key].open_position(
            asset, timeframe, decision, entry_price, size, features, market_id,
            model_prob=p_up, regime=self.current_regime, conditions=conditions,
        )
        self.pending[market_id] = {
            "trade_id": trade_id, "asset": asset, "timeframe": timeframe,
            "seconds_remaining": snapshot.get("seconds_remaining"), "opened_at": time.time(),
        }
        logger.info(
            f"OPEN [{asset}-{timeframe}] {decision} entry={entry_price:.3f} size=${size:.2f} "
            f"P(UP)={p_up:.3f} regime={self.current_regime}"
        )

    def _check_resolutions(self):
        for market_id, info in list(self.pending.items()):
            elapsed = time.time() - info["opened_at"]
            if elapsed < (info.get("seconds_remaining") or 300):
                continue
            outcome = self.polymarket.get_resolution(market_id)
            if outcome is None:
                continue
            outcome_up = outcome == "UP"
            key = (info["asset"], info["timeframe"])
            pnl = self.executors[key].close_position(info["trade_id"], outcome_up)
            won = pnl > 0
            self.adaptive_state.record_close(won)
            if won:
                self.day_wins += 1
            else:
                self.day_losses += 1
            logger.info(f"CLOSE [{info['asset']}-{info['timeframe']}] outcome={outcome} pnl=${pnl:+.2f}")
            del self.pending[market_id]

    def _maybe_capture_window(self, key, snapshot, features, window_sec):
        market_id = snapshot.get("market_id")
        seconds_remaining = snapshot.get("seconds_remaining")
        if market_id is None or seconds_remaining is None or market_id in self.pending_learning:
            return
        if seconds_remaining <= window_sec * WINDOW_LEARN_FRACTION:
            self.pending_learning[market_id] = {
                "asset": key[0], "timeframe": key[1], "features": features,
                "captured_at": time.time(), "seconds_remaining": seconds_remaining,
            }

    def _check_window_resolutions(self):
        for market_id, info in list(self.pending_learning.items()):
            elapsed = time.time() - info["captured_at"]
            if elapsed < info["seconds_remaining"]:
                continue
            outcome = self.polymarket.get_resolution(market_id)
            if outcome is None:
                continue
            outcome_up = outcome == "UP"
            key = (info["asset"], info["timeframe"])
            self.models[key].learn(info["features"], outcome_up)
            logger.info(
                f"LEARN [{info['asset']}-{info['timeframe']}] outcome={outcome} "
                f"(n_examples={self.models[key].n_examples})"
            )
            del self.pending_learning[market_id]

    def _maybe_print_summary(self):
        now = time.time()
        if now - self.last_summary < CONSOLE_SUMMARY_INTERVAL_SEC:
            return
        self.last_summary = now

        def row(timeframe):
            parts = []
            for asset in ASSETS:
                s = self.last_scores.get((asset, timeframe))
                if not s:
                    continue
                p_up = s.get("p_up")
                decision = s.get("decision") or "HOLD"
                p_str = f"{p_up:.2f}" if p_up is not None else "n/a"
                parts.append(f"{asset} P={p_str}→{decision}")
            return " | ".join(parts)

        open_5m = self.risk_manager.open_positions.get("5m", 0)
        open_15m = self.risk_manager.open_positions.get("15m", 0)
        total = self.day_wins + self.day_losses
        win_rate = (self.day_wins / total * 100) if total else 0.0
        news_sent = self.news.sentiment_1h()
        tp_remaining = self.take_profit.take_profit_remaining(self.risk_manager.daily_pnl)
        whale_buys = sum(self.whales[a].whale_buy_count() for a in ASSETS)

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        print(
            f"[{ts} UTC]\n"
            f"5M:  {row('5m')}\n"
            f"15M: {row('15m')}\n"
            f"Regime: {self.current_regime} | Temp: {self.adaptive_state.temperature} | "
            f"News: neutral({news_sent:+.1f})\n"
            f"Open: 5M={open_5m} 15M={open_15m} | Today: {self.risk_manager.daily_pnl:+.2f} "
            f"({self.day_wins}W/{self.day_losses}L {win_rate:.1f}%)\n"
            f"Streak: {'+' if self.risk_manager.loss_streak == 0 else '-'}{self.risk_manager.loss_streak} | "
            f"Take-profit: ${tp_remaining:.2f} to go | Whales: {whale_buys} buys",
            flush=True,
        )

    def _export_status(self):
        data = {
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scores": {
                f"{asset}-{timeframe}": {
                    "p_up": s.get("p_up"),
                    "decision": s.get("decision"),
                    "yes_price": s["snapshot"].get("yes_price"),
                    "no_price": s["snapshot"].get("no_price"),
                    "spot_price": s["snapshot"].get("spot_price"),
                    "seconds_remaining": s["snapshot"].get("seconds_remaining"),
                    "book_depth_yes": s["snapshot"].get("book_depth_yes"),
                    "book_depth_no": s["snapshot"].get("book_depth_no"),
                    "bid_ask_spread": s["snapshot"].get("bid_ask_spread"),
                }
                for (asset, timeframe), s in self.last_scores.items()
            },
            "open_positions": [
                {**self.executors[(p["asset"], p["timeframe"])].open_positions[p["trade_id"]], "market_id": mid}
                for mid, p in self.pending.items()
                if p["trade_id"] in self.executors[(p["asset"], p["timeframe"])].open_positions
            ],
            "day_pnl": self.risk_manager.daily_pnl,
            "day_wins": self.day_wins,
            "day_losses": self.day_losses,
            "loss_streak": self.risk_manager.loss_streak,
            "paused": time.time() < self.risk_manager.paused_until,
            "regime": self.current_regime,
            "temperature": self.adaptive_state.temperature,
            "news_sentiment_1h": self.news.sentiment_1h(),
            "fear_greed": self.fear_greed.normalized(),
            "take_profit_remaining": self.take_profit.take_profit_remaining(self.risk_manager.daily_pnl),
            "model_examples": {
                f"{asset}-{timeframe}": self.models[(asset, timeframe)].n_examples
                for timeframe in TIMEFRAMES for asset in ASSETS
            },
        }
        try:
            os.makedirs(os.path.dirname(BOT_STATUS_FILE), exist_ok=True)
            tmp = BOT_STATUS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, default=str)
            os.replace(tmp, BOT_STATUS_FILE)
        except Exception as e:
            logger.error(f"status export error: {e}")
        self.adaptive_state.export()


if __name__ == "__main__":
    Bot().run_forever()
