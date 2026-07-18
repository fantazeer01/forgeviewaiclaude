"""v3 main trading loop: gather market_feed -> build features -> model P(UP)
-> risk filters -> paper trade -> settle on window close -> train the model
on EVERY resolved window (not just real trades)."""

import datetime
import json
import logging
import os
import time

from config.settings import (
    ASSETS, TIMEFRAMES, CONTEXT_POLL_INTERVAL_SEC, CONSOLE_SUMMARY_INTERVAL_SEC,
    BOT_STATUS_FILE, RISK_STATE_FILE, model_weights_path,
)
from core.market_feed import MarketFeed
from core.feature_engine import build_features, CrossMarketState
from core.model import OnlineModel
from core.risk_manager import RiskManager
from core.executor import Executor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Window-close learning fires once per window at its midpoint (generalizes
# the "150s before close" 5-min rule to any timeframe: 150s = 300s/2).
WINDOW_LEARN_FRACTION = 0.5


class Bot:
    def __init__(self):
        self.market_feed = MarketFeed()
        self.cross_market = CrossMarketState()
        self.risk_manager = RiskManager(state_file=RISK_STATE_FILE)

        self.models = {}
        self.executors = {}
        for asset in ASSETS:
            for timeframe in TIMEFRAMES:
                key = (asset, timeframe)
                model = OnlineModel(
                    weights_file=model_weights_path(asset, timeframe), asset=asset, timeframe=timeframe
                )
                self.models[key] = model
                self.executors[key] = Executor(model, self.risk_manager)

        self.pending = {}          # market_id -> {position_id, asset, timeframe, seconds_remaining, opened_at}
        self.pending_learning = {}  # market_id -> {asset, timeframe, features, captured_at, seconds_remaining}
        self.day_wins = 0
        self.day_losses = 0
        self.last_summary = 0.0
        self.last_scores = {}  # (asset, timeframe) -> {snapshot, features, decision}

    def start(self):
        self.market_feed.start()

    def run_forever(self):
        self.start()
        try:
            while True:
                self.tick()
                time.sleep(CONTEXT_POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self.market_feed.stop()

    def tick(self):
        btc_mom_5m = self.market_feed.binance.get_momentum_bps("BTC", 5)
        for asset in ASSETS:
            if asset != "BTC":
                asset_mom_5m = self.market_feed.binance.get_momentum_bps(asset, 5)
                self.cross_market.update(asset, btc_mom_5m, asset_mom_5m)

        for timeframe, window_sec in TIMEFRAMES.items():
            btc_snapshot = self.market_feed.snapshot("BTC", timeframe)
            self._process(("BTC", timeframe), btc_snapshot, window_sec, None, None)
            for asset in ASSETS:
                if asset == "BTC":
                    continue
                snapshot = self.market_feed.snapshot(asset, timeframe)
                correlation = self.cross_market.correlation(asset)
                self._process((asset, timeframe), snapshot, window_sec, btc_snapshot, correlation)

        self._check_resolutions()
        self._check_window_resolutions()
        self._maybe_print_summary()
        self._export_status()

    def _process(self, key, snapshot, window_sec, btc_snapshot, correlation):
        asset, timeframe = key
        features = build_features(snapshot, window_sec, btc_snapshot, correlation)
        result = self.models[key].decide(features, snapshot.get("seconds_remaining"))
        self.last_scores[key] = {**result, "snapshot": snapshot, "features": features}
        self._maybe_trade(key, snapshot, features, result)
        self._maybe_capture_window(key, snapshot, features, window_sec)

    def _maybe_trade(self, key, snapshot, features, result):
        asset, timeframe = key
        decision = result.get("decision")
        market_id = snapshot.get("market_id")
        if decision not in ("YES", "NO") or market_id is None or market_id in self.pending:
            return
        ok, reason = self.risk_manager.can_open_trade(timeframe)
        if not ok:
            logger.info(f"SKIP [{asset}-{timeframe}] risk blocked: {reason}")
            return
        yes_price = features["yes_price"]
        entry_price = yes_price if decision == "YES" else 1 - yes_price
        p_up = result["p_up"]
        win_probability = p_up if decision == "YES" else 1 - p_up
        size = self.risk_manager.position_size(win_probability, entry_price)
        if size <= 0:
            return
        position_id = self.executors[key].open_position(
            asset, timeframe, decision, entry_price, size, features, market_id
        )
        self.pending[market_id] = {
            "position_id": position_id,
            "asset": asset,
            "timeframe": timeframe,
            "seconds_remaining": snapshot.get("seconds_remaining"),
            "opened_at": time.time(),
        }
        logger.info(
            f"OPEN [{asset}-{timeframe}] {decision} entry={entry_price:.3f} size=${size:.2f} P(UP)={p_up:.3f}"
        )

    def _check_resolutions(self):
        for market_id, info in list(self.pending.items()):
            elapsed = time.time() - info["opened_at"]
            if elapsed < (info.get("seconds_remaining") or 300):
                continue
            outcome = self.market_feed.polymarket.get_resolution(market_id)
            if outcome is None:
                continue
            outcome_up = outcome == "UP"
            key = (info["asset"], info["timeframe"])
            pnl = self.executors[key].close_position(info["position_id"], outcome_up)
            if pnl > 0:
                self.day_wins += 1
            else:
                self.day_losses += 1
            logger.info(f"CLOSE [{info['asset']}-{info['timeframe']}] outcome={outcome} pnl=${pnl:+.2f}")
            del self.pending[market_id]

    def _maybe_capture_window(self, key, snapshot, features, window_sec):
        """Once per window, at its midpoint, record a feature snapshot for
        training -- independent of whether a real trade was also opened."""
        market_id = snapshot.get("market_id")
        seconds_remaining = snapshot.get("seconds_remaining")
        if market_id is None or seconds_remaining is None or market_id in self.pending_learning:
            return
        if seconds_remaining <= window_sec * WINDOW_LEARN_FRACTION:
            self.pending_learning[market_id] = {
                "asset": key[0],
                "timeframe": key[1],
                "features": features,
                "captured_at": time.time(),
                "seconds_remaining": seconds_remaining,
            }

    def _check_window_resolutions(self):
        for market_id, info in list(self.pending_learning.items()):
            elapsed = time.time() - info["captured_at"]
            if elapsed < info["seconds_remaining"]:
                continue
            outcome = self.market_feed.polymarket.get_resolution(market_id)
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
                yes = s["snapshot"].get("yes_price")
                p_up = s.get("p_up")
                decision = s.get("decision") or "HOLD"
                yes_str = f"{yes:.2f}" if yes is not None else "n/a"
                p_str = f"{p_up:.2f}" if p_up is not None else "n/a"
                parts.append(f"{asset} yes={yes_str} P={p_str}→{decision}")
            return " | ".join(parts)

        open_5m = self.risk_manager.open_positions.get("5m", 0)
        open_15m = self.risk_manager.open_positions.get("15m", 0)
        total = self.day_wins + self.day_losses
        win_rate = (self.day_wins / total * 100) if total else 0.0
        paused = time.time() < self.risk_manager.paused_until

        examples_str = " ".join(
            f"{asset}-{timeframe}={self.models[(asset, timeframe)].n_examples}"
            for timeframe in TIMEFRAMES for asset in ASSETS
        )

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        print(
            f"[{ts} UTC]\n"
            f"5MIN:  {row('5m')}\n"
            f"15MIN: {row('15m')}\n"
            f"Open: 5M={open_5m} 15M={open_15m} | Today: {self.risk_manager.daily_pnl:+.2f} "
            f"({self.day_wins}W/{self.day_losses}L {win_rate:.1f}%) | "
            f"Streak: {self.risk_manager.loss_streak} | Pause: {'Yes' if paused else 'No'}\n"
            f"Examples: {examples_str}",
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
                    "spot_price": s["snapshot"].get("spot_price"),
                    "seconds_remaining": s["snapshot"].get("seconds_remaining"),
                }
                for (asset, timeframe), s in self.last_scores.items()
            },
            "open_positions": [
                {**self.executors[(p["asset"], p["timeframe"])].open_positions[p["position_id"]], "market_id": mid}
                for mid, p in self.pending.items()
                if p["position_id"] in self.executors[(p["asset"], p["timeframe"])].open_positions
            ],
            "day_pnl": self.risk_manager.daily_pnl,
            "day_wins": self.day_wins,
            "day_losses": self.day_losses,
            "loss_streak": self.risk_manager.loss_streak,
            "paused": time.time() < self.risk_manager.paused_until,
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


if __name__ == "__main__":
    Bot().run_forever()
