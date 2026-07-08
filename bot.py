"""Main trading loop: gather context -> build features -> ensemble decision
-> risk-check -> paper trade -> settle on window close -> train models."""

import datetime
import json
import logging
import os
import time

from config.settings import (
    ASSETS, CONTEXT_POLL_INTERVAL_SEC, CONSOLE_SUMMARY_INTERVAL_SEC, BOT_STATUS_FILE,
    momentum_weights_path, volume_weights_path, WARMUP_TRADE_SIZE_USD,
)
from core.market_context import MarketContext
from core.feature_engine import build_features
from core.ensemble import Ensemble
from core.risk_manager import RiskManager
from core.executor import Executor
from models.momentum_model import MomentumModel
from models.volume_model import VolumeModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class Bot:
    def __init__(self):
        self.context = MarketContext()
        # Each asset gets its own momentum/volume model (and therefore its
        # own ensemble and executor) -- a resolved BTC trade only ever
        # trains BTC's weights, never SOL's or ETH's.
        self.momentum_models = {a: MomentumModel(weights_file=momentum_weights_path(a)) for a in ASSETS}
        self.volume_models = {a: VolumeModel(weights_file=volume_weights_path(a)) for a in ASSETS}
        self.ensembles = {a: Ensemble(self.momentum_models[a], self.volume_models[a]) for a in ASSETS}
        self.risk_manager = RiskManager()  # shared: bankroll/exposure limits are portfolio-wide
        self.executors = {
            a: Executor(self.momentum_models[a], self.volume_models[a], self.risk_manager) for a in ASSETS
        }
        self.pending = {}  # market_id -> {position_id, asset, seconds_remaining, opened_at}
        self.day_wins = 0
        self.day_losses = 0
        self.day_pnl = 0.0
        self.last_summary = 0.0
        self.last_scores = {}

    def start(self):
        self.context.start()

    def run_forever(self):
        self.start()
        try:
            while True:
                self.tick()
                time.sleep(CONTEXT_POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self.context.stop()

    def tick(self):
        for asset in ASSETS:
            snapshot = self.context.snapshot(asset)
            features = build_features(snapshot)
            result = self.ensembles[asset].decide(features, snapshot.get("fear_greed"), snapshot.get("hour_utc"))
            self.last_scores[asset] = {**result, "snapshot": snapshot}
            self._maybe_trade(asset, snapshot, features, result)
        self._check_resolutions()
        self._maybe_print_summary()
        self._export_status()

    def _maybe_trade(self, asset, snapshot, features, result):
        market_id = snapshot.get("market_id")
        decision = result.get("decision")
        if decision is None or market_id is None or market_id in self.pending:
            return
        ok, reason = self.risk_manager.can_open_trade()
        if not ok:
            logger.info(f"SKIP [{asset}] risk blocked: {reason}")
            return
        yes_price = features["yes_price"]
        entry_price = yes_price if decision == "YES" else 1 - yes_price
        mode = result.get("mode", "live")
        if mode == "warmup":
            size = WARMUP_TRADE_SIZE_USD
        else:
            win_probability = result["final_score"] if decision == "YES" else 1 - result["final_score"]
            size = self.risk_manager.position_size(win_probability, entry_price)
        if size <= 0:
            return
        position_id = self.executors[asset].open_position(asset, decision, entry_price, size, features, market_id)
        self.pending[market_id] = {
            "position_id": position_id,
            "asset": asset,
            "seconds_remaining": snapshot.get("seconds_remaining"),
            "opened_at": time.time(),
        }
        score_str = f" score={result['final_score']:.3f}" if result.get("final_score") is not None else ""
        logger.info(
            f"OPEN [{asset}] {decision} mode={mode} entry={entry_price:.3f} size=${size:.2f}{score_str}"
        )

    def _check_resolutions(self):
        for market_id, info in list(self.pending.items()):
            elapsed = time.time() - info["opened_at"]
            if elapsed < (info.get("seconds_remaining") or 300):
                continue
            outcome = self.context.get_resolution(market_id)
            if outcome is None:
                continue
            outcome_up = outcome == "UP"
            pnl = self.executors[info["asset"]].close_position(info["position_id"], outcome_up)
            self.day_pnl += pnl
            if pnl > 0:
                self.day_wins += 1
            else:
                self.day_losses += 1
            logger.info(f"CLOSE [{info['asset']}] outcome={outcome} pnl=${pnl:+.2f}")
            del self.pending[market_id]

    def _maybe_print_summary(self):
        now = time.time()
        if now - self.last_summary < CONSOLE_SUMMARY_INTERVAL_SEC:
            return
        self.last_summary = now
        parts = []
        for asset in ASSETS:
            s = self.last_scores.get(asset)
            if not s:
                continue
            snap = s["snapshot"]
            spot = snap.get("spot_price")
            yes = snap.get("yes_price")
            score = s.get("final_score")
            spot_str = f"{spot:.0f}" if spot is not None else "n/a"
            yes_str = f"{yes:.2f}" if yes is not None else "n/a"
            score_str = f"{score:.2f}" if score is not None else "n/a"
            parts.append(f"{asset}: spot={spot_str} yes={yes_str} score={score_str}")
        n_examples_total = sum(self._examples(asset) for asset in ASSETS)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        print(
            f"[{ts}] " + " | ".join(parts) +
            f"\nOpen: {len(self.pending)} | Today: {self.day_pnl:+.2f} ({self.day_wins}W/{self.day_losses}L)"
            f" | Model: {n_examples_total} examples",
            flush=True,
        )

    def _examples(self, asset: str) -> int:
        return min(self.momentum_models[asset].n_examples, self.volume_models[asset].n_examples)

    def _export_status(self):
        data = {
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scores": {
                asset: {
                    "final_score": s.get("final_score"),
                    "decision": s.get("decision"),
                    "yes_price": s["snapshot"].get("yes_price"),
                    "spot_price": s["snapshot"].get("spot_price"),
                }
                for asset, s in self.last_scores.items()
            },
            "open_positions": [
                {**self.executors[p["asset"]].open_positions[p["position_id"]], "market_id": mid}
                for mid, p in self.pending.items()
                if p["position_id"] in self.executors[p["asset"]].open_positions
            ],
            "day_pnl": self.day_pnl,
            "day_wins": self.day_wins,
            "day_losses": self.day_losses,
            "model_examples": {asset: self._examples(asset) for asset in ASSETS},
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
