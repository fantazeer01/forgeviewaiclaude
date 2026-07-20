"""Layer 5 (hands): paper-mode order execution. Opens/closes positions,
logs to data/trades/paper_trades_v4.jsonl with the full v4 record shape
(including a top-10-feature snapshot for later analysis), and feeds
resolved outcomes back into the online model for training."""

import datetime
import json
import logging
import os
import uuid

from config.settings import PAPER_MODE, PAPER_TRADES_LOG, TAKER_FEE_RATE

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, model, risk_manager, paper_mode: bool = PAPER_MODE, trade_history=None,
                 pattern_memory=None):
        self.paper_mode = paper_mode
        self.model = model
        self.risk_manager = risk_manager
        self.trade_history = trade_history
        self.pattern_memory = pattern_memory
        self.open_positions = {}

    def open_position(self, asset: str, timeframe: str, side: str, entry_price: float, size_usd: float,
                       features: dict, market_id: str, model_prob: float = None, regime: str = None,
                       conditions: dict = None) -> str:
        if not self.paper_mode:
            raise NotImplementedError("live trading is not implemented -- PAPER_MODE must stay True")
        trade_id = str(uuid.uuid4())
        self.open_positions[trade_id] = {
            "trade_id": trade_id,
            "asset": asset,
            "timeframe": timeframe,
            "side": side,
            "entry_price": entry_price,
            "size_usd": size_usd,
            "features": features,
            "market_id": market_id,
            "model_prob": model_prob,
            "regime": regime,
            "conditions": conditions or {},
            "opened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.risk_manager.register_open(timeframe)
        return trade_id

    def close_position(self, trade_id: str, outcome_up: bool) -> float:
        position = self.open_positions.pop(trade_id, None)
        if position is None:
            return 0.0
        won = (position["side"] == "YES" and outcome_up) or (position["side"] == "NO" and not outcome_up)
        pnl = self._settle_pnl(position, won)
        closed_at = datetime.datetime.now(datetime.timezone.utc)
        self._log_trade(position, outcome_up, won, pnl, closed_at)
        self.risk_manager.register_close(position["timeframe"], pnl)
        self.model.learn(position["features"], outcome_up)
        if self.trade_history is not None:
            self.trade_history.record_close(closed_at, won)
        if self.pattern_memory is not None:
            self.pattern_memory.record(position.get("conditions", {}), won, pnl, closed_at)
        return pnl

    def _settle_pnl(self, position: dict, won: bool) -> float:
        """shares = size_usd / entry_price
        fee = shares * TAKER_FEE_RATE * entry_price * (1 - entry_price)
        pnl_win  = shares * (1 - entry_price) - fee
        pnl_loss = -size_usd - fee
        TAKER_FEE_RATE is 0.0 for now -- the formula is wired in for when
        it isn't."""
        entry_price = position["entry_price"]
        size = position["size_usd"]
        if not entry_price or entry_price <= 0:
            return 0.0
        shares = size / entry_price
        fee = shares * TAKER_FEE_RATE * entry_price * (1 - entry_price)
        if won:
            return round(shares * (1 - entry_price) - fee, 2)
        return round(-size - fee, 2)

    def _features_snapshot(self, features: dict) -> dict:
        top_names = self.model.top_feature_names(10)
        return {name: features.get(name) for name in top_names}

    def _log_trade(self, position: dict, outcome_up: bool, won: bool, pnl: float, closed_at: datetime.datetime):
        record = {
            "trade_id": position["trade_id"],
            "asset": position["asset"],
            "timeframe": position["timeframe"],
            "side": position["side"],
            "entry_price": position["entry_price"],
            "size_usd": position["size_usd"],
            "market_id": position["market_id"],
            "opened_at": position["opened_at"],
            "closed_at": closed_at.isoformat(),
            "outcome_up": outcome_up,
            "won": won,
            "pnl_usd": pnl,
            "model_prob": position.get("model_prob"),
            "regime": position.get("regime"),
            "features_snapshot": self._features_snapshot(position["features"]),
        }
        try:
            os.makedirs(os.path.dirname(PAPER_TRADES_LOG), exist_ok=True)
            with open(PAPER_TRADES_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Executor trade log error: {e}")
