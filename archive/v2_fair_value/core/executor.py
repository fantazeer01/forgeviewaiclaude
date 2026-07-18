"""Paper-mode order execution: opens/closes positions, logs to
data/trades/paper_trades.jsonl, and feeds resolved outcomes back into the
online models for training."""

import datetime
import json
import logging
import os
import uuid

from config.settings import PAPER_MODE, PAPER_TRADES_LOG

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, momentum_model, volume_model, risk_manager, paper_mode: bool = PAPER_MODE):
        self.paper_mode = paper_mode
        self.momentum_model = momentum_model
        self.volume_model = volume_model
        self.risk_manager = risk_manager
        self.open_positions = {}

    def open_position(self, asset: str, side: str, entry_price: float, size_usd: float,
                       features: dict, market_id: str) -> str:
        if not self.paper_mode:
            raise NotImplementedError("live trading is not implemented -- PAPER_MODE must stay True")
        position_id = str(uuid.uuid4())
        self.open_positions[position_id] = {
            "position_id": position_id,
            "asset": asset,
            "side": side,
            "entry_price": entry_price,
            "size_usd": size_usd,
            "features": features,
            "market_id": market_id,
            "opened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.risk_manager.register_open()
        return position_id

    def close_position(self, position_id: str, outcome_up: bool) -> float:
        position = self.open_positions.pop(position_id, None)
        if position is None:
            return 0.0
        won = (position["side"] == "YES" and outcome_up) or (position["side"] == "NO" and not outcome_up)
        pnl = self._settle_pnl(position, won)
        self._log_trade(position, outcome_up, won, pnl)
        self.risk_manager.register_close(pnl)
        self.momentum_model.learn(position["features"], outcome_up)
        self.volume_model.learn(position["features"], outcome_up)
        return pnl

    def _settle_pnl(self, position: dict, won: bool) -> float:
        contract_price = position["entry_price"]
        size = position["size_usd"]
        if not contract_price or contract_price <= 0:
            return 0.0
        shares = size / contract_price
        return round(shares * (1 - contract_price), 2) if won else round(-size, 2)

    def _log_trade(self, position: dict, outcome_up: bool, won: bool, pnl: float):
        record = {
            **{k: v for k, v in position.items() if k != "features"},
            "outcome_up": outcome_up,
            "won": won,
            "pnl": pnl,
            "closed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            os.makedirs(os.path.dirname(PAPER_TRADES_LOG), exist_ok=True)
            with open(PAPER_TRADES_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Executor trade log error: {e}")
