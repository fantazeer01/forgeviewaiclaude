import json
import os
import logging
import datetime
import uuid
from typing import Optional
from dataclasses import dataclass, asdict
from config.settings import PAPER_TRADE_SIZE_USD, MAX_OPEN_POSITIONS, MAX_DAILY_LOSS_USD, MAX_LOSS_STREAK, TRADES_LOG
from core.dedup_guard import DedupGuard
from core.state_manager import StateManager
from core.repricing_detector import RepricingSignal

logger = logging.getLogger(__name__)

@dataclass
class PaperTrade:
    trade_id: str
    market_id: str
    asset: str
    direction: str
    entry_price: float
    size_usd: float
    size_tokens: float
    signal_confidence: float
    signal_reason: str
    signal_source: str
    open_ts: str
    minutes_at_open: float
    status: str = "open"
    close_ts: Optional[str] = None
    close_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    result: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

class PaperTradingEngine:
    def __init__(self, state: StateManager, dedup: DedupGuard):
        self.state = state
        self.dedup = dedup
        self._open_trades: dict[str, PaperTrade] = {}
        self._restore_open_trades()

    def can_open(self) -> tuple[bool, str]:
        if self.state.is_stopped():
            return False, f"System stopped: {self.state.get('stop_reason')}"
        if len(self._open_trades) >= MAX_OPEN_POSITIONS:
            return False, f"Max open positions reached"
        if self.state.get("daily_loss_usd", 0.0) >= MAX_DAILY_LOSS_USD:
            self.state.stop_system(f"Daily loss limit hit")
            return False, "Daily loss limit hit"
        if self.state.get("loss_streak", 0) >= MAX_LOSS_STREAK:
            self.state.stop_system(f"Loss streak limit hit")
            return False, "Loss streak hit"
        return True, ""

    def open_trade(self, signal: RepricingSignal, source: str = "repricing") -> Optional[PaperTrade]:
        ok, reason = self.can_open()
        if not ok:
            logger.warning(f"Cannot open: {reason}")
            return None
        if self.dedup.is_duplicate(signal.market_id):
            logger.info(f"Duplicate skipped: {signal.market_id}")
            return None
        entry_price = signal.yes_price if signal.direction == "YES" else signal.no_price
        if entry_price <= 0:
            return None
        trade = PaperTrade(
            trade_id=str(uuid.uuid4())[:8],
            market_id=signal.market_id,
            asset=signal.asset,
            direction=signal.direction,
            entry_price=entry_price,
            size_usd=PAPER_TRADE_SIZE_USD,
            size_tokens=PAPER_TRADE_SIZE_USD / entry_price,
            signal_confidence=signal.confidence,
            signal_reason=signal.reason,
            signal_source=source,
            open_ts=datetime.datetime.utcnow().isoformat(),
            minutes_at_open=signal.minutes_remaining,
        )
        self._open_trades[signal.market_id] = trade
        self.dedup.mark_open(signal.market_id)
        self._append_log(trade.to_dict())
        logger.info(f"OPENED: {trade.asset} {trade.direction} @ {entry_price:.3f} id={trade.trade_id}")
        return trade

    def close_trade(self, market_id: str, outcome: str) -> Optional[PaperTrade]:
        trade = self._open_trades.get(market_id)
        if not trade:
            return None
        if outcome == trade.direction:
            close_price, result = 1.0, "WIN"
        else:
            close_price, result = 0.0, "LOSS"
        pnl = trade.size_tokens * (close_price - trade.entry_price)
        trade.close_ts = datetime.datetime.utcnow().isoformat()
        trade.close_price = close_price
        trade.pnl_usd = round(pnl, 4)
        trade.result = result
        trade.status = result.lower()
        wins = self.state.get("wins", 0)
        losses = self.state.get("losses", 0)
        total_pnl = self.state.get("total_pnl_usd", 0.0)
        daily_loss = self.state.get("daily_loss_usd", 0.0)
        loss_streak = self.state.get("loss_streak", 0)
        if result == "WIN":
            wins += 1
            loss_streak = 0
        else:
            losses += 1
            loss_streak += 1
            daily_loss += abs(pnl)
        self.state.update({
            "wins": wins, "losses": losses,
            "total_pnl_usd": round(total_pnl + pnl, 4),
            "daily_loss_usd": round(daily_loss, 4),
            "loss_streak": loss_streak,
            "total_trades": self.state.get("total_trades", 0) + 1,
        })
        if daily_loss >= MAX_DAILY_LOSS_USD:
            self.state.stop_system(f"Daily loss limit hit")
        if loss_streak >= MAX_LOSS_STREAK:
            self.state.stop_system(f"Loss streak limit hit")
        del self._open_trades[market_id]
        self.dedup.mark_closed(market_id)
        self._append_log(trade.to_dict())
        logger.info(f"CLOSED: {trade.asset} {trade.direction} -> {result} PnL=${pnl:+.3f}")
        return trade

    def get_open_trades(self) -> list[PaperTrade]:
        return list(self._open_trades.values())

    def _restore_open_trades(self):
        if not os.path.exists(TRADES_LOG):
            return
        open_by_market: dict[str, dict] = {}
        try:
            with open(TRADES_LOG) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    mid = entry.get("market_id")
                    if not mid:
                        continue
                    if entry.get("status") == "open":
                        open_by_market[mid] = entry
                    else:
                        open_by_market.pop(mid, None)
            for mid, entry in open_by_market.items():
                trade = PaperTrade(**{k: entry.get(k) for k in PaperTrade.__dataclass_fields__})
                self._open_trades[mid] = trade
        except Exception as e:
            logger.error(f"restore error: {e}")

    def _append_log(self, data: dict):
        os.makedirs(os.path.dirname(TRADES_LOG), exist_ok=True)
        with open(TRADES_LOG, "a") as f:
            f.write(json.dumps(data) + "\n")
