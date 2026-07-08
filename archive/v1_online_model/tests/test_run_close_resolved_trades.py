import pytest

from core.dedup_guard import DedupGuard
from core.online_model import OnlineQuantModel
from core.paper_trading_engine import PaperTradingEngine
from core.pnl_tracker import PnLTracker
from core.repricing_detector import RepricingSignal
from core.state_manager import StateManager
from reporting.stats_reporter import StatsReporter
from reporting.telegram_reporter import TelegramReporter
from run import _close_resolved_trades


class FakeFetcher:
    """A market that resolves YES as soon as asked."""
    def __init__(self, winning_outcome="YES"):
        self.winning_outcome = winning_outcome

    def get_market_resolution(self, market_id):
        return {"closed": True}

    def resolve_outcome(self, resolution):
        return self.winning_outcome


@pytest.fixture
def wiring(tmp_path, monkeypatch):
    trades_log = tmp_path / "trades.jsonl"
    monkeypatch.setattr("core.paper_trading_engine.TRADES_LOG", str(trades_log))
    monkeypatch.setattr("core.pnl_tracker.TRADES_LOG", str(trades_log))
    # _close_resolved_trades() also calls _export_execution_cycle("settle", ...)
    # -- without this, tests silently overwrite the real data/execution_cycle.json.
    monkeypatch.setattr("run.EXECUTION_CYCLE_FILE", str(tmp_path / "execution_cycle.json"))
    state = StateManager(state_file=str(tmp_path / "state.json"))
    dedup = DedupGuard(state_file=str(tmp_path / "dedup.json"))
    engine = PaperTradingEngine(state, dedup)
    tracker = PnLTracker()
    stats_rep = StatsReporter(tracker, state)
    online_model = OnlineQuantModel(state_path=str(tmp_path / "online_model.pkl"), warmup_trades=5)
    tg = TelegramReporter()
    return engine, tracker, stats_rep, online_model, tg


def open_trade(engine, direction):
    signal = RepricingSignal(
        asset="BTC", market_id="m1", direction=direction,
        yes_price=0.1, no_price=0.9, confidence=0.7, reason="test",
    )
    return engine.open_trade(signal, source="online_model", size_usd=10.0)


def test_no_direction_win_trains_model_with_yes_lost(wiring, mocker):
    # A NO-direction bet that WINS means the market resolved NO, i.e. YES
    # lost -- the online model's training label must reflect "YES lost"
    # (outcome=0), not "our trade won" (which would wrongly be 1).
    engine, tracker, stats_rep, online_model, tg = wiring
    trade = open_trade(engine, direction="NO")
    online_model.record_features(trade.market_id, {"yes_price": 0.1})
    resolve_spy = mocker.spy(online_model, "resolve")

    fetcher = FakeFetcher(winning_outcome="NO")
    _close_resolved_trades(engine, fetcher, tg, tracker, stats_rep, online_model)

    resolve_spy.assert_called_once_with(trade.market_id, 0)


def test_yes_direction_win_trains_model_with_yes_won(wiring, mocker):
    engine, tracker, stats_rep, online_model, tg = wiring
    trade = open_trade(engine, direction="YES")
    online_model.record_features(trade.market_id, {"yes_price": 0.1})
    resolve_spy = mocker.spy(online_model, "resolve")

    fetcher = FakeFetcher(winning_outcome="YES")
    _close_resolved_trades(engine, fetcher, tg, tracker, stats_rep, online_model)

    resolve_spy.assert_called_once_with(trade.market_id, 1)


def test_no_direction_loss_trains_model_with_yes_won(wiring, mocker):
    # A NO-direction bet that LOSES means the market resolved YES -- the
    # label must be 1 (YES won), even though our own trade lost.
    engine, tracker, stats_rep, online_model, tg = wiring
    trade = open_trade(engine, direction="NO")
    online_model.record_features(trade.market_id, {"yes_price": 0.9})
    resolve_spy = mocker.spy(online_model, "resolve")

    fetcher = FakeFetcher(winning_outcome="YES")
    _close_resolved_trades(engine, fetcher, tg, tracker, stats_rep, online_model)

    resolve_spy.assert_called_once_with(trade.market_id, 1)
