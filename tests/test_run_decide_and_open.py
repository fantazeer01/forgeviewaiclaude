import pytest

from config.settings import EXTREME_REVERSION_SIZE_USD
from core.dedup_guard import DedupGuard
from core.online_model import OnlineQuantModel
from core.paper_trading_engine import PaperTradingEngine
from core.repricing_detector import RepricingSignal
from core.state_manager import StateManager
from reporting.telegram_reporter import TelegramReporter
from run import _decide_and_open


def make_market(market_id="m1", asset="BTC", yes_price=0.9, no_price=0.1):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price,
        "no_price": no_price, "minutes_remaining": 3.0,
    }


@pytest.fixture
def engine(tmp_path, monkeypatch):
    trades_log = tmp_path / "trades.jsonl"
    monkeypatch.setattr("core.paper_trading_engine.TRADES_LOG", str(trades_log))
    state = StateManager(state_file=str(tmp_path / "state.json"))
    dedup = DedupGuard(state_file=str(tmp_path / "dedup.json"))
    return PaperTradingEngine(state, dedup)


@pytest.fixture
def warmed_model(tmp_path):
    model = OnlineQuantModel(state_path=str(tmp_path / "online_model.pkl"), warmup_trades=5)
    model._n_updates = model.warmup_trades
    return model


def test_extreme_reversion_signal_uses_flat_conservative_size(engine, warmed_model, mocker):
    mocker.patch.object(warmed_model, "predict_proba_one", return_value=0.1)  # P(NO)=0.9, clears threshold
    combined_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="NO",
        yes_price=0.9, no_price=0.1, confidence=0.95, reason="extreme reversion: test",
        is_extreme_reversion=True,
    )
    trade = _decide_and_open(engine, warmed_model, make_market(), combined_signal, {}, TelegramReporter())
    assert trade is not None
    assert trade.direction == "NO"
    assert trade.size_usd == EXTREME_REVERSION_SIZE_USD


def test_normal_band_signal_uses_kelly_size_not_flat_size(engine, warmed_model, mocker):
    mocker.patch.object(warmed_model, "predict_proba_one", return_value=0.9)  # P(YES)=0.9
    combined_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="YES",
        yes_price=0.5, no_price=0.5, confidence=0.95, reason="combined(order_book)=0.95",
        is_extreme_reversion=False,
    )
    trade = _decide_and_open(
        engine, warmed_model, make_market(yes_price=0.5, no_price=0.5), combined_signal, {}, TelegramReporter(),
    )
    assert trade is not None
    assert trade.direction == "YES"
    assert trade.size_usd == warmed_model.kelly_size(0.95)
    assert trade.size_usd != EXTREME_REVERSION_SIZE_USD
