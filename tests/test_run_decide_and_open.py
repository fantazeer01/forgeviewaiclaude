import pytest

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


def test_normal_band_signal_uses_full_kelly_size_uncapped(engine, warmed_model, mocker):
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
    assert trade.size_usd == warmed_model.kelly_size(0.95)  # $25, no cap in the proven band


def test_very_extreme_no_trade_is_capped_at_5_despite_high_confidence(engine, warmed_model, mocker):
    # The exact incident this was added for: a BTC NO trade at
    # yes_price=0.985 (very extreme) sized the full kelly $25 off a high
    # confidence signal. It must now be capped to $5 regardless.
    mocker.patch.object(warmed_model, "predict_proba_one", return_value=0.05)  # P(NO)=0.95, clears threshold
    combined_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="NO",
        yes_price=0.985, no_price=0.015, confidence=0.95, reason="extreme reversion: test",
        is_extreme_reversion=True,
    )
    trade = _decide_and_open(
        engine, warmed_model, make_market(yes_price=0.985, no_price=0.015), combined_signal, {}, TelegramReporter(),
    )
    assert trade is not None
    assert trade.direction == "NO"
    assert warmed_model.kelly_size(0.95) == pytest.approx(25.0)  # would have been $25 uncapped
    assert trade.size_usd == pytest.approx(5.0)


def test_moderately_extreme_yes_trade_is_capped_at_10_despite_high_confidence(engine, warmed_model, mocker):
    mocker.patch.object(warmed_model, "predict_proba_one", return_value=0.95)  # P(YES)=0.95
    combined_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="YES",
        yes_price=0.15, no_price=0.85, confidence=0.95, reason="extreme reversion: test",
        is_extreme_reversion=True,
    )
    trade = _decide_and_open(
        engine, warmed_model, make_market(yes_price=0.15, no_price=0.85), combined_signal, {}, TelegramReporter(),
    )
    assert trade is not None
    assert trade.size_usd == pytest.approx(10.0)


def test_cap_never_increases_size_above_what_kelly_would_give(engine, warmed_model, mocker):
    # At low confidence, kelly_size() itself may already be below the price
    # cap -- the cap must never bump size UP to meet it.
    mocker.patch.object(warmed_model, "predict_proba_one", return_value=0.1)
    combined_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="NO",
        yes_price=0.985, no_price=0.015, confidence=0.61, reason="extreme reversion: test",
        is_extreme_reversion=True,
    )
    trade = _decide_and_open(
        engine, warmed_model, make_market(yes_price=0.985, no_price=0.015), combined_signal, {}, TelegramReporter(),
    )
    assert trade is not None
    assert warmed_model.kelly_size(0.61) == pytest.approx(5.0)  # already at/below the $5 cap
    assert trade.size_usd == pytest.approx(5.0)
