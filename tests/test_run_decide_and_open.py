import pytest

from config.settings import WARMUP_FLAT_SIZE_USD
from core.dedup_guard import DedupGuard
from core.online_model import OnlineQuantModel
from core.paper_trading_engine import PaperTradingEngine
from core.repricing_detector import RepricingSignal
from core.state_manager import StateManager
from reporting.telegram_reporter import TelegramReporter
from run import _decide_and_open


def make_market(market_id="m1", asset="BTC", yes_price=0.5, no_price=0.5):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price,
        "no_price": no_price, "minutes_remaining": 3.0,
    }


@pytest.fixture
def engine(tmp_path, monkeypatch):
    trades_log = tmp_path / "trades.jsonl"
    monkeypatch.setattr("core.paper_trading_engine.TRADES_LOG", str(trades_log))
    # _decide_and_open() also calls _log_signal() and _export_execution_cycle()
    # on a successful open -- without these, tests silently write real rows
    # into data/signals_log.jsonl and data/execution_cycle.json.
    monkeypatch.setattr("run.SIGNALS_LOG", str(tmp_path / "signals_log.jsonl"))
    monkeypatch.setattr("run.EXECUTION_CYCLE_FILE", str(tmp_path / "execution_cycle.json"))
    state = StateManager(state_file=str(tmp_path / "state.json"))
    dedup = DedupGuard(state_file=str(tmp_path / "dedup.json"))
    return PaperTradingEngine(state, dedup)


@pytest.fixture
def warmed_model(tmp_path):
    model = OnlineQuantModel(state_path=str(tmp_path / "online_model.pkl"), warmup_trades=5)
    model._n_updates = model.warmup_trades
    return model


@pytest.fixture
def warming_up_model(tmp_path):
    # n_updates stays at 0 (default) < warmup_trades -- is_warmed_up() is False
    return OnlineQuantModel(state_path=str(tmp_path / "online_model.pkl"), warmup_trades=5)


def test_normal_band_signal_uses_kelly_size(engine, warmed_model, mocker):
    mocker.patch.object(warmed_model, "predict_proba_one", return_value=0.9)  # P(YES)=0.9
    combined_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="YES",
        yes_price=0.5, no_price=0.5, confidence=0.95, reason="combined(order_book)=0.95",
    )
    trade = _decide_and_open(
        engine, warmed_model, make_market(yes_price=0.5, no_price=0.5), combined_signal, {}, TelegramReporter(),
    )
    assert trade is not None
    assert trade.direction == "YES"
    # kelly_size() is now keyed off the model's own win_probability (0.9) AND
    # this market's real yes_price (0.5), not signal_combiner confidence.
    assert trade.size_usd == warmed_model.kelly_size(0.9, 0.5)


def test_decisive_signal_propagates_from_combiner_into_the_logged_signal(engine, warmed_model, mocker):
    # 2026-07-06 signal quality pass: _decide_and_open builds a NEW
    # RepricingSignal from decide()'s own reason -- decisive_signal must be
    # explicitly carried over from combined_signal or it's silently lost.
    mocker.patch.object(warmed_model, "predict_proba_one", return_value=0.9)
    logged_signal = {}

    def fake_log_signal(signal):
        logged_signal["signal"] = signal

    mocker.patch("run._log_signal", side_effect=fake_log_signal)
    combined_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="YES",
        yes_price=0.5, no_price=0.5, confidence=0.95, reason="combined(momentum)=0.95",
        decisive_signal="momentum",
    )
    trade = _decide_and_open(
        engine, warmed_model, make_market(yes_price=0.5, no_price=0.5), combined_signal, {}, TelegramReporter(),
    )
    assert trade is not None
    assert logged_signal["signal"].decisive_signal == "momentum"


def test_no_direction_trade_sized_off_no_side_probability_and_price(engine, warmed_model, mocker):
    # 2026-07-07 NO-direction resurrection: win_probability from decide() is
    # always raw P(YES) regardless of direction -- a NO trade must size off
    # P(NO)=1-win_probability and NO's own entry price (no_price), not
    # win_probability/yes_price directly, or kelly_size() gets the wrong
    # side's price/probability pair entirely.
    mocker.patch.object(warmed_model, "predict_proba_one", return_value=0.1)  # P(YES)=0.1, P(NO)=0.9
    no_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="NO",
        yes_price=0.85, no_price=0.15, confidence=0.95, reason="mean reversion",
    )
    trade = _decide_and_open(
        engine, warmed_model, make_market(yes_price=0.85, no_price=0.15), no_signal, {}, TelegramReporter(),
    )
    assert trade is not None
    assert trade.direction == "NO"
    assert trade.entry_price == 0.15
    assert trade.size_usd == warmed_model.kelly_size(0.9, 0.15)
    assert trade.size_usd != warmed_model.kelly_size(0.1, 0.85)  # sanity: NOT the YES-side pair


def test_warmup_trade_sizes_flat_regardless_of_combiner_confidence(engine, warming_up_model):
    # 2026-07-07 reversal: warm-up trades must NOT use kelly_size() -- a
    # freshly-reset, unproven model shouldn't be sized off real Kelly purely
    # off combiner confidence during the one period whose only job is
    # safely accumulating training examples.
    strong_combiner_signal = RepricingSignal(
        asset="BTC", market_id="m1", direction="YES",
        yes_price=0.5, no_price=0.5, confidence=0.95, reason="combined(momentum)=0.95",
    )
    trade = _decide_and_open(
        engine, warming_up_model, make_market(yes_price=0.5, no_price=0.5),
        strong_combiner_signal, {}, TelegramReporter(),
    )
    assert trade is not None
    assert trade.size_usd == WARMUP_FLAT_SIZE_USD
