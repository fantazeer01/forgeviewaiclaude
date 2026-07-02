import datetime

from core.quant_signal import QuantSignalGenerator
from core.state_manager import StateManager


class FakeDataset:
    def __init__(self):
        self.signals = []
        self.resolutions = []

    def log_signal(self, **kwargs):
        self.signals.append(kwargs)

    def log_resolution(self, **kwargs):
        self.resolutions.append(kwargs)


class FakeModel:
    def __init__(self, probability):
        self.probability = probability
        self.calls = 0

    def predict_proba_one(self, features):
        self.calls += 1
        return self.probability


class FakeFetcher:
    def __init__(self, order_book_top=None, resolution=None, resolved_direction=None):
        self.order_book_top = order_book_top
        self.resolution = resolution
        self.resolved_direction = resolved_direction
        self.resolution_calls = []

    def get_order_book_top(self, token_id):
        return self.order_book_top

    def get_market_resolution(self, market_id):
        self.resolution_calls.append(market_id)
        return self.resolution

    def resolve_outcome(self, resolution):
        return self.resolved_direction


def make_market(market_id="m1", asset="BTC", yes_price=0.5, no_price=0.5,
                 minutes_remaining=3.0, up_token_id="up-tok", volume_24h=100.0):
    return {
        "market_id": market_id, "asset": asset, "yes_price": yes_price,
        "no_price": no_price, "minutes_remaining": minutes_remaining,
        "up_token_id": up_token_id, "volume_24h": volume_24h,
    }


def test_process_market_returns_none_when_no_model_loaded(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher()
    qsg = QuantSignalGenerator(state, fetcher, dataset=FakeDataset(), model=None)

    signal = qsg.process_market(make_market())

    assert signal is None


def test_process_market_still_updates_feature_history_without_model(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher()
    qsg = QuantSignalGenerator(state, fetcher, dataset=FakeDataset(), model=None)

    qsg.process_market(make_market())

    assert "m1" in qsg.features._price_history


def test_process_market_fires_signal_when_model_confidence_above_threshold(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_ask_price": 0.51,
                                           "total_bid_depth": 100.0, "total_ask_depth": 50.0})
    model = FakeModel(probability=0.72)
    qsg = QuantSignalGenerator(state, fetcher, dataset=FakeDataset(), model=model)

    signal = qsg.process_market(make_market(yes_price=0.5, no_price=0.5))

    assert signal is not None
    assert signal.direction == "YES"
    assert signal.confidence == 0.72
    assert "0.720" in signal.reason


def test_process_market_returns_none_when_model_confidence_below_threshold(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_ask_price": 0.51,
                                           "total_bid_depth": 100.0, "total_ask_depth": 50.0})
    model = FakeModel(probability=0.40)
    qsg = QuantSignalGenerator(state, fetcher, dataset=FakeDataset(), model=model)

    signal = qsg.process_market(make_market())

    assert signal is None


def test_process_market_respects_confidence_threshold_boundary(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_ask_price": 0.51,
                                           "total_bid_depth": 100.0, "total_ask_depth": 50.0})
    model = FakeModel(probability=0.55)
    qsg = QuantSignalGenerator(state, fetcher, dataset=FakeDataset(), model=model, confidence_threshold=0.55)

    signal = qsg.process_market(make_market())

    assert signal is not None


def test_process_market_returns_none_outside_minutes_remaining_window(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_ask_price": 0.51,
                                           "total_bid_depth": 100.0, "total_ask_depth": 50.0})
    model = FakeModel(probability=0.90)
    qsg = QuantSignalGenerator(state, fetcher, dataset=FakeDataset(), model=model)

    assert qsg.process_market(make_market(minutes_remaining=0.5)) is None
    assert qsg.process_market(make_market(minutes_remaining=4.6)) is None


def test_process_market_respects_cooldown(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    state.set("last_signal_ts", {"BTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_ask_price": 0.51,
                                           "total_bid_depth": 100.0, "total_ask_depth": 50.0})
    model = FakeModel(probability=0.90)
    qsg = QuantSignalGenerator(state, fetcher, dataset=FakeDataset(), model=model)

    signal = qsg.process_market(make_market())

    assert signal is None


def test_process_market_sets_cooldown_after_firing(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_ask_price": 0.51,
                                           "total_bid_depth": 100.0, "total_ask_depth": 50.0})
    model = FakeModel(probability=0.90)
    qsg = QuantSignalGenerator(state, fetcher, dataset=FakeDataset(), model=model)

    qsg.process_market(make_market())

    assert "BTC" in (state.get("last_signal_ts") or {})


def test_process_market_logs_signal_with_features_and_model_probability(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_ask_price": 0.51,
                                           "total_bid_depth": 100.0, "total_ask_depth": 50.0})
    model = FakeModel(probability=0.80)
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(state, fetcher, dataset=dataset, model=model)

    signal = qsg.process_market(make_market(yes_price=0.5, no_price=0.5))

    assert signal is not None
    assert len(dataset.signals) == 1
    logged = dataset.signals[0]
    assert logged["sample_id"] == "m1"
    assert logged["asset"] == "BTC"
    assert logged["direction"] == "YES"
    assert logged["entry_price"] == 0.5
    assert logged["model_probability"] == 0.80
    assert logged["features"]["yes_price"] == 0.5
    assert "m1" in qsg._pending


def test_resolve_pending_logs_win_outcome_and_clears_pending(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(resolution={"closed": True}, resolved_direction="YES")
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(state, fetcher, dataset=dataset, model=None)
    qsg._pending["m1"] = {"asset": "BTC", "features": {"yes_price": 0.4}, "entry_price": 0.4,
                           "direction": "YES", "model_probability": 0.7}

    qsg.resolve_pending()

    assert "m1" not in qsg._pending
    assert len(dataset.resolutions) == 1
    assert dataset.resolutions[0]["outcome"] == 1
    assert dataset.resolutions[0]["model_probability"] == 0.7


def test_resolve_pending_logs_loss_outcome_when_direction_mismatches(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(resolution={"closed": True}, resolved_direction="NO")
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(state, fetcher, dataset=dataset, model=None)
    qsg._pending["m1"] = {"asset": "BTC", "features": {"yes_price": 0.4}, "entry_price": 0.4,
                           "direction": "YES", "model_probability": 0.7}

    qsg.resolve_pending()

    assert dataset.resolutions[0]["outcome"] == 0


def test_resolve_pending_leaves_entry_pending_when_not_yet_resolved(tmp_path):
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(resolution=None)
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(state, fetcher, dataset=dataset, model=None)
    qsg._pending["m1"] = {"asset": "BTC", "features": {}, "entry_price": 0.4,
                           "direction": "YES", "model_probability": None}

    qsg.resolve_pending()

    assert "m1" in qsg._pending
    assert dataset.resolutions == []
