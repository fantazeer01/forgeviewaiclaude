import datetime

from core.quant_signal import QuantSignalGenerator
from core.repricing_detector import RepricingDetector
from core.state_manager import StateManager


class FakeDataset:
    def __init__(self):
        self.signals = []
        self.resolutions = []

    def log_signal(self, **kwargs):
        self.signals.append(kwargs)

    def log_resolution(self, **kwargs):
        self.resolutions.append(kwargs)


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


def test_process_market_updates_feature_history_even_without_signal(tmp_path):
    detector = RepricingDetector()
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher()
    qsg = QuantSignalGenerator(detector, state, fetcher, dataset=FakeDataset())

    signal = qsg.process_market(make_market())

    assert signal is None
    assert "m1" in qsg.features._price_history


def test_process_market_fires_signal_identically_to_repricing_logic(tmp_path):
    detector = RepricingDetector()
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_bid_size": 10.0,
                                           "best_ask_price": 0.51, "best_ask_size": 5.0})
    qsg = QuantSignalGenerator(detector, state, fetcher, dataset=FakeDataset())

    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]

    signal = qsg.process_market(make_market(yes_price=0.5, no_price=0.5))

    assert signal is not None
    assert signal.direction == "YES"
    assert signal.confidence == 0.95


def test_process_market_logs_signal_with_features_when_signal_fires(tmp_path):
    detector = RepricingDetector()
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(order_book_top={"best_bid_price": 0.49, "best_bid_size": 10.0,
                                           "best_ask_price": 0.51, "best_ask_size": 5.0})
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(detector, state, fetcher, dataset=dataset)

    old_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=70)
    detector._price_history["m1"] = [{"ts": old_ts, "yes": 0.9, "no": 0.1}]

    signal = qsg.process_market(make_market(yes_price=0.5, no_price=0.5))

    assert signal is not None
    assert len(dataset.signals) == 1
    logged = dataset.signals[0]
    assert logged["sample_id"] == "m1"
    assert logged["market_id"] == "m1"
    assert logged["asset"] == "BTC"
    assert logged["direction"] == "YES"
    assert logged["entry_price"] == 0.5
    assert logged["features"]["yes_price"] == 0.5
    assert logged["features"]["order_book_imbalance"] is not None
    assert "m1" in qsg._pending


def test_process_market_does_not_log_when_no_signal(tmp_path):
    detector = RepricingDetector()
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher()
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(detector, state, fetcher, dataset=dataset)

    qsg.process_market(make_market())

    assert dataset.signals == []
    assert qsg._pending == {}


def test_resolve_pending_logs_win_outcome_and_clears_pending(tmp_path):
    detector = RepricingDetector()
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(resolution={"closed": True}, resolved_direction="YES")
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(detector, state, fetcher, dataset=dataset)
    qsg._pending["m1"] = {"asset": "BTC", "features": {"yes_price": 0.4}, "entry_price": 0.4, "direction": "YES"}

    qsg.resolve_pending()

    assert "m1" not in qsg._pending
    assert len(dataset.resolutions) == 1
    assert dataset.resolutions[0]["outcome"] == 1
    assert dataset.resolutions[0]["sample_id"] == "m1"


def test_resolve_pending_logs_loss_outcome_when_direction_mismatches(tmp_path):
    detector = RepricingDetector()
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(resolution={"closed": True}, resolved_direction="NO")
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(detector, state, fetcher, dataset=dataset)
    qsg._pending["m1"] = {"asset": "BTC", "features": {"yes_price": 0.4}, "entry_price": 0.4, "direction": "YES"}

    qsg.resolve_pending()

    assert dataset.resolutions[0]["outcome"] == 0


def test_resolve_pending_leaves_entry_pending_when_not_yet_resolved(tmp_path):
    detector = RepricingDetector()
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(resolution=None)
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(detector, state, fetcher, dataset=dataset)
    qsg._pending["m1"] = {"asset": "BTC", "features": {}, "entry_price": 0.4, "direction": "YES"}

    qsg.resolve_pending()

    assert "m1" in qsg._pending
    assert dataset.resolutions == []


def test_resolve_pending_leaves_entry_pending_when_outcome_none(tmp_path):
    detector = RepricingDetector()
    state = StateManager(state_file=str(tmp_path / "state.json"))
    fetcher = FakeFetcher(resolution={"closed": False}, resolved_direction=None)
    dataset = FakeDataset()
    qsg = QuantSignalGenerator(detector, state, fetcher, dataset=dataset)
    qsg._pending["m1"] = {"asset": "BTC", "features": {}, "entry_price": 0.4, "direction": "YES"}

    qsg.resolve_pending()

    assert "m1" in qsg._pending
    assert dataset.resolutions == []
