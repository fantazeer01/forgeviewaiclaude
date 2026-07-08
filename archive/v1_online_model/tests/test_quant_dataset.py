import json

from core.quant_dataset import QuantDataset


def make_features():
    return {"yes_price": 0.4, "no_price": 0.6, "price_velocity": 0.001,
            "price_acceleration": None, "order_book_imbalance": 0.2,
            "volume_24h": 1000.0, "time_remaining_pct": 0.6, "spread": 0.02}


def test_log_signal_writes_row_with_null_outcome(tmp_path):
    dataset = QuantDataset(log_path=str(tmp_path / "quant.jsonl"))
    dataset.log_signal("s1", "m1", "BTC", make_features(), 0.40, "YES", "2026-01-01T00:00:00+00:00")
    with open(dataset.log_path) as f:
        row = json.loads(f.readline())
    assert row["sample_id"] == "s1"
    assert row["stage"] == "signal"
    assert row["outcome"] is None
    assert row["entry_price"] == 0.40
    assert row["direction"] == "YES"
    assert row["features"] == make_features()


def test_log_resolution_writes_row_with_outcome(tmp_path):
    dataset = QuantDataset(log_path=str(tmp_path / "quant.jsonl"))
    dataset.log_resolution("s1", "m1", "BTC", make_features(), 0.40, "YES", 1, "2026-01-01T00:05:00+00:00")
    with open(dataset.log_path) as f:
        row = json.loads(f.readline())
    assert row["stage"] == "resolution"
    assert row["outcome"] == 1


def test_load_labeled_examples_empty_when_file_missing(tmp_path):
    dataset = QuantDataset(log_path=str(tmp_path / "missing.jsonl"))
    assert dataset.load_labeled_examples() == []


def test_load_labeled_examples_excludes_unresolved_signals(tmp_path):
    dataset = QuantDataset(log_path=str(tmp_path / "quant.jsonl"))
    dataset.log_signal("s1", "m1", "BTC", make_features(), 0.40, "YES", "t1")
    dataset.log_signal("s2", "m2", "ETH", make_features(), 0.45, "YES", "t2")
    dataset.log_resolution("s1", "m1", "BTC", make_features(), 0.40, "YES", 1, "t3")
    examples = dataset.load_labeled_examples()
    assert len(examples) == 1
    assert examples[0]["sample_id"] == "s1"
    assert examples[0]["outcome"] == 1


def test_load_labeled_examples_dedupes_by_sample_id_keeping_latest(tmp_path):
    dataset = QuantDataset(log_path=str(tmp_path / "quant.jsonl"))
    dataset.log_signal("s1", "m1", "BTC", make_features(), 0.40, "YES", "t1")
    dataset.log_resolution("s1", "m1", "BTC", make_features(), 0.40, "YES", 0, "t2")
    examples = dataset.load_labeled_examples()
    assert len(examples) == 1
    assert examples[0]["outcome"] == 0
    assert examples[0]["stage"] == "resolution"


def test_load_labeled_examples_ignores_blank_lines(tmp_path):
    log_path = tmp_path / "quant.jsonl"
    dataset = QuantDataset(log_path=str(log_path))
    dataset.log_resolution("s1", "m1", "BTC", make_features(), 0.40, "YES", 1, "t1")
    with open(log_path, "a") as f:
        f.write("\n\n")
    assert len(dataset.load_labeled_examples()) == 1


def test_load_labeled_examples_counts_multiple_labeled_samples(tmp_path):
    dataset = QuantDataset(log_path=str(tmp_path / "quant.jsonl"))
    for i in range(5):
        dataset.log_signal(f"s{i}", f"m{i}", "BTC", make_features(), 0.40, "YES", "t")
        dataset.log_resolution(f"s{i}", f"m{i}", "BTC", make_features(), 0.40, "YES", i % 2, "t")
    assert len(dataset.load_labeled_examples()) == 5
