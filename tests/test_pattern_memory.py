from layer6_memory.pattern_memory import PatternMemory, price_bucket, hour_bucket
from config.settings import PATTERN_MIN_TRADES_FOR_SIGNAL, PATTERN_BREAKEVEN_AVG_PNL


def _memory(tmp_path):
    return PatternMemory(path=str(tmp_path / "patterns.json"))


# 25. Pattern memory correctly computes win rate per condition set.
def test_win_rate_computed_per_conditions(tmp_path):
    memory = _memory(tmp_path)
    conditions_a = {"regime": "RANGE", "hour_bucket": "06-12", "price_bucket": "mid", "asset": "BTC"}
    conditions_b = {"regime": "TRENDING_UP", "hour_bucket": "06-12", "price_bucket": "mid", "asset": "BTC"}

    memory.record(conditions_a, won=True, pnl=2.0)
    memory.record(conditions_a, won=True, pnl=3.0)
    memory.record(conditions_a, won=False, pnl=-1.0)
    memory.record(conditions_b, won=False, pnl=-5.0)

    perf_a = memory.get_historical_performance(conditions_a)
    assert perf_a["n_trades"] == 3
    assert perf_a["win_rate"] == 2 / 3
    assert perf_a["avg_pnl"] == (2.0 + 3.0 - 1.0) / 3

    perf_b = memory.get_historical_performance(conditions_b)
    assert perf_b["n_trades"] == 1
    assert perf_b["win_rate"] == 0.0


def test_no_history_returns_zero_trades(tmp_path):
    memory = _memory(tmp_path)
    perf = memory.get_historical_performance({"regime": "RANGE", "hour_bucket": "00-06", "price_bucket": "low", "asset": "SOL"})
    assert perf == {"n_trades": 0, "win_rate": None, "avg_pnl": None}


def test_should_avoid_requires_minimum_sample_and_negative_avg_pnl(tmp_path):
    memory = _memory(tmp_path)
    conditions = {"regime": "RANGE", "hour_bucket": "12-18", "price_bucket": "high", "asset": "ETH"}
    for _ in range(PATTERN_MIN_TRADES_FOR_SIGNAL - 1):
        memory.record(conditions, won=False, pnl=-1.0)
    assert memory.should_avoid(conditions) is False  # not enough samples yet

    memory.record(conditions, won=False, pnl=-1.0)  # now at the minimum
    assert PATTERN_BREAKEVEN_AVG_PNL == 0.0
    assert memory.should_avoid(conditions) is True


def test_persists_across_restart(tmp_path):
    path = str(tmp_path / "patterns.json")
    memory = PatternMemory(path=path)
    conditions = {"regime": "RANGE", "hour_bucket": "18-24", "price_bucket": "mid", "asset": "BTC"}
    memory.record(conditions, won=True, pnl=1.5)

    reloaded = PatternMemory(path=path)
    perf = reloaded.get_historical_performance(conditions)
    assert perf["n_trades"] == 1


def test_price_and_hour_buckets():
    assert price_bucket(0.3) == "low"
    assert price_bucket(0.5) == "mid"
    assert price_bucket(0.7) == "high"
    assert hour_bucket(7) == "06-12"
    assert hour_bucket(0) == "00-06"
