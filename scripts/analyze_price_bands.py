"""
Reads data/price_history.jsonl (raw, unconditioned yes_price samples for
BTC/ETH, written every poll tick by run.py's _log_price_history()) and
reports what % of observed time each asset's yes_price spends in each band:

    <0.20, 0.20-0.35, 0.35-0.45, 0.45-0.60, >0.60

Unlike data/quant_features.jsonl (only written when the old repricing rule
fires -- a biased sample), this is a straight tick-by-tick sample, so the
percentages here are a real answer to "where does price actually spend its
time," usable for setting SIGNAL_COMBINER_MIN/MAX_YES_PRICE from data.

Usage: python scripts/analyze_price_bands.py [path/to/price_history.jsonl]
"""
import json
import os
import sys
from collections import defaultdict

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "price_history.jsonl")

BANDS = [
    ("<0.20", lambda p: p < 0.20),
    ("0.20-0.35", lambda p: 0.20 <= p < 0.35),
    ("0.35-0.45", lambda p: 0.35 <= p < 0.45),
    ("0.45-0.60", lambda p: 0.45 <= p <= 0.60),
    (">0.60", lambda p: p > 0.60),
]


def load_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def band_breakdown(prices):
    n = len(prices)
    if n == 0:
        return None
    return [(label, sum(1 for p in prices if pred(p)), n) for label, pred in BANDS]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    rows = load_rows(path)
    print(f"Loaded {len(rows)} rows from {path}")
    if not rows:
        print("No data yet.")
        return

    by_asset = defaultdict(list)
    for r in rows:
        if r.get("asset") and r.get("yes_price") is not None:
            by_asset[r["asset"]].append(r["yes_price"])

    ts = sorted(r["timestamp"] for r in rows if r.get("timestamp"))
    if ts:
        print(f"Span: {ts[0]} .. {ts[-1]}")
    print()

    for asset in sorted(by_asset):
        prices = by_asset[asset]
        print(f"{asset} (n={len(prices)}):")
        for label, count, n in band_breakdown(prices):
            print(f"  {label:>10s}: {100*count/n:5.1f}%  (n={count})")
        print()

    all_prices = [p for ps in by_asset.values() for p in ps]
    print(f"COMBINED (n={len(all_prices)}):")
    for label, count, n in band_breakdown(all_prices):
        print(f"  {label:>10s}: {100*count/n:5.1f}%  (n={count})")


if __name__ == "__main__":
    main()
