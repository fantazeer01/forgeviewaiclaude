import json
import os
import logging
from config.settings import QUANT_FEATURES_LOG

logger = logging.getLogger(__name__)


class QuantDataset:
    """Append-only log of feature snapshots for future model training.

    Each sample is written twice: once at signal time (stage="signal",
    outcome=None) and once at resolution time (stage="resolution",
    outcome=1 for WIN / 0 for LOSS), both rows sharing the same sample_id.
    Readers dedupe by sample_id keeping the latest row -- the same
    append-and-dedupe pattern used by paper_trades.jsonl -- so a labeled
    example is any sample_id whose latest row has a non-null outcome.
    """

    def __init__(self, log_path: str = QUANT_FEATURES_LOG):
        self.log_path = log_path

    def log_signal(self, sample_id: str, market_id: str, asset: str, features: dict,
                    entry_price: float, direction: str, timestamp: str,
                    model_probability: float = None):
        self._append({
            "sample_id": sample_id,
            "market_id": market_id,
            "asset": asset,
            "direction": direction,
            "entry_price": entry_price,
            "features": features,
            "model_probability": model_probability,
            "stage": "signal",
            "outcome": None,
            "timestamp": timestamp,
        })

    def log_resolution(self, sample_id: str, market_id: str, asset: str, features: dict,
                        entry_price: float, direction: str, outcome: int, timestamp: str,
                        model_probability: float = None):
        self._append({
            "sample_id": sample_id,
            "market_id": market_id,
            "asset": asset,
            "direction": direction,
            "entry_price": entry_price,
            "features": features,
            "model_probability": model_probability,
            "stage": "resolution",
            "outcome": outcome,
            "timestamp": timestamp,
        })

    def load_labeled_examples(self) -> list[dict]:
        """Dedupe by sample_id (keeping the latest row) and return only the
        ones that reached resolution, i.e. outcome is not None."""
        if not os.path.exists(self.log_path):
            return []
        by_id: dict[str, dict] = {}
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    sid = row.get("sample_id")
                    if not sid:
                        continue
                    by_id[sid] = row
        except Exception as e:
            logger.error(f"QuantDataset load error: {e}")
            return []
        return [r for r in by_id.values() if r.get("outcome") is not None]

    def _append(self, row: dict):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.error(f"QuantDataset append error: {e}")
