import json
import logging
import os
from config.settings import STATE_FILE

logger = logging.getLogger(__name__)

class DedupGuard:
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self._open_market_ids: set[str] = set()
        self._load()

    def is_duplicate(self, market_id: str) -> bool:
        return market_id in self._open_market_ids

    def mark_open(self, market_id: str):
        self._open_market_ids.add(market_id)
        self._save()

    def mark_closed(self, market_id: str):
        self._open_market_ids.discard(market_id)
        self._save()

    def _load(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            self._open_market_ids = set(data.get("open_market_ids", []))
        except Exception as e:
            logger.warning(f"DedupGuard load error: {e}")

    def _save(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        try:
            existing = {}
            if os.path.exists(self.state_file):
                with open(self.state_file) as f:
                    existing = json.load(f)
            existing["open_market_ids"] = list(self._open_market_ids)
            with open(self.state_file, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.error(f"DedupGuard save error: {e}")
