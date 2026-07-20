"""Layer 1 (eyes): Alternative.me Fear & Greed Index. Unlike news_feed, this
falls back to the last cached value (disk or in-memory) on any API error,
rather than going neutral -- the index barely moves within a day, so a
slightly stale reading is far more useful than a fabricated neutral one."""

import datetime
import json
import logging
import os
import time
from typing import Optional

import requests

from config.settings import FEAR_GREED_API_URL, FEAR_GREED_POLL_INTERVAL_SEC, FEAR_GREED_CACHE_FILE

logger = logging.getLogger(__name__)


class FearGreedIndex:
    def __init__(self, session: requests.Session = None, cache_path: str = FEAR_GREED_CACHE_FILE):
        self.session = session or requests.Session()
        self.cache_path = cache_path
        self._value = None            # today's index, 0-100
        self._previous_value = None   # ~24h-ago index, 0-100
        self._last_poll = 0.0
        self._load_cache()

    def poll(self, now: float = None):
        now = time.time() if now is None else now
        if now - self._last_poll < FEAR_GREED_POLL_INTERVAL_SEC and self._last_poll > 0:
            return
        self._last_poll = now
        result = self._fetch()
        if result is None:
            return  # keep whatever's already loaded from cache/last successful poll
        self._value, self._previous_value = result
        self._write_cache()

    def _fetch(self) -> Optional[tuple]:
        try:
            resp = self.session.get(FEAR_GREED_API_URL, params={"limit": 2}, timeout=10)
            resp.raise_for_status()
            entries = resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"FearGreedIndex fetch error: {e}")
            return None
        if not entries:
            return None
        try:
            today = float(entries[0]["value"])
            yesterday = float(entries[1]["value"]) if len(entries) > 1 else today
            return today, yesterday
        except (KeyError, ValueError, TypeError):
            return None

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
            self._value = data.get("value")
            self._previous_value = data.get("previous_value")
        except Exception as e:
            logger.warning(f"FearGreedIndex cache load error: {e}")

    def _write_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "value": self._value,
                    "previous_value": self._previous_value,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }, f)
            os.replace(tmp, self.cache_path)
        except Exception as e:
            logger.error(f"FearGreedIndex cache write error: {e}")

    def normalized(self) -> float:
        """0.0 (extreme fear) .. 1.0 (extreme greed); 0.5 neutral fallback
        only when there's truly never been a value, cached or fetched."""
        if self._value is None:
            return 0.5
        return self._value / 100.0

    def change_24h(self) -> float:
        """Raw point change (0-100 scale), not normalized -- 0.0 if either
        side of the comparison is unavailable."""
        if self._value is None or self._previous_value is None:
            return 0.0
        return self._value - self._previous_value
