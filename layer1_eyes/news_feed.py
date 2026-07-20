"""Layer 1 (eyes): CryptoPanic news sentiment. Requires an auth token
(CRYPTOPANIC_API_KEY env var) -- with no token, or on any API error, this
always reports a neutral 0.0 sentiment rather than trusting stale data."""

import datetime
import json
import logging
import os
import statistics
import time
from typing import Optional

import requests

from config.settings import (
    CRYPTOPANIC_API_BASE, CRYPTOPANIC_API_KEY, NEWS_POLL_INTERVAL_SEC,
    NEWS_SENTIMENT_WINDOW_HOURS, NEWS_CACHE_FILE, NEWS_MAJOR_SOURCES,
)

logger = logging.getLogger(__name__)


class NewsFeed:
    def __init__(self, session: requests.Session = None, api_key: str = CRYPTOPANIC_API_KEY,
                 cache_path: str = NEWS_CACHE_FILE):
        self.session = session or requests.Session()
        self.api_key = api_key
        self.cache_path = cache_path
        self._posts = []  # [{title, source, currencies, sentiment, published_at}]
        self._last_poll = 0.0
        self._available = False

    def poll(self, now: float = None):
        now = time.time() if now is None else now
        if now - self._last_poll < NEWS_POLL_INTERVAL_SEC and self._last_poll > 0:
            return
        self._last_poll = now
        posts = self._fetch()
        if posts is None:
            self._available = False
            self._posts = []
            return
        self._available = True
        self._posts = posts
        self._write_cache(posts)

    def _fetch(self) -> Optional[list]:
        if not self.api_key:
            return None
        try:
            resp = self.session.get(
                CRYPTOPANIC_API_BASE, params={"auth_token": self.api_key, "public": "true"}, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"NewsFeed fetch error: {e}")
            return None

        posts = []
        for item in data.get("results", []):
            try:
                votes = item.get("votes", {}) or {}
                positive = float(votes.get("positive", 0) or 0)
                negative = float(votes.get("negative", 0) or 0)
                total_votes = positive + negative
                sentiment = (positive - negative) / total_votes if total_votes > 0 else 0.0
                source = ((item.get("source") or {}).get("domain") or "").lower()
                currencies = [c.get("code") for c in (item.get("currencies") or []) if c.get("code")]
                published_at = _parse_iso(item.get("published_at"))
                if published_at is None:
                    continue
                posts.append({
                    "title": item.get("title", ""),
                    "source": source,
                    "currencies": currencies,
                    "sentiment": max(-1.0, min(1.0, sentiment)),
                    "published_at": published_at,
                })
            except Exception:
                continue
        return posts

    def _write_cache(self, posts: list):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            serializable = [{**p, "published_at": p["published_at"].isoformat()} for p in posts]
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"posts": serializable, "cached_at": _now_iso()}, f)
            os.replace(tmp, self.cache_path)
        except Exception as e:
            logger.error(f"NewsFeed cache write error: {e}")

    def _recent_posts(self, now: datetime.datetime = None) -> list:
        if not self._available:
            return []
        now = now or datetime.datetime.now(datetime.timezone.utc)
        cutoff = now - datetime.timedelta(hours=NEWS_SENTIMENT_WINDOW_HOURS)
        return [p for p in self._posts if p["published_at"] >= cutoff]

    def sentiment_1h(self) -> float:
        recent = self._recent_posts()
        if not recent:
            return 0.0
        return statistics.mean(p["sentiment"] for p in recent)

    def count_1h(self) -> int:
        return len(self._recent_posts())

    def has_major_1h(self) -> bool:
        return any(p["source"] in NEWS_MAJOR_SOURCES for p in self._recent_posts())


def _parse_iso(value) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
