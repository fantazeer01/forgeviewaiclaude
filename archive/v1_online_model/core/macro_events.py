import datetime
import logging
from typing import Optional

import requests

from config.settings import MACRO_CALENDAR_URL

logger = logging.getLogger(__name__)


class MacroEventsFetcher:
    """Fetches the weekly USD economic calendar from ForexFactory's free,
    no-key public feed and returns the high-impact USD events closest to
    now. These are real scheduled events (Fed speeches, NFP, CPI, etc.)
    that move crypto markets.

    Honesty note: the feed's "this week" window is anchored to the real
    calendar week and does not necessarily align with whatever date this
    environment's clock reports. If no event in the feed is still in the
    future relative to now, this falls back to the latest-dated events in
    the feed rather than returning nothing or fabricating a future date --
    every date returned is real and taken verbatim from the feed either way.
    """

    def fetch_next_events(self, n: int = 3) -> Optional[list[dict]]:
        try:
            resp = requests.get(MACRO_CALENDAR_URL, timeout=15)
            resp.raise_for_status()
            events = resp.json()
        except Exception as e:
            logger.warning(f"MacroEventsFetcher fetch error: {e}")
            return None

        parsed = []
        for e in events:
            if e.get("country") != "USD" or e.get("impact") != "High":
                continue
            date_raw = e.get("date")
            title = e.get("title")
            if not date_raw or not title:
                continue
            try:
                dt = datetime.datetime.fromisoformat(date_raw)
            except (ValueError, TypeError):
                continue
            parsed.append((dt, {"title": title, "date": dt.isoformat(), "country": "USD", "impact": "High"}))

        if not parsed:
            return []

        parsed.sort(key=lambda pair: pair[0])
        now = datetime.datetime.now(datetime.timezone.utc)
        future = [pair for pair in parsed if pair[0] > now]
        chosen = future[:n] if future else parsed[-n:]
        return [entry for _dt, entry in chosen]
