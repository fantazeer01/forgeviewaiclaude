import requests
import json
import logging
import datetime
import time
from typing import Optional
from config.settings import POLYMARKET_GAMMA_BASE

logger = logging.getLogger(__name__)

class MarketFetcher:
    ASSET_SLUG_PREFIX = {
        "BTC": "btc",
        "ETH": "eth",
        "SOL": "sol",
    }
    WINDOW_SEC = 300

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_active_5min_markets(self) -> list[dict]:
        try:
            now = time.time()
            boundary = self._current_boundary(now)
            slugs = [f"{prefix}-updown-5m-{boundary}" for prefix in self.ASSET_SLUG_PREFIX.values()]
            markets = self._fetch_markets_by_slug(slugs)
            result = []
            for m in markets:
                asset = self._asset_from_slug(m.get("slug", ""))
                if asset is None:
                    continue
                parsed = self._parse_market(m, asset, now)
                if parsed:
                    result.append(parsed)
            logger.info(f"Found {len(result)} active 5-min markets")
            return result
        except Exception as e:
            logger.error(f"MarketFetcher error: {e}")
            return []

    def _current_boundary(self, now: float) -> int:
        now_int = int(now)
        return now_int - (now_int % self.WINDOW_SEC)

    def _asset_from_slug(self, slug: str) -> Optional[str]:
        for asset, prefix in self.ASSET_SLUG_PREFIX.items():
            if slug.startswith(f"{prefix}-updown-5m-"):
                return asset
        return None

    def _fetch_markets_by_slug(self, slugs: list[str]) -> list[dict]:
        url = f"{POLYMARKET_GAMMA_BASE}/markets"
        params = [("slug", s) for s in slugs] + [("limit", len(slugs))]
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("markets", [])

    def _parse_market(self, m: dict, asset: str, now: float) -> Optional[dict]:
        try:
            if m.get("closed"):
                return None
            outcomes = json.loads(m.get("outcomes", "[]"))
            prices = json.loads(m.get("outcomePrices", "[]"))
            price_by_label = {
                str(label).strip().lower(): float(price)
                for label, price in zip(outcomes, prices)
            }
            yes_price = price_by_label.get("up", 0.5)
            no_price = price_by_label.get("down", 0.5)
            end_date = m.get("endDate") or ""
            minutes_remaining = 5.0
            if end_date:
                try:
                    end_dt = datetime.datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    now_dt = datetime.datetime.now(datetime.timezone.utc)
                    minutes_remaining = max(0.0, (end_dt - now_dt).total_seconds() / 60)
                except Exception:
                    pass
            return {
                "market_id": m.get("conditionId") or m.get("id", ""),
                "asset": asset,
                "question": m.get("question", ""),
                "end_date_iso": end_date,
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": float(m.get("volumeNum") or m.get("volume") or 0),
                "minutes_remaining": minutes_remaining,
            }
        except Exception as e:
            logger.warning(f"_parse_market error: {e}")
            return None
