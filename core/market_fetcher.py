import requests
import logging
import datetime
from typing import Optional
from config.settings import POLYMARKET_GAMMA_BASE

logger = logging.getLogger(__name__)

class MarketFetcher:
    ASSET_KEYWORDS = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "eth"],
        "SOL": ["solana", "sol"],
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_active_5min_markets(self) -> list[dict]:
        try:
            markets = self._fetch_gamma_markets()
            result = []
            for m in markets:
                asset = self._detect_asset(m.get("question", ""))
                if asset is None:
                    continue
                if not self._is_5min_updown(m):
                    continue
                parsed = self._parse_market(m, asset)
                if parsed:
                    result.append(parsed)
            logger.info(f"Found {len(result)} active 5-min markets")
            return result
        except Exception as e:
            logger.error(f"MarketFetcher error: {e}")
            return []

    def _fetch_gamma_markets(self) -> list[dict]:
        url = f"{POLYMARKET_GAMMA_BASE}/markets"
        params = {"active": "true", "closed": "false", "limit": 200}
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("markets", [])

    def _detect_asset(self, question: str) -> Optional[str]:
        q = question.lower()
        for asset, keywords in self.ASSET_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return asset
        return None

    def _is_5min_updown(self, market: dict) -> bool:
        question = market.get("question", "").lower()
        has_direction = any(w in question for w in ["up", "down", "above", "below"])
        duration = market.get("duration", 0)
        if isinstance(duration, (int, float)):
            return has_direction and 240 <= duration <= 360
        return has_direction and any(w in question for w in ["5", "five", "minute"])

    def _parse_market(self, m: dict, asset: str) -> Optional[dict]:
        try:
            tokens = m.get("tokens", [])
            yes_price, no_price = 0.5, 0.5
            for t in tokens:
                outcome = (t.get("outcome") or "").upper()
                price = float(t.get("price", 0.5))
                if outcome == "YES":
                    yes_price = price
                elif outcome == "NO":
                    no_price = price
            end_date = m.get("endDate") or m.get("end_date_iso") or ""
            minutes_remaining = 5.0
            if end_date:
                try:
                    end_dt = datetime.datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    now = datetime.datetime.now(datetime.timezone.utc)
                    minutes_remaining = max(0, (end_dt - now).total_seconds() / 60)
                except Exception:
                    pass
            return {
                "market_id": m.get("conditionId") or m.get("id", ""),
                "asset": asset,
                "question": m.get("question", ""),
                "end_date_iso": end_date,
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": float(m.get("volume", 0)),
                "minutes_remaining": minutes_remaining,
            }
        except Exception as e:
            logger.warning(f"_parse_market error: {e}")
            return None
