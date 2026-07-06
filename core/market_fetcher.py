import requests
import json
import logging
import datetime
import time
import os
import statistics
from typing import Optional
from config.settings import (
    POLYMARKET_GAMMA_BASE, POLYMARKET_API_BASE, LATENCY_LOG, LATENCY_WINDOW,
    API_STATS_LOG, API_STATS_EXPORT_INTERVAL_SEC,
)

logger = logging.getLogger(__name__)

class MarketFetcher:
    # This is the real gate on which assets' markets get fetched from
    # Polymarket at all (get_active_5min_markets() only ever builds slugs
    # from this dict's values) -- config.settings.REPRICING_FROZEN["assets"]
    # is documentation only and has no functional effect.
    ASSET_SLUG_PREFIX = {
        "BTC": "btc",
        "ETH": "eth",
        "SOL": "sol",
    }
    WINDOW_SEC = 300

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._latencies_ms: list[float] = []
        self.api_call_count = 0
        self._api_call_ts: list[float] = []
        self._last_api_stats_export = 0.0

    def _timed_get(self, url, **kwargs):
        """Every real Polymarket HTTP call in this class goes through here so
        data/latency.json reflects genuinely measured round-trip time, not a
        guess -- used only by the dashboard's LATENCY widget, never on any
        trading-critical decision path. Also the single point where every
        real API call is counted for data/api_stats.json (header bar's API
        calls/min field)."""
        start = time.monotonic()
        self._record_api_call(start)
        try:
            return self.session.get(url, **kwargs)
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            self._record_latency(elapsed_ms)

    def _record_api_call(self, now: float):
        self.api_call_count += 1
        self._api_call_ts.append(now)
        self._maybe_export_api_stats(now)

    def _maybe_export_api_stats(self, now: float):
        if now - self._last_api_stats_export < API_STATS_EXPORT_INTERVAL_SEC:
            return
        self._last_api_stats_export = now
        self._export_api_stats(now)

    def _export_api_stats(self, now: float):
        cutoff = now - 60
        self._api_call_ts = [t for t in self._api_call_ts if t > cutoff]
        data = {
            "calls_last_minute": len(self._api_call_ts),
            "calls_total": self.api_call_count,
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            os.makedirs(os.path.dirname(API_STATS_LOG), exist_ok=True)
            tmp_path = API_STATS_LOG + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, API_STATS_LOG)
        except Exception as e:
            logger.error(f"api stats export error: {e}")

    def _record_latency(self, elapsed_ms: float):
        self._latencies_ms.append(elapsed_ms)
        if len(self._latencies_ms) > LATENCY_WINDOW:
            self._latencies_ms.pop(0)
        self._export_latency_status(elapsed_ms)

    def _export_latency_status(self, last_ms: float):
        sorted_lat = sorted(self._latencies_ms)
        avg_ms = statistics.mean(sorted_lat)
        p99_idx = min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.99))
        p99_ms = sorted_lat[p99_idx]
        data = {
            "avg_ms": round(avg_ms, 1),
            "p99_ms": round(p99_ms, 1),
            "last_ms": round(last_ms, 1),
            "sample_count": len(sorted_lat),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            os.makedirs(os.path.dirname(LATENCY_LOG), exist_ok=True)
            tmp_path = LATENCY_LOG + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, LATENCY_LOG)
        except Exception as e:
            logger.error(f"latency export error: {e}")

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

    def ping(self) -> bool:
        """Lightweight reachability check for the dashboard's exchange-status
        widget only -- not used by any trading-critical path (which already
        has its own per-call error handling and never depends on this)."""
        try:
            resp = self._timed_get(f"{POLYMARKET_GAMMA_BASE}/markets", params={"limit": 1}, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"MarketFetcher ping error: {e}")
            return False

    def get_market_resolution(self, condition_id: str) -> Optional[dict]:
        try:
            url = f"{POLYMARKET_API_BASE}/markets/{condition_id}"
            resp = self._timed_get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"MarketFetcher error fetching resolution condition_id={condition_id}: {e}")
            return None

    def resolve_outcome(self, resolution: dict) -> Optional[str]:
        try:
            if not resolution.get("closed"):
                return None
            for token in resolution.get("tokens", []):
                if not token.get("winner"):
                    continue
                outcome = str(token.get("outcome", "")).strip().lower()
                if outcome == "up":
                    return "YES"
                if outcome == "down":
                    return "NO"
            return None
        except Exception as e:
            logger.warning(f"resolve_outcome error: {e}")
            return None

    def _asset_from_slug(self, slug: str) -> Optional[str]:
        for asset, prefix in self.ASSET_SLUG_PREFIX.items():
            if slug.startswith(f"{prefix}-updown-5m-"):
                return asset
        return None

    def _fetch_markets_by_slug(self, slugs: list[str]) -> list[dict]:
        url = f"{POLYMARKET_GAMMA_BASE}/markets"
        params = [("slug", s) for s in slugs] + [("limit", len(slugs))]
        resp = self._timed_get(url, params=params, timeout=15)
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
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            token_by_label = {
                str(label).strip().lower(): str(token_id)
                for label, token_id in zip(outcomes, token_ids)
            }
            up_token = token_by_label.get("up")
            down_token = token_by_label.get("down")
            if not up_token or not down_token:
                return None
            yes_price = self._token_mid_price(up_token)
            no_price = self._token_mid_price(down_token)
            if yes_price is None or no_price is None:
                return None
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
                "up_token_id": up_token,
                "down_token_id": down_token,
                "volume_24h": float(m.get("volume24hr") or 0),
            }
        except Exception as e:
            logger.warning(f"_parse_market error: {e}")
            return None

    def _token_mid_price(self, token_id: str) -> Optional[float]:
        try:
            book = self._fetch_order_book(token_id)
            bid = self._best_price(book.get("bids"), highest=True)
            ask = self._best_price(book.get("asks"), highest=False)
            if bid is not None and ask is not None:
                return (bid + ask) / 2
            return ask if ask is not None else bid
        except Exception as e:
            logger.warning(f"_token_mid_price error token={token_id}: {e}")
            return None

    def _fetch_order_book(self, token_id: str) -> dict:
        url = f"{POLYMARKET_API_BASE}/book"
        resp = self._timed_get(url, params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _best_price(self, levels: Optional[list], highest: bool) -> Optional[float]:
        prices = []
        for level in levels or []:
            try:
                prices.append(float(level.get("price")))
            except (TypeError, ValueError, AttributeError):
                continue
        if not prices:
            return None
        return max(prices) if highest else min(prices)

    def get_order_book_top(self, token_id: str) -> Optional[dict]:
        try:
            book = self._fetch_order_book(token_id)
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            bid = self._best_level(bids, highest=True)
            ask = self._best_level(asks, highest=False)
            return {
                "best_bid_price": bid[0] if bid else None,
                "best_bid_size": bid[1] if bid else None,
                "best_ask_price": ask[0] if ask else None,
                "best_ask_size": ask[1] if ask else None,
                "total_bid_depth": self._total_size(bids),
                "total_ask_depth": self._total_size(asks),
            }
        except Exception as e:
            logger.warning(f"get_order_book_top error token={token_id}: {e}")
            return None

    def _best_level(self, levels: Optional[list], highest: bool) -> Optional[tuple]:
        best = None
        for level in levels or []:
            try:
                price = float(level.get("price"))
                size = float(level.get("size", 0))
            except (TypeError, ValueError, AttributeError):
                continue
            if best is None or (highest and price > best[0]) or (not highest and price < best[0]):
                best = (price, size)
        return best

    def _total_size(self, levels: Optional[list]) -> float:
        total = 0.0
        for level in levels or []:
            try:
                total += float(level.get("size", 0))
            except (TypeError, ValueError, AttributeError):
                continue
        return total
