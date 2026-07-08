import logging
from typing import Optional

import requests

from config.settings import (
    COINGECKO_API_BASE, MARKET_BIAS_BULLISH_THRESHOLD, MARKET_BIAS_BEARISH_THRESHOLD,
    FEAR_GREED_API_BASE,
)

logger = logging.getLogger(__name__)


class MarketBiasFetcher:
    """Fetches BTC/ETH/SOL spot price and 24h change from CoinGecko's free,
    no-key public API. This is a real, independent external data source --
    distinct from Polymarket's own YES/NO contract prices -- used to derive
    a simple directional bias label from BTC's 24h change, which
    signals/repricing_signal.py uses to gate YES-direction signals during a
    falling market. SOL is tracked for the dashboard's SOL/BTC spot
    correlation only -- SOL isn't traded (REPRICING_FROZEN.assets excludes
    it) and doesn't feed the bias calculation.
    """

    def fetch(self) -> Optional[dict]:
        try:
            resp = requests.get(
                COINGECKO_API_BASE,
                params={"ids": "bitcoin,ethereum,solana", "vs_currencies": "usd", "include_24hr_change": "true"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            btc = data.get("bitcoin") or {}
            eth = data.get("ethereum") or {}
            sol = data.get("solana") or {}
            btc_price, btc_change = btc.get("usd"), btc.get("usd_24h_change")
            eth_price, eth_change = eth.get("usd"), eth.get("usd_24h_change")
            sol_price, sol_change = sol.get("usd"), sol.get("usd_24h_change")
            if None in (btc_price, btc_change, eth_price, eth_change):
                logger.warning("MarketBiasFetcher: incomplete CoinGecko response")
                return None
            return {
                "btc_price": btc_price,
                "btc_24h_change": btc_change,
                "eth_price": eth_price,
                "eth_24h_change": eth_change,
                # SOL is best-effort: never blocks the (BTC-driven) bias result
                "sol_price": sol_price,
                "sol_24h_change": sol_change,
                "market_bias": self.bias_from_change(btc_change),
            }
        except Exception as e:
            logger.warning(f"MarketBiasFetcher fetch error: {e}")
            return None

    @staticmethod
    def bias_from_change(btc_24h_change: float) -> str:
        if btc_24h_change > MARKET_BIAS_BULLISH_THRESHOLD:
            return "BULLISH"
        if btc_24h_change < MARKET_BIAS_BEARISH_THRESHOLD:
            return "BEARISH"
        return "NEUTRAL"


class FearGreedFetcher:
    """Fetches the current Crypto Fear & Greed Index from alternative.me's
    free, no-key public API -- a real, independent third sentiment source
    (distinct from both Polymarket contract prices and CoinGecko spot
    prices)."""

    def fetch(self) -> Optional[dict]:
        try:
            resp = requests.get(FEAR_GREED_API_BASE, params={"limit": 1}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data") or []
            if not rows:
                return None
            row = rows[0]
            value_raw = row.get("value")
            classification = row.get("value_classification")
            if value_raw is None or not classification:
                return None
            return {"value": int(value_raw), "classification": classification}
        except Exception as e:
            logger.warning(f"FearGreedFetcher fetch error: {e}")
            return None
