import logging
from typing import Optional

import requests

from config.settings import (
    COINGECKO_API_BASE, MARKET_BIAS_BULLISH_THRESHOLD, MARKET_BIAS_BEARISH_THRESHOLD,
)

logger = logging.getLogger(__name__)


class MarketBiasFetcher:
    """Fetches BTC/ETH spot price and 24h change from CoinGecko's free,
    no-key public API. This is a real, independent external data source --
    distinct from Polymarket's own YES/NO contract prices -- used to derive
    a simple directional bias label from BTC's 24h change, which
    signals/repricing_signal.py uses to gate YES-direction signals during a
    falling market.
    """

    def fetch(self) -> Optional[dict]:
        try:
            resp = requests.get(
                COINGECKO_API_BASE,
                params={"ids": "bitcoin,ethereum", "vs_currencies": "usd", "include_24hr_change": "true"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            btc = data.get("bitcoin") or {}
            eth = data.get("ethereum") or {}
            btc_price, btc_change = btc.get("usd"), btc.get("usd_24h_change")
            eth_price, eth_change = eth.get("usd"), eth.get("usd_24h_change")
            if None in (btc_price, btc_change, eth_price, eth_change):
                logger.warning("MarketBiasFetcher: incomplete CoinGecko response")
                return None
            return {
                "btc_price": btc_price,
                "btc_24h_change": btc_change,
                "eth_price": eth_price,
                "eth_24h_change": eth_change,
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
