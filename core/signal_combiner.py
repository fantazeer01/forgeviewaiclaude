import datetime
import json
import logging
import os
from typing import Optional

from config.settings import (
    SIGNAL_COMBINER_WEIGHTS, SIGNAL_COMBINER_THRESHOLD, SIGNAL_COMBINER_STATUS_FILE,
    SIGNAL_COMBINER_MIN_YES_PRICE, SIGNAL_COMBINER_MAX_YES_PRICE,
)
from core.repricing_detector import RepricingSignal
from core.signals.order_book_signal import OrderBookSignalGenerator
from core.signals.momentum_signal import MomentumSignalGenerator
from core.signals.volume_signal import VolumeSignalGenerator
from core.signals.correlation_signal import CorrelationFilter

logger = logging.getLogger(__name__)


class SignalCombiner:
    """Combines 3 independent trading signals into one weighted-confidence
    decision (quant-only mode -- repricing was removed entirely, not just
    down-weighted; see QUANT_ONLY_MODE in config/settings.py):

        order_book:  0.25  (core/signals/order_book_signal.py)
        momentum:    0.25  (core/signals/momentum_signal.py)
        volume:      0.15  (core/signals/volume_signal.py)

    Final confidence is a weighted AVERAGE of only the signals that actually
    fired this tick (weights renormalized among the active subset), not a
    weighted sum with silent zero-padding for signals that didn't fire --
    e.g. if only momentum (0.25) and volume (0.15) fire, the combined
    confidence is (0.25*c_momentum + 0.15*c_volume) / (0.25+0.15), not
    divided by some larger total. A trade only fires when that combined
    confidence exceeds SIGNAL_COMBINER_THRESHOLD (0.60).

    core/signals/correlation_signal.py's CorrelationFilter is a FILTER, not
    one of the weighted signals: if it blocks (BTC/ETH correlation > 0.8
    and BTC just dropped, for an ETH market), combine() returns None
    immediately regardless of how strong the other signals are.

    A second FILTER (also not one of the weighted signals): yes_price must
    fall within [SIGNAL_COMBINER_MIN_YES_PRICE, SIGNAL_COMBINER_MAX_YES_PRICE]
    (0.45-0.60), per docs/polymarket/DECISIONS.md D-002's finding that price
    is the one factor that actually predicts win rate. None of the 3 signals
    above look at price at all, so without this gate combine() would happily
    fire on a market at yes_price=0.165 with the same confidence as one at
    0.50.
    """

    SIGNAL_NAMES = ("order_book", "momentum", "volume")

    def __init__(self, order_book_gen: Optional[OrderBookSignalGenerator] = None,
                 momentum_gen: Optional[MomentumSignalGenerator] = None,
                 volume_gen: Optional[VolumeSignalGenerator] = None,
                 correlation_filter: Optional[CorrelationFilter] = None,
                 status_path: str = SIGNAL_COMBINER_STATUS_FILE):
        self.order_book_gen = order_book_gen or OrderBookSignalGenerator()
        self.momentum_gen = momentum_gen or MomentumSignalGenerator()
        self.volume_gen = volume_gen or VolumeSignalGenerator()
        self.correlation_filter = correlation_filter or CorrelationFilter()
        self.status_path = status_path
        self._status: dict[str, dict] = {}
        self._stats_date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        self._signal_stats: dict[str, dict] = {
            name: {"fired_today": 0, "last_fired_at": None} for name in self.SIGNAL_NAMES
        }
        self._load_signal_stats()

    def _load_signal_stats(self):
        """Restores today's fired-count/last-fired-at from the previously
        exported status file, so a bot restart doesn't reset the "fired
        today" counters back to zero mid-day. If the persisted stats are
        from a prior UTC day, they're stale and _reset_stats_if_new_day()
        (called from the normal update path) will zero them out instead."""
        if not os.path.exists(self.status_path):
            return
        try:
            with open(self.status_path) as f:
                data = json.load(f)
            stats = data.get("signal_stats")
            stats_date = data.get("signal_stats_date")
            if stats and stats_date == self._stats_date:
                for name in self.SIGNAL_NAMES:
                    if name in stats:
                        self._signal_stats[name] = {
                            "fired_today": stats[name].get("fired_today", 0),
                            "last_fired_at": stats[name].get("last_fired_at"),
                        }
        except Exception as e:
            logger.error(f"SignalCombiner signal_stats load error: {e}")

    def _reset_stats_if_new_day(self):
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if today != self._stats_date:
            self._stats_date = today
            self._signal_stats = {
                name: {"fired_today": 0, "last_fired_at": None} for name in self.SIGNAL_NAMES
            }

    def _record_fire(self, name: str, fired: bool):
        if fired:
            self._signal_stats[name]["fired_today"] += 1
            self._signal_stats[name]["last_fired_at"] = (
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            )

    @staticmethod
    def _signal_summary(signal: Optional[RepricingSignal]) -> dict:
        if signal is None:
            return {"fired": False, "confidence": None, "reason": None}
        return {"fired": True, "confidence": signal.confidence, "reason": signal.reason}

    def combine(self, market: dict, fetcher,
                btc_eth_correlation: Optional[float]) -> Optional[RepricingSignal]:
        asset = market["asset"]
        market_id = market["market_id"]
        self._reset_stats_if_new_day()

        # feed rolling histories every tick, regardless of whether anything fires
        self.momentum_gen.update(market_id, market["yes_price"])
        if asset == "BTC":
            self.correlation_filter.update_btc_price(market["yes_price"])
        self.volume_gen.record_volume(asset, market.get("volume_24h", 0.0))

        blocked = self.correlation_filter.should_block(asset, btc_eth_correlation)
        if blocked:
            self._status[asset] = {
                "order_book": self._signal_summary(None),
                "momentum": self._signal_summary(None),
                "volume": self._signal_summary(None),
                "correlation_filter_blocked": True,
                "price_out_of_band": False,
                "combined_confidence": None,
                "fired": False,
            }
            self._export_status()
            logger.info(f"Signal combiner blocked by correlation filter: {asset} {market_id}")
            return None

        # PRICE BAND FILTER (bug fix 2026-07-04): none of order_book/momentum/
        # volume signals below look at yes_price at all, so without this
        # check trades were firing at prices like 0.165/0.345 that
        # docs/polymarket/DECISIONS.md D-002 found have a materially lower
        # win rate than the 0.45-0.60 band -- this is the live-path
        # equivalent of the old repricing_detector's min_yes_price/
        # max_yes_price, which stopped applying to real trades once
        # QUANT_ONLY_MODE removed repricing from the trading path entirely.
        yes_price = market["yes_price"]
        price_out_of_band = not (SIGNAL_COMBINER_MIN_YES_PRICE <= yes_price <= SIGNAL_COMBINER_MAX_YES_PRICE)
        if price_out_of_band:
            self._status[asset] = {
                "order_book": self._signal_summary(None),
                "momentum": self._signal_summary(None),
                "volume": self._signal_summary(None),
                "correlation_filter_blocked": False,
                "price_out_of_band": True,
                "combined_confidence": None,
                "fired": False,
            }
            self._export_status()
            logger.info(
                f"Signal combiner blocked by price band: {asset} {market_id} "
                f"yes_price={yes_price} not in [{SIGNAL_COMBINER_MIN_YES_PRICE}, {SIGNAL_COMBINER_MAX_YES_PRICE}]"
            )
            return None

        order_book_signal = self.order_book_gen.generate(market, fetcher)
        momentum_signal = self.momentum_gen.generate(market)
        volume_signal = self.volume_gen.generate(market)
        self._record_fire("order_book", order_book_signal is not None)
        self._record_fire("momentum", momentum_signal is not None)
        self._record_fire("volume", volume_signal is not None)

        active = {}
        if order_book_signal is not None:
            active["order_book"] = order_book_signal
        if momentum_signal is not None:
            active["momentum"] = momentum_signal
        if volume_signal is not None:
            active["volume"] = volume_signal

        combined_confidence = None
        result = None
        if active:
            total_weight = sum(SIGNAL_COMBINER_WEIGHTS[name] for name in active)
            combined_confidence = sum(
                SIGNAL_COMBINER_WEIGHTS[name] * sig.confidence for name, sig in active.items()
            ) / total_weight
            if combined_confidence > SIGNAL_COMBINER_THRESHOLD:
                result = RepricingSignal(
                    asset=asset, market_id=market_id, direction="YES",
                    yes_price=market["yes_price"], no_price=market["no_price"],
                    confidence=round(combined_confidence, 3),
                    reason=f"combined({','.join(sorted(active.keys()))})={combined_confidence:.3f}",
                    minutes_remaining=market.get("minutes_remaining", 5.0),
                )

        self._status[asset] = {
            "order_book": self._signal_summary(order_book_signal),
            "momentum": self._signal_summary(momentum_signal),
            "volume": self._signal_summary(volume_signal),
            "correlation_filter_blocked": False,
            "price_out_of_band": False,
            "combined_confidence": round(combined_confidence, 3) if combined_confidence is not None else None,
            "fired": result is not None,
        }
        self._export_status()
        return result

    def _export_status(self):
        """Small JSON snapshot per asset the dashboard can read to show each
        of the 3 signals' status separately, plus a top-level "signal_stats"
        summary (today's fired count and last-fired-at timestamp per signal
        type, across all assets) -- this is display-only telemetry, not
        consumed by any trading logic."""
        try:
            os.makedirs(os.path.dirname(self.status_path), exist_ok=True)
            data = dict(self._status)
            data["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            data["signal_stats"] = self._signal_stats
            data["signal_stats_date"] = self._stats_date
            tmp_path = self.status_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, self.status_path)
        except Exception as e:
            logger.error(f"SignalCombiner status export error: {e}")
