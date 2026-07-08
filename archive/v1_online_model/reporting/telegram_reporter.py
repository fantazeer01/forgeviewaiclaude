import logging
import os
from config.settings import TELEGRAM_ENABLED

logger = logging.getLogger(__name__)

class TelegramReporter:
    def __init__(self):
        self.enabled = TELEGRAM_ENABLED
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if self.enabled and (not self.token or not self.chat_id):
            logger.warning("Telegram: missing credentials, disabling")
            self.enabled = False

    def send_signal(self, signal, trade):
        if not self.enabled: return
        self._send(
            f"Signal: {signal.asset} {signal.direction}\n"
            f"Reason: {signal.reason}\n"
            f"Confidence: {signal.confidence:.0%}\n"
            f"Time left: {signal.minutes_remaining:.1f} min\n"
            f"Trade #{trade.trade_id} OPENED"
        )

    def send_close(self, trade, stats: dict):
        if not self.enabled: return
        icon = "WIN" if trade.result == "WIN" else "LOSS"
        self._send(
            f"{icon}: {trade.asset} {trade.direction}\n"
            f"PnL: ${trade.pnl_usd:+.4f}\n"
            f"Win rate: {stats['win_rate']*100:.1f}%\n"
            f"Total PnL: ${stats['total_pnl_usd']:+.4f}"
        )

    def send_stop(self, reason: str):
        if not self.enabled: return
        self._send(f"SYSTEM STOPPED: {reason}")

    def send_text(self, text: str):
        if not self.enabled: return
        self._send(text)

    def _send(self, text: str):
        try:
            import requests
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=10
            ).raise_for_status()
        except Exception as e:
            logger.warning(f"Telegram failed: {e}")
