import json
import os
import logging
import datetime
from config.settings import STATE_FILE

logger = logging.getLogger(__name__)

class StateManager:
    DEFAULT_STATE = {
        "open_market_ids": [],
        "daily_loss_usd": 0.0,
        "loss_streak": 0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl_usd": 0.0,
        "last_signal_ts": {},
        "system_stopped": False,
        "stop_reason": "",
        "last_daily_reset": "",
        "session_start": "",
        # 2026-07-08: replaces the old, dashboard-only "session_start_clean"
        # field (which was written exclusively by scripts/dashboard_server.py's
        # POST /api/reset-session, never by this class) -- consolidated so
        # there's a single, consistent session-boundary timestamp used by
        # both stats_calculator.py's session/ vs alltime stats.json blocks
        # and the dashboard's "Reset Session" button.
        "session_start_ts": "",
    }

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.state = {}
        self._load()

    def get(self, key: str, default=None):
        return self.state.get(key, default)

    def set(self, key: str, value):
        self.state[key] = value
        self._save()

    def update(self, updates: dict):
        self.state.update(updates)
        self._save()

    def is_stopped(self) -> bool:
        return self.state.get("system_stopped", False)

    def stop_system(self, reason: str):
        logger.warning(f"SYSTEM STOP: {reason}")
        self.state["system_stopped"] = True
        self.state["stop_reason"] = reason
        self._save()

    def reset_daily(self):
        self.state["daily_loss_usd"] = 0.0
        self.state["loss_streak"] = 0
        self._save()

    def _load(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    saved = json.load(f)
                self.state = {**self.DEFAULT_STATE, **saved}
                self.state["session_start"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self._ensure_session_start_ts(saved)
                return
            except Exception as e:
                logger.warning(f"StateManager load error: {e}")
        self.state = dict(self.DEFAULT_STATE)
        self.state["session_start"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.state["session_start_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._save()

    def _ensure_session_start_ts(self, saved: dict):
        """If session_start_ts isn't already on disk, set it now (and save
        immediately, since _load()'s normal success path doesn't otherwise
        call _save()) -- one-time migration (2026-07-08): if the old
        session_start_clean field is present, reuse ITS value so upgrading
        doesn't silently reset whatever session the user was already
        looking at; otherwise (a genuinely first-ever start) use now."""
        if self.state.get("session_start_ts"):
            return
        self.state["session_start_ts"] = (
            saved.get("session_start_clean")
            or datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        self._save()

    def _save(self):
        try:
            # Merge with whatever is currently on disk before overwriting, so a
            # key written by another process (e.g. the dashboard server's
            # session_start_clean reset) isn't wiped out by our next save of
            # this process's in-memory state.
            on_disk = {}
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file) as f:
                        on_disk = json.load(f)
                except Exception:
                    on_disk = {}
            merged = {**on_disk, **self.state}
            tmp_path = self.state_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(merged, f, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception as e:
            logger.error(f"StateManager save error: {e}")
