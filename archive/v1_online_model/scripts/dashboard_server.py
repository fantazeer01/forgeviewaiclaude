"""
Static file server for dashboard_pro.html / dashboard.html that also exposes
one small write endpoint the plain `python -m http.server` can't provide:

    POST /api/reset-session
        Sets data/state.json's "session_start_ts" to the current UTC time
        and returns {"status": "ok", "session_start_ts": ...} -- nothing
        else in state.json or anywhere else is touched. This is what the
        dashboard's "Reset Session" button calls -- it does NOT touch
        data/paper_trades.jsonl (or any other data file: model weights,
        price_history.jsonl, no_shadow.jsonl, model_health_log.jsonl,
        quant_features.jsonl are all untouched), so all history stays fully
        intact; dashboard_pro.html's own client-side scopeTradesToSession()
        just starts scoping to trades opened after this new timestamp.
        The write retries up to 3x with backoff on OSError, since this file
        is also written by run.py's StateManager and the two os.replace()
        calls can transiently collide with WinError 5 on Windows.

Run from the repo root (same directory this replaces `python -m http.server`
for):

    python scripts/dashboard_server.py [port]   # default port 8080
"""
import datetime
import http.server
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config.settings import STATE_FILE  # noqa: E402

DEFAULT_PORT = 8080
LOG_FILE = os.path.join(REPO_ROOT, "data", "server.log")


def _log_line(msg: str):
    """All of this server's own logging (startup line + per-request access
    log) goes here instead of stdout/stderr -- this process has nothing to do
    with the trading bot's terminal output reaching the browser (this file
    only serves static files + one JSON POST endpoint, never injects process
    logs into a response body), but there's no reason for it to spam whatever
    terminal it happens to be launched from either."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=REPO_ROOT, **kwargs)

    def log_message(self, fmt, *args):
        _log_line("[dashboard_server] " + (fmt % args))

    def do_POST(self):
        if self.path == "/api/reset-session":
            self._handle_reset_session()
        else:
            self.send_error(404, "Not found")

    def _handle_reset_session(self):
        state_path = os.path.join(REPO_ROOT, STATE_FILE)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            # run.py's own StateManager._save() does the same read-modify-write
            # + os.replace() dance on this same file -- on Windows, two
            # processes racing to os.replace() the same target can transiently
            # collide with WinError 5 (access denied) rather than one just
            # blocking. Retrying a few times with backoff clears it without
            # needing real cross-process locking.
            for attempt in range(3):
                try:
                    on_disk = {}
                    if os.path.exists(state_path):
                        with open(state_path) as f:
                            on_disk = json.load(f)
                    on_disk["session_start_ts"] = now_iso
                    os.makedirs(os.path.dirname(state_path), exist_ok=True)
                    tmp_path = state_path + ".tmp"
                    with open(tmp_path, "w") as f:
                        json.dump(on_disk, f, indent=2)
                    os.replace(tmp_path, state_path)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(0.05 * (2 ** attempt))
            body = json.dumps({"status": "ok", "session_start_ts": now_iso}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_error(500, f"reset-session failed: {e}")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = http.server.ThreadingHTTPServer(("", port), DashboardRequestHandler)
    _log_line(f"Serving {REPO_ROOT} on http://localhost:{port} (POST /api/reset-session enabled)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
