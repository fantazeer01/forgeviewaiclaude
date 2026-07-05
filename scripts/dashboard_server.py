"""
Static file server for dashboard_pro.html / dashboard.html that also exposes
one small write endpoint the plain `python -m http.server` can't provide:

    POST /api/reset-session
        Sets data/state.json's "session_start_clean" to the current UTC time
        and returns it as JSON. This is what the dashboard's "Reset Session"
        button calls -- it does NOT touch data/paper_trades.jsonl, so all
        historical trades stay intact; the dashboard just starts computing
        its visible stats from trades opened after this timestamp.

Run from the repo root (same directory this replaces `python -m http.server`
for):

    python scripts/dashboard_server.py [port]   # default port 8080
"""
import datetime
import http.server
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from config.settings import STATE_FILE  # noqa: E402

DEFAULT_PORT = 8080


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=REPO_ROOT, **kwargs)

    def log_message(self, fmt, *args):
        # keep default stderr logging, just tagged for clarity
        sys.stderr.write("[dashboard_server] " + (fmt % args) + "\n")

    def do_POST(self):
        if self.path == "/api/reset-session":
            self._handle_reset_session()
        else:
            self.send_error(404, "Not found")

    def _handle_reset_session(self):
        state_path = os.path.join(REPO_ROOT, STATE_FILE)
        try:
            on_disk = {}
            if os.path.exists(state_path):
                with open(state_path) as f:
                    on_disk = json.load(f)
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            on_disk["session_start_clean"] = now_iso
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            tmp_path = state_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(on_disk, f, indent=2)
            os.replace(tmp_path, state_path)
            body = json.dumps({"session_start_clean": now_iso}).encode("utf-8")
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
    print(f"Serving {REPO_ROOT} on http://localhost:{port} (POST /api/reset-session enabled)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
