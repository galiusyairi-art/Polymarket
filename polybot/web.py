"""The dashboard: a small web server that shows the engine's state and takes the Live tab's controls."""
import base64
import hmac
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("polybot")
STATIC = os.path.join(os.path.dirname(__file__), "static")


def make_handler(engine, password=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep the console for trading events

        def _authorized(self):
            if not password:
                return True
            header = self.headers.get("Authorization", "")
            if header.startswith("Basic "):
                try:
                    _, _, given = base64.b64decode(header[6:]).decode().partition(":")
                except ValueError:
                    return False
                return hmac.compare_digest(given, password)
            return False

        def _send(self, code, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _guard(self):
            if self._authorized():
                return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="polybot"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def do_GET(self):
            if self.path == "/healthz":  # for the hosting platform's health check; reveals nothing
                self._send(200, {"ok": True})
                return
            if not self._guard():
                return
            if self.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, engine.state())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard():
                return
            # Browsers send Origin on cross-site POSTs; refuse those so another
            # website can't flip the Live switch through your browser.
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host"):
                self._send(403, {"error": "cross-site request refused"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/live/arm":
                    engine.live.arm(bool(body.get("on")))
                elif self.path == "/api/live/settings":
                    engine.live.update_settings(
                        stake=float(body["stake"]) if body.get("stake") is not None else None,
                        daily_loss_limit=float(body["daily_loss_limit"])
                        if body.get("daily_loss_limit") is not None else None,
                    )
                elif self.path == "/api/live/stop":
                    engine.live.emergency_stop()
                elif self.path == "/api/lab/research":
                    engine.start_research()
                    self._send(200, {"ok": True})
                    return
                else:
                    self._send(404, {"error": "not found"})
                    return
                self._send(200, {"ok": True, "live": engine.live.status()})
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})

    return Handler


def serve(engine, host, port, password=None):
    server = ThreadingHTTPServer((host, port), make_handler(engine, password))
    log.info("Dashboard: http://%s:%d", "localhost" if host in ("127.0.0.1", "0.0.0.0") else host, port)
    server.serve_forever()
