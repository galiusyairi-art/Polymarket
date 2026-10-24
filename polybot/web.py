"""The dashboard: a small web server that shows the engine's state and takes the Live tab's controls.

Sign-in is a password page that sets a signed cookie for 30 days. The cookie is derived
from the password, so changing POLYBOT_PASSWORD signs everyone out. Wrong passwords are
rate-limited per client address.
"""
import base64
import gzip
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("polybot")
STATIC = os.path.join(os.path.dirname(__file__), "static")
COOKIE = "polybot_session"
SESSION_DAYS = 30
MAX_FAILURES = 10  # per address per 10 minutes

LOGIN_PAGE = """<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>כניסה · בוט פולימרקט</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --text:#16181d; --muted:#646b78; --line:#e3e6eb; --accent:#3b5bdb; --bad:#c92a2a; }
@media (prefers-color-scheme: dark) { :root { --bg:#111318; --panel:#1a1d24; --text:#e8eaee; --muted:#9aa1ad; --line:#2b2f38; --accent:#7c95ff; --bad:#ff6b6b; } }
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; display:grid; place-items:center; background:var(--bg); color:var(--text);
  font:16px/1.5 system-ui, -apple-system, "Segoe UI", Arial, sans-serif; padding:16px; }
form { width:100%; max-width:360px; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:24px; }
h1 { font-size:20px; margin:0 0 4px; } p { color:var(--muted); margin:0 0 18px; font-size:14px; }
input { width:100%; padding:12px; font:inherit; border:1px solid var(--line); border-radius:10px; background:var(--bg); color:var(--text); }
button { width:100%; margin-top:12px; padding:12px; font:inherit; font-weight:600; border:0; border-radius:10px; background:var(--accent); color:#fff; cursor:pointer; }
.err { color:var(--bad); font-size:14px; margin-top:10px; min-height:1.5em; }
</style></head><body>
<form method="post" action="/login">
  <h1>בוט פולימרקט</h1>
  <p>הכנס את הסיסמה כדי להיכנס.</p>
  <input type="password" name="password" autocomplete="current-password" autofocus required placeholder="סיסמה">
  <button type="submit">כניסה</button>
  <div class="err">__ERROR__</div>
</form></body></html>"""


def session_token(password):
    """What the session cookie holds: a keyed hash of the password, never the password."""
    return hmac.new(hashlib.sha256(password.encode()).digest(), b"polybot-session-v1", hashlib.sha256).hexdigest()


def password_matches(given, password):
    # Compare as bytes: hmac.compare_digest rejects str with non-ASCII characters
    # (a Hebrew password would crash the check instead of passing it).
    return hmac.compare_digest(given.strip().encode(), password.encode())


def make_handler(engine, password=None):
    password = (password or "").strip() or None
    token = session_token(password) if password else None
    failures = {}  # address -> [timestamps]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep the console for trading events

        @property
        def route(self):
            return urllib.parse.urlsplit(self.path).path

        def _client(self):
            # Behind the hosting platform's proxy the real address is the first forwarded one.
            forwarded = self.headers.get("X-Forwarded-For")
            return forwarded.split(",")[0].strip() if forwarded else self.client_address[0]

        def _authorized(self):
            if not password:
                return True
            cookies = {}
            for part in (self.headers.get("Cookie") or "").split(";"):
                name, _, value = part.strip().partition("=")
                cookies[name] = value
            if hmac.compare_digest(cookies.get(COOKIE, "").encode(), token.encode()):
                return True
            header = self.headers.get("Authorization", "")
            if header.startswith("Basic "):  # still accepted, for scripts
                try:
                    _, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                except (ValueError, UnicodeDecodeError):
                    return False
                return password_matches(given, password)
            return False

        def _send(self, code, body, content_type="application/json; charset=utf-8", headers=()):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            compress = len(data) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or "")
            if compress:
                # The dashboard refreshes every few seconds; hosts meter bandwidth.
                data = gzip.compress(data, compresslevel=5)
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            if compress:
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location, headers=()):
            self._send(303, b"", "text/plain", headers=(("Location", location), *headers))

        def _login_page(self, error="", code=200):
            self._send(code, LOGIN_PAGE.replace("__ERROR__", error).encode(), "text/html; charset=utf-8")

        def _guard(self):
            if self._authorized():
                return True
            if self.route.startswith("/api/"):
                self._send(401, {"error": "sign in required"})
            else:
                self._redirect("/login")
            return False

        def do_GET(self):
            if self.route == "/healthz":  # for the hosting platform's health check; reveals nothing
                self._send(200, {"ok": True})
                return
            if self.route == "/health":
                # For an outside uptime monitor: 503 while any alert is open (feeds down,
                # data not persisted), so a monitor can mail when the bot is in trouble or gone.
                problems = engine._alerts(time.time())
                self._send(503 if problems else 200, {"ok": not problems, "problems": len(problems)})
                return
            if self.route == "/login":
                if self._authorized():
                    self._redirect("/")
                else:
                    self._login_page()
                return
            if self.route == "/logout":
                self._redirect("/login", headers=(("Set-Cookie", f"{COOKIE}=; Path=/; Max-Age=0"),))
                return
            if not self._guard():
                return
            if self.route in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.route == "/api/calc/trades":
                self._send(200, engine.calc_trades())
            elif self.route == "/api/state":
                try:
                    data = engine.state()
                except Exception as e:
                    # Answer with the reason instead of dropping the connection, so the page
                    # can tell "the server had a hiccup" from "the server is gone".
                    log.exception("Dashboard state failed")
                    self._send(500, {"error": f"{type(e).__name__}: {e}"})
                    return
                self._send(200, data)
            else:
                self._send(404, {"error": "not found"})

        def _login(self):
            if not password:
                self._redirect("/")
                return
            client, now = self._client(), time.time()
            recent = [t for t in failures.get(client, []) if now - t < 600]
            failures[client] = recent
            if len(recent) >= MAX_FAILURES:
                self._login_page("יותר מדי ניסיונות. נסה שוב בעוד כמה דקות.", 429)
                return
            length = min(int(self.headers.get("Content-Length") or 0), 4096)
            form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            given = (form.get("password") or [""])[0]
            if not password_matches(given, password):
                recent.append(now)
                log.warning("Failed sign-in from %s", client)
                self._login_page("סיסמה שגויה.", 401)
                return
            secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
            cookie = f"{COOKIE}={token}; Path=/; Max-Age={SESSION_DAYS * 86400}; HttpOnly; SameSite=Lax{secure}"
            self._redirect("/", headers=(("Set-Cookie", cookie),))

        def do_POST(self):
            if self.route == "/login":
                self._login()
                return
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
                if self.route == "/api/live/arm":
                    engine.live.arm(bool(body.get("on")))
                elif self.route == "/api/live/settings":
                    engine.live.update_settings(
                        stake=float(body["stake"]) if body.get("stake") is not None else None,
                        daily_loss_limit=float(body["daily_loss_limit"])
                        if body.get("daily_loss_limit") is not None else None,
                    )
                elif self.route == "/api/live/stop":
                    engine.live.emergency_stop()
                elif self.route == "/api/notify/settings":
                    engine.notifier.update(email=body.get("email"), open=body.get("open"),
                                           close=body.get("close"), daily=body.get("daily"),
                                           problems=body.get("problems"), quiet=body.get("quiet"),
                                           quiet_from=body.get("quiet_from"), quiet_to=body.get("quiet_to"),
                                           quiet_problems=body.get("quiet_problems"),
                                           open_every=body.get("open_every"), close_every=body.get("close_every"),
                                           lab=body.get("lab"))
                    self._send(200, {"ok": True, "notify": engine.notifier.status()})
                    return
                elif self.route == "/api/notify/test":
                    try:
                        engine.notifier.send_test()
                    except Exception as e:
                        self._send(400, {"error": str(e)})
                        return
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/mode":
                    engine.set_mode(bool(body.get("live")), confirmed=bool(body.get("confirmed")))
                    self._send(200, {"ok": True, "mode": engine.mode()})
                    return
                elif self.route == "/api/wallet/check":
                    try:
                        result = engine.live.check_connection()
                    except Exception as e:
                        self._send(400, {"error": str(e)})
                        return
                    self._send(200, dict(result, ok=True))
                    return
                elif self.route == "/api/notify/summary":
                    try:
                        engine.notifier.send_summary_now()
                    except Exception as e:
                        self._send(400, {"error": str(e)})
                        return
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/calc/save":
                    engine.save_calculator(body.get("targets") or [], body.get("min_ratio"), body.get("max_ratio"),
                                           body.get("strategies"))
                elif self.route == "/api/lab/auto_push":
                    engine.set_auto_push(bool(body.get("on")), "live" if body.get("mode") == "live" else "sim")
                elif self.route == "/api/strategy/push":
                    mode = "live" if body.get("mode") == "live" else "sim"
                    if body.get("recommended"):
                        changes = [(c["asset"], c["minutes"], c["variant"]) for c in engine.recommendation(mode)["changes"]]
                    else:
                        changes = [(str(a), int(m), str(v)) for a, m, v in body["changes"]]
                    engine.push(changes, mode=mode)
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/stocks/push":
                    engine.push_stocks(str(body["variant"]))
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/stocks/remove":
                    engine.remove_stocks()
                    self._send(200, {"ok": True})
                    return
                elif self.route in ("/api/trading/sim", "/api/trading/live"):
                    if not body.get("confirmed"):
                        raise ValueError("צריך אישור")
                    engine.save_trading(self.route.rsplit("/", 1)[1], body)
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/sim/strategies":
                    if not body.get("confirmed"):
                        raise ValueError("צריך אישור")
                    engine.set_sim_strategies(bool(body.get("crypto")), bool(body.get("stocks")),
                                              str(body["stocks_variant"]) if body.get("stocks_variant") else None)
                    if "stake" in body or "daily_loss" in body or "max_open" in body:
                        engine.set_sim_limits(body.get("stake"), body.get("daily_loss") or 0,
                                              body.get("max_open") or {"crypto": 0, "stocks": 0})
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/live/strategies":
                    if not body.get("confirmed"):
                        raise ValueError("צריך אישור")
                    if body.get("stocks_variant"):
                        engine.push_stocks(str(body["stocks_variant"]))
                    # Real money mirrors the simulation: what goes live also runs there.
                    on = engine.sim_strategies()
                    engine.set_sim_strategies(on["crypto"] or bool(body.get("crypto")),
                                              on["stocks"] or bool(body.get("stocks")))
                    engine.live.set_strategy("crypto", bool(body.get("crypto")), confirmed=True)
                    engine.live.set_strategy("stocks", bool(body.get("stocks")), confirmed=True)
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/live/strategy":
                    engine.live.set_strategy(str(body["name"]), bool(body.get("on")), confirmed=bool(body.get("confirmed")))
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/strategy/rollback":
                    engine.rollback(int(body["id"]), "live" if body.get("mode") == "live" else "sim")
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/main/promote":
                    engine.promote(str(body["asset"]), int(body["minutes"]), str(body["variant"]),
                                   "live" if body.get("mode") == "live" else "sim")
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/risk/settings":
                    engine.set_risk_reward(str(body["mode"]), body.get("ratio"))
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/costs/settings":
                    engine.set_operating_costs(body["hosting"])
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/main/demote":
                    engine.demote(str(body["asset"]), int(body["minutes"]))
                    self._send(200, {"ok": True})
                    return
                elif self.route == "/api/lab/research":
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
