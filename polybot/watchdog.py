"""The guard (the owner, 06.11.2026: "yes, add the guard"): the site stopped answering and stayed so until the
server was restarted by hand. Every minute it looks at three things: the trading loop still turns (it records its
own lag four times a second), the web server answers, and the page's data (Engine.state) is ready within
STATE_TIMEOUT seconds. FAILS bad looks in a row and the bot goes down by itself, the IB Gateway connection closed
first; Fly starts it again (fly.toml: restart always). Never in its first GRACE seconds, and at most MAX_RESTARTS
in RESTART_WINDOW: past that it stops restarting (a loop of restarts is worse than one stuck server) and says so.
After a restart of its own, one mail with why."""
import json
import logging
import os
import threading
import time
import urllib.request

log = logging.getLogger("polybot")


class Watchdog:
    EVERY = 60  # seconds between looks
    GRACE = 600  # seconds after a start before it may act (a start loads a lot)
    LOOP_SILENT = 180  # seconds without the trading loop's heartbeat
    STATE_TIMEOUT = 45  # seconds the page's data may take
    WEB_TIMEOUT = 20  # seconds the web server may take for /healthz
    FAILS = 3  # bad looks in a row before it restarts
    MAX_RESTARTS, RESTART_WINDOW = 3, 6 * 3600

    def __init__(self, engine, port, stop=None, exit=os._exit, clock=time.time):
        self.engine, self.port, self.stop, self.exit, self.clock = engine, port, stop, exit, clock
        self.bad = 0
        self.last = None  # the last look: {"ts", "problems"}
        self._probe = None  # the page's data being worked out (a thread), while it runs

    # ---------- the looks ----------
    def _loop_problem(self, now):
        e = self.engine
        if not getattr(e, "streaming", False):
            return None
        lags = getattr(getattr(e, "health", None), "loop_lag", None)
        beat = lags[-1][0] if lags else getattr(e, "started_at", now)
        if now - beat > self.LOOP_SILENT:
            return f"לולאת המסחר לא זזה כבר {int(now - beat)} שניות"
        return None

    def _web_problem(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=self.WEB_TIMEOUT) as r:
                if r.status != 200:
                    return f"שרת האתר החזיר {r.status}"
        except Exception as e:
            return f"שרת האתר לא ענה: {type(e).__name__}"
        return None

    def _state_problem(self):
        if self._probe is not None and self._probe.is_alive():
            return f"נתוני הדף לא מוכנים כבר יותר מ-{self.STATE_TIMEOUT} שניות (עדיין מחכה לקודם)"
        done = {}

        def go():
            try:
                self.engine.state()
                done["ok"] = True
            except Exception as e:  # (a page that fails is the page's problem, not a stuck server)
                done["ok"] = True
                done["error"] = f"{type(e).__name__}: {e}"
        self._probe = threading.Thread(target=go, daemon=True, name="watchdog-state")
        self._probe.start()
        self._probe.join(self.STATE_TIMEOUT)
        if not done.get("ok"):
            return f"נתוני הדף לא היו מוכנים תוך {self.STATE_TIMEOUT} שניות"
        self._probe = None
        return None

    def look(self, now=None):
        now = now or self.clock()
        problems = [p for p in (self._loop_problem(now), self._web_problem(), self._state_problem()) if p]
        self.last = {"ts": now, "problems": problems}
        return problems

    # ---------- what it does about them ----------
    def tick(self, now=None):
        now = now or self.clock()
        if now - getattr(self.engine, "started_at", now) < self.GRACE:
            return None
        problems = self.look(now)
        self.bad = self.bad + 1 if problems else 0
        if self.bad >= self.FAILS:
            return self.restart(problems, now)
        return None

    def _history(self):
        try:
            return json.loads(self.engine.store.get_setting("watchdog_restarts") or "[]")
        except (TypeError, ValueError):
            return []

    def restart(self, problems, now=None):
        now = now or self.clock()
        recent = [t for t in self._history() if now - t < self.RESTART_WINDOW]
        if len(recent) >= self.MAX_RESTARTS:
            if self.bad == self.FAILS:  # (said once a run of bad looks)
                log.error("Watchdog: stuck (%s), but already %d restarts in %d hours: not restarting again",
                          "; ".join(problems), len(recent), self.RESTART_WINDOW // 3600)
            return "held"
        store = self.engine.store
        store.set_setting("watchdog_restarts", json.dumps(recent + [now]))
        store.set_setting("watchdog_last", json.dumps({"ts": now, "why": problems, "told": False}))
        log.error("Watchdog: the bot is stuck (%s): restarting it", "; ".join(problems))
        try:
            if self.stop:
                self.stop()
        finally:
            self.exit(1)
        return "restarted"

    def report(self):
        """After a restart of its own: one mail with why (the owner wants mail only for a real problem: this is one)."""
        try:
            last = json.loads(self.engine.store.get_setting("watchdog_last") or "null")
        except (TypeError, ValueError):
            return False
        if not last or last.get("told"):
            return False
        last["told"] = True
        self.engine.store.set_setting("watchdog_last", json.dumps(last))
        n = getattr(self.engine, "notifier", None)
        try:
            s = n.settings()
            if not (n.configured() and s["email"] and s["problems"]):
                return False
            from .notify import _shell
            import html
            when = time.strftime("%H:%M", time.localtime(last["ts"]))
            items = "".join(f"<li>{html.escape(p)}</li>" for p in last.get("why") or [])
            n._safe_send("🐕 הבוט נתקע והופעל מחדש לבד", _shell(
                "🐕 הבוט נתקע והופעל מחדש לבד", f"<p>ב-{when} השומר ראה שלוש פעמים ברצף שהבוט לא עונה:</p><ul>{items}</ul>"
                "<p>הוא הפעיל את הבוט מחדש, והבוט חזר לעבוד. אין צורך לעשות כלום. אם זה חוזר, כדאי לספר לי.</p>", n.url))
            return True
        except Exception as e:
            log.warning("Watchdog mail: %s", e)
            return False

    def start(self):
        def run():
            self.report()
            while True:
                time.sleep(self.EVERY)
                try:
                    self.tick()
                except Exception as e:  # (the guard never brings the bot down by a fault of its own)
                    log.warning("Watchdog: %s", e)
        threading.Thread(target=run, daemon=True, name="watchdog").start()
