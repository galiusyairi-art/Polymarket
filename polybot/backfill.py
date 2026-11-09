"""Years of 1-minute bars from IBKR's history (the owner, 09.11.2026, "way 2: IBKR's history"): what the experiments
lab needs to judge the intraday strategies on years of real market data, not the month Yahoo keeps.

Only while the market is closed (weekdays 17:30 to 8:00 New York, and the weekend), on the data connection of its own
(never the orders'), one session at a time, newest first: the hundred stocks of Atlas and SPY, QQQ, each a day of 1-minute
bars of the regular hours, into the same day files the lab reads (scalp.Bars; a day already there is never fetched
again). Gently: a few requests at a time, a pause between days (IBKR's pacing). Never past the disk's room: it stops
while less than FREE_MIN is free (the volume is shared with the bot's database). As days arrive, the lab runs again
(labrun.py: each day's results kept, so only the new days are computed, in a process of its own)."""
import datetime as dt
import logging
import shutil
import threading
import time
from zoneinfo import ZoneInfo

from . import equity, scalp

log = logging.getLogger("polybot")
NY = ZoneInfo("America/New_York")

YEARS = 5
FREE_MIN = 1.5e9  # bytes left free on the volume at the least
MIN_SHARE = 0.5  # a day is kept only with at least this share of the symbols
BAD_DAYS = 3  # days in a row with too little: stop until the next window (said, with IBKR's own words)
PAUSE = 2.0  # seconds between two days
RUN_EVERY = 20  # days fetched between two runs of the lab


def symbols():
    return list(equity.UNIVERSE) + ["SPY", "QQQ"]


def in_window(now):
    """The market closed, and away from the live lab's morning (from 8:00) and the evening's recording (to 17:30): the
    weekend all day, a weekday's night."""
    t = dt.datetime.fromtimestamp(now, NY)
    m = t.hour * 60 + t.minute
    return t.weekday() >= 5 or m < 8 * 60 or m >= 17 * 60 + 30


class Backfill:
    def __init__(self, atlas):
        self.a = atlas
        self.running = False
        self.stop = threading.Event()

    def state(self):
        return self.a._get("backfill") or {}

    def wanted(self, today):
        """The sessions of the last YEARS years not in the day files yet, newest first (the daily data's calendar)."""
        with self.a.eq.lock:
            spy = self.a.eq.prepared.get("SPY")
        if not spy:
            return []
        cut = (dt.date.fromisoformat(today) - dt.timedelta(days=int(YEARS * 365.25))).isoformat()
        have = set(self.a.lab_bars.days())
        return [d for d in reversed(spy["day"]) if cut <= d < today and d not in have]

    def due(self, now):
        S = self.a.settings()
        if self.running or not S.get("backfill_on", True) or not S.get("ibkr_on") or not in_window(now):
            return False
        st = self.state()
        if now - (st.get("tried") or 0) < 1800 and st.get("stopped"):
            return False  # (stopped on a problem: tried again half an hour later)
        main = self.a.ibkr
        return bool(main is not None and main.connected())

    def run(self, now=None, gateway=None, deadline=None):
        """Fetches sessions until the window closes, the disk's room ends, or nothing is left. The state saved."""
        now = now or time.time()
        self.running = True
        today = dt.datetime.fromtimestamp(now, NY).date().isoformat()
        st = dict(self.state(), tried=now, stopped=None, running=True)
        self.a._set("backfill", st)
        bad, since_run = 0, 0
        try:
            gw = gateway or self.a.data_gateway()
            if not gw.connected():
                gw.connect()
            todo = self.wanted(today)
            st.update(left=len(todo), target=len(todo) + len(self.a.lab_bars.days()))
            for d in todo:
                t = time.time()
                if self.stop.is_set() or (deadline and t >= deadline) or (not deadline and not in_window(t)):
                    st["stopped"] = None
                    break
                free = shutil.disk_usage(self.a.lab_bars.folder).free
                if free < FREE_MIN:
                    st["stopped"] = f"נעצר: נשארו {free / 1e9:.1f}GB פנויים בדיסק (המינימום {FREE_MIN / 1e9:.1f}GB)"
                    break
                got, errors = gw.bars_day(symbols(), d)
                need = MIN_SHARE * len(symbols())
                if len(got) >= need:
                    by = {s: [b for b in bars if scalp.day_of(b[0]) == d and scalp.OPEN <= scalp.minute_of(b[0]) < 960]
                          for s, bars in got.items()}
                    self.a.lab_bars.save(d, {s: b for s, b in by.items() if b})
                    st.update(done=(st.get("done") or 0) + 1, oldest=d if not st.get("oldest") or d < st["oldest"] else st["oldest"],
                              last=time.time(), error=None, left=max(0, (st.get("left") or 1) - 1))
                    bad = 0
                    since_run += 1
                else:
                    bad += 1
                    st["error"] = f"{d}: רק {len(got)} מניות מתוך {len(symbols())}" + (f" · אינטראקטיב: {errors['_ibkr']}" if errors.get("_ibkr") else "")
                    if bad >= BAD_DAYS:
                        st["stopped"] = "נעצר אחרי 3 ימים בלי נתונים: " + st["error"]
                        break
                self.a._set("backfill", dict(st))
                if since_run >= RUN_EVERY:
                    since_run = 0
                    self.a.lab_run()
                time.sleep(PAUSE)
            if since_run:
                self.a.lab_run()
        except Exception as e:
            st["stopped"] = f"{type(e).__name__}: {e}"[:300]
            log.warning("Backfill: %s", st["stopped"])
        finally:
            self.running = False
            st["running"] = False
            self.a._set("backfill", st)
        log.warning("Backfill: %s sessions in all, oldest %s%s", st.get("done"), st.get("oldest"),
                    f" ({st['stopped']})" if st.get("stopped") else "")
        return st

    def tick(self, now=None):
        now = now or time.time()
        if self.due(now):
            threading.Thread(target=self.run, args=(now,), daemon=True, name="atlas-backfill").start()
            return True
        return False

    def view(self, now=None):
        st = self.state()
        have = self.a.lab_bars.days()
        return {**{k: st.get(k) for k in ("done", "oldest", "last", "error", "stopped", "running", "left", "tried")},
                "days": len(have), "first": have[0] if have else None, "years": YEARS}
