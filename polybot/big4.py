"""The four (the owner, 23.11.2026: "four assets: TQQQ, BITX, SNDK and one more of your choice. The system runs only on
them, on one-minute candles, analyses, and makes me trades, long or short ... an expert of these four only ... a new tab,
a clear switch of live trading or simulation only, the four charts, a mark on the chart where it went in and where it
sold, the profit of each ... a short stop, a trailing stop, a great weight on risk against reward").

The fourth is PLTR: of fifteen liquid, volatile candidates it moves least with the other three (its daily moves against
TQQQ's 0.49, BITX's 0.41, SNDK's 0.15), trades four billion dollars a day and can be sold short.

The rule (measured before it was built, on 60 days of 5-minute bars of the four, costs taken): intraday momentum by the
day's noise band. For each minute of the day, the usual distance of the price from the day's open over the last LOOK days
is the day's noise at that minute. A price beyond BAND times that noise, above the higher of the open and yesterday's close
or below the lower of them, is no longer noise: with it, long or short. Decisions only on the half hours (every look in
between is a whipsaw: the same rule looked at every 5 minutes made a third of the money). The stop trails on the band and
on the day's average price, whichever is nearer the price, checked on the half hours; a guard stop GUARD of the price away
works inside the minute; everything is flat before the close.

What the 60 days said, and what they did not: 77 trades, 49% won, the average win +2.33% against an average loss of -1.0%
(1 to 2.3), +0.63% a trade, both halves of the period in profit, the worst trade -4.2% (-2.6% with the guard). And the
things the owner asked for that the data refused: a stop a short way off (one 5-minute range) lost on every setup tried,
-0.25 to -0.39 of the risk a trade, because noise takes the stop out and the cost of two crossings is a large part of a small
risk; a fixed target of 2 or 3 times the risk, or entries only where the risk is under 1-2%, took the profit away (the
money is made by the few days that run). The reward against the risk is kept the way that works: losses cut on the half
hour, profits left to the close. Sixty days in a falling market are not proof: this starts in the simulation.

BITX made nothing in those 60 days (its spread is a seventh of a percent each way): it runs in the simulation with the rest
and its own record says whether it earns its place.

Real money: the switch is here and the amount beside it, but orders are not sent yet (LIVE_OPEN): the rule sells short and
closes on the half hour, which the broker's bracket orders here do not do, and the path must be tried against the broker
connected before a dollar goes through it.
"""
import logging
import threading
import time

from . import scalp

log = logging.getLogger("polybot")

SYMBOLS = ("TQQQ", "BITX", "SNDK", "PLTR")
NAMES = {"TQQQ": "נאסד\"ק 100 ממונף פי 3", "BITX": "ביטקוין ממונף פי 2", "SNDK": "סנדיסק", "PLTR": "פלנטיר"}
BAND = 1.75  # times the minute's usual distance from the open
LOOK = 14  # days the usual distance is taken over
STEP = 30  # minutes between decisions
FIRST, LAST_ENTRY, FLAT = 30, 330, 385  # minutes after the open: the first decision (10:00), the last entry (15:00), all out (15:55)
GUARD = 0.025  # the guard stop, of the price: inside the minute
COST = {"TQQQ": 0.0004, "BITX": 0.0015, "SNDK": 0.0006, "PLTR": 0.0004}  # a crossing of the spread, of the price, each way
SIZE = 500.0  # dollars a trade, until the owner sets his
SIZE_MAX = 50_000.0
LIVE_OPEN = False  # (see the module's last paragraph)
OPEN = scalp.OPEN  # 9:30 New York, in minutes of the day
POLL = 30  # seconds between two reads of the minutes while the market is open (four requests each)
KEEP_DAYS = 45
TRADES_KEPT = 400


def minute(ts):
    """Minutes since the open of the bar's day."""
    return scalp.minute_of(ts) - OPEN


def noise(days):
    """days: the last days' bars, oldest first, each [(ts, o, h, l, c, v)]. The usual distance of the price from the day's
    open at each minute of the day, over the last LOOK days: {minute: fraction}."""
    by = {}
    for bars in days[-LOOK:]:
        if not bars:
            continue
        o0 = bars[0][1]
        last = None
        seen = {}
        for b in bars:
            seen[minute(b[0])] = abs(b[4] / o0 - 1)
        for m in range(0, 390):  # (a minute with no trade keeps the distance of the one before it)
            if m in seen:
                last = seen[m]
            if last is not None:
                by.setdefault(m, []).append(last)
    return {m: sum(v) / len(v) for m, v in by.items() if len(v) >= min(LOOK, 5)}


class Day:
    """One symbol's day, a minute at a time: the bars go in in order (feed), the position and the trades come out."""

    def __init__(self, symbol, prev_close, sigma, band=BAND, guard=GUARD, cost=None):
        self.symbol, self.prev_close, self.sigma, self.band, self.guard = symbol, prev_close, sigma, band, guard
        self.cost = COST.get(symbol, 0.0005) if cost is None else cost
        self.open = None
        self.vsum = self.pv = 0.0
        self.vwap = None
        self.last_m = -1
        self.last = None  # the last bar
        self.pos = None  # {"side", "entry", "ts", "stop", "guard", "risk"}
        self.trades = []
        self.up = self.lo = None

    def bands(self, m):
        s = self.sigma.get(min(max(m, 0), 389))
        if s is None or self.open is None or not self.prev_close:
            return None, None
        return max(self.open, self.prev_close) * (1 + self.band * s), min(self.open, self.prev_close) * (1 - self.band * s)

    def _out(self, price, ts, why):
        p = self.pos
        ret = p["side"] * (price - p["entry"]) / p["entry"] - 2 * self.cost
        self.trades.append({"symbol": self.symbol, "side": p["side"], "entry": p["entry"], "in": p["ts"], "exit": price, "out": ts,
                            "ret": ret, "risk": p["risk"], "why": why})
        self.pos = None

    def feed(self, bar):
        """One closed minute bar. Returns what happened: None, ("in", pos), or ("out", trade)."""
        ts, o, h, l, c, v = bar
        m = minute(ts)
        if m <= self.last_m:
            return None
        if self.open is None:
            self.open = o
        tp = (h + l + c) / 3
        self.vsum += v
        self.pv += tp * v
        self.vwap = self.pv / self.vsum if self.vsum else c
        self.up, self.lo = self.bands(m)
        crossed = (m + 1) // STEP > (self.last_m + 1) // STEP  # (this bar closes on a half hour, or is the first after one)
        self.last_m, self.last = m, bar
        p = self.pos
        if p:  # the guard works inside the minute; a bar that opens beyond it goes out at its open
            if p["side"] > 0 and l <= p["guard"]:
                self._out(min(o, p["guard"]), ts, "guard")
                return "out", self.trades[-1]
            if p["side"] < 0 and h >= p["guard"]:
                self._out(max(o, p["guard"]), ts, "guard")
                return "out", self.trades[-1]
            if m >= FLAT:
                self._out(c, ts, "close")
                return "out", self.trades[-1]
        if not crossed or self.up is None or m + 1 < FIRST:
            return None
        if p:
            trail = max(self.up, self.vwap) if p["side"] > 0 else min(self.lo, self.vwap)
            if (p["side"] > 0 and c < trail) or (p["side"] < 0 and c > trail):
                self._out(c, ts, "trail")
                return "out", self.trades[-1]
            p["stop"] = max(p["stop"], trail) if p["side"] > 0 else min(p["stop"], trail)
            return None
        if m + 1 > LAST_ENTRY:
            return None
        side = 1 if c > self.up else -1 if c < self.lo else 0
        if not side:
            return None
        stop = max(self.up, self.vwap) if side > 0 else min(self.lo, self.vwap)
        self.pos = {"side": side, "entry": c, "ts": ts, "stop": stop, "guard": c * (1 - side * self.guard),
                    "risk": max(abs(c - stop) / c, 0.002)}
        return "in", self.pos


def replay(symbol, days, band=BAND):
    """The rule over recorded days (oldest first): every trade it would have made, once there are enough days before it."""
    out = []
    for i in range(1, len(days)):
        sigma = noise(days[:i])
        if len(sigma) < 300 or not days[i] or not days[i - 1]:
            continue
        d = Day(symbol, days[i - 1][-1][4], sigma, band)
        for b in days[i]:
            d.feed(b)
        if d.pos and d.last:
            d._out(d.last[4], d.last[0], "close")
        out += d.trades
    return out


def summary(trades):
    n = len(trades)
    wins = [t["ret"] for t in trades if t["ret"] > 0]
    losses = [t["ret"] for t in trades if t["ret"] <= 0]
    return {"n": n, "wins": len(wins), "ret": sum(t["ret"] for t in trades),
            "win": sum(wins) / len(wins) if wins else None, "loss": sum(losses) / len(losses) if losses else None}


def rsi(closes, n=14):
    if len(closes) <= n:
        return None
    gain = loss = 0.0
    for a, b in zip(closes[-n - 1:-1], closes[-n:]):
        gain += max(b - a, 0)
        loss += max(a - b, 0)
    return 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)


class Big4:
    """The four's book (the simulation's), their minutes and what the page shows. State under the Atlas key "big4"."""

    def __init__(self, atlas, folder, fetch=scalp.fetch_minutes):
        self.atlas, self.fetch = atlas, fetch
        self.bars = scalp.Bars(folder)
        self.lock = threading.RLock()
        self.days = {}  # symbol -> {day: [bars]} (the closed days, from the files)
        self.today = {}  # symbol -> Day
        self.today_bars = {}  # symbol -> [bars] of the session being read
        self.day = None
        self.status = {"ok": None, "error": None, "at": None, "source": "Yahoo"}
        self.record = {}  # symbol -> the rule over the recorded days (what it knows of each)
        self._polled = 0.0
        self._loaded = False

    # ---------- the stored state ----------
    def state(self):
        S = self.atlas._get("big4") or {}
        S.setdefault("settings", {})
        S["settings"].setdefault("mode", "sim")
        S["settings"].setdefault("size", SIZE)
        S.setdefault("trades", [])
        S.setdefault("open", {})
        return S

    def set(self, mode=None, size=None, confirm=False):
        S = self.state()
        if size is not None:
            size = float(size)
            if not 10 <= size <= SIZE_MAX:
                raise ValueError(f"גודל טרייד: בין $10 ל-${SIZE_MAX:,.0f}")
            S["settings"]["size"] = size
        if mode is not None:
            if mode not in ("sim", "live"):
                raise ValueError("מצב לא מוכר")
            if mode == "live":
                if not LIVE_OPEN:
                    raise ValueError("המסחר החי של 4 הגדולות עוד לא נפתח: קודם הסימולציה צוברת תוצאות, והוראות השורט והיציאה "
                                     "נבדקות מול IBKR מחובר. עד אז כל הטריידים כאן בסימולציה בלבד")
                if not confirm:
                    raise ValueError("מסחר בכסף אמיתי צריך את האישור שלך")
            S["settings"]["mode"] = mode
        self.atlas._set("big4", S)
        return S["settings"]

    # ---------- the minutes ----------
    def _load(self):
        """The recorded days, once: from the files, and from the source whatever is missing of the last weeks."""
        for day in self.bars.days()[-KEEP_DAYS:]:
            try:
                for sym, rows in self.bars.load(day).items():
                    if sym in SYMBOLS:
                        self.days.setdefault(sym, {})[day] = rows
            except Exception as e:
                log.warning("The four: a day's file (%s): %s", day, e)
        self._loaded = True

    def backfill(self, now=None):
        """The closed days the files lack: the last week, or as far back as the source keeps minutes (about four weeks,
        seven days a request) when little is recorded yet."""
        now = now or time.time()
        today = scalp.day_of(now) if self._session(now) else None  # (the session being made is kept at its close)
        got = {}
        for sym in SYMBOLS:
            have = self.days.setdefault(sym, {})
            end = now
            for _ in range(1 if len(have) >= LOOK + 5 else 4):
                try:
                    rows = self.fetch(sym, end - 7 * 86400, end)
                except Exception as e:
                    self.status.update(ok=False, error=f"{sym}: {type(e).__name__}: {e}"[:160])
                    break
                for b in rows:
                    d = scalp.day_of(b[0])
                    if d != today and d not in have:
                        got.setdefault(d, {}).setdefault(sym, []).append(tuple(b))
                end -= 7 * 86400
                time.sleep(0.3)
        for d, by in got.items():
            full = {s: rows for s, rows in by.items() if len(rows) >= 200}  # (a whole session, not a piece of one)
            if not full:
                continue
            try:
                old = self.bars.load(d) if d in self.bars.days() else {}
            except Exception:
                old = {}
            old.update(full)
            self.bars.save(d, old)
            for s, rows in full.items():
                self.days.setdefault(s, {})[d] = rows
        self._record()

    def _record(self):
        """What the rule would have made on each of the four over the recorded days: what it knows of each."""
        for sym in SYMBOLS:
            ds = self.days.get(sym) or {}
            days = [ds[d] for d in sorted(ds)]
            tr = replay(sym, days)
            self.record[sym] = dict(summary(tr), days=len(days))

    @staticmethod
    def _session(now):
        t = time.gmtime(now)
        return t.tm_wday < 5 and 0 <= minute(now) < 392

    def _start_day(self, day):
        self.day = day
        self.today, self.today_bars = {}, {}
        for sym in SYMBOLS:
            ds = self.days.get(sym) or {}
            before = [ds[d] for d in sorted(ds) if d < day]
            sigma = noise(before)
            if before and len(sigma) >= 300:
                self.today[sym] = Day(sym, before[-1][-1][4], sigma)

    def tick(self, now=None):
        """Once every few seconds: while the market is open the day's minutes are read and the rule looks at each new one."""
        now = now or time.time()
        with self.lock:
            if not self._loaded:
                self._load()
                self.backfill(now)
            if not self._session(now):
                if self.day and scalp.day_of(now) == self.day and minute(now) >= 395 and self.today_bars:
                    self._close_day(now)
                return
            if now - self._polled < POLL:
                return
            self._polled = now
            day = scalp.day_of(now)
            if day != self.day:
                self.backfill(now)
                self._start_day(day)
            start = now - (minute(now) + 5) * 60
            ok = True
            for sym in SYMBOLS:
                try:
                    rows = [tuple(b) for b in self.fetch(sym, start, now + 60) if scalp.day_of(b[0]) == day]
                except Exception as e:
                    ok = False
                    self.status.update(error=f"{sym}: {type(e).__name__}: {e}"[:160])
                    continue
                # (the minute still being made is left for the next read: only closed minutes are decided on)
                rows = [b for b in rows if b[0] + 60 <= now]
                self.today_bars[sym] = rows
                d = self.today.get(sym)
                if d is None:
                    continue
                for b in rows:
                    if minute(b[0]) > d.last_m:
                        what = d.feed(b)
                        if what:
                            self._book(sym, d, what)
            self.status.update(ok=ok, at=now, **({"error": None} if ok else {}))

    def _book(self, sym, d, what):
        S = self.state()
        size = float(S["settings"]["size"])
        kind, x = what
        if kind == "in":
            S["open"][sym] = {"side": x["side"], "entry": x["entry"], "in": x["ts"], "stop": x["stop"], "guard": x["guard"],
                              "shares": size / x["entry"], "size": size, "mode": "sim"}
            log.warning("THE FOUR (sim) %s %s at %.2f, stop %.2f", sym, "LONG" if x["side"] > 0 else "SHORT", x["entry"], x["stop"])
        else:
            o = S["open"].pop(sym, None) or {"shares": size / x["entry"], "size": size, "mode": "sim"}
            pnl = x["ret"] * o["shares"] * x["entry"]
            S["trades"].append({"symbol": sym, "side": x["side"], "entry": x["entry"], "exit": x["exit"], "in": x["in"], "out": x["out"],
                                "ret": x["ret"], "pnl": round(pnl, 2), "size": o["size"], "why": x["why"], "mode": o.get("mode", "sim")})
            S["trades"] = S["trades"][-TRADES_KEPT:]
            log.warning("THE FOUR (sim) %s out at %.2f: %+.2f%% $%+.2f (%s)", sym, x["exit"], x["ret"] * 100, pnl, x["why"])
        self.atlas._set("big4", S)

    def _close_day(self, now):
        """After the close: anything still open goes out at the last price, and the day's minutes are kept."""
        for sym, d in self.today.items():
            if d.pos and d.last:
                d._out(d.last[4], d.last[0], "close")
                self._book(sym, d, ("out", d.trades[-1]))
        by = {s: rows for s, rows in self.today_bars.items() if len(rows) >= 200}
        if by:
            try:
                old = self.bars.load(self.day) if self.day in self.bars.days() else {}
                old.update(by)
                self.bars.save(self.day, old)
                for s, rows in by.items():
                    self.days.setdefault(s, {})[self.day] = rows
                self._record()
            except Exception as e:
                log.warning("The four: keeping the day: %s", e)
        self.today_bars = {}

    def loop(self, stop):
        stop.wait(20)
        while not stop.is_set():
            try:
                self.tick()
            except Exception as e:
                log.warning("The four: %s", e)
                self.status.update(ok=False, error=f"{type(e).__name__}: {e}"[:160])
            stop.wait(5)

    # ---------- the page ----------
    def view(self, now=None):
        now = now or time.time()
        with self.lock:
            S = self.state()
            cards = []
            for sym in SYMBOLS:
                rows = self.today_bars.get(sym)
                day = self.day
                live_day = bool(rows)
                if not rows:  # (no session being read: the last recorded one)
                    ds = self.days.get(sym) or {}
                    day = max(ds) if ds else None
                    rows = ds.get(day) or []
                d = self.today.get(sym) if live_day else None
                if d is None and rows:  # (the recorded day, run again for its bands and marks)
                    ds = self.days.get(sym) or {}
                    before = [ds[x] for x in sorted(ds) if x < day]
                    sigma = noise(before)
                    if before and len(sigma) >= 300:
                        d = Day(sym, before[-1][-1][4], sigma)
                        for b in rows:
                            d.feed(b)
                pts, up, lo, vw = [], [], [], []
                if rows:
                    vsum = pv = 0.0
                    for b in rows:
                        m = minute(b[0])
                        vsum += b[5]
                        pv += (b[2] + b[3] + b[4]) / 3 * b[5]
                        pts.append([m, round(b[4], 4)])
                        vw.append([m, round(pv / vsum if vsum else b[4], 4)])
                        if d is not None:
                            u, l_ = d.bands(m)
                            if u is not None:
                                up.append([m, round(u, 4)])
                                lo.append([m, round(l_, 4)])
                trades = [t for t in S["trades"] if t["symbol"] == sym]
                marks = []
                for t in trades:
                    if day and scalp.day_of(t["in"]) == day:
                        marks.append({"m": minute(t["in"]), "price": t["entry"], "kind": "in", "side": t["side"]})
                        marks.append({"m": minute(t["out"]), "price": t["exit"], "kind": "out", "side": t["side"], "pnl": t["pnl"]})
                o = S["open"].get(sym)
                if o and day and scalp.day_of(o["in"]) == day:
                    marks.append({"m": minute(o["in"]), "price": o["entry"], "kind": "in", "side": o["side"]})
                last = rows[-1][4] if rows else None
                closes = [b[4] for b in rows]
                prev = d.prev_close if d else None
                u_now, l_now = (up[-1][1], lo[-1][1]) if up else (None, None)
                where = None
                if last is not None and u_now is not None and u_now > l_now:
                    where = (last - l_now) / (u_now - l_now)  # 0: on the lower band, 1: on the upper; outside: beyond the noise
                vol = sum(b[5] for b in rows)
                usual = None
                ds = self.days.get(sym) or {}
                past = [ds[x] for x in sorted(ds) if not day or x < day][-LOOK:]
                if rows and past:
                    m_now = minute(rows[-1][0])
                    same = [sum(b[5] for b in p if minute(b[0]) <= m_now) for p in past]
                    usual = sum(same) / len(same) if same else None
                cards.append({
                    "symbol": sym, "name": NAMES.get(sym, ""), "day": day, "live_day": live_day, "last": last, "prev": prev,
                    "change": (last / prev - 1) if last and prev else None, "points": pts, "up": up, "lo": lo, "vwap": vw,
                    "marks": marks, "where": where, "rsi": rsi(closes[::5]) if len(closes) > 80 else None,
                    "relvol": (vol / usual) if usual else None,
                    "open": dict(o, now=last, ret=(o["side"] * (last - o["entry"]) / o["entry"]) if last else None,
                                 pnl=round(o["side"] * (last - o["entry"]) * o["shares"], 2) if last else None) if o else None,
                    "total": dict(summary(trades), pnl=round(sum(t["pnl"] for t in trades), 2)),
                    "trades": trades[-60:][::-1], "known": self.record.get(sym), "ready": sym in self.today or not live_day,
                    "state": ("long" if o and o["side"] > 0 else "short" if o else
                              "above" if where is not None and where > 1 else "below" if where is not None and where < 0 else "inside"),
                })
            allt = S["trades"]
            return {"ts": now, "settings": S["settings"], "live_open": LIVE_OPEN, "session": self._session(now), "status": self.status,
                    "cards": cards, "total": dict(summary(allt), pnl=round(sum(t["pnl"] for t in allt), 2)),
                    "rule": {"band": BAND, "look": LOOK, "step": STEP, "guard": GUARD}}
