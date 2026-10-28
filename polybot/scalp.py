"""Atlas's experiments lab, experiment 1: the in-play opening-range break with a rising stop.

Designed by five expert panels and a plenary (27.10.2026) from the owner's idea: scan the market, find
the stocks in play, enter by the right rules, lose a fixed few tens of dollars when wrong, let a winner
run under a stop that rises with the price from the first moment, and get out of a trade that doesn't
move. The panels' honest verdict: entering all the time with tight stops is how day traders lose
(fixed costs of $3-6 a round trip are 10-30% of a $30 risk); what has a chance is selective: the day's
most active stocks, a break of the first five minutes' high, a stop sized by the stock's own movement.
Nothing here is proven: the lab exists to find out, against a random-entry placebo.

Phase 0 (this module now): 1-minute bars from Yahoo (about the last 30 days, then every day after the
close), stored as one file a day; eight variants, set before any data, run on them in event time with
honest fills (decisions on closed bars, the low first inside a bar, spreads and costs paid); results in
R (the $30 risked), each variant pooled over every stock, judged only at checkpoints of trades and days.
No verdict is possible yet with a month of bars and assumed spreads: phase 0 checks the plumbing and
the trade counts. Phase 1 (five years of minute history, real-time quotes) needs the owner's approval.
"""

import datetime as dt
import gzip
import json
import logging
import math
import os
import random
import time
import urllib.request
from zoneinfo import ZoneInfo

log = logging.getLogger("polybot")
NY = ZoneInfo("America/New_York")

R = 30.0  # dollars risked a trade (every variant: judged in R)
OPEN, ORB_END, ENTRY_END, FLAT = 570, 575, 630, 955  # minutes after midnight, New York: 9:30, 9:35, 10:30, 15:55
BLOCK = (599, 602)  # 9:59-10:02: no entries (the 10:00 releases)
BASE_DAYS = 14  # days of history for the normal volume and range
MAX_OPEN, MAX_OPEN_R, MAX_PER_MIN = 8, 6.0, 3
SOFT_LOSS, HARD_LOSS = -5.0, -10.0  # R a day: no new entries / out of everything
CHECKPOINTS = ((300, 20), (600, 40), (1200, 80), (2400, 160))  # (trades, days), both needed

# The variants, set before any data (the plenary's section 4). Each is the baseline but for what it lists.
BASE = {"entry": "orb", "window": (ORB_END, ENTRY_END), "stop_mult": 1.0, "stop_floor": 0.0025, "stop_cap": 0.01,
        "trail": "entry", "ladder": True, "time_stop": (10, 0.5), "stall": 20, "rvol": 2.0, "top": 20,
        "per_symbol": 1, "gap": 0, "hold": False, "placebo": False, "universe_only": False, "daily_atr": None,
        "spread_mult": 10}
VARIANTS = {
    "V0": dict(BASE, placebo=True),
    "V1": dict(BASE),
    "V2": dict(BASE, entry="high15", window=(ORB_END, 930), stop_mult=0.5, stop_floor=0.0015, ladder=False,
               time_stop=(2, 0.3), stall=None, per_symbol=3, gap=10, spread_mult=5),
    "V3": dict(BASE, trail="1R"),
    "V4": dict(BASE, time_stop=None, stall=None),
    "V5": dict(BASE, stop_mult=2.0),
    "V6": dict(BASE, universe_only=True),
    "V7": dict(BASE, daily_atr=0.10, trail=None, ladder=False, time_stop=None, stall=None, hold=True),
    # (28.10.2026, the traders' panel: Toby Crabel's narrow range, 1990: a quiet day before the break)
    "V8": dict(BASE, universe_only=True, nr7=True),
}
NAMES = {
    "V0": "פלצבו: אותן מניות ויציאות, כניסה אקראית (ביקורת)",
    "V1": "הבסיס: פריצת שיא 5 הדקות הראשונות, סטופ עולה מהכניסה, 10 דקות בלי תזוזה ← יוצאים",
    "V2": "הניסוח של הבעלים: כל היום, פריצת שיא 15 דקות, סטופ צמוד, יציאה אחרי 2 דקות בלי תזוזה",
    "V3": "הסטופ נגרר רק מרווח של פי 1 מהסיכון",
    "V4": "בלי 'לא זז ← יוצאים' (בודק אם הכלל שווה משהו)",
    "V5": "סטופ רחב: פי 2 מהתנודה הרגילה",
    "V6": "בלי הסורק: כל המניות של אטלס (בודק אם הסורק שווה משהו)",
    "V7": "העתק המחקר שפורסם: סטופ 10% מהתנודה היומית, עד הסגירה",
    "V8": "יום שקט לפני (NR7 של טובי קרבל): רק מניות שאתמול היה הטווח הכי צר ב-7 ימים",
}


# ---------- costs (IBKR fixed pricing and the regulators' fees, checked 27.10.2026) ----------
def commission(shares, price):
    return min(max(1.0, 0.005 * shares), 0.01 * shares * price)


def sell_fees(shares, price):
    return 20.60e-6 * shares * price + min(9.79, 0.000195 * shares)


def half_spread(price, minute):
    """The assumed half spread (phase 0 has no quotes): a large cap's, wider in the first quarter hour."""
    h = max(0.005, 0.0001 * price)
    return 2 * h if minute < OPEN + 15 else h


# ---------- the bars ----------
def minute_of(ts):
    t = dt.datetime.fromtimestamp(ts, NY)
    return t.hour * 60 + t.minute


def day_of(ts):
    return dt.datetime.fromtimestamp(ts, NY).date().isoformat()


def fetch_minutes(symbol, start, end, opener=None):
    """1-minute bars of the regular session between two times (Yahoo: at most 7 days a request, the last
    30 days or so): [(ts, o, h, l, c, v)]."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={int(start)}&period2={int(end)}"
           "&interval=1m&includePrePost=false")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (polybot research)"})
    with (opener or urllib.request.urlopen)(req, timeout=20) as r:
        data = json.loads(r.read())
    res = data["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    out = []
    for i, ts in enumerate(res.get("timestamp") or []):
        o, h, lo, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, lo, c):
            continue
        if OPEN <= minute_of(ts) < 960:
            out.append((int(ts), float(o), float(h), float(lo), float(c), float(v or 0)))
    return out


class Bars:
    """The recorded days: one gzip JSON file a day, {symbol: [[ts, o, h, l, c, v], ...]}, beside the
    database (not in it: a month of minutes for a hundred stocks is tens of megabytes)."""

    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def days(self):
        return sorted(f[:10] for f in os.listdir(self.folder) if f.endswith(".json.gz"))

    def load(self, day):
        with gzip.open(os.path.join(self.folder, f"{day}.json.gz"), "rt") as f:
            return {s: [tuple(b) for b in bars] for s, bars in json.load(f).items()}

    def save(self, day, by_symbol):
        path = os.path.join(self.folder, f"{day}.json.gz")
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump(by_symbol, f, separators=(",", ":"))
        os.replace(tmp, path)


def record(bars, symbols, now=None, fetch=fetch_minutes, pause=0.2):
    """The closed days not yet recorded, as far back as the source keeps minutes (about 29 days): each
    symbol fetched a week at a time. Returns the days recorded."""
    now = now or time.time()
    today = day_of(now)
    t = dt.datetime.fromtimestamp(now, NY)
    done_today = t.weekday() < 5 and (t.hour, t.minute) >= (16, 5)
    have = set(bars.days())
    start = now - 29 * 86400
    by_day = {}
    for sym in symbols:
        s = start
        while s < now:
            e = min(now, s + 7 * 86400)
            try:
                for b in fetch(sym, s, e):
                    d = day_of(b[0])
                    if d in have or (d == today and not done_today):
                        continue
                    by_day.setdefault(d, {}).setdefault(sym, []).append(b)
            except Exception as ex:
                log.warning("Lab bars %s: %s", sym, ex)
            s = e
            time.sleep(pause)
    most = max((len(by) for by in by_day.values()), default=0)
    for d, by in by_day.items():
        if len(by) >= 0.5 * most:  # (a day the source mostly missed isn't kept: it would mislead)
            bars.save(d, {s: sorted(set(map(tuple, b))) for s, b in by.items()})
    return sorted(by_day)


# ---------- the experiment ----------
def five_minute(bars):
    """1-minute bars into 5-minute ones (by the clock), [(minute, o, h, l, c, v)]."""
    out = {}
    for ts, o, h, lo, c, v in bars:
        m = minute_of(ts) // 5 * 5
        x = out.get(m)
        out[m] = (m, o, h, lo, c, v) if x is None else (m, x[1], max(x[2], h), min(x[3], lo), c, x[5] + v)
    return [out[m] for m in sorted(out)]


def baselines(history):
    """From the days before: each symbol's normal first-5-minute volume and its normal 5-minute range in
    the first hour (the opening ATR), {symbol: (volume, atr5)}."""
    vols, ranges = {}, {}
    for day in history[-BASE_DAYS:]:
        for sym, bars in day.items():
            five = five_minute(bars)
            if five and five[0][0] == OPEN:
                vols.setdefault(sym, []).append(five[0][5])
            prev = None
            for m, o, h, lo, c, v in five:
                if m >= ENTRY_END:
                    break
                tr = h - lo if prev is None else max(h, prev) - min(lo, prev)
                ranges.setdefault(sym, []).append(tr)
                prev = c
    return {s: (sum(vols[s]) / len(vols[s]), sum(ranges[s]) / len(ranges[s]))
            for s in vols if s in ranges and len(vols[s]) >= 5}


def scan(day, base, P, symbols=None, nr7=None):
    """The day's narrow net: the stocks whose first 5 minutes traded at least rvol times their normal
    volume, closed green; the top P["top"] by that. (V6: every stock, no filter.) {symbol: info}."""
    out = []
    for sym, bars in day.items():
        if symbols is not None and sym not in symbols or sym in ("SPY", "QQQ") or sym not in base:
            continue
        if P.get("nr7") and sym not in (nr7 or ()):
            continue
        first = [b for b in bars if minute_of(b[0]) < ORB_END]
        if len(first) < 3:
            continue
        o, h, c, v = first[0][1], max(b[2] for b in first), first[-1][4], sum(b[5] for b in first)
        if c < 10:
            continue
        rvol = v / base[sym][0] if base[sym][0] else 0
        if not P["universe_only"] and (rvol < P["rvol"] or c <= o):
            continue
        out.append({"symbol": sym, "rvol": rvol, "orh": h, "atr5": base[sym][1], "price": c, "green": c > o})
    out.sort(key=lambda x: -x["rvol"])
    return {x["symbol"]: x for x in (out if P["universe_only"] else out[:P["top"]])}


class Trade:
    __slots__ = ("sym", "entry", "shares", "d", "stop", "best", "t_in", "m_in", "last_best", "cost", "why", "exit",
                 "m_out", "pnl", "r", "trail_d", "full")

    def risk_left(self):
        return max(0.0, (self.entry - self.stop) * self.shares) / R


def run_day(day, base, P, daily_atr=None, symbols=None, seed=None, placebo_n=None, nr7=None, partial=False):
    """One variant over one day, in event time on the minute bars: [closed trade dicts]. partial: the day
    so far (livelab.py, in real time): the trades still open are returned as they stand ("open": True,
    their stop now, marked at the last price), not closed at the last bar."""
    net = scan(day, base, P, symbols, nr7)
    if not net:
        return []
    by_min = {}
    for sym in list(net) + ["SPY"]:
        for b in day.get(sym, []):
            by_min.setdefault(minute_of(b[0]), {})[sym] = b
    minutes = sorted(by_min)
    if not minutes:
        return []
    last_min = minutes[-1]
    flat = FLAT if partial else min(FLAT, last_min)  # (a half day closes early; the day so far isn't over)
    rng = random.Random(seed)
    plan = {}
    if P["placebo"]:
        # As many trades as the baseline made that day, in its narrow net, at random names and minutes.
        names = sorted(net)
        for _ in range(placebo_n or 0):
            plan.setdefault(rng.choice(names), []).append(rng.randrange(ORB_END, ENTRY_END))
    vwap = {s: [0.0, 0.0] for s in net}
    hi15 = {s: [] for s in net}
    tr1 = {s: [] for s in net}
    prev_c = {}
    spy = []
    open_, closed, entries, last_entry = [], [], {}, {}
    realized, stops_at, paused_until, stopped_new, flatten_all = 0.0, [], -1, False, False
    pending_exit = []  # (trade, why) to leave at the next bar's open
    for m in minutes:
        bars = by_min[m]
        if "SPY" in bars:
            spy.append((m, bars["SPY"][4]))
        # 1. Exits decided on the last bar, at this bar's open (the bot's own: time stop, stall, flatten).
        for t, why in pending_exit:
            b = bars.get(t.sym)
            if t in open_ and b:
                _close(t, b[1] - half_spread(b[1], m), m, why)
                open_.remove(t)
                closed.append(t)
                realized += t.r
        pending_exit = []
        # 2. The open trades on this bar: the low first (the stop), then the high raises the stop.
        for t in list(open_):
            b = bars.get(t.sym)
            if not b or m <= t.m_in:
                continue
            ts, o, h, lo, c, v = b
            hs = half_spread(o, m)
            hit = "trail" if t.stop > t.entry else "stop"
            if o <= t.stop:
                _close(t, o - hs - 0.0002 * o, m, hit)
            elif lo <= t.stop:
                _close(t, t.stop - hs - 0.0002 * t.stop, m, hit)
            if t.why:
                open_.remove(t)
                closed.append(t)
                realized += t.r
                if t.r <= -0.9:
                    stops_at.append(m)
                continue
            if h > t.best:
                t.best, t.last_best = h, m
            mfe = (t.best - t.entry) / t.d
            trail = t.trail_d
            if P["ladder"]:
                if mfe >= 4:
                    trail = 0.5 * t.d
                elif mfe >= 2:
                    trail = 0.75 * t.d
                if mfe >= 1:
                    t.stop = max(t.stop, t.entry + 0.1 * R / t.shares)
                floor = max(2 * _mean(tr1[t.sym][-14:]), 0.0015 * c, 10 * half_spread(c, m))
                trail = max(trail, floor) if mfe >= 2 else trail
            if P["trail"] == "entry" or (P["trail"] == "1R" and mfe >= 1):
                t.stop = max(t.stop, t.best - trail)
            if P["time_stop"] and m - t.m_in >= P["time_stop"][0] and mfe < P["time_stop"][1] and not t.full:
                pending_exit.append((t, "time"))
                t.full = True
            elif P["stall"] and m - t.last_best >= P["stall"] and (c - t.entry) * t.shares < R and not t.full:
                pending_exit.append((t, "stall"))
                t.full = True
            elif m >= flat - 1 and not t.full:
                pending_exit.append((t, "close"))
                t.full = True
        # 3. The day's limits: marked to market at -10R everything goes; at -5R realized, no new entries.
        marked = realized + sum((bars[t.sym][4] - t.entry) * t.shares / R for t in open_ if t.sym in bars)
        if marked <= HARD_LOSS and not flatten_all:
            flatten_all = True
            pending_exit += [(t, "daily limit") for t in open_ if not t.full]
            for t in open_:
                t.full = True
        if realized <= SOFT_LOSS:
            stopped_new = True
        recent = [x for x in stops_at if m - x <= 15]
        if len(recent) >= 3:
            paused_until, stops_at = m + 30, []
        spy_drop = len(spy) >= 16 and spy[-1][1] / spy[-16][1] - 1 <= -0.005
        # 4. Entries on this bar's prices (signals from the bars before it).
        can = (not stopped_new and not flatten_all and m >= P["window"][0] and m < P["window"][1]
               and not (BLOCK[0] <= m < BLOCK[1]) and m >= paused_until and not spy_drop)
        this_min = 0
        for sym, info in net.items():
            b = bars.get(sym)
            if not b:
                continue
            ts, o, h, lo, c, v = b
            if can and this_min < MAX_PER_MIN and len(open_) < MAX_OPEN and entries.get(sym, 0) < P["per_symbol"] \
                    and m - last_entry.get(sym, -999) >= P["gap"] and not any(t.sym == sym for t in open_):
                trigger = None
                if P["placebo"]:
                    if m in plan.get(sym, []):
                        trigger = o
                elif P["entry"] == "orb":
                    level = info["orh"] + 0.01
                    if h >= level and prev_c.get(sym, 0) > _vwap(vwap[sym]):
                        trigger = level
                elif P["entry"] == "high15" and len(hi15[sym]) >= 15:
                    level = max(hi15[sym][-15:]) + 0.01
                    if h >= level:
                        trigger = level
                if trigger:
                    t = _enter(sym, trigger, o, m, info, P, daily_atr)
                    if t and sum(x.risk_left() for x in open_) + t.risk_left() <= MAX_OPEN_R:
                        open_.append(t)
                        entries[sym] = entries.get(sym, 0) + 1
                        last_entry[sym] = m
                        this_min += 1
            # (the bar's own numbers, for the next minute's decisions)
            vw = vwap[sym]
            vw[0] += (h + lo + c) / 3 * v
            vw[1] += v
            hi15[sym].append(h)
            p = prev_c.get(sym)
            tr1[sym].append(h - lo if p is None else max(h, p) - min(lo, p))
            prev_c[sym] = c
    if partial:
        rows = [_row(t) for t in closed]
        for t in open_:
            b = day.get(t.sym, [])[-1:]
            px = b[0][4] if b else t.entry
            rows.append({"symbol": t.sym, "in": t.m_in, "out": None, "entry": round(t.entry, 2), "exit": None,
                         "shares": t.shares, "stop0": round(t.entry - t.d, 2), "stop": round(t.stop, 2),
                         "best": round(t.best, 2), "why": None, "open": True, "exiting": bool(t.full),
                         "pnl": round((px - t.entry) * t.shares - t.cost, 2), "r": round(((px - t.entry) * t.shares - t.cost) / R, 3),
                         "cost": round(t.cost, 2)})
        return rows
    for t in open_:  # (what's still open at the last bar: out at its close)
        b = day.get(t.sym, [])[-1:]
        if b:
            _close(t, b[0][4] - half_spread(b[0][4], last_min), last_min, "close")
            closed.append(t)
    return [_row(t) for t in closed]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _vwap(vw):
    return vw[0] / vw[1] if vw[1] else 0.0


def _enter(sym, trigger, o, m, info, P, daily_atr):
    """A fill at the trigger (or the bar's open when it gapped through), plus half the spread and a basis
    point; the order's limit is the trigger plus max(2c, 0.05%): above it, no fill. Sized to R."""
    price = max(trigger, o)
    hs = half_spread(price, m)
    fill = price + hs + 0.0001 * price
    if not P["placebo"] and fill > trigger + max(0.02, 0.0005 * trigger) + hs:
        return None  # (gapped past the order's limit: a real order wouldn't have filled)
    if P["daily_atr"]:
        if not (daily_atr or {}).get(sym):
            return None
        d = max(P["daily_atr"] * daily_atr[sym], 0.05)  # (the published rule: no floor or cap but a nickel)
    else:
        d = P["stop_mult"] * info["atr5"]
        d = min(max(d, P["stop_floor"] * fill, 2 * P["spread_mult"] * hs, 0.05), P["stop_cap"] * fill)
    shares = int(R / (d + hs + 0.01))
    shares = min(shares, int(10_000 / fill))
    if shares < 1 or shares * d < 0.6 * R:
        return None
    cost = 2 * commission(shares, fill) + shares * 2 * hs + sell_fees(shares, fill)
    if cost > 0.15 * R:
        return None  # (the cost gate: more than 15% of the risk goes to costs)
    t = Trade()
    t.sym, t.entry, t.shares, t.d, t.stop, t.best = sym, fill, shares, d, fill - d, fill
    t.t_in, t.m_in, t.last_best, t.why, t.exit, t.m_out, t.full = None, m, m, None, None, None, False
    t.trail_d = d
    t.cost = commission(shares, fill)
    return t


def _close(t, price, m, why):
    t.exit, t.m_out, t.why = price, m, why
    t.cost += commission(t.shares, price) + sell_fees(t.shares, price)
    t.pnl = (price - t.entry) * t.shares - t.cost
    t.r = t.pnl / R


def _row(t):
    return {"symbol": t.sym, "in": t.m_in, "out": t.m_out, "entry": round(t.entry, 2), "exit": round(t.exit, 2),
            "shares": t.shares, "stop0": round(t.entry - t.d, 2), "best": round(t.best, 2), "why": t.why,
            "pnl": round(t.pnl, 2), "r": round(t.r, 3), "cost": round(t.cost, 2)}


def run_all(bars, daily_atr=None, symbols=None, nr7=None):
    """Every variant over every recorded day with a few before it: {variant: [trades with "day"]}, and
    the days run. Deterministic (the placebo's seed is the date). daily_atr: {day: {symbol: the daily ATR
    as of the day before}} (V7)."""
    days = bars.days()
    loaded = {}
    out = {k: [] for k in VARIANTS}
    run = []
    for i, d in enumerate(days):
        if i < 5:
            continue  # (too little history for the normal volume and range)
        hist = [loaded.get(x) or loaded.setdefault(x, bars.load(x)) for x in days[max(0, i - BASE_DAYS):i]]
        day = loaded.get(d) or loaded.setdefault(d, bars.load(d))
        base = baselines(hist)
        atr = (daily_atr or {}).get(d)
        n_base = None
        for k in ("V1",) + tuple(x for x in VARIANTS if x != "V1"):
            P = VARIANTS[k]
            rows = run_day(day, base, P, atr, symbols if P["universe_only"] else None, seed=d,
                           placebo_n=n_base, nr7=(nr7 or {}).get(d))
            if k == "V1":
                n_base = len(rows)
            out[k] += [dict(r, day=d) for r in rows]
        run.append(d)
        for x in list(loaded):  # (keep only the window in memory)
            if x < days[max(0, i - BASE_DAYS)]:
                loaded.pop(x)
    return out, run


def day_z(trades, days):
    """Mean daily net R over its standard error (every day counted, a day without trades as 0)."""
    if len(days) < 2:
        return None
    per = {d: 0.0 for d in days}
    for t in trades:
        per[t["day"]] = per.get(t["day"], 0.0) + t["r"]
    xs = list(per.values())
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean / math.sqrt(var / len(xs)) if var > 0 else None


def board(results, days, names=None, control="V0"):
    """One row a variant: the trades, wins, average win and loss, money and R after costs, the day-z, the
    paired day-z against the placebo, and where it stands (collecting until the first checkpoint)."""
    names = names or NAMES
    placebo = results.get(control, [])
    rows = []
    for k, trades in results.items():
        n = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        z = day_z(trades, days)
        diff = None
        if k != control and len(days) >= 2:
            per = {d: 0.0 for d in days}
            for t in trades:
                per[t["day"]] += t["r"]
            for t in placebo:
                per[t["day"]] -= t["r"]
            xs = list(per.values())
            mean = sum(xs) / len(xs)
            var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
            diff = mean / math.sqrt(var / len(xs)) if var > 0 else None
        reached = [c for c in CHECKPOINTS if n >= c[0] and len(days) >= c[1]]
        nxt = next((c for c in CHECKPOINTS if c not in reached), None)
        why = {}
        for t in trades:
            why[t["why"]] = why.get(t["why"], 0) + 1
        rows.append({"variant": k, "rule": names[k], "trades": n, "days": len(days),
                     "win_rate": len(wins) / n if n else None,
                     "avg_win_r": _mean([t["r"] for t in wins]) if wins else None,
                     "avg_loss_r": _mean([t["r"] for t in losses]) if losses else None,
                     "pnl": round(sum(t["pnl"] for t in trades), 2), "r": round(sum(t["r"] for t in trades), 2),
                     "r_per_trade": _mean([t["r"] for t in trades]) if n else None,
                     "cost_r": _mean([t["cost"] / R for t in trades]) if n else None,
                     "day_z": z, "vs_placebo": diff, "exits": why,
                     "status": "collecting" if not reached else "undecided",
                     "next": {"trades": nxt[0], "days": nxt[1]} if nxt else None})
    return rows
