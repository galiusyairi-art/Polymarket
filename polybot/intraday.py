"""Atlas's experiments lab, experiment 2: intraday strategies on the index ETFs (SPY, QQQ), from the research.

The owner (28.10.2026): the intraday strategies that take a fair, realistic profit without needless risk, after
a wide search (papers, books, the field's best known traders) and four panels. What passed the filter, each
with its source, and each set before any data:

- The noise area (Zarattini, Aziz & Barbon, "Beat the Market", Swiss Finance Institute 2024): each minute's
  normal move from the open, over the last 14 days, around the day's open (the gap counted); at the hour and
  the half hour from 10:00, a price out of that area is a trend: in its direction, under a stop at the area's
  edge or the VWAP, whichever is nearer, out at the close. The paper: 2007-2024, 19.6% a year net, Sharpe 1.33;
  replications find less.
- The last half hour (Gao, Han, Li & Zhou, JFE 2018; Baltussen, Da, Lammers & Martens, JFE 2021): the day's
  move until 15:30 continues into the close (the option dealers' hedging and the late rebalancing). Small
  (R² 2-3%) but found in 60 markets over 46 years; stronger on the big days.
- The VWAP trend (Zarattini & Aziz 2023): long above the day's VWAP, out below. The paper's numbers didn't
  replicate; a band around the VWAP helped the replications: tested here with one, as the claim to check.

Every variant is judged as experiment 1's: $30 risked a trade (R), after the costs, by the day, against the
control (long at 10:00, out at the close, every day: does the timing add anything to just being in the market
during the day?). Rejected by the panels (the reasons are in the page's spec): the 5-minute ORB on QQQ (net
zero in replication), the gap fill (its fill rates are the day's range, not an edge), the indicator setups
without published evidence.
"""

from .scalp import NY, OPEN, R, commission, minute_of, sell_fees  # noqa: F401  (NY: the callers' clock)

ETFS = ("SPY", "QQQ")
NOISE_DAYS = 14
CHECKS = tuple(range(600, 960, 30))  # 10:00, 10:30, ... 15:30: the noise area's decisions
LAST_HALF = 930  # 15:30
CLOSE = 959  # the last bar, 15:59
NOTIONAL_CAP = 25_000.0
MIN_STOP = 0.001  # (a stop nearer than 0.1% is taken as 0.1%: never a giant position on a hair's distance)

VARIANTS = {
    "N0": {"kind": "control"},
    "N1": {"kind": "noise", "sides": ("long",)},
    "N2": {"kind": "noise", "sides": ("long", "short")},
    "N3": {"kind": "last", "sides": ("long",), "min_move": 0.0},
    "N4": {"kind": "last", "sides": ("long", "short"), "min_move": 0.005},
    "N5": {"kind": "vwap", "band": 0.001},
}
NAMES = {
    "N0": "ביקורת: קונים ב-10:00 ויוצאים בסגירה, כל יום (האם התזמון מוסיף משהו?)",
    "N1": "אזור הרעש: יציאה מעליו ← לונג, סטופ על הגבול או ה-VWAP, עד הסגירה (Zarattini, Aziz, Barbon 2024)",
    "N2": "אזור הרעש בשני הכיוונים, כמו במאמר (שורט רק בסימולציה)",
    "N3": "חצי השעה האחרונה: עלה מהסגירה של אתמול עד 15:30 ← לונג עד הסגירה (Gao ואח' 2018)",
    "N4": "חצי השעה האחרונה, רק בימים של תזוזה מעל 0.5%, בשני הכיוונים (Baltussen ואח' 2021)",
    "N5": "מגמת VWAP: מעל ה-VWAP ב-0.1% ← לונג, מתחת ← יוצאים (Zarattini, Aziz 2023: הטענה לבדיקה)",
}


def half_spread(price, minute):
    """An index ETF's: a cent wide (half a cent each way), double in the first five minutes, plus 0.5bp."""
    return (0.01 if minute < OPEN + 5 else 0.005) + 0.00005 * price


def by_minute(bars):
    return {minute_of(b[0]): b for b in bars}


def sigma(history, sym):
    """Each minute's normal distance from the open (|close / open - 1|), averaged over the days before:
    {minute: sigma}."""
    acc = {}
    for day in history[-NOISE_DAYS:]:
        bars = day.get(sym) or []
        if not bars:
            continue
        o = bars[0][1]
        for b in bars:
            acc.setdefault(minute_of(b[0]), []).append(abs(b[4] / o - 1))
    return {m: sum(x) / len(x) for m, x in acc.items() if len(x) >= 5}


class _T:
    __slots__ = ("sym", "side", "entry", "shares", "stop", "m_in", "best", "exit", "m_out", "why", "pnl", "r", "cost")


def _open(sym, side, price, stop, m):
    hs = half_spread(price, m)
    fill = price + hs if side == "long" else price - hs
    d = max(abs(fill - stop), MIN_STOP * fill)
    shares = int(min(R / d, NOTIONAL_CAP / fill))
    if shares < 1:
        return None
    t = _T()
    t.sym, t.side, t.entry, t.shares, t.m_in, t.best, t.why = sym, side, fill, shares, m, fill, None
    t.stop = fill - d if side == "long" else fill + d
    t.cost = commission(shares, fill) + (sell_fees(shares, fill) if side == "short" else 0.0)
    return t


def _close(t, price, m, why):
    hs = half_spread(price, m)
    fill = price - hs if t.side == "long" else price + hs
    t.exit, t.m_out, t.why = fill, m, why
    t.cost += commission(t.shares, fill) + (sell_fees(t.shares, fill) if t.side == "long" else 0.0)
    move = fill - t.entry if t.side == "long" else t.entry - fill
    t.pnl = move * t.shares - t.cost
    t.r = t.pnl / R


def _row(t):
    return {"symbol": t.sym, "side": t.side, "in": t.m_in, "out": t.m_out, "entry": round(t.entry, 2),
            "exit": round(t.exit, 2), "shares": t.shares, "stop0": round(t.stop, 2), "why": t.why,
            "pnl": round(t.pnl, 2), "r": round(t.r, 3), "cost": round(t.cost, 2)}


def _stop_hit(t, b):
    """The bar's own move against the open trade (the worse side first): the price it left at, or None."""
    _, o, h, lo, c, v = b
    if t.side == "long":
        if o <= t.stop:
            return o
        if lo <= t.stop:
            return t.stop
    else:
        if o >= t.stop:
            return o
        if h >= t.stop:
            return t.stop
    return None


def run_day(day, prev_day, history, P, sym):
    """One variant over one day of one ETF, decisions on closed bars, fills at the next bar's open:
    [closed trades]."""
    bars = day.get(sym) or []
    prev = (prev_day or {}).get(sym) or []
    if len(bars) < 300 or not prev:
        return []
    M = by_minute(bars)
    minutes = sorted(M)
    last = minutes[-1]
    day_open, prev_close = bars[0][1], prev[-1][4]
    vw, vv = 0.0, 0.0
    sig = sigma(history, sym) if P["kind"] == "noise" else {}
    t, out, pending = None, [], None  # (pending: a decision on this bar, filled at the next one's open)
    for i, m in enumerate(minutes):
        b = M[m]
        _, o, h, lo, c, v = b
        if pending:  # 1. the last bar's decision, at this bar's open
            what, side, stop = pending
            pending = None
            if what == "in" and t is None:
                t = _open(sym, side, o, stop if stop else (o * (1 - 0.004) if side == "long" else o * (1 + 0.004)), m)
            elif what == "out" and t is not None:
                _close(t, o, m, side)
                out.append(t)
                t = None
        if t is not None and m > t.m_in:  # 2. the stop, inside the bar
            hit = _stop_hit(t, b)
            if hit is not None:
                _close(t, hit, m, "stop")
                out.append(t)
                t = None
        vw += (h + lo + c) / 3 * v
        vv += v
        vwap = vw / vv if vv else c
        if m >= last or m >= CLOSE:  # 3. the close
            if t is not None:
                _close(t, c, m, "close")
                out.append(t)
                t = None
            break
        if P["kind"] == "control":
            if m == 599 and t is None:
                pending = ("in", "long", None)
        elif P["kind"] == "noise":
            s = sig.get(m)
            if s:
                up, down = max(day_open, prev_close) * (1 + s), min(day_open, prev_close) * (1 - s)
                if t is not None:  # (the stop trails: the area's edge or the VWAP, the nearer one, never back)
                    t.stop = max(t.stop, max(up, vwap)) if t.side == "long" else min(t.stop, min(down, vwap))
                elif m + 1 in CHECKS:  # (the bar closing at the hour / half hour)
                    if c > up and "long" in P["sides"]:
                        pending = ("in", "long", max(up, vwap))
                    elif c < down and "short" in P["sides"]:
                        pending = ("in", "short", min(down, vwap))
        elif P["kind"] == "last":
            if m + 1 == LAST_HALF and t is None:
                move = c / prev_close - 1
                if abs(move) >= P["min_move"] and move != 0:
                    side = "long" if move > 0 else "short"
                    if side in P["sides"]:
                        pending = ("in", side, None)
        elif P["kind"] == "vwap":
            band = P["band"]
            if m >= OPEN + 1:
                if t is None and c > vwap * (1 + band):
                    pending = ("in", "long", vwap * (1 - band))
                elif t is not None and c < vwap * (1 - band):
                    pending = ("out", "vwap", None)
    return [_row(x) for x in out]


def run_all(bars, symbols=ETFS):
    """Every variant over every recorded day (with the noise area's days before it), both ETFs: {variant:
    [trades with "day"]}, and the days run."""
    days = bars.days()
    loaded, out, run = {}, {k: [] for k in VARIANTS}, []
    get = lambda d: loaded.get(d) or loaded.setdefault(d, _etfs(bars.load(d), symbols))
    for i, d in enumerate(days):
        if i < 5:
            continue
        hist = [get(x) for x in days[max(0, i - NOISE_DAYS):i]]
        day, prev = get(d), get(days[i - 1])
        for k, P in VARIANTS.items():
            for sym in symbols:
                out[k] += [dict(r, day=d) for r in run_day(day, prev, hist, P, sym)]
        run.append(d)
        for x in list(loaded):
            if x < days[max(0, i - NOISE_DAYS)]:
                loaded.pop(x)
    return out, run


def _etfs(day, symbols):
    return {s: day[s] for s in symbols if s in day}


def board(results, days):
    from . import scalp
    return scalp.board(results, days, names=NAMES, control="N0")
