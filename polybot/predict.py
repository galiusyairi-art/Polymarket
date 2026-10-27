"""Predicting the move (the owner's idea of 27.10.2026): does the way the price is moving, or the way
people are buying, say who will win a round better than Polymarket's own price already does?

Signals set before any data, each read at three moments before the close (2 minutes, 1 minute, 30
seconds left):
  - mom: the coin's own move (its price then against a minute earlier);
  - flow: who was buying: every trade in the round's market that minute, signed (someone buying Up or
    selling Down pushes Up; buying Down or selling Up pushes Down), in shares;
  - brk15, brk60 (the owner's idea of 28.10.2026, what works in stocks): a breakout: the price over the
    last 15 minutes' (or hour's) high, until the minute before, or under its low; how far, signed.
Two questions for each:
  - does it know more than the market? following it against the market's middle price (win minus the
    price) should be worth more than nothing;
  - does it pay? buying the side it points to, at the ask, the taker fee included; on the strongest
    third of its readings (a weak reading is mostly noise).
Rounds that end together move together (the coins follow each other): the score counts each closing
moment once (clustered), and a finding has to hold in the older half and the newer half alike. Twelve
tests: an edge needs a score of 3 (a lucky one among twelve reaches it about 1 time in 50).

The recording keeps only the last weeks of prices and trades (cfg.keep_ticks_days), so each decided
round's readings are kept in a table of their own (pred_rows) as soon as it ends: the study keeps
everything it has seen, beyond what the recording keeps. Technical shapes (support, resistance,
Fibonacci) wait: in a 5- or 15-minute round they are mostly the same move again, and many shapes
tested at once would find something by chance; they come in only if these show a signal.
"""

import math

from .model import taker_fee_per_share

MOMENTS = (120, 60, 30)  # seconds left
LOOKBACK = 60  # seconds before each moment
NEAR = 4  # a reading counts within this many seconds of its moment (lab-only markets record every 5 s)
FEATURES = ("mom", "flow", "brk15", "brk60")
BREAKOUTS = (15, 60)  # minutes of price behind a breakout
NEED_DAYS = 7
Z_FOUND = 3.0
Z_PROMISING = 2.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS pred_rows (
    asset TEXT, minutes INTEGER, window_start INTEGER, moment INTEGER, end_ts INTEGER,
    up_bid REAL, up_ask REAL, down_bid REAL, down_ask REAL, mom REAL, flow REAL, winner TEXT,
    brk15 REAL, brk60 REAL,
    PRIMARY KEY (asset, minutes, window_start, moment)
);
CREATE TABLE IF NOT EXISTS pred_done (asset TEXT, minutes INTEGER, window_start INTEGER, PRIMARY KEY (asset, minutes, window_start));
"""


def _near(rows, at):
    best = None
    for r in rows:
        d = abs(r[0] - at)
        if d <= NEAR and (best is None or d < best[0]):
            best = (d, r)
    return best and best[1]


def readings(ticks, trades, minutes):
    """One round's readings: {moment: (up_bid, up_ask, down_bid, down_ask, mom, flow)}.
    ticks: (seconds_left, price, up_bid, up_ask, down_bid, down_ask); trades: (seconds_left, outcome,
    size, taker)."""
    out = {}
    for m in MOMENTS:
        if m + LOOKBACK > minutes * 60:
            continue
        now, back = _near(ticks, m), _near(ticks, m + LOOKBACK)
        if not now or not all(x and 0 < x < 1 for x in (now[2], now[3], now[4], now[5])):
            continue
        mom = now[1] / back[1] - 1 if back and back[1] and now[1] else None
        flow = 0.0
        for sl, outcome, size, taker in trades:
            if m <= sl < m + LOOKBACK and size:
                up = (outcome == "up") == (taker == "BUY")
                flow += size if up else -size
        out[m] = (now[2], now[3], now[4], now[5], mom, flow)
    return out


def breakout(prices, at, minutes):
    """prices: (ts, high, low, last) buckets, oldest first. How far the price at `at` stands over the high
    of the `minutes` before the last minute (positive), or under its low (negative); 0 inside; None
    without the history."""
    prior = [p for p in prices if at - minutes * 60 <= p[0] < at - 60]
    now = [p for p in prices if at - 60 <= p[0] <= at]
    if len(prior) < minutes * 6 or not now:  # (at least half the buckets: a real history)
        return None
    hi, lo, last = max(p[1] for p in prior), min(p[2] for p in prior), now[-1][3]
    return last / hi - 1 if last > hi else last / lo - 1 if last < lo else 0.0


def _schema(db):
    db.executescript(SCHEMA)
    cols = {r[1] for r in db.execute("PRAGMA table_info(pred_rows)")}
    if "brk15" not in cols:  # (an older table: the breakouts added, the rounds still recorded read again)
        import time as _t
        db.execute("ALTER TABLE pred_rows ADD COLUMN brk15 REAL")
        db.execute("ALTER TABLE pred_rows ADD COLUMN brk60 REAL")
        # (by the round's start alone, never a scan of the recording: it holds the database while it runs)
        db.execute("DELETE FROM pred_done WHERE window_start >= ?", (_t.time() - 13 * 86400,))
        db.commit()


def record(db, limit=200, budget=3.0):
    """The decided rounds not read yet: their readings into pred_rows (kept for good). Returns how many
    rounds were read. Short: at most `limit` rounds and about `budget` seconds a call, everything read
    first and each round written and committed on its own, so the database is never held for long (the
    bot writes to it all the time)."""
    import time as _t
    started = _t.time()
    _schema(db)
    todo = [tuple(r) for r in db.execute(
        "SELECT w.asset, w.minutes, w.window_start, w.winner FROM windows w LEFT JOIN pred_done d ON d.asset = w.asset"
        " AND d.minutes = w.minutes AND d.window_start = w.window_start WHERE d.asset IS NULL AND w.winner IN ('up', 'down')"
        " ORDER BY w.window_start LIMIT ?", (limit,)).fetchall()]
    # (each coin's price over the batch, in 5-second buckets, read once: its highs and lows, for the breakouts)
    history = {}
    for asset in {r[0] for r in todo}:
        ends = [start + minutes * 60 for a, minutes, start, _ in todo if a == asset]
        history[asset] = [tuple(r) for r in db.execute(
            "SELECT CAST(ts / 5 AS INTEGER) * 5 AS b, MAX(price), MIN(price), price FROM ticks WHERE asset = ? AND ts BETWEEN ? AND ?"
            " AND price > 0 GROUP BY b ORDER BY b", (asset, min(ends) - max(MOMENTS) - max(BREAKOUTS) * 60, max(ends)))]
    import bisect
    done = 0
    for asset, minutes, start, winner in todo:
        key = (asset, minutes, start)
        ticks = [tuple(r) for r in db.execute(
            "SELECT seconds_left, price, up_bid, up_ask, down_bid, down_ask FROM ticks WHERE asset = ? AND minutes = ?"
            " AND window_start = ?", key)]
        trades = [tuple(r) for r in db.execute(
            "SELECT seconds_left, outcome, size, taker FROM trades WHERE asset = ? AND minutes = ? AND window_start = ?", key)]
        got = readings(ticks, trades, minutes)
        end = start + minutes * 60
        H = history.get(asset) or []
        times = [p[0] for p in H]
        prices = H[bisect.bisect_left(times, end - max(MOMENTS) - max(BREAKOUTS) * 60):bisect.bisect_right(times, end)]
        for m, (ub, ua, db_, da, mom, flow) in got.items():
            b15, b60 = (breakout(prices, end - m, n) for n in BREAKOUTS)
            db.execute("INSERT OR REPLACE INTO pred_rows (asset, minutes, window_start, moment, end_ts, up_bid, up_ask, down_bid,"
                       " down_ask, mom, flow, winner, brk15, brk60) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (asset, minutes, start, m, end, ub, ua, db_, da, mom, flow, winner, b15, b60))
        db.execute("INSERT OR REPLACE INTO pred_done VALUES (?, ?, ?)", key)
        db.commit()
        done += 1
        if _t.time() - started > budget:
            break
    return done


def _score(points):
    """points: (cluster, value). The mean, and its score counting each cluster once."""
    if not points:
        return None, None
    n = len(points)
    mean = sum(v for _, v in points) / n
    by = {}
    for c, v in points:
        s = by.setdefault(c, [0.0, 0])
        s[0] += v
        s[1] += 1
    var = sum((s - k * mean) ** 2 for s, k in by.values())
    return mean, (mean * n / math.sqrt(var) if var > 0 else None)


def study(db, fee_rate=0.07, fee_exponent=1.0, since=0):
    """Every signal at every moment over the readings since `since` (the engine: the last 90 days)."""
    _schema(db)
    cur = db.execute("SELECT * FROM pred_rows WHERE end_ts >= ?", (since,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, tuple(r))) for r in cur]
    if not rows:
        return {"rounds": 0, "days": 0.0, "need_days": NEED_DAYS, "rows": [], "state": "collecting", "progress": 0.0}
    ends = sorted(r["end_ts"] for r in rows)
    days = (ends[-1] - ends[0]) / 86400
    half = ends[len(ends) // 2]
    out = []
    for f in FEATURES:
        for m in MOMENTS:
            got = [r for r in rows if r["moment"] == m and r[f]]
            # the strongest third of the readings, within each market (a coin's moves and trades differ in size)
            cut = {}
            for k in {(r["asset"], r["minutes"]) for r in got}:
                xs = sorted(abs(r[f]) for r in got if (r["asset"], r["minutes"]) == k)
                cut[k] = xs[(2 * len(xs)) // 3] if xs else 0
            edge, pay, old, new = [], [], [], []
            for r in got:
                s = 1 if r[f] > 0 else -1
                up = 1.0 if r["winner"] == "up" else 0.0
                edge.append((r["end_ts"], s * (up - (r["up_bid"] + r["up_ask"]) / 2)))
                if abs(r[f]) >= cut[(r["asset"], r["minutes"])]:
                    ask = r["up_ask"] if s > 0 else r["down_ask"]
                    won = up if s > 0 else 1 - up
                    p = won - ask - taker_fee_per_share(ask, fee_rate, fee_exponent)
                    pay.append((r["end_ts"], p))
                    (old if r["end_ts"] < half else new).append((r["end_ts"], p))
            e_mean, e_z = _score(edge)
            p_mean, p_z = _score(pay)
            out.append({"feature": f, "moment": m, "n": len(edge), "edge": e_mean, "edge_z": e_z,
                        "strong": len(pay), "pay": p_mean, "pay_z": p_z,
                        "old": _score(old)[0], "new": _score(new)[0]})
    enough = days >= NEED_DAYS
    holds = lambda r: (r["old"] or 0) > 0 and (r["new"] or 0) > 0
    if enough and any((r["pay_z"] or 0) >= Z_FOUND and holds(r) for r in out):
        state = "found"
    elif enough and any((r["edge_z"] or 0) >= Z_FOUND for r in out):
        state = "real_not_paying"  # (it knows more than the market, not enough to pay the fee)
    elif days >= 1 and any((r["pay_z"] or 0) >= Z_PROMISING or (r["edge_z"] or 0) >= Z_FOUND for r in out):
        state = "promising"
    elif enough:
        state = "nothing"
    else:
        state = "collecting"
    best = max(out, key=lambda r: r["edge_z"] or -9)
    return {"rounds": len({(r["asset"], r["minutes"], r["window_start"]) for r in rows}), "days": days,
            "need_days": NEED_DAYS, "rows": out, "best": {k: best[k] for k in ("feature", "moment")},
            "state": state, "progress": min(1.0, days / NEED_DAYS)}
