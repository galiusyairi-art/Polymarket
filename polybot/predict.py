"""Predicting the move (the owner's idea of 27.10.2026): does the way the price is moving, or the way
people are buying, say who will win a round better than Polymarket's own price already does?

Two signals, set before any data, each read at three moments before the close (2 minutes, 1 minute,
30 seconds left) over the minute before that moment:
  - mom: the coin's own move (its price then against a minute earlier);
  - flow: who was buying: every trade in the round's market that minute, signed (someone buying Up or
    selling Down pushes Up; buying Down or selling Up pushes Down), in shares.
Two questions for each:
  - does it know more than the market? following it against the market's middle price (win minus the
    price) should be worth more than nothing;
  - does it pay? buying the side it points to, at the ask, the taker fee included; on the strongest
    third of its readings (a weak reading is mostly noise).
Rounds that end together move together (the coins follow each other): the score counts each closing
moment once (clustered), and a finding has to hold in the older half and the newer half alike. Six
tests: an edge needs a score of 3 (a lucky one among six reaches it about 1 time in 100).

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
FEATURES = ("mom", "flow")
NEED_DAYS = 7
Z_FOUND = 3.0
Z_PROMISING = 2.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS pred_rows (
    asset TEXT, minutes INTEGER, window_start INTEGER, moment INTEGER, end_ts INTEGER,
    up_bid REAL, up_ask REAL, down_bid REAL, down_ask REAL, mom REAL, flow REAL, winner TEXT,
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


def record(db, limit=3000):
    """The decided rounds not read yet: their readings into pred_rows (kept for good). Returns how many
    rounds were read."""
    db.executescript(SCHEMA)
    todo = db.execute(
        "SELECT w.asset, w.minutes, w.window_start, w.winner FROM windows w LEFT JOIN pred_done d ON d.asset = w.asset"
        " AND d.minutes = w.minutes AND d.window_start = w.window_start WHERE d.asset IS NULL AND w.winner IN ('up', 'down')"
        " ORDER BY w.window_start LIMIT ?", (limit,)).fetchall()
    for asset, minutes, start, winner in [tuple(r) for r in todo]:
        key = (asset, minutes, start)
        ticks = [tuple(r) for r in db.execute(
            "SELECT seconds_left, price, up_bid, up_ask, down_bid, down_ask FROM ticks WHERE asset = ? AND minutes = ?"
            " AND window_start = ?", key)]
        trades = [tuple(r) for r in db.execute(
            "SELECT seconds_left, outcome, size, taker FROM trades WHERE asset = ? AND minutes = ? AND window_start = ?", key)]
        for m, (ub, ua, db_, da, mom, flow) in readings(ticks, trades, minutes).items():
            db.execute("INSERT OR REPLACE INTO pred_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (asset, minutes, start, m, start + minutes * 60, ub, ua, db_, da, mom, flow, winner))
        db.execute("INSERT OR REPLACE INTO pred_done VALUES (?, ?, ?)", key)
    db.commit()
    return len(todo)


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
    db.executescript(SCHEMA)
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
