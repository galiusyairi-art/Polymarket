"""Why orders don't go through, and whether a little room would pay (the owner's questions of 28.10.2026).

Real money buys fill-or-kill at the price the strategy saw: all of it, or nothing. Half a second later,
when the order reaches Polymarket, the shares may be gone. The lab and the simulation send their orders
the same way (engine._arrive), and every order the book refused is kept with what was left at its
price by then and what 1 or 2 cents more would have filled (store.fill_tests). Two answers:

  - why: the shares all gone at that price (others were faster: a smaller amount wouldn't help), or
    some left but fewer than wanted (a smaller amount would have gone through);
  - room: the orders that 1 or 2 cents more would have filled, settled: do they make money after the
    dearer price and the fee? On the simulation's own orders (the strategies that run, real money's
    twins), judged when there are enough of them; the lab's, all its rules, shown beside.
"""

import math

MIN_SHARES = 5  # Polymarket's least order: what a smaller amount could still buy
ROOMS = (1, 2, 3, 5)  # cents (3 and 5 measured from 29.10.2026: the rows marked wide)
CHEAP = 0.20  # a price under it: a cheap side, whose win pays 4x and more (the owner's question of 29.10.2026)
NEED = 100  # refused orders a room would have filled, settled, before a verdict on it
Z = 2.0
NEAR_MS = 2.0  # a server in the same data centre as Polymarket's order book: the round trip left there


def _score(points):
    """points: (cluster, pnl, paid). The return a dollar, and its score counting each round once."""
    if not points:
        return None, None
    paid = sum(p for _, _, p in points)
    pnl = sum(x for _, x, _ in points)
    if paid <= 0:
        return None, None
    roi = pnl / paid
    by = {}
    for c, x, p in points:
        s = by.setdefault(c, [0.0, 0.0])
        s[0] += x
        s[1] += p
    var = sum((x - roi * p) ** 2 for x, p in by.values())
    return roi, (pnl - 0) / math.sqrt(var) if var > 0 else None


def study(db, since=0, live_since=None):
    rows = [tuple(r) for r in db.execute(
        "SELECT f.mode, f.side, f.want, f.left_at_limit, f.cost1, f.fee1, f.cost2, f.fee2, w.winner,"
        " f.asset || f.minutes || f.window_start, f.cost3, f.fee3, f.cost5, f.fee5, f.wide, f.limit_price"
        " FROM fill_tests f LEFT JOIN windows w ON w.asset = f.asset"
        " AND w.minutes = f.minutes AND w.window_start = f.window_start WHERE f.ts >= ?", (since,))]
    sent = {m: n for m, n in db.execute(
        "SELECT mode, COUNT(*) FROM bets WHERE mode IN ('lab', 'paper') AND status IN ('filled', 'rejected')"
        " AND ts >= ? GROUP BY mode", (since,))}
    live = db.execute("SELECT COUNT(*), SUM(status = 'rejected') FROM bets WHERE mode = 'live' AND ts >= ?",
                      (live_since or since,)).fetchone()
    # Why: every refused order of the lab and the simulation.
    n = len(rows)
    gone = sum(1 for r in rows if r[3] <= 1e-9)
    some = sum(1 for r in rows if 1e-9 < r[3] < r[2])
    least = sum(1 for r in rows if r[3] >= MIN_SHARES)
    why = {"refused": n, "gone": gone, "some_left": some, "least_would_fill": least,
           "sent": sum(sent.values()), "refused_share": n / sum(sent.values()) if sum(sent.values()) else None,
           "live_sent": live[0] or 0, "live_refused": live[1] or 0,
           "state": "collecting" if n < 50 else "found",
           "verdict": None if n < 50 else ("speed" if gone >= 0.6 * n else "size" if least >= 0.5 * n else "both")}
    # Room: what 1, 2, 3 or 5 cents more would have filled, settled (3 and 5: only the orders they were tried on).
    cols = {1: (4, 5), 2: (6, 7), 3: (10, 11), 5: (12, 13)}

    def _room(rs, k):
        ci, fi = cols[k]
        rs = [r for r in rs if k <= 2 or r[14]]
        pts = [(r[9], r[2] * (r[8] == r[1]) - r[ci] - r[fi], r[ci] + r[fi]) for r in rs
               if r[ci] is not None and r[8] in ("up", "down")]
        roi, z = _score(pts)
        return {"cents": k, "settled": len(pts), "would_fill": sum(1 for r in rs if r[ci] is not None), "of": len(rs),
                "pnl": sum(p[1] for p in pts), "roi": roi, "z": z,
                "wins": sum(1 for r in rs if r[ci] is not None and r[8] == r[1])}

    room = {}
    for k in ROOMS:
        for who in ("paper", "lab"):
            room[f"{who}{k}"] = dict(_room([r for r in rows if r[0] == who], k), who=who)
    # The cheap sides (under 20 cents): the simulation's refused orders there, how many of them won (the big
    # wins real money missed), and what each room would have filled of them and made.
    cheap_rows = [r for r in rows if r[0] == "paper" and r[15] is not None and r[15] < CHEAP]
    cheap = {"refused": len(cheap_rows), "settled": sum(1 for r in cheap_rows if r[8] in ("up", "down")),
             "won": sum(1 for r in cheap_rows if r[8] == r[1]),
             "rooms": {str(k): _room(cheap_rows, k) for k in ROOMS}}
    best = max((room[f"paper{k}"] for k in (1, 2)), key=lambda x: x["z"] if x["z"] is not None else -9)
    enough = best["settled"] >= NEED
    room_state = ("found" if enough and (best["z"] or 0) >= Z and (best["roi"] or 0) > 0 else
                  "nothing" if enough and (best["roi"] or 0) <= 0 else
                  "promising" if (best["roi"] or 0) > 0 and best["settled"] >= NEED // 4 else "collecting")
    return {"why": why, "room": room, "cheap": cheap, "room_state": room_state, "best": best["cents"],
            "room_progress": min(1.0, best["settled"] / NEED), "why_progress": min(1.0, n / 50), "need": NEED}


REACH = (1, 2, 3, 5, 8, 10, 15, 20)  # cents over the price seen
TARGET = 0.85  # the share of the trades to get into (the owner, 30.10.2026)
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))


def reach(db, since, mode, fee_rate=0.0, fee_exponent=1.0):
    """What it takes to get into most of the very same trades (the owner, 30.10.2026: "85% and more, the same
    trades I'm meant to get into"). Of the orders `mode` sent ("paper": the simulation's, "live": real money's)
    since the price each refused one needed was recorded (fill_tests.need): the share that went through, and
    with R cents more, the refused ones whose whole order needed at most their price + R, and what those would
    have made paying what they needed (its deepest level: a bound, the cheaper levels first in fact). The least
    R reaching TARGET, overall and by the price the order wanted."""
    from .model import taker_fee_per_share
    first = db.execute("SELECT MIN(ts) FROM fill_tests WHERE mode = ? AND need IS NOT NULL AND ts >= ?", (mode, since)).fetchone()[0]
    if first is None:
        return {"mode": mode, "sent": 0, "filled": 0, "refused": 0, "rate": None, "rooms": [], "r85": None, "bands": [],
                "state": "collecting", "progress": 0.0}
    sent, filled = db.execute("SELECT COUNT(*), COALESCE(SUM(status = 'filled'), 0) FROM bets WHERE mode = ?"
                              " AND status IN ('filled', 'rejected') AND ts >= ?", (mode, first)).fetchone()
    rows = db.execute(
        "SELECT f.limit_price, f.need, f.want, f.side, w.winner, f.asset || f.minutes || f.window_start FROM fill_tests f"
        " LEFT JOIN windows w ON w.asset = f.asset AND w.minutes = f.minutes AND w.window_start = f.window_start"
        " WHERE f.mode = ? AND f.ts >= ? AND f.need IS NOT NULL", (mode, first)).fetchall()

    def curve(rs, n_sent, n_filled):
        out = []
        for R in REACH:
            extra = [r for r in rs if r[1] < 1 and r[1] - r[0] <= R / 100 + 1e-9]
            pts = [(r[5], r[2] * ((r[4] == r[3]) - r[1] - taker_fee_per_share(r[1], fee_rate, fee_exponent)),
                    r[2] * (r[1] + taker_fee_per_share(r[1], fee_rate, fee_exponent))) for r in extra if r[4] in ("up", "down")]
            roi, z = _score(pts)
            out.append({"cents": R, "rate": (n_filled + len(extra)) / n_sent if n_sent else None, "extra": len(extra),
                        "settled": len(pts), "pnl": sum(p[1] for p in pts), "roi": roi, "z": z})
        r85 = next((x["cents"] for x in out if x["rate"] is not None and x["rate"] >= TARGET), None)
        return out, r85

    rooms, r85 = curve(rows, sent, filled)
    bands = []
    for lo, hi in BANDS:
        rs = [r for r in rows if lo <= r[0] < hi]
        # (a band's own share: its refused ones against all it sent, the filled ones counted by their price)
        b_filled = db.execute("SELECT COUNT(*) FROM bets WHERE mode = ? AND status = 'filled' AND ts >= ? AND avg_price >= ?"
                              " AND avg_price < ?", (mode, first, lo, hi)).fetchone()[0]
        b_rooms, b85 = curve(rs, b_filled + len(rs), b_filled)
        bands.append({"lo": lo, "hi": hi, "refused": len(rs), "filled": b_filled, "rooms": b_rooms, "r85": b85})
    at = next((x for x in rooms if x["cents"] == r85), None)
    state = ("collecting" if len(rows) < 30 else
             "found" if at and at["settled"] >= 30 and (at["pnl"] or 0) > 0 else
             "nothing" if at and at["settled"] >= 30 else "promising")
    return {"mode": mode, "since": first, "sent": sent, "filled": filled, "refused": len(rows),
            "rate": filled / sent if sent else None, "rooms": rooms, "r85": r85, "bands": bands, "state": state,
            "progress": min(1.0, len(rows) / 30)}


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def speed(db, since=0, our_delay=0.5):
    """How fast we are, and how fast we'd need to be (the owner's question of 28.10.2026): our real
    orders' time from the signal to the order sent and Polymarket's answer; the network's round trip to
    the order book; and the race: for each order the book refused, how long after our signal someone
    else bought those shares (the first taker's buy of that side at our price or under, in the round's
    trades). Shares gone within a hair of the signal: no speed of ours would have got them; gone a
    little before our order arrived: a faster order would have."""
    T = [tuple(r) for r in db.execute("SELECT decide, answer FROM order_timing WHERE ts >= ?", (since,))]
    P = [r[0] for r in db.execute("SELECT ms FROM net_ping WHERE ts >= ?", (since,))]
    rows = [tuple(r) for r in db.execute(
        "SELECT ts, asset, minutes, window_start, side, limit_price FROM fill_tests WHERE ts >= ? ORDER BY ts DESC LIMIT 500",
        (since,))]
    race = []
    for ts, asset, minutes, start, side, limit in rows:
        t = db.execute("SELECT MIN(ts) FROM trades WHERE asset = ? AND minutes = ? AND window_start = ? AND outcome = ?"
                       " AND taker = 'BUY' AND price <= ? AND ts >= ? AND ts <= ?",
                       (asset, minutes, start, side, limit + 1e-9, ts - 0.05, ts + 3)).fetchone()[0]
        if t is not None:
            race.append(max(0.0, t - ts))
    ours = (_q([d + a for d, a in T], 0.5) if T else None) or our_delay
    within = lambda s: sum(1 for x in race if x <= s) / len(race) if race else None
    lost_close = sum(1 for x in race if ours / 2 < x <= ours) / len(race) if race else None
    # A server next to Polymarket's (the owner's question of 29.10.2026): the order reaches the book half a
    # round trip after it is sent (the answer's other half comes back after the match), so a server there
    # would reach it half the ping sooner, less the half left there (NEAR_MS): how many of the refused
    # orders' shares were taken in that time, just before our order arrived.
    ping = _q(P, 0.5)
    saved = max(0.0, ping - NEAR_MS) / 2000 if ping is not None else None
    arrive = max(0.0, ours - ping / 2000) if ping is not None else ours
    near = (sum(1 for x in race if arrive - saved < x <= arrive) / len(race)) if race and saved is not None else None
    return {"orders": len(T), "decide": _q([d for d, _ in T], 0.5), "answer": _q([a for _, a in T], 0.5),
            "answer_p90": _q([a for _, a in T], 0.9), "ping": ping, "pings": len(P), "ours": ours,
            "race_n": len(race), "race_median": _q(race, 0.5), "within_100": within(0.1), "within_250": within(0.25),
            "within_ours": within(ours), "halving_saves": lost_close, "near_ms": NEAR_MS,
            "near_saved": saved, "arrive": arrive, "near_saves": near,
            "state": "found" if len(race) >= 50 else "collecting", "progress": min(1.0, len(race) / 50)}
