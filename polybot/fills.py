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
NEED = 100  # refused orders a room would have filled, settled, before a verdict on it
Z = 2.0


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
        " f.asset || f.minutes || f.window_start FROM fill_tests f LEFT JOIN windows w ON w.asset = f.asset"
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
    # Room: what 1 or 2 cents more would have filled, settled.
    room = {}
    for k, (ci, fi) in ((1, (4, 5)), (2, (6, 7))):
        for who in ("paper", "lab"):
            pts = [(r[9], r[2] * (r[8] == r[1]) - r[ci] - r[fi], r[ci] + r[fi]) for r in rows
                   if r[0] == who and r[ci] is not None and r[8] in ("up", "down")]
            roi, z = _score(pts)
            would = sum(1 for r in rows if r[0] == who and r[ci] is not None)
            room[f"{who}{k}"] = {"cents": k, "who": who, "settled": len(pts), "would_fill": would,
                                 "of": sum(1 for r in rows if r[0] == who), "pnl": sum(p[1] for p in pts),
                                 "roi": roi, "z": z}
    best = max((room[f"paper{k}"] for k in (1, 2)), key=lambda x: x["z"] if x["z"] is not None else -9)
    enough = best["settled"] >= NEED
    room_state = ("found" if enough and (best["z"] or 0) >= Z and (best["roi"] or 0) > 0 else
                  "nothing" if enough and (best["roi"] or 0) <= 0 else
                  "promising" if (best["roi"] or 0) > 0 and best["settled"] >= NEED // 4 else "collecting")
    return {"why": why, "room": room, "room_state": room_state, "best": best["cents"],
            "room_progress": min(1.0, best["settled"] / NEED), "why_progress": min(1.0, n / 50), "need": NEED}


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
        "SELECT ts, asset, minutes, window_start, side, limit_price FROM fill_tests WHERE ts >= ? ORDER BY ts DESC LIMIT 2000",
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
    return {"orders": len(T), "decide": _q([d for d, _ in T], 0.5), "answer": _q([a for _, a in T], 0.5),
            "answer_p90": _q([a for _, a in T], 0.9), "ping": _q(P, 0.5), "pings": len(P), "ours": ours,
            "race_n": len(race), "race_median": _q(race, 0.5), "within_100": within(0.1), "within_250": within(0.25),
            "within_ours": within(ours), "halving_saves": lost_close,
            "state": "found" if len(race) >= 50 else "collecting", "progress": min(1.0, len(race) / 50)}
