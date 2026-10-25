"""The market check: where Polymarket's own price is wrong, and by how much, after the fee.

No model and no guess about the direction. For every round that was decided, the price each side
was offered at (the best ask) at a few moments before the close, against which side won. Grouped
by the time left and the price, it says how often a side offered at, say, 90-95 cents with half a
minute left really wins. Where it wins more often than it costs (price plus the taker fee), buying
it is worth money; where less, the market is right or the other side is the buy.

One sample per round and moment (the first price seen as the round enters that stretch), so a
round recorded every second counts the same as one recorded every few seconds. Both sides of a
round are looked at (each has its own price).

Many groups are looked at, so some look good by chance: a group counts as an edge only when its
pessimistic edge (the win rate's lower bound, minus the cost) is above zero and the edge held in
both the older and the newer half of the rounds.
"""

import math

from .model import taker_fee_per_share

# Seconds left before the close: the stretches looked at, from the earliest.
TIMES = ((600, 300), (300, 180), (180, 120), (120, 60), (60, 30), (30, 15), (15, 5))
PRICE_STEP = 0.05  # price groups of 5 cents
Z = 1.96  # the pessimistic edge: the win rate's lower bound at about 95%
MIN_ROUNDS = 50  # fewer samples than this in a group: shown, never called an edge


def wilson_lower(wins, n, z=Z):
    if not n:
        return None
    p = wins / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def _time_group(seconds_left):
    for hi, lo in TIMES:
        if lo <= seconds_left < hi:
            return f"{lo}-{hi}"
    return None


def study(db, cfg):
    """{"groups": [...], "rounds": n, "days": d, "since": ts, "until": ts}, best pessimistic edge first."""
    case = " ".join(f"WHEN t.seconds_left >= {lo} AND t.seconds_left < {hi} THEN '{lo}-{hi}'" for hi, lo in TIMES)
    rows = db.execute(
        # SQLite: with MAX(), the bare columns come from the row that has the maximum, i.e. the first
        # price seen as the round entered the stretch.
        f"SELECT t.asset, t.minutes, t.window_start, CASE {case} END AS tg, MAX(t.seconds_left) AS sl,"
        " t.up_ask, t.down_ask, w.winner FROM ticks t JOIN windows w ON w.asset = t.asset AND"
        " w.minutes = t.minutes AND w.window_start = t.window_start"
        f" WHERE w.winner IN ('up', 'down') AND t.seconds_left >= {TIMES[-1][1]} AND t.seconds_left < {TIMES[0][0]}"
        " GROUP BY t.asset, t.minutes, t.window_start, tg").fetchall()
    rows = [r for r in rows if r["tg"]]
    if not rows:
        return {"groups": [], "rounds": 0, "days": 0, "since": None, "until": None}
    starts = sorted({r["window_start"] for r in rows})
    middle = starts[len(starts) // 2]  # the older half of the rounds, and the newer
    rounds = len({(r["asset"], r["minutes"], r["window_start"]) for r in rows})
    groups = {}
    for r in rows:
        for side, ask in (("up", r["up_ask"]), ("down", r["down_ask"])):
            if ask is None or not 0 < ask < 1:
                continue
            band = min(int(ask / PRICE_STEP), int(1 / PRICE_STEP) - 1)
            g = groups.setdefault((r["tg"], band), {"n": 0, "wins": 0, "cost": 0.0, "halves": [[0, 0, 0.0], [0, 0, 0.0]],
                                                    "markets": set()})
            cost = ask + taker_fee_per_share(ask, cfg.fee_rate, cfg.fee_exponent)
            won = r["winner"] == side
            g["n"] += 1
            g["wins"] += won
            g["cost"] += cost
            g["markets"].add((r["asset"], r["minutes"]))
            h = g["halves"][r["window_start"] >= middle]
            h[0] += 1
            h[1] += won
            h[2] += cost
    days = max((starts[-1] - starts[0]) / 86400, 1 / 24)
    out = []
    for (tg, band), g in groups.items():
        n, cost = g["n"], g["cost"] / g["n"]
        rate = g["wins"] / n
        low = wilson_lower(g["wins"], n)
        halves = [(h[1] / h[0] - h[2] / h[0]) if h[0] else None for h in g["halves"]]
        lo, hi = (int(x) for x in tg.split("-"))
        out.append({
            "time": tg, "time_lo": lo, "time_hi": hi,
            "price_lo": round(band * PRICE_STEP, 2), "price_hi": round((band + 1) * PRICE_STEP, 2),
            "n": n, "wins": g["wins"], "win_rate": rate, "cost": cost,
            "edge": rate - cost,  # dollars a $1 share makes on average (price and fee paid)
            "roi": (rate - cost) / cost if cost else None,
            "edge_low": low - cost if low is not None else None,  # the pessimistic edge
            "older": halves[0], "newer": halves[1],
            "per_day": n / days, "markets": len(g["markets"]),
            "edge_found": bool(n >= MIN_ROUNDS and low is not None and low - cost > 0
                               and all(x is not None and x > 0 for x in halves)),
        })
    out.sort(key=lambda g: -(g["edge_low"] if g["edge_low"] is not None else -9))
    return {"groups": out, "rounds": rounds, "days": days, "since": starts[0], "until": starts[-1]}
