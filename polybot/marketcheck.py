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


def study(db, cfg, ticks="ticks", windows="windows", times=TIMES, fee=True):
    """{"groups": [...], "rounds": n, "days": d, "since": ts, "until": ts}, best pessimistic edge first.
    fee=False: the cost is the price alone."""
    case = " ".join(f"WHEN t.seconds_left >= {lo} AND t.seconds_left < {hi} THEN '{lo}-{hi}'" for hi, lo in times)
    rows = db.execute(
        # SQLite: with MAX(), the bare columns come from the row that has the maximum, i.e. the first
        # price seen as the round entered the stretch.
        f"SELECT t.asset, t.minutes, t.window_start, CASE {case} END AS tg, MAX(t.seconds_left) AS sl,"
        f" t.up_ask, t.down_ask, w.winner FROM {ticks} t JOIN {windows} w ON w.asset = t.asset AND"
        " w.minutes = t.minutes AND w.window_start = t.window_start"
        f" WHERE w.winner IN ('up', 'down') AND t.seconds_left >= {times[-1][1]} AND t.seconds_left < {times[0][0]}"
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
            cost = ask + (taker_fee_per_share(ask, cfg.fee_rate, cfg.fee_exponent) if fee else 0.0)
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


# ---------- The maker check ----------
# The same question from the other side of the book. Every trade in the markets has a taker, who
# bought or sold at the price on offer, and a maker, whose resting order it filled. Had we been that
# maker, the trade would have made us: (what the taker paid - $1 if their side won) a share when
# they bought, (that side's $1 if it won - what we paid) when they sold. No fee for the maker
# (Polymarket's maker rebates come on top and are left out). Grouped by the time left, the price and
# who was the taker, it shows where the people and bots taking prices give money away, and where
# they are the ones who know (the move had begun: makers lose there).
# The trades of one round share one outcome, so a round counts once in the statistics (its trades
# in a group summed), not each trade apart.
MAKER_TIMES = ((900, 600), (600, 300), (300, 180), (180, 120), (120, 60), (60, 30), (30, 15), (15, 0))
MAKER_STEP = 0.10  # price groups of 10 cents


def maker_study(db, cfg, trades="trades", windows="windows", times=MAKER_TIMES):
    case = " ".join(f"WHEN t.seconds_left >= {lo} AND t.seconds_left < {hi} THEN '{lo}-{hi}'" for hi, lo in times)
    rows = db.execute(
        f"SELECT t.asset, t.minutes, t.window_start, CASE {case} END AS tg, CAST(t.price / {MAKER_STEP} AS INTEGER) AS band,"
        " t.taker, t.outcome, SUM(t.size) AS shares, SUM(t.size * t.price) AS paid, COUNT(*) AS n, w.winner"
        f" FROM {trades} t JOIN {windows} w ON w.asset = t.asset AND w.minutes = t.minutes AND w.window_start = t.window_start"
        " WHERE w.winner IN ('up', 'down') AND t.taker IN ('BUY', 'SELL') AND t.price > 0 AND t.price < 1 AND t.size > 0"
        " GROUP BY t.asset, t.minutes, t.window_start, tg, band, t.taker, t.outcome").fetchall()
    rows = [r for r in rows if r["tg"]]
    if not rows:
        return {"groups": [], "trades": 0, "rounds": 0, "days": 0, "volume": 0.0}
    starts = sorted({r["window_start"] for r in rows})
    middle = starts[len(starts) // 2]
    days = max((starts[-1] - starts[0]) / 86400, 1 / 24)
    per_round = {}  # (group, round) -> [pnl, shares, trades, dollars]
    for r in rows:
        won = r["winner"] == r["outcome"]
        pnl = r["paid"] - r["shares"] * won if r["taker"] == "BUY" else r["shares"] * won - r["paid"]
        x = per_round.setdefault(((r["tg"], min(r["band"], int(1 / MAKER_STEP) - 1), r["taker"]),
                                  (r["asset"], r["minutes"], r["window_start"])), [0.0, 0.0, 0, 0.0])
        x[0] += pnl
        x[1] += r["shares"]
        x[2] += r["n"]
        x[3] += r["paid"]
    groups = {}
    for (g, rnd), x in per_round.items():
        groups.setdefault(g, []).append((rnd, *x))
    out = []
    for (tg, band, taker), rs in groups.items():
        n, shares, pnl = len(rs), sum(x[2] for x in rs), sum(x[1] for x in rs)
        if not shares:
            continue
        per = pnl / shares  # dollars a share made for the maker
        resid = sum((x[1] - per * x[2]) ** 2 for x in rs)
        se = math.sqrt(resid * n / (n - 1)) / shares if n > 1 else None
        halves = []
        for newer in (False, True):
            h = [x for x in rs if (x[0][2] >= middle) == newer]
            hs = sum(x[2] for x in h)
            halves.append(sum(x[1] for x in h) / hs if hs else None)
        low = per - Z * se if se is not None else None
        lo, hi = (int(v) for v in tg.split("-"))
        out.append({
            "time": tg, "time_lo": lo, "time_hi": hi, "taker": taker,
            "price_lo": round(band * MAKER_STEP, 2), "price_hi": round((band + 1) * MAKER_STEP, 2),
            "rounds": n, "trades": sum(x[3] for x in rs), "shares": shares, "dollars": sum(x[4] for x in rs),
            "per_share": per, "pnl": pnl, "per_share_low": low, "older": halves[0], "newer": halves[1],
            "per_day": shares / days,
            "edge_found": bool(n >= MIN_ROUNDS and low is not None and low > 0
                               and all(h is not None and h > 0 for h in halves)),
        })
    out.sort(key=lambda g: -(g["per_share_low"] if g["per_share_low"] is not None else -9))
    return {"groups": out, "trades": sum(r["n"] for r in rows), "rounds": len({(r["asset"], r["minutes"], r["window_start"]) for r in rows}),
            "days": days, "volume": sum(r["paid"] for r in rows), "since": starts[0], "until": starts[-1]}
