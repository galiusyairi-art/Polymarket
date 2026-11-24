"""Numbers for the dashboard and the readiness gate. Every function takes a read-only sqlite connection."""
import math
import time

from .lab import describe, money_weighted
from .model import taker_fee_per_share


STRATEGY_SQL = {None: "", "crypto": " AND minutes > 0", "stocks": " AND minutes = 0"}


def bet_summary(db, mode, since=None, min_ratio=0.0, strategy=None, keys=None):
    """Settled and open bets of one mode. strategy: None for both main strategies, "crypto"
    or "stocks" for one. With min_ratio, only the bets that pass that risk/reward filter
    (what a win pays against what a loss costs), as if it had been on. keys: only these
    (asset, minutes, variant) rules."""
    which, table = STRATEGY_SQL[strategy], "bets"
    if keys is not None:
        q = lambda x: "'" + str(x).replace("'", "''") + "'"
        listed = ", ".join(q("|".join(str(x) for x in k)) for k in keys) or "''"
        which += (f" AND variant IN ({', '.join(sorted({q(k[2]) for k in keys})) or chr(39) * 2})"
                  f" AND (asset || '|' || minutes || '|' || variant) IN ({listed})")
        # (by the rules' own trades: the lab has millions, and by the mode alone the database reads them all)
        table = "bets INDEXED BY bets_variant"
    rows = db.execute(
        f"SELECT cost + fee AS paid, pnl FROM {table} WHERE mode = ? AND status = 'filled' AND pnl IS NOT NULL"
        f" AND ts >= ? AND (cost + fee) * (1 + ?) <= shares + 1e-9{which} ORDER BY ts",
        (mode, since or 0, min_ratio or 0.0),
    ).fetchall()
    open_rows = db.execute(
        f"SELECT COUNT(*), COALESCE(SUM(cost + fee), 0) FROM {table}"
        f" WHERE mode = ? AND status = 'filled' AND pnl IS NULL AND ts >= ?{which}",
        (mode, since or 0),
    ).fetchone()
    paid = sum(r["paid"] for r in rows)
    pnl = sum(r["pnl"] for r in rows)
    return {
        "settled": len(rows),
        "open": open_rows[0],
        "open_risk": open_rows[1],
        "wins": sum(1 for r in rows if r["pnl"] > 0),
        "paid": paid,
        "pnl": pnl,
        "roi": pnl / paid if paid else None,
        "rows": [(r["pnl"], r["paid"]) for r in rows if r["paid"]],
    }


def strategy_mix_empty():
    """The count with no strategies in it (before the first one)."""
    return {"counts": {"hot": 0, "swinging": 0, "losing": 0, "few": 0}, "top": [], "worst": [], "active": 0, "bets": 0,
            "pnl": 0.0, "roi": None}


def strategy_mix(db, cfg, since, current=None):
    """Lab variants bucketed by how their settled bets since `since` did.

    current: the (asset, minutes, variant) keys the lab runs now; bets of retired
    variants stay in the database but are not counted.
    """
    sql = ("SELECT asset, minutes, variant, COUNT(*) AS n, SUM(pnl) AS pnl, SUM(cost + fee) AS paid FROM bets{by}"
           " WHERE mode = 'lab' AND pnl IS NOT NULL AND cost + fee > 0 AND ts >= ?{which} GROUP BY asset, minutes, variant")
    names = sorted({v for _, _, v in current}) if current is not None else None
    if names is not None and len(names) <= 40:
        # (only the rules that run, by their own index: by the mode alone the database walks the lab's whole history,
        # millions of trades, once a period, every pass; 19.11.2026)
        by = (" INDEXED BY bets_variant"
              if db.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'bets_variant'").fetchone() else "")
        rows = [r for v in names for r in db.execute(sql.format(by=by, which=" AND variant = ?"), (since, v)).fetchall()]
    else:
        rows = db.execute(sql.format(by="", which=""), (since,)).fetchall()
    buckets = {"hot": [], "swinging": [], "losing": [], "few": []}
    if current is not None:
        rows = [r for r in rows if (r["asset"], r["minutes"], r["variant"]) in current]
    for r in rows:
        roi = r["pnl"] / r["paid"]
        entry = {"asset": r["asset"], "minutes": r["minutes"], "variant": r["variant"],
                 "rule": describe(r["variant"]), "bets": r["n"], "pnl": r["pnl"], "roi": roi}
        if r["n"] < cfg.dash_min_bets:
            buckets["few"].append(entry)
        elif roi >= cfg.dash_hot_roi:
            buckets["hot"].append(entry)
        elif roi <= cfg.dash_losing_roi:
            buckets["losing"].append(entry)
        else:
            buckets["swinging"].append(entry)
    for name in ("hot", "swinging"):
        buckets[name].sort(key=lambda e: -e["roi"])
    buckets["losing"].sort(key=lambda e: e["roi"])
    paid = sum(r["paid"] for r in rows)
    return {
        "counts": {k: len(v) for k, v in buckets.items()},
        "top": buckets["hot"][:3],
        "worst": buckets["losing"][:3],
        "active": len(rows),
        "bets": sum(r["n"] for r in rows),
        "pnl": sum(r["pnl"] for r in rows),
        "roi": sum(r["pnl"] for r in rows) / paid if paid else None,
    }


def recent_bets(db, mode, limit=50, strategy=None):
    rows = db.execute(
        "SELECT id, ts, asset, minutes, window_start, shares, side, seconds_left, fair_prob, avg_price, cost + fee AS paid, cost, edge, status,"
        " error, winner, pnl, price_age, book_age, market, variant, notified, mail_open, mail_close,"
        # (the round's result, also for an order that didn't go through: what it would have made)
        " (SELECT w.winner FROM windows w WHERE w.asset = bets.asset AND w.minutes = bets.minutes"
        f" AND w.window_start = bets.window_start) AS round_winner FROM bets WHERE mode = ?{STRATEGY_SQL[strategy]}"
        " ORDER BY ts DESC LIMIT ?",
        (mode, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def with_need(db, rows, mode="live"):
    """Each refused order of `rows` with "need": the price the whole of it needed when it arrived, from its own fill
    test (the test's time is the order's, so the tests' index on the time finds it). Asked apart from the list's own
    query, one small look a refused order, and never in the page's way: a look that fails leaves that order without it.
    (24.11.2026: it went into the list's query at first, and that night the page's data was never ready after a start;
    whatever the reason, the page's main query is as it was again.)"""
    for b in rows:
        b["need"] = None
        if b.get("status") == "filled" or not b.get("variant") or b.get("ts") is None:
            continue
        try:
            r = db.execute(
                "SELECT need FROM fill_tests WHERE ts BETWEEN ? AND ? AND mode = ? AND asset = ? AND minutes = ? AND window_start = ?"
                " AND variant = ? AND side = ? ORDER BY ABS(ts - ?) LIMIT 1",
                (b["ts"] - 3, b["ts"] + 3, mode, b["asset"], b["minutes"], b["window_start"], b["variant"], b["side"], b["ts"])).fetchone()
            b["need"] = r[0] if r else None
        except Exception:
            pass
    return rows


def would_have(b):
    """A real order's result had it gone through as sent (the owner, 29.10.2026): a filled one's own result; one
    that didn't go through, the shares its amount bought at its price, paid for, once the round is decided;
    else None (not decided, or no price)."""
    if b.get("status") == "filled":
        return b.get("pnl")
    won = b.get("winner") or b.get("round_winner")
    price, cost, paid = b.get("avg_price"), b.get("cost"), b.get("paid")
    if won not in ("up", "down") or not price or not cost or paid is None:
        return None
    return round((cost / price if won == b.get("side") else 0.0) - paid, 2)


BOOK_GONE = 0.99  # a fill test's "need" at or over this: the book hadn't the order's shares at any price


def would_at_book(b, fee_rate=0.0, fee_exponent=1.0):
    """A refused order's result at the price the book really had when it arrived (the owner, 24.11.2026, of a day of
    refused orders each showing what it "would have made": "is there nothing to do to catch these trades?". There was
    nothing to catch: the figure was at the price seen when the order was sent, 63 cents on average, and the book the
    order met stood at 78; at that price the same orders made about nothing): the order's dollars buying at the
    deepest price the whole of it needed, with the fee, once the round is decided. {"price", "pnl"}; {"price": None}
    when the book had none of it at any price; None when it filled, isn't decided, or no fill test was kept for it."""
    try:
        if b.get("status") == "filled" or b.get("need") is None:
            return None
        need, paid = float(b["need"]), b.get("paid")
        if need >= BOOK_GONE:
            return {"price": None, "pnl": None}
        won = b.get("winner") or b.get("round_winner")
        if won not in ("up", "down") or not paid or need <= 0:
            return {"price": need, "pnl": None}
        per = need + fee_rate * (need * (1.0 - need)) ** fee_exponent
        return {"price": need, "pnl": round((paid / per if won == b.get("side") else 0.0) - paid, 2)}
    except Exception:  # (a figure beside a row: never what keeps the page from loading)
        return None


def feed_agreement(db, mode):
    """How often our reading of the rules ("twap" or "point") matched Polymarket's actual result, all markets."""
    rows = db.execute(
        f"SELECT open_{mode} AS o, close_{mode} AS c, winner FROM windows"
        f" WHERE winner IS NOT NULL AND open_{mode} IS NOT NULL AND close_{mode} IS NOT NULL"
    ).fetchall()
    agree = sum(("up" if r["c"] >= r["o"] else "down") == r["winner"] for r in rows)
    return {"windows": len(rows), "agree": agree, "rate": agree / len(rows) if rows else None}


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else None


def analysis(db, cfg, markets=None):
    """The slower numbers: prediction accuracy over all finished windows."""
    first = db.execute("SELECT MIN(ts) FROM ticks").fetchone()[0]
    last = db.execute("SELECT MAX(ts) FROM ticks").fetchone()[0]
    markets = markets or [(a, cfg.window_minutes) for a in cfg.assets]
    where = " OR ".join("(t.asset = ? AND t.minutes = ?)" for _ in markets)
    # (summed inside the database, never read out: 19.11.2026, the recorded seconds of two weeks, a million and more
    # rows, were read into the server's memory at every pass, most of a gigabyte beside the bot's own, and the system
    # ended the bot every quarter of an hour; it also took half of the one processor)
    o = "(CASE WHEN w.winner = 'up' THEN 1.0 ELSE 0.0 END)"
    mid = "((t.up_bid + t.up_ask) / 2)"
    n, sq, mn, msq, wins = db.execute(
        f"SELECT COUNT(*), SUM((t.fair_up - {o}) * (t.fair_up - {o})),"
        f" SUM(t.up_ask IS NOT NULL AND t.up_bid IS NOT NULL),"
        f" SUM(CASE WHEN t.up_ask IS NOT NULL AND t.up_bid IS NOT NULL THEN ({mid} - {o}) * ({mid} - {o}) END),"
        " COUNT(DISTINCT t.asset || '|' || t.window_start)"
        " FROM ticks t JOIN windows w USING (asset, minutes, window_start)"
        f" WHERE ({where}) AND w.winner IS NOT NULL AND t.fair_up IS NOT NULL",
        [x for m in markets for x in m],
    ).fetchone()
    return {
        "computed_at": time.time(),
        "hours": (last - first) / 3600 if first else 0.0,
        "finished_windows": wins or 0,
        "brier_ours": sq / n if n else None,
        "brier_market": msq / mn if mn else None,
    }


def readiness(cfg, hours, feed, paper):
    """Is the simulation strong enough to risk real money? Every check must pass."""
    rows = paper["rows"]
    n = len(rows)
    _, lower = money_weighted(rows, 1.65)  # 95% one-sided: profit is unlikely to be luck
    if n < cfg.dash_min_bets:
        lower = None  # a handful of bets can't tell skill from luck either way
    tail = rows[-cfg.ready_recent_bets:]
    recent = money_weighted(tail, 1.65)[0] if len(tail) >= min(cfg.ready_recent_bets, 10) else None
    checks = [
        {"id": "hours", "value": hours, "target": cfg.ready_min_hours,
         "passed": hours >= cfg.ready_min_hours},
        {"id": "feed", "value": feed["rate"], "target": cfg.ready_min_feed_agreement,
         "windows": feed["windows"], "min_windows": cfg.ready_min_windows,
         "passed": feed["windows"] >= cfg.ready_min_windows
         and (feed["rate"] or 0) >= cfg.ready_min_feed_agreement},
        {"id": "bets", "value": n, "target": cfg.ready_min_bets, "passed": n >= cfg.ready_min_bets},
        {"id": "roi", "value": paper["roi"], "target": cfg.ready_min_roi,
         "passed": paper["roi"] is not None and paper["roi"] >= cfg.ready_min_roi},
        {"id": "luck", "value": lower, "target": 0.0, "passed": lower is not None and lower > 0},
        {"id": "recent", "value": recent, "target": 0.0, "count": cfg.ready_recent_bets,
         "passed": recent is not None and recent > 0},
    ]
    return {"ready": all(c["passed"] for c in checks), "checks": checks}


def calibration(db):
    """Does the model's probability match reality? Settled paper and lab bets, bucketed by
    the probability the model gave the side it bought, against how often that side won."""
    # (counted inside the database, a bucket a row: read out one by one, the lab's millions of trades took hundreds of
    # megabytes of the server's memory at every pass; 19.11.2026)
    los = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    case = ("CASE WHEN fair_prob < 0 OR fair_prob > 1 THEN -1 "
            + " ".join(f"WHEN fair_prob < {hi} THEN {i}" for i, hi in enumerate(los[1:])) + f" ELSE {len(los) - 1} END")
    got = {r[0]: r[1:] for r in db.execute(
        f"SELECT {case} AS b, COUNT(*), SUM(fair_prob), SUM(pnl > 0) FROM bets WHERE mode IN ('paper', 'lab')"
        " AND pnl IS NOT NULL AND minutes > 0 AND fair_prob IS NOT NULL AND variant NOT LIKE 'pair%'"
        " AND variant != 'manual' AND COALESCE(winner, '') != 'sold' GROUP BY b").fetchall()}
    buckets = []
    for i, lo in enumerate(los):
        hi = {0.0: 0.2, 0.2: 0.4, 0.4: 0.5}.get(lo, round(lo + 0.1, 1))
        if i in got:
            k, said, won = got[i]
            buckets.append({"lo": lo, "hi": hi, "bets": k, "said": said / k, "happened": (won or 0) / k})
    n = sum(v[0] for v in got.values())
    return {
        "bets": n,
        "buckets": buckets,
        "said": sum(v[1] for v in got.values()) / n if n else None,
        "happened": sum(v[2] or 0 for v in got.values()) / n if n else None,
    }
