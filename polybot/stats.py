"""Numbers for the dashboard and the readiness gate. Every function takes a read-only sqlite connection."""
import math
import time

from .model import taker_fee_per_share


def bet_summary(db, mode, since=None):
    rows = db.execute(
        "SELECT cost + fee AS paid, pnl FROM bets WHERE mode = ? AND status = 'filled' AND pnl IS NOT NULL"
        " AND ts >= ? ORDER BY ts",
        (mode, since or 0),
    ).fetchall()
    open_rows = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(cost + fee), 0) FROM bets"
        " WHERE mode = ? AND status = 'filled' AND pnl IS NULL AND ts >= ?",
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
        "returns": [r["pnl"] / r["paid"] for r in rows if r["paid"]],
    }


def recent_bets(db, mode, limit=50):
    rows = db.execute(
        "SELECT ts, asset, side, seconds_left, fair_prob, avg_price, cost + fee AS paid, edge, status, error,"
        " winner, pnl FROM bets WHERE mode = ? ORDER BY ts DESC LIMIT ?",
        (mode, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def feed_agreement(db):
    rows = db.execute(
        "SELECT open_price, our_end_price, winner FROM windows"
        " WHERE winner IS NOT NULL AND open_price IS NOT NULL AND our_end_price IS NOT NULL"
    ).fetchall()
    agree = sum(("up" if r["our_end_price"] >= r["open_price"] else "down") == r["winner"] for r in rows)
    return {"windows": len(rows), "agree": agree, "rate": agree / len(rows) if rows else None}


def replay(rows, cfg, min_edge):
    """Re-run the entry rule at another threshold using top-of-book prices.

    Optimistic: assumes the best ask had enough size for a 1-share bet.
    rows: (asset, window_start, seconds_left, fair_up, up_ask, down_ask, winner), time-ordered.
    Returns (bets, wins, pnl_per_share_total).
    """
    taken = set()
    bets = wins = 0
    pnl = 0.0
    for asset, start, left, fair_up, up_ask, down_ask, winner in rows:
        if (asset, start) in taken or not (cfg.min_seconds_left <= left <= cfg.max_seconds_left):
            continue
        for side, prob, ask in (("up", fair_up, up_ask), ("down", 1 - fair_up, down_ask)):
            if ask is None:
                continue
            fee = taker_fee_per_share(ask, cfg.fee_rate, cfg.fee_exponent)
            if prob - ask - fee >= min_edge:
                taken.add((asset, start))
                bets += 1
                won = side == winner
                wins += won
                pnl += (1.0 if won else 0.0) - ask - fee
                break
    return bets, wins, pnl


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else None


def analysis(db, cfg):
    """The slower numbers: prediction accuracy and what-if thresholds over all finished windows."""
    first, last = db.execute("SELECT MIN(ts), MAX(ts) FROM ticks").fetchone()
    ticks = db.execute(
        "SELECT t.asset, t.window_start, t.seconds_left, t.fair_up, t.up_ask, t.down_ask, w.winner, t.up_bid"
        " FROM ticks t JOIN windows w USING (asset, window_start)"
        " WHERE w.winner IS NOT NULL AND t.fair_up IS NOT NULL ORDER BY t.ts"
    ).fetchall()
    outcome = lambda t: 1.0 if t[6] == "up" else 0.0
    ours = [(t[3], outcome(t)) for t in ticks]
    market = [((t[7] + t[4]) / 2, outcome(t)) for t in ticks if t[4] is not None and t[7] is not None]
    rows = [tuple(t)[:7] for t in ticks]
    whatif = []
    for edge in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
        b, w, p = replay(rows, cfg, edge)
        whatif.append({"min_edge": edge, "bets": b, "win_rate": w / b if b else None,
                       "pnl_per_share": p / b if b else None})
    return {
        "computed_at": time.time(),
        "hours": (last - first) / 3600 if first else 0.0,
        "finished_windows": len({(t[0], t[1]) for t in ticks}),
        "brier_ours": brier(ours),
        "brier_market": brier(market),
        "whatif": whatif,
    }


def readiness(cfg, hours, feed, paper, recent):
    """Is the simulation strong enough to risk real money? Every check must pass."""
    returns = paper["returns"]
    n = len(returns)
    lower = None
    if n >= 2:
        mean = sum(returns) / n
        sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
        lower = mean - 1.65 * sd / math.sqrt(n)  # 95% one-sided: profit is unlikely to be luck
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


def recent_roi(returns, count):
    tail = returns[-count:]
    return sum(tail) / len(tail) if len(tail) >= min(count, 10) else None
