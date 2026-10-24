"""Risk/reward research, per strategy: which filter makes this strategy the most money.

Every settled lab trade of a strategy is kept with what it paid per share and how it ended, so
any filter can be replayed on its history at once: "had it bought only up to 67c (a win paying
at least $5 for every $10), what would it have made?". The filter is chosen on the oldest 70%
of the strategy's trades and kept only if it also wins on the newest 30%, trades it was not
chosen on. Choosing the best of nine filters on the same trades always finds one that looks
great by luck; the newest trades are the test it has to pass.

The aim is the most money, so filters are compared by total profit in the pessimistic scenario
(the return minus what may be luck, times the money it puts to work), not by return per dollar:
a filter that leaves out most trades can show a fine return and still make less.
"""
from .lab import Variant, money_weighted

GRID = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)  # a win paying at least $0..$10 for every $10
TRAIN_SHARE = 0.7
MIN_TRAIN = 20  # trades a filter keeps on the older part before it can be chosen
MIN_TEST = 10  # trades it keeps on the newer part before it can be trusted
MARGIN = 0.03  # on the newer part, at least 3 points of return better than what runs now
GAP = 30  # new trades of a strategy between two changes of its filter
Z = 1.65


def _stats(rows):
    """rows: (pnl, paid). Count, wins, profit, money put in, return, pessimistic return."""
    paid = sum(p for _, p in rows)
    roi, low = money_weighted(rows, Z) if rows else (None, None)
    return {"trades": len(rows), "wins": sum(1 for x, _ in rows if x > 0), "pnl": sum(x for x, _ in rows),
            "paid": paid, "roi": roi, "pess": low}


def _kept(trades, r):
    """The trades a filter keeps: those that paid no more than it allows per share (fee included)."""
    cap = 1.0 / (1.0 + r) if r else 1.0
    return [(pnl, paid) for per_share, pnl, paid in trades if per_share <= cap + 1e-9]


def _score(st):
    """Total profit in the pessimistic scenario (None: too little to say)."""
    return None if st["pess"] is None else st["pess"] * st["paid"]


def study(trades, current=0.0):
    """One strategy. trades: [(cost per share, pnl, paid)] oldest first. current: the filter in
    force where it runs. Returns the curve on all its trades, and the verdict."""
    curve = [dict(_stats(_kept(trades, r)), ratio=r) for r in GRID]
    n = len(trades)
    out = {"curve": curve, "n": n, "current": current, "recommended": None, "verdict": "waiting"}
    if n < MIN_TRAIN + MIN_TEST:
        return out
    cut = int(n * TRAIN_SHARE)
    train, test = trades[:cut], trades[cut:]
    best, best_score = 0.0, None
    for r in GRID:
        st = _stats(_kept(train, r))
        if st["trades"] < MIN_TRAIN:
            continue
        sc = _score(st)
        if sc is not None and (best_score is None or sc > best_score + 1e-9):
            best, best_score = r, sc
    t_best, t_now = _stats(_kept(test, best)), _stats(_kept(test, current))
    out.update(train_n=len(train), test_n=len(test), chosen=best, test_best=t_best, test_now=t_now)
    if best == current:
        out.update(recommended=current, verdict="keep")
    elif (t_best["trades"] >= MIN_TEST and (t_best["roi"] or 0) > 0 and t_best["pnl"] >= t_now["pnl"]
          and (t_best["roi"] or 0) - (t_now["roi"] or 0) >= MARGIN):
        out.update(recommended=best, verdict="change")
    else:
        out.update(recommended=current, verdict="not_proven")  # looked better on the old trades only
    return out


def research(db, keys, current=None):
    """For each strategy in keys {(asset, minutes, variant)}: its own settled lab trades, studied.
    A rule that carries its own filter is studied as the rule it is (nothing to add to it)."""
    current = current or {}
    if not keys:
        return {}
    variants = sorted({v for _, _, v in keys})
    rows = db.execute(
        "SELECT asset, minutes, variant, cost + fee AS paid, shares, pnl FROM bets WHERE mode = 'lab'"
        " AND status = 'filled' AND pnl IS NOT NULL AND shares > 0 AND cost + fee > 0"
        f" AND variant IN ({','.join('?' * len(variants))}) ORDER BY ts", variants).fetchall()
    by = {}
    for r in rows:
        k = (r["asset"], r["minutes"], r["variant"])
        if k in keys:
            by.setdefault(k, []).append((r["paid"] / r["shares"], r["pnl"], r["paid"]))
    out = {}
    for k in keys:
        try:
            own = Variant.from_key(k[2]).min_ratio
        except Exception:
            continue
        if own:
            continue
        out[k] = study(by.get(k, []), current.get(k, 0.0))
    return out
