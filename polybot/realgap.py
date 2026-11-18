"""What real money gets of what the lab measures (the owner, 18.11.2026, after his own trades were read: on the rounds
real money got in, it made about what the lab's copy of the rule made there, a cent a share dearer; on the rounds the
book refused it, the lab's copy made by far the most. The orders that don't get in are the ones that would have won).

Every real order of the last DAYS days (the ones that got in and the ones refused), beside the lab's own trade of the
same rule in the same round:

- filled: the rounds real money got in. The lab's return there, and real money's own.
- refused: the rounds real money only met refusals. The lab's return there (real money made nothing).
- the gap: the lab's return on the rounds real money tried (got in or not), less real money's return on what it got.
  That many points come off a lab result before it says anything about real money.

Each trade counts the same (its own return, profit over what it paid): a lab trade is $10 and a real one a few dollars,
and one large trade must not speak for the rest. Read only once there are enough of both kinds (NEED_FILLED, NEED_REFUSED).
"""

import math
import time

from .store import connect

DAYS = 60
NEED_FILLED = 100
NEED_REFUSED = 40
Z = 1.645


def _mean_se(xs):
    n = len(xs)
    if not n:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def study(db, now=None, days=DAYS):
    now = now or time.time()
    since = now - days * 86400
    real = {}  # (variant, asset, minutes, round) -> {"paid", "pnl", "px", "side", "refused"}
    for variant, asset, minutes, w, status, cost, fee, shares, pnl, side in db.execute(
            "SELECT variant, asset, minutes, window_start, status, cost, fee, shares, pnl, side FROM bets "
            "WHERE mode = 'live' AND ts >= ? AND variant IS NOT NULL", (since,)):
        x = real.setdefault((variant, asset, minutes, w), {"paid": 0.0, "pnl": 0.0, "shares": 0.0, "side": side, "refused": 0, "open": False})
        if status == "filled" and (shares or 0) > 0:
            if pnl is None:
                x["open"] = True
                continue
            x["paid"] += (cost or 0) + (fee or 0)
            x["pnl"] += pnl
            x["shares"] += shares
        elif status == "rejected":
            x["refused"] += 1
    rules = {k[:3] for k in real}
    lab = {}
    for variant, asset, minutes in rules:
        for w, side, paid, pnl, shares in db.execute(
                "SELECT window_start, side, cost + fee, pnl, shares FROM bets INDEXED BY bets_variant WHERE variant = ? AND mode = 'lab' "
                "AND asset = ? AND minutes = ? AND ts >= ? AND status = 'filled' AND pnl IS NOT NULL AND shares > 0",
                (variant, asset, minutes, since)):
            if paid and (variant, asset, minutes, w) not in lab:  # (a rule's first trade of the round, as real money's)
                lab[(variant, asset, minutes, w)] = (pnl / paid, paid / shares, side)
    lab_f, real_f, lab_r, price, by = [], [], [], [], {}
    for k, x in real.items():
        if x["open"] or k not in lab:
            continue
        r, px, side = lab[k]
        row = by.setdefault(k[:3], {"filled": 0, "refused": 0})
        if x["paid"] > 0:
            lab_f.append(r)
            real_f.append(x["pnl"] / x["paid"])
            row["filled"] += 1
            if side == x["side"]:
                price.append(x["paid"] / x["shares"] - px)
        elif x["refused"]:
            lab_r.append(r)
            row["refused"] += 1
    a, a_se = _mean_se(lab_f)
    c, c_se = _mean_se(real_f)
    b, b_se = _mean_se(lab_r)
    tried, tried_se = _mean_se(lab_f + lab_r)
    orders = sum(1 for x in real.values() if x["paid"] > 0 or x["refused"])
    out = {"ts": now, "days": days, "orders": orders, "refused_orders": sum(1 for x in real.values() if not x["paid"] and x["refused"]),
           "filled": len(lab_f), "refused": len(lab_r), "need": {"filled": NEED_FILLED, "refused": NEED_REFUSED},
           "lab_filled": a, "real_filled": c, "lab_refused": b, "lab_tried": tried,
           "price": sum(price) / len(price) if price else None,
           "rules": sorted(({"variant": k[0], "asset": k[1], "minutes": k[2], **v} for k, v in by.items()),
                           key=lambda r: -(r["filled"] + r["refused"]))[:8]}
    if tried is not None and c is not None:
        gap = tried - c
        se = math.sqrt((tried_se or 0) ** 2 + (c_se or 0) ** 2) if tried_se is not None and c_se is not None else None
        out.update(gap=gap, gap_low=None if se is None else gap - Z * se, gap_high=None if se is None else gap + Z * se)
    out["state"] = ("none" if not (lab_f or lab_r) else
                    "measuring" if len(lab_f) < NEED_FILLED or len(lab_r) < NEED_REFUSED else "measured")
    return out


def study_db(db_path, now=None, days=DAYS):
    db = connect(db_path)
    try:
        return study(db, now, days)
    finally:
        db.close()
