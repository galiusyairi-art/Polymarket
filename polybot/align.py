"""The simulation against real money, round by round (the owner, 21.11.2026: "again a very large gap between the
simulation and the live trading; the simulation makes a lot of money and the live trading loses. We must align, and
understand how the live trading makes money: I do see that on certain kinds of trades one can profit steadily").

Every order of the two over the last days, the simulation's and real money's, put side by side by the round, the rule
and the side it went for. Where the two part, and what each parting is worth:

- both: the same round, rule and side, both got in. The same trade: the difference is the price paid.
- refused: the simulation got in and the book refused real money. The simulation's result there is money real money
  could not have had.
- never sent: the simulation traded and real money sent no order there at all. By why: a rule real money doesn't run;
  real money wasn't trading at that hour (off, or stopped by the day's loss limit); or it was trading and held this one
  back (a cap, a filter, the amount).
- real money's alone, and the rounds real money got in and the simulation was refused.

And real money's own results by rule, by the price paid, by the seconds left and by coin, beside the simulation's: the
kinds of trades that make money for real, and the ones that only do in the simulation.

Dollars are each one's own (the simulation's amount a trade isn't real money's); the return on the dollar is what
compares. Few trades say little: every figure comes with how many trades it stands on.
"""

import time

from .store import connect

NEAR = 1200  # seconds: real money "was trading at that hour" when it sent any order this close to the simulation's
BANDS = ((0.0, 0.2, "עד 20¢"), (0.2, 0.4, "20-40¢"), (0.4, 0.6, "40-60¢"), (0.6, 0.8, "60-80¢"), (0.8, 1.01, "מעל 80¢"))
LEFT = ((0, 30, "30 שניות אחרונות"), (30, 60, "30-60 שניות"), (60, 120, "1-2 דקות"), (120, 10 ** 9, "יותר מ-2 דקות"))

# (how old the price and the book were when the order was decided on: the older of the two; real money's are kept from
# 21.11.2026, and an order of real money's from before takes the simulation's of the same moment)
AGES = ((0.0, 0.1, "עד עשירית שנייה"), (0.1, 0.3, "0.1-0.3 שנ׳"), (0.3, 1.0, "0.3-1 שנ׳"), (1.0, 10 ** 9, "שנייה ויותר"))
SAME_MOMENT = 1.0  # seconds between real money's order and the simulation's that count as the same decision

COLS = ("mode", "variant", "status", "error", "ts", "asset", "minutes", "window_start", "side", "seconds_left", "avg_price",
        "shares", "cost", "fee", "pnl", "price_age", "book_age")


def _sum(rows):
    """Orders into one line: sent, in, settled, won, paid, pnl, the return on the dollar, the average price."""
    sent = len(rows)
    got = [r for r in rows if r["status"] == "filled"]
    done = [r for r in got if r["pnl"] is not None]
    paid = sum((r["cost"] or 0) + (r["fee"] or 0) for r in done)
    pnl = sum(r["pnl"] for r in done)
    return {"sent": sent, "in": len(got), "settled": len(done), "won": sum(1 for r in done if r["pnl"] > 0),
            "paid": round(paid, 2), "pnl": round(pnl, 2), "roi": pnl / paid if paid > 0 else None,
            "price": sum(r["avg_price"] or 0 for r in got) / len(got) if got else None}


def _band(x, bands):
    for lo, hi, name in bands:
        if lo <= x < hi:
            return name
    return bands[-1][2]


def study(db, now=None, days=3.0, names=None):
    now = now or time.time()
    since = now - days * 86400
    rows = {"paper": [], "live": []}
    for mode in rows:
        for r in db.execute(f"SELECT {', '.join(COLS)} FROM bets WHERE mode = ? AND ts >= ? AND minutes > 0"
                            " AND status IN ('filled', 'rejected')", (mode, since)):
            rows[mode].append(dict(zip(COLS, r)))
    key = lambda r: (r["asset"], r["minutes"], r["window_start"], r["variant"], r["side"])
    by = {m: {} for m in rows}
    for m, rs in rows.items():
        for r in rs:
            by[m].setdefault(key(r), []).append(r)
    P, L = by["paper"], by["live"]
    is_in = lambda rs: any(r["status"] == "filled" for r in rs)
    flat = lambda keys, side: [r for k in keys for r in side[k]]

    both = [k for k in L if k in P]
    cell = {"both_in": [k for k in both if is_in(P[k]) and is_in(L[k])],
            "sim_in_live_refused": [k for k in both if is_in(P[k]) and not is_in(L[k])],
            "sim_refused_live_in": [k for k in both if not is_in(P[k]) and is_in(L[k])],
            "both_refused": [k for k in both if not is_in(P[k]) and not is_in(L[k])]}
    same = {name: {"rounds": len(keys), "sim": _sum(flat(keys, P)), "live": _sum(flat(keys, L))} for name, keys in cell.items()}
    dearer = []
    for k in cell["both_in"]:
        p = [r["avg_price"] for r in P[k] if r["status"] == "filled" and r["avg_price"]]
        l_ = [r["avg_price"] for r in L[k] if r["status"] == "filled" and r["avg_price"]]
        if p and l_:
            dearer.append(sum(l_) / len(l_) - sum(p) / len(p))
    same["both_in"]["dearer"] = sum(dearer) / len(dearer) if dearer else None
    same["both_in"]["dearer_share"] = sum(1 for x in dearer if x > 0.005) / len(dearer) if dearer else None

    # the simulation's alone: why real money sent nothing there
    live_rules = {r["variant"] for r in rows["live"]}
    live_ts = sorted(r["ts"] for r in rows["live"])

    def near(ts):
        import bisect
        i = bisect.bisect_left(live_ts, ts - NEAR)
        return i < len(live_ts) and live_ts[i] <= ts + NEAR
    alone = {"other_rule": [], "not_trading": [], "held_back": []}
    for k in P:
        if k in L:
            continue
        why = ("other_rule" if k[3] not in live_rules else
               "held_back" if any(near(r["ts"]) for r in P[k]) else "not_trading")
        alone[why].append(k)
    sim_alone = {why: dict(_sum(flat(keys, P)), rounds=len(keys)) for why, keys in alone.items()}
    live_alone = dict(_sum(flat([k for k in L if k not in P], L)), rounds=sum(1 for k in L if k not in P))

    # by rule, and by the kind of trade
    label = lambda v: v
    if names:
        short = {}
        for full, name in names.items():
            v = full.split("|", 2)[-1]
            short.setdefault(v, set()).add(name)
        label = lambda v: (next(iter(short[v])) + " · " + v) if len(short.get(v, ())) == 1 else v
    rules = []
    for v in sorted({r["variant"] for m in rows for r in rows[m]}):
        s = _sum([r for r in rows["paper"] if r["variant"] == v])
        l_ = _sum([r for r in rows["live"] if r["variant"] == v])
        if s["sent"] or l_["sent"]:
            rules.append({"rule": v, "name": label(v), "sim": s, "live": l_})
    rules.sort(key=lambda x: (-x["live"]["settled"], -x["sim"]["settled"]))

    def split(pick, order):
        out = []
        for name in order:
            s = _sum([r for r in rows["paper"] if r["status"] == "filled" and pick(r) == name])
            l_ = _sum([r for r in rows["live"] if r["status"] == "filled" and pick(r) == name])
            if s["in"] or l_["in"]:
                out.append({"name": name, "sim": s, "live": l_})
        return out
    # by how old the data was: every order sent, the refused ones too (an old book is where they are refused)
    def age(r):
        if r["book_age"] is None and r["mode"] == "live":
            near = [p for p in P.get(key(r), ()) if p["book_age"] is not None and abs(p["ts"] - r["ts"]) <= SAME_MOMENT]
            r = min(near, key=lambda p: abs(p["ts"] - r["ts"])) if near else r
        if r["book_age"] is None:
            return None
        return _band(max(r["book_age"], r["price_age"] or 0.0), AGES)
    for m in rows:
        for r in rows[m]:
            r["age"] = age(r)
    ages = []
    for name in [b[2] for b in AGES]:
        s_, l_ = (_sum([r for r in rows[m] if r["age"] == name]) for m in ("paper", "live"))
        if s_["sent"] or l_["sent"]:
            ages.append({"name": name, "sim": s_, "live": l_})
    kinds = {"age": ages,
             "price": split(lambda r: _band(r["avg_price"] or 0, BANDS), [b[2] for b in BANDS]),
             "left": split(lambda r: _band(r["seconds_left"] or 0, LEFT), [b[2] for b in LEFT]),
             "asset": split(lambda r: f"{(r['asset'] or '').upper()} · {r['minutes']} דק׳",
                            sorted({f"{(r['asset'] or '').upper()} · {r['minutes']} דק׳" for m in rows for r in rows[m]}))}

    # real orders that got in and that were refused: how long each took
    timing = {}
    try:
        for status, n, d, a in db.execute(
                "SELECT status, COUNT(*), AVG(decide), AVG(answer) FROM order_timing WHERE ts >= ? GROUP BY status", (since,)):
            xs = sorted(r[0] for r in db.execute("SELECT decide + answer FROM order_timing WHERE ts >= ? AND status = ?", (since, status)))
            timing[status] = {"n": n, "decide": d, "answer": a, "median": xs[len(xs) // 2] if xs else None,
                              "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))] if xs else None}
    except Exception:
        timing = {}
    refused_why = {}
    for r in rows["live"]:
        if r["status"] == "rejected":
            e = (r["error"] or "").lower()
            k = ("book" if "nothing filled" in e or "no orders found" in e or "fak" in e or "fully filled" in e else
                 "balance" if "balance" in e or "allowance" in e else "other")
            refused_why[k] = refused_why.get(k, 0) + 1
    return {"ts": now, "days": days, "since": since,
            "total": {"sim": _sum(rows["paper"]), "live": _sum(rows["live"])},
            "same": same, "sim_alone": sim_alone, "live_alone": live_alone,
            "rules": rules[:40], "kinds": kinds, "timing": timing, "refused_why": refused_why}


def study_db(db_path, now=None, days=3.0, names=None):
    db = connect(db_path)
    try:
        return study(db, now, days, names)
    finally:
        db.close()
