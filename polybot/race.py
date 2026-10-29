"""The race for the shares (the owner, 29.10.2026: "the ones refused would have made money; we must find a way
into them too", and three panels' answer). Four experiments on what is recorded already, none of them trading:

  A. The simulation, racing: each of its fills again, as if others' orders were there too. Within our real
     delay after the signal (the order's way to the book), others' buys of the same side at our price or under
     (the round's trades) took shares from the book; when they took what our order would have found, real money
     would have been refused there. Every strategy's figures now and racing.
  B. The intent kept open (the owner's idea): a refused order isn't given up; for 5, 15 or 30 seconds, every
     second, the most a share may cost is counted again (our model's chance now, less the rule's edge and the
     fee, within its risk/reward filter), and the order buys at the first offer within it that stayed there
     a second. Judged by the owner's measure: the same win rate and return as the strategy's own fills.
  C. The room that pays most, by price: of the simulation's refused orders, by the price they wanted, what 1,
     2, 3 and 5 cents more would have filled and made; the best, where it makes money.
  D. Standing in the book instead of racing: at the signal, a bid at the price seen less a cent or two, kept
     30 seconds or until our model no longer sees the edge at it (then cancelled); filled when sellers trade
     under it (they went through it: the pessimistic way). No fee for the maker.

Each is judged on enough cases and on the money (a score that counts each round once).
"""

import json
import math

from . import lab
from .model import taker_fee_per_share

NEED = 100  # cases (decided) before a verdict
Z = 2.0
WINDOWS = (5, 15, 30)  # seconds the intent stays open
STEADY = 1.0  # an offer counts when it stayed this long (a flicker is someone else's already)
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
ROOMS = (1, 2, 3, 5)
BID_UNDER = (0.0, 0.01, 0.02)  # D: the bid, under the price the signal saw
BID_FOR = 30.0
DAYS = 14


TAKERS = ("twap", "twapt", "twap3")  # the rules that buy on our model's edge by taking the offer (not twapm: a maker)


def _plain(key):
    """A rule that buys on our model's edge by taking the offer (the ones the race is about); else None."""
    try:
        v = lab.Variant.from_key(key)
    except Exception:
        return None
    return v if v.resolution in TAKERS else None


def score(points):
    """points: (round, pnl, paid). The return a dollar, and its z counting each round once."""
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
    return roi, pnl / math.sqrt(var) if var > 0 else None


def summary(points, wins=None):
    roi, z = score(points)
    return {"n": len(points), "pnl": round(sum(p[1] for p in points), 2), "paid": round(sum(p[2] for p in points), 2),
            "roi": roi, "z": z, "win": (wins / len(points)) if points and wins is not None else None}


def _verdict(s, base=None):
    """found: enough cases, making money and sure of it (and, against a base, keeping at least 80% of its return)."""
    if not s or s["n"] < NEED:
        return "promising" if s and s["n"] >= NEED // 4 and (s["roi"] or 0) > 0 else "collecting"
    if (s["roi"] or 0) <= 0:
        return "nothing"
    if base and base.get("roi") and base["roi"] > 0 and s["roi"] < 0.8 * base["roi"]:
        return "nothing"
    return "found" if (s["z"] or 0) >= Z else "promising"


# ---------- A. the simulation, racing ----------
def racing(db, since, arrive):
    rows = db.execute(
        "SELECT ts, asset, minutes, window_start, side, variant, shares, avg_price, cost + fee, pnl, book FROM bets"
        " WHERE mode = 'paper' AND status = 'filled' AND pnl IS NOT NULL AND ts >= ? ORDER BY ts DESC LIMIT 20000",
        (since,)).fetchall()
    per, now_pts, race_pts, lost, unsure = {}, [], [], 0, 0
    wins_now = wins_race = 0
    for ts, asset, minutes, start, side, variant, shares, price, paid, pnl, book in rows:
        try:
            levels = json.loads(book) if book else None
        except ValueError:
            levels = None
        top = max((p for p, _ in levels), default=price) if levels else price
        depth = sum(s for _, s in levels) if levels else None
        took = db.execute("SELECT COALESCE(SUM(size), 0) FROM trades WHERE asset = ? AND minutes = ? AND window_start = ?"
                          " AND outcome = ? AND taker = 'BUY' AND price <= ? AND ts > ? AND ts <= ?",
                          (asset, minutes, start, side, top + 1e-9, ts, ts + arrive)).fetchone()[0] or 0.0
        if depth is None:
            unsure += 1
            gone = took > 0  # (without the book's depth: any buy at our price before us is taken as the race lost)
        else:
            gone = took > depth - shares + 1e-9
        rnd = f"{asset}{minutes}{start}"
        k = f"{asset}|{minutes}|{variant}"
        s = per.setdefault(k, {"now": [], "race": [], "wins_now": 0, "wins_race": 0})
        s["now"].append((rnd, pnl, paid))
        now_pts.append((rnd, pnl, paid))
        wins_now += pnl > 0
        s["wins_now"] += pnl > 0
        if gone:
            lost += 1
        else:
            s["race"].append((rnd, pnl, paid))
            race_pts.append((rnd, pnl, paid))
            wins_race += pnl > 0
            s["wins_race"] += pnl > 0
    rows_out = []
    for k, s in per.items():
        a, m, v = k.split("|", 2)
        rows_out.append({"asset": a, "minutes": int(m), "variant": v, "now": summary(s["now"], s["wins_now"]),
                         "race": summary(s["race"], s["wins_race"])})
    rows_out.sort(key=lambda r: -r["now"]["n"])
    N, R = summary(now_pts, wins_now), summary(race_pts, wins_race)
    # (the verdict: does the simulation's money survive the race? found: yes; nothing: it was the race's money)
    state = "collecting" if N["n"] < NEED else ("found" if (R["roi"] or 0) > 0 and (R["z"] or 0) >= Z else
                                                "nothing" if (R["roi"] or 0) <= 0 else "promising")
    return {"now": N, "race": R, "lost": lost, "unsure": unsure, "arrive": arrive, "rows": rows_out[:40],
            "state": state, "progress": min(1.0, N["n"] / NEED)}


# ---------- B. the intent kept open ----------
def _ticks(db, asset, minutes, start, t0, t1):
    return db.execute("SELECT ts, seconds_left, fair_up, up_ask, down_ask FROM ticks WHERE asset = ? AND minutes = ?"
                      " AND window_start = ? AND ts > ? AND ts <= ? ORDER BY ts", (asset, minutes, start, t0, t1)).fetchall()


def intent_fill(ticks, side, rule, fee_rate, fee_exp, until):
    """The first offer within the most the intent may pay, held STEADY: (price, ts) or None."""
    held = None
    for ts, left, fair_up, up_ask, down_ask in ticks:
        if ts > until:
            break
        ask = up_ask if side == "up" else down_ask
        if fair_up is None or ask is None or ask <= 0 or ask >= 1 or (left is not None and left < rule.min_seconds_left):
            held = None
            continue
        fair = fair_up if side == "up" else 1 - fair_up
        per = ask + taker_fee_per_share(ask, fee_rate, fee_exp)
        ok = fair - per >= rule.min_edge - 1e-9 and (not rule.ratio or 1 - per >= rule.ratio * per - 1e-12) \
            and rule.passes(ask, fair)
        if not ok:
            held = None
            continue
        if held is None or abs(held[0] - ask) > 1e-9:
            held = (ask, ts)
        if ts - held[1] >= STEADY - 1e-9:
            return ask, ts
    return None


def intent(db, since, fee_rate, fee_exp):
    refused = db.execute(
        "SELECT f.ts, f.asset, f.minutes, f.window_start, f.side, f.variant, f.want, f.limit_price, w.winner"
        " FROM fill_tests f LEFT JOIN windows w ON w.asset = f.asset AND w.minutes = f.minutes AND w.window_start = f.window_start"
        " WHERE f.ts >= ? AND f.mode IN ('paper', 'live') ORDER BY f.ts DESC LIMIT 5000", (since,)).fetchall()
    out = {str(W): {"tried": 0, "filled": 0, "pts": [], "wins": 0, "paid_more": 0.0} for W in WINDOWS}
    keys = set()
    for ts, asset, minutes, start, side, variant, want, limit, winner in refused:
        rule = _plain(variant)
        if not rule or winner not in ("up", "down"):
            continue
        keys.add(variant)
        ticks = _ticks(db, asset, minutes, start, ts, ts + max(WINDOWS))
        for W in WINDOWS:
            o = out[str(W)]
            o["tried"] += 1
            got = intent_fill(ticks, side, rule, fee_rate, fee_exp, ts + W)
            if not got:
                continue
            price = got[0]
            per = price + taker_fee_per_share(price, fee_rate, fee_exp)
            pnl = want * ((winner == side) - per)
            o["filled"] += 1
            o["wins"] += winner == side
            o["paid_more"] += price - limit
            o["pts"].append((f"{asset}{minutes}{start}", pnl, want * per))
    # The base: the same strategies' own fills in the simulation over the same days (the owner's measure).
    base_pts, base_wins = [], 0
    for key in keys:
        for ts, a, m, start, pnl, paid in db.execute(
                "SELECT ts, asset, minutes, window_start, pnl, cost + fee FROM bets WHERE mode = 'paper' AND status = 'filled'"
                " AND pnl IS NOT NULL AND variant = ? AND ts >= ?", (key, since)):
            base_pts.append((f"{a}{m}{start}", pnl, paid))
            base_wins += pnl > 0
    base = summary(base_pts, base_wins)
    res = {}
    for W, o in out.items():
        s = summary(o["pts"], o["wins"])
        res[W] = dict(s, tried=o["tried"], filled=o["filled"], fill_rate=o["filled"] / o["tried"] if o["tried"] else None,
                      paid_more=o["paid_more"] / o["filled"] if o["filled"] else None, state=_verdict(s, base))
    best = max(res.values(), key=lambda r: (r["state"] == "found", r["pnl"]))
    return {"windows": res, "base": base, "best": next(W for W, r in res.items() if r is best),
            "state": best["state"], "progress": min(1.0, best["n"] / NEED)}


# ---------- C. the room that pays most, by price ----------
def shading(db, since):
    rows = db.execute(
        "SELECT f.limit_price, f.side, f.want, f.cost1, f.fee1, f.cost2, f.fee2, f.cost3, f.fee3, f.cost5, f.fee5, f.wide, w.winner,"
        " f.asset || f.minutes || f.window_start FROM fill_tests f LEFT JOIN windows w ON w.asset = f.asset"
        " AND w.minutes = f.minutes AND w.window_start = f.window_start WHERE f.mode = 'paper' AND f.ts >= ?", (since,)).fetchall()
    cols = {1: (3, 4), 2: (5, 6), 3: (7, 8), 5: (9, 10)}
    out = []
    for lo, hi in BANDS:
        rs = [r for r in rows if r[0] is not None and lo <= r[0] < hi]
        band = {"lo": lo, "hi": hi, "refused": len(rs), "rooms": {}}
        for k in ROOMS:
            ci, fi = cols[k]
            tried = [r for r in rs if k <= 2 or r[11]]
            pts = [(r[13], r[2] * (r[12] == r[1]) - r[ci] - r[fi], r[ci] + r[fi]) for r in tried
                   if r[ci] is not None and r[12] in ("up", "down")]
            s = summary(pts, sum(1 for r in tried if r[ci] is not None and r[12] == r[1]))
            band["rooms"][str(k)] = dict(s, tried=len(tried), would_fill=sum(1 for r in tried if r[ci] is not None))
        ok = [(k, r) for k, r in band["rooms"].items() if r["n"] >= 30 and (r["roi"] or 0) > 0]
        band["best"] = max(ok, key=lambda x: x[1]["pnl"])[0] if ok else None
        band["state"] = ("found" if band["best"] and (band["rooms"][band["best"]]["z"] or 0) >= Z
                         and band["rooms"][band["best"]]["n"] >= NEED else "promising" if band["best"] else
                         "nothing" if sum(r["n"] for r in band["rooms"].values()) >= NEED else "collecting")
        out.append(band)
    found = [b for b in out if b["state"] == "found"]
    n = max((r["n"] for b in out for r in b["rooms"].values()), default=0)
    return {"bands": out, "state": "found" if found else "promising" if any(b["best"] for b in out) else
            "nothing" if all(b["state"] == "nothing" for b in out) else "collecting", "progress": min(1.0, n / NEED)}


# ---------- D. standing in the book ----------
def bid_fill(db, asset, minutes, start, side, bid, t0, ticks, rule):
    """The bid at `bid` from t0 for BID_FOR seconds: (fill ts) when sellers traded under it, None when it was
    cancelled (our model no longer sees the edge at it) or never reached."""
    cancel = None
    for ts, left, fair_up, *_ in ticks:
        if fair_up is None:
            continue
        fair = fair_up if side == "up" else 1 - fair_up
        if fair - bid < rule.min_edge - 1e-9 or (left is not None and left < rule.min_seconds_left):
            cancel = ts
            break
    until = min(t0 + BID_FOR, cancel) if cancel else t0 + BID_FOR
    hit = db.execute("SELECT MIN(ts) FROM trades WHERE asset = ? AND minutes = ? AND window_start = ? AND outcome = ?"
                     " AND taker = 'SELL' AND price < ? AND ts > ? AND ts <= ?",
                     (asset, minutes, start, side, bid - 1e-9, t0, until)).fetchone()[0]
    return hit


def standing(db, since):
    signals = db.execute(
        "SELECT b.ts, b.asset, b.minutes, b.window_start, b.side, b.variant, b.shares, b.avg_price, w.winner FROM bets b"
        " LEFT JOIN windows w ON w.asset = b.asset AND w.minutes = b.minutes AND w.window_start = b.window_start"
        " WHERE b.mode = 'paper' AND b.status IN ('filled', 'rejected') AND b.ts >= ? AND b.shares > 0"
        " ORDER BY b.ts DESC LIMIT 5000", (since,)).fetchall()
    out = {str(u): {"tried": 0, "pts": [], "wins": 0} for u in BID_UNDER}
    taker_pts, taker_wins = [], 0
    for ts, asset, minutes, start, side, variant, shares, price, winner in signals:
        rule = _plain(variant)
        if not rule or winner not in ("up", "down") or not price:
            continue
        ticks = _ticks(db, asset, minutes, start, ts, ts + BID_FOR)
        for u in BID_UNDER:
            bid = round(price - u, 2)
            if bid <= 0.01:
                continue
            o = out[str(u)]
            o["tried"] += 1
            if bid_fill(db, asset, minutes, start, side, bid, ts, ticks, rule) is None:
                continue
            o["pts"].append((f"{asset}{minutes}{start}", shares * ((winner == side) - bid), shares * bid))
            o["wins"] += winner == side
    for ts, a, m, start, pnl, paid in db.execute(
            "SELECT ts, asset, minutes, window_start, pnl, cost + fee FROM bets WHERE mode = 'paper' AND status = 'filled'"
            " AND pnl IS NOT NULL AND ts >= ?", (since,)):
        taker_pts.append((f"{a}{m}{start}", pnl, paid))
        taker_wins += pnl > 0
    base = summary(taker_pts, taker_wins)
    res = {}
    for u, o in out.items():
        s = summary(o["pts"], o["wins"])
        res[u] = dict(s, tried=o["tried"], fill_rate=len(o["pts"]) / o["tried"] if o["tried"] else None, state=_verdict(s, base))
    best = max(res.values(), key=lambda r: (r["state"] == "found", r["pnl"]))
    return {"bids": res, "base": base, "best": next(u for u, r in res.items() if r is best), "seconds": BID_FOR,
            "state": best["state"], "progress": min(1.0, best["n"] / NEED)}


def study(db_path, now, arrive, fee_rate, fee_exp):
    from .store import connect
    db = connect(db_path)
    try:
        since = now - DAYS * 86400
        out = {"ts": now, "days": DAYS}
        for name, fn in (("racing", lambda: racing(db, since, arrive)), ("intent", lambda: intent(db, since, fee_rate, fee_exp)),
                         ("shading", lambda: shading(db, since)), ("standing", lambda: standing(db, since))):
            try:
                out[name] = fn()
            except Exception as e:  # (one experiment's failure never hides the others)
                out[name] = {"state": "collecting", "error": f"{type(e).__name__}: {e}"[:200]}
        return out
    finally:
        db.close()


def _go(conn, *args):
    import os
    try:
        os.nice(15)
    except Exception:
        pass
    try:
        conn.send(("ok", study(*args)))
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}"[:300]))
    finally:
        conn.close()


def study_apart(db_path, now, arrive, fee_rate, fee_exp, timeout=1800):
    """In a process of its own at the lowest priority: the trading never waits behind it."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_go, args=(b, db_path, now, arrive, fee_rate, fee_exp), daemon=True, name="race-study")
    p.start()
    b.close()
    try:
        if not a.poll(timeout):
            raise TimeoutError("the race study took too long")
        what, out = a.recv()
    finally:
        p.join(5)
        if p.is_alive():
            p.terminate()
    if what != "ok":
        raise RuntimeError(out)
    return out
