"""Built for real money (the owner, 30.10.2026: "the simulation makes money because everything gets in and on its
conditions; real money doesn't. Take the winning strategies' formula, put real money's problems into it, and find the
way around them or the way to use them: the strategy that makes money steadily with real money").

Not a lab of rules but a set of ideas, each tested on real money's own log: every real order (the ones that got in and
the ones the book refused), the round's real winner, the prices recorded each second around it and what each refused
order would have needed. Every idea runs again on each new real order. A panel of a hundred (traders of these markets,
market makers, execution and latency engineers, statisticians, risk managers) set them before any data:

  1. The fills that come: is what gets in worse than what doesn't (the book gives us the trades the fast ones left)?
  2. Each strategy as real money gets it: its share that gets in times what those make: dollars an order sent.
  3. Calm or moving: an order sent while the price is running is the race's; one sent in a calm second isn't.
  4. Waiting in the book after a refusal: a bid at the price seen, 30 seconds, filled when sellers come back to it.
  5. Trying again within the edge: the next 15 seconds, an offer still within the strategy's edge.
  6. One order a round and side: when several strategies fire together they race each other for the same shares.
  7. When in the round: the share that gets in, and its result, by the time left.
  8. Paying more: what a cent or more over the price seen would have filled, and made.

And at the bottom, the answer they add up to: which strategies, how to enter, when not to: the strategy for real
money, with what it would have made a day on the log (it grows with each new real order).
"""

DAYS = 14
NEED = 100  # real orders before the answer is read
NEED_IDEA = 40  # cases before an idea is judged
MOVE = 0.02  # the side's offer moved this much over the 3 seconds before: "moving"
REST_FOR = 30
RETRY_FOR = 15
ROOMS = (1, 2, 3, 5, 10)


def _fee(rate, exp):
    return lambda p: rate * (p * (1 - p)) ** exp


def _edge(variant):
    try:
        return float(str(variant).split("|")[1])
    except (IndexError, ValueError):
        return None


def _orders(db, since):
    rows = db.execute(
        "SELECT b.id, b.ts, b.asset, b.minutes, b.window_start, b.side, b.seconds_left, b.fair_prob, b.avg_price, b.shares,"
        " b.cost, b.fee, b.status, b.pnl, b.variant, b.seen, w.winner FROM bets b LEFT JOIN windows w ON w.asset = b.asset"
        " AND w.minutes = b.minutes AND w.window_start = b.window_start WHERE b.mode = 'live' AND b.minutes > 0"
        " AND b.status IN ('filled', 'rejected') AND b.ts >= ? ORDER BY b.ts", (since,)).fetchall()
    out = []
    for (i, ts, a, m, w, side, left, fair, price, shares, cost, fee, status, pnl, variant, seen, winner) in rows:
        if winner not in ("up", "down") or not price or not cost:
            continue
        won = side == winner
        shares = shares or cost / price
        res = pnl if status == "filled" and pnl is not None else (shares if won else 0.0) - (cost + (fee or 0.0))
        out.append({"id": i, "ts": ts, "asset": a, "minutes": m, "window": w, "side": side, "left": left or 0.0,
                    "fair": fair, "price": price, "seen": seen or price, "shares": shares, "paid": cost + (fee or 0.0),
                    "filled": status == "filled", "res": res, "won": won, "variant": variant or ""})
    return out


def _sum(xs):
    n = len(xs)
    if not n:
        return {"n": 0}
    pnl = sum(x["res"] for x in xs)
    paid = sum(x["paid"] for x in xs)
    return {"n": n, "pnl": round(pnl, 2), "roi": pnl / paid if paid else None, "win": sum(1 for x in xs if x["won"]) / n}


def _ticks(db, o, t0, t1):
    col = "up_ask" if o["side"] == "up" else "down_ask"
    return db.execute(f"SELECT ts, {col}, fair_up FROM ticks WHERE asset = ? AND minutes = ? AND window_start = ?"
                      " AND ts BETWEEN ? AND ? ORDER BY ts", (o["asset"], o["minutes"], o["window"], t0, t1)).fetchall()


def _state(n, better, need=NEED_IDEA):
    return "collecting" if n < need else "found" if better else "nothing"


def study(db, now, fee_rate=0.07, fee_exp=1.0):
    fee = _fee(fee_rate, fee_exp)
    O = _orders(db, now - DAYS * 86400)
    F = [o for o in O if o["filled"]]
    R = [o for o in O if not o["filled"]]
    tracks = []

    # 1. The fills that come.
    f, r = _sum(F), _sum(R)
    worse = bool(f.get("n") and r.get("n") and f["roi"] is not None and r["roi"] is not None and r["roi"] > f["roi"] + 0.05)
    tracks.append({"key": "adverse", "n": len(O), "need": NEED_IDEA * 2, "filled": f, "refused": r,
                   "state": _state(min(f.get("n", 0), r.get("n", 0)), worse), "worse": worse})

    # 2. Each strategy as real money gets it.
    by = {}
    for o in O:
        by.setdefault((o["asset"], o["minutes"], o["variant"]), []).append(o)
    strat = []
    for (a, m, v), xs in by.items():
        got = [x for x in xs if x["filled"]]
        strat.append({"asset": a, "minutes": m, "variant": v, "sent": len(xs), "filled": len(got), "fill": len(got) / len(xs),
                      "pnl": round(sum(x["res"] for x in got), 2),
                      "per_sent": sum(x["res"] for x in got) / len(xs), "sim": round(sum(x["res"] for x in xs), 2)})
    strat.sort(key=lambda s: -s["per_sent"])
    good = [s for s in strat if s["sent"] >= 20 and s["per_sent"] > 0]
    tracks.append({"key": "strategies", "n": len(O), "need": NEED, "rows": strat[:20],
                   "state": _state(max((s["sent"] for s in strat), default=0), bool(good), 20)})

    # 3. Calm or moving, 4. waiting in the book, 5. trying again.
    calm, moving, rest, retry = [], [], [], []
    for o in O:
        T = _ticks(db, o, o["ts"] - 4, o["ts"] + REST_FOR)
        before = [a for ts, a, _ in T if ts <= o["ts"] and a]
        if len(before) >= 2:
            (moving if abs(before[-1] - before[0]) >= MOVE - 1e-9 else calm).append(o)
        if o["filled"]:
            continue
        # (a bid at the price seen, live after a second, filled once an offer comes to it or under: no fee)
        hit = next((ts for ts, a, _ in T if ts >= o["ts"] + 1 and a and a <= o["seen"] + 1e-9), None)
        sh = o["paid"] / o["seen"] if o["seen"] else 0
        if hit is not None:
            rest.append(dict(o, res=(sh if o["won"] else 0.0) - sh * o["seen"], paid=sh * o["seen"]))
        edge = _edge(o["variant"])
        for ts, a, fu in T:
            if ts < o["ts"] + 1 or ts > o["ts"] + RETRY_FOR or not a or fu is None or edge is None:
                continue
            prob = fu if o["side"] == "up" else 1 - fu
            if prob - a - fee(a) >= edge - 1e-9:
                per = a + fee(a)
                n = o["paid"] / per
                retry.append(dict(o, res=(n if o["won"] else 0.0) - n * per, paid=n * per, price=a))
                break
    c, mv = _sum(calm), _sum(moving)
    calm_better = bool(c.get("n") and mv.get("n") and (c["pnl"] / c["n"]) > (mv["pnl"] / mv["n"]) + 0.05)
    fill_c = sum(1 for o in calm if o["filled"]) / len(calm) if calm else None
    fill_m = sum(1 for o in moving if o["filled"]) / len(moving) if moving else None
    tracks.append({"key": "calm", "n": len(calm) + len(moving), "need": NEED_IDEA * 2, "calm": dict(c, fill=fill_c),
                   "moving": dict(mv, fill=fill_m), "state": _state(min(len(calm), len(moving)), calm_better), "better": calm_better})
    rs = _sum(rest)
    tracks.append({"key": "rest", "n": len(R), "need": NEED_IDEA, "tried": len(R), "got": rs,
                   "state": _state(len(R), bool(rs.get("n") and rs["pnl"] > 0))})
    rt = _sum(retry)
    tracks.append({"key": "retry", "n": len(R), "need": NEED_IDEA, "tried": len(R), "got": rt,
                   "state": _state(len(R), bool(rt.get("n") and rt["pnl"] > 0))})

    # 6. One order a round and side.
    groups, cur = [], []
    for o in sorted(O, key=lambda x: (x["asset"], x["minutes"], x["window"], x["side"], x["ts"])):
        if cur and (o["asset"], o["minutes"], o["window"], o["side"]) == (cur[0]["asset"], cur[0]["minutes"], cur[0]["window"],
                                                                          cur[0]["side"]) and o["ts"] - cur[-1]["ts"] <= 2:
            cur.append(o)
        else:
            if len(cur) > 1:
                groups.append(cur)
            cur = [o]
    if len(cur) > 1:
        groups.append(cur)
    first = [g[0] for g in groups]
    later = [x for g in groups for x in g[1:]]
    ff = sum(1 for o in first if o["filled"]) / len(first) if first else None
    fl = sum(1 for o in later if o["filled"]) / len(later) if later else None
    together = bool(ff is not None and fl is not None and fl < ff - 0.15)
    tracks.append({"key": "together", "n": len(groups), "need": 20, "groups": len(groups), "first_fill": ff,
                   "later_fill": fl, "state": _state(len(groups), together, 20), "better": together})

    # 7. When in the round.
    buckets = []
    for lo, hi, name in ((0, 30, "0-30"), (30, 60, "30-60"), (60, 120, "60-120"), (120, 1e9, "120+")):
        xs = [o for o in O if lo <= o["left"] < hi]
        got = [x for x in xs if x["filled"]]
        buckets.append({"name": name, "sent": len(xs), "fill": len(got) / len(xs) if xs else None,
                        "per_sent": sum(x["res"] for x in got) / len(xs) if xs else None})
    bad_time = [b["name"] for b in buckets if b["sent"] >= 20 and (b["per_sent"] or 0) < 0]
    tracks.append({"key": "timing", "n": len(O), "need": NEED, "buckets": buckets, "avoid": bad_time,
                   "state": _state(len(O), bool(bad_time), NEED)})

    # 8. Paying more.
    tests = db.execute("SELECT f.ts, f.asset, f.minutes, f.window_start, f.side, f.want, f.limit_price, f.need, w.winner FROM fill_tests f"
                       " LEFT JOIN windows w ON w.asset = f.asset AND w.minutes = f.minutes AND w.window_start = f.window_start"
                       " WHERE f.mode = 'live' AND f.ts >= ?", (now - DAYS * 86400,)).fetchall()
    rooms = {}
    for c_ in ROOMS:
        got = [(want, lim + c_ / 100, winner == side) for ts, a, m, w, side, want, lim, need_, winner in tests
               if need_ is not None and need_ < 99 and need_ <= lim + c_ / 100 + 1e-9 and winner in ("up", "down")]
        pnl = sum(want * (1.0 if won else 0.0) - want * (p + fee(p)) for want, p, won in got)
        rooms[str(c_)] = {"n": len(got), "pnl": round(pnl, 2)}
    best_room = max((k for k in rooms if rooms[k]["n"] >= 10), key=lambda k: rooms[k]["pnl"], default=None)
    tracks.append({"key": "room", "n": len(tests), "need": NEED_IDEA, "rooms": rooms, "best": best_room,
                   "state": _state(len(tests), bool(best_room and rooms[best_room]["pnl"] > 0))})

    # The answer: what the ideas found, put together, and what it would have made on the log.
    days = max(1.0, (now - min((o["ts"] for o in O), default=now)) / 86400)
    pick = {(s["asset"], s["minutes"], s["variant"]) for s in good}
    chosen = [o for o in O if (not pick or (o["asset"], o["minutes"], o["variant"]) in pick)]
    if bad_time:
        chosen = [o for o in chosen if not any(_in(o["left"], b) for b in bad_time)]
    if calm_better:
        calm_ids = {o["id"] for o in calm}
        chosen = [o for o in chosen if o["id"] in calm_ids]
    base = sum(o["res"] for o in chosen if o["filled"])
    extra, how, code = 0.0, "לקנות מההצעה כמו היום", ""
    ids = {o["id"] for o in chosen}
    add_rest = sum(x["res"] for x in rest if x["id"] in ids)
    add_retry = sum(x["res"] for x in retry if x["id"] in ids)
    if max(add_rest, add_retry) > 0:
        extra, how, code = (add_rest, "אחרי דחייה: הצעת קנייה שמחכה בספר במחיר שנראה, 30 שניות", f"r{REST_FOR}") if add_rest >= add_retry \
            else (add_retry, "אחרי דחייה: לנסות שוב 15 שניות, בכל הצעה שעוד בתוך היתרון", f"i{RETRY_FOR}")
    # (the final check's first half: the same answer on the later 30% of the log alone, which chose nothing)
    cut = sorted(o["ts"] for o in O)[int(len(O) * 0.7)] if O else now
    late = [o for o in chosen if o["ts"] >= cut]
    late_ids = {o["id"] for o in late}
    late_pnl = sum(o["res"] for o in late if o["filled"]) + sum(
        x["res"] for x in (rest if code.startswith("r") else retry if code else []) if x["id"] in late_ids)
    late_days = max(0.5, (now - cut) / 86400)
    ready = len(O) >= NEED
    final = {"ready": ready, "orders": len(O), "need": NEED, "strategies": [list(k) for k in sorted(pick)], "how": how, "together": together,
             "calm_only": calm_better, "avoid": bad_time, "room": best_room if best_room and rooms[best_room]["pnl"] > 0 else None,
             "pnl": round(base + extra, 2), "per_day": round((base + extra) / days, 2),
             "now_per_day": round(sum(o["res"] for o in O if o["filled"]) / days, 2), "days": round(days, 1),
             "how_code": code, "late_per_day": round(late_pnl / late_days, 2), "late_orders": len(late)}
    done = sum(1 for t in tracks if t["state"] in ("found", "nothing"))
    return {"ts": now, "tracks": tracks, "final": final, "progress": done / len(tracks), "orders": len(O)}


def _in(left, name):
    lo, hi = {"0-30": (0, 30), "30-60": (30, 60), "60-120": (60, 120), "120+": (120, 1e9)}[name]
    return lo <= left < hi


def study_db(db_path, now, fee_rate, fee_exp):
    from .store import connect
    db = connect(db_path)
    try:
        return study(db, now, fee_rate, fee_exp)
    finally:
        db.close()


def _go(conn, *args):
    import os
    try:
        os.nice(15)
    except Exception:
        pass
    try:
        conn.send(("ok", study_db(*args)))
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}"[:300]))
    finally:
        conn.close()


def study_apart(db_path, now, fee_rate, fee_exp, timeout=900):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_go, args=(b, db_path, now, fee_rate, fee_exp), daemon=True, name="livefit")
    p.start()
    b.close()
    try:
        if not a.poll(timeout):
            raise TimeoutError("the real-money study took too long")
        what, out = a.recv()
    finally:
        p.join(5)
        if p.is_alive():
            p.terminate()
    if what != "ok":
        raise RuntimeError(out)
    return out
