"""Chains and the floor price (the owner, 31.10.2026).

Chains: "wins come together and losses come in a chain". The switch it suggests: real money stops after K losing
rounds in a row, and starts again after M winning rounds in a row (the simulation keeps trading meanwhile, so the
wins are counted on it). Tried for every K and M on the rounds the strategies already traded in the simulation:
  - the rounds in the order they ended, each the strategies' profit and cost together (a round is one result);
  - what the switch kept, against what the same money made on average ("over the average": a switch that keeps the
    better rounds is worth something, one that only trades less isn't);
  - the honest bar: the same rounds shuffled many times (any chain in them is gone) and the same switch run on
    them; a real chain has to beat almost every shuffle, and hold in the old half and the new half alike;
  - and the plain question under it: after a win, is a win more likely than after a loss? (a runs test)

The floor: real money's lowest price a share may be bought at, from the trades themselves (the owner, 31.10.2026:
"the system should do it by itself, as part of the research"). The strategies' trades of the last days in price
bands, cheapest first; while a band clearly loses (enough trades, losing on the dollar, beyond chance), the floor
rises over it. A band that stops losing brings it down again on the next pass.
"""
import math
import random

KS = (1, 2, 3, 4)  # losing rounds in a row that stop real money
MS = (1, 2, 3)  # winning rounds in a row (on the simulation) that start it again
NEED = 150  # rounds before a verdict
SHUFFLES = 300
P_FOUND, P_PROMISING = 0.01, 0.05
DAYS = 14

FLOOR_DAYS = 7
FLOOR_BANDS = (0, 10, 20, 30, 40, 50)  # cents: the bands [0, 10), [10, 20) ... and the floor at most 50
FLOOR_MIN_N = 30
FLOOR_Z = -1.0  # a band's loss beyond chance (its trades' return on the dollar, as a z score)
OWNERS = ("manual", "lockguard", "autojoin")  # (the owner's own buys and the locks: not a strategy's choice)


def _run(seq, k, m):
    """The switch over rounds [(pnl, paid)]: (pnl, paid, rounds) of what real money kept."""
    on, lost, won = True, 0, 0
    pnl = paid = 0.0
    n = 0
    for p, c in seq:
        if on:
            pnl += p
            paid += c
            n += 1
            lost = lost + 1 if p <= 0 else 0
            if lost >= k:
                on, won = False, 0
        else:
            won = won + 1 if p > 0 else 0
            if won >= m:
                on, lost = True, 0
    return pnl, paid, n


def _roi(seq):
    paid = sum(c for _, c in seq)
    return sum(p for p, _ in seq) / paid if paid else 0.0


def _over(seq, k, m):
    pnl, paid, n = _run(seq, k, m)
    return pnl - _roi(seq) * paid


def study(seq, shuffles=SHUFFLES, seed=7):
    """seq: the rounds [(pnl, paid)] in the order they ended."""
    n = len(seq)
    total, paid_all = sum(p for p, _ in seq), sum(c for _, c in seq)
    out = {"rounds": n, "need": NEED, "pnl": total, "paid": paid_all, "rows": [], "best": None,
           "progress": min(1.0, n / NEED), "state": "collecting"}
    if n < 20 or not paid_all:
        return out
    wins = [p > 0 for p, _ in seq]
    after_w = [b for a, b in zip(wins, wins[1:]) if a]
    after_l = [b for a, b in zip(wins, wins[1:]) if not a]
    n1, n2 = sum(wins), n - sum(wins)
    runs = 1 + sum(1 for a, b in zip(wins, wins[1:]) if a != b)
    mu = 2 * n1 * n2 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1) if n > 1 else 0
    out.update(win_rate=n1 / n, win_after_win=sum(after_w) / len(after_w) if after_w else None,
               win_after_loss=sum(after_l) / len(after_l) if after_l else None,
               runs_z=(runs - mu) / math.sqrt(var) if var > 0 else None)  # (below 0: fewer runs than chance, chains)
    rng = random.Random(seed)
    shuffled = [rng.sample(seq, n) for _ in range(shuffles)]
    half = n // 2
    for k in KS:
        for m in MS:
            pnl, paid, kept = _run(seq, k, m)
            over = pnl - _roi(seq) * paid
            p = (1 + sum(1 for s in shuffled if _over(s, k, m) >= over)) / (1 + shuffles)
            out["rows"].append({"k": k, "m": m, "pnl": pnl, "paid": paid, "kept": kept, "roi": pnl / paid if paid else None,
                                "over": over, "p": p, "gain": pnl - total, "old": _over(seq[:half], k, m),
                                "new": _over(seq[half:], k, m)})
    best = max(out["rows"], key=lambda r: r["over"])
    out["best"] = [best["k"], best["m"]]
    if n >= NEED:
        out["state"] = ("found" if best["p"] <= P_FOUND and best["old"] > 0 and best["new"] > 0 and best["gain"] > 0
                        else "promising" if best["p"] <= P_PROMISING and best["over"] > 0 else "nothing")
    return out


def floor(bets):
    """bets: [(price, pnl, paid)]. The bands, cheapest first, and the floor in cents (0: none)."""
    bands, cents = [], 0
    rising = True
    for lo, hi in zip(FLOOR_BANDS, FLOOR_BANDS[1:]):
        rows = [(p, c) for price, p, c in bets if lo <= price * 100 < hi and c]
        n = len(rows)
        r = [p / c for p, c in rows]
        mean = sum(r) / n if n else None
        sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1)) if n > 1 else 0.0
        z = (mean / sd * math.sqrt(n) if sd > 0 else -math.inf if mean < 0 else None) if n > 1 else None  # (all lost alike: sure)
        paid = sum(c for _, c in rows)
        losing = n >= FLOOR_MIN_N and paid and sum(p for p, _ in rows) < 0 and z is not None and z <= FLOOR_Z
        if rising and losing:
            cents = hi
        else:
            rising = False
        bands.append({"lo": lo, "hi": hi, "n": n, "pnl": sum(p for p, _ in rows), "paid": paid,
                      "roi": sum(p for p, _ in rows) / paid if paid else None, "wins": sum(1 for p, _ in rows if p > 0),
                      "z": None if z is None or math.isinf(z) else z, "losing": bool(losing)})
    return {"cents": cents, "bands": bands, "trades": len(bets), "days": FLOOR_DAYS}


def study_db(db_path, now, keys):
    """keys: the (asset, minutes, variant) the chains are read on (real money's strategies, else the simulation's)."""
    from .store import connect
    db = connect(db_path)
    try:
        rounds = {}
        for a, m, v, w, pnl, paid in db.execute(
                "SELECT asset, minutes, variant, window_start, pnl, cost + fee FROM bets WHERE mode = 'paper'"
                " AND status = 'filled' AND pnl IS NOT NULL AND minutes > 0 AND ts >= ?", (now - DAYS * 86400,)):
            if (a, m, v) in keys:
                r = rounds.setdefault((w + m * 60, a, m), [0.0, 0.0])
                r[0] += pnl
                r[1] += paid
        seq = [(p, c) for _, (p, c) in sorted(rounds.items())]
        own = ",".join(f"'{x}'" for x in OWNERS)
        bets = db.execute(
            "SELECT avg_price, pnl, cost + fee FROM bets WHERE mode IN ('paper', 'live') AND status = 'filled'"
            f" AND pnl IS NOT NULL AND minutes > 0 AND variant NOT IN ({own}) AND ts >= ? AND avg_price > 0",
            (now - FLOOR_DAYS * 86400,)).fetchall()
    finally:
        db.close()
    return {"ts": now, "chains": study(seq), "floor": floor(bets), "keys": len(keys)}


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


def study_apart(db_path, now, keys, timeout=900):
    """In a process of its own at the lowest priority (the shuffles take the processor a few seconds)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_go, args=(b, db_path, now, set(keys)), daemon=True, name="chains")
    p.start()
    b.close()
    try:
        if not a.poll(timeout):
            raise TimeoutError("the chains study took too long")
        what, out = a.recv()
    finally:
        p.join(5)
        if p.is_alive():
            p.terminate()
    if what != "ok":
        raise RuntimeError(out)
    return out
