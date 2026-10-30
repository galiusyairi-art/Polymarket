"""Both sides, locked (the owner, 30.10.2026: "in one click I made a trade that can't lose; find me the bot that
does it every time"). Up and Down of a round always pay $1 together, so holding the same number of shares of
each, bought for less than $1 a pair in all (fees included), makes money whoever wins. Buying both at the same
moment almost never costs under $1 (the two offers add up to $1.01-1.02); the way in is through time: one side
now, the other when the price swings and it turns cheap. Five panels (30.10.2026) set three
ways to try, side by side in the lab on every round, with real orders' fills:

  A. Leg, then lock ("bothA"): one side (where our model sees an edge, or where it is cheap), then the same
     number of shares of the other side once the pair costs at most T, by taking the offer or by a bid that
     waits in the book (no fee). With "i": when it hasn't locked a minute before the close and our model says
     the first side is losing, the other side even at a small loss (up to INSURE a pair): the loss capped.
  B. Keep buying until locked ("bothB", the owner's "buys until it succeeds"): small slices of whichever side
     is cheap, more of a side only under its own average, at most twice the other side's shares, until the
     pair is locked at T or the budget is spent.
  C. Two bids ("bothC"): a bid on each side under our model's price, the two adding up to T; once one is
     filled, the other waits at what still locks the pair. A market maker's way: no fee on either side.

The honest part (the probability panel): before the second leg is in, the first is an ordinary bet. A price
that is fair moves up as often as down, so locking a profit of d after paying p happens about p/(p+d) of the
time, and the rest of the time the first leg can lose all it cost: on fair prices it all adds up to zero, less
the fees. What can make it more than zero: a first leg bought cheaper than our model's chance, makers' prices
(no fee, and the spread on our side), and Polymarket's prices swinging further than the coin's (people
panicking). Which of them is real here, the lab finds out; the replay below gives a first answer at once.

The decisions here (step) and the fills (fills) are the same for the lab and for the replay on the recorded
seconds, so the two measure the same thing.
"""

import math
from dataclasses import dataclass, field

SIDES = ("up", "down")
PAIR = 10.0  # shares a pair a round (pays $10 whoever wins)
SLICE = 2.0  # B: shares a slice
BUDGET = 30.0  # B: the most a round puts in, in dollars
RATIO = 2.0  # B: a side at most twice the other's shares
DROP = 0.05  # B: more of a side only this far under its own average (a share, fee included)
GAP = 2.0  # B: seconds between two of its orders
LAST = 5  # no new order this close to the end
INSURE_AT = 60  # A with "i": the last minute
INSURE = 1.05  # A with "i": the most a pair may cost then (a loss of 5c a pair, capped)
LEAST = 1.0  # shares: less is a crumb, not a fill


def _fee(rate, exp):
    return lambda p: rate * (p * (1.0 - p)) ** exp


def spec(resolution):
    """A lock rule's parts, from its name: {"kind": "A"/"B"/"C", "T", "trig", "wait", "insure"}."""
    r = resolution
    kind = r[4]
    if kind == "A":
        return {"kind": "A", "trig": r[5], "T": int(r[6:8]) / 100, "wait": r[8] == "w", "insure": r.endswith("i")}
    return {"kind": kind, "T": int(r[5:7]) / 100, "trig": "", "wait": kind == "C", "insure": False}


@dataclass
class Pos:
    """A rule's position in a round, and its orders on their way or resting."""
    sh: dict = field(default_factory=lambda: {"up": 0.0, "down": 0.0})
    paid: dict = field(default_factory=lambda: {"up": 0.0, "down": 0.0})
    takes: list = field(default_factory=list)  # [{"side", "shares", "limit", "due"}]: taking the offer
    bids: dict = field(default_factory=dict)  # side -> {"price", "shares", "live"}: waiting in the book
    last: float = -1e18  # (when its last order went: none yet)
    done: bool = False
    insured: bool = False
    tried: int = 0
    refused: int = 0
    scale: float = 1.0  # the amount: PAIR shares a pair (and B's slices and budget) times this

    @property
    def total(self):
        return self.paid["up"] + self.paid["down"]

    @property
    def locked(self):
        """The profit whoever wins (None: not locked)."""
        low = min(self.sh.values())
        return low - self.total if low >= LEAST and low - self.total >= -1e-9 else None


def _best(levels):
    return min((p for p, s in levels or () if s > 0), default=None)


def fills(pos, now, asks, fee):
    """The orders that reached the book by now meet it: an offer taken fills what is there up to its limit
    (fill-and-kill, like a real one); a bid fills when sellers trade under it (they went through it: the
    pessimistic way), at its price and with no fee. [(side, shares, cost, fee, top, how)]"""
    out, keep = [], []
    for o in pos.takes:
        if o["due"] > now:
            keep.append(o)
            continue
        got = cost = f = top = 0.0
        for price, size in sorted(asks.get(o["side"]) or ()):
            if price > o["limit"] + 1e-9 or got >= o["shares"] - 1e-9:
                break
            take = min(size, o["shares"] - got)
            got, cost, f, top = got + take, cost + take * price, f + take * fee(price), price
        if got >= LEAST:
            out.append((o["side"], got, cost, f, top, "take"))
        else:
            pos.refused += 1
    pos.takes = keep
    for side in list(pos.bids):
        b = pos.bids[side]
        ask = _best(asks.get(side))
        if now >= b["live"] and ask is not None and ask < b["price"] - 1e-9:
            out.append((side, b["shares"], b["shares"] * b["price"], 0.0, b["price"], "rest"))
            del pos.bids[side]
    for side, got, cost, f, _, _ in out:
        pos.sh[side] += got
        pos.paid[side] += cost + f
    return out


def _limit(x_max, fee, tick):
    """The highest price whose share, fee included, costs at most x_max (None: none)."""
    p = math.floor(x_max / tick + 1e-9) * tick
    while p >= tick - 1e-9 and p + fee(p) > x_max + 1e-12:
        p -= tick
    return round(p, 4) if p >= tick - 1e-9 else None


def _take(pos, side, shares, limit, now, delay):
    pos.takes.append({"side": side, "shares": shares, "limit": limit, "due": now + delay})
    pos.tried += 1
    pos.last = now


def _bid(pos, side, price, shares, now, delay):
    b = pos.bids.get(side)
    if b and abs(b["price"] - price) < 1e-9 and abs(b["shares"] - shares) < 1e-9:
        return
    pos.bids[side] = {"price": price, "shares": shares, "live": now + delay}  # (a new or moved bid: on its way again)
    if not b:
        pos.tried += 1


def step(v, pos, now, left, fair_up, asks, fee, tick=0.01, delay=0.5):
    """A lock rule's decision now (v: its lab.Variant; left: seconds to the close; fair_up: our model's chance of
    Up, or None). Adds orders to pos; fills() fills them."""
    if pos.done:
        return
    if pos.locked is not None:
        pos.done = True
        pos.bids.clear()
        return
    if left < LAST:
        pos.bids.clear()
        return
    if pos.takes:
        return  # (one order at a time: the next decision knows what the last one got)
    S = spec(v.resolution)
    best = {s: _best(asks.get(s)) for s in SIDES}
    fair = {"up": fair_up, "down": 1.0 - fair_up} if fair_up is not None else None
    in_window = v.min_seconds_left <= left <= v.max_seconds_left
    has = sum(1 for s in SIDES if pos.sh[s] >= LEAST)
    if not has:
        if S["kind"] == "C":
            _two_bids(v, S, pos, now, in_window, fair, best, tick, delay)
        elif in_window and now - pos.last >= GAP:
            _first(v, S, pos, now, fair, best, fee, delay, tick)
        return
    long = max(SIDES, key=lambda s: pos.sh[s])
    short = "down" if long == "up" else "up"
    need = pos.sh[long] - pos.sh[short]
    if S["kind"] == "C" and long in pos.bids:
        del pos.bids[long]
    if need < LEAST:
        if S["kind"] != "B":
            pos.done = True  # (both in, not for a profit: the insurance's capped loss)
            pos.bids.clear()
            return
    else:
        x_max = (S["T"] * pos.sh[long] - pos.total) / need  # the most a share of the short side may cost
        if S["insure"] and not pos.insured and left <= INSURE_AT and fair and fair[long] < 0.5:
            x_ins = (INSURE * pos.sh[long] - pos.total) / need
            if best[short] is not None and best[short] + fee(best[short]) <= x_ins + 1e-12:
                pos.bids.clear()
                pos.insured = True
                _take(pos, short, need, _limit(x_ins, fee, tick), now, delay)
                return
        if S["wait"]:
            price = math.floor(x_max / tick + 1e-9) * tick
            if best[short] is not None:
                price = min(price, best[short] - tick)  # (resting: never crossing into a taker's order)
            if price >= tick - 1e-9:
                _bid(pos, short, round(price, 4), need, now, delay)
            return
        if best[short] is not None and best[short] + fee(best[short]) <= x_max + 1e-12:
            _take(pos, short, need, _limit(x_max, fee, tick), now, delay)
            return
        if S["kind"] == "B" and pos.sh[long] and best[short] is not None and now - pos.last >= GAP:
            # (not the whole lock yet: a slice of the short side, if it pairs with the long side's average at T)
            avg_long = pos.paid[long] / pos.sh[long]
            if best[short] + fee(best[short]) + avg_long <= S["T"] + 1e-12:
                _take(pos, short, min(SLICE * pos.scale, need), best[short], now, delay)
                return
    if S["kind"] == "B":
        _average(v, pos, now, left, best, fee, delay)


def _first(v, S, pos, now, fair, best, fee, delay, tick):
    """A and B's first side: A where our model sees the edge (or the cheapest under its cap), B the cheapest
    side under its cap."""
    pick = None
    for s in SIDES:
        a = best[s]
        if a is None:
            continue
        if S["kind"] == "A" and S["trig"] == "m":
            if fair is None:
                continue
            edge = fair[s] - a - fee(a)
            if edge >= v.min_edge - 1e-9 and (pick is None or edge > pick[0]):
                pick = (edge, s)
        elif a <= v.min_edge + 1e-9 and (pick is None or -a > pick[0]):
            pick = (-a, s)
    if pick:
        # (up to the most the rule still pays, not just the price seen: an offer gone by the time the order
        # arrives is the usual refusal, and the next level up is still within the rule)
        s = pick[1]
        top = _limit(fair[s] - v.min_edge, fee, tick) if S["trig"] == "m" else round(v.min_edge, 4)
        _take(pos, s, (PAIR if S["kind"] == "A" else SLICE) * pos.scale, max(best[s], top or 0.0), now, delay)


def _average(v, pos, now, left, best, fee, delay):
    """B: another slice of a side that turned cheaper than its own average (or of the side it has none of yet,
    under its cap), within the budget and twice the other side's shares."""
    slice_, budget = SLICE * pos.scale, BUDGET * pos.scale
    if now - pos.last < GAP or not v.min_seconds_left <= left or pos.total >= budget:
        return
    for s in sorted(SIDES, key=lambda x: best[x] if best[x] is not None else 9):
        a = best[s]
        if a is None or a > v.min_edge + 1e-9 or pos.total + slice_ * (a + fee(a)) > budget:
            continue
        other = pos.sh["down" if s == "up" else "up"]
        if pos.sh[s] + slice_ > RATIO * max(other, slice_) + 1e-9:
            continue
        avg = pos.paid[s] / pos.sh[s] if pos.sh[s] else None
        if avg is None or a + fee(a) <= avg - DROP + 1e-12:
            _take(pos, s, slice_, a, now, delay)
            return


def _two_bids(v, S, pos, now, in_window, fair, best, tick, delay):
    """C before either fills: a bid on each side (1 - T) / 2 under our model's price, never crossing the offer."""
    if not in_window or fair is None:
        pos.bids.clear()
        return
    h = (1.0 - S["T"]) / 2
    for s in SIDES:
        price = math.floor((fair[s] - h) / tick + 1e-9) * tick
        if best[s] is not None:
            price = min(price, best[s] - tick)
        if price >= tick - 1e-9:
            _bid(pos, s, round(price, 4), PAIR * pos.scale, now, delay)
        else:
            pos.bids.pop(s, None)


# ---------------------------------------------------------------------------------------------------------
# The replay: every rule at once on the recorded seconds (each market's model and best prices, every second
# on the markets the strategies run, every 5 seconds on the others), against each round's real winner. A first
# answer, not a verdict: it takes the best offer on record whatever the book really held, and a recorded second
# is coarser than the lab's every update. Judged on the later 40% of the rounds, after an earlier 60% shown.

DAYS = 7
SPLIT = 0.6
NEED = 100  # rounds a rule's later part needs before a verdict
DEEP = 1e9  # (a recorded price has no depth: the whole order at it)


def _round(rules, rows, fee, delay, tick):
    """One round's recorded seconds through every rule: {key: Pos}."""
    pos = {v.key: Pos() for v in rules}
    for ts, left, fair_up, up_ask, down_ask in rows:
        asks = {"up": [(up_ask, DEEP)] if up_ask else [], "down": [(down_ask, DEEP)] if down_ask else []}
        for v in rules:
            p = pos[v.key]
            fills(p, ts, asks, fee)
            step(v, p, ts, left, fair_up, asks, fee, tick, delay)
    return pos


def _sum(points):
    n = len(points)
    if not n:
        return {"n": 0}
    pnl = sum(x for x, _, _ in points)
    paid = sum(p for _, p, _ in points)
    mean = pnl / n
    sd = math.sqrt(sum((x - mean) ** 2 for x, _, _ in points) / (n - 1)) if n > 1 else 0.0
    locked = [x for x, _, k in points if k]
    open_ = [x for x, _, k in points if not k]
    return {"n": n, "pnl": round(pnl, 2), "paid": round(paid, 2), "roi": pnl / paid if paid else None,
            "z": mean / sd * math.sqrt(n) if sd else None, "locked": len(locked), "lock_rate": len(locked) / n,
            "lock_avg": sum(locked) / len(locked) if locked else None, "open_avg": sum(open_) / len(open_) if open_ else None,
            "per_round": mean}


def replay(db, now, rules_for, fee_rate, fee_exp, days=DAYS, tick=0.01, delay=0.5):
    """rules_for(minutes): the lock rules. {"rows": [per rule and market length], "rounds", "state", ...}"""
    from . import lab
    fee = _fee(fee_rate, fee_exp)
    since = now - days * 86400
    winners = {(a, m, w): x for a, m, w, x in db.execute(
        "SELECT asset, minutes, window_start, winner FROM windows WHERE winner IN ('up', 'down') AND window_start >= ?", (since,))}
    markets = db.execute("SELECT DISTINCT asset, minutes FROM windows WHERE window_start >= ? AND minutes > 0", (since,)).fetchall()
    res = {}  # (key, minutes) -> [(window_start, pnl, paid, locked)]
    rounds = 0
    for asset, minutes in markets:
        rules = rules_for(minutes)
        if not rules:
            continue
        cur, rows = None, []

        def flush():
            nonlocal rounds
            if cur is None or (asset, minutes, cur) not in winners or len(rows) < 10:
                return
            winner = winners[(asset, minutes, cur)]
            rounds += 1
            for key, p in _round(rules, rows, fee, delay, tick).items():
                if p.total <= 0:
                    continue
                res.setdefault((key, minutes), []).append((cur, p.sh[winner] - p.total, p.total, p.locked is not None))
        # (a day at a time, each read whole before it is replayed: a read held open for minutes keeps the
        # database's write-ahead file from emptying, and it grows on the disk all that time)
        for day in range(int(since), int(now), 86400):
            for w, ts, left, fair_up, ua, da in db.execute(
                    "SELECT window_start, ts, seconds_left, fair_up, up_ask, down_ask FROM ticks WHERE asset = ? AND minutes = ?"
                    " AND window_start >= ? AND window_start < ? ORDER BY window_start, ts",
                    (asset, minutes, day, day + 86400)).fetchall():
                if w != cur:
                    flush()
                    cur, rows = w, []
                rows.append((ts, left, fair_up, ua, da))
        flush()
    out = []
    for (key, minutes), pts in res.items():
        pts.sort()
        cut = int(len(pts) * SPLIT)
        early = _sum([(x, p, k) for _, x, p, k in pts[:cut]])
        late = _sum([(x, p, k) for _, x, p, k in pts[cut:]])
        state = ("collecting" if late["n"] < NEED else "found" if (late["z"] or 0) >= 2.0 and late["pnl"] > 0
                 else "nothing" if late["pnl"] <= 0 else "promising")
        out.append({"key": key, "minutes": minutes, "rule": lab.describe(key), "all": _sum([(x, p, k) for _, x, p, k in pts]),
                    "early": early, "late": late, "state": state})
    out.sort(key=lambda r: -(r["late"].get("per_round") if r["late"]["n"] else -9))
    states = {r["state"] for r in out}
    state = ("found" if "found" in states else "promising" if "promising" in states else
             "nothing" if out and states == {"nothing"} else "collecting")
    most = max((r["late"]["n"] for r in out), default=0)
    return {"ts": now, "days": days, "rounds": rounds, "rows": out, "state": state,
            "progress": 1.0 if state in ("found", "nothing") else min(1.0, most / NEED)}


def study(db_path, now, fee_rate, fee_exp):
    from . import lab
    from .store import connect
    db = connect(db_path)
    try:
        return replay(db, now, lab.both_for, fee_rate, fee_exp)
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


def study_apart(db_path, now, fee_rate, fee_exp, timeout=1800):
    """In a process of its own at the lowest priority: the trading never waits behind it."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_go, args=(b, db_path, now, fee_rate, fee_exp), daemon=True, name="lock-replay")
    p.start()
    b.close()
    try:
        if not a.poll(timeout):
            raise TimeoutError("the lock replay took too long")
        what, out = a.recv()
    finally:
        p.join(5)
        if p.is_alive():
            p.terminate()
    if what != "ok":
        raise RuntimeError(out)
    return out
