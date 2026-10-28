"""The workshop (the owner, 29.10.2026): a strategy that makes money, cloned into the lab to be made better,
without touching it where it runs. Two questions: more trades (it misses many), and more precise ones.

- candidates(): the versions of it tried, set before any data: each one step away on one thing the lab
  can run as is (the edge needed, the entry window, several entries, the least chance, a price group, a
  risk/reward floor), and a few pairs of them. Each goes into the live lab beside the clone (the same rule
  under its own record), so the answer comes on the lab's realistic fills, a trade at a time.
- replay(): the same versions at once on the recorded seconds (two weeks of every market's model and best
  prices, and each round's winner): an early answer, not a verdict (it buys at the best price on record,
  however much the book really held). Judged on the later 40% of the rounds only, against the original on
  the same rounds (paired), after an earlier 60% it is shown but not judged on.
- misses(): where real money and the lab parted: the rounds the lab's copy traded and real money didn't,
  what they made in the lab, and which of them real money tried and the book refused (and whether a cent
  or two more would have filled it).
"""

import dataclasses
import math

from . import lab
from .model import taker_fee_per_share

SPLIT = 0.6  # the replay's earlier part (shown) / later part (judged)
STAKE = 10.0


def candidates(key, minutes):
    """[{"key", "goal", "what"}]: the versions of `key` the workshop tries (goal: "more" / "precise" / "both")."""
    v = lab.Variant.from_key(key)
    v = dataclasses.replace(v, filter_ratio=0.0, stake=0.0, fill="")
    length = minutes * 60
    out = []

    def add(goal, what, **ch):
        k = dataclasses.replace(v, **ch).key
        if k != v.key and k not in [x["key"] for x in out]:
            out.append({"key": k, "goal": goal, "what": what})
    # More trades: a lower edge, an earlier start, several entries.
    for de in (-0.02, -0.04):
        if v.min_edge + de >= 0.04:
            add("more", f"פער של {(v.min_edge + de) * 100:.0f} נקודות במקום {v.min_edge * 100:.0f}", min_edge=round(v.min_edge + de, 2))
    for hi in (int(v.max_seconds_left * 1.5), v.max_seconds_left * 2):
        hi = min(hi, length - 30)
        if hi > v.max_seconds_left:
            add("more", f"מתחיל לחפש {hi} שניות לפני הסגירה במקום {v.max_seconds_left}", max_seconds_left=hi)
    if v.resolution == "twap":
        add("more", f"עד {lab.MULTI_ENTRIES} כניסות בסבב (שליש מהסכום בכל אחת) במקום אחת", resolution="twap3")
    # More precise: our model surer of the side, a price group, a risk/reward floor, a higher edge.
    for c in (0.60, 0.70):
        add("precise", f"רק כשהסיכוי שלנו לצד {c * 100:.0f}% ומעלה", min_chance=c)
    for lo, hi in ((0.30, 0.85), (0.40, 0.80)):
        add("precise", f"רק במחיר {lo * 100:.0f}¢–{hi * 100:.0f}¢ למניה (בלי הזולות והיקרות מדי)", price_lo=lo, price_hi=hi)
    add("precise", "רק כשניצחון מרוויח לפחות $3 על כל $10 (קונים עד 77¢)", min_ratio=0.3)
    add("precise", f"פער של {(v.min_edge + 0.02) * 100:.0f} נקודות במקום {v.min_edge * 100:.0f}", min_edge=round(v.min_edge + 0.02, 2))
    # Both: more of them, the shaky ones left out.
    if v.min_edge - 0.02 >= 0.04:
        add("both", f"פער {(v.min_edge - 0.02) * 100:.0f} + סיכוי 70% ומעלה", min_edge=round(v.min_edge - 0.02, 2), min_chance=0.70)
        add("both", f"פער {(v.min_edge - 0.02) * 100:.0f} + מחיר 30¢–85¢", min_edge=round(v.min_edge - 0.02, 2), price_lo=0.30, price_hi=0.85)
    hi15 = min(int(v.max_seconds_left * 1.5), length - 30)
    if hi15 > v.max_seconds_left:
        add("both", f"מתחיל {hi15} שניות לפני + סיכוי 70% ומעלה", max_seconds_left=hi15, min_chance=0.70)
    return out


def _decide(v, fair_up, up_ask, down_ask, fee_rate, fee_exp):
    """The rule's buy on these best prices: (side, ask, paid a share with the fee, chance) or None."""
    best = None
    for side, p, ask in (("up", fair_up, up_ask), ("down", 1 - fair_up, down_ask)):
        if ask is None or not 0 < ask < 1:
            continue
        paid = ask + taker_fee_per_share(ask, fee_rate, fee_exp)
        edge = p - paid
        if edge < v.min_edge - 1e-12:
            continue
        if v.ratio and 1 - paid < v.ratio * paid - 1e-12:
            continue
        if not v.passes(ask, p):
            continue
        if best is None or edge > best[4]:
            best = (side, ask, paid, p, edge)
    return best


def replay(db, asset, minutes, keys, fee_rate=0.07, fee_exp=1.0, since=None, stake=STAKE):
    """Each rule over the recorded seconds of a market: {key: [{"w", "ts", "side", "paid", "p", "pnl", "cost"}]}
    and the rounds run (those with a winner). One entry a round (three, a third each, for "twap3",
    lab.ENTRY_GAP seconds apart); bought at the best price on record, the whole amount."""
    rules = {k: lab.Variant.from_key(k) for k in keys}
    top = max(v.max_seconds_left for v in rules.values())
    winners = dict(db.execute("SELECT window_start, winner FROM windows WHERE asset = ? AND minutes = ? AND winner IN ('up', 'down')"
                              + (" AND window_start >= ?" if since else ""), (asset, minutes, since) if since else (asset, minutes)).fetchall())
    out = {k: [] for k in rules}
    rows = db.execute("SELECT window_start, ts, seconds_left, fair_up, up_ask, down_ask FROM ticks WHERE asset = ? AND minutes = ?"
                      " AND seconds_left >= 0 AND seconds_left <= ? AND fair_up IS NOT NULL" + (" AND window_start >= ?" if since else "")
                      + " ORDER BY window_start, ts", (asset, minutes, top, since) if since else (asset, minutes, top))
    state, cur, run = {}, None, set()
    for w, ts, left, fair, ua, da in rows:
        if w not in winners:
            continue
        if w != cur:
            cur, state = w, {}
            run.add(w)
        for k, v in rules.items():
            if not v.min_seconds_left <= left <= v.max_seconds_left:
                continue
            n, last = state.get(k, (0, -1e9))
            if n >= lab.entries(v) or ts - last < lab.ENTRY_GAP:
                continue
            d = _decide(v, fair, ua, da, fee_rate, fee_exp)
            if d is None:
                continue
            side, ask, paid, p, _ = d
            part = stake / lab.entries(v)
            shares = part / paid
            pnl = (shares - part) if winners[w] == side else -part
            out[k].append({"w": w, "ts": ts, "side": side, "paid": round(paid, 4), "p": round(p, 4), "pnl": round(pnl, 4), "cost": part})
            state[k] = (n + 1, ts)
    return out, sorted(run)


def _stats(trades):
    n, cost = len(trades), sum(t["cost"] for t in trades)
    pnl = sum(t["pnl"] for t in trades)
    roi, pess = lab.money_weighted([(t["pnl"], t["cost"]) for t in trades], 1.645) if trades else (None, None)
    return {"n": n, "wins": sum(1 for t in trades if t["pnl"] > 0), "win_rate": sum(1 for t in trades if t["pnl"] > 0) / n if n else None,
            "pnl": round(pnl, 2), "roi": roi, "pess": pess, "avg_paid": sum(t["paid"] for t in trades) / n if n else None}


def judge(results, rounds, base):
    """Each rule on the replay's earlier and later rounds, and against the original on the later ones, round
    by round (the same rounds: the difference of the two's money each round, its mean over its standard
    error). {key: {"early", "late", "vs": {"z", "extra"}, "verdict"}}."""
    if not rounds:
        return {}
    cut = rounds[int(len(rounds) * SPLIT)] if len(rounds) > 1 else rounds[0]
    late = [w for w in rounds if w >= cut]
    per = lambda trades: {w: sum(t["pnl"] for t in trades if t["w"] == w) for w in late}
    b = per(results.get(base, []))
    out = {}
    for k, trades in results.items():
        early, later = [t for t in trades if t["w"] < cut], [t for t in trades if t["w"] >= cut]
        row = {"early": _stats(early), "late": _stats(later), "all": _stats(trades)}
        if k != base:
            mine = per(trades)
            d = [mine[w] - b[w] for w in late]
            mean = sum(d) / len(d) if d else 0.0
            var = sum((x - mean) ** 2 for x in d) / (len(d) - 1) if len(d) > 1 else 0.0
            z = mean / math.sqrt(var / len(d)) if var > 0 else None
            row["vs"] = {"z": z, "extra": round(sum(d), 2), "more_trades": len(later) - sum(1 for t in results.get(base, []) if t["w"] >= cut)}
            bl = out.get(base, {}).get("late") or _stats([t for t in results.get(base, []) if t["w"] >= cut])
            better = z is not None and z >= 1.0 and sum(d) > 0
            worse = z is not None and z <= -1.0
            row["verdict"] = ("better" if better and (row["late"]["roi"] or 0) >= (bl["roi"] or 0) - 0.02 else
                              "worse" if worse else "same") if row["late"]["n"] >= 20 else "few"
        out[k] = row
    return out


def misses(db, asset, minutes, key, since):
    """Where real money and the lab's copy of the same rule parted since `since`: the rounds each traded, the
    lab's rounds real money didn't (what they made in the lab), and of those the ones real money tried and the
    book refused (fill_tests), with whether a cent or two more would have filled them."""
    q = "SELECT window_start, pnl, cost + fee FROM bets WHERE mode = ? AND variant = ? AND asset = ? AND minutes = ? AND ts >= ? AND shares > 0"
    labw = {}
    for w, pnl, cost in db.execute(q + " AND pnl IS NOT NULL", ("lab", key, asset, minutes, since)):
        x = labw.setdefault(w, [0.0, 0.0])
        x[0] += pnl or 0
        x[1] += cost or 0
    livew = {w for w, *_ in db.execute(q, ("live", key, asset, minutes, since))}
    refused = {}
    for w, c1, c2 in db.execute("SELECT window_start, cost1, cost2 FROM fill_tests WHERE mode = 'live' AND variant = ? AND asset = ?"
                                " AND minutes = ? AND ts >= ?", (key, asset, minutes, since)):
        r = refused.setdefault(w, [False, False])
        r[0] |= c1 is not None
        r[1] |= c2 is not None
    missed = [w for w in labw if w not in livew]
    tried = [w for w in missed if w in refused]
    return {"lab": len(labw), "live": len(livew), "both": len(set(labw) & livew), "missed": len(missed),
            "missed_pnl": round(sum(labw[w][0] for w in missed), 2), "missed_cost": round(sum(labw[w][1] for w in missed), 2),
            "refused": len(tried), "refused_pnl": round(sum(labw[w][0] for w in tried), 2),
            "room1": sum(1 for w in tried if refused[w][0]), "room2": sum(1 for w in tried if refused[w][1]),
            "not_tried": len(missed) - len(tried), "live_only": len(livew - set(labw))}


def fidelity(db, asset, minutes, key, rounds, results):
    """How close the replay is to the lab's own trades of the original on the same rounds: (lab trades, replay
    trades, the share of the lab's rounds the replay traded too, the two's win rates)."""
    if not rounds:
        return None
    rows = db.execute("SELECT window_start, pnl FROM bets WHERE mode = 'lab' AND variant = ? AND asset = ? AND minutes = ?"
                      " AND window_start >= ? AND window_start <= ? AND pnl IS NOT NULL AND shares > 0",
                      (key, asset, minutes, rounds[0], rounds[-1])).fetchall()
    if not rows:
        return None
    lw = {w for w, _ in rows}
    rw = {t["w"] for t in results.get(key, [])}
    return {"lab": len(rows), "replay": len(results.get(key, [])), "same_rounds": len(lw & rw) / len(lw),
            "lab_win": sum(1 for _, p in rows if (p or 0) > 0) / len(rows),
            "replay_win": (sum(1 for t in results.get(key, []) if t["pnl"] > 0) / len(results[key])) if results.get(key) else None}


def study(db_path, asset, minutes, base, keys, fee_rate, fee_exp, now):
    """The whole study from the database file (a process of its own on the server: the replay reads every
    recorded second of two weeks): the replay judged, its fidelity, and where real money missed."""
    from .store import connect
    db = connect(db_path)
    try:
        results, rounds = replay(db, asset, minutes, [base] + list(keys), fee_rate, fee_exp)
        first_live = db.execute("SELECT MIN(ts) FROM bets WHERE mode = 'live' AND variant = ? AND asset = ? AND minutes = ?",
                                (base, asset, minutes)).fetchone()[0]
        since = max(first_live or 0, now - 14 * 86400)
        return {"ts": now, "rounds": len(rounds), "first": rounds[0] if rounds else None, "last": rounds[-1] if rounds else None,
                "judge": judge(results, rounds, base), "fidelity": fidelity(db, asset, minutes, base, rounds, results),
                "misses": misses(db, asset, minutes, base, since) if first_live else None, "since": since}
    finally:
        db.close()
