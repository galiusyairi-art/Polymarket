"""The brain's layer: the choices made on top of the strategies (how much a trade, the most it may lose,
the price group, the least chance, the model's agreement, which strategies run), each either the
brain's (from the data, applied by itself) or the owner's. This module only reads the data and scores
the choices; engine.py applies them (Engine.brain_auto) and serves the brain hub.

The one rule for every choice: of the options with enough trades, the one keeping the most money
without its single best trade (a profit that isn't one lucky trade). A filter costs trades, so it has to
earn its place: it's on only while it does clearly better (SWITCH times the score, and MIN_GAIN more)
than no filter at all; otherwise the brain takes it off. And a filter in force changes to another only
when that one does clearly better too, so the brain doesn't flip back and forth.
"""

MIN_TRADES = 100
SWITCH = 1.25
MIN_GAIN = 25.0


def score(groups, keep):
    """The trades of the groups `keep(group)` says yes to: (score, trades, pnl, win rate)."""
    part = [g for g in groups if g["trades"] and keep(g)]
    n = sum(g["trades"] for g in part)
    pnl = sum(g["pnl"] for g in part)
    top = max((g["best"] for g in part if g.get("best") is not None), default=0.0)
    wins = sum(g["wins"] for g in part)
    return {"score": pnl - max(top, 0.0), "trades": n, "pnl": pnl, "win_rate": wins / n if n else None}


def pick_threshold(groups, options):
    """Of the thresholds `options` (None: no threshold, everything), the one scoring best: groups from
    the threshold up. Returns (threshold, its score) or (None, everything's score) when none does better."""
    everything = score(groups, lambda g: True)
    best = (None, everything)
    for t in options:
        sc = score(groups, lambda g, t=t: g["lo"] >= t - 1e-9)
        if sc["trades"] >= MIN_TRADES and sc["score"] > best[1]["score"]:
            best = (t, sc)
    return best


def better(new, current):
    """Is a new choice's score clearly better than the one in force (so worth a change)?"""
    return new["score"] >= max(current["score"] * SWITCH, current["score"] + MIN_GAIN)


def decide(current, current_score, best, best_score, everything):
    """A filter's next value: `best` (the best option, None: none qualifies) when it does clearly better
    than no filter, else None (no filter). The one in force is kept while it still earns its place and
    the new one isn't clearly better than it. Returns (value, changed)."""
    if everything.get("trades", MIN_TRADES) < MIN_TRADES:
        return current, False  # (too little data to say anything: what's in force stays)
    want = best if best is not None and better(best_score, everything) else None
    if current is None:
        return want, want is not None
    if not better(current_score, everything):
        return want, want != current  # (no longer earns its place)
    if want is not None and want != current and better(best_score, current_score):
        return want, True
    return current, False


def gap_groups(rows):
    """The makers' trades by how far our model's chance was over the price paid (rows: (gap, pnl)),
    in groups like the price table's."""
    edges = (-1.0, 0.0, 0.03, 0.06, 0.10, 1.0001)
    groups = [{"lo": lo, "hi": hi, "trades": 0, "wins": 0, "pnl": 0.0, "best": None, "worst": None}
              for lo, hi in zip(edges, edges[1:])]
    for gap, pnl in rows:
        g = next((g for g in groups if gap < g["hi"]), groups[-1])
        g["trades"] += 1
        g["wins"] += pnl > 0
        g["pnl"] += pnl
        g["best"] = pnl if g["best"] is None else max(g["best"], pnl)
        g["worst"] = pnl if g["worst"] is None else min(g["worst"], pnl)
    for g in groups:
        n = g["trades"]
        g.update(win_rate=g["wins"] / n if n else None, avg=g["pnl"] / n if n else None)
    return groups


def risk_groups(rounds):
    """Rounds (all strategies in one market's round together) by how much they had at risk (rows:
    (paid, pnl)): what a cap on a round's loss would have kept out."""
    edges = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1e12)
    groups = [{"lo": lo, "hi": hi, "trades": 0, "wins": 0, "pnl": 0.0, "best": None, "worst": None}
              for lo, hi in zip(edges, edges[1:])]
    for paid, pnl in rounds:
        g = next(g for g in groups if paid < g["hi"] + 1e-9)
        g["trades"] += 1
        g["wins"] += pnl > 0
        g["pnl"] += pnl
        g["best"] = pnl if g["best"] is None else max(g["best"], pnl)
        g["worst"] = pnl if g["worst"] is None else min(g["worst"], pnl)
    for g in groups:
        n = g["trades"]
        g.update(win_rate=g["wins"] / n if n else None, avg=g["pnl"] / n if n else None)
    return groups


def wallet_stake(balance, least=5.0):
    """Real money's amount a trade from the wallet: 2% of it, Polymarket's smallest order at least."""
    if not balance or balance <= 0:
        return None
    return max(least, float(int(balance * 0.02)))
