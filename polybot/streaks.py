"""Streaks: do wins and losses come in runs, across every strategy at once? (the owner's idea of 27.10.2026)

The owner saw whole stretches when every strategy won together, and stretches when they all lost. If
that is real, a switch is worth something: trade only while the recent results of all the strategies
together are green ("hot"), stop the moment they turn red ("cold"), and wait for green again.

The test, on trades already made (nothing new has to be collected):
  - every trade's signal is what the strategies' trades that had already ended said before it began:
    their profit over what they cost in the last L (minutes in Polybot, trading days in Atlas). The
    signal is read at the start of the trade's time bucket, so it never knows anything from later;
  - hot (that result over 0) against cold (0 or under), for a few L set before any data;
  - the honest bar (a placebo): the same signal, slid along the time line to where it has nothing to
    do with the trades, many times. Runs that show up by chance show up there too; a real one has to
    beat almost all the slid copies (p <= 0.01: four lookbacks, so a lucky one has a 4% chance), and
    hold in the old half and the new half of the data alike;
  - and what it would have caught: the share of the winning stretches' profit made while hot, and
    the share of the losing stretches' loss avoided while cold.
"""
import bisect
import random

P_FOUND = 0.01
P_PROMISING = 0.05
SHIFTS = 200


def from_events(events):
    """(entry, ended, pnl, paid) trades as study()'s two lists: by when they began and by when they ended."""
    events = [e for e in events if e[3] and e[3] > 0 and e[1] is not None]
    return [(e[0], e[2], e[3], 1) for e in events], [(e[1], e[2], e[3], 1) for e in events]


def study(started, ended, lookbacks, bucket, need_span, min_recent=10, shifts=SHIFTS, seed=7):
    """started / ended: the trades as (time, profit, cost, how many) by when they began and by when
    they ended, in one time unit (seconds, or a trading day's index); rows may already be summed (the
    database sums them). lookbacks: the L values to try, in that unit. bucket: the entries' time
    bucket (its start is when the signal is read). need_span: how much time before a verdict.
    Returns the summary and a row per lookback."""
    started = [r for r in started if r[2] and r[2] > 0]
    ended = sorted(r for r in ended if r[0] is not None and r[2] and r[2] > 0)
    if not started:
        return {"trades": 0, "span": 0, "need_span": need_span, "rows": [], "best": None, "state": "collecting",
                "progress": 0.0}
    first, last = min(r[0] for r in started), max(r[0] for r in started)
    span = last - first
    b0, b1 = int(first // bucket), int(last // bucket)
    # the trades by the bucket they began in: profit, cost, count
    out_pnl, out_paid, out_n = {}, {}, {}
    for t, pnl, paid, n in started:
        b = int(t // bucket)
        out_pnl[b] = out_pnl.get(b, 0.0) + pnl
        out_paid[b] = out_paid.get(b, 0.0) + paid
        out_n[b] = out_n.get(b, 0) + n
    # the trades by when they ended, as running totals (what had ended before a moment: a bisection)
    times = [r[0] for r in ended]
    cum_p, cum_n = [0.0], [0]
    for _, pnl, _, n in ended:
        cum_p.append(cum_p[-1] + pnl)
        cum_n.append(cum_n[-1] + n)
    buckets = sorted(out_n)
    half = buckets[len(buckets) // 2] if buckets else b0
    rng = random.Random(seed)
    width = b1 - b0 + 1
    offsets = [rng.randint(max(1, width // 10), max(1, width - width // 10)) for _ in range(shifts)] if width > 2 else []
    rows = []
    for L in lookbacks:
        # the signal at every bucket's start: +1 hot, -1 cold, 0 too few trades had ended
        sig = {}
        for b in range(b0, b1 + 1):
            t = b * bucket
            i, j = bisect.bisect_left(times, t - L), bisect.bisect_left(times, t)
            if cum_n[j] - cum_n[i] < min_recent:
                sig[b] = 0
            else:
                sig[b] = 1 if cum_p[j] - cum_p[i] > 0 else -1
        real = _split(buckets, sig.get, out_pnl, out_paid, out_n)
        old = _split([b for b in buckets if b < half], sig.get, out_pnl, out_paid, out_n)
        new = _split([b for b in buckets if b >= half], sig.get, out_pnl, out_paid, out_n)
        diff = _diff(real)
        placebo = []
        for s in offsets:
            got = _split(buckets, lambda b, s=s: sig.get(b0 + (b - b0 + s) % width, 0), out_pnl, out_paid, out_n)
            d = _diff(got)
            if d is not None:
                placebo.append(d)
        p = z = None
        if diff is not None and placebo:
            p = (1 + sum(1 for d in placebo if d >= diff)) / (1 + len(placebo))
            m = sum(placebo) / len(placebo)
            sd = (sum((d - m) ** 2 for d in placebo) / len(placebo)) ** 0.5
            z = (diff - m) / sd if sd > 0 else None
        rows.append({"lookback": L, "hot": real["hot"], "cold": real["cold"], "all": real["all"], "diff": diff,
                     "p": p, "z": z, "old_diff": _diff(old), "new_diff": _diff(new),
                     "caught": real["caught"], "avoided": real["avoided"], "kept": real["kept"]})
    ok = [r for r in rows if r["p"] is not None]
    best = min(ok, key=lambda r: (r["p"], -(r["diff"] or 0)), default=None)
    enough = span >= need_span
    holds = lambda r: (r["old_diff"] or 0) > 0 and (r["new_diff"] or 0) > 0
    if best and best["p"] <= P_FOUND and holds(best) and enough:
        state = "found" if (best["hot"]["roi"] or 0) > 0 else "real_not_paying"
    elif best and best["p"] <= P_PROMISING and (best["diff"] or 0) > 0:
        state = "promising"
    elif enough:
        state = "nothing"
    else:
        state = "collecting"
    return {"trades": sum(r[3] for r in started), "span": span, "need_span": need_span, "rows": rows, "best": best and best["lookback"],
            "state": state, "progress": min(1.0, span / need_span) if need_span else 1.0}


def _split(buckets, sig, out_pnl, out_paid, out_n):
    """The trades split by the signal at their bucket: hot, cold (and too early to say)."""
    acc = {k: [0.0, 0.0, 0] for k in ("hot", "cold", "all")}
    win_all = loss_all = win_hot = loss_cold = 0.0
    for b in buckets:
        s = sig(b)
        p, c, n = out_pnl[b], out_paid[b], out_n[b]
        for k in ("all",) + (("hot",) if s > 0 else ("cold",) if s < 0 else ()):
            acc[k][0] += p
            acc[k][1] += c
            acc[k][2] += n
        if s:
            if p > 0:
                win_all += p
                win_hot += p if s > 0 else 0.0
            else:
                loss_all += p
                loss_cold += p if s < 0 else 0.0
    out = {k: {"pnl": v[0], "paid": v[1], "trades": v[2], "roi": v[0] / v[1] if v[1] else None} for k, v in acc.items()}
    out["caught"] = win_hot / win_all if win_all else None
    out["avoided"] = loss_cold / loss_all if loss_all else None
    known = out["hot"]["trades"] + out["cold"]["trades"]
    out["kept"] = out["hot"]["trades"] / known if known else None
    return out


def _diff(split):
    h, c = split["hot"], split["cold"]
    if not h["trades"] or not c["trades"] or h["roi"] is None or c["roi"] is None:
        return None
    return h["roi"] - c["roi"]
