"""The proof test (the owner, 16.11.2026, after the web research found no strategy documented to make a small bot money
on the short rounds: "go and do it; and a card of its own on the control board, the whole progress, whether I'm up or
down, how much, and when we get to a final answer").

One frozen rule with real money at the smallest size the exchange takes, to settle cheaply whether the one signal the
lab showed anything on is an edge or luck (the best of 983 rules shows about +6..7.5% by luck alone):

- BTC, the 5-minute rounds only; the lab's rule as it was measured there: our model gives the side at least 12 points
  over its offer, with 10 to 120 seconds left in the round;
- entered as a resting bid a cent under the offer (post-only: no fee, a small rebate), never by taking; it rests 30
  seconds or until the model no longer sees the edge at it; held to the round's end; no stop, no hedge;
- 5 shares a bet (the exchange's least), one bet a round, at most DAY_FILLS fills a day, and no new bet on a day whose
  loss (with what is still open) reached DAY_STOP;
- judged on real fills alone, at three checkpoints fixed ahead (STAGES), never between them: stopped when even the
  hopeful end of the error range is under +2%, passed when even the cautious end is above zero.

Nothing here changes while the test runs: the rule, the size and the checkpoints are constants of the code."""
import json
import time

ASSET, MINUTES = "btc", 5
VARIANT = "twap|0.12|10-120|es1"
SHARES = 5.0
DAY_FILLS = 100
DAY_STOP = 50.0
# (settled fills, stopped under this return, passed over this return): the report's table (for a mean price near 55c)
STAGES = ((500, -0.06, None), (1500, -0.025, 0.045), (4000, -0.008, 0.028))
REBATE = 0.2  # the makers' share of the takers' fees (Polymarket's fee schedule, read 15.11.2026): an estimate only
MARK_AT = (10, 30)  # seconds after a fill the book's middle is read against the price paid (the early warning)
MARK_WARN = -0.0035  # against us by more than the rebate, on average: run over
MARK_LEAST = 300


def rows(db, since):
    return [dict(r) for r in db.execute(
        "SELECT id, ts, window_start, side, seconds_left, fair_prob, avg_price, shares, cost, fee, pnl FROM bets"
        " WHERE mode = 'live' AND status = 'filled' AND asset = ? AND minutes = ? AND variant = ? AND ts >= ? ORDER BY ts",
        (ASSET, MINUTES, VARIANT, since or 0))]


def judge(done):
    """The checkpoints on the settled fills, in order: each stage's return on the first n of them, and the verdict
    (None: not there yet; "failed" / "passed" at the first stage that says so; "unproven" past the last)."""
    stages, verdict = [], None
    for n, stop, go in STAGES:
        s = {"n": n, "stop": stop, "go": go, "roi": None, "result": None}
        if len(done) >= n and verdict is None:
            first = done[:n]
            paid = sum(b["cost"] + (b["fee"] or 0) for b in first)
            s["roi"] = sum(b["pnl"] for b in first) / paid if paid else 0.0
            s["result"] = "failed" if s["roi"] < stop else "passed" if go is not None and s["roi"] > go else "on"
            if s["result"] != "on":
                verdict = s["result"]
        stages.append(s)
    if verdict is None and len(done) >= STAGES[-1][0]:
        verdict = "unproven"
    return stages, verdict


def marks(db, bets, kept):
    """The book's middle MARK_AT seconds after each fill, against the price paid (dollars a share, over zero: for us):
    kept by the bet's id (the prices are deleted after some days)."""
    changed = False
    now = time.time()
    for b in bets:
        k = str(b["id"])
        if k in kept or now < b["ts"] + max(MARK_AT) + 5:
            continue
        out = []
        for d in MARK_AT:
            r = db.execute("SELECT up_bid, up_ask, down_bid, down_ask FROM ticks WHERE asset = ? AND minutes = ? AND window_start = ?"
                           " AND ts >= ? ORDER BY ts LIMIT 1", (ASSET, MINUTES, b["window_start"], b["ts"] + d)).fetchone()
            bid, ask = (r[0], r[1]) if r and b["side"] == "up" else (r[2], r[3]) if r else (None, None)
            out.append(round((bid + ask) / 2 - b["avg_price"], 4) if bid is not None and ask is not None else None)
        kept[k] = out
        changed = True
    return changed


def numbers(db, state, day_start, fee_rate, kept=None):
    """Everything the card shows, from the real fills alone."""
    bets = rows(db, state.get("since"))
    done = [b for b in bets if b["pnl"] is not None]
    paid = sum(b["cost"] + (b["fee"] or 0) for b in done)
    pnl = sum(b["pnl"] for b in done)
    wins = sum(1 for b in done if b["pnl"] > 0)
    price = sum(b["cost"] for b in done) / sum(b["shares"] for b in done) if done else None
    stages, verdict = judge(done)
    today = [b for b in bets if b["ts"] >= day_start]
    open_cost = sum(b["cost"] + (b["fee"] or 0) for b in bets if b["pnl"] is None)
    today_pnl = sum(b["pnl"] for b in today if b["pnl"] is not None)
    nxt = next((s for s in stages if s["result"] is None), None)
    first = bets[0]["ts"] if bets else None
    days = max((time.time() - first) / 86400, 1 / 24) if first else None
    pace = len(done) / days if days else None
    curve, run = [], 0.0
    for i, b in enumerate(done, 1):
        run += b["pnl"]
        curve.append((i, round(run, 2)))
    step = max(1, len(curve) // 200)
    curve = curve[::step] + (curve[-1:] if curve and (len(curve) - 1) % step else [])
    mk = None
    if kept is not None:
        vals = [[kept[str(b["id"])][i] for b in bets if str(b["id"]) in kept and kept[str(b["id"])][i] is not None]
                for i in range(len(MARK_AT))]
        mk = [{"after": d, "n": len(v), "avg": sum(v) / len(v) if v else None} for d, v in zip(MARK_AT, vals)]
    return {
        "fills": len(bets), "settled": len(done), "open": len(bets) - len(done), "wins": wins,
        "win_rate": wins / len(done) if done else None, "price": price, "pnl": round(pnl, 2), "paid": round(paid, 2),
        "roi": pnl / paid if paid else None, "stages": stages, "verdict": verdict,
        "next": nxt and {"n": nxt["n"], "left": nxt["n"] - len(done),
                         "days": (nxt["n"] - len(done)) / pace if pace else None},
        "last": {"n": STAGES[-1][0], "left": max(0, STAGES[-1][0] - len(done)),
                 "days": max(0, STAGES[-1][0] - len(done)) / pace if pace else None},
        "pace": pace, "first": first,
        "today": {"fills": len(today), "pnl": round(today_pnl, 2), "at_risk": round(open_cost, 2),
                  "max_fills": DAY_FILLS, "stop": DAY_STOP},
        "rebate": round(sum(REBATE * fee_rate * b["avg_price"] * (1 - b["avg_price"]) * b["shares"] for b in bets), 2),
        "marks": mk, "curve": curve,
        "recent": [{k: b[k] for k in ("ts", "side", "avg_price", "shares", "pnl", "seconds_left")} for b in bets[-12:]][::-1],
    }


def day_block(n):
    """Why no new bet today (None: go on)."""
    t = n["today"]
    if t["fills"] >= DAY_FILLS:
        return f"מבחן ההוכחה: {t['fills']} מילויים היום (התקרה היומית)"
    if t["at_risk"] - t["pnl"] >= DAY_STOP:
        return f"מבחן ההוכחה: ההפסד של היום הגיע ל-${DAY_STOP:.0f} (עם מה שעוד פתוח): ממשיך מחר"
    return None


def load(text, default):
    try:
        return json.loads(text or "null") or default
    except ValueError:
        return default
