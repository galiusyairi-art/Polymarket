"""החכמה: one strategy that reads its coin's market by itself (the owner, 31.10.2026: "one strategy that profits
steadily, knows all it needs to know, no switches and no cents of mine: it reads the market of its coin, knows the
weekend, knows when and how to enter, under real money's conditions always").

What the week taught (a panel of three over all our studies, 31.10.2026), and what this does about each:
  - our model on its own was never better than Polymarket's price: on the trades it chose it promised 61-63% and won
    42-52%, while the price was about right. Choosing "model minus price is big" chose the model's own mistakes.
    So the probability here starts from the market's price and lets our model (and the order flow) move it only as
    far as they earned on rounds already settled ("market only" is the prior, our model has to earn every weight);
  - the market changes under it (a quiet weekend, a calm or a wild hour): the fit has the weekend and the volatility
    of the moment among its inputs, forgets the old (a half-life of three days), and is fitted per coin and round
    length on top of all coins together;
  - an edge on paper isn't one: the strategy enters only where its chance, less the price, the fee and what real
    money really pays over the price seen, stays above zero even at the low end of how sure the fit is about it
    (one-sided 95%: a statistical bar, not a setting);
  - when and how to enter: the rounds' time and the price are split into cells; a cell the walk-forward check (the
    fit of the days before, tried on a day it never saw) shows clearly losing is closed, and a price band real money
    clearly loses in (the cheap side late: 0 of 23) is closed too, in the simulation exactly as with real money;
  - real money's environment: its measured delay and price (the simulation's orders already go at both), what its
    orders paid over the price they saw, how many went through, and how its trades really ended;
  - it stops by itself: a CUSUM of how much less it wins than it said (per coin and round length) stops real money
    when it drifts, the simulation keeps going and real money starts again once that's back.

Data: `smart_rows`, a few samples a round (at fixed times before the close, from the recorded seconds and the
round's result), built as rounds end and kept 30 days; everything else is read from `bets` and `windows`.
"""
import datetime
import math

Z = 1.645  # one-sided 95%: the low end of the fit's chance that an entry needs above its cost
HALF_LIFE = 3 * 86400  # the old forgotten: a round three days ago weighs half of one now
HISTORY_DAYS = 10
KEEP_DAYS = 30
PRIOR = 20.0  # all coins together, pulled toward "the market's price only" (worth 20 rounds of evidence)
MARKET_PRIOR = 40.0  # one coin and round length, pulled toward all coins together
BUCKETS = {5: (20, 45, 75, 105, 150, 200, 250), 15: (30, 60, 120, 200, 300, 420, 540, 660, 800)}
BANDS = (0, 15, 35, 65, 101)  # price bands, cents
TEST_DAYS = 7  # walk-forward: the last days, each tried on a fit of the days before it
MIN_TRAIN_DAYS = 2
NEED_DAYS, NEED_TRADES, NEED_WEEKEND = 5, 100, 1  # real money: tried on this many days and rounds, a weekend among them
CELL_MIN = 30  # trades before a cell can be closed
LIVE_MIN = 10  # real money's trades in a price band before it can close it
CUSUM_K = 0.03  # the drift (won less than said) it allows a trade before it counts
NAMES = ("bias", "market", "model", "gap_time", "model_vol", "model_weekend", "market_time", "flow")
MU0 = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # "the market's price only"
MIN_STAKE = 1.0


def logit(p):
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def market_p(up_bid, up_ask, down_bid, down_ask):
    """The market's chance of Up: the middle of Up's book, else of Down's, else Up's offer against both offers."""
    if up_bid and up_ask:
        return (up_bid + up_ask) / 2
    if down_bid and down_ask:
        return 1 - (down_bid + down_ask) / 2
    if up_ask and down_ask:
        return up_ask / (up_ask + down_ask)
    return None


def weekend(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).weekday() >= 5


def features(model_up, market_up, seconds_left, minutes, vol, vol_mid, ts, flow, flow_scale):
    m, q = logit(model_up), logit(market_up)
    tau = max(0.0, min(1.0, seconds_left / (minutes * 60)))
    v = math.log(vol / vol_mid) if vol and vol_mid and vol > 0 and vol_mid > 0 else 0.0
    v = max(-2.0, min(2.0, v))
    w = 1.0 if weekend(ts) else 0.0
    f = math.tanh((flow or 0.0) / flow_scale) if flow_scale else 0.0
    return (1.0, q, m, (m - q) * tau, m * v, m * w, q * tau, f)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _solve(A, b):
    """A x = b by Gaussian elimination (A small and positive definite here)."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(n):
            if r != c and M[r][c]:
                k = M[r][c] / M[c][c]
                M[r] = [x - k * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _inverse(A):
    n = len(A)
    cols = [_solve(A, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    if any(c is None for c in cols):
        return None
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _loss(rows, beta, mu, lam):
    total = 0.0
    for x, y, w in rows:
        z = _dot(beta, x)
        # log(1 + e^z) - y z, steady for large |z|
        total += w * ((max(z, 0) + math.log1p(math.exp(-abs(z)))) - (z if y else 0.0))
    return total + 0.5 * lam * sum((b - m) ** 2 for b, m in zip(beta, mu))


def fit(rows, mu=MU0, lam=PRIOR, beta=None, iterations=12):
    """rows: (features, up won, weight). The weighted logistic fit with a Gaussian prior around mu (lam: its weight in
    rounds), by Newton's method, only ever downhill. Returns (beta, covariance) or (mu, None) without data."""
    n = len(mu)
    if not rows:
        return list(mu), None
    beta = list(beta or mu)
    current = _loss(rows, beta, mu, lam)
    H = None
    for _ in range(iterations):
        g = [lam * (b - m) for b, m in zip(beta, mu)]
        H = [[lam if i == j else 0.0 for j in range(n)] for i in range(n)]
        for x, y, w in rows:
            p = sigmoid(_dot(beta, x))
            r, s = w * (p - (1.0 if y else 0.0)), w * p * (1 - p)
            for i in range(n):
                xi = x[i]
                if not xi:
                    continue
                g[i] += r * xi
                si = s * xi
                Hi = H[i]
                for j in range(i, n):
                    Hi[j] += si * x[j]
        for i in range(n):
            for j in range(i):
                H[i][j] = H[j][i]
        step = _solve(H, g)
        if step is None:
            break
        t = 1.0
        while t > 1e-3:
            nb = [b - t * d for b, d in zip(beta, step)]
            loss = _loss(rows, nb, mu, lam)
            if loss <= current:
                break
            t /= 2
        else:
            break
        moved = max(abs(a - b) for a, b in zip(nb, beta))
        beta, current = nb, loss
        if moved < 1e-5:
            break
    return beta, (_inverse(H) if H else None)


def predict(beta, cov, x):
    """(chance of Up, its standard error as a chance)."""
    p = sigmoid(_dot(beta, x))
    if not cov:
        return p, 0.0
    var = sum(x[i] * cov[i][j] * x[j] for i in range(len(x)) for j in range(len(x)))
    return p, p * (1 - p) * math.sqrt(max(0.0, var))


def band(price):
    c = price * 100
    return next(i for i, (lo, hi) in enumerate(zip(BANDS, BANDS[1:])) if lo <= c < hi) if 0 <= c < BANDS[-1] else len(BANDS) - 2


def bucket(minutes, seconds_left):
    b = BUCKETS.get(minutes) or BUCKETS[15]
    return min(range(len(b)), key=lambda i: abs(b[i] - seconds_left))


def cost(price, fee):
    return price + fee(price)


def decide(p_up, sd, up_ask, down_ask, fee, slip=(0.0,) * 4, closed=(), floor=0.0, z=Z):
    """The entry, if any: (side, price seen, chance of the side, the low end of its edge, band). The edge at the
    offer, less the fee and what real money pays over the price seen in its band, at the low end of how sure the fit
    is. Not under real money's floor, not in a closed band."""
    best = None
    for side, prob, ask in (("up", p_up, up_ask), ("down", 1 - p_up, down_ask)):
        if not ask or ask <= 0 or ask >= 1 or ask < floor - 1e-9:
            continue
        b = band(ask)
        if b in closed:
            continue
        low = prob - z * sd - cost(ask, fee) - (slip[b] if b < len(slip) else 0.0)
        if low > 0 and (best is None or low > best[3]):
            best = (side, ask, prob, low, b)
    return best


def best_low(p_up, sd, up_ask, down_ask, fee, slip=(0.0,) * 4, z=Z):
    """How close the better side is to an entry, floor and closed bands aside: the low end of its edge (below 0: that
    much short), or None without offers. For the owner's "why no trades"."""
    out = None
    for prob, ask in ((p_up, up_ask), (1 - p_up, down_ask)):
        if not ask or ask <= 0 or ask >= 1:
            continue
        b = band(ask)
        low = prob - z * sd - cost(ask, fee) - (slip[b] if b < len(slip) else 0.0)
        out = low if out is None else max(out, low)
    return out


def stake_for(base, low, price, fee):
    """How much of the round's amount: a quarter of Kelly on the low end of the edge, as a share of the full amount
    at a 10% Kelly (more sure: more of it), never under a third of it, never over all of it."""
    per = cost(price, fee)
    kelly = low / max(1e-6, 1 - per)
    return max(MIN_STAKE, round(base * min(1.0, max(0.3, kelly * 10)), 2))


# ---------- the rows: a few samples a round, from the recorded seconds ----------

SCHEMA = """CREATE TABLE IF NOT EXISTS smart_rows (
    asset TEXT, minutes INTEGER, window_start INTEGER, ts REAL, seconds_left REAL, model REAL, market REAL,
    up_ask REAL, down_ask REAL, vol REAL, flow REAL, up_won INTEGER);
CREATE INDEX IF NOT EXISTS smart_rows_ts ON smart_rows (ts);
CREATE INDEX IF NOT EXISTS smart_rows_window ON smart_rows (asset, minutes, window_start);"""


def _twap(points, t0, t1):
    pts = [(ts, p) for ts, p in points if ts <= t1]
    if not pts:
        return None
    inside = [(ts, p) for ts, p in pts if ts >= t0]
    before = [p for ts, p in pts if ts < t0]
    start = before[-1] if before else (inside[0][1] if inside else None)
    if start is None or t1 <= t0:
        return start
    acc, cur_t, cur_p = 0.0, t0, start
    for ts, p in inside:
        acc += cur_p * (ts - cur_t)
        cur_t, cur_p = ts, p
    acc += cur_p * (t1 - cur_t)
    return acc / (t1 - t0)


def window_rows(asset, minutes, start, ticks, trades, open_twap, up_won, twap_seconds=60):
    """ticks: [(ts, seconds_left, price, open_price, vol, up_bid, up_ask, down_bid, down_ask)] in time order;
    trades: [(ts, outcome, size, taker)]. One sample at each of the round's times (the recorded second nearest it)."""
    import bisect
    from .model import fair_up_probability_twap
    out = []
    end = start + minutes * 60
    ok = [t for t in ticks if t[2] and t[4] and t[6] and t[8]]
    lefts = [-t[1] for t in ok]  # (time order: the seconds left only fall)
    trades = sorted((t for t in trades if t[2]), key=lambda t: t[0])
    tts = [t[0] for t in trades]
    prices = [(t[0], t[2]) for t in ticks if t[2]]
    for target in BUCKETS.get(minutes) or BUCKETS[15]:
        i = bisect.bisect_left(lefts, -target)
        near = [ok[j] for j in range(max(0, i - 3), min(len(ok), i + 3)) if abs(ok[j][1] - target) <= 8]
        if not near:
            continue
        ts, left, price, open_price, vol, ub, ua, db, da = min(near, key=lambda t: abs(t[1] - target))
        o = open_twap or open_price
        q = market_p(ub, ua, db, da)
        if not o or q is None:
            continue
        realized = _twap(prices, end - twap_seconds, ts) if left < twap_seconds else None
        m = fair_up_probability_twap(price, o, left, vol, twap_seconds, realized)
        flow = sum((s if (side == "up") == (taker == "BUY") else -s)
                   for t2, side, s, taker in trades[bisect.bisect_left(tts, ts - 60):bisect.bisect_right(tts, ts)])
        out.append((asset, minutes, start, ts, left, m, q, ua, da, vol, flow, 1 if up_won else 0))
    return out


def build(db, now, markets=None):
    """The rounds ended since the last pass (or the recorded days, the first time) into smart_rows."""
    db.executescript(SCHEMA)
    done = {(a, m): w for a, m, w in db.execute("SELECT asset, minutes, MAX(window_start) FROM smart_rows GROUP BY asset, minutes")}
    first = now - KEEP_DAYS * 86400
    added = 0
    rounds = db.execute("SELECT asset, minutes, window_start, winner, open_twap FROM windows WHERE winner IN ('up', 'down')"
                        " AND minutes > 0 AND window_start >= ? ORDER BY window_start", (first,)).fetchall()
    for a, m, w, winner, open_twap in rounds:
        if markets is not None and (a, m) not in markets:
            continue
        if w <= done.get((a, m), -1):
            continue
        ticks = db.execute("SELECT ts, seconds_left, price, open_price, vol, up_bid, up_ask, down_bid, down_ask FROM ticks"
                           " WHERE asset = ? AND minutes = ? AND window_start = ? ORDER BY ts", (a, m, w)).fetchall()
        if not ticks:
            continue
        trades = db.execute("SELECT ts, outcome, size, taker FROM trades WHERE asset = ? AND minutes = ? AND window_start = ?",
                            (a, m, w)).fetchall()
        rows = window_rows(a, m, w, ticks, trades, open_twap, winner == "up")
        if rows:
            db.executemany("INSERT INTO smart_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            added += len(rows)
    db.execute("DELETE FROM smart_rows WHERE ts < ?", (first,))
    db.commit()
    return added


# ---------- the fit, the walk-forward check, and what real money taught ----------

def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def scales(rows):
    """Per market: the usual volatility and the usual size of the flow (the fit's yardsticks)."""
    out = {}
    for key in {(r[0], r[1]) for r in rows}:
        mine = [r for r in rows if (r[0], r[1]) == key]
        out[key] = {"vol": _median([r[9] for r in mine if r[9]]), "flow": _median([abs(r[10]) for r in mine if r[10]]) or 0.0}
    return out


def _xs(rows, sc, now, half_life=HALF_LIFE):
    out = []
    for a, m, w, ts, left, model, market, ua, da, vol, flow, won in rows:
        s = sc.get((a, m)) or {}
        n = len(BUCKETS.get(m) or BUCKETS[15])
        out.append((features(model, market, left, m, vol, s.get("vol"), ts, flow, s.get("flow")), won,
                    0.5 ** (max(0.0, now - ts) / half_life) / n))  # (a round weighs one, its samples share it)
    return out


def fit_all(rows, now, markets, warm=None):
    """All coins together, then each market on top of it. {key: {beta, cov}}, with "all" for the pooled."""
    sc = scales(rows)
    xs = _xs(rows, sc, now)
    warm = warm or {}
    pooled, pcov = fit(xs, MU0, PRIOR, (warm.get("all") or {}).get("beta"))
    out = {"all": {"beta": pooled, "cov": pcov}}
    for key in markets:
        mine = [x for x, r in zip(xs, rows) if (r[0], r[1]) == key]
        b, c = fit(mine, pooled, MARKET_PRIOR, (warm.get(key) or {}).get("beta") or pooled)
        out[key] = {"beta": b, "cov": c or pcov, "scale": sc.get(key) or {}, "rows": len(mine)}
    return out


def _day(ts):
    return int(ts // 86400)


def walk_forward(rows, now, markets, fee, slip=(0.0,) * 4, live_closed=(), floor=0.0, days=TEST_DAYS):
    """Each of the last days, tried on a fit of the days before it (never on itself): the log loss against the market's
    price alone, and the trades the rule would have made (the first sample a round with an entry), at the offer, the
    fee and real money's slip."""
    all_days = sorted({_day(r[3]) for r in rows})
    tests = [d for d in all_days[-days:] if sum(1 for x in all_days if x < d) >= MIN_TRAIN_DAYS]
    trades, loss = [], {}
    warm = None
    for d in tests:
        start = d * 86400
        train = [r for r in rows if start - HISTORY_DAYS * 86400 <= r[3] < start]
        test = [r for r in rows if start <= r[3] < start + 86400 and (r[0], r[1]) in markets]
        if not train or not test:
            continue
        fits = fit_all(train, start, markets, warm)
        warm = fits
        rounds = {}
        for r in test:
            rounds.setdefault((r[0], r[1], r[2]), []).append(r)
        for (a, m, w), rs in rounds.items():
            F = fits.get((a, m))
            if not F:
                continue
            s = F.get("scale") or {}
            entered = None
            for r in sorted(rs, key=lambda r: r[3]):
                x = features(r[5], r[6], r[4], m, r[9], s.get("vol"), r[3], r[10], s.get("flow"))
                p, sd = predict(F["beta"], F["cov"], x)
                L = loss.setdefault((a, m), [0.0, 0.0, 0])
                y = r[11]
                L[0] -= math.log(max(1e-9, p if y else 1 - p))
                L[1] -= math.log(max(1e-9, min(0.99, max(0.01, r[6])) if y else 1 - min(0.99, max(0.01, r[6]))))
                L[2] += 1
                if entered is None:
                    e = decide(p, sd, r[7], r[8], fee, slip, live_closed, floor)
                    if e:
                        side, price, prob, low, b = e
                        won = (y == 1) == (side == "up")
                        per = cost(price, fee)
                        entered = {"asset": a, "minutes": m, "window": w, "ts": r[3], "day": d, "weekend": weekend(r[3]),
                                   "side": side, "price": price, "prob": prob, "won": won, "band": b,
                                   "bucket": bucket(m, r[4]), "roi": (1 / per - 1) if won else -1.0, "low": low}
            if entered:
                trades.append(entered)
    return trades, loss, tests


def summary(trades):
    n = len(trades)
    if not n:
        return {"n": 0}
    r = [t["roi"] for t in trades]
    mean = sum(r) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1)) if n > 1 else 0.0
    wins = sum(1 for t in trades if t["won"])
    return {"n": n, "roi": mean, "sd": sd, "low": mean - 1.2816 * sd / math.sqrt(n) if n > 1 else None,  # (90% one-sided)
            "high": mean + 1.0 * sd / math.sqrt(n) if n > 1 else None, "wins": wins, "win_rate": wins / n,
            "chance": sum(t["prob"] for t in trades) / n, "price": sum(t["price"] for t in trades) / n,
            "days": len({t["day"] for t in trades}), "weekend_days": len({t["day"] for t in trades if t["weekend"]})}


def closed_cells(trades):
    """(minutes, bucket, band) cells the walk-forward check shows clearly losing: enough trades, and even a standard
    error over their mean is under zero."""
    cells = {}
    for t in trades:
        cells.setdefault((t["minutes"], t["bucket"], t["band"]), []).append(t)
    out = []
    for key, ts in cells.items():
        s = summary(ts)
        if s["n"] >= CELL_MIN and s["high"] is not None and s["high"] < 0:
            out.append(key)
    return sorted(out)


def live_facts(db, now, days=14):
    """What real money taught, by price band (all its automatic trades, not the owner's own buys): what its orders
    paid over the price they saw, how many went through, and how they ended. A band whose trades won clearly fewer
    than their price said is closed (the simulation's and real money's alike: the same conditions)."""
    rows = db.execute("SELECT status, avg_price, seen, best_ask, pnl FROM bets WHERE mode = 'live' AND ts >= ? AND minutes > 0"
                      " AND variant NOT IN ('manual', 'lockguard', 'autojoin')", (now - days * 86400,)).fetchall()
    bands = []
    for i, (lo, hi) in enumerate(zip(BANDS, BANDS[1:])):
        mine = [r for r in rows if lo <= ((r[2] or r[3] or r[1] or 0) * 100) < hi]
        filled = [r for r in mine if r[0] == "filled"]
        slips = [r[1] - r[2] for r in filled if r[1] and r[2]]
        done = [r for r in filled if r[4] is not None and r[1]]
        wins = sum(1 for r in done if r[4] > 0)
        expect = sum(r[1] for r in done)  # (what the price said they'd win)
        var = sum(r[1] * (1 - r[1]) for r in done)
        z = (wins - expect) / math.sqrt(var) if var > 0 else None
        bands.append({"lo": lo, "hi": hi, "orders": len(mine), "filled": len(filled),
                      "fill_rate": len(filled) / len(mine) if mine else None,
                      "slip": max(0.0, sum(slips) / len(slips)) if slips else 0.0, "settled": len(done), "wins": wins,
                      "expect": expect, "z": z, "closed": bool(len(done) >= LIVE_MIN and z is not None and z <= -2.0)})
    return bands


def cusum(bets):
    """bets: [(chance said, won)] in time order. One-sided CUSUM of winning less than said: (S, its alarm level)."""
    if not bets:
        return 0.0, None
    s = 0.0
    for p, won in bets:
        s = max(0.0, s + (p - (1.0 if won else 0.0)) - CUSUM_K)
    h = 4 * math.sqrt(sum(p * (1 - p) for p, _ in bets) / len(bets))
    return s, h


def state_of(wf, now_weekend, alarm):
    """What a market may do now: "off" (no edge shown: it doesn't enter), "trial" (the simulation only), "proven"
    (real money too, where the owner runs it), with the reason in Hebrew."""
    if not wf or not wf.get("n"):
        return "off", "עוד אין לה הוכחה: בודקת על ימים שהיא לא ראתה"
    part = wf.get("weekend" if now_weekend else "weekday") or {}
    when = "בסוף שבוע" if now_weekend else "באמצע שבוע"
    if part.get("n", 0) >= CELL_MIN and (part.get("roi") or 0) <= 0:
        return "off", f"{when} היא לא מרוויחה בבדיקה ({part['roi'] * 100:+.1f}% על {part['n']} טריידים): לא נכנסת עכשיו"
    if (wf.get("roi") or 0) <= 0:
        return "off", f"בבדיקה על ימים שלא ראתה היא לא מרוויחה ({wf['roi'] * 100:+.1f}%): לא נכנסת"
    if alarm:
        return "trial", "השומר עצר את הכסף האמיתי: היא ניצחה פחות ממה שאמרה. בסימולציה ממשיכה עד שזה חוזר"
    need = []
    if wf.get("days", 0) < NEED_DAYS:
        need.append(f"{wf.get('days', 0)}/{NEED_DAYS} ימים")
    if wf.get("n", 0) < NEED_TRADES:
        need.append(f"{wf.get('n', 0)}/{NEED_TRADES} טריידים")
    if wf.get("weekend_days", 0) < NEED_WEEKEND:
        need.append("סוף שבוע")
    if wf.get("low") is None or wf["low"] <= 0:
        need.append("רווח בטוח (הגבול התחתון מעל 0)")
    if need:
        return "trial", "מרוויחה בבדיקה, עוד לא מוכחת: חסר " + ", ".join(need)
    return "proven", f"מוכחת: {wf['roi'] * 100:+.1f}% על {wf['n']} טריידים ב-{wf['days']} ימים שלא ראתה"


def study(db, now, markets, fee, floor=0.0):
    """The whole pass: new rows, what real money taught, the walk-forward check, and the fit to trade on now."""
    added = build(db, now)
    rows = db.execute("SELECT asset, minutes, window_start, ts, seconds_left, model, market, up_ask, down_ask, vol, flow,"
                      " up_won FROM smart_rows WHERE ts >= ?", (now - (HISTORY_DAYS + TEST_DAYS) * 86400,)).fetchall()
    rows = [tuple(r) for r in rows]
    facts = live_facts(db, now)
    slip = tuple(b["slip"] for b in facts)
    live_closed = tuple(i for i, b in enumerate(facts) if b["closed"])
    markets = {tuple(k) for k in markets}
    trades, loss, tests = walk_forward(rows, now, markets, fee, slip, live_closed, floor)
    recent = [r for r in rows if r[3] >= now - HISTORY_DAYS * 86400]
    fits = fit_all(recent, now, markets) if recent else {}
    cells = closed_cells(trades)
    out = {"ts": now, "rows": len(rows), "added": added, "tests": len(tests), "slip": slip, "live_closed": list(live_closed),
           "live": facts, "cells": [list(c) for c in cells], "names": list(NAMES), "markets": {}}
    for key in sorted(markets):
        mine = [t for t in trades if (t["asset"], t["minutes"]) == key]
        wf = summary(mine)
        wf["weekend"] = summary([t for t in mine if t["weekend"]])
        wf["weekday"] = summary([t for t in mine if not t["weekend"]])
        L = loss.get(key)
        F = fits.get(key) or {}
        out["markets"][f"{key[0]}|{key[1]}"] = {
            "asset": key[0], "minutes": key[1], "wf": wf,
            # (above 0: our probability beat the market's price alone on days it never saw)
            "loss_gain": (L[1] - L[0]) / L[2] if L and L[2] else None, "samples": L[2] if L else 0,
            "beta": F.get("beta"), "cov": F.get("cov"), "scale": F.get("scale") or {}, "fit_rows": F.get("rows", 0)}
    return out


def study_db(db_path, now, markets, fee_rate, fee_exponent, floor):
    from .model import taker_fee_per_share
    from .store import connect
    fee = lambda p: taker_fee_per_share(p, fee_rate, fee_exponent)  # noqa: E731
    db = connect(db_path)
    try:
        return study(db, now, markets, fee, floor)
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


def study_apart(*args, timeout=1500):
    """In a process of its own at the lowest priority (fits over the days, a few minutes of the processor)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_go, args=(b,) + tuple(args), daemon=True, name="smart")
    p.start()
    b.close()
    try:
        if not a.poll(timeout):
            raise TimeoutError("the smart study took too long")
        what, out = a.recv()
    finally:
        p.join(5)
        if p.is_alive():
            p.terminate()
    if what != "ok":
        raise RuntimeError(out)
    return out

