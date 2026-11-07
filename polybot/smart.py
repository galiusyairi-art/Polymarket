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
  - it stops by itself: when its last rounds in a market clearly win less than their price asked (money_guard), real money
    stops there, the simulation keeps going and real money starts again once that's no longer so.

Data: `smart_rows`, a few samples a round (at fixed times before the close, from the recorded seconds and the
round's result), built as rounds end and kept 30 days; everything else is read from `bets` and `windows`.
"""
import datetime
import math

try:  # (the fit in numpy when it's there: the same Newton steps, some fifty times faster; 03.11.2026, the study on 14
    import numpy as _np  # markets ran past its time limit on the server's one shared processor and never finished)
except Exception:  # pragma: no cover
    _np = None

Z = 1.645  # one-sided 95%: the low end of the fit's chance that an entry needs above its cost
HALF_LIFE = 3 * 86400  # the old forgotten: a round three days ago weighs half of one now
HISTORY_DAYS = 10
KEEP_DAYS = 30
PRIOR = 20.0  # all coins together, pulled toward "the market's price only" (worth 20 rounds of evidence)
MARKET_PRIOR = 40.0  # one coin and round length, pulled toward all coins together
BUCKETS = {5: (20, 45, 75, 105, 150, 200, 250), 15: (30, 60, 120, 200, 300, 420, 540, 660, 800)}
# price bands, cents (the dear end apart since 01.11.2026: 16 of 22 won and the day still lost, a loss at 97c wiping 32
# wins; 65-100 as one band let the 70c trades hide the 97c ones)
BANDS = (0, 15, 35, 65, 85, 93, 101)
NB = len(BANDS) - 1
TEST_DAYS = 7  # walk-forward: the last days, each tried on a fit of the days before it
MIN_TRAIN_DAYS = 2
NEED_DAYS, NEED_TRADES, NEED_WEEKEND = 5, 100, 1  # real money: tried on this many days and rounds, a weekend among them
CELL_MIN = 30  # trades before a cell can be closed
LIVE_MIN = 10  # real money's trades in a price band before it can close it
# The committee (the owner, 02.11.2026: "a precise panel that is part of the strategy and enters by itself"): each expert
# is one input of the same fit, its weight starting at 0 ("the market's price only") and earned only on settled rounds
# it never saw. The market itself; our model (with the time left, the volatility and the weekend); the buying flow; the
# dear end (the market's own late calibration past 85c: where one loss at 97c wiped 32 wins); and the spread (trust our
# model more where the book is wide).
NAMES = ("bias", "market", "model", "gap_time", "model_vol", "model_weekend", "market_time", "flow",
         "dear", "dear_time", "gap_spread", "momentum", "futures")
MU0 = (0.0, 1.0) + (0.0,) * (len(NAMES) - 2)  # "the market's price only"
EXPERTS = (("השוק", ("bias", "market", "market_time")), ("המודל שלנו", ("model", "gap_time", "model_vol", "model_weekend")),
           ("זרימת הקניות", ("flow",)), ("הקצה היקר", ("dear", "dear_time")), ("המרווח בספר", ("gap_spread",)),
           ("המומנטום", ("momentum",)), ("החוזים העתידיים", ("futures",)))
DEAR = math.log(0.85 / 0.15)
U_FLOOR = 0.01  # the least model error assumed at any price: what no fit can see (a flip in the last seconds, a feed)
U_PRIOR = 30  # trades' worth of "no error beyond the floor", shrinking a band's measured gap
MIN_STAKE = 1.0
# "החכמה הקלאסית" (the panel, 04.11.2026, the owner agreeing): החכמה as it was at 1.289, when it traded often and well: its
# first 8 inputs, the low end of its edge on the fit's own uncertainty only (no model-error term), no gate from real money.
# It runs beside the present one in the simulation, on the same rounds, so the two are compared on what they really do.
CLASSIC_K = 8
CLASSIC_U = (0.0,) * NB


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


def features(model_up, market_up, seconds_left, minutes, vol, vol_mid, ts, flow, flow_scale, spread=0.0, mom=None,
             model_fut=None):
    """mom: the coin's log move over the last minute (in the round); model_fut: our model's chance fed by Binance's
    futures (they move first: lead.py), None without them."""
    m, q = logit(model_up), logit(market_up)
    tau = max(0.0, min(1.0, seconds_left / (minutes * 60)))
    v = math.log(vol / vol_mid) if vol and vol_mid and vol > 0 and vol_mid > 0 else 0.0
    v = max(-2.0, min(2.0, v))
    w = 1.0 if weekend(ts) else 0.0
    f = math.tanh((flow or 0.0) / flow_scale) if flow_scale else 0.0
    dear = math.copysign(max(0.0, abs(q) - DEAR), q)
    sp = max(0.0, min(0.1, spread or 0.0)) * 10
    mo = math.tanh(mom / (vol * math.sqrt(60)) / 2) if mom and vol and vol > 0 else 0.0
    fu = max(-3.0, min(3.0, logit(model_fut) - m)) if model_fut is not None else 0.0
    return (1.0, q, m, (m - q) * tau, m * v, m * w, q * tau, f, dear, dear * tau, (m - q) * sp, mo, fu)


def spread_of(up_ask, down_ask):
    return max(0.0, (up_ask or 0) + (down_ask or 0) - 1) if up_ask and down_ask else 0.0


def votes(beta, x):
    """What each expert of the committee says now, in logit points over "the market's price only"."""
    w = dict(zip(NAMES, [b - m for b, m in zip(beta, MU0)]))
    xs = dict(zip(NAMES, x))
    return [(name, sum(w.get(k, 0.0) * xs.get(k, 0.0) for k in keys)) for name, keys in EXPERTS]


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


def _fit_np(rows, mu, lam, beta, iterations):
    """fit() in numpy: the same loss, gradient, Hessian, steps and line search, on arrays."""
    np = _np
    n = len(mu)
    X = np.array([list(x[:n]) + [0.0] * (n - len(x)) for x, _, _ in rows], dtype=float)
    y = np.array([1.0 if r[1] else 0.0 for r in rows])
    w = np.array([r[2] for r in rows], dtype=float)
    return _fit_arrays(X, y, w, mu, lam, beta, iterations)


def _fit_arrays(X, y, w, mu, lam, beta=None, iterations=12):
    """The fit on arrays: X (rows x inputs), y (1 when Up won), w (weights)."""
    np = _np
    n = len(mu)
    if not len(y):
        return list(mu), None
    m = np.array(mu, dtype=float)
    b = np.array(list(beta or mu)[:n] + list(mu)[len(beta or mu):], dtype=float)

    def loss(v):
        z = X @ v
        return float(np.sum(w * (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z))) - y * z))) + 0.5 * lam * float(np.sum((v - m) ** 2))
    current, H = loss(b), None
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30.0, 30.0)))
        g = lam * (b - m) + X.T @ (w * (p - y))
        H = lam * np.eye(n) + (X * (w * p * (1 - p))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        t, nb = 1.0, None
        while t > 1e-3:
            nb = b - t * step
            nl = loss(nb)
            if nl <= current:
                break
            t /= 2
        else:
            break
        moved = float(np.max(np.abs(nb - b)))
        b, current = nb, nl
        if moved < 1e-5:
            break
    cov = None
    if H is not None:
        try:
            cov = np.linalg.inv(H).tolist()
        except np.linalg.LinAlgError:
            cov = None
    return [float(v) for v in b], cov


def fit(rows, mu=MU0, lam=PRIOR, beta=None, iterations=12):
    """rows: (features, up won, weight). The weighted logistic fit with a Gaussian prior around mu (lam: its weight in
    rounds), by Newton's method, only ever downhill. Returns (beta, covariance) or (mu, None) without data."""
    n = len(mu)
    if not rows:
        return list(mu), None
    if _np is not None:
        return _fit_np(rows, mu, lam, beta, iterations)
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
    """(chance of Up, its standard error as a chance). A fit with fewer inputs (made before an expert joined) reads the
    first of them."""
    n = min(len(beta), len(x))
    x = x[:n]
    p = sigmoid(_dot(beta, x))
    if not cov or len(cov) < n:
        return p, 0.0
    var = sum(x[i] * cov[i][j] * x[j] for i in range(n) for j in range(n))
    return p, p * (1 - p) * math.sqrt(max(0.0, var))


def band(price):
    c = price * 100
    return next(i for i, (lo, hi) in enumerate(zip(BANDS, BANDS[1:])) if lo <= c < hi) if 0 <= c < BANDS[-1] else len(BANDS) - 2


def bucket(minutes, seconds_left):
    b = BUCKETS.get(minutes) or BUCKETS[15]
    return min(range(len(b)), key=lambda i: abs(b[i] - seconds_left))


def cost(price, fee):
    return price + fee(price)


def unsure(sd, u, b):
    """How unsure, as a chance: the fit's own uncertainty with the model's error measured in this price band (it grows at
    the extremes, where the fit's own shrinks to nothing: at 97c 1.645 x sd alone was about 1c)."""
    ub = u[b] if u and b < len(u) else U_FLOOR
    return math.sqrt(sd * sd + ub * ub)


# The dear end (the panel, 07.11.2026, the owner agreeing on the simulation's data: of 20 signals 13 were refused, "the price
# moved", nearly all at 80-98c in the round's last minute): there the book moves faster than an order, and an edge of a cent
# leaves the order no room. At DEAR_PRICE and over, an entry needs DEAR_ROOM of careful edge (real room), and none in the
# round's last DEAR_LAST seconds. החכמה's rule; the classic keeps 1.289's, beside it to compare.
DEAR_PRICE, DEAR_ROOM, DEAR_LAST = 0.80, 0.02, 30


def dear_blocked(ask, low, left):
    """At the dear end without real room, or in the round's last seconds: no entry (see DEAR_PRICE)."""
    return ask >= DEAR_PRICE and (low < DEAR_ROOM or (left is not None and left < DEAR_LAST))


def decide(p_up, sd, up_ask, down_ask, fee, slip=(0.0,) * NB, closed=(), floor=0.0, z=Z, u=None, left=None, dear=False):
    """The entry, if any: (side, price seen, chance of the side, the low end of its edge, band). The edge at the
    offer, less the fee and what real money pays over the price seen in its band, at the low end of how sure the fit
    is. Not under real money's floor, not in a closed band; with dear, not at the dear end without real room (left: the
    round's seconds left)."""
    best = None
    for side, prob, ask in (("up", p_up, up_ask), ("down", 1 - p_up, down_ask)):
        if not ask or ask <= 0 or ask >= 1 or ask < floor - 1e-9:
            continue
        b = band(ask)
        if b in closed:
            continue
        low = prob - z * unsure(sd, u, b) - cost(ask, fee) - (slip[b] if b < len(slip) else 0.0)
        if dear and low > 0 and dear_blocked(ask, low, left):
            continue
        if low > 0 and (best is None or low > best[3]):
            best = (side, ask, prob, low, b)
    return best


def best_low(p_up, sd, up_ask, down_ask, fee, slip=(0.0,) * NB, z=Z, u=None):
    """How close the better side is to an entry, floor and closed bands aside: the low end of its edge (below 0: that
    much short), or None without offers. For the owner's "why no trades"."""
    out = None
    for prob, ask in ((p_up, up_ask), (1 - p_up, down_ask)):
        if not ask or ask <= 0 or ask >= 1:
            continue
        b = band(ask)
        low = prob - z * unsure(sd, u, b) - cost(ask, fee) - (slip[b] if b < len(slip) else 0.0)
        out = low if out is None else max(out, low)
    return out


R_WINS = 8  # one loss may cost at most this many wins of the same trade


def stake_for(base, low, price, fee):
    """How much of the round's amount (02.11.2026, the panel: the old rule put the full amount on every dear trade, where
    one loss wipes 30 wins): a quarter of Kelly on the careful chance, the full amount from a 5% quarter Kelly, a third at
    least; and never more than lets one loss cost R_WINS wins of the same trade (at 97c about a quarter of the amount, at
    92c two thirds, at 85c and under all of it).
    Polymarket's least order is the caller's (it grows the amount to it, within the round's amount)."""
    per = cost(price, fee)
    f = max(0.0, low / max(1e-6, 1 - per)) / 4
    cap = min(1.0, R_WINS * (1 - per) / per) if per > 0 else 1.0
    return max(MIN_STAKE, round(base * min(cap, max(0.3, min(1.0, 20 * f))), 2))


# ---------- the rows: a few samples a round, from the recorded seconds ----------

SCHEMA = """CREATE TABLE IF NOT EXISTS smart_rows (
    asset TEXT, minutes INTEGER, window_start INTEGER, ts REAL, seconds_left REAL, model REAL, market REAL,
    up_ask REAL, down_ask REAL, vol REAL, flow REAL, up_won INTEGER, mom REAL, fut REAL);
CREATE INDEX IF NOT EXISTS smart_rows_ts ON smart_rows (ts);
CREATE INDEX IF NOT EXISTS smart_rows_window ON smart_rows (asset, minutes, window_start);
CREATE TABLE IF NOT EXISTS smart_done (asset TEXT, minutes INTEGER, window_start INTEGER,
    PRIMARY KEY (asset, minutes, window_start));"""
BUILD_BUDGET = 480  # seconds of building in one pass: the rest goes on in the next (02.11.2026)
BUILD_HOLD = 2.0  # seconds a write is held at most: the trading's own writes never wait on the rows


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
    """ticks: [(ts, seconds_left, price, open_price, vol, up_bid, up_ask, down_bid, down_ask, fut_price)] in time order;
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
    pts = [t for t, _ in prices]
    for target in BUCKETS.get(minutes) or BUCKETS[15]:
        i = bisect.bisect_left(lefts, -target)
        near = [ok[j] for j in range(max(0, i - 3), min(len(ok), i + 3)) if abs(ok[j][1] - target) <= 8]
        if not near:
            continue
        ts, left, price, open_price, vol, ub, ua, db, da, *more = min(near, key=lambda t: abs(t[1] - target))
        fut_price = more[0] if more else None
        o = open_twap or open_price
        q = market_p(ub, ua, db, da)
        if not o or q is None:
            continue
        realized = _twap(prices, end - twap_seconds, ts) if left < twap_seconds else None
        m = fair_up_probability_twap(price, o, left, vol, twap_seconds, realized)
        flow = sum((s if (side == "up") == (taker == "BUY") else -s)
                   for t2, side, s, taker in trades[bisect.bisect_left(tts, ts - 60):bisect.bisect_right(tts, ts)])
        j = bisect.bisect_right(pts, ts - 60) - 1  # (the price a minute before, in the round)
        mom = math.log(price / prices[j][1]) if j >= 0 and prices[j][1] and ts - prices[j][0] <= 70 else None
        mf = fair_up_probability_twap(fut_price, o, left, vol, twap_seconds, realized) if fut_price else None
        out.append((asset, minutes, start, ts, left, m, q, ua, da, vol, flow, 1 if up_won else 0, mom, mf))
    return out


def build(db, now, markets=None, budget=BUILD_BUDGET, clock=None):
    """The rounds not built yet into smart_rows: of the markets asked (all, without them), of the days the study reads,
    the newest first. Written a little at a time (02.11.2026: one write held for the whole first build of four new coins
    and of the momentum's new column locked the trading's own writes out for minutes, and was cut off by the study's
    time limit before it was ever kept, again and again): each round marked built as it goes, so a pass cut short goes
    on from there in the next."""
    import time as _t
    clock = clock or _t.monotonic
    db.executescript(SCHEMA)
    if "mom" not in {r[1] for r in db.execute("PRAGMA table_info(smart_rows)")}:
        # (02.11.2026: the momentum and the futures joined the committee: the rows built again from the recorded seconds)
        db.execute("DROP TABLE smart_rows")
        db.execute("DROP TABLE IF EXISTS smart_done")
        db.executescript(SCHEMA)
    if not db.execute("SELECT 1 FROM smart_done LIMIT 1").fetchone():  # (the rows built before the marks: marked once)
        db.execute("INSERT OR IGNORE INTO smart_done SELECT DISTINCT asset, minutes, window_start FROM smart_rows")
        db.commit()
    first = now - (HISTORY_DAYS + TEST_DAYS + 1) * 86400
    done = {(a, m, w) for a, m, w in db.execute("SELECT asset, minutes, window_start FROM smart_done WHERE window_start >= ?", (first,))}
    added = 0
    fut_col = "fut_price" if "fut_price" in {r[1] for r in db.execute("PRAGMA table_info(ticks)")} else "NULL"
    rounds = db.execute("SELECT asset, minutes, window_start, winner, open_twap FROM windows WHERE winner IN ('up', 'down')"
                        " AND minutes > 0 AND window_start >= ? ORDER BY window_start DESC", (first,)).fetchall()
    start = held = clock()
    for a, m, w, winner, open_twap in rounds:
        if (markets is not None and (a, m) not in markets) or (a, m, w) in done:
            continue
        if budget is not None and clock() - start > budget:
            break
        ticks = db.execute("SELECT ts, seconds_left, price, open_price, vol, up_bid, up_ask, down_bid, down_ask, %s FROM ticks"
                           " WHERE asset = ? AND minutes = ? AND window_start = ? ORDER BY ts" % fut_col, (a, m, w)).fetchall()
        rows = []
        if ticks:
            trades = db.execute("SELECT ts, outcome, size, taker FROM trades WHERE asset = ? AND minutes = ? AND window_start = ?",
                                (a, m, w)).fetchall()
            rows = window_rows(a, m, w, ticks, trades, open_twap, winner == "up")
        if rows:
            db.executemany("INSERT INTO smart_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            added += len(rows)
        db.execute("INSERT OR IGNORE INTO smart_done VALUES (?, ?, ?)", (a, m, w))
        if clock() - held >= BUILD_HOLD:
            db.commit()
            held = clock()
    db.commit()
    old = now - KEEP_DAYS * 86400
    db.execute("DELETE FROM smart_rows WHERE ts < ?", (old,))
    db.execute("DELETE FROM smart_done WHERE window_start < ?", (old,))
    db.commit()
    return added


def progress(db, now, markets):
    """(rounds built, rounds there are) of the markets, over the days the study reads: how far the rows are."""
    first = now - (HISTORY_DAYS + TEST_DAYS + 1) * 86400
    keys = {tuple(k) for k in markets}
    total = sum(n for a, m, n in db.execute("SELECT asset, minutes, COUNT(*) FROM windows WHERE winner IN ('up', 'down')"
                                            " AND minutes > 0 AND window_start >= ? GROUP BY asset, minutes", (first,))
                if (a, m) in keys)
    done = sum(n for a, m, n in db.execute("SELECT asset, minutes, COUNT(*) FROM smart_done WHERE window_start >= ?"
                                           " GROUP BY asset, minutes", (first,)) if (a, m) in keys)
    return min(done, total), total


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
    for a, m, w, ts, left, model, market, ua, da, vol, flow, won, *extra in rows:
        s = sc.get((a, m)) or {}
        n = len(BUCKETS.get(m) or BUCKETS[15])
        mom, fut = (extra + [None, None])[:2]
        out.append((features(model, market, left, m, vol, s.get("vol"), ts, flow, s.get("flow"), spread_of(ua, da), mom, fut), won,
                    0.5 ** (max(0.0, now - ts) / half_life) / n))  # (a round weighs one, its samples share it)
    return out


def _design(rows, sc, now, half_life=HALF_LIFE, k=None):
    """_xs() as arrays (X, y, w), filled a row at a time: no list of inputs per row is kept (04.11.2026, memory).
    k: the first k inputs only."""
    np = _np
    k = k or len(NAMES)
    X = np.empty((len(rows), k))
    y = np.empty(len(rows))
    w = np.empty(len(rows))
    for i, (a, m, _w, ts, left, model, market, ua, da, vol, flow, won, *extra) in enumerate(rows):
        s = sc.get((a, m)) or {}
        n = len(BUCKETS.get(m) or BUCKETS[15])
        mom, fut = (extra + [None, None])[:2]
        X[i] = features(model, market, left, m, vol, s.get("vol"), ts, flow, s.get("flow"), spread_of(ua, da), mom, fut)[:k]
        y[i] = 1.0 if won else 0.0
        w[i] = 0.5 ** (max(0.0, now - ts) / half_life) / n
    return X, y, w


def fit_all(rows, now, markets, warm=None, k=None):
    """All coins together, then each market on top of it. {key: {beta, cov}}, with "all" for the pooled. k: the first k
    inputs only (החכמה הקלאסית)."""
    sc = scales(rows)
    warm = warm or {}
    mu0 = MU0[:k] if k else MU0
    if _np is not None:
        np = _np
        X, y, w = _design(rows, sc, now, k=k)
        pooled, pcov = _fit_arrays(X, y, w, mu0, PRIOR, (warm.get("all") or {}).get("beta"))
        out = {"all": {"beta": pooled, "cov": pcov}}
        keys = [(r[0], r[1]) for r in rows]
        for key in markets:
            mask = np.fromiter((k == key for k in keys), dtype=bool, count=len(keys))
            b, c = _fit_arrays(X[mask], y[mask], w[mask], pooled, MARKET_PRIOR, (warm.get(key) or {}).get("beta") or pooled)
            if not mask.any():
                c = None
            out[key] = {"beta": b, "cov": c or pcov, "scale": sc.get(key) or {}, "rows": int(mask.sum())}
        return out
    xs = _xs(rows, sc, now)
    if k:
        xs = [(x[:k], y, w) for x, y, w in xs]
    pooled, pcov = fit(xs, mu0, PRIOR, (warm.get("all") or {}).get("beta"))
    out = {"all": {"beta": pooled, "cov": pcov}}
    for key in markets:
        mine = [x for x, r in zip(xs, rows) if (r[0], r[1]) == key]
        b, c = fit(mine, pooled, MARKET_PRIOR, (warm.get(key) or {}).get("beta") or pooled)
        out[key] = {"beta": b, "cov": c or pcov, "scale": sc.get(key) or {}, "rows": len(mine)}
    return out


def _day(ts):
    return int(ts // 86400)


# ---------- the check on columns (04.11.2026: the rows as Python objects ran the check out of memory on the server, 24
# times in a row; as numpy columns they take a tenth, and the inputs are computed for all rows at once) ----------

class _Table:
    """smart_rows as columns: a (asset index), m, w, ts, left, model, market, ua, da, vol, flow, won, mom, fut (None as
    nan); assets: the names behind a."""
    COLS = ("m", "w", "ts", "left", "model", "market", "ua", "da", "vol", "flow", "won", "mom", "fut")

    def __init__(self, chunks):
        np = _np
        assets, parts = {}, []
        for rows in chunks:
            if not rows:
                continue
            a = np.array([assets.setdefault(r[0], len(assets)) for r in rows], dtype=np.int16)
            cols = [np.array([np.nan if i >= len(r) or r[i] is None else r[i] for r in rows], dtype=float) for i in range(1, 14)]
            parts.append([a] + cols)
        self.assets = list(assets)
        if parts:
            self.a = np.concatenate([p[0] for p in parts])
            for j, name in enumerate(self.COLS):
                setattr(self, name, np.concatenate([p[j + 1] for p in parts]))
        else:
            self.a = np.empty(0, dtype=np.int16)
            for name in self.COLS:
                setattr(self, name, np.empty(0))
        self.m = self.m.astype(int)
        self.w = self.w.astype(np.int64)
        self.day = np.floor(self.ts / 86400).astype(np.int64)

    def __len__(self):
        return len(self.ts)

    def key_mask(self, key, idx=None):
        np = _np
        if key[0] not in self.assets:
            return np.zeros(len(self) if idx is None else len(idx), dtype=bool)
        a = self.assets.index(key[0])
        if idx is None:
            return (self.a == a) & (self.m == key[1])
        return (self.a[idx] == a) & (self.m[idx] == key[1])


def _table(rows):
    return _Table([list(rows)])


def _cursor_table(cur, size=50000):
    def chunks():
        while True:
            got = cur.fetchmany(size)
            if not got:
                return
            yield got
    return _Table(chunks())


def _median_np(x):
    x = _np.sort(x)
    return float(x[len(x) // 2]) if len(x) else None


def _scales_np(T, idx):
    """scales() on the table's rows idx: per market, the usual volatility and the usual size of the flow."""
    np = _np
    out = {}
    a, m, vol, flow = T.a[idx], T.m[idx], T.vol[idx], T.flow[idx]
    for ai, mi in ({(int(x), int(y)) for x, y in zip(*np.unique(np.stack([a, m]), axis=1))} if len(idx) else set()):
        sel = (a == ai) & (m == mi)
        v, f = vol[sel], flow[sel]
        v = v[~np.isnan(v) & (v != 0)]
        f = np.abs(f[~np.isnan(f) & (f != 0)])
        out[(T.assets[int(ai)], int(mi))] = {"vol": _median_np(v), "flow": _median_np(f) or 0.0}
    return out


def _logit_np(p):
    p = _np.clip(p, 0.01, 0.99)
    return _np.log(p / (1 - p))


def _features_np(T, idx, vol_mid, flow_scale, k=None):
    """features() for the rows idx at once; vol_mid and flow_scale per row (nan without them)."""
    np = _np
    k = k or len(NAMES)
    model, market, left, mins = T.model[idx], T.market[idx], T.left[idx], T.m[idx]
    vol, ts, flow, ua, da, mom, fut = T.vol[idx], T.ts[idx], T.flow[idx], T.ua[idx], T.da[idx], T.mom[idx], T.fut[idx]
    m, q = _logit_np(model), _logit_np(market)
    tau = np.clip(left / (mins * 60), 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (vol > 0) & (vol_mid > 0)
        v = np.where(ok, np.log(np.where(ok, vol / np.where(ok, vol_mid, 1.0), 1.0)), 0.0)
        v = np.clip(v, -2.0, 2.0)
        wk = (((T.day[idx] + 3) % 7) >= 5).astype(float)
        fl = np.nan_to_num(flow, nan=0.0)
        fs = np.nan_to_num(flow_scale, nan=0.0)
        f = np.where(fs != 0, np.tanh(fl / np.where(fs != 0, fs, 1.0)), 0.0)
        dear = np.copysign(np.maximum(0.0, np.abs(q) - DEAR), q)
        spread = np.where((np.nan_to_num(ua) != 0) & (np.nan_to_num(da) != 0), np.maximum(0.0, np.nan_to_num(ua) + np.nan_to_num(da) - 1), 0.0)
        sp = np.clip(spread, 0.0, 0.1) * 10
        okm = (np.nan_to_num(mom) != 0) & (vol > 0)
        mo = np.where(okm, np.tanh(np.nan_to_num(mom) / (np.where(okm, vol, 1.0) * math.sqrt(60)) / 2), 0.0)
        fu = np.where(~np.isnan(fut), np.clip(_logit_np(np.nan_to_num(fut, nan=0.5)) - m, -3.0, 3.0), 0.0)
    one = np.ones(len(idx))
    cols = (one, q, m, (m - q) * tau, m * v, m * wk, q * tau, f, dear, dear * tau, (m - q) * sp, mo, fu)
    return np.stack(cols[:k], axis=1) if len(idx) else np.empty((0, k))


def _row_scales(T, idx, sc):
    """Each row's market yardsticks: (vol_mid, flow_scale) arrays, nan where its market has none."""
    np = _np
    vm, fs = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    for (asset, minutes), s in sc.items():
        sel = T.key_mask((asset, minutes), idx)
        vm[sel] = s.get("vol") if s.get("vol") is not None else np.nan
        fs[sel] = s.get("flow") if s.get("flow") is not None else np.nan
    return vm, fs


def _fit_all_table(T, idx, now, markets, warm=None, k=None, half_life=HALF_LIFE):
    """fit_all() on the table's rows idx."""
    np = _np
    sc = _scales_np(T, idx)
    warm = warm or {}
    mu0 = MU0[:k] if k else MU0
    vm, fs = _row_scales(T, idx, sc)
    X = _features_np(T, idx, vm, fs, k)
    y = (T.won[idx] > 0).astype(float)
    nb = np.where(T.m[idx] == 5, len(BUCKETS[5]), len(BUCKETS[15])).astype(float)
    for mm, b in BUCKETS.items():
        nb[T.m[idx] == mm] = len(b)
    w = 0.5 ** (np.maximum(0.0, now - T.ts[idx]) / half_life) / nb
    pooled, pcov = _fit_arrays(X, y, w, mu0, PRIOR, (warm.get("all") or {}).get("beta"))
    out = {"all": {"beta": pooled, "cov": pcov}}
    for key in markets:
        mask = T.key_mask(key, idx)
        b, c = _fit_arrays(X[mask], y[mask], w[mask], pooled, MARKET_PRIOR, (warm.get(key) or {}).get("beta") or pooled)
        if not mask.any():
            c = None
        out[key] = {"beta": b, "cov": c or pcov, "scale": sc.get(key) or {}, "rows": int(mask.sum())}
    return out


def _walk_forward_table(T, now, markets, fee, slip=(0.0,) * NB, live_closed=(), floor=0.0, days=TEST_DAYS, u=None, k=None):
    """walk_forward() on the table: the same days, fits, chances, losses and entries."""
    np = _np
    all_days = sorted(set(int(d) for d in np.unique(T.day)))
    tests = [d for d in all_days[-days:] if sum(1 for x in all_days if x < d) >= MIN_TRAIN_DAYS]
    trades, loss = [], {}
    warm = None
    in_markets = np.zeros(len(T), dtype=bool)
    for key in markets:
        in_markets |= T.key_mask(key)
    for d in tests:
        start = d * 86400
        train = np.nonzero((T.ts >= start - HISTORY_DAYS * 86400) & (T.ts < start))[0]
        test = np.nonzero((T.ts >= start) & (T.ts < start + 86400) & in_markets)[0]
        if not len(train) or not len(test):
            continue
        fits = _fit_all_table(T, train, start, markets, warm, k=k)
        warm = fits
        for key in markets:
            F = fits.get(key)
            sel = test[T.key_mask(key, test)]
            if not F or not len(sel):
                continue
            s = F.get("scale") or {}
            vm = np.full(len(sel), s.get("vol") if s.get("vol") is not None else np.nan)
            fsc = np.full(len(sel), s.get("flow") if s.get("flow") is not None else np.nan)
            X = _features_np(T, sel, vm, fsc)
            beta = np.array(F["beta"], dtype=float)
            n = min(len(beta), X.shape[1])
            Xn = X[:, :n]
            p = 1.0 / (1.0 + np.exp(-np.clip(Xn @ beta[:n], -30.0, 30.0)))
            cov = F.get("cov")
            if cov and len(cov) >= n:
                C = np.array(cov, dtype=float)[:n, :n]
                var = np.einsum("ij,jk,ik->i", Xn, C, Xn)
                sd = p * (1 - p) * np.sqrt(np.maximum(0.0, var))
            else:
                sd = np.zeros(len(sel))
            y = T.won[sel] > 0
            mk = np.clip(T.market[sel], 0.01, 0.99)
            L = loss.setdefault(key, [0.0, 0.0, 0])
            L[0] += float(-np.sum(np.log(np.maximum(1e-9, np.where(y, p, 1 - p)))))
            L[1] += float(-np.sum(np.log(np.where(y, mk, 1 - mk))))
            L[2] += int(len(sel))
            order = np.lexsort((T.ts[sel], T.w[sel]))
            entered_w = None
            for j in order:
                i = sel[j]
                w = int(T.w[i])
                if w == entered_w:
                    continue
                ua, da = float(T.ua[i]), float(T.da[i])
                e = decide(float(p[j]), float(sd[j]), None if ua != ua else ua, None if da != da else da, fee, slip, live_closed,
                           floor, u=u, left=float(T.left[i]), dear=k is None)
                if e:
                    side, price, prob, low, b = e
                    won = bool(y[j]) == (side == "up")
                    per = cost(price, fee)
                    ts = float(T.ts[i])
                    trades.append({"asset": key[0], "minutes": key[1], "window": w, "ts": ts, "day": d, "weekend": weekend(ts),
                                   "side": side, "price": price, "prob": prob, "won": won, "band": b,
                                   "bucket": bucket(key[1], float(T.left[i])), "roi": (1 / per - 1) if won else -1.0, "low": low})
                    entered_w = w
    return trades, loss, tests


def walk_forward(rows, now, markets, fee, slip=(0.0,) * NB, live_closed=(), floor=0.0, days=TEST_DAYS, u=None, k=None):
    """Each of the last days, tried on a fit of the days before it (never on itself): the log loss against the market's
    price alone, and the trades the rule would have made (the first sample a round with an entry), at the offer, the
    fee and real money's slip."""
    if _np is not None:
        return _walk_forward_table(_table(rows), now, markets, fee, slip, live_closed, floor, days, u, k)
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
        fits = fit_all(train, start, markets, warm, k=k)
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
                x = features(r[5], r[6], r[4], m, r[9], s.get("vol"), r[3], r[10], s.get("flow"), spread_of(r[7], r[8]),
                             r[12] if len(r) > 12 else None, r[13] if len(r) > 13 else None)
                p, sd = predict(F["beta"], F["cov"], x)
                L = loss.setdefault((a, m), [0.0, 0.0, 0])
                y = r[11]
                L[0] -= math.log(max(1e-9, p if y else 1 - p))
                L[1] -= math.log(max(1e-9, min(0.99, max(0.01, r[6])) if y else 1 - min(0.99, max(0.01, r[6]))))
                L[2] += 1
                if entered is None:
                    e = decide(p, sd, r[7], r[8], fee, slip, live_closed, floor, u=u, left=r[4], dear=k is None)
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


# The guard (01.11.2026, the owner: "what does the guard need to let it trade? in the simulation it makes money"): on
# money, not on its own words. Per market, its last GUARD_N rounds (the simulation's, real money's and the guard's
# shadows, one a round): how many won against how many their price asked for (the price and the fee a share). Real money
# stops when it clearly wins less than the price asks (z -2, from GUARD_MIN rounds) and goes again once that's no longer
# so (z over -1). Old rounds leave the window, so it never stays stuck on a loss of last week. (Until then: a CUSUM of
# winning less than its own chance said, over all its history: at 97c one loss stopped it for hours, while it made money.)
GUARD_N, GUARD_MIN, GUARD_STOP, GUARD_GO = 40, 20, -2.0, -1.0


def money_guard(trades):
    """trades: [(what a share cost, the fee in; won)], the newest last. {n, wins, expect, z} over the last GUARD_N."""
    last = trades[-GUARD_N:]
    wins = sum(1 for _, w in last if w)
    expect = sum(c for c, _ in last)
    var = sum(c * (1 - c) for c, _ in last)
    return {"n": len(last), "wins": wins, "expect": expect, "z": (wins - expect) / math.sqrt(var) if var > 0 else None}


def own_facts(db, now, keys, days=14):
    """What its own trades taught, by price band: the simulation's, real money's and the shadows its guard held back
    ("guarded"), all on real money's conditions. Each against what its price said it had to win (the price and the fee a
    share): a band where it clearly wins less (as real money's own lessons, z -2) closes to real money; the simulation
    keeps trading it, so the band learns its way back open."""
    by = {}
    rank = {"live": 0, "paper": 1, "guarded": 2}
    for a, m, w, mode, *r in db.execute(
            "SELECT asset, minutes, window_start, mode, avg_price, shares, cost, fee, pnl FROM bets WHERE variant IN (%s)"
            " AND mode IN ('paper', 'live', 'guarded') AND status = 'filled' AND pnl IS NOT NULL AND ts >= ? AND shares > 0"
            % ",".join("?" * len(keys)), tuple(keys) + (now - days * 86400,)):
        k = (a, m, w)  # (one a round: the simulation and real money took the same signal; real money's when it has one)
        if k not in by or rank[mode] < by[k][0]:
            by[k] = (rank[mode], r)
    rows = [r for _, r in by.values()]
    bands = []
    for lo, hi in zip(BANDS, BANDS[1:]):
        mine = [r for r in rows if r[0] and lo <= r[0] * 100 < hi]
        wins = sum(1 for r in mine if r[4] > 0)
        need = [min(0.999, (r[2] + r[3]) / r[1]) for r in mine]  # (what a share cost, the fee in: the chance it had to win)
        expect, var = sum(need), sum(q * (1 - q) for q in need)
        z = (wins - expect) / math.sqrt(var) if var > 0 else None
        bands.append({"lo": lo, "hi": hi, "n": len(mine), "wins": wins, "expect": expect, "z": z,
                      "pnl": sum(r[4] for r in mine), "paid": sum(r[2] + r[3] for r in mine),
                      # (and at the dear end, where one loss wipes dozens of wins and the z test reacts too slowly: losing
                      # money over 10 trades or more is enough)
                      "closed": bool(len(mine) >= LIVE_MIN and ((z is not None and z <= -2.0)
                                                               or (lo >= 85 and sum(r[4] for r in mine) < 0)))})
    return bands


def model_error(trades):
    """Per price band, the model's error on trades it took on days it never saw: what it promised over what they won,
    shrunk toward none (U_PRIOR trades' worth), plus its standard error; never under U_FLOOR."""
    out = []
    for b in range(NB):
        mine = [t for t in trades if t["band"] == b]
        n = len(mine)
        gap = sum(t["prob"] - (1.0 if t["won"] else 0.0) for t in mine) / (n + U_PRIOR)
        se = math.sqrt(sum(t["prob"] * (1 - t["prob"]) for t in mine)) / (n + U_PRIOR)
        out.append(round(max(U_FLOOR, max(0.0, gap) + se), 4))
    return out


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


def study(db, now, markets, fee, floor=0.0, u_prev=None):
    """The whole pass: new rows, what real money taught, the walk-forward check, and the fit to trade on now."""
    added = build(db, now, {tuple(k) for k in markets})
    built, total = progress(db, now, markets)
    cur = db.execute("SELECT asset, minutes, window_start, ts, seconds_left, model, market, up_ask, down_ask, vol, flow,"
                     " up_won, mom, fut FROM smart_rows WHERE ts >= ?", (now - (HISTORY_DAYS + TEST_DAYS) * 86400,))
    if _np is not None:
        rows = _cursor_table(cur)  # (columns, read a part at a time: 04.11.2026, the memory)
    else:
        rows = [tuple(r) for r in cur.fetchall()]
    facts = live_facts(db, now)
    own = own_facts(db, now, [f"smart|0.00|5-{m * 60 - 5}" for m in (5, 15)])
    slip = tuple(b["slip"] for b in facts)
    live_closed = tuple(i for i, b in enumerate(facts) if b["closed"])
    markets = {tuple(k) for k in markets}
    # (the check on days it never saw tries the rule that trades: its model error as the last check measured it, not the
    # floor alone; 04.11.2026, the panel: the check proved a looser rule than the one that traded)
    u_check = tuple(u_prev) if u_prev and len(u_prev) == NB else (U_FLOOR,) * NB
    if _np is not None:
        T = rows
        trades, loss, tests = _walk_forward_table(T, now, markets, fee, slip, live_closed, floor, u=u_check)
        recent = _np.nonzero(T.ts >= now - HISTORY_DAYS * 86400)[0]
        fits = _fit_all_table(T, recent, now, markets) if len(recent) else {}
        # (החכמה הקלאסית: the same days and rounds, its own rule)
        ctrades, closs, _ = _walk_forward_table(T, now, markets, fee, slip, live_closed, floor, u=CLASSIC_U, k=CLASSIC_K)
        cfits = _fit_all_table(T, recent, now, markets, k=CLASSIC_K) if len(recent) else {}
    else:
        trades, loss, tests = walk_forward(rows, now, markets, fee, slip, live_closed, floor, u=u_check)
        recent = [r for r in rows if r[3] >= now - HISTORY_DAYS * 86400]
        fits = fit_all(recent, now, markets) if recent else {}
        ctrades, closs, _ = walk_forward(rows, now, markets, fee, slip, live_closed, floor, u=CLASSIC_U, k=CLASSIC_K)
        cfits = fit_all(recent, now, markets, k=CLASSIC_K) if recent else {}
    u = model_error(trades)
    cells = closed_cells(trades)
    out = {"ts": now, "rows": len(rows), "added": added, "tests": len(tests), "built": built, "rounds": total, "slip": slip, "live_closed": list(live_closed),
           "live": facts, "own": own, "own_closed": [i for i, b in enumerate(own) if b["closed"]],
           "bands": list(BANDS), "cells": [list(c) for c in cells], "names": list(NAMES), "u": u, "markets": {}}
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
    classic = {"cells": [list(c) for c in closed_cells(ctrades)], "markets": {}}
    for key in sorted(markets):
        mine = [t for t in ctrades if (t["asset"], t["minutes"]) == key]
        wf = summary(mine)
        wf["weekend"] = summary([t for t in mine if t["weekend"]])
        wf["weekday"] = summary([t for t in mine if not t["weekend"]])
        L, F = closs.get(key), cfits.get(key) or {}
        classic["markets"][f"{key[0]}|{key[1]}"] = {
            "asset": key[0], "minutes": key[1], "wf": wf, "loss_gain": (L[1] - L[0]) / L[2] if L and L[2] else None,
            "beta": F.get("beta"), "cov": F.get("cov"), "scale": F.get("scale") or {}}
    out["classic"] = classic
    out["u_check"] = list(u_check)
    return out


def study_db(db_path, now, markets, fee_rate, fee_exponent, floor, u_prev=None):
    from .model import taker_fee_per_share
    from .store import connect
    fee = lambda p: taker_fee_per_share(p, fee_rate, fee_exponent)  # noqa: E731
    db = connect(db_path)
    db.row_factory = None  # (plain tuples: a Row each would hold the rows twice over)
    try:
        return study(db, now, markets, fee, floor, u_prev)
    finally:
        db.close()


STUDY_MEMORY = 1000 * 2 ** 20  # bytes the check may hold: past it, a MemoryError it reports, never the server's memory


def _cli():
    """The check as a plain process of its own (python -c "from polybot.smart import _cli; _cli()"): its arguments as JSON on
    stdin, its answer as JSON on stdout. It loads only what the check needs, never the bot (04.11.2026: a process made with
    multiprocessing loaded the whole bot again before starting), at the lowest priority, the first the system kills when
    memory runs out, and never over STUDY_MEMORY."""
    import json
    import os
    import sys
    args = json.load(sys.stdin)
    try:
        os.nice(15)
    except Exception:
        pass
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except Exception:
        pass
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_DATA, (STUDY_MEMORY, STUDY_MEMORY))
    except Exception:
        pass
    try:
        out = {"ok": study_db(*args)}
    except MemoryError:
        out = {"error": f"MemoryError: the check needed more than {STUDY_MEMORY // 2 ** 20} MB"}
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}"[:300]}
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


def study_apart(*args, timeout=2400):
    """In a plain process of its own (_cli), at the lowest priority (a few minutes of the processor at most)."""
    import json
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONPATH=root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    try:
        p = subprocess.run([sys.executable, "-c", "from polybot.smart import _cli; _cli()"], input=json.dumps(list(args)),
                           capture_output=True, text=True, timeout=timeout, cwd=root, env=env)
    except subprocess.TimeoutExpired:
        raise TimeoutError("the smart study took too long")
    if p.returncode in (-9, 137):
        raise RuntimeError("the check's process was killed by the system (out of memory)")
    try:
        out = json.loads(p.stdout)
    except ValueError:
        raise RuntimeError(f"the check's process ended without an answer (exit code {p.returncode}): {p.stderr.strip()[-200:]}")
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["ok"]

