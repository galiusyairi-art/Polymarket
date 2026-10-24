"""Automatic calibration: make the model's probabilities match how often things really happen.

The model turns a price move into a probability through a z-score. If it is overconfident
(says 80% for things that happen 65% of the time), its z-scores are too large; scaling them
by a factor k < 1 fixes that, k > 1 fixes the opposite. k is fitted on settled bets by
maximum likelihood, using the model's raw probabilities (never the calibrated ones, which
would feed on themselves).

Settled bets are far from independent: dozens of lab variants bet the same side of the same
window. So bets are first collapsed to one per (market, window, side), and the fitted k is
pulled toward 1 in proportion to how little evidence there is. It is applied only once there
is enough evidence and when it improves the fit by a margin.

Blending with the market (market_blend): when we disagree with Polymarket's price, the
price is often right (faster traders see the move first). So the probability used for
decisions can be a learned mix, p = sigmoid(a * logit(model) + b * logit(market price)).
a and b are fitted on the older 70% of resolved windows and kept only if the mix beats the
model alone on the newest 30%, which it never saw. If the model adds nothing, a goes to 0,
the mix equals the market, and no edge is left to bet on: the system stops trading where it
has no advantage instead of losing slowly.
"""
import math
from statistics import NormalDist

_N = NormalDist()
GRID = [round(0.30 + 0.01 * i, 2) for i in range(121)]  # 0.30 .. 1.50


def apply(p, k):
    """The probability p with its z-score scaled by k."""
    if p is None or k == 1.0:
        return p
    p = min(max(p, 1e-6), 1 - 1e-6)
    return _N.cdf(k * _N.inv_cdf(p))


def _loss(rows, k):
    total = 0.0
    for p, won, weight in rows:
        q = min(max(apply(p, k), 1e-9), 1 - 1e-9)
        total -= weight * (math.log(q) if won else math.log(1 - q))
    return total / sum(w for _, _, w in rows)


def fit(rows):
    """rows: (raw probability of the side bought, won, weight). Returns (k, loss at k, loss at 1)."""
    losses = {k: _loss(rows, k) for k in GRID}
    best = min(losses, key=losses.get)
    return best, losses[best], losses[1.0]


def calibrate(db, cfg):
    rows = [
        (r[0], bool(r[1]), 1.0) for r in db.execute(
            "SELECT AVG(COALESCE(fair_raw, fair_prob)), MAX(pnl > 0) FROM bets"
            " WHERE mode IN ('paper', 'lab') AND pnl IS NOT NULL AND fair_prob IS NOT NULL AND minutes > 0"
            " GROUP BY asset, minutes, window_start, side"
        ).fetchall() if r[0] is not None
    ]
    n = len(rows)
    out = {"k": 1.0, "fitted": None, "samples": n, "applied": False,
           "min_samples": cfg.calib_min_samples, "loss_raw": None, "loss_cal": None}
    if n < cfg.calib_min_samples:
        return out
    k, loss_k, loss_1 = fit(rows)
    shrunk = 1.0 + (k - 1.0) * n / (n + cfg.calib_prior_samples)
    out.update(fitted=k, loss_raw=loss_1, loss_cal=_loss(rows, shrunk))
    if loss_1 - out["loss_cal"] > cfg.calib_min_gain:
        out.update(k=round(shrunk, 3), applied=True)
    return out


def _logit(p):
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))


def blend(model, market, a, b):
    """The decision probability: the model's and the market's views, mixed with learned weights."""
    z = a * _logit(model) + b * _logit(market)
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def _blend_loss(samples, a, b):
    total = 0.0
    for x, y, won in samples:
        z = max(-30.0, min(30.0, a * x + b * y))
        p = min(max(1 / (1 + math.exp(-z)), 1e-9), 1 - 1e-9)
        total -= math.log(p) if won else math.log(1 - p)
    return total / len(samples)


def fit_blend(samples, iterations=30):
    """samples: (logit model, logit market, up won). Maximum likelihood by Newton's method (a two-weight
    logistic regression, a few passes over the data); weights below zero mean "ignore" and are clipped."""
    a, b = 1.0, 0.0
    for _ in range(iterations):
        ga = gb = haa = hab = hbb = 0.0
        for x, y, won in samples:
            p = 1 / (1 + math.exp(-max(-30.0, min(30.0, a * x + b * y))))
            r, w = (1.0 if won else 0.0) - p, p * (1 - p)
            ga += r * x
            gb += r * y
            haa += w * x * x
            hab += w * x * y
            hbb += w * y * y
        haa, hbb = haa + 1e-6, hbb + 1e-6
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        size = abs(da) + abs(db)
        if size > 0.5:  # a damped step: never jump far on a bad curvature estimate
            da, db = da * 0.5 / size, db * 0.5 / size
        current = _blend_loss(samples, a, b)
        step = 1.0
        while step > 1e-3:  # and only ever downhill, inside sane bounds
            na, nb = min(3.0, max(0.0, a + step * da)), min(3.0, max(0.0, b + step * db))
            if _blend_loss(samples, na, nb) <= current:
                break
            step /= 2
        else:
            break
        if abs(na - a) + abs(nb - b) < 1e-6:
            a, b = na, nb
            break
        a, b = na, nb
    return round(a, 3), round(b, 3)


def market_blend(db, cfg):
    """Learn how much to trust the model against Polymarket's price, from resolved windows."""
    winners = {(a, m, w): r for a, m, w, r in db.execute(
        "SELECT asset, minutes, window_start, winner FROM windows WHERE winner IS NOT NULL AND minutes > 0")}
    samples, seen = [], set()
    for a, m, w, left, raw, bid, ask in db.execute(
            "SELECT asset, minutes, window_start, seconds_left, fair_raw, up_bid, up_ask FROM ticks"
            " WHERE fair_raw IS NOT NULL AND up_bid IS NOT NULL AND up_ask IS NOT NULL"
            " AND seconds_left BETWEEN 15 AND 600 AND minutes > 0 ORDER BY window_start, ts"):
        key = (a, m, w)
        if key not in winners or (key, int(left // 60)) in seen:
            continue
        seen.add((key, int(left // 60)))  # one sample per window per minute: they are far from independent
        samples.append((w, _logit(raw), _logit((bid + ask) / 2), winners[key] == "up", key))
    keys = sorted({s[4] for s in samples}, key=lambda k: k[2])[-cfg.blend_max_windows:]  # the recent past
    samples = [s[:4] for s in samples if s[4] in set(keys)]
    windows = sorted({s[0] for s in samples})  # window start times, for splitting old from new
    out = {"a": 1.0, "b": 0.0, "applied": False, "windows": len(keys), "samples": len(samples),
           "min_windows": cfg.blend_min_windows, "loss_model": None, "loss_market": None, "loss_blend": None}
    if len(keys) < cfg.blend_min_windows or len(windows) < 4:
        return out
    cut = windows[int(len(windows) * 0.7)]
    train = [s[1:] for s in samples if s[0] < cut]
    test = [s[1:] for s in samples if s[0] >= cut]
    a, b = fit_blend(train)
    out.update(loss_model=_blend_loss(test, 1.0, 0.0), loss_market=_blend_loss(test, 0.0, 1.0),
               loss_blend=_blend_loss(test, a, b), fitted_on_train=(a, b))
    if out["loss_blend"] < out["loss_model"] - cfg.blend_min_gain:
        a, b = fit_blend([s[1:] for s in samples])
        out.update(a=a, b=b, applied=True)
    return out
