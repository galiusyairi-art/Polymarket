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
