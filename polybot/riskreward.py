"""Risk/reward: what a win pays against what a loss costs, per share.

On a binary market, a share bought at price p (fee included) wins 1 - p and loses p, so a
ratio of R means paying at most 1 / (1 + R) per share: $7 for every $10 (0.7) is at most 59c.
A rule may carry a minimum ratio of its own (lab.Variant.min_ratio); the lab tests each rule
with and without one, so a filter that helps is proven like any other rule.
"""


def passes_range(cost_per_share, lo, hi):
    """A win pays at least lo and at most hi (0: no bound) per $1 at risk."""
    win = 1.0 - cost_per_share
    return (not lo or win >= lo * cost_per_share - 1e-12) and (not hi or win <= hi * cost_per_share + 1e-12)


def max_price(ratio):
    """Highest price per share (fee included) a ratio allows."""
    return 1.0 / (1.0 + ratio) if ratio else 1.0
