"""The risk/reward filter: bet only when what a win pays is at least `ratio` times what a loss costs.

On a binary market, a share bought at price p (fee included) wins 1 - p and loses p, so a
ratio of R means paying at most 1 / (1 + R) per share: 1:1 is at most 50c, 1:3 at most 25c.

The filter applies to the main strategy only (simulation and real money); the lab keeps
betting without it, so its bets show what each ratio would have done. That is what the
automatic choice reads: on the settled bets, for each ratio, the bets that pass it and how
they did, judged by the same pessimistic return as everything else. Only the main crypto
strategy's own simulated bets count: the filter applies to it, and the lab's many
experimental rules (most of them weak on purpose) would say nothing about it.
"""
from . import lab

RATIOS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
MODES = ("off", "manual", "auto")


def max_price(ratio):
    """Highest price per share (fee included) a ratio allows."""
    return 1.0 / (1.0 + ratio) if ratio else 1.0


def passes(cost_per_share, ratio):
    return not ratio or 1.0 - cost_per_share >= ratio * cost_per_share - 1e-12


def table(db, cfg):
    """For each ratio, how the main strategy's settled simulated bets that pass it did."""
    rows = db.execute(
        "SELECT shares, cost + fee AS paid, pnl FROM bets WHERE mode = 'paper' AND minutes > 0 AND status = 'filled'"
        " AND pnl IS NOT NULL AND shares > 0"
    ).fetchall()
    out = []
    for ratio in RATIOS:
        kept = [(r["pnl"], r["paid"]) for r in rows if r["paid"] > 0 and passes(r["paid"] / r["shares"], ratio)]
        roi, low = lab.money_weighted(kept, cfg.lab_oos_z)
        out.append({"ratio": ratio, "max_price": max_price(ratio), "bets": len(kept),
                    "wins": sum(1 for pnl, _ in kept if pnl > 0), "pnl": sum(pnl for pnl, _ in kept),
                    "roi": roi, "roi_pessimistic": low})
    return out


def choose(rows, min_bets):
    """The ratio the data favours: the best pessimistic return among ratios with enough bets.

    Returns (ratio, reason). Stays at no filter until some filter beats it on the evidence.
    """
    ok = [r for r in rows if r["bets"] >= min_bets and r["roi_pessimistic"] is not None]
    if not ok:
        return 0.0, "waiting"
    best = max(ok, key=lambda r: r["roi_pessimistic"])
    base = next((r for r in ok if r["ratio"] == 0.0), None)
    if base is not None and best["roi_pessimistic"] <= base["roi_pessimistic"]:
        return 0.0, "no_filter_best"
    return best["ratio"], "best"
