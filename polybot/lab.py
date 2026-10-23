"""The lab: many variants of the strategy, run on paper side by side on every market, and an honest ranking.

Testing hundreds of variants guarantees that some look great by luck. The ranking
therefore scores each variant by a pessimistic estimate of its return per dollar
(mean minus 2.5 standard errors, stricter than the 1.65 used for the single main
strategy), requires a minimum number of bets, and requires the more recent half of
its bets to be profitable too. A variant that only looks good on old data, or on a
handful of bets, is not recommended.
"""
import math
from dataclasses import dataclass

EDGES = (0.03, 0.05, 0.08, 0.12)
ENTRY_WINDOWS = {
    15: ((15, 60), (15, 180), (15, 600)),
    5: ((10, 45), (10, 120), (10, 240)),
}
# Measured against real outcomes: the 60s TWAP reading matched Polymarket in 61 of 61
# resolved windows, the point reading in 58 of 61, so only TWAP variants are run.
# Windows still record both readings, and the dashboard keeps checking them.
RESOLUTIONS = ("twap",)


@dataclass(frozen=True)
class Variant:
    min_edge: float
    min_seconds_left: int
    max_seconds_left: int
    resolution: str

    @property
    def key(self):
        return f"{self.resolution}|{self.min_edge:.2f}|{self.min_seconds_left}-{self.max_seconds_left}"

    @classmethod
    def from_key(cls, key):
        resolution, edge, window = key.split("|")
        lo, hi = window.split("-")
        return cls(float(edge), int(lo), int(hi), resolution)


def variants_for(minutes):
    entries = ENTRY_WINDOWS.get(minutes) or ((15, minutes * 60 * 2 // 3),)
    return [Variant(e, lo, hi, r) for r in RESOLUTIONS for e in EDGES for lo, hi in entries]


def describe(key):
    """A variant's rule in plain Hebrew."""
    v = Variant.from_key(key)
    when = f"{v.max_seconds_left // 60} הדקות האחרונות" if v.max_seconds_left >= 120 else f"{v.max_seconds_left} השניות האחרונות"
    reading = "ממוצע 60 שניות" if v.resolution == "twap" else "מחיר בשנייה המדויקת"
    return f"פער של {v.min_edge * 100:.0f} נקודות ומעלה, ב-{when}, לפי {reading}"


def _stats(returns, z):
    n = len(returns)
    mean = sum(returns) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1)) if n > 1 else float("inf")
    return mean, mean - z * sd / math.sqrt(n) if n > 1 else None


def leaderboard(db, cfg):
    """Every lab variant's paper results, best first, plus the top picks and per-market totals."""
    rows = db.execute(
        "SELECT asset, minutes, variant, ts, pnl, cost + fee AS paid FROM bets"
        " WHERE mode = 'lab' AND pnl IS NOT NULL AND cost + fee > 0 ORDER BY ts"
    ).fetchall()
    groups = {}
    for r in rows:
        groups.setdefault((r["asset"], r["minutes"], r["variant"]), []).append(r)

    board = []
    for (asset, minutes, variant), bets in groups.items():
        returns = [b["pnl"] / b["paid"] for b in bets]
        n = len(returns)
        mean, lower = _stats(returns, cfg.lab_z)
        half = returns[n // 2:]
        recent = sum(half) / len(half) if half else None
        paid = sum(b["paid"] for b in bets)
        pnl = sum(b["pnl"] for b in bets)
        if n < cfg.lab_min_bets:
            status = "collecting"
        elif lower is not None and lower > 0 and recent is not None and recent > 0:
            status = "proven"
        elif mean > 0:
            status = "promising"
        else:
            status = "losing"
        board.append({
            "asset": asset, "minutes": minutes, "variant": variant, "rule": describe(variant),
            "bets": n, "wins": sum(1 for r in returns if r > 0), "paid": paid, "pnl": pnl,
            "roi": mean, "roi_pessimistic": lower, "roi_recent": recent, "status": status,
            "first_ts": bets[0]["ts"], "last_ts": bets[-1]["ts"],
        })
    order = {"proven": 0, "promising": 1, "collecting": 2, "losing": 3}
    board.sort(key=lambda b: (order[b["status"]], -(b["roi_pessimistic"] if b["roi_pessimistic"] is not None
                                                     else -1e9), -b["bets"]))

    # Top picks: proven variants on three different markets (variants of one market
    # are highly correlated, so three of them would be one bet, not three).
    picks, seen = [], set()
    for b in board:
        if b["status"] == "proven" and (b["asset"], b["minutes"]) not in seen:
            picks.append(b)
            seen.add((b["asset"], b["minutes"]))
        if len(picks) == 3:
            break
    candidates = [b for b in board if b["status"] == "promising"][:5]

    markets = {}
    for r in db.execute(
        "SELECT asset, minutes, COUNT(*) AS windows, SUM(winner IS NOT NULL) AS resolved FROM windows"
        " GROUP BY asset, minutes"
    ).fetchall():
        markets[(r["asset"], r["minutes"])] = {"asset": r["asset"], "minutes": r["minutes"],
                                               "windows": r["windows"], "resolved": r["resolved"] or 0,
                                               "bets": 0, "best": None}
    for b in board:
        m = markets.setdefault((b["asset"], b["minutes"]), {"asset": b["asset"], "minutes": b["minutes"],
                                                            "windows": 0, "resolved": 0, "bets": 0, "best": None})
        m["bets"] += b["bets"]
        if m["best"] is None:
            m["best"] = {k: b[k] for k in ("variant", "rule", "status", "roi", "roi_pessimistic", "bets")}
    return {
        "board": board,
        "picks": picks,
        "candidates": candidates,
        "markets": sorted(markets.values(), key=lambda m: (m["asset"], m["minutes"])),
        "settled_bets": len(rows),
    }
