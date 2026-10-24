"""The lab: many variants of the strategy, run on paper side by side on every market, and an honest ranking.

Testing hundreds of variants guarantees that some look great by luck, so a variant goes
through stages, each judged on data it hasn't been judged on before:

1. collecting: fewer than lab_min_bets settled bets.
2. candidate: its pessimistic return (mean minus 2.5 standard errors, stricter than the
   1.65 used for the single main strategy) is positive and its recent half is profitable.
   From that moment on, its bets are counted separately: new data, unseen when it was picked.
3. proven: on that new data alone, at least lab_oos_min_bets bets and a positive pessimistic
   return. A candidate whose new data loses is marked failed.

The search near winners (explore) adds neighbours of promising variants: a slightly
different edge or timing. They start at stage 1 like any other, on their own new data.
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
RESOLUTIONS = ("twap", "twapt")  # twapt: the same reading priced with fat tails (sudden jumps)


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


# Maker variants ("twapm"): instead of buying an existing offer (and paying the taker fee),
# post a buy order at our own price and wait for a seller. No fee, a better price, but filled
# only when the market comes to us, which is often when it moves against us: the lab measures it.
MAKER_EDGES = (0.03, 0.05)


def variants_for(minutes):
    entries = ENTRY_WINDOWS.get(minutes) or ((15, minutes * 60 * 2 // 3),)
    taker = [Variant(e, lo, hi, r) for r in RESOLUTIONS for e in EDGES for lo, hi in entries]
    maker = [Variant(e, lo, hi, "twapm") for e in MAKER_EDGES for lo, hi in entries[1:]]
    return taker + maker


def describe(key):
    """A variant's rule in plain Hebrew."""
    v = Variant.from_key(key)
    when = f"{v.max_seconds_left // 60} הדקות האחרונות" if v.max_seconds_left >= 120 else f"{v.max_seconds_left} השניות האחרונות"
    if v.resolution == "twapm":
        return f"פקודה ממתינה במחיר שלנו (בלי עמלה), פער של {v.min_edge * 100:.0f} נקודות, ב-{when}"
    reading = {"twap": "ממוצע 60 שניות", "twapt": "ממוצע 60 שניות, עם קפיצות פתאומיות (זנבות עבים)"}.get(
        v.resolution, "מחיר בשנייה המדויקת")
    return f"פער של {v.min_edge * 100:.0f} נקודות ומעלה, ב-{when}, לפי {reading}"


def money_weighted(rows, z):
    """(return per dollar, pessimistic return per dollar) for bets given as (pnl, paid).

    Return = total profit / total staked, so it always agrees with the dollar figures.
    Pessimistic = that minus z standard errors, with each bet weighted by its stake and
    the effective number of bets reflecting uneven stakes. None with under two bets.
    """
    paid = sum(p for _, p in rows)
    if paid <= 0:
        return None, None
    mean = sum(x for x, _ in rows) / paid
    n_eff = paid ** 2 / sum(p * p for _, p in rows)
    if n_eff < 2:
        return mean, None
    var = sum(p * (x / p - mean) ** 2 for x, p in rows) / paid
    return mean, mean - z * math.sqrt(var * n_eff / (n_eff - 1) / n_eff)


def neighbours(key, minutes):
    """Variants a small step away from `key`: edge one or two points either side, and an entry
    window a third shorter or half again longer."""
    v = Variant.from_key(key)
    length = minutes * 60
    out = set()
    for de in (-0.02, -0.01, 0.01, 0.02):
        edge = round(v.min_edge + de, 2)
        if 0.01 <= edge <= 0.20:
            out.add(Variant(edge, v.min_seconds_left, v.max_seconds_left, v.resolution).key)
    for factor in (2 / 3, 1.5):
        hi = int(round(v.max_seconds_left * factor / 5) * 5)
        if v.min_seconds_left + 20 <= hi <= length - 30 and hi != v.max_seconds_left:
            out.add(Variant(v.min_edge, v.min_seconds_left, hi, v.resolution).key)
    out.discard(key)
    return sorted(out)


def _bets_for_positive_lower(n, mean, lower):
    """How many bets the pessimistic return needs to turn positive, if the strategy keeps
    returning what it returns now: the uncertainty shrinks like 1/sqrt(bets)."""
    if mean is None or lower is None or mean <= 0 or n <= 0:
        return None
    spread = (mean - lower) * math.sqrt(n)
    return max(n, math.ceil((spread / mean) ** 2) + 1)


def next_stage(cfg, status, n, mean, lower, oos_n, oos_roi, oos_lower, first_ts, last_ts):
    """What it takes to reach the next stage: bets so far, bets needed (None: not while it
    loses), and how many bets an hour it places (to turn that into time)."""
    hours = (last_ts - first_ts) / 3600 if last_ts > first_ts else 0
    rate = n / hours if hours >= 0.25 else None
    if status == "collecting":
        return {"to": "promising", "have": n, "need": cfg.lab_min_bets, "rate": rate}
    if status in ("promising", "losing"):
        return {"to": "candidate", "have": n, "need": _bets_for_positive_lower(n, mean, lower), "rate": rate}
    if status == "candidate":
        need = cfg.lab_oos_min_bets
        if oos_roi is not None and oos_roi > 0 and oos_n >= cfg.dash_min_bets:
            need = max(need, _bets_for_positive_lower(oos_n, oos_roi, oos_lower) or need)
        return {"to": "proven", "have": oos_n, "need": need if (oos_roi or 0) > 0 or oos_n < cfg.dash_min_bets else None,
                "rate": rate}
    return None


def leaderboard(db, cfg, current=None, candidates=None, dynamic=None):
    """Every lab variant's paper results, best first, plus the top picks and per-market totals.

    current: the (asset, minutes, variant) keys the lab runs now; retired variants are left out.
    candidates: {(asset, minutes, variant): when it became a candidate}.
    dynamic: keys added by the search near winners (marked as such).
    """
    candidates = candidates or {}
    dynamic = dynamic or set()
    rows = db.execute(
        "SELECT asset, minutes, variant, ts, pnl, cost + fee AS paid FROM bets"
        " WHERE mode = 'lab' AND pnl IS NOT NULL AND cost + fee > 0 ORDER BY ts"
    ).fetchall()
    groups = {}
    for r in rows:
        key = (r["asset"], r["minutes"], r["variant"])
        if current is None or key in current:
            groups.setdefault(key, []).append(r)

    board = []
    for (asset, minutes, variant), bets in groups.items():
        rows_ = [(b["pnl"], b["paid"]) for b in bets]
        n = len(rows_)
        mean, lower = money_weighted(rows_, cfg.lab_z)
        recent, _ = money_weighted(rows_[n // 2:], cfg.lab_z)
        paid = sum(b["paid"] for b in bets)
        pnl = sum(b["pnl"] for b in bets)
        key = (asset, minutes, variant)
        since = candidates.get(key)
        oos = [(b["pnl"], b["paid"]) for b in bets if since is not None and b["ts"] >= since]
        oos_roi, oos_lower = money_weighted(oos, cfg.lab_oos_z) if oos else (None, None)
        becomes_candidate = False
        if since is not None:
            if len(oos) >= cfg.lab_oos_min_bets and oos_lower is not None and oos_lower > 0:
                status = "proven"
            elif len(oos) >= cfg.lab_oos_min_bets and (oos_roi or 0) <= 0:
                status = "failed"
            else:
                status = "candidate"
        elif n < cfg.lab_min_bets:
            status = "collecting"
        elif lower is not None and lower > 0 and recent is not None and recent > 0:
            status, becomes_candidate = "candidate", True
        elif mean > 0:
            status = "promising"
        else:
            status = "losing"
        board.append({
            "asset": asset, "minutes": minutes, "variant": variant, "rule": describe(variant),
            "bets": n, "wins": sum(1 for x, _ in rows_ if x > 0), "paid": paid, "pnl": pnl,
            # With only a handful of bets a pessimistic figure means nothing; don't show one.
            "roi": mean, "roi_pessimistic": lower if n >= cfg.dash_min_bets else None,
            "roi_recent": recent if n >= cfg.dash_min_bets else None, "status": status,
            "first_ts": bets[0]["ts"], "last_ts": bets[-1]["ts"],
            "candidate_since": since, "new_candidate": becomes_candidate,
            "oos_bets": len(oos), "oos_roi": oos_roi,
            "oos_pessimistic": oos_lower if len(oos) >= cfg.dash_min_bets else None,
            "auto": key in dynamic,
            "next": next_stage(cfg, status, n, mean, lower, len(oos), oos_roi, oos_lower,
                               bets[0]["ts"], bets[-1]["ts"]),
        })
    order = {"proven": 0, "candidate": 1, "promising": 2, "collecting": 3, "failed": 4, "losing": 5}
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
    leading = [b for b in board if b["status"] in ("candidate", "promising")][:5]

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
        "candidates": leading,
        "stage_counts": {s: sum(1 for b in board if b["status"] == s) for s in order},
        "markets": sorted(markets.values(), key=lambda m: (m["asset"], m["minutes"])),
        "settled_bets": sum(len(bets) for bets in groups.values()),
        "active": len(groups),
        "pnl": sum(b["pnl"] for b in board),
        "paid": sum(b["paid"] for b in board),
    }
