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
    # Risk/reward, part of the rule itself: buy only where a win pays at least this much per
    # dollar at risk (0.7 = $7 for every $10). 0 = no filter. The lab tests the same rule with
    # and without it, so a filter that helps is proven like anything else and then runs as is.
    min_ratio: float = 0.0
    # A filter put on the rule where it runs (the simulation or real money), by the system's
    # research on this rule's own trades or by the owner. Not part of the rule's name: the
    # strategy stays the same strategy, with the same record.
    filter_ratio: float = 0.0
    # The amount a round where it runs (the simulation or real money), set by the system's research
    # on this rule's own trades or by the owner; 0 = the mode's amount. Not part of the name either.
    stake: float = 0.0

    @property
    def ratio(self):
        """The risk/reward filter in force: the rule's own, or the one put on it, whichever is stricter."""
        return max(self.min_ratio, self.filter_ratio)

    @property
    def key(self):
        base = f"{self.resolution}|{self.min_edge:.2f}|{self.min_seconds_left}-{self.max_seconds_left}"
        return base + (f"|r{self.min_ratio:.1f}" if self.min_ratio else "")

    @classmethod
    def from_key(cls, key):
        resolution, edge, window, *rest = key.split("|")
        lo, hi = window.split("-")
        ratio = float(rest[0][1:]) if rest and rest[0].startswith("r") else 0.0
        return cls(float(edge), int(lo), int(hi), resolution, ratio)

    def with_ratio(self, ratio):
        return Variant(self.min_edge, self.min_seconds_left, self.max_seconds_left, self.resolution, ratio)


# The risk/reward filters the lab tries on promising rules and on the owner's own: a win paying
# at least $3, $5 or $7 for every $10 at risk (buying up to 77¢, 67¢, 59¢ a share, fee included).
RATIOS = (0.3, 0.5, 0.7)


def takes_ratio(v):
    """Rules a risk/reward filter applies to: the usual taker rules (not maker orders, whose price
    is their own, nor lottery tickets, which are capped at a few cents anyway)."""
    return v.resolution in ("twap", "twapt", "twap3", "point")


# Maker variants ("twapm"): instead of buying an existing offer (and paying the taker fee),
# post a buy order at our own price and wait for a seller. No fee, a better price, but filled
# only when the market comes to us, which is often when it moves against us: the lab measures it.
MAKER_EDGES = (0.03, 0.05)

# Both sides ("pair<T>"): buy the side that turned cheap (at most min_edge a share, fee included),
# then the other side the moment the two together cost at most T cents a share (fees included),
# the same number of shares of each. One of the two always pays $1 a share, so once both are in, the
# round makes (1 - pair cost) a share whoever wins. The risk is the second side never getting that
# cheap: then the first is an ordinary bet on its side. Known on Polymarket as "pair cost" or
# "dump and hedge". The first side only while at least min_seconds_left remain, so there is time
# for the second; the second until PAIR_LAST seconds before the close.
# On a first day of recordings the cheap first legs (15-25c) with a low target (80-85c) did best
# and the dearer ones lost; the grid spans both, and the lab's stages decide.
PAIR_CAPS = (0.15, 0.20, 0.25, 0.30, 0.40)
PAIR_TARGETS = (80, 85, 90, 95)
PAIR_FIRST = {15: ((120, 885), (450, 885)), 5: ((60, 285), (150, 285))}
PAIR_LAST = 5

# Several entries ("twap3"): the same rule, but up to three entries a window (a third of the stake
# each, the edge checked again each time, ENTRY_GAP seconds apart) instead of one: whether
# going in more than once beats a single bet of the whole amount.
MULTI_ENTRIES = 3
MULTI_EDGES = (0.05, 0.08)
ENTRY_GAP = 20


def variants_for(minutes):
    entries = ENTRY_WINDOWS.get(minutes) or ((15, minutes * 60 * 2 // 3),)
    taker = [Variant(e, lo, hi, r) for r in RESOLUTIONS for e in EDGES for lo, hi in entries]
    maker = [Variant(e, lo, hi, "twapm") for e in MAKER_EDGES for lo, hi in entries[1:]]
    multi = [Variant(e, lo, hi, "twap3") for e in MULTI_EDGES for lo, hi in entries]
    firsts = PAIR_FIRST.get(minutes) or ((minutes * 60 // 4, minutes * 60 - 15),)
    pairs = [Variant(cap, lo, hi, f"pair{t}") for t in PAIR_TARGETS for cap in PAIR_CAPS for lo, hi in firsts]
    return taker + maker + multi + pairs


def family(key):
    """Which part of the lab a variant belongs to: the both-sides rules have their own tab."""
    return "pair" if key.startswith("pair") else "main"


def reading(v):
    """The model reading a variant decides on (the special kinds all use the 60-second average)."""
    return "twap" if v.resolution == "twapm" or is_lotto(v) or v.resolution == "twap3" else v.resolution


def is_lotto(v):
    return v.resolution.startswith("lotto")


def is_pair(v):
    return v.resolution.startswith("pair")


def pair_target(v):
    """The most the two sides may cost together, per share (fees included)."""
    return int(v.resolution[4:]) / 100


def lotto_min_fair(v):
    return int(v.resolution[5:] or 0) / 100


def entries(v):
    """How many times a variant may enter in one window."""
    return MULTI_ENTRIES if v.resolution == "twap3" else 1


def describe(key):
    """A variant's rule in plain Hebrew."""
    v = Variant.from_key(key)
    when = f"{v.max_seconds_left // 60} הדקות האחרונות" if v.max_seconds_left >= 120 else f"{v.max_seconds_left} השניות האחרונות"
    if is_pair(v):
        lo = f"{v.min_seconds_left // 60} דקות" if v.min_seconds_left >= 120 else f"{v.min_seconds_left} שניות"
        return (f"שני הצדדים: קונים צד כשהוא עד {v.min_edge * 100:.0f}¢, ואת הצד השני כשהשניים יחד עולים עד"
                f" {pair_target(v) * 100:.0f}¢ למניה (עמלות כלולות). הצד הראשון רק כשנשארו לפחות {lo}")
    if is_lotto(v):
        least = lotto_min_fair(v)
        return (f"כרטיס לוטו: קונים רק עד {v.min_edge * 100:.0f}¢ למניה, ב-{when}"
                + (f", כשהמודל נותן לצד הזה לפחות {least * 100:.0f}%" if least else ""))
    if v.resolution == "twap3":
        return f"פער של {v.min_edge * 100:.0f} נקודות ומעלה, ב-{when}, עד {MULTI_ENTRIES} כניסות (שליש מהסכום בכל אחת)" + ratio_text(v)
    if v.resolution == "twapm":
        return f"פקודה ממתינה במחיר שלנו (בלי עמלה), פער של {v.min_edge * 100:.0f} נקודות, ב-{when}"
    reading = {"twap": "ממוצע 60 שניות", "twapt": "ממוצע 60 שניות, עם קפיצות פתאומיות (זנבות עבים)"}.get(
        v.resolution, "מחיר בשנייה המדויקת")
    return f"פער של {v.min_edge * 100:.0f} נקודות ומעלה, ב-{when}, לפי {reading}" + ratio_text(v)


def ratio_text(v):
    if not v.min_ratio:
        return ""
    return f", רק כשניצחון מרוויח לפחות ${v.min_ratio * 10:g} על כל $10 (קונים עד {100 / (1 + v.min_ratio):.0f}¢)"


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
            out.add(Variant(edge, v.min_seconds_left, v.max_seconds_left, v.resolution, v.min_ratio).key)
    for factor in (2 / 3, 1.5):
        hi = int(round(v.max_seconds_left * factor / 5) * 5)
        if v.min_seconds_left + 20 <= hi <= length - 30 and hi != v.max_seconds_left:
            out.add(Variant(v.min_edge, v.min_seconds_left, hi, v.resolution, v.min_ratio).key)
    out.discard(key)
    return sorted(out)


def ratio_versions(key):
    """The same rule with each risk/reward filter the lab tries (none for rules it doesn't fit)."""
    v = Variant.from_key(key)
    if not takes_ratio(v) or v.min_ratio:
        return []
    return [v.with_ratio(r).key for r in RATIOS]


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


def leaderboard(db, cfg, current=None, candidates=None, dynamic=None, since=None):
    """Every lab variant's paper results, best first, plus the top picks and per-market totals.

    current: the (asset, minutes, variant) keys the lab runs now; retired variants are left out.
    candidates: {(asset, minutes, variant): when it became a candidate}.
    dynamic: keys added by the search near winners (marked as such).
    since: count only bets placed from then on (the owner started counting afresh).
    """
    candidates = candidates or {}
    dynamic = dynamic or set()
    rows = db.execute(
        "SELECT asset, minutes, variant, ts, pnl, cost + fee AS paid, window_start, side FROM bets"
        " WHERE mode = 'lab' AND pnl IS NOT NULL AND cost + fee > 0 AND ts >= ? ORDER BY ts", (since or 0,)
    ).fetchall()
    groups = {}
    for r in rows:
        key = (r["asset"], r["minutes"], r["variant"])
        if current is None or key in current:
            groups.setdefault(key, []).append(r)

    board = []
    for (asset, minutes, variant), bets in groups.items():
        locked = None
        if family(variant) == "pair":
            bets, locked = _rounds(bets)
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
            # The same distance above: what a lucky run of the same strategy could look like.
            "roi_optimistic": 2 * mean - lower if lower is not None and n >= cfg.dash_min_bets else None,
            "roi_recent": recent if n >= cfg.dash_min_bets else None, "status": status,
            "first_ts": bets[0]["ts"], "last_ts": bets[-1]["ts"],
            "candidate_since": since, "new_candidate": becomes_candidate,
            "oos_bets": len(oos), "oos_roi": oos_roi,
            "oos_pessimistic": oos_lower if len(oos) >= cfg.dash_min_bets else None,
            "auto": key in dynamic, "family": family(variant), "locked": locked,
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
        "per_round": per_round([r for r in rows if current is None or (r["asset"], r["minutes"], r["variant"]) in current]),
        "families": {f: _family_summary(f, board, rows, current, order) for f in ("main", "pair")},
    }


def _rounds(bets):
    """A both-sides rule's legs, one row per round (its two legs together), and how many rounds
    got both legs (locked: they make money whoever wins)."""
    by = {}
    for b in bets:
        r = by.setdefault(b["window_start"], {"ts": b["ts"], "pnl": 0.0, "paid": 0.0, "window_start": b["window_start"],
                                              "side": "pair", "sides": set()})
        r["pnl"] += b["pnl"]
        r["paid"] += b["paid"]
        r["sides"].add(b["side"])
    rounds = sorted(by.values(), key=lambda r: r["ts"])
    return rounds, sum(1 for r in rounds if len(r["sides"]) == 2)


def _family_summary(fam, board, rows, current, order):
    mine = [b for b in board if b["family"] == fam]
    return {"active": len(mine), "settled_bets": sum(b["bets"] for b in mine),
            "locked": sum(b["locked"] or 0 for b in mine) if fam == "pair" else None,
            "pnl": sum(b["pnl"] for b in mine), "paid": sum(b["paid"] for b in mine),
            "stage_counts": {s: sum(1 for b in mine if b["status"] == s) for s in order},
            "per_round": per_round([r for r in rows if family(r["variant"]) == fam and
                                    (current is None or (r["asset"], r["minutes"], r["variant"]) in current)])}


def per_round(rows, stake=10.0):
    """The lab's result counted once per round and side: many variants buying the same round would
    each count the same opportunity (which, for real, only the first buyer gets). Each round-side's
    return per dollar (averaged over the variants that bought it) times one stake. A both-sides
    rule's two legs are one position (its "side" is the pair)."""
    pairs, rest = {}, []
    for r in rows:
        if "variant" not in r.keys() or family(r["variant"]) != "pair":
            rest.append(r)
            continue
        k = (r["asset"], r["minutes"], r["variant"], r["window_start"])
        m = pairs.setdefault(k, {"asset": r["asset"], "minutes": r["minutes"], "window_start": r["window_start"],
                                 "side": "pair", "pnl": 0.0, "paid": 0.0})
        m["pnl"] += r["pnl"]
        m["paid"] += r["paid"]
    # (profit, money in) per entry: a stake's worth at the entry's return, or a pair as it was bought
    # (a $10 pair is 10 pairs of shares, so it is already one stake when both legs are in; a round
    # left with only its cheap first leg put less in, and counts as what it was).
    rounds = {}
    for r in rest:
        if r["paid"] > 0:
            rounds.setdefault((r["asset"], r["minutes"], r["window_start"], r["side"]), []).append(
                (stake * r["pnl"] / r["paid"], stake))
    for r in pairs.values():
        if r["paid"] > 0:
            rounds.setdefault((r["asset"], r["minutes"], r["window_start"], "pair"), []).append((r["pnl"], r["paid"]))
    avg = [(sum(x for x, _ in v) / len(v), sum(p for _, p in v) / len(v)) for v in rounds.values()]
    pnl, paid = sum(x for x, _ in avg), sum(p for _, p in avg)
    return {"rounds": len(rounds), "pnl": pnl, "roi": pnl / paid if paid else None,
            "wins": sum(1 for x, _ in avg if x > 0)}
