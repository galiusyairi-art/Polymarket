"""Why did we stop making money? (the owner, 31.10.2026: "the starred strategies made money Sunday to Friday, and
since then only lose: check them all, thoroughly").

Each starred strategy's simulated trades, before and after a cut (by default the moment the simulation started to
trade on real money's conditions: the real delay, and paying up to a few cents over the price seen to be filled).
Three causes told apart:
  - the price: what a trade paid over the price it saw (the slip), and what the trades after the cut would have made
    at the price they saw ("at the price seen"): if that is as good as before, it is the paying more;
  - the model: its chance against how often the trades won (calibration); a model that now promises more than it
    wins is the market moving away from it (a quieter weekend, other people);
  - chance: the drop against how much such returns jump about (a z score); a drop within chance is no finding.
"""
import math

NEED = 20  # trades on each side of the cut before a verdict
Z = 2.0


def _part(rows, fee):
    n = len(rows)
    if not n:
        return {"n": 0}
    roi = [r["pnl"] / r["paid"] for r in rows]
    seen = []
    for r in rows:
        p = r["seen"] if r["seen"] else r["avg"]
        eff = p + fee(p)
        seen.append((1.0 / eff - 1.0) if r["won"] else -1.0)
    mean = sum(roi) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in roi) / (n - 1)) if n > 1 else 0.0
    span = max(1.0, (max(r["ts"] for r in rows) - min(r["ts"] for r in rows)) / 3600)
    wins = sum(1 for r in rows if r["won"])
    return {"n": n, "wins": wins, "win_rate": wins / n, "chance": sum(r["chance"] for r in rows if r["chance"] is not None) / max(1, sum(1 for r in rows if r["chance"] is not None)),
            "roi": mean, "sd": sd, "roi_seen": sum(seen) / n, "pnl": sum(r["pnl"] for r in rows),
            "price": sum(r["avg"] for r in rows) / n, "slip": sum(r["avg"] - r["seen"] for r in rows if r["seen"]) / max(1, sum(1 for r in rows if r["seen"])),
            "per_hour": n / span, "cheap": sum(1 for r in rows if r["avg"] < 0.20) / n}


def verdict(before, after):
    """(code, Hebrew words) for one strategy."""
    if (before.get("n") or 0) < NEED or (after.get("n") or 0) < NEED:
        return "few", "מעט מדי טריידים כדי לומר"
    drop = before["roi"] - after["roi"]
    se = math.sqrt(before["sd"] ** 2 / before["n"] + after["sd"] ** 2 / after["n"]) or 1e-9
    z = drop / se
    price = after["roi_seen"] - after["roi"]  # (what paying over the price seen cost after the cut)
    # (the win rate is a far steadier measure than a return on cheap shares: a drop in it is often clear where the
    # return's still "chance", 31.10.2026: 52% to 42% over 1,978 trades, z 4, while the return's z was 1)
    nb, na = before["n"], after["n"]
    pool = (before["wins"] + after["wins"]) / (nb + na)
    zw = (before["win_rate"] - after["win_rate"]) / (math.sqrt(pool * (1 - pool) * (1 / nb + 1 / na)) or 1e-9)
    gap_b, gap_a = before["win_rate"] - before["chance"], after["win_rate"] - after["chance"]
    if z < Z and zw >= Z and gap_a <= -0.05 and price < 0.03:
        return "model", (f"המודל: אחוז הניצחונות ירד מ-{before['win_rate'] * 100:.0f}% ל-{after['win_rate'] * 100:.0f}% (ירידה ברורה, ציון {zw:.1f})"
                         f" בזמן שהמודל ממשיך להבטיח {after['chance'] * 100:.0f}%. הפער בין המודל למציאות גדל מ-{gap_b * 100:.0f} ל-{gap_a * 100:.0f} נקודות."
                         " המחיר ששילמנו לא השתנה. השוק זז מתחת למודל (למשל סוף שבוע שקט יותר)")
    if z < Z:
        return "chance", (f"הירידה בגבולות המקרה (ציון {z:.1f}, צריך {Z:g}): בהימורים זולים התוצאות קופצות הרבה"
                          + (f". אבל המחיר לבד לקח {price * 100:.0f}% מהתשואה ({after['slip'] * 100:.1f}¢ מעל המחיר שנראה)" if price >= 0.03 else ""))
    model = after["win_rate"] - after["chance"]  # (below 0: the model promises more than it wins)
    if price >= drop / 2:
        return "price", f"המחיר: אחרי השינוי משלמים {after['slip'] * 100:.1f}¢ מעל המחיר שנראה, וזה לוקח {price * 100:.0f}% מהתשואה"
    if model <= -0.05:
        return "model", (f"המודל: הוא נותן {after['chance'] * 100:.0f}% והטריידים מנצחים רק {after['win_rate'] * 100:.0f}%:"
                         " השוק השתנה מתחתיו (למשל סוף שבוע שקט יותר)")
    return "other", f"ירידה אמיתית (ציון {z:.1f}), לא מהמחיר ולא מהכיול לבד"


def study(db, keys, cut, fee, names=None, now=None):
    """keys: the starred (asset, minutes, variant); cut: the moment compared around."""
    names = names or {}
    out, every = [], {"before": [], "after": []}
    for a, m, v in keys:
        rows = []
        for ts, avg, seen, prob, side, pnl, cost, fee_paid in db.execute(
                "SELECT ts, avg_price, best_ask, fair_prob, side, pnl, cost, fee FROM bets WHERE mode = 'paper'"
                " AND status = 'filled' AND pnl IS NOT NULL AND asset = ? AND minutes = ? AND variant = ?", (a, m, v)):
            paid = (cost or 0) + (fee_paid or 0)
            if not paid or not avg:
                continue
            rows.append({"ts": ts, "avg": avg, "seen": seen, "chance": prob, "pnl": pnl, "paid": paid, "won": pnl > 0})
        b = [r for r in rows if r["ts"] < cut]
        f = [r for r in rows if r["ts"] >= cut]
        every["before"] += b
        every["after"] += f
        B, A = _part(b, fee), _part(f, fee)
        code, words = verdict(B, A)
        out.append({"asset": a, "minutes": m, "variant": v, "name": names.get(f"{a}|{m}|{v}", ""),
                    "before": B, "after": A, "verdict": code, "words": words})
    B, A = _part(every["before"], fee), _part(every["after"], fee)
    code, words = verdict(B, A)
    return {"cut": cut, "now": now, "rows": out, "all": {"before": B, "after": A, "verdict": code, "words": words}}
