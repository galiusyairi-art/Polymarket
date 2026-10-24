"""The stake that makes the most money: every settled trade replayed at other stakes.

Each bet keeps the levels of the order book its rule would have bought (bets.book). A bigger
stake walks further up those levels, so each share costs more and the edge shrinks; past the
depth on offer the order isn't filled at all (all or nothing, like a real order). Replaying
every settled trade at each stake shows where the total profit peaks. This is about making the
most money per trade, not about the wallet's size.
"""
import json

from .lab import Variant, entries
from .model import taker_fee_per_share

STAKES = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500)
MIN_TRADES = 30  # below this many trades at a stake, its total is too much luck to recommend
RECENT = 3000  # the newest settled trades with a recorded book


def replay(levels, stake, side, winner, cfg):
    """Profit of buying `stake` dollars from `levels` [(price, shares)], or None when the levels
    can't take the whole stake (the order would be killed)."""
    shares = paid = 0.0
    for price, size in sorted(levels):
        per_share = price + taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
        take = min(size, (stake - paid) / per_share)
        shares += take
        paid += take * per_share
        if paid >= stake * 0.999:
            return (shares if side == winner else 0.0) - paid, paid
    return None


def curve(db, cfg, keys, mode="lab"):
    """For each stake: trades that would have filled, wins, profit and return, over the settled
    trades of the strategies in `keys` {(asset, minutes, variant)}. The stake is the one set for a
    round; a rule that enters several times splits it, as it does when trading."""
    if not keys:
        return {"rows": [], "trades": 0, "best": None, "min_trades": MIN_TRADES}
    variants = sorted({v for _, _, v in keys})
    rows = db.execute(
        "SELECT asset, minutes, variant, side, winner, book FROM bets WHERE mode = ? AND book IS NOT NULL"
        f" AND winner IS NOT NULL AND status = 'filled' AND variant IN ({','.join('?' * len(variants))})"
        " ORDER BY id DESC LIMIT ?", (mode, *variants, RECENT)).fetchall()
    trades = []
    for asset, minutes, variant, side, winner, book in rows:
        if (asset, minutes, variant) not in keys:
            continue
        try:
            split = entries(Variant.from_key(variant))
        except Exception:
            split = 1
        trades.append((json.loads(book), side, winner, split))
    out = []
    for stake in STAKES:
        n = wins = 0
        pnl = paid = 0.0
        for levels, side, winner, split in trades:
            r = replay(levels, stake / split, side, winner, cfg)
            if r is None:
                continue
            n += 1
            wins += side == winner
            pnl += r[0]
            paid += r[1]
        out.append({"stake": stake, "trades": n, "wins": wins, "pnl": pnl, "paid": paid,
                    "roi": pnl / paid if paid else None})
    usable = [r for r in out if r["trades"] >= MIN_TRADES]
    best = max(usable, key=lambda r: r["pnl"], default=None)
    return {"rows": out, "trades": len(trades), "min_trades": MIN_TRADES,
            "best": best["stake"] if best and best["pnl"] > 0 else None,
            "reason": "best" if best and best["pnl"] > 0 else ("losing" if best else "waiting")}


def depth_now(asks, prob, min_edge, cfg):
    """Right now: how many dollars this side of the book offers with at least `min_edge`."""
    dollars = 0.0
    for price, size in sorted(asks or ()):
        per_share = price + taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
        if prob - per_share < min_edge:
            break
        dollars += size * per_share
    return dollars
