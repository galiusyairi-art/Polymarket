"""The wallets that win the race (panel 3, 29.10.2026): who takes the shares before us, and how they do it.

Every trade on Polymarket is public, with the wallet that made it (Polymarket's own data API: free, no key;
the same on-chain data Dune sells access to). For the rounds of the markets we trade (first the ones where
our orders were refused), every taker's trade: each wallet's rounds, trades, money, profit, how early before
the end it buys, and how often it bought the same side within a couple of seconds of one of our refused
orders (it raced us). Read only: it never trades.
"""

import json
import time
import urllib.parse

from . import feeds

DATA = "https://data-api.polymarket.com"
ROUNDS = 120  # rounds a study
RACE = (-1.0, 2.0)  # seconds around our refused order (the API's times are whole seconds)
GAP = 0.3  # between two requests


def _condition(slug, cache):
    if slug in cache:
        return cache[slug]
    markets = feeds.get_json(f"{feeds.GAMMA}/markets?slug={urllib.parse.quote(slug)}&closed=true", timeout=10)
    cid = markets[0].get("conditionId") if markets else None
    if cid:
        cache[slug] = cid
    return cid


def _trades(cid):
    out, offset = [], 0
    while offset < 3000:
        page = feeds.get_json(f"{DATA}/trades?market={cid}&limit=500&offset={offset}&takerOnly=true", timeout=15)
        if not page:
            break
        out += page
        if len(page) < 500:
            break
        offset += 500
        time.sleep(GAP)
    return out


def study(db, cache, sleep=time.sleep, fetch_trades=_trades, condition=_condition):
    refused = {}
    for ts, a, m, start, side in db.execute("SELECT ts, asset, minutes, window_start, side FROM fill_tests"
                                            " WHERE ts >= ?", (time.time() - 14 * 86400,)):
        refused.setdefault((a, m, start), []).append((ts, side))
    rounds = db.execute("SELECT asset, minutes, window_start, slug, winner FROM windows WHERE winner IN ('up', 'down')"
                        " AND slug IS NOT NULL ORDER BY window_start DESC LIMIT 2000").fetchall()
    rounds.sort(key=lambda r: ((r[0], r[1], r[2]) not in refused, -r[2]))  # (ours refused first, newest first)
    W, done, errors = {}, 0, []
    total_volume = 0.0
    for a, m, start, slug, winner in rounds[:ROUNDS]:
        try:
            cid = condition(slug, cache)
            if not cid:
                continue
            sleep(GAP)
            trades = fetch_trades(cid)
            sleep(GAP)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}"[:160])
            if len(errors) >= 5 and not done:
                break
            continue
        done += 1
        end = start + m * 60
        ours = refused.get((a, m, start), [])
        for t in trades:
            try:
                who = str(t.get("proxyWallet") or "").lower()
                side, out = str(t.get("side") or "").upper(), str(t.get("outcome") or "").lower()
                size, price, ts = float(t.get("size") or 0), float(t.get("price") or 0), float(t.get("timestamp") or 0)
            except (TypeError, ValueError):
                continue
            if not who or out not in ("up", "down") or size <= 0:
                continue
            w = W.setdefault(who, {"wallet": who, "name": t.get("pseudonym") or t.get("name") or "", "rounds": set(),
                                   "trades": 0, "buys": 0, "volume": 0.0, "pnl": 0.0, "won": 0, "left": [], "raced": 0, "price": 0.0})
            won = out == winner
            pnl = size * (won - price) if side == "BUY" else size * (price - won)
            w["rounds"].add(start)
            w["trades"] += 1
            w["volume"] += size * price
            w["pnl"] += pnl
            total_volume += size * price
            if side == "BUY":
                w["buys"] += 1
                w["won"] += won
                w["price"] += price
                w["left"].append(end - ts)
                if any(our_side == out and RACE[0] <= ts - t0 <= RACE[1] for t0, our_side in ours):
                    w["raced"] += 1
    rows = []
    for w in W.values():
        left = sorted(w["left"])
        rows.append({"wallet": w["wallet"], "name": w["name"], "rounds": len(w["rounds"]), "trades": w["trades"],
                     "volume": round(w["volume"], 2), "pnl": round(w["pnl"], 2), "raced": w["raced"],
                     "win": w["won"] / w["buys"] if w["buys"] else None, "avg_price": w["price"] / w["buys"] if w["buys"] else None,
                     "left_median": left[len(left) // 2] if left else None,
                     "share": w["volume"] / total_volume if total_volume else None})
    top = sorted(rows, key=lambda r: -r["pnl"])[:15]
    racers = sorted((r for r in rows if r["raced"]), key=lambda r: -r["raced"])[:10]
    big10 = sum(r["volume"] for r in sorted(rows, key=lambda r: -r["volume"])[:10])
    return {"ts": time.time(), "rounds": done, "wallets": len(rows), "volume": round(total_volume, 2),
            "top10_share": big10 / total_volume if total_volume else None, "top": top, "racers": racers,
            "refused_rounds": sum(1 for r in rounds[:ROUNDS] if (r[0], r[1], r[2]) in refused),
            "errors": errors[:3], "state": "found" if done >= 50 and racers else "collecting" if done < 50 else "nothing",
            "progress": min(1.0, done / 50)}


def dumps_cache(cache):
    return json.dumps(dict(list(cache.items())[-5000:]))
