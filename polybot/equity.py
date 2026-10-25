"""Stocks (the "שוק ההון" area), phase 1: US stocks' daily prices, the market's picture, and a swing
backtest the owner can play with. A world apart from Polymarket: its own tables (eq_*), its own
thread; a fault here never touches the trading there.

The owner's way, as rules (an experienced trader's, and the expert panel's):
- a stop always, set before the entry and not too tight (a multiple of the stock's daily range, ATR);
- the take-profit rises with the profit, leaving room to breathe (a trailing stop), and a trade that
  was winning doesn't turn into a loss (the stop moves to the entry);
- risk 0.5% to 2% of the account a trade (the loss if the stop is hit), a position at most $10K;
- reward to risk at least 3 to 1;
- the market's picture first: its primary and secondary trend, fear (VIX), how many stocks rise.

Daily bars from Yahoo Finance (prices adjusted for splits), two years, refreshed after each close.
Entries at the next day's open (a signal is known only at the close), exits on the day's range: a
stop and a target hit the same day count as the stop (the worse case). Costs: Interactive Brokers'
commission ($0.005 a share, $1 at least, each way) and a little slippage.
"""

import datetime as dt
import json
import logging
import math
import os
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

log = logging.getLogger("polybot")
NY = ZoneInfo("America/New_York")

# The serious ones: about a hundred of the largest, most traded US stocks.
UNIVERSE = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM", "V", "MA", "UNH", "XOM", "LLY",
    "JNJ", "PG", "HD", "COST", "ABBV", "MRK", "PEP", "KO", "WMT", "BAC", "CVX", "ADBE", "CRM", "NFLX", "AMD", "INTC",
    "ORCL", "CSCO", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC", "ADI", "PANW", "CRWD", "SNOW", "PLTR", "UBER", "ABNB",
    "SHOP", "COIN", "MSTR", "SMCI", "ARM", "ASML", "TSM", "DIS", "NKE", "MCD", "SBUX", "BA", "CAT", "DE", "GE", "HON",
    "LMT", "RTX", "UPS", "FDX", "GS", "MS", "C", "WFC", "AXP", "SCHW", "BLK", "PYPL", "INTU", "NOW", "ISRG", "TMO",
    "DHR", "ABT", "PFE", "AMGN", "GILD", "VRTX", "REGN", "BKNG", "MAR", "CMG", "LULU", "TGT", "LOW", "DELL", "ANET",
    "MRVL", "ON", "HOOD", "RBLX", "DDOG", "NET", "ZS",
)
BENCH = ("SPY", "QQQ", "DIA", "IWM", "^VIX", "BTC-USD", "ETH-USD")
INDEX_NAMES = {"SPY": "S&P 500 (SPY)", "QQQ": "נאסד\"ק 100 (QQQ)", "DIA": "דאו ג'ונס (DIA)", "IWM": "ראסל 2000 (IWM)",
               "^VIX": "מדד הפחד (VIX)", "BTC-USD": "ביטקוין", "ETH-USD": "את'ריום"}
HISTORY = {"SPY": "10y", "^VIX": "10y"}  # the S&P's and fear's longer history: the base rates of what came next
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval=1d"
REFRESH_HOURS = 6


def completed(day, symbol, now=None):
    """A day's bar is final only once its session closed: a stock's at 16:00 New York (a few minutes' margin
    for the last prints), a coin's (it trades all day) at midnight UTC. Before that it is a bar still forming:
    its close is the price of the moment and its volume part of the day's."""
    now = now or time.time()
    if symbol.endswith("-USD"):
        return day < dt.datetime.fromtimestamp(now, dt.timezone.utc).date().isoformat()
    t = dt.datetime.fromtimestamp(now, NY)
    today = t.date().isoformat()
    return day < today or (day == today and (t.hour, t.minute) >= (16, 20))


def fetch_yahoo(symbol, rng="2y"):
    """[(day, open, high, low, close, volume)] of a symbol's daily bars, oldest first (completed sessions only)."""
    url = YAHOO.format(urllib.parse.quote(symbol), rng)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (polybot research)", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    res = (data.get("chart") or {}).get("result") or []
    if not res:
        raise ValueError(f"no data: {(data.get('chart') or {}).get('error')}")
    res = res[0]
    off = (res.get("meta") or {}).get("gmtoffset") or 0
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    out = []
    for i, ts in enumerate(res.get("timestamp") or []):
        o, h, lo, c, v = (q.get(k, [None])[i] if i < len(q.get(k, [])) else None for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, lo, c):
            continue
        day = dt.datetime.fromtimestamp(ts + off, dt.timezone.utc).date().isoformat()
        if completed(day, symbol):
            out.append((day, float(o), float(h), float(lo), float(c), float(v or 0)))
    return out


# ---------- indicators ----------
def sma(xs, n):
    out, s = [None] * len(xs), 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= n:
            s -= xs[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def wilder(xs, n):
    """Wilder's smoothing (as ATR and RSI use)."""
    out, avg = [None] * len(xs), None
    for i, x in enumerate(xs):
        if x is None:
            continue
        avg = x if avg is None else (avg * (n - 1) + x) / n
        out[i] = avg if i >= n - 1 else None
    return out


def atr(h, lo, c, n=14):
    tr = [h[0] - lo[0]] + [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))]
    return wilder(tr, n)


def rsi(c, n):
    up = [None] + [max(0.0, c[i] - c[i - 1]) for i in range(1, len(c))]
    dn = [None] + [max(0.0, c[i - 1] - c[i]) for i in range(1, len(c))]
    au, ad = wilder(up, n), wilder(dn, n)
    return [None if a is None or b is None else (100.0 if b == 0 else 100 - 100 / (1 + a / b)) for a, b in zip(au, ad)]


def prepare(bars):
    """A symbol's bars with the indicators the strategies read."""
    if not bars:
        return None
    day, o, h, lo, c, v = (list(x) for x in zip(*bars))
    p = {"day": day, "o": o, "h": h, "l": lo, "c": c, "v": v, "sma20": sma(c, 20), "sma50": sma(c, 50),
         "sma200": sma(c, 200), "atr": atr(h, lo, c, 14), "rsi2": rsi(c, 2), "vol20": sma(v, 20)}
    p["hh20"] = [None if i < 20 else max(h[i - 20:i]) for i in range(len(c))]  # the 20 days before
    p["index"] = {d: i for i, d in enumerate(day)}
    return p


# ---------- the market's picture ----------
def trend(p, i, short, long_):
    """'up', 'down' or 'side': the close against two averages and the longer one's slope."""
    c, a, b = p["c"][i], p[short][i], p[long_][i]
    if a is None or b is None:
        return None
    back = p[long_][i - 20] if i >= 20 else None
    rising = back is not None and b > back
    if c > b and a > b and rising:
        return "up"
    if c < b and a < b and back is not None and b < back:
        return "down"
    return "side"


def picture(prepared, day=None):
    """The market now (or on `day`): primary trend (S&P 500 against its 200-day average), secondary (50 and
    20 days), fear (VIX), breadth (the share of the stocks above their 50-day average), and a verdict."""
    spy, vix = prepared.get("SPY"), prepared.get("^VIX")
    if not spy:
        return None
    i = spy["index"].get(day, len(spy["c"]) - 1) if day else len(spy["c"]) - 1
    d = spy["day"][i]
    above = total = 0
    for s in UNIVERSE:
        p = prepared.get(s)
        j = p and p["index"].get(d)
        if j is None or p["sma50"][j] is None:
            continue
        total += 1
        above += p["c"][j] > p["sma50"][j]
    breadth = above / total if total else None
    v = None
    if vix and d in vix["index"]:
        k = vix["index"][d]
        v = {"now": vix["c"][k], "avg20": vix["sma20"][k]}
    primary, secondary = trend(spy, i, "sma50", "sma200"), trend(spy, i, "sma20", "sma50")
    fear = None if not v else "calm" if v["now"] < 15 else "normal" if v["now"] < 25 else "fear"
    score = (primary == "up") + (secondary == "up") + (breadth is not None and breadth > 0.5) + (fear in ("calm", "normal"))
    score -= (primary == "down") + (secondary == "down") + (fear == "fear")
    verdict = "risk_on" if score >= 3 else "risk_off" if score <= 0 else "mixed"
    qqq = prepared.get("QQQ")
    q = qqq and qqq["index"].get(d)
    return {"day": d, "spy": spy["c"][i], "spy_sma50": spy["sma50"][i], "spy_sma200": spy["sma200"][i],
            "spy_5d": spy["c"][i] / spy["c"][i - 5] - 1 if i >= 5 else None,
            "spy_20d": spy["c"][i] / spy["c"][i - 20] - 1 if i >= 20 else None,
            "qqq_20d": qqq["c"][q] / qqq["c"][q - 20] - 1 if q is not None and q >= 20 else None,
            "primary": primary, "secondary": secondary, "vix": v, "fear": fear,
            "breadth": breadth, "stocks": total, "score": score, "verdict": verdict}


def indices(prepared):
    """The leading indices (and Bitcoin and Ether): the last close and its moves over 1, 5 and 20 days."""
    out = []
    for sym in BENCH:
        p = prepared.get(sym)
        if not p or len(p["c"]) < 21:
            continue
        c = p["c"]
        out.append({"symbol": sym, "name": INDEX_NAMES.get(sym, sym), "day": p["day"][-1], "close": c[-1],
                    "d1": c[-1] / c[-2] - 1, "d5": c[-1] / c[-6] - 1, "d20": c[-1] / c[-21] - 1,
                    "above50": p["sma50"][-1] is not None and c[-1] > p["sma50"][-1],
                    "above200": p["sma200"][-1] is not None and c[-1] > p["sma200"][-1]})
    return out


def _pivots(p, i, back, side, k=5):
    """Swing points in the last `back` days: lows (side 'l') or highs ('h') with k days on each side above / below."""
    xs = p[side]
    out = []
    for j in range(max(k, i - back), i - k + 1):
        win = xs[j - k:j + k + 1]
        if (side == "l" and xs[j] == min(win)) or (side == "h" and xs[j] == max(win)):
            out.append((p["day"][j], xs[j]))
    return out


def technical(p):
    """The S&P's technical picture now: momentum, the distance from its averages, the year's range, the
    nearest supports and resistances (recent swing points) and the Fibonacci levels of the last big swing."""
    i = len(p["c"]) - 1
    c = p["c"][i]
    r14 = rsi(p["c"], 14)[i]
    lows, highs = _pivots(p, i, 120, "l"), _pivots(p, i, 120, "h")
    support = sorted({round(x, 2) for _, x in lows if x < c}, reverse=True)[:2]
    resistance = sorted({round(x, 2) for _, x in highs if x > c})[:2]
    year = slice(max(0, i - 251), i + 1)
    hi_j = max(range(max(0, i - 120), i + 1), key=lambda j: p["h"][j])
    lo_j = min(range(max(0, i - 120), i + 1), key=lambda j: p["l"][j])
    H, L = p["h"][hi_j], p["l"][lo_j]
    rising = lo_j < hi_j  # (the swing went up: retracements are measured down from the high)
    fib = {f"{int(f * 1000) / 10:g}": round(H - f * (H - L) if rising else L + f * (H - L), 2) for f in (0.236, 0.382, 0.5, 0.618, 0.786)}
    return {"close": c, "rsi14": r14, "atr": p["atr"][i],
            "vs50": c / p["sma50"][i] - 1 if p["sma50"][i] else None, "vs200": c / p["sma200"][i] - 1 if p["sma200"][i] else None,
            "high52": max(p["h"][year]), "low52": min(p["l"][year]), "support": support, "resistance": resistance,
            "fib": {"from": round(L if rising else H, 2), "to": round(H if rising else L, 2), "rising": rising, "levels": fib}}


def outlook(spy, vix):
    """What came next, in the S&P's own history, on the days like today (above or below its 200-day average,
    momentum stretched or not, fear high or low): how often it was higher 5 and 20 trading days later, and by
    how much. Base rates, not a prophecy."""
    c = spy["c"]
    if len(c) < 260:
        return None
    r14 = rsi(c, 14)
    vix_at = {d: vix["c"][k] for k, d in enumerate(vix["day"])} if vix else {}

    def state(i):
        if spy["sma200"][i] is None or r14[i] is None:
            return None
        v = vix_at.get(spy["day"][i])
        return (c[i] > spy["sma200"][i], 0 if r14[i] < 35 else 2 if r14[i] > 65 else 1,
                None if v is None else (0 if v < 15 else 1 if v < 25 else 2))
    now = state(len(c) - 1)
    if now is None:
        return None
    out = {"state": {"above200": now[0], "momentum": ["low", "mid", "high"][now[1]],
                     "fear": None if now[2] is None else ["calm", "normal", "fear"][now[2]]}}
    for h in (5, 20):
        rets = []
        for i in range(200, len(c) - h):
            s = state(i)
            if s and s[0] == now[0] and s[1] == now[1] and (now[2] is None or s[2] is None or s[2] == now[2]):
                rets.append(c[i + h] / c[i] - 1)
        rets.sort()
        out[f"d{h}"] = {"n": len(rets), "up": sum(1 for x in rets if x > 0) / len(rets) if rets else None,
                        "median": rets[len(rets) // 2] if rets else None,
                        "low": rets[int(len(rets) * 0.1)] if rets else None, "high": rets[int(len(rets) * 0.9)] if rets else None}
    return out


def chart(p, days=180):
    """The last `days` daily candles and the 50 and 200-day averages, for a chart."""
    k = max(0, len(p["c"]) - days)
    return {"day": p["day"][k:], "o": p["o"][k:], "h": p["h"][k:], "l": p["l"][k:], "c": p["c"][k:],
            "sma50": p["sma50"][k:], "sma200": p["sma200"][k:]}


# ---------- the strategies (swing) ----------
# Each: the signal at a day's close (the stop under the next open's entry), or None. Named after the
# traders whose published methods they follow.
STRATEGIES = {
    "pullback": {"name": "חזרה לממוצע במגמה עולה", "by": "לארי קונורס",
                 "rule": "מניה מעל ממוצע 200 יום ירדה חזק כמה ימים (RSI של יומיים מתחת לסף): קונים בפתיחה של מחר",
                 "params": {"rsi_max": 10}},
    "breakout": {"name": "פריצה עם מחזור", "by": "מארק מינרביני · קריסטיאן קולמאגי",
                 "rule": "מניה במגמה עולה (מעל ממוצע 50 שמעל 200) נסגרת מעל השיא של 20 הימים הקודמים, במחזור גבוה מהרגיל",
                 "params": {"vol_mult": 1.5}},
    "fib": {"name": "תיקון פיבונאצ'י במגמה", "by": "תצורות פיבונאצ'י (הפאנל)",
            "rule": "מניה במגמה עולה מתקנת לאזור 38.2%–61.8% מהמהלך האחרון ונסגרת בירוק מעל אתמול: קונים מחר; הסטופ מתחת ל-78.6%",
            "params": {"swing_days": 30}},
    "gap": {"name": "פער פתיחה עם מחזור (חדשות)", "by": "קריסטיאן קולמאגי (Episodic Pivot)",
            "rule": "מניה פותחת בפער של 4% ומעלה במחזור כפול מהרגיל ונסגרת מעל הפתיחה: קונים מחר; הסטופ מתחת לשפל של יום הפער",
            "params": {"gap": 0.04, "vol_mult": 2.0}},
}

DEFAULTS = {
    "account": 100_000.0, "risk_pct": 1.0, "max_position": 10_000.0, "max_open": 10,
    "stop_atr": 2.0,  # the stop's distance in daily ranges (ATR): not too tight
    "rr": 3.0,  # the take-profit at this many times the risk (0: none, only the trailing stop)
    "breakeven_r": 1.0,  # once the trade made this many times its risk, the stop moves to the entry (0: never)
    "trail_r": 1.5, "trail_atr": 2.5,  # from this profit on, the stop trails the best close by this many ATRs
    "max_hold": 15,  # trading days at most
    "regime": "primary",  # buy only when the market allows: "off", "primary" (S&P above its 200-day), "both" (and its 50-day)
    "commission": 0.005, "min_commission": 1.0, "slippage": 0.0005,
}


def signal(kind, p, i, P):
    """The stop for an entry at the next open, if the strategy signals at day i's close."""
    c, a = p["c"][i], p["atr"][i]
    if a is None or p["sma200"][i] is None or i < 1:
        return None
    if kind == "pullback":
        r = p["rsi2"][i]
        if c > p["sma200"][i] and r is not None and r < P.get("rsi_max", 10):
            return c - P["stop_atr"] * a
    elif kind == "breakout":
        hh, v20 = p["hh20"][i], p["vol20"][i]
        if (hh is not None and v20 and c > hh and p["v"][i] > P.get("vol_mult", 1.5) * v20
                and p["sma50"][i] is not None and c > p["sma50"][i] > p["sma200"][i]):
            return c - P["stop_atr"] * a
    elif kind == "fib":
        n = int(P.get("swing_days", 30))
        if i < 2 * n or p["sma50"][i] is None or not (c > p["sma200"][i] and p["sma50"][i] > p["sma200"][i]):
            return None
        hi_j = max(range(i - n, i - 2), key=lambda j: p["h"][j])
        H = p["h"][hi_j]
        L = min(p["l"][hi_j - n:hi_j + 1])
        span = H - L
        if span < 3 * a:
            return None
        top, bottom = H - 0.382 * span, H - 0.618 * span
        if bottom <= p["l"][i] <= top and c > p["o"][i] and c > p["c"][i - 1]:
            return H - 0.786 * span
    elif kind == "gap":
        v20 = p["vol20"][i]
        if (v20 and p["o"][i] >= p["c"][i - 1] * (1 + P.get("gap", 0.04)) and p["v"][i] >= P.get("vol_mult", 2.0) * v20
                and c > p["o"][i]):
            return p["l"][i] - 0.2 * a
    return None


def _score(kind, p, i):
    """Among the day's signals, which first when the slots are few: the strongest move in volume."""
    v20 = p["vol20"][i]
    return (p["v"][i] / v20) if v20 else 0.0


def plan_for(kind, params=None):
    """A strategy's full rules: the defaults, its own, and any given."""
    return {**DEFAULTS, **STRATEGIES[kind]["params"], **(params or {})}


def new_book(account):
    return {"cash": account, "open": [], "pending": [], "trades": [], "curve": [], "last": None}


def _fee(P, shares):
    return max(P["min_commission"], P["commission"] * shares)


def _close(book, t, d, price, why, P):
    price *= 1 - P["slippage"]
    book["cash"] += t["shares"] * price - _fee(P, t["shares"])
    pnl = (price - t["entry"]) * t["shares"] - t["fees"] - _fee(P, t["shares"])
    book["trades"].append({"id": t.get("id") or f"{t['strategy']}:{t['symbol']}:{t['day']}", "strategy": t["strategy"],
                           "symbol": t["symbol"], "in": t["day"], "out": d,
                           "entry": round(t["entry"], 2), "exit": round(price, 2), "stop0": round(t["stop0"], 2),
                           "target": round(t["target"], 2) if t["target"] else None, "shares": t["shares"],
                           "pnl": round(pnl, 2), "r": round(pnl / t["risk_usd"], 2) if t["risk_usd"] else None,
                           "days": t["held"], "why": why})


def market_ok(spy, d, regime):
    si = spy["index"].get(d)
    if si is None:
        return False
    if regime not in ("primary", "both"):
        return True
    s200 = spy["sma200"][si]
    ok = s200 is not None and spy["c"][si] > s200
    if regime == "both":
        ok = ok and spy["sma50"][si] is not None and spy["c"][si] > spy["sma50"][si]
    return ok


def step(book, prepared, d, plans, port, signals=True, symbols=UNIVERSE):
    """One trading day for a portfolio: yesterday's signals enter at today's open, the open trades meet
    today's range, today's signals wait for tomorrow's open (when the market allows). plans: {strategy:
    its rules}; port: the account's rules (risk a trade, the largest position, how many open, the filter)."""
    # 1. Yesterday's signals, at today's open.
    for kind, sym, stop in book["pending"]:
        p = prepared.get(sym)
        i = p and p["index"].get(d)
        if i is None:
            continue
        P = plans[kind]
        entry = p["o"][i] * (1 + P["slippage"])
        if stop >= entry or any(t["symbol"] == sym for t in book["open"]):
            continue  # (it opened under the stop, or the stock is held already)
        per = entry - stop
        shares = int(min(port["account"] * port["risk_pct"] / 100 / per, port["max_position"] / entry))
        if shares < 1 or shares * entry + _fee(P, shares) > book["cash"] or len(book["open"]) >= port["max_open"]:
            continue
        book["cash"] -= shares * entry + _fee(P, shares)
        book["open"].append({"id": f"{kind}:{sym}:{d}", "strategy": kind, "symbol": sym, "day": d, "entry": entry, "stop": stop, "stop0": stop,
                             "shares": shares, "target": entry + P["rr"] * per if P["rr"] else None, "risk": per,
                             "risk_usd": per * shares, "fees": _fee(P, shares), "best": entry, "held": 0})
    book["pending"] = []
    # 2. The open trades on today's range.
    still = []
    for t in book["open"]:
        p, P = prepared.get(t["symbol"]), plans.get(t["strategy"]) or plan_for(t["strategy"])
        i = p and p["index"].get(d)
        if i is None:
            still.append(t)
            continue
        o, h, lo, c = p["o"][i], p["h"][i], p["l"][i], p["c"][i]
        t["held"] += 1
        if t["day"] != d and o <= t["stop"]:
            _close(book, t, d, o, "stop" if t["stop"] < t["entry"] else "trail" if t["stop"] > t["entry"] else "breakeven", P)
        elif t["target"] and t["day"] != d and o >= t["target"]:
            _close(book, t, d, o, "target", P)
        elif lo <= t["stop"]:
            _close(book, t, d, t["stop"], "stop" if t["stop"] < t["entry"] else "trail" if t["stop"] > t["entry"] else "breakeven", P)
        elif t["target"] and h >= t["target"]:
            _close(book, t, d, t["target"], "target", P)
        elif t["held"] >= P["max_hold"]:
            _close(book, t, d, c, "time", P)
        else:
            # After the day: a trade that made breakeven_r risks nothing any more; past trail_r the stop trails.
            t["best"] = max(t["best"], c)
            if P["breakeven_r"] and h >= t["entry"] + P["breakeven_r"] * t["risk"]:
                t["stop"] = max(t["stop"], t["entry"])
            a = p["atr"][i]
            if P["trail_atr"] and a and h >= t["entry"] + P["trail_r"] * t["risk"]:
                t["stop"] = max(t["stop"], t["best"] - P["trail_atr"] * a)
            still.append(t)
    book["open"] = still
    # 3. Today's signals, for tomorrow's open.
    free = int(port["max_open"]) - len(book["open"])
    if signals and free > 0 and market_ok(prepared["SPY"], d, port["regime"]):
        held = {t["symbol"] for t in book["open"]}
        found = []
        for kind, P in plans.items():
            for sym in symbols:
                p = prepared.get(sym)
                i = p and p["index"].get(d)
                if i is None or sym in held:
                    continue
                stop = signal(kind, p, i, P)
                if stop is not None:
                    found.append((_score(kind, p, i), kind, sym, stop))
        found.sort(reverse=True)
        seen = set()
        for _, kind, sym, stop in found:
            if sym not in seen and len(book["pending"]) < free:
                seen.add(sym)
                book["pending"].append((kind, sym, stop))
    # 4. The account at today's close.
    value = book["cash"]
    for t in book["open"]:
        p = prepared[t["symbol"]]
        i = p["index"].get(d)
        value += t["shares"] * (p["c"][i] if i is not None else t["entry"])
    book["curve"].append((d, round(value, 2)))
    book["last"] = d


def portfolio(prepared, plans, port, symbols=UNIVERSE):
    """Several strategies as one account over the whole history (each trade managed by its own rules)."""
    spy = prepared.get("SPY")
    if not spy:
        return None
    days = spy["day"]
    book = new_book(port["account"])
    for di, d in enumerate(days):
        step(book, prepared, d, plans, port, signals=di < len(days) - 1, symbols=symbols)
    for t in book["open"]:
        p = prepared[t["symbol"]]
        _close(book, t, days[-1], p["c"][-1], "open", plans.get(t["strategy"]) or plan_for(t["strategy"]))
    trades = sorted(book["trades"], key=lambda t: t["in"])
    cut = days[int(len(days) * 0.7)] if days else None
    return {"params": port, "plans": plans, "trades": trades, "curve": book["curve"], "stats": stats(trades, book["curve"], port),
            "split": {"cut": cut, "old": stats([t for t in trades if t["in"] < cut], None, port),
                      "new": stats([t for t in trades if t["in"] >= cut], None, port)}}


def backtest(prepared, kind, params=None, symbols=UNIVERSE):
    """One strategy as an account, with the owner's money rules."""
    P = plan_for(kind, params)
    return portfolio(prepared, {kind: P}, P, symbols=symbols)


def stats(trades, curve, P):
    done = [t for t in trades if t["why"] != "open"]
    n = len(done)
    wins = [t["pnl"] for t in done if t["pnl"] > 0]
    losses = [t["pnl"] for t in done if t["pnl"] <= 0]
    out = {"trades": n, "wins": len(wins), "win_rate": len(wins) / n if n else None,
           "avg_win": sum(wins) / len(wins) if wins else None, "avg_loss": sum(losses) / len(losses) if losses else None,
           "expectancy": sum(t["pnl"] for t in done) / n if n else None,
           "avg_r": sum(t["r"] for t in done if t["r"] is not None) / n if n else None,
           "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None,
           "pnl": sum(t["pnl"] for t in done), "avg_days": sum(t["days"] for t in done) / n if n else None,
           # (out at the entry once a winner turned: neither a win nor a real loss, only the commissions)
           "scratches": sum(1 for t in done if t["why"] == "breakeven"),
           "open": sum(1 for t in trades if t["why"] == "open"), "open_pnl": sum(t["pnl"] for t in trades if t["why"] == "open")}
    out["pnl_pct"] = out["pnl"] / P["account"] if P and P.get("account") else None
    if curve:
        peak, dd = P["account"], 0.0
        for _, v in curve:
            peak = max(peak, v)
            dd = max(dd, (peak - v) / peak)
        out.update(max_drawdown=dd, final=curve[-1][1], ret=curve[-1][1] / P["account"] - 1)
        by = {}
        for t in done:
            y = by.setdefault(t["in"][:4], {"trades": 0, "pnl": 0.0})
            y["trades"] += 1
            y["pnl"] += t["pnl"]
        out["by_year"] = by
    return out


INTEGERS = ("max_open", "max_hold", "swing_days")  # (counts: whole numbers, whatever the page sends)
STAGES = ("יש נתונים", "רווחית בבקטסט", "מחזיקה על נתונים חדשים", "סימולציה במערכת", "מחוברת לברוקר (Paper)")


def stage(result, data_ok):
    """How far a strategy is from real money: 0 to len(STAGES) steps done, the next step's name."""
    done = 0
    if data_ok:
        done = 1
        s, new = result["stats"], result["split"]["new"]
        if s["trades"] >= 30 and (s["expectancy"] or 0) > 0 and (s["profit_factor"] or 0) > 1.3:
            done = 2
            if new["trades"] >= 15 and (new["expectancy"] or 0) > 0 and (new["profit_factor"] or 0) > 1.1:
                done = 3  # (the simulation in the system and the broker come in the next phases)
    return {"done": done, "of": len(STAGES), "next": STAGES[done] if done < len(STAGES) else None}


class EquityStore:
    """The stocks area's own database file, next to Polymarket's: nothing done here (a new table, a
    reload, a bug) ever writes to Polymarket's data."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.db.execute("CREATE TABLE IF NOT EXISTS eq_daily (symbol TEXT, day TEXT, o REAL, h REAL, l REAL, c REAL,"
                        " v REAL, PRIMARY KEY (symbol, day))")
        self.db.commit()

    def _x(self, sql, args=()):
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
            self.db.commit()
            return rows

    def add_bars(self, rows):
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO eq_daily VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            self.db.commit()

    def close(self):
        self.db.close()


def move_out_of(main, eq):
    """Once: the bars the first version kept in Polymarket's database move to the stocks area's own."""
    if not main._x("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'eq_daily'"):
        return 0
    rows = main._x("SELECT symbol, day, o, h, l, c, v FROM eq_daily")
    if rows:
        eq.add_bars([tuple(r) for r in rows])
    main._x("DROP TABLE eq_daily")
    log.warning("Stocks: %d daily bars moved to their own database (%s)", len(rows), eq.path)
    return len(rows)


class Equities:
    """The data (refreshed after each close), the prepared series and each strategy's standing."""

    def __init__(self, store, fetch=fetch_yahoo):
        self.store, self.fetch = store, fetch
        self.lock = threading.Lock()
        self.prepared, self.version = {}, 0
        self.summary = None
        self.status = {"symbols": 0, "of": len(UNIVERSE) + len(BENCH), "last_day": None, "refreshed": None,
                       "errors": [], "running": False}
        self._load()

    def _load(self):
        rows = self.store._x("SELECT symbol, day, o, h, l, c, v FROM eq_daily ORDER BY symbol, day")
        by = {}
        for sym, *bar in rows:
            by.setdefault(sym, []).append(tuple(bar))
        with self.lock:
            self.prepared = {s: prepare(b) for s, b in by.items() if len(b) >= 30}
            self.version += 1
            self.summary = None
        self.status["symbols"] = len(self.prepared)
        spy = self.prepared.get("SPY")
        self.status["last_day"] = spy["day"][-1] if spy else None

    def refresh(self):
        """Every symbol's bars: two years the first time, the last month after that."""
        self.status["running"] = True
        errors = []
        try:
            have = dict(self.store._x("SELECT symbol, MAX(day) FROM eq_daily GROUP BY symbol"))
            # (bars of a session still forming, stored before that was checked, are dropped)
            for sym in BENCH + UNIVERSE:
                for (day,) in self.store._x("SELECT day FROM eq_daily WHERE symbol = ? ORDER BY day DESC LIMIT 2", (sym,)):
                    if not completed(day, sym):
                        self.store._x("DELETE FROM eq_daily WHERE symbol = ? AND day = ?", (sym, day))
            for sym in BENCH + UNIVERSE:
                try:
                    bars = self.fetch(sym, "1mo" if sym in have else HISTORY.get(sym, "2y"))
                    if sym in have and bars and self._rewritten(sym, bars):
                        # Yahoo adjusts a stock's whole history after a split (or a correction): the stored
                        # history no longer matches, so all of it is fetched again.
                        full = self.fetch(sym, HISTORY.get(sym, "2y"))
                        if full:
                            self.store._x("DELETE FROM eq_daily WHERE symbol = ?", (sym,))
                            bars = full
                            self.status.setdefault("rewritten", []).append(sym)
                            log.warning("Stocks: %s's history changed at the source (a split?): fetched again", sym)
                except Exception as e:
                    errors.append(f"{sym}: {type(e).__name__}: {e}"[:160])
                    continue
                if bars:
                    self.store.add_bars([(sym, *b) for b in bars])
                time.sleep(0.25)  # (polite to the source)
        finally:
            self.status.update(running=False, errors=errors[-20:], refreshed=time.time())
        self._load()
        log.warning("Stocks: %d symbols, last day %s, %d errors", self.status["symbols"], self.status["last_day"], len(errors))

    def _rewritten(self, sym, bars, tolerance=0.01):
        """Whether the source's bars disagree with the stored ones on the days both have (a close more than
        1% apart): the stored history is stale."""
        stored = dict(self.store._x("SELECT day, c FROM eq_daily WHERE symbol = ? AND day >= ?", (sym, bars[0][0])))
        return any(day in stored and stored[day] and abs(c / stored[day] - 1) > tolerance for day, _, _, _, c, _ in bars)

    def due(self, now=None):
        last = self.status["refreshed"]
        return last is None or (now or time.time()) - last >= REFRESH_HOURS * 3600

    def run(self, stop):
        while not stop.is_set():
            try:
                if self.due():
                    self.refresh()
            except Exception as e:
                log.warning("Stocks refresh: %s", e)
            stop.wait(600)

    def start(self, stop):
        threading.Thread(target=self.run, args=(stop,), daemon=True, name="equities").start()

    def data_ok(self):
        spy = self.prepared.get("SPY")
        return bool(spy and len(spy["day"]) >= 250 and self.status["symbols"] >= 0.8 * (len(UNIVERSE) + len(BENCH)))

    def backtest(self, kind, params=None):
        if kind not in STRATEGIES:
            raise ValueError("bad request")
        clean = {}
        for k, v in (params or {}).items():
            if k in DEFAULTS or k in STRATEGIES[kind]["params"]:
                clean[k] = v if isinstance(DEFAULTS.get(k), str) else int(float(v)) if k in INTEGERS else float(v)
        if "regime" in clean and clean["regime"] not in ("off", "primary", "both"):
            raise ValueError("bad request")
        if not 0.1 <= clean.get("risk_pct", 1.0) <= 5:
            raise ValueError("סיכון לטרייד: בין 0.1% ל-5%")
        with self.lock:
            prepared = self.prepared
        out = backtest(prepared, kind, clean)
        if out is None:
            raise ValueError("עוד אין נתונים: הטעינה הראשונה לוקחת כמה דקות")
        out["stage"] = stage(out, self.data_ok())
        return out

    def view(self):
        """The area's page: the data, the market's picture, each strategy at its defaults."""
        with self.lock:
            prepared, version = self.prepared, self.version
        if self.summary is None or self.summary[0] != version:
            strategies = []
            for kind, s in STRATEGIES.items():
                r = backtest(prepared, kind) if prepared.get("SPY") else None
                strategies.append({"key": kind, **{k: s[k] for k in ("name", "by", "rule", "params")},
                                   "stats": r and {k: v for k, v in r["stats"].items() if k != "by_year"},
                                   "new": r and r["split"]["new"], "stage": stage(r, self.data_ok()) if r else None})
            self.summary = (version, {"picture": picture(prepared), "strategies": strategies})
        return dict(self.summary[1], status=self.status, defaults=DEFAULTS, stages=STAGES,
                    universe=len(UNIVERSE))
