"""The owner's trade ticket (10.11.2026: "a big, standing-out 'add a trade' button: a popup with a search for the stock,
autocomplete and its logo; then a full, exact set of tools to enter it: a big chart with lines, Fibonacci, a ruler and
every time frame, the volume under it; the news, four key figures of the company, the analysts' counts, and one final
gauge: buy / don't / your call, from a deep pass over the news, the technical picture, the fundamentals, the analysts and
the market; a recommended stop and take-profit; then the buy / sell, the quantity in dollars, a trailing stop of
1/2/3/5%, at the market, under or over the price; and all of it on the chart, dragged there too").

The data: Yahoo's search (the symbols, as they are typed, and the stock's news) and its chart (every interval with its
volume), read by the server (the browser can't: their pages refuse other sites). The technical picture is computed here,
from the daily candles; the market's from Atlas' own data (the S&P against its averages, the VIX, and what came next on
days like today). The rest (the news' tone, the company's figures, the analysts) is Claude's, with the web, each with
its source, kept for half an hour a stock (and counted in the research's cost)."""
import json
import logging
import math
import re
import threading
import time
import urllib.parse
import urllib.request

log = logging.getLogger("polybot")

UA = {"User-Agent": "Mozilla/5.0 (polybot ticket)"}
# Each time frame: Yahoo's interval and how far back (its own limits: a minute's bars for the last week, five minutes'
# and half an hour's for two months); "1y" is built here from the monthly bars.
FRAMES = {"1m": ("1m", "5d"), "5m": ("5m", "1mo"), "30m": ("30m", "2mo"), "1h": ("60m", "6mo"), "1d": ("1d", "2y"),
          "1wk": ("1wk", "10y"), "1mo": ("1mo", "max"), "1y": ("1mo", "max")}
SEARCH_KEEP = 3600  # seconds a search answer is kept
CHART_KEEP = {"1m": 20, "5m": 60, "30m": 120, "1h": 300, "1d": 600, "1wk": 1800, "1mo": 3600, "1y": 3600}
ANALYSIS_KEEP = 1800  # seconds a stock's analysis is kept (asked again: the kept one, unless "again")
MODEL = "claude-opus-5-5"
TRAILS = (1.0, 2.0, 3.0, 5.0)
SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

_cache, _lock = {}, threading.Lock()


def _get(url, timeout=12, opener=None):
    req = urllib.request.Request(url, headers=UA)
    with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
        return json.loads(r.read())


def _kept(key, keep, make):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < keep:
            return hit[1]
    val = make()
    with _lock:
        _cache[key] = (now, val)
        if len(_cache) > 500:
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:100]:
                _cache.pop(k, None)
    return val


def clean_symbol(s):
    s = (s or "").strip().upper()
    if not SYMBOL.match(s):
        raise ValueError("סימבול לא תקין")
    return s


# ---------- the search ----------
def parse_search(data):
    """Yahoo's search answer -> [{"symbol", "name", "exchange", "type"}]: US stocks and ETFs only."""
    out = []
    for q in (data or {}).get("quotes") or []:
        sym, kind = q.get("symbol") or "", q.get("quoteType") or ""
        if kind not in ("EQUITY", "ETF") or "." in sym and not re.match(r"^[A-Z]+\.[A-Z]$", sym) or not SYMBOL.match(sym):
            continue
        if q.get("exchange") and q["exchange"] not in ("NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NAS", "NYS", "PNK", "OQB", "OQX"):
            continue  # (Tel Aviv, London, Toronto...: the account trades the US ones)
        out.append({"symbol": sym, "name": q.get("longname") or q.get("shortname") or sym,
                    "exchange": q.get("exchDisp") or q.get("exchange") or "", "type": "ETF" if kind == "ETF" else "מניה"})
    return out


def search(q, fetch=_get):
    q = (q or "").strip()
    if not q or len(q) > 40:
        return []
    url = ("https://query2.finance.yahoo.com/v1/finance/search?" +
           urllib.parse.urlencode({"q": q, "quotesCount": 10, "newsCount": 0, "listsCount": 0, "enableFuzzyQuery": "false"}))
    return _kept(("s", q.upper()), SEARCH_KEEP, lambda: parse_search(fetch(url)))[:8]


# ---------- the chart ----------
def parse_chart(data):
    """Yahoo's chart answer -> {"bars": [[ts, o, h, l, c, v]], "price", "prev_close", "currency", "name", "exchange"}."""
    res = ((data or {}).get("chart") or {}).get("result") or []
    if not res:
        err = ((data or {}).get("chart") or {}).get("error") or {}
        raise ValueError(err.get("description") or "אין נתונים למניה הזאת")
    r = res[0]
    meta, ts = r.get("meta") or {}, r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    bars = []
    for i, t in enumerate(ts):
        o, h, lo, c, v = (q.get(k, [None] * len(ts))[i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, lo, c):
            continue
        bars.append([int(t), round(float(o), 4), round(float(h), 4), round(float(lo), 4), round(float(c), 4), float(v or 0)])
    return {"bars": bars, "price": meta.get("regularMarketPrice") or (bars[-1][4] if bars else None),
            "prev_close": meta.get("chartPreviousClose") or meta.get("previousClose"), "currency": meta.get("currency") or "USD",
            "name": meta.get("longName") or meta.get("shortName") or meta.get("symbol"), "exchange": meta.get("exchangeName")}


def yearly(bars):
    """Monthly bars into yearly ones (New York's calendar years)."""
    import datetime as dt
    out = {}
    for t, o, h, lo, c, v in bars:
        y = dt.datetime.fromtimestamp(t, dt.timezone.utc).year
        if y not in out:
            out[y] = [int(dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc).timestamp()), o, h, lo, c, v]
        else:
            b = out[y]
            b[2], b[3], b[4], b[5] = max(b[2], h), min(b[3], lo), c, b[5] + v
    return [out[y] for y in sorted(out)]


def chart(symbol, frame, fetch=_get):
    symbol = clean_symbol(symbol)
    if frame not in FRAMES:
        raise ValueError("טווח זמן לא מוכר")
    interval, rng = FRAMES[frame]
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?" +
           urllib.parse.urlencode({"interval": interval, "range": rng, "includePrePost": "false"}))

    def make():
        out = parse_chart(fetch(url))
        if frame == "1y":
            out["bars"] = yearly(out["bars"])
        out.update(symbol=symbol, frame=frame)
        return out
    return _kept(("c", symbol, frame), CHART_KEEP[frame], make)


# ---------- the news ----------
def parse_news(data):
    out = []
    for n in (data or {}).get("news") or []:
        if n.get("title") and n.get("link"):
            out.append({"title": n["title"], "url": n["link"], "source": n.get("publisher") or "",
                        "ts": n.get("providerPublishTime"), "tickers": n.get("relatedTickers") or []})
    return out


def news(symbol, fetch=_get):
    symbol = clean_symbol(symbol)
    url = ("https://query2.finance.yahoo.com/v1/finance/search?" +
           urllib.parse.urlencode({"q": symbol, "quotesCount": 0, "newsCount": 10}))
    return _kept(("n", symbol), 900, lambda: parse_news(fetch(url)))


# ---------- the technical picture (computed, from the daily candles) ----------
def _sma(xs, n):
    return sum(xs[-n:]) / n if len(xs) >= n else None


def _rsi(c, n=14):
    if len(c) <= n:
        return None
    gains = [max(0.0, c[i] - c[i - 1]) for i in range(1, len(c))]
    losses = [max(0.0, c[i - 1] - c[i]) for i in range(1, len(c))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    for g, lo in zip(gains[n:], losses[n:]):
        ag, al = (ag * (n - 1) + g) / n, (al * (n - 1) + lo) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def _atr(h, lo, c, n=14):
    trs = [h[i] - lo[i] if i == 0 else max(h[i], c[i - 1]) - min(lo[i], c[i - 1]) for i in range(len(c))]
    return sum(trs[-n:]) / min(n, len(trs)) if trs else None


def levels(h, lo, k=3, lookback=120):
    """The swing highs and lows of the last `lookback` days (a bar higher / lower than the k on each side)."""
    h, lo = h[-lookback:], lo[-lookback:]
    highs = [h[i] for i in range(k, len(h) - k) if h[i] == max(h[i - k:i + k + 1])]
    lows = [lo[i] for i in range(k, len(lo) - k) if lo[i] == min(lo[i - k:i + k + 1])]
    return highs, lows


def technical(bars, side="long"):
    """From the daily bars: the trend, the momentum, the key levels, where the price stands among them, a score 0-100
    for entering on `side`, and the stop and take-profit they suggest (the stop as a trailing distance too)."""
    if len(bars) < 30:
        return None
    o, h, lo, c = ([b[k] for b in bars] for k in (1, 2, 3, 4))
    price = c[-1]
    s20, s50, s200 = _sma(c, 20), _sma(c, 50), _sma(c, 200)
    r = _rsi(c)
    a = _atr(h, lo, c)
    highs, lows = levels(h, lo)
    res = min((x for x in highs if x > price * 1.003), default=max(h[-252:]))
    sup = max((x for x in lows if x < price * 0.997), default=min(lo[-60:]))
    hi52, lo52 = max(h[-252:]), min(lo[-252:])
    up = sum(1 for m in (s20, s50, s200) if m and price > m)
    trend = "עולה" if up == 3 or (up >= 2 and s50 and s200 and s50 > s200) else "יורדת" if up == 0 else "מעורבת"
    # where it stands: a breakout coming (just under the resistance), at a support, extended, or in the middle
    near_res, near_sup = (res - price) / price < 0.02, a and (price - sup) < a
    key = ("ממש מתחת להתנגדות: אם היא פורצת, זו נקודת כניסה" if near_res else
           "על תמיכה: נקודת כניסה עם סטופ קרוב" if near_sup else
           "מתוחה רחוק מעל הממוצעים: כניסה עכשיו היא מרדף" if s20 and price > s20 * 1.08 else "באמצע הטווח, בלי נקודת מפתח")
    long = side != "short"
    score = 50.0
    score += (up - 1.5) * 12 * (1 if long else -1)
    if r is not None:
        score += (8 if 45 <= r <= 65 else -10 if r > 75 else 6 if r < 30 else 0) * (1 if long else -1)
    if near_sup:
        score += 10 if long else -5
    if near_res:
        score += 4 if long else 8
    if s20 and price > s20 * 1.08:
        score += -12 if long else 6
    score = max(0.0, min(100.0, score))
    if long:
        stop = max(min(sup - 0.2 * a, price - a), price - 3 * a) if a else price * 0.97
        stop = min(stop, price * 0.995)
        risk = price - stop
        target = res if res - price >= 1.5 * risk else price + 2 * risk
    else:
        stop = min(max(res + 0.2 * a, price + a), price + 3 * a) if a else price * 1.03
        stop = max(stop, price * 1.005)
        risk = stop - price
        target = sup if price - sup >= 1.5 * risk else price - 2 * risk
    dist = abs(price - stop) / price * 100
    trail = min(TRAILS, key=lambda t: abs(t - dist))
    rd = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 2)
    return {"price": rd(price), "sma20": rd(s20), "sma50": rd(s50), "sma200": rd(s200), "rsi": rd(r), "atr": rd(a),
            "atr_pct": rd(a / price * 100) if a else None, "support": rd(sup), "resistance": rd(res), "high52": rd(hi52), "low52": rd(lo52),
            "trend": trend, "key": key, "score": round(score), "stop": rd(stop), "target": rd(target), "stop_pct": rd(dist),
            "trail": trail, "rr": rd(abs(target - price) / risk) if risk else None}


def market(atlas):
    """The market now (Atlas' own data): the S&P against its averages, its last 5 days, the VIX, what came next on days
    like today; a score 0-100 for buying."""
    from . import equity
    with atlas.eq.lock:
        spy, vix = atlas.eq.prepared.get("SPY"), atlas.eq.prepared.get("^VIX")
    if not spy:
        return None
    c = spy["c"]
    s50, s200 = spy["sma50"][-1], spy["sma200"][-1]
    v = vix["c"][-1] if vix else None
    d5 = c[-1] / c[-6] - 1 if len(c) > 6 else None
    try:
        ol = equity.outlook(spy, vix)
    except Exception:
        ol = None
    up20 = ((ol or {}).get("d20") or {}).get("up")
    score = 50.0 + (12 if s200 and c[-1] > s200 else -15) + (8 if s50 and c[-1] > s50 else -8)
    score += (6 if v is not None and v < 16 else -14 if v is not None and v > 25 else 0)
    score += (d5 or 0) * 300
    if up20 is not None:
        score += (up20 - 0.55) * 60
    score = max(0.0, min(100.0, score))
    return {"spy": round(c[-1], 2), "above50": bool(s50 and c[-1] > s50), "above200": bool(s200 and c[-1] > s200),
            "d5": d5, "vix": v and round(v, 2), "up20": up20, "median20": ((ol or {}).get("d20") or {}).get("median"),
            "score": round(score)}


# ---------- the deep pass (Claude, with the web) ----------
SYSTEM = ("You are the research desk of a US-stock trading system, writing for its owner, a trader. You search the web "
          "carefully and report only what you found a source for; never invent a number, a rating or a headline.")

PROMPT = """Today is {day}. The owner considers a {side} trade in {symbol} ({name}) now, at about ${price}.
The system's own technical reading of the daily chart: {tech}
The market now (the system's data): {market}

Search the web now and judge, for a {side} trade over the next days to weeks:
1. News of the last 7 days about {symbol}: what happened, and is it good or bad for the stock.
2. The fundamentals: exactly 4 key figures (pick the most telling: e.g. market cap, P/E or forward P/E, revenue growth,
   margins, debt, guidance), each with its value.
3. The analysts now: how many rate it strong buy, buy, hold, sell, strong sell (the latest consensus you can find, with
   its source), and the average price target.
4. Weigh it all with the technical reading and the market, and give a verdict.

Answer with ONLY one JSON object inside a ```json block:
{{"news": [{{"title": "...", "url": "...", "source": "...", "date": "YYYY-MM-DD", "tone": "good|bad|neutral"}}],
  "stats": [{{"label": "Hebrew label", "value": "the figure", "note": "a few Hebrew words: good / weak and why"}}],
  "analysts": {{"strong_buy": n, "buy": n, "hold": n, "sell": n, "strong_sell": n, "target": number or null, "source": "..."}},
  "scores": {{"news": 0-100, "fundamentals": 0-100, "analysts": 0-100}},
  "verdict": "buy|avoid|your_call", "summary": "3-5 sentences in Hebrew: why, what to watch",
  "risks": ["Hebrew, short"]}}
Scores are for the {side} side (100 = strongly supports entering {side})."""


def parse_analysis(text):
    m = re.search(r"```json\s*(\{.*\})\s*```", text or "", re.S) or re.search(r"(\{.*\})", text or "", re.S)
    if not m:
        raise ValueError("לא חזר ניתוח")
    a = json.loads(m.group(1))
    sc = a.get("scores") or {}
    clamp = lambda x: None if x is None else max(0, min(100, int(float(x))))
    A = a.get("analysts") or {}
    num = lambda x: int(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()) else 0
    return {"news": [n for n in (a.get("news") or []) if isinstance(n, dict) and n.get("title")][:8],
            "stats": [s for s in (a.get("stats") or []) if isinstance(s, dict) and s.get("label")][:4],
            "analysts": {k: num(A.get(k)) for k in ("strong_buy", "buy", "hold", "sell", "strong_sell")}
            | {"target": A.get("target") if isinstance(A.get("target"), (int, float)) else None, "source": A.get("source") or ""},
            "scores": {k: clamp(sc.get(k)) for k in ("news", "fundamentals", "analysts")},
            "verdict": a.get("verdict") if a.get("verdict") in ("buy", "avoid", "your_call") else "your_call",
            "summary": str(a.get("summary") or "")[:1500], "risks": [str(x)[:200] for x in (a.get("risks") or [])][:5]}


WEIGHTS = {"technical": 0.30, "news": 0.15, "fundamentals": 0.20, "analysts": 0.15, "market": 0.20}


def combine(parts):
    """The final gauge: the parts' weighted score (those that came), and its verdict."""
    got = {k: v for k, v in parts.items() if v is not None}
    if not got:
        return None, None
    w = sum(WEIGHTS[k] for k in got)
    score = round(sum(WEIGHTS[k] * v for k, v in got.items()) / w)
    return score, "buy" if score >= 65 else "avoid" if score <= 40 else "your_call"


def ask_claude(prompt, client=None):
    """(text, used) of one deep pass with the web (watchlist.ask's way: the server's search loop resumed)."""
    from . import research
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    tools = [{"type": "web_search_20260209", "name": "web_search"}, {"type": "web_fetch_20260209", "name": "web_fetch"}]
    user = {"role": "user", "content": prompt}
    content, used = [], {"searches": 0, "input": 0, "output": 0, "cost": 0.0}
    for _ in range(5):
        messages = [user] + ([{"role": "assistant", "content": content}] if content else [])
        with client.messages.stream(model=MODEL, max_tokens=8000, system=SYSTEM, tools=tools, messages=messages) as stream:
            resp = stream.get_final_message()
        content = list(content) + list(resp.content)
        u = getattr(resp, "usage", None)
        if u is not None:
            stu = getattr(u, "server_tool_use", None)
            n = (getattr(stu, "web_search_requests", 0) or 0) if stu is not None else 0
            used["input"] += getattr(u, "input_tokens", 0) or 0
            used["output"] += getattr(u, "output_tokens", 0) or 0
            used["searches"] += n
            used["cost"] += research.cost_of(MODEL, u) + n * 0.01
        if resp.stop_reason != "pause_turn":
            break
    return "".join(getattr(b, "text", "") or "" for b in content if getattr(b, "type", "") == "text"), used
