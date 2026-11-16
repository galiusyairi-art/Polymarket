"""The portfolio bot (the owner, 17.11.2026, after the research found no strategy that beats the market for a small
account: "build the portfolio bot inside Atlas; for now without real money; a tab of its own: what it does and how, a
switch for real money or simulation only, and then its trades").

It holds the whole market with half the pain: 60% in a fund of every stock in the world (VT), 20% and 20% in two
trend-following funds (DBMF, KMLM) that often rise when stocks fall. It predicts nothing. Each day after the close it
looks at the three weights; when one has drifted more than BAND from its target, or a year passed since the last
rebalance, it sells a little of what grew and buys a little of what shrank, back to 60/20/20, in whole shares, the
next session. Dividends come in as cash. The rules are fixed: nothing here learns or changes them.

Two ways to run, the owner's switch: "sim", a book of its own at real prices (filled at the next session's open, with
IBKR's commission and a little slippage), and "live", the same decisions as real orders at IBKR on the bot's own
ledger (only the shares it bought itself are ever sold; nothing else in the account is touched).
"""
import datetime as dt
import json
import logging
import math
import time
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

log = logging.getLogger("polybot")
NY = ZoneInfo("America/New_York")

TARGETS = {"VT": 0.60, "DBMF": 0.20, "KMLM": 0.20}
NAMES = {"VT": "כל מניות העולם", "DBMF": "רודף מגמה 1", "KMLM": "רודף מגמה 2"}
BAND = 0.05  # a weight this far (absolute) from its target: rebalance
YEARLY = 365  # days: rebalance at least this often
SLIP = 0.0005  # the price a little worse than the open (half the spread of these funds)
ROOM = 0.003  # real money: a limit this far past the price now (filled at once, never chased far)
HISTORY_FROM = "2020-12-03"  # the first day all three funds trade (KMLM's)
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range={}&interval=1d&events=div"


def fee(qty, price):
    """IBKR's fixed commission: half a cent a share, a dollar at least, 1% of the trade at most."""
    return round(min(max(1.0, 0.005 * qty), 0.01 * qty * price), 2) if qty else 0.0


def fetch(symbol, rng="10y"):
    """{"day": [...], "o", "c", "a" (dividend-adjusted close), "div": {day: amount}} of a fund's daily bars."""
    req = urllib.request.Request(YAHOO.format(urllib.parse.quote(symbol), rng), headers={"User-Agent": "Mozilla/5.0 (polybot)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        res = json.loads(r.read())["chart"]["result"][0]
    q, adj = res["indicators"]["quote"][0], (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    off = (res.get("meta") or {}).get("gmtoffset") or 0
    day = lambda ts: dt.datetime.fromtimestamp(ts + off, dt.timezone.utc).date().isoformat()
    out = {"day": [], "o": [], "c": [], "a": [], "div": {}}
    for i, ts in enumerate(res.get("timestamp") or []):
        o, c = q["open"][i], q["close"][i]
        if o is None or c is None:
            continue
        out["day"].append(day(ts))
        out["o"].append(float(o))
        out["c"].append(float(c))
        out["a"].append(float(adj[i]) if adj and adj[i] is not None else float(c))
    for d in ((res.get("events") or {}).get("dividends") or {}).values():
        out["div"][day(d["date"])] = float(d["amount"])
    return out


def weights(shares, prices, cash):
    total = cash + sum(shares.get(s, 0) * prices[s] for s in TARGETS)
    return total, {s: (shares.get(s, 0) * prices[s] / total if total > 0 else 0.0) for s in TARGETS}


def due(shares, prices, cash, last_rebalance, today):
    """Why rebalance now (None: no need)."""
    if not any(shares.get(s) for s in TARGETS):
        return "first"
    _, w = weights(shares, prices, cash)
    far = max(TARGETS, key=lambda s: abs(w[s] - TARGETS[s]))
    if abs(w[far] - TARGETS[far]) > BAND + 1e-9:
        return f"drift:{far}:{w[far]:.3f}"
    if last_rebalance and (dt.date.fromisoformat(today) - dt.date.fromisoformat(last_rebalance)).days >= YEARLY:
        return "yearly"
    return None


def plan(shares, prices, cash):
    """The orders back to the targets, in whole shares: sells first (their money pays the buys), buys only what the cash
    (after the sells and every commission) covers. [(symbol, "SELL"/"BUY", qty)]"""
    total, _ = weights(shares, prices, cash)
    want = {s: int(total * TARGETS[s] // prices[s]) for s in TARGETS}
    sells = [(s, "SELL", shares.get(s, 0) - want[s]) for s in TARGETS if shares.get(s, 0) > want[s]]
    money = cash + sum(q * prices[s] - fee(q, prices[s]) for s, _, q in sells)
    buys = []
    for s in sorted(TARGETS, key=lambda s: -TARGETS[s]):
        q = want[s] - shares.get(s, 0)
        while q > 0 and q * prices[s] * (1 + SLIP) + fee(q, prices[s]) > money:
            q -= 1
        if q > 0:
            buys.append((s, "BUY", q))
            money -= q * prices[s] * (1 + SLIP) + fee(q, prices[s])
    return [x for x in sells + buys if x[2] > 0]


def new_book(capital, start):
    return {"cash": float(capital), "shares": {}, "start": start, "capital": float(capital), "last_day": None,
            "last_rebalance": None, "pending": [], "curve": [], "trades": [], "dividends": 0.0, "fees": 0.0}


def step(book, data, d):
    """One session for the simulated book: yesterday's decision filled at today's open, today's dividends paid, the
    book valued at the close, and today's decision taken for tomorrow."""
    idx = {s: data[s]["day"].index(d) if d in data[s]["day"] else None for s in TARGETS}
    if any(i is None for i in idx.values()):
        return False
    px = lambda s, k: data[s][k][idx[s]]
    for s, action, q in book["pending"]:
        price = px(s, "o") * (1 + SLIP if action == "BUY" else 1 - SLIP)
        have = book["shares"].get(s, 0)
        if action == "SELL":
            q = min(q, have)
        elif q * price + fee(q, price) > book["cash"]:
            q = int(max(0.0, book["cash"] - 1.0) // (price * 1.0001))
        if q <= 0:
            continue
        f = fee(q, price)
        book["cash"] += (-1 if action == "BUY" else 1) * q * price - f
        book["shares"][s] = have + (q if action == "BUY" else -q)
        book["fees"] += f
        book["trades"].append({"day": d, "symbol": s, "action": action, "qty": q, "price": round(price, 2), "fee": f,
                               "why": book.get("why_pending")})
    book["pending"] = []
    for s in TARGETS:
        amount = data[s]["div"].get(d)
        if amount and book["shares"].get(s):
            got = round(amount * book["shares"][s], 2)
            book["cash"] += got
            book["dividends"] += got
            book["trades"].append({"day": d, "symbol": s, "action": "DIV", "qty": book["shares"][s], "price": amount,
                                   "fee": 0.0, "amount": got, "why": "dividend"})
    prices = {s: px(s, "c") for s in TARGETS}
    total, _ = weights(book["shares"], prices, book["cash"])
    book["curve"].append((d, round(total, 2)))
    why = due(book["shares"], prices, book["cash"], book["last_rebalance"], d)
    if why:
        book["pending"] = plan(book["shares"], prices, book["cash"])
        book["why_pending"] = why
        if book["pending"]:
            book["last_rebalance"] = d
    book["last_day"] = d
    return True


def history(data, capital, irx=None):
    """The same rules from the first day all three trade, against holding VT alone and T-bills: {"curve", "vt", "cash",
    "years", "stats"}."""
    days = [d for d in data["VT"]["day"] if d >= HISTORY_FROM and all(d in data[s]["day"] for s in TARGETS)]
    book = new_book(capital, days[0])
    for d in days:
        step(book, data, d)
    vt = data["VT"]
    a0 = vt["a"][vt["day"].index(days[0])]
    vt_curve = [(d, round(capital * vt["a"][vt["day"].index(d)] / a0, 2)) for d in days]
    cash_curve, v = [], float(capital)
    for d in days:
        y = (irx or {}).get(d)
        v *= 1 + ((y if y is not None else 0.0) / 100) / 252
        cash_curve.append((d, round(v, 2)))

    def stats(curve):
        vals = [x for _, x in curve]
        yrs = max(1e-9, len(vals) / 252)
        peak, dd = vals[0], 0.0
        for x in vals:
            peak = max(peak, x)
            dd = min(dd, x / peak - 1)
        return {"cagr": (vals[-1] / vals[0]) ** (1 / yrs) - 1, "dd": dd, "end": vals[-1]}

    def by_year(curve):
        ends = {}
        for d, x in curve:
            ends[d[:4]] = x  # (each year's last value)
        res, last = {}, curve[0][1]
        for y in sorted(ends):
            res[y] = ends[y] / last - 1
            last = ends[y]
        return res
    step_n = max(1, len(days) // 300)
    thin = lambda c: c[::step_n] + ([c[-1]] if (len(c) - 1) % step_n else [])
    return {"from": days[0], "to": days[-1], "capital": capital,
            "curve": thin(book["curve"]), "vt": thin(vt_curve), "cash": thin(cash_curve),
            "stats": {"bot": stats(book["curve"]), "vt": stats(vt_curve), "cash": stats(cash_curve)},
            "years": {"bot": by_year(book["curve"]), "vt": by_year(vt_curve), "cash": by_year(cash_curve)},
            "trades": sum(1 for t in book["trades"] if t["action"] in ("BUY", "SELL")),
            "rebalances": len({t["day"] for t in book["trades"] if t["action"] in ("BUY", "SELL")}),
            "fees": round(book["fees"], 2), "dividends": round(book["dividends"], 2)}


class PortBot:
    """The bot inside Atlas: its state in Atlas' store ("portbot"), its pass on Atlas' minute, its page."""
    DATA_EVERY = 3600  # seconds between two reads of the funds' daily bars
    LIVE_FROM, LIVE_TO = (9, 45), (15, 45)  # real money's orders: well inside the session

    def __init__(self, atlas, fetch=fetch):
        self.a, self.fetch = atlas, fetch
        self.data, self.data_ts, self.error = None, 0.0, None
        self._hist = (None, None)

    # ---------- state ----------
    def state(self):
        return self.a._get("portbot") or {"mode": "sim", "capital": 5000.0, "sim": None, "live": None}

    def save(self, st):
        self.a._set("portbot", st)

    def load(self, now=None, force=False):
        now = now or time.time()
        if self.data and not force and now - self.data_ts < self.DATA_EVERY:
            return self.data
        try:
            data = {s: self.fetch(s) for s in TARGETS}
            try:
                irx = self.fetch("^IRX")
                data["^IRX"] = dict(zip(irx["day"], irx["c"]))
            except Exception:
                data["^IRX"] = {}
            self.data, self.data_ts, self.error = data, now, None
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"[:200]
            log.warning("Portfolio bot data: %s", e)
        return self.data

    # ---------- the owner's ----------
    def set_mode(self, mode, confirm=False):
        if mode not in ("sim", "live"):
            raise ValueError("מצב לא מוכר")
        st = self.state()
        if mode == "live" and st.get("mode") != "live":
            if not confirm:
                raise ValueError("כסף אמיתי צריך את האישור שלך")
            if not self.a.settings().get("ibkr_on"):
                raise ValueError("החיבור לאינטראקטיב כבוי בהגדרות של אטלס")
            st["live"] = st.get("live") or {"ledger": {}, "cash": float(st.get("capital") or 5000.0), "orders": [],
                                            "trades": [], "curve": [], "last_rebalance": None, "start": time.time(),
                                            "capital": float(st.get("capital") or 5000.0)}
        st["mode"] = mode
        self.save(st)
        self.a.note("portbot", "🧺 בוט התיק: " + ("כסף אמיתי" if mode == "live" else "סימולציה בלבד"), "")
        return self.view()

    def set_capital(self, capital):
        capital = float(capital)
        if not 1000 <= capital <= 1_000_000:
            raise ValueError("סכום בין $1,000 ל-$1,000,000")
        st = self.state()
        if st.get("mode") == "live" and (st.get("live") or {}).get("ledger"):
            raise ValueError("בכסף אמיתי הסכום נקבע כשהבוט התחיל")
        st["capital"] = capital
        st["sim"] = None  # (the simulation starts again from the new amount)
        if st.get("live") is not None and not (st["live"].get("ledger") or {}):
            st["live"].update(cash=capital, capital=capital)
        self.save(st)
        return self.view()

    def reset_sim(self):
        st = self.state()
        st["sim"] = None
        self.save(st)
        return self.view()

    # ---------- the minute's pass ----------
    def tick(self, now=None, gateway=None):
        now = now or time.time()
        if self.a._get("portbot") is None:
            return  # (not opened yet: it starts the day the owner first opens its tab)
        data = self.load(now)
        if not data:
            return
        st = self.state()
        t = dt.datetime.fromtimestamp(now, NY)
        today = t.date().isoformat()
        days = [d for d in data["VT"]["day"] if all(d in data[s]["day"] for s in TARGETS)]
        if not days:
            return
        # The simulation: every finished session since it started (today's once the close is in).
        sim = st.get("sim")
        if sim is None:
            sim = st["sim"] = new_book(st.get("capital") or 5000.0, today)
        closed = [d for d in days if d > (sim["last_day"] or "") and d >= sim["start"]
                  and (d < today or (t.hour, t.minute) >= (16, 30))]
        for d in closed:
            step(sim, data, d)
        sim["curve"] = sim["curve"][-2000:]
        sim["trades"] = sim["trades"][-1000:]
        if st.get("mode") == "live":
            try:
                with self.a.ib_lock:  # (one hand on the gateway at a time, as Atlas' own pass)
                    self._live(st, data, now, gateway)
            except Exception as e:
                st.setdefault("live", {})["error"] = f"{type(e).__name__}: {e}"[:300]
                log.warning("Portfolio bot (real money): %s", e)
        self.save(st)

    def _price(self, sym, data):
        q = self.a.quotes.get(sym)
        if q and time.time() - q.get("ts", 0) < 120:
            return float(q["price"])
        return data[sym]["c"][-1]

    def _live(self, st, data, now, gateway=None):
        """Real money: the same rule on the bot's own ledger; orders as limits a little past the price, for the day,
        only well inside the session; each one followed until filled or gone."""
        L = st["live"]
        gw = gateway or self.a.ibkr
        if gw is None or not getattr(gw, "connected", lambda: True)():
            L["why"] = "אין חיבור לאינטראקטיב"
            return
        t = dt.datetime.fromtimestamp(now, NY)
        today = t.date().isoformat()
        # 1. The orders on their way.
        open_orders = [o for o in L["orders"] if o["status"] == "sent"]
        if open_orders:
            known = gw.orders()
            for o in open_orders:
                k = known.get(o["id"])
                if k and k["status"] == "Filled":
                    self._filled(L, o, k["filled"], k["avg_price"], today)
                elif k and k["status"] in ("Cancelled", "ApiCancelled", "Inactive"):
                    if k.get("filled"):
                        self._filled(L, o, k["filled"], k["avg_price"], today)
                    o.update(status="gone", why=k["status"])
                    if k["status"] == "Inactive":
                        self.a.ibkr_alert("בוט התיק: אינטראקטיב דחתה הוראה", k.get("why") or "Inactive", o["symbol"], now)
                elif not k and o["day"] < today:
                    o.update(status="gone", why="לא התמלאה עד הסגירה")
            if any(o["status"] == "sent" for o in L["orders"]):
                L["why"] = "ממתין למילוי ההוראות"
                return
        prices = {s: self._price(s, data) for s in TARGETS}
        total, _ = weights(L["ledger"], prices, L["cash"])
        if not L["curve"] or L["curve"][-1][0] != today:
            L["curve"].append((today, round(total, 2)))
        else:
            L["curve"][-1] = (today, round(total, 2))
        L["curve"] = L["curve"][-2000:]
        # (a rebalance whose sales are done goes on to its buys, though the weights alone may no longer ask for it)
        why = due(L["ledger"], prices, L["cash"], L.get("last_rebalance"), today) or L.get("rebalancing")
        if not why:
            L["why"] = "הכל מאוזן: אין מה לעשות"
            return
        if t.weekday() >= 5 or not (self.LIVE_FROM <= (t.hour, t.minute) < self.LIVE_TO):
            L["why"] = "צריך איזון: ההוראות יישלחו במסחר הבא (בין 9:45 ל-15:45 בניו יורק)"
            return
        if any(o["day"] == today and o["status"] == "gone" for o in L["orders"]):
            L["why"] = "הוראה של היום לא התמלאה או נדחתה: ננסה שוב במסחר הבא"
            return
        orders = plan(L["ledger"], prices, L["cash"])
        if not orders:
            L["rebalancing"], L["last_rebalance"] = None, today
            L["why"] = "הכל מאוזן: אין מה לעשות"
            return
        sells = [x for x in orders if x[1] == "SELL"]
        for s, action, q in (sells or orders):  # (the sales first: their money pays the buys, at the next pass)
            limit = round(prices[s] * (1 + ROOM if action == "BUY" else 1 - ROOM), 2)
            oid = gw.add_order(s, action, int(q), limit, True)
            L["orders"].append({"id": oid, "symbol": s, "action": action, "qty": int(q), "limit": limit, "day": today,
                                "sent": now, "status": "sent", "why": why})
        L["orders"] = L["orders"][-300:]
        if sells:
            L["rebalancing"] = why
        else:
            L["rebalancing"], L["last_rebalance"] = None, today
        L["why"] = "נשלחו הוראות איזון"

    def _filled(self, L, o, qty, price, today):
        qty = int(qty or 0)
        if not qty:
            return
        f = fee(qty, price)
        sign = 1 if o["action"] == "BUY" else -1
        L["ledger"][o["symbol"]] = L["ledger"].get(o["symbol"], 0) + sign * qty
        L["cash"] += -sign * qty * price - f
        o.update(status="filled", filled=qty, price=round(price, 2))
        L["trades"].append({"day": today, "symbol": o["symbol"], "action": o["action"], "qty": qty, "price": round(price, 2),
                            "fee": f, "why": o.get("why")})
        L["trades"] = L["trades"][-1000:]

    # ---------- the page ----------
    def view(self):
        if self.a._get("portbot") is None:
            self.save(self.state())  # (its first look: the simulation starts today)
        if not self.data:
            self.load()
            self.tick()
        st = self.state()
        data = self.data
        out = {"mode": st.get("mode", "sim"), "capital": st.get("capital", 5000.0), "targets": TARGETS, "names": NAMES,
               "band": BAND, "yearly": YEARLY, "error": self.error}
        prices = {s: data[s]["c"][-1] for s in TARGETS} if data else None
        for k in ("sim", "live"):
            b = st.get(k)
            if not b:
                continue
            shares = b.get("shares") if k == "sim" else b.get("ledger")
            total, w = weights(shares or {}, prices, b["cash"]) if prices else (None, {})
            out[k] = {"cash": round(b["cash"], 2), "shares": shares, "value": total and round(total, 2),
                      "weights": w, "capital": b.get("capital"), "start": b.get("start"),
                      "curve": b.get("curve", [])[-400:], "trades": list(reversed(b.get("trades", [])[-60:])),
                      "pending": b.get("pending") if k == "sim" else [o for o in b.get("orders", []) if o["status"] == "sent"],
                      "last_rebalance": b.get("last_rebalance"), "dividends": round(b.get("dividends", 0.0), 2),
                      "fees": round(b.get("fees", 0.0), 2), "why": b.get("why"), "error": b.get("error")}
        if data:
            key = (data["VT"]["day"][-1], st.get("capital"))
            if self._hist[0] != key:
                try:
                    self._hist = (key, history(data, float(st.get("capital") or 5000.0), data.get("^IRX")))
                except Exception as e:
                    log.warning("Portfolio bot history: %s", e)
            out["history"] = self._hist[1]
            out["prices"] = prices
            out["as_of"] = data["VT"]["day"][-1]
        return out
