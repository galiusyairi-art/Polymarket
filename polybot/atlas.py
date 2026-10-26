"""Atlas, the stocks bot: everything that runs on top of equity.py's data, in its own database file.

- The simulation in the system: one account trading forward from the day it was switched on, a day at a
  time (the same engine as the backtest: signals at the close, entries at the next open, the owner's money
  rules), kept in the database. Open trades are priced live (every few minutes, while the market is open).
- The brain, learning all the time: in turn for each strategy it tries a small change of its rules on the
  first 60% of the history (training), and takes it only if it does better there and also holds on the
  next 20% (validation: so it doesn't learn luck). The last 20% (the locked test) it never looks at: that's
  the honest result on data it didn't see. Each change is written down.
- News: the headlines of a few public feeds (markets, the economy, the Fed), the important ones marked, and
  every couple of hours a short summary in Hebrew by Claude (when the server has its key).
- After each close: the day's review by the expert panel (Claude), from the day's trades, the market's
  picture, what the brain changed and the news, with a plan for tomorrow.
"""

import datetime as dt
import json
import logging
import os
import random
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from . import equity

log = logging.getLogger("polybot")
NY = ZoneInfo("America/New_York")

SETTINGS = {"account": 100_000.0, "risk_pct": 1.0, "max_position": 10_000.0, "max_open": 10, "regime": "primary",
            "strategies": {k: True for k in equity.STRATEGIES}, "paper_on": True, "learn_on": True, "ibkr_on": False}

# The brain's search: the values each rule may take (the target at least 3 times the risk, the owner's rule).
SEARCH = {"stop_atr": (1.5, 2.0, 2.5, 3.0), "rr": (3.0, 4.0, 5.0, 0.0), "breakeven_r": (0.0, 0.5, 1.0, 1.5),
          "trail_r": (1.0, 1.5, 2.0), "trail_atr": (0.0, 2.0, 2.5, 3.0), "max_hold": (5, 10, 15, 20),
          "rsi_max": (5, 10, 15), "vol_mult": (1.3, 1.5, 2.0), "swing_days": (20, 30, 40), "gap": (0.03, 0.04, 0.06),
          "entry": equity.ENTRIES, "limit_atr": (0.25, 0.5, 1.0), "max_stop_pct": (0.04, 0.06, 0.08, 0.10, 0.0)}
NAMES = {"stop_atr": "סטופ (ATR)", "rr": "יעד (פי הסיכון)", "breakeven_r": "סטופ לכניסה אחרי", "trail_r": "סטופ נגרר מ-",
         "trail_atr": "מרחק הסטופ הנגרר", "max_hold": "ימים לכל היותר", "rsi_max": "RSI מתחת ל-", "vol_mult": "מחזור פי",
         "swing_days": "ימי המהלך", "gap": "פער מינימלי", "entry": "כניסה", "limit_atr": "כניסה מתחת לסגירה (ATR)",
         "max_stop_pct": "סטופ לכל היותר (%)"}
MIN_TRADES = 30  # on the training data, for a change to count
MIN_VAL_TRADES = 10  # on the validation data
BRAIN_VERSION = 4  # (1 chose on the data now in the locked test; 2 on entries at the open only; 3 without the stop's cap: dropped)
LEARN_SECONDS = 3  # between two tries

FEEDS = (
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("הפד", "https://www.federalreserve.gov/feeds/press_all.xml"),
)
IMPORTANT = re.compile(r"\b(fed|fomc|powell|rate|rates|inflation|cpi|ppi|pce|jobs|payrolls?|unemployment|tariffs?|war|"
                       r"sanctions?|recession|trump|white house|treasury|yields?|earnings|guidance|downgrade|upgrade|"
                       r"bankrupt\w*|merger|acquisition|opec|oil|china|s&p|nasdaq|dow)\b", re.I)
# (headlines about a household's money or about people, not the market: a "rate" of a mortgage isn't the Fed's)
NOT_MARKET = re.compile(r"\b(mortgages?|refinanc\w*|credit cards?|retire\w*|401\(k\)|social security|personal finance|"
                        r"savings accounts?|cd rates?|lawsuits?|sues?|sued|celebrit\w*|wedding|divorce|horoscope|recipes?|"
                        r"my (husband|wife|son|daughter|mom|dad)|i'm \d+|ask an advisor|moneyist)\b", re.I)
NEWS_SUMMARY_HOURS = 2
MODEL = "claude-opus-5-5"


class Atlas:
    def __init__(self, equities):
        self.eq = equities
        self.store = equities.store
        self.store._x("CREATE TABLE IF NOT EXISTS atlas_kv (key TEXT PRIMARY KEY, value TEXT)")
        self.lock = threading.Lock()
        self.quotes = {}  # symbol -> {"price", "ts", "bars": [(ts, o, h, l, c)]}: today's, while the market is open
        self.status = {"learning": False, "tried": 0, "news_error": None, "review_error": None, "quotes_error": None}
        self._tests = (None, None)
        self.ibkr = None  # the IB Gateway connection (ibkr.py), made when the owner switches it on

    # ---------- the stored state ----------
    def _get(self, key, default=None):
        rows = self.store._x("SELECT value FROM atlas_kv WHERE key = ?", (key,))
        return json.loads(rows[0][0]) if rows else default

    def _set(self, key, value):
        self.store._x("INSERT OR REPLACE INTO atlas_kv VALUES (?, ?)", (key, json.dumps(value)))

    def settings(self):
        s = {**SETTINGS, **self._get("settings", {})}
        s["strategies"] = {**SETTINGS["strategies"], **(s.get("strategies") or {})}
        return s

    def set_settings(self, changes):
        s = self.settings()
        for k, v in (changes or {}).items():
            if k == "strategies":
                s["strategies"].update({x: bool(on) for x, on in v.items() if x in equity.STRATEGIES})
            elif k == "regime":
                if v not in ("off", "primary", "both"):
                    raise ValueError("bad request")
                s[k] = v
            elif k in ("paper_on", "learn_on", "ibkr_on"):
                s[k] = bool(v)
            elif k in ("account", "risk_pct", "max_position", "max_open"):
                v = float(v)
                if k == "risk_pct" and not 0.1 <= v <= 2.0:
                    raise ValueError("סיכון לטרייד: בין 0.1% ל-2% (הכלל שלך)")
                if k == "max_open":
                    v = int(v)
                if v <= 0:
                    raise ValueError("bad request")
                s[k] = v
        self._set("settings", s)
        return s

    def port(self):
        s = self.settings()
        return {**equity.DEFAULTS, **{k: s[k] for k in ("account", "risk_pct", "max_position", "max_open", "regime")}}

    def brain(self):
        return self._get("brain", {}) if self._get("brain_v") == BRAIN_VERSION else {}

    def plans(self, kinds=None):
        """The system's rules for each strategy: the brain's, once it learned something, else the defaults;
        the account's rules from the settings."""
        s, b, port = self.settings(), self.brain(), self.port()
        kinds = kinds or [k for k, on in s["strategies"].items() if on]
        return {k: {**equity.plan_for(k, (b.get(k) or {}).get("params")), **{x: port[x] for x in
                    ("account", "risk_pct", "max_position", "max_open", "regime")}} for k in kinds}

    # ---------- the simulation in the system ----------
    def paper_book(self):
        return self._get("paper")

    def paper_step(self):
        """The simulation, from where it stopped to the last day with data (or its first day, when started)."""
        s = self.settings()
        if not s["paper_on"]:
            return
        with self.eq.lock:
            prepared = self.eq.prepared
        spy = prepared.get("SPY")
        if not spy:
            return
        book = self.paper_book()
        plans, port = self.plans(), self.port()
        if not plans:
            return
        with self.lock:
            if book is None:
                book = equity.new_book(port["account"])
                book["started"] = spy["day"][-1]
                days = [spy["day"][-1]]  # (it starts with today's signals, for tomorrow's open)
            else:
                days = [d for d in spy["day"] if d > (book["last"] or "")]
            before_open = {t.get("id") for t in book["open"]}
            before_closed = len(book["trades"])
            for d in days:
                book["pending"] = [tuple(x) for x in book["pending"]]
                equity.step(book, prepared, d, plans, port, signals=True)
            if days:
                book["skipped"] = book.get("skipped", [])[-100:]
                self._set("paper", book)
                # The trades that opened and closed, for the mails (a trade opened and closed in the days
                # caught up on gets both).
                closed = book["trades"][before_closed:]
                opened = [t for t in book["open"] if t.get("id") not in before_open]
                opened += [{**t, "day": t["in"]} for t in closed if t["id"] not in before_open]
                self.queue_mails("sim", opened, closed)
                log.warning("Atlas: the simulation is at %s (%d open, %d closed)", book["last"], len(book["open"]), len(book["trades"]))

    # ---------- the mails' queue (the notifier sends them: notify.py, with atlas_mail.py's texts) ----------
    def queue_mails(self, mode, opened, closed, now=None):
        if not opened and not closed:
            return
        now = now or time.time()
        q = self._get("mail_queue", [])
        keep = ("id", "strategy", "symbol", "day", "in", "out", "entry", "exit", "stop", "stop0", "target", "shares",
                "risk_usd", "pnl", "r", "why", "days")
        for kind, rows in (("open", opened), ("close", closed)):
            for t in rows:
                q.append({"kind": kind, "mode": mode, "ts": now, "trade": {k: t[k] for k in keep if k in t}})
        self._set("mail_queue", q[-500:])

    def take_mails(self, kind, mode):
        """The queued mails of one kind (open / close) and mode (sim / live), removed from the queue."""
        q = self._get("mail_queue", [])
        mine = [m for m in q if m["kind"] == kind and m["mode"] == mode]
        if mine:
            self._set("mail_queue", [m for m in q if not (m["kind"] == kind and m["mode"] == mode)])
        return mine

    def peek_mails(self):
        return self._get("mail_queue", [])

    def mail_status(self):
        return self._get("mail_status", {})

    def record_mail(self, trades, kind, ok, why=None):
        st = self.mail_status()
        for t in trades:
            st.setdefault(t["id"], {})[kind] = {"ts": time.time(), "ok": ok, "why": why}
        if len(st) > 3000:
            st = dict(list(st.items())[-3000:])
        self._set("mail_status", st)

    def paper_reset(self):
        with self.lock:
            self.store._x("DELETE FROM atlas_kv WHERE key = 'paper'")
        self.paper_step()

    def paper_view(self):
        """The account: its value now (open trades at the live price when there is one), the money made today,
        this week and this month, the open trades (with their charts) and the closed ones."""
        book = self.paper_book()
        if not book:
            return None
        with self.eq.lock:
            prepared = self.eq.prepared
        value_close = book["curve"][-1][1] if book["curve"] else book["cash"]
        live, opens = book["cash"], []
        for t in book["open"]:
            p = prepared.get(t["symbol"])
            q = self.quotes.get(t["symbol"])
            last = q["price"] if q else (p["c"][-1] if p else t["entry"])
            P = equity.plan_for(t["strategy"])
            out = t["shares"] * last * (1 - P["slippage"]) - equity._fee(P, t["shares"])  # (as if sold now, as the curve counts)
            live += out
            k = max(0, len(p["c"]) - 60) if p else 0
            opens.append({**{x: t[x] for x in ("strategy", "symbol", "day", "shares", "held")},
                          "entry": round(t["entry"], 2), "stop": round(t["stop"], 2), "stop0": round(t["stop0"], 2),
                          "target": round(t["target"], 2) if t["target"] else None, "price": last,
                          "pnl": round(out - t["entry"] * t["shares"] - t.get("fees", 0), 2), "risk_usd": round(t["risk_usd"], 2),
                          "live": bool(q), "quote_ts": q and q["ts"],
                          "chart": p and {"day": p["day"][k:], "o": p["o"][k:], "h": p["h"][k:], "l": p["l"][k:], "c": p["c"][k:]},
                          "today": q and q.get("bars")})
        curve = book["curve"]

        def since(days):
            if not curve:
                return None
            cut = (dt.date.fromisoformat(curve[-1][0]) - dt.timedelta(days=days)).isoformat()
            base = next((v for d, v in reversed(curve) if d <= cut), book.get("start_value", curve[0][1]))
            return live - base
        prev = curve[-2][1] if len(curve) > 1 else (curve[0][1] if curve else book["cash"])
        return {"started": book.get("started"), "last": book["last"], "cash": book["cash"], "value": live,
                "value_close": value_close, "account": self.settings()["account"],
                "today": live - (value_close if any(o["live"] for o in opens) else prev),
                "week": since(7), "month": since(30), "total": live - self.settings()["account"],
                "open_risk": sum(max(0.0, (t["entry"] - t["stop"]) * t["shares"]) for t in book["open"]),
                "exposure": sum(o["price"] * o["shares"] for o in opens),
                "open": opens, "pending": [self._pending(x, prepared) for x in book["pending"]],
                "trades": book["trades"][-500:], "curve": curve[-260:]}

    def _pending(self, x, prepared):
        """A signal waiting for tomorrow's open: its stop, and the entry and target as if it opened at the
        last close (an estimate: the real ones come from the open's price)."""
        kind, sym, stop, *rest = x
        ceiling, order = (list(rest) + [None, None])[:2]
        p = prepared.get(sym)
        P = self.plans([kind]).get(kind) if kind in equity.STRATEGIES else None
        est = order[1] if order else (p["c"][-1] if p else None)
        target = est + P["rr"] * (est - stop) if est and P and P["rr"] and est > stop else None
        return {"strategy": kind, "symbol": sym, "stop": round(stop, 2), "est_entry": est and round(est, 2),
                "est_target": target and round(target, 2), "ceiling": ceiling and round(ceiling, 2),
                "order": order and order[0]}

    # ---------- entering during the day, when the price says so ----------
    def intraday_entries(self, now=None, fetch=None):
        """While the market is open: each signal waiting for today enters once its entry order would
        fill on today's prices so far (5-minute bars), as the end of the day's step would have it
        (the same rules, enter_pending); entered, it's in the book and its mail goes now, not after
        the close. What isn't filled by the close is gone then (step)."""
        now = now or time.time()
        book = self.paper_book()
        if not book or not book["pending"] or not self.market_open(now):
            return []
        today = dt.datetime.fromtimestamp(now, NY).date().isoformat()
        if (book["last"] or "") >= today:
            return []  # (today was stepped already)
        fetch = fetch or fetch_intraday
        bars = {}

        def bar_of(sym):
            if sym not in bars:
                try:
                    q = fetch(sym)
                    day = [b for b in q.get("bars") or [] if dt.datetime.fromtimestamp(b[0], NY).date().isoformat() == today]
                    bars[sym] = (day[0][1], max(b[2] for b in day), min(b[3] for b in day)) if day else None
                except Exception as e:
                    self.status["quotes_error"] = f"{sym}: {type(e).__name__}: {e}"[:160]
                    bars[sym] = None
            return bars[sym]
        with self.lock:
            entered = equity.enter_pending(book, today, self.plans(), self.port(), bar_of)
            if entered:
                book["pending"] = [tuple(x) for x in book["pending"]]
                self._set("paper", book)
                self.queue_mails("sim", entered, [])
                log.warning("Atlas: entered %s during the day", ", ".join(t["symbol"] for t in entered))
        return entered

    # ---------- live prices of the open trades ----------
    def market_open(self, now=None):
        t = dt.datetime.fromtimestamp(now or time.time(), NY)
        return t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0)

    def refresh_quotes(self, fetch=None):
        book = self.paper_book()
        if not book or not book["open"]:
            return
        fetch = fetch or fetch_intraday
        for t in book["open"]:
            try:
                self.quotes[t["symbol"]] = fetch(t["symbol"])
                self.status["quotes_error"] = None
            except Exception as e:
                self.status["quotes_error"] = f"{t['symbol']}: {type(e).__name__}: {e}"[:160]

    # ---------- the brain: learning all the time ----------
    def learn_once(self, kind=None, rnd=random):
        """One try: a small change of one strategy's rules, kept if it does better on the older data and
        still makes money on the newer. Returns the record of the try."""
        with self.eq.lock:
            prepared = self.eq.prepared
        if not prepared.get("SPY"):
            return None
        kinds = [k for k, on in self.settings()["strategies"].items() if on] or list(equity.STRATEGIES)
        kind = kind or kinds[self.status["tried"] % len(kinds)]
        b = self.brain()
        mine = b.get(kind) or {}
        port = self.port()
        base = {**(mine.get("params") or {})}
        if not mine.get("old") or not mine.get("val"):  # (the strategy's starting point, once)
            r = equity.backtest(prepared, kind, {**base, **{x: port[x] for x in ("account", "risk_pct", "max_position", "max_open", "regime")}})
            mine = {"params": base, "old": r["split"]["old"], "val": r["split"]["val_stats"], "tried": 0, "kept": 0, "history": []}
        keys = [k for k in SEARCH if k in equity.DEFAULTS or k in equity.STRATEGIES[kind]["params"]]
        cand = dict(base)
        for k in rnd.sample(keys, rnd.choice((1, 2))):
            cand[k] = rnd.choice([v for v in SEARCH[k] if v != equity.plan_for(kind, base).get(k)] or SEARCH[k])
        r = equity.backtest(prepared, kind, {**cand, **{x: port[x] for x in ("account", "risk_pct", "max_position", "max_open", "regime")}})
        # (the locked test, r["split"]["new"], is never read here)
        old, val = r["split"]["old"], r["split"]["val_stats"]
        better = old["trades"] >= MIN_TRADES and (old["expectancy"] or 0) > (mine["old"].get("expectancy") or 0) and (old["profit_factor"] or 0) > 1
        held = ((val["trades"] or 0) >= MIN_VAL_TRADES and (val["expectancy"] or 0) > 0
                and (val["expectancy"] or 0) >= (mine["val"].get("expectancy") or 0))
        change = {k: cand[k] for k in cand if base.get(k, equity.plan_for(kind).get(k)) != cand[k]}
        rec = {"ts": time.time(), "strategy": kind, "change": change, "old": old["expectancy"], "val": val["expectancy"],
               "kept": bool(better and held), "why": "kept" if better and held else "failed_new" if better else "not_better"}
        mine["tried"] = mine.get("tried", 0) + 1
        if rec["kept"]:
            mine.update(params=cand, old=old, val=val, since=time.time())
            mine["kept"] = mine.get("kept", 0) + 1
            mine["history"] = (mine.get("history") or [])[-29:] + [rec]
            log.warning("Atlas brain: %s now %s (training %+.0f$/trade, validation %+.0f$/trade)", kind, change,
                        old["expectancy"] or 0, val["expectancy"] or 0)
        elif rec["why"] == "failed_new":
            mine["history"] = (mine.get("history") or [])[-29:] + [rec]  # (worth showing: it would have learned luck)
        b[kind] = mine
        self._set("brain", b)
        self._set("brain_v", BRAIN_VERSION)
        self.status["tried"] += 1
        return rec

    def learn_loop(self, stop):
        while not stop.is_set():
            try:
                if self.settings()["learn_on"]:
                    self.status["learning"] = True
                    self.learn_once()
                else:
                    self.status["learning"] = False
            except Exception as e:
                log.warning("Atlas brain: %s", e)
                stop.wait(30)
            stop.wait(LEARN_SECONDS)

    # ---------- news ----------
    def refresh_news(self, fetch=None):
        fetch = fetch or fetch_feed
        have = {n["link"]: n for n in self._get("news", [])}
        errors = []
        for source, url in FEEDS:
            try:
                for item in fetch(url):
                    if item["link"] and item["link"] not in have:
                        have[item["link"]] = {**item, "source": source}
            except Exception as e:
                errors.append(f"{source}: {type(e).__name__}")
        for n in have.values():
            n["important"] = important(n)
        news = sorted(have.values(), key=lambda n: -(n.get("ts") or 0))[:300]
        self._set("news", news)
        self.status["news_error"] = ", ".join(errors) or None
        return news

    def news_summary(self, now=None):
        """A short summary of what matters in the news for the market, every NEWS_SUMMARY_HOURS (Claude)."""
        now = now or time.time()
        last = self._get("news_summary")
        if last and now - last["ts"] < NEWS_SUMMARY_HOURS * 3600 or not os.environ.get("ANTHROPIC_API_KEY"):
            return last
        heads = [n for n in self._get("news", []) if (n.get("ts") or 0) > now - 86400 and not NOT_MARKET.search(n.get("title") or "")][:60]
        if not heads:
            return last
        text = ask_claude(
            "אתה אנליסט שוק ההון בצוות של סוחר מניות מנוסה. כתוב בעברית פשוטה, קצר וחד.",
            "הכותרות של היממה האחרונה (מקור, שעה, כותרת):\n" + "\n".join(
                f"- {n['source']} · {dt.datetime.fromtimestamp(n['ts'], NY).strftime('%H:%M') if n.get('ts') else ''} · {n['title']}" for n in heads)
            + "\n\nכתוב עד 6 נקודות: מה באמת חשוב עכשיו לשוק האמריקאי ולמה, אילו מניות או סקטורים מושפעים, "
              "ומה הסנטימנט הכללי (חיובי / שלילי / מעורב). התעלם מכותרות שלא נוגעות לשוק (כסף אישי, רכילות). בלי הקדמות.")
        out = {"ts": now, "text": text}
        self._set("news_summary", out)
        return out

    # ---------- the panel's review after each close ----------
    def daily_review(self, picture, now=None, force=False):
        now = now or time.time()
        today = dt.datetime.fromtimestamp(now, NY)
        book = self.paper_book()
        reviews = self._get("reviews", [])
        day = today.date().isoformat()
        if not force and (today.weekday() >= 5 or (today.hour, today.minute) < (16, 45)
                          or any(r["day"] == day for r in reviews) or not book or book["last"] != day):
            return None
        if not book:
            return None
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.status["review_error"] = "חסר ANTHROPIC_API_KEY בשרת"
            return None
        closed = [t for t in book["trades"] if t["out"] == day]
        brain_today = [h for k, b in self.brain().items() for h in (b.get("history") or []) if h["ts"] > now - 86400]
        facts = {"day": day, "market": picture, "closed_today": closed, "open": book["open"],
                 "account": {"value": book["curve"][-1][1] if book["curve"] else None, "start": self.settings()["account"]},
                 "brain_changes_today": brain_today,
                 "news": [n["title"] for n in self._get("news", []) if n.get("important")][:25]}
        text = ask_claude(
            "אתה פאנל של סוחרי מניות מהטובים בעולם (מארק מינרביני, פול טודור ג'ונס, לינדה ראשקי, לארי קונורס, "
            "קריסטיאן קולמאגי, בריאן שאנון), שמסכם יום מסחר לסוחר מנוסה. כתוב בעברית פשוטה וחדה. אל תמציא נתונים: "
            "רק מה שבעובדות.",
            json.dumps(facts, ensure_ascii=False, default=str)
            + "\n\nכתוב 4 חלקים עם כותרות: 1. מה קרה היום בשוק. 2. הטריידים של היום: מה עבד ומה לא, ולמה. "
              "3. מה המערכת למדה ומה לשנות. 4. התוכנית למחר: מה לחפש, ממה להיזהר, ובאיזה גודל מנות.",
            max_tokens=2500)
        rec = {"day": day, "ts": now, "text": text}
        self._set("reviews", (reviews + [rec])[-30:])
        self.status["review_error"] = None
        return rec

    # ---------- IBKR: the simulation's trades as real orders in the owner's paper account ----------
    IBKR_EVERY = 60  # seconds between two looks

    def ibkr_tick(self, now=None, gateway=None):
        """Send the new signals (after each close, for the next open), follow the orders (fills, the
        stop moved as the simulation's, a sale at the next open when the simulation left on time) and
        record each trade beside the simulation's. Paper accounts only."""
        from . import ibkr
        now = now or time.time()
        if not self.settings().get("ibkr_on"):
            return None
        gw = gateway or self.ibkr or ibkr.IbGateway()
        self.ibkr = gw
        st = self._get("ibkr", {"orders": {}})
        try:
            accounts = gw.connect()
        except Exception as e:
            self.status["ibkr_error"] = f"לא מחובר ל-IB Gateway ({gw.host}:{gw.port}): {type(e).__name__}: {e}"[:300]
            return None
        if not accounts or not all(a.startswith(ibkr.PAPER_PREFIX) for a in accounts):
            self.status["ibkr_error"] = f"החשבון ({', '.join(accounts) or '?'}) אינו חשבון Paper: לא נשלח כלום"
            return None
        self.status.update(ibkr_error=None, ibkr_account=accounts[0], ibkr_ts=now)
        book = self.paper_book()
        t = dt.datetime.fromtimestamp(now, NY)
        session = t.weekday() < 5 and (9, 28) <= (t.hour, t.minute) < (16, 0)
        port, orders = self.port(), st["orders"]
        # 1. The simulation's new signals, for the next open (sent only while the market is closed:
        # an on-open order sent during the session would wait for tomorrow's).
        if book and not session:
            with self.eq.lock:
                prepared = self.eq.prepared
            for kind, sym, stop, *rest in book["pending"]:
                key = f"{kind}:{sym}:{book['last']}"
                if key in orders or kind not in equity.STRATEGIES:
                    continue
                p = prepared.get(sym)
                ceiling, order = (list(rest) + [None, None])[:2]
                est = ceiling or (order[1] if order else None) or (p["c"][-1] if p else None)
                if not est or est <= stop:
                    continue
                qty = int(min(port["account"] * port["risk_pct"] / 100 / (est - stop), port["max_position"] / est))
                if qty < 1:
                    continue
                P = self.plans([kind])[kind]
                rr = P["rr"]
                if P.get("max_stop_pct") and est - stop > P["max_stop_pct"] * est:
                    stop = est * (1 - P["max_stop_pct"])  # (as the simulation caps it; moved again once in)
                target = est + rr * (est - stop) if rr else None
                try:
                    ids = gw.place_bracket(sym, qty, ceiling, target, stop, order)
                except Exception as e:
                    self.status["ibkr_error"] = f"{sym}: {type(e).__name__}: {e}"[:300]
                    continue
                orders[key] = {"key": key, "strategy": kind, "symbol": sym, "qty": qty, "stop": round(stop, 2), "order": order,
                               "ceiling": ceiling and round(ceiling, 2), "target": target and round(target, 2), "rr": rr,
                               "ids": ids, "status": "sent", "sent": now, "signal_day": book["last"]}
                log.warning("IBKR: %s %d shares, entry %s (ceiling %s), stop %.2f", sym, qty, order or "at the open", ceiling, stop)
        # 2. Where each order stands.
        live = [o for o in orders.values() if o["status"] in ("sent", "open")]
        if live:
            try:
                known, held = gw.orders(), gw.positions()
            except Exception as e:
                self.status["ibkr_error"] = f"{type(e).__name__}: {e}"[:300]
                known, held = None, None
            for o in live if known is not None else []:
                par = known.get(o["ids"]["parent"])
                if o["status"] == "sent":
                    if par and par["status"] == "Filled" or (not par and o["symbol"] in held):
                        price = par["avg_price"] if par else held[o["symbol"]][1]
                        o.update(status="open", entry=round(price, 2), filled=par["filled"] if par else held[o["symbol"]][0],
                                 entry_day=t.date().isoformat() if session or t.hour >= 16 else o.get("entry_day"))
                        o["entry_day"] = o["entry_day"] or t.date().isoformat()
                        if o["ids"].get("target") and o["rr"]:
                            o["target"] = round(price + o["rr"] * (price - o["stop"]), 2)  # (from the real entry)
                            gw.modify(o["ids"]["target"], lmtPrice=o["target"])
                    elif par and par["status"] in ("Cancelled", "ApiCancelled", "Inactive") and not par["filled"]:
                        o.update(status="skipped", why=("לא הגיעה למחיר הכניסה עד הסגירה" if o.get("order")
                                                        else "המניה פתחה מעל התקרה (לא רודפים)" if o["ceiling"] else par["status"]))
                    continue
                for leg, why in (("stop", "stop"), ("target", "target"), ("exit", "time")):
                    x = known.get(o["ids"].get(leg)) if o["ids"].get(leg) else None
                    if x and x["status"] == "Filled":
                        exit_ = x["avg_price"]
                        why = ("trail" if o["stop"] > o["entry"] else "breakeven" if abs(o["stop"] - o["entry"]) < 0.01 else "stop") if leg == "stop" else why
                        fee = 2 * max(1.0, 0.005 * o["filled"])
                        o.update(status="closed", exit=round(exit_, 2), why=why, closed=now,
                                 pnl=round((exit_ - o["entry"]) * o["filled"] - fee, 2))
                        break
        # 3. The stop as the simulation's, and the simulation's time exits.
        if book:
            sim_open = {x.get("id"): x for x in book["open"]}
            sim_closed = {x["id"]: x for x in book["trades"][-300:]}
            for o in [x for x in orders.values() if x["status"] == "open"]:
                sid = f"{o['strategy']}:{o['symbol']}:{o.get('entry_day')}"
                s_t = sim_open.get(sid)
                if s_t and s_t["stop"] > o["stop"] + 0.005:
                    if gw.modify(o["ids"]["stop"], auxPrice=round(s_t["stop"], 2)):
                        o["stop"] = round(s_t["stop"], 2)
                elif sid in sim_closed and sim_closed[sid]["why"] in ("time", "open") and not o.get("exit_sent") and not session:
                    for leg in ("target", "stop"):
                        if o["ids"].get(leg):
                            gw.cancel(o["ids"][leg])
                    o["ids"]["exit"] = gw.sell_at_open(o["symbol"], o["filled"])
                    o["exit_sent"] = now
        if len(orders) > 400:
            for k in sorted(orders, key=lambda k: orders[k]["sent"])[:len(orders) - 400]:
                if orders[k]["status"] in ("closed", "skipped"):
                    orders.pop(k)
        self._set("ibkr", st)
        return st

    def ibkr_view(self):
        """The orders and trades in the IBKR account, each beside the simulation's same trade."""
        st = self._get("ibkr", {"orders": {}})
        book = self.paper_book() or {"open": [], "trades": []}
        sims = {x.get("id"): x for x in book["open"]}
        sims.update({x["id"]: x for x in book["trades"][-500:]})
        rows = []
        for o in sorted(st["orders"].values(), key=lambda o: -o["sent"]):
            sim = sims.get(f"{o['strategy']}:{o['symbol']}:{o.get('entry_day')}") if o.get("entry_day") else None
            rows.append({**{k: o.get(k) for k in ("key", "strategy", "symbol", "qty", "filled", "stop", "ceiling", "target",
                                                   "status", "entry", "exit", "why", "pnl", "signal_day", "entry_day")},
                         "sim": sim and {"entry": round(sim["entry"], 2), "exit": sim.get("exit"), "pnl": sim.get("pnl"),
                                         "shares": sim.get("shares"), "why": sim.get("why")}})
        return {"on": self.settings().get("ibkr_on"), "account": self.status.get("ibkr_account"),
                "error": self.status.get("ibkr_error"), "checked": self.status.get("ibkr_ts"),
                "host": self.ibkr.host if self.ibkr else None, "rows": rows[:200]}

    # ---------- the loops ----------
    def tick(self, now=None):
        """Every minute: the simulation after new data, the open trades' prices while the market is open,
        the news, and the day's review once it's due."""
        now = now or time.time()
        try:
            self.paper_step()
        except Exception as e:
            log.warning("Atlas simulation: %s", e)
        if self.market_open(now) and now - self.status.get("quotes_ts", 0) >= 120:
            self.status["quotes_ts"] = now
            try:
                self.intraday_entries(now)
            except Exception as e:
                log.warning("Atlas entries during the day: %s", e)
            self.refresh_quotes()
        if now - self.status.get("news_ts", 0) >= 600:
            self.status["news_ts"] = now
            try:
                self.refresh_news()
                self.news_summary(now)
            except Exception as e:
                self.status["news_error"] = f"{type(e).__name__}: {e}"[:160]
        if now - self.status.get("ibkr_tried", 0) >= self.IBKR_EVERY:
            self.status["ibkr_tried"] = now
            try:
                self.ibkr_tick(now)
            except Exception as e:
                self.status["ibkr_error"] = f"{type(e).__name__}: {e}"[:300]
        try:
            with self.eq.lock:
                prepared = self.eq.prepared
            self.daily_review(equity.picture(prepared), now)
        except Exception as e:
            self.status["review_error"] = f"{type(e).__name__}: {e}"[:160]

    def run(self, stop):
        while not stop.is_set():
            self.tick()
            stop.wait(60)

    def start(self, stop):
        threading.Thread(target=self.run, args=(stop,), daemon=True, name="atlas").start()
        threading.Thread(target=self.learn_loop, args=(stop,), daemon=True, name="atlas-brain").start()

    # ---------- the pages ----------
    def strategy_tests(self):
        """Each strategy at the system's rules (the brain's), by part of the history: training, validation
        and the locked test (kept until the data or the rules change)."""
        with self.eq.lock:
            prepared, version = self.eq.prepared, self.eq.version
        if not prepared.get("SPY"):
            return {}
        plans, port = self.plans(list(equity.STRATEGIES)), self.port()
        key = (version, json.dumps(plans, sort_keys=True, default=str))
        if self._tests[0] != key:
            out = {}
            for k, P in plans.items():
                r = equity.portfolio(prepared, {k: P}, port)
                out[k] = {x: r["split"][x] for x in ("old", "val_stats", "new", "robust", "val", "cut")}
            self._tests = (key, out)
        return self._tests[1]

    def view(self):
        with self.eq.lock:
            prepared = self.eq.prepared
        spy = prepared.get("SPY")
        brain = self.brain()
        news = self._get("news", [])
        tests = self.strategy_tests()
        return {
            "settings": self.settings(), "status": {**self.eq.status, **{k: v for k, v in self.status.items()
                                                                        if k not in ("quotes_ts", "ibkr_tried")}},
            "picture": equity.picture(prepared), "indices": equity.indices(prepared),
            "technical": equity.technical(spy) if spy and len(spy["c"]) > 260 else None,
            "outlook": equity.outlook(spy, prepared.get("^VIX")) if spy else None,
            "spy_chart": equity.chart(spy) if spy else None,
            "strategies": [{"key": k, **{x: v[x] for x in ("name", "by", "rule")}, "on": self.settings()["strategies"].get(k, True),
                            "brain": {x: (brain.get(k) or {}).get(x) for x in ("params", "old", "val", "tried", "kept", "since")},
                            "test": tests.get(k),
                            "history": ((brain.get(k) or {}).get("history") or [])[-10:]} for k, v in equity.STRATEGIES.items()],
            "learned": sorted((h for b in brain.values() for h in (b.get("history") or [])), key=lambda h: -h["ts"])[:20],
            "paper": self.paper_view(),
            "news": sorted(({**n, "important": important(n)} for n in news), key=lambda n: (not n["important"], -(n.get("ts") or 0)))[:40],
            "news_summary": self._get("news_summary"), "review": (self._get("reviews", []) or [None])[-1],
            "names": NAMES, "stages": equity.STAGES, "defaults": equity.DEFAULTS, "market_open": self.market_open(),
            "mails": self.mail_status(), "ibkr": self.ibkr_view(),
        }

    def backtest(self, kinds, mode="system", params=None):
        """The calendar's backtest: the system's rules (the brain's) or the owner's, for one or more strategies
        as one account."""
        kinds = [k for k in kinds or [] if k in equity.STRATEGIES] or [k for k, on in self.settings()["strategies"].items() if on]
        with self.eq.lock:
            prepared = self.eq.prepared
        if not prepared.get("SPY"):
            raise ValueError("עוד אין נתונים: הטעינה הראשונה לוקחת כמה דקות")
        port = self.port()
        if mode == "custom":
            clean = {}
            for k, v in (params or {}).items():
                if k in equity.DEFAULTS or any(k in s["params"] for s in equity.STRATEGIES.values()):
                    clean[k] = v if isinstance(equity.DEFAULTS.get(k), str) else int(float(v)) if k in equity.INTEGERS else float(v)
            if not 0.1 <= clean.get("risk_pct", 1.0) <= 5:
                raise ValueError("סיכון לטרייד: בין 0.1% ל-5%")
            if "regime" in clean and clean["regime"] not in ("off", "primary", "both"):
                raise ValueError("מצב השוק: off / primary / both בלבד")
            if "entry" in clean and clean["entry"] not in equity.ENTRIES:
                raise ValueError("כניסה: open / stop / limit בלבד")
            port = {**port, **{k: clean[k] for k in ("account", "risk_pct", "max_position", "max_open", "regime") if k in clean}}
            plans = {k: {**equity.plan_for(k, clean), **{x: port[x] for x in ("account", "risk_pct", "max_position", "max_open", "regime")}} for k in kinds}
        else:
            plans = self.plans(kinds)
        out = equity.portfolio(prepared, plans, port)
        out["stage"] = equity.stage(out, self.eq.data_ok())
        brain = self.brain()
        # Which rules ran, and when: the brain changes them every few minutes, so a result is stamped.
        out["stamp"] = {"ts": time.time(), "mode": mode, "data_last": prepared["SPY"]["day"][-1],
                        "brain": {k: {x: (brain.get(k) or {}).get(x) for x in ("kept", "tried", "since")} for k in kinds}}
        if mode != "custom" and any((brain.get(k) or {}).get("params") for k in kinds):
            # The same strategies at their default rules, side by side: what the brain's changes are worth.
            d = equity.portfolio(prepared, {k: {**equity.plan_for(k), **{x: port[x] for x in
                                 ("account", "risk_pct", "max_position", "max_open", "regime")}} for k in kinds}, port)
            out["default"] = {"stats": {k: v for k, v in d["stats"].items() if k != "by_year"}, "split": d["split"]}
        return out


def important(n):
    """A headline that matters to the market: the Fed's, or one naming what moves it (and not about a
    household's money or a person's affairs)."""
    t = n.get("title") or ""
    return n.get("source") == "הפד" or bool(IMPORTANT.search(t) and not NOT_MARKET.search(t))


def fetch_intraday(symbol):
    """Today's 5-minute bars and the price now (Yahoo, delayed a few minutes at most)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=5m"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (polybot research)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    res = data["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    bars = [(ts, q["open"][i], q["high"][i], q["low"][i], q["close"][i]) for i, ts in enumerate(res.get("timestamp") or [])
            if q["close"][i] is not None]
    price = (res.get("meta") or {}).get("regularMarketPrice") or (bars[-1][4] if bars else None)
    if price is None:
        raise ValueError("no price")
    return {"price": float(price), "ts": time.time(), "bars": bars}


def fetch_feed(url):
    """An RSS feed's items: title, link, time."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (polybot research)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        root = ET.fromstring(r.read())
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        when = item.findtext("pubDate")
        ts = None
        if when:
            try:
                ts = dt.datetime.strptime(when.strip()[:31].replace("GMT", "+0000"), "%a, %d %b %Y %H:%M:%S %z").timestamp()
            except ValueError:
                ts = None
        if title:
            out.append({"title": title, "link": link, "ts": ts})
    return out


def ask_claude(system, prompt, max_tokens=1500):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                  messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in resp.content).strip()
