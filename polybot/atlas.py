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

from . import equity, intraday, livelab, mentor, scalp, sizing, streaks

log = logging.getLogger("polybot")
NY = ZoneInfo("America/New_York")

SETTINGS = {"account": 100_000.0, "risk_pct": 1.0, "max_position": 10_000.0, "max_open": 10, "regime": "primary",
            "strategies": {k: True for k in equity.STRATEGIES}, "paper_on": True, "learn_on": True, "ibkr_on": False,
            # IBKR: the paper account, or real money ("live"), where each trade is only live_size dollars
            # until the owner sees it all works (his rule of 28.10.2026)
            "ibkr_mode": "paper", "live_size": 100.0,
            # (with the connection on: True sends every signal by itself; False, each waits for the owner's
            # yes on the hub: the same three modes as Polymarket's, simulation / on approval / automatic)
            "ibkr_auto": True,
            # The owner's two overrides of the system (everything else is the system's): the size of a real
            # trade (live_size) and the target as a multiple of the risk (None: each strategy's own, the
            # brain's; 0: no target, the trailing stop only). The target applies to the simulation too, so
            # the two run under the same rules.
            "rr_override": None,
            # Real money's own guards (the owner's settings, 28.10.2026; 0: none): no new entry once today's
            # loss in the real account reaches live_daily_loss dollars, or with live_max_open positions held.
            "live_daily_loss": 0.0, "live_max_open": 0,
            # (and the most a real trade may lose at its stop, in dollars; 0: the account's risk rule only)
            "live_risk": 0.0,
            # The mentor's signals (mentor.py, the owner 28.10.2026: automatic from the start, switched off here):
            # on / automatic (False: each waits for his yes), the stop when the mentor gives none (percent under
            # the entry; trailing or fixed), and whose messages count.
            "mentor_on": True, "mentor_auto": True, "mentor_stop_pct": 6.0, "mentor_trail": False,
            "mentor_author": "Raz Gamliel",
            # The live lab (livelab.py, the owner 28.10.2026): experiment 1 every day in real time on the whole
            # market (each variant's simulation); the ones the owner moves to real money trade there (lanes).
            "livelab_on": True,
            # The size and the number open at once (sizing.py, the owner 29.10.2026): the research's answer becomes
            # every strategy's by itself (False: shown only); the most the account may fall, its bound.
            "sizing_auto": True, "sizing_dd": 20.0,
            # Each strategy's place (the owner, 29.10.2026: lab -> simulation -> real money, as Polybot's): in the
            # simulation and with real money, "on" / "paused" (no new entries, its open trades go on) / "out".
            # None: from "strategies" (the simulation's), and real money as the simulation (the behaviour till then).
            "lanes": None}
LIVE_SIZE_MAX = 50_000.0  # (a typo's guard, not a limit)

# The brain's search: the values each rule may take (the target at least 3 times the risk, the owner's rule).
SEARCH = {"stop_atr": (1.5, 2.0, 2.5, 3.0), "rr": (3.0, 4.0, 5.0, 0.0), "breakeven_r": (0.0, 0.5, 1.0, 1.5),
          "trail_r": (1.0, 1.5, 2.0), "trail_atr": (2.0, 2.5, 3.0), "max_hold": (5, 10, 15, 20),
          "rsi_max": (5, 10, 15), "vol_mult": (1.3, 1.5, 2.0), "swing_days": (20, 30, 40), "gap": (0.03, 0.04, 0.06),
          "entry": equity.ENTRIES, "limit_atr": (0.25, 0.5, 1.0), "max_stop_pct": (0.04, 0.06, 0.08, 0.10, 0.0)}
NAMES = {"stop_atr": "סטופ (ATR)", "rr": "יעד (פי הסיכון)", "breakeven_r": "סטופ לכניסה אחרי", "trail_r": "סטופ נגרר מ-",
         "trail_atr": "מרחק הסטופ הנגרר", "max_hold": "ימים לכל היותר", "rsi_max": "RSI מתחת ל-", "vol_mult": "מחזור פי",
         "swing_days": "ימי המהלך", "gap": "פער מינימלי", "entry": "כניסה", "limit_atr": "כניסה מתחת לסגירה (ATR)",
         "max_stop_pct": "סטופ לכל היותר (%)"}
MIN_TRADES = 30  # on the training data, for a change to count
MIN_VAL_TRADES = 10  # on the validation data
BRAIN_VERSION = 5  # (1 chose on the data now in the locked test; 2 on entries at the open only; 3 without the stop's cap;
# 4 with a stop that rose only from some profit on: dropped)
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
        self.ib_lock = threading.Lock()
        self._mail_lock = threading.Lock()  # (the mail queue: added to by the trading, taken by the mail loop)  # (the minute's IBKR pass and the owner's own orders, one at a time)
        self._notes_lock = threading.Lock()
        self.lab_bars = scalp.Bars(os.path.join(os.path.dirname(os.path.abspath(self.store.path)), "atlas_minutes"))
        self._lab_busy = False
        self._streaks_busy = False
        self._sizing_busy = False
        self.mentor = mentor.Mentor(self)
        self.livelab = livelab.LiveLab(self)

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
                if s.get("lanes"):
                    for x, on in v.items():
                        if x in equity.STRATEGIES and (s["lanes"].get("sim") or {}).get(x) != "paused":
                            s["lanes"].setdefault("sim", {})[x] = "on" if on else "out"
            elif k == "regime":
                if v not in ("off", "primary", "both"):
                    raise ValueError("bad request")
                s[k] = v
            elif k in ("paper_on", "learn_on", "ibkr_on", "ibkr_auto", "mentor_on", "mentor_auto", "mentor_trail"):
                if k in ("ibkr_on", "ibkr_auto", "mentor_on", "mentor_auto") and v and s.get("ibkr_mode") == "live" and not s.get(k) \
                        and not (changes or {}).get("confirm_live"):
                    raise ValueError("מסחר בכסף אמיתי צריך את האישור שלך")
                s[k] = bool(v)
            elif k == "ibkr_mode":
                if v not in ("paper", "live"):
                    raise ValueError("bad request")
                if v == "live" and s.get("ibkr_mode") != "live" and not (changes or {}).get("confirm_live"):
                    raise ValueError("מעבר לכסף אמיתי צריך את האישור שלך")
                s[k] = v
            elif k == "rr_override":
                if v in (None, "", "system"):
                    s[k] = None
                else:
                    v = float(v)
                    if not (v == 0 or 1 <= v <= 10):
                        raise ValueError("יחס סיכון / סיכוי: בין 1 ל-10 (או 0: בלי יעד, רק הסטופ הנגרר)")
                    s[k] = v
            elif k in ("live_daily_loss", "live_max_open", "live_risk"):
                v = float(v or 0)
                if v < 0 or (k == "live_max_open" and v > 100) or (k in ("live_daily_loss", "live_risk") and v > LIVE_SIZE_MAX):
                    raise ValueError("bad request")
                s[k] = int(v) if k == "live_max_open" else v
            elif k == "mentor_stop_pct":
                v = float(v)
                if not 1 <= v <= 30:
                    raise ValueError("הסטופ כשהמנטור לא נותן: בין 1% ל-30%")
                s[k] = v
            elif k in ("lanes", "lanes_since"):
                continue  # (set_lane only)
            elif k in ("livelab_on", "sizing_auto"):
                s[k] = bool(v)
            elif k == "sizing_dd":
                v = float(v)
                if not 5 <= v <= 50:
                    raise ValueError("הנפילה המקסימלית: בין 5% ל-50%")
                s[k] = v
            elif k == "livelab_paper":
                continue  # (gone: no paper account in Atlas, the owner 29.10.2026)
            elif k == "mentor_author":
                s[k] = str(v or "").strip()[:60]
            elif k == "live_size":
                v = float(v)
                if not 10 <= v <= LIVE_SIZE_MAX:
                    raise ValueError(f"סכום לטרייד בכסף אמיתי: בין $10 ל-${LIVE_SIZE_MAX:,.0f}")
                s[k] = v
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

    LANE_STATES = ("on", "paused", "out")

    def lanes(self, s=None):
        """{"sim": {strategy: state}, "live": {...}} (see SETTINGS["lanes"]): the swing strategies, and the
        intraday ones (livelab.LIVE: out of both till the owner moves them)."""
        s = s or self.settings()
        stored = s.get("lanes") or {}
        sim = {k: (stored.get("sim") or {}).get(k) or ("on" if s["strategies"].get(k, True) else "out") for k in equity.STRATEGIES}
        live = {k: (stored.get("live") or {}).get(k) or sim[k] for k in equity.STRATEGIES}
        for k in livelab.LIVE:
            sim[k] = (stored.get("sim") or {}).get(k) or "out"
            live[k] = (stored.get("live") or {}).get(k) or "out"
        return {"sim": sim, "live": live}

    def set_lane(self, mode, kind, state, confirm_live=False):
        """A strategy into / out of the simulation or real money, or paused there (the owner's)."""
        intraday = kind in livelab.LIVE
        if mode not in ("sim", "live") or not (kind in equity.STRATEGIES or intraday) or state not in self.LANE_STATES:
            raise ValueError("bad request")
        s = self.settings()
        L = self.lanes(s)
        if mode == "live" and state == "on" and L["live"][kind] != "on" and (s.get("ibkr_mode") == "live" or intraday) and not confirm_live:
            raise ValueError("מסחר בכסף אמיתי צריך את האישור שלך")
        if L[mode][kind] == "out" and state == "on":  # (its record there starts today)
            s.setdefault("lanes_since", {}).setdefault(mode, {})[kind] = dt.datetime.now(NY).date().isoformat()
        L[mode][kind] = state
        s["lanes"] = L
        if mode == "sim" and not intraday:  # (the simulation's own switch, read across the code)
            s["strategies"][kind] = state == "on"
        self._set("settings", s)
        name = scalp.NAMES[kind] if intraday else equity.STRATEGIES[kind]["name"]
        self.note("lane", f"{'🧪 סימולציה' if mode == 'sim' else '💵 כסף אמיתי'} · {kind if intraday else ''} {name[:60]}",
                  {"on": "פועלת", "paused": "מושהית", "out": "הוצאה"}[state])
        return self.lanes()

    def port(self):
        s = self.settings()
        return {**equity.DEFAULTS, **{k: s[k] for k in ("account", "risk_pct", "max_position", "max_open", "regime")}}

    def brain(self):
        return self._get("brain", {}) if self._get("brain_v") == BRAIN_VERSION else {}

    def plans(self, kinds=None, system=False):
        """The system's rules for each strategy: the brain's, once it learned something, else the defaults;
        the account's rules from the settings."""
        s, b, port = self.settings(), self.brain(), self.port()
        kinds = kinds or [k for k, on in s["strategies"].items() if on]
        out = {k: {**equity.plan_for(k, (b.get(k) or {}).get("params")), **{x: port[x] for x in
                   ("account", "risk_pct", "max_position", "max_open", "regime")}} for k in kinds}
        if s.get("rr_override") is not None and not system:  # (the owner's target, for the simulation and IBKR alike)
            for P in out.values():
                P["rr"] = s["rr_override"]
        return out

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
        # (every strategy's rules for the trades it has open; only the simulation's running ones enter)
        plans, port = self.plans(list(equity.STRATEGIES)), self.port()
        active = {k for k, st in self.lanes()["sim"].items() if st == "on"}
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
                equity.step(book, prepared, d, plans, port, signals=True, active=active)
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

    def run_audit(self, now=None):
        """The audit of every calculation (equity.audit_trades), on the server's own data: the backtest of the
        system's rules over all the history, and the simulation as it ran; and the two set side by side over
        the simulation's days (the same rules on the same days: a large gap says something differs between
        the backtest and the simulation as it runs). Kept, and shown in the settings."""
        now = now or time.time()
        with self.eq.lock:
            prepared = self.eq.prepared
        if not prepared.get("SPY"):
            raise ValueError("אין עדיין נתוני מניות")
        plans, port = self.plans(), self.port()
        r = equity.portfolio(prepared, plans, port)
        bt = equity.audit_trades(prepared, r["trades"], plans, port, r["curve"])
        bt.update(stats={k: r["stats"].get(k) for k in ("trades", "win_rate", "avg_win", "avg_loss", "profit_factor", "pnl", "max_drawdown")},
                  first=r["curve"][0][0] if r["curve"] else None, last=r["curve"][-1][0] if r["curve"] else None)
        book = self.paper_book() or {}
        sim = None
        if book.get("trades") or book.get("open"):
            started = book.get("started") or ""
            curve = [x for x in book.get("curve", []) if x[0] >= started]
            sim = equity.audit_trades(prepared, list(book.get("trades", [])), plans, port, curve, book.get("cash"), book.get("open", []))
            closed = book.get("trades", [])
            wins = [t for t in closed if t["pnl"] > 0]
            same = [t for t in r["trades"] if t["in"] >= started]
            sim.update(started=started, closed=len(closed), win_rate=len(wins) / len(closed) if closed else None,
                       pnl=sum(t["pnl"] for t in closed),
                       backtest_same_days={"trades": len(same), "pnl": sum(t["pnl"] for t in same),
                                           "win_rate": sum(1 for t in same if t["pnl"] > 0) / len(same) if same else None},
                       in_both=len(({t["id"] for t in closed} | {t.get("id") for t in book.get("open", [])}) & {t["id"] for t in same}))
        out = {"ts": now, "backtest": bt, "sim": sim,
               "ok": not bt["problems"] and not (sim and sim["problems"])}
        self._set("audit", out)
        if not out["ok"]:
            n = len(bt["problems"]) + len((sim or {}).get("problems") or [])
            self.note("alert", "🔬 בדיקת החישובים מצאה בעיות", f"{n} ממצאים: הפירוט בהגדרות", "bad", now=now)
        return out

    def _quiet_safe(self):
        try:
            return self.why_quiet()
        except Exception as e:  # (a diagnosis never stops the page)
            log.warning("Atlas why_quiet: %s", e)
            return None

    def why_quiet(self, now=None):
        """Why no new trade came in (the owner's question of 28.10.2026), every gate between a signal and an
        entry, in order: the data (is the last day there?), the market's filter, room in the account (open
        trades against the most allowed, cash), the signals the last day gave, the ones waiting for their
        price (how far it is), the ones passed over, and, for real money, what wasn't sent and why."""
        now = now or time.time()
        with self.eq.lock:
            prepared = self.eq.prepared
        spy = prepared.get("SPY")
        book, port, plans = self.paper_book() or {}, self.port(), self.plans()
        if not spy or not book:
            return None
        last = spy["day"][-1]
        t = dt.datetime.fromtimestamp(now, NY)
        expect = t.date()
        if (t.hour, t.minute) < (16, 30):  # (today's close isn't in before the evening)
            expect -= dt.timedelta(days=1)
        while expect.weekday() >= 5:
            expect -= dt.timedelta(days=1)
        si = spy["index"][last]
        s200, s50, c = spy["sma200"][si], spy["sma50"][si], spy["c"][si]
        market = equity.market_ok(spy, last, port["regime"])
        held = {x["symbol"] for x in book.get("open", [])}
        found = []
        for kind, P in plans.items():
            for sym in equity.UNIVERSE:
                p = prepared.get(sym)
                i = p and p["index"].get(last)
                if i is not None and sym not in held and equity.signal(kind, p, i, P) is not None:
                    found.append({"strategy": kind, "symbol": sym})
        pend = []
        for kind, sym, stop, *rest in book.get("pending", []):
            ceiling, order = (list(rest) + [None, None])[:2]
            q = self.quotes.get(sym)
            p = prepared.get(sym)
            price = q["price"] if q else (p["c"][-1] if p else None)
            pend.append({"strategy": kind, "symbol": sym, "order": order and order[0], "trigger": order and order[1],
                         "price": price, "ceiling": ceiling})
        entries = sorted({x["day"] for x in book.get("open", [])} | {x["in"] for x in book.get("trades", [])})
        S = self.settings()
        live = [o for o in self._get("ibkr", {"orders": {}})["orders"].values()
                if o.get("mode") == S.get("ibkr_mode") and o["status"] == "skipped" and now - o["sent"] <= 4 * 86400]
        free = int(port["max_open"]) - len(book.get("open", []))
        lsig = []
        if S.get("ibkr_on"):
            for kind, sym, stop, *rest in self.live_signals(book):
                p = prepared.get(sym)
                px = (rest[1][1] if len(rest) > 1 and rest[1] else None) or (p["c"][-1] if p else None)
                lsig.append({"strategy": kind, "symbol": sym, "price": px,
                             "too_dear": bool(S.get("ibkr_mode") == "live" and px and px > S.get("live_size", 0))})
        return {"live_signals": lsig, "last_day": last, "expected_day": expect.isoformat(), "data_ok": last >= expect.isoformat(),
                "data_error": ((self.eq.status or {}).get("errors") or [None])[0], "refreshed": (self.eq.status or {}).get("refreshed"),
                "regime": port["regime"], "market_ok": market, "spy": c, "sma200": s200, "sma50": s50,
                "open": len(book.get("open", [])), "max_open": int(port["max_open"]), "free": free,
                "cash": book.get("cash"), "max_position": port["max_position"],
                "signals": found[:30], "signals_n": len(found), "pending": pend,
                "skipped": [x for x in book.get("skipped", []) if x["day"] >= (entries[-1] if entries else "")][-10:],
                "last_entry": entries[-1] if entries else None,
                "live_on": bool(S.get("ibkr_on")), "live_mode": S.get("ibkr_mode"), "live_size": S.get("live_size"),
                "live_skipped": [{k: o.get(k) for k in ("symbol", "strategy", "why", "signal_day")} for o in live][-10:]}

    # ---------- the notifications (the bell in the header): everything the system said, newest first ----------
    NOTES_KEEP = 300

    def note(self, kind, title, text="", level="info", now=None):
        """kind: trade / alert / ibkr / brain / review; level: info / good / bad."""
        with self._notes_lock:
            n = self._get("notes", [])
            n.append({"ts": now or time.time(), "kind": kind, "title": title, "text": text, "level": level})
            self._set("notes", n[-self.NOTES_KEEP:])

    def notes_view(self):
        seen = self._get("notes_seen", 0)
        n = self._get("notes", [])
        return {"items": n[-150:][::-1], "unread": sum(1 for x in n if x["ts"] > seen), "seen": seen}

    def notes_seen(self, now=None):
        """The owner opened the bell: the count goes, read or not."""
        self._set("notes_seen", now or time.time())
        return {"ok": True}

    # ---------- the mails' queue (the notifier sends them: notify.py, with atlas_mail.py's texts) ----------
    def queue_mails(self, mode, opened, closed, now=None):
        if not opened and not closed:
            return
        now = now or time.time()
        label = "כסף אמיתי" if mode == "live" else "סימולציה"
        for t in opened:
            self.note("trade", f"📈 נפתח {t['symbol']}", f"{equity.STRATEGIES.get(t['strategy'], {}).get('name', t['strategy'])} · {label}"
                      + (f" · {t['shares']} מניות ב-${t['entry']:,.2f}" if t.get("entry") and t.get("shares") else ""), now=now)
        for t in closed:
            pnl = t.get("pnl") or 0
            self.note("trade", f"{'✅' if pnl > 0 else '❌'} נסגר {t['symbol']} · {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}",
                      f"{equity.STRATEGIES.get(t['strategy'], {}).get('name', t['strategy'])} · {label}"
                      + (f" · {t['r']:+.2f}R" if t.get("r") is not None else ""), "good" if pnl > 0 else "bad", now)
        keep = ("id", "strategy", "symbol", "day", "in", "out", "entry", "exit", "stop", "stop0", "target", "shares",
                "risk_usd", "pnl", "r", "why", "days")
        with self._mail_lock:  # (the minute's loop adds, the mail loop takes: one at a time, nothing lost)
            q = self._get("mail_queue", [])
            for kind, rows in (("open", opened), ("close", closed)):
                for t in rows:
                    q.append({"kind": kind, "mode": mode, "ts": now, "trade": {k: t[k] for k in keep if k in t}})
            self._set("mail_queue", q[-500:])

    def take_mails(self, kind, mode):
        """The queued mails of one kind (open / close) and mode (sim / live), removed from the queue."""
        with self._mail_lock:
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
                          "id": t.get("id") or f"{t['strategy']}:{t['symbol']}:{t['day']}", "adds": t.get("adds") or [],
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
        fetch = fetch or self.quote
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

    def intraday_stops(self, now=None):
        """While the market is open (the owner's rule): each open trade's stop rises with the prices as
        they come (today's 5-minute bars, then the price now), at the distance it started with, and
        never comes down; a price back down to the stop closes the trade there and then. (At IBKR the
        stop is the broker's own trailing stop: it moves tick by tick.)"""
        now = now or time.time()
        if not self.market_open(now):
            return []
        today = dt.datetime.fromtimestamp(now, NY).date().isoformat()
        closed = []
        with self.lock:
            book = self.paper_book()
            if not book or not book["open"] or (book["last"] or "") >= today:
                return []
            plans, still = self.plans(list(equity.STRATEGIES)), []
            for t in book["open"]:
                q = self.quotes.get(t["symbol"])
                if not q:
                    still.append(t)
                    continue
                P, dist, out = plans.get(t["strategy"]) or equity.plan_for(t["strategy"]), equity.trail_distance(t), None
                if t["day"] != today:  # (a trade in since today: its bars before the entry aren't its own)
                    for ts, o, h, lo, c in q.get("bars") or []:
                        if ts <= t.get("bar_ts", 0) or dt.datetime.fromtimestamp(ts, NY).date().isoformat() != today:
                            continue
                        if lo <= t["stop"]:
                            out = min(t["stop"], o)  # (opened under it: at that price)
                            break
                        t["best"] = max(t["best"], h)
                        t["stop"] = max(t["stop"], t["best"] - dist)
                        t["bar_ts"] = ts
                price = q["price"]
                if out is None and price <= t["stop"]:
                    out = price
                if out is None:
                    t["best"] = max(t["best"], price)
                    t["stop"] = max(t["stop"], t["best"] - dist)
                    still.append(t)
                    continue
                why = "trail" if t["stop"] > t["entry"] + 1e-9 else "breakeven" if abs(t["stop"] - t["entry"]) < 1e-9 else "stop"
                equity._close(book, t, today, out, why, P)
                closed.append(book["trades"][-1])
            book["open"] = still
            self._set("paper", book)
        if closed:
            self.queue_mails("sim", [], closed)
            log.warning("Atlas: stopped out during the day: %s", ", ".join(t["symbol"] for t in closed))
        return closed

    # ---------- live prices of the open trades ----------
    def market_open(self, now=None):
        t = dt.datetime.fromtimestamp(now or time.time(), NY)
        return t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0)

    def quote(self, sym, now=None):
        """A stock's price now and today's 5-minute bars: in real time from IBKR while the gateway is
        connected and has a real-time price for it (the bot's user's market data), else Yahoo's (a few
        minutes late). Which one served last is shown in the settings."""
        now = now or time.time()
        gw = self._data_gateway(now)
        if gw is not None:
            try:
                q = gw.quote(sym, now)
                if q:
                    self.status.update(quotes_source="ibkr", quotes_source_ts=now, ibkr_data_error=None)
                    return q
                self.status["ibkr_data_error"] = f"{sym}: אין מחיר בזמן אמת מאינטראקטיב"
            except Exception as e:
                self.status["ibkr_data_error"] = f"{sym}: {type(e).__name__}: {e}"[:160]
        q = dict(fetch_intraday(sym), source="yahoo")
        self.status.update(quotes_source="yahoo", quotes_source_ts=now)
        return q

    DATA_RETRY = 600  # seconds between two tries to reach the gateway for prices only

    def _data_gateway(self, now):
        """The gateway for prices: the orders' connection when it's up; with orders off, one of its own
        for prices only (nothing is sent on it), tried every DATA_RETRY seconds. Only on the server, where
        the gateway runs (FLY_APP_NAME / IBKR_HOST)."""
        from . import ibkr
        gw = self.ibkr
        if gw is not None and gw.connected():
            return gw
        if not (os.environ.get("FLY_APP_NAME") or os.environ.get("IBKR_HOST")):
            return None
        if now - self.status.get("ibkr_data_tried", 0) < self.DATA_RETRY:
            return None
        self.status["ibkr_data_tried"] = now
        try:
            with self.ib_lock:
                gw = self.ibkr = gw or ibkr.IbGateway()
                gw.connect()
            return gw
        except Exception as e:
            self.status["ibkr_data_error"] = f"החיבור לנתונים: {type(e).__name__}: {e}"[:160]
            return None

    def refresh_quotes(self, fetch=None):
        book = self.paper_book()
        if not book or not book["open"]:
            return
        fetch = fetch or self.quote
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
            self.note("brain", f"🧠 המוח שינה את {equity.STRATEGIES[kind]['name']}",
                      ", ".join(f"{NAMES.get(k, k)} {v}" for k, v in change.items()) + f" · אימות {val['expectancy'] or 0:+.0f}$ לטרייד", "good")
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
        self.note("review", f"🎙️ סיכום היום של הפאנל ({day})", "נכתב: בדשבורד, למטה", now=now)
        self.status["review_error"] = None
        return rec

    # ---------- IBKR: the simulation's trades as real orders in the owner's paper account ----------
    IBKR_EVERY = 60  # seconds between two looks

    # ---------- connecting again (the owner, 29.10.2026: his login on the IBKR app takes the gateway's) ----------
    # IBKR keeps one session a login: when the owner logs in on the app, the gateway's ends. The way back is the
    # gateway's: IBC (the program that logs it in) is asked to log in to IBKR again (RECONNECTACCOUNT, as
    # Ctrl-Alt-R), and if that didn't bring it back, to restart with the session's own credentials (RESTART: no
    # second approval); our side drops its connection (a stuck one too) for a new one. On the owner's button,
    # and by itself every RECONNECT_EVERY seconds while the gateway doesn't answer.
    RECONNECT_EVERY = 180

    def reconnect(self, now=None, by="owner"):
        from . import ibkr
        now = now or time.time()
        old = self.ibkr
        host, port = (old.host, old.port) if old else (ibkr.default_host(), None)
        tries = int(self.status.get("reconnect_tries") or 0)
        cmd = "RESTART" if tries % 2 else "RECONNECTACCOUNT"
        try:
            answer, sent = ibkr.gateway_command(host, cmd), True
        except Exception as e:
            answer, sent = f"{type(e).__name__}: {e}", False
        self.ibkr = ibkr.IbGateway(host, port)
        if old is not None:
            threading.Thread(target=old.close, daemon=True, name="ibkr-close").start()
        rec = {"ts": now, "by": by, "cmd": cmd, "sent": sent, "answer": answer[:200], "try": tries + 1}
        self.status.update(reconnect_tries=tries + 1, reconnect=rec, ibkr_tried=0)  # (the next minute's look connects)
        log.info("Atlas reconnect (%s): %s %s", by, cmd, "sent" if sent else answer)
        return rec

    def clear_alerts(self, now=None):
        """The owner read the past failures: they leave the hub (kept in the log; a new one shows again)."""
        self._set("ibkr_alerts_cleared", now or time.time())
        return True

    def _auto_reconnect(self, now):
        since = self.status.get("ibkr_down_since")
        last = (self.status.get("reconnect") or {}).get("ts") or 0
        if (self.settings().get("ibkr_on") and since and now - since >= self.RECONNECT_EVERY
                and now - last >= self.RECONNECT_EVERY):
            self.reconnect(now, by="auto")
            return True
        return False

    # ---------- what went wrong with IBKR: said at once (the hub, a popup, a mail), with where ----------
    ALERT_REPEAT = 1800  # the same failure again within this: counted, not said again
    DOWN_GRACE = 240  # seconds the gateway may not answer (a restart) before it is an alert

    def ibkr_alert(self, where, what, symbol=None, now=None):
        now = now or time.time()
        self.status["ibkr_error"] = f"{where}{' · ' + symbol if symbol else ''}: {what}"[:300]
        alerts = self._get("ibkr_alerts", [])
        last = next((a for a in reversed(alerts) if a["where"] == where and a.get("symbol") == symbol and a["what"] == what), None)
        if last and now - last["ts"] < self.ALERT_REPEAT:
            last["n"], last["ts"] = last.get("n", 1) + 1, now
        else:
            alerts.append({"ts": now, "where": where, "what": str(what)[:400], "symbol": symbol, "n": 1,
                           "mode": self.settings().get("ibkr_mode", "paper")})
            self.note("alert", f"⚠️ {where}{' · ' + symbol if symbol else ''}", str(what)[:300], "bad", now)
            q = self._get("alert_mails", [])
            q.append(alerts[-1])
            self._set("alert_mails", q[-50:])
            log.warning("IBKR failure: %s %s: %s", where, symbol or "", what)
        self._set("ibkr_alerts", alerts[-50:])

    def _live_mail(self, o, kind, now):
        """A real-money trade at IBKR (not the paper account's: the simulation's own mails tell those) into
        the mails' queue, as the simulation's are."""
        if o.get("mode") != "live":
            return
        t = {"id": o["key"], "strategy": o["strategy"], "symbol": o["symbol"], "day": o.get("entry_day"), "in": o.get("entry_day"),
             "entry": o.get("entry"), "stop": o["stop"], "stop0": o.get("stop0", o["stop"]), "target": o.get("target"),
             "shares": o.get("filled") or o["qty"], "risk_usd": o.get("risk_usd")}
        if kind == "open":
            self.queue_mails("live", [t], [], now)
        else:
            out = dt.datetime.fromtimestamp(now, NY).date().isoformat()
            t.update(out=out, exit=o.get("exit"), pnl=o.get("pnl"), r=o["pnl"] / o["risk_usd"] if o.get("risk_usd") else None,
                     why=o.get("why"), days=(dt.date.fromisoformat(out) - dt.date.fromisoformat(o["entry_day"])).days if o.get("entry_day") else None)
            self.queue_mails("live", [], [t], now)

    @staticmethod
    def _signals_fresh(book, now):
        """Whether the day's signals are the last close's (sent only then): after 16:00 on a weekday, today's;
        before, the weekday before. Between the close and the evening's data, yesterday's signals are a day old
        and wait (they'd go into tomorrow's open on a stale close)."""
        t = dt.datetime.fromtimestamp(now, NY)
        want = t.date()
        if not (t.weekday() < 5 and t.hour >= 16):
            want -= dt.timedelta(days=1)
            while want.weekday() >= 5:
                want -= dt.timedelta(days=1)
        return str(book.get("last") or "") >= want.isoformat()

    def ibkr_tick(self, now=None, gateway=None):
        """Send the new signals (after each close, for the next open), follow the orders (fills, the
        stop moved as the simulation's, a sale at the next open when the simulation left on time) and
        record each trade beside the simulation's. Paper accounts only."""
        with self.ib_lock:
            return self._ibkr_tick(now, gateway)

    def _ibkr_tick(self, now=None, gateway=None):
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
            # (a minute or two down is a restart's (a deploy, the gateway's own): said only if it lasts)
            since = self.status.setdefault("ibkr_down_since", now)
            what = f"{gw.host}:{gw.port}: {type(e).__name__}: {e}"
            if now - since >= self.DOWN_GRACE:
                self.ibkr_alert("החיבור ל-IB Gateway", what + f" (כבר {int((now - since) // 60)} דקות)", now=now)
            else:
                self.status["ibkr_error"] = f"החיבור ל-IB Gateway: {what}"[:300]
            return None
        if self.status.pop("ibkr_down_since", None) and self.status.get("reconnect"):
            self.status["reconnect"] = dict(self.status["reconnect"], back=now)  # (it came back)
        self.status["reconnect_tries"] = 0
        mode = self.settings().get("ibkr_mode", "paper")
        problem = ibkr.account_problem(accounts, mode)
        if problem:
            self.ibkr_alert("החשבון", problem, now=now)
            return None
        self.status.update(ibkr_error=None, ibkr_account=accounts[0], ibkr_ts=now)
        acc = self._read_account(gw, mode, now)
        book = self.paper_book()
        t = dt.datetime.fromtimestamp(now, NY)
        session = t.weekday() < 5 and (9, 28) <= (t.hour, t.minute) < (16, 0)
        port, orders = self.port(), st["orders"]
        # 1. The simulation's new signals, for the next open (sent only while the market is closed:
        # an on-open order sent during the session would wait for tomorrow's).
        if book and not session and self._signals_fresh(book, now):
            with self.eq.lock:
                prepared = self.eq.prepared
            auto = self.settings().get("ibkr_auto", True)
            for kind, sym, stop, *rest in self.live_signals(book):
                key = f"{kind}:{sym}:{book['last']}"
                if key in orders or kind not in equity.STRATEGIES:
                    continue
                if not auto and key not in st.get("approved", []):
                    continue  # (on approval: it waits on the hub for the owner's yes)
                p = prepared.get(sym)
                ceiling, order = (list(rest) + [None, None])[:2]
                guard = self._live_guard(acc, orders, now) if mode == "live" else None
                if guard:  # (the owner's limits on the real account: recorded, so he sees what was left out and why)
                    orders[key] = {"key": key, "strategy": kind, "symbol": sym, "qty": 0, "stop": round(stop, 2), "order": order,
                                   "ceiling": ceiling and round(ceiling, 2), "target": None, "rr": None, "ids": {},
                                   "mode": mode, "status": "skipped", "sent": now, "signal_day": book["last"], "why": guard}
                    continue
                est = ceiling or (order[1] if order else None) or (p["c"][-1] if p else None)
                if not est or est <= stop:
                    continue
                qty = int(min(port["account"] * port["risk_pct"] / 100 / (est - stop), port["max_position"] / est))
                if mode == "live":  # (real money: only live_size dollars a trade, until the owner says otherwise)
                    qty = min(qty, int(self.settings()["live_size"] // est))
                    risk_cap = self.settings().get("live_risk") or 0
                    if risk_cap:  # (and a loss at the stop of at most live_risk dollars: the owner's cap)
                        qty = min(qty, int(risk_cap // (est - stop)))
                    if qty < 1:
                        orders[key] = {"key": key, "strategy": kind, "symbol": sym, "qty": 0, "stop": round(stop, 2), "order": order,
                                       "ceiling": ceiling and round(ceiling, 2), "target": None, "rr": None, "ids": {}, "mode": mode,
                                       "status": "skipped", "sent": now, "signal_day": book["last"],
                                       "why": f"מניה אחת עולה יותר מ-${self.settings()['live_size']:,.0f}"
                                       if est > self.settings()["live_size"] else
                                       f"מניה אחת מסכנת יותר מ-${self.settings().get('live_risk') or 0:,.0f} עד הסטופ"}
                        continue
                if qty < 1:
                    continue
                P = self.plans([kind])[kind]
                rr = P["rr"]
                if P.get("max_stop_pct") and est - stop > P["max_stop_pct"] * est:
                    stop = est * (1 - P["max_stop_pct"])  # (as the simulation caps it; moved again once in)
                target = est + rr * (est - stop) if rr else None
                try:
                    ids = gw.place_bracket(sym, qty, ceiling, target, stop, order, trail=est - stop)
                except Exception as e:
                    self.ibkr_alert("שליחת הזמנת כניסה", f"{type(e).__name__}: {e}", sym, now)
                    continue
                orders[key] = {"key": key, "strategy": kind, "symbol": sym, "qty": qty, "stop": round(stop, 2), "order": order,
                               "trail": round(est - stop, 2), "mode": mode, "stop0": round(stop, 2),
                               "ceiling": ceiling and round(ceiling, 2), "target": target and round(target, 2), "rr": rr,
                               "ids": ids, "status": "sent", "sent": now, "signal_day": book["last"]}
                log.warning("IBKR: %s %d shares, entry %s (ceiling %s), stop %.2f", sym, qty, order or "at the open", ceiling, stop)
                self.note("ibkr", f"📤 נשלח לאינטראקטיב: {sym}", f"{qty} מניות · {'כסף אמיתי' if mode == 'live' else 'Paper'} · סטופ ${stop:,.2f}", now=now)
        # 2. Where each order stands.
        # (only the orders of the account the gateway is in now: the other kind's stay as they were left)
        live = [o for o in orders.values() if o["status"] in ("sent", "open") and o.get("mode", "paper") == mode]
        if live:
            try:
                known, held = gw.orders(), gw.positions()
            except Exception as e:
                self.ibkr_alert("קריאת מצב ההזמנות", f"{type(e).__name__}: {e}", now=now)
                known, held = None, None
            for o in live if known is not None else []:
                par = known.get(o["ids"]["parent"])
                if o["status"] == "sent":
                    if par and par["status"] == "Filled" or (not par and o["symbol"] in held):
                        price = par["avg_price"] if par else held[o["symbol"]][1]
                        o.update(status="open", entry=round(price, 2), filled=par["filled"] if par else held[o["symbol"]][0],
                                 entry_day=o.get("entry_day") if o.get("manual") else
                                 t.date().isoformat() if session or t.hour >= 16 else o.get("entry_day"))
                        o["entry_day"] = o["entry_day"] or t.date().isoformat()
                        o["risk_usd"] = round(max(0.0, price - o["stop"]) * o["filled"], 2)
                        if o["ids"].get("target") and o["rr"]:
                            o["target"] = round(price + o["rr"] * (price - o["stop"]), 2)  # (from the real entry)
                            try:
                                gw.modify(o["ids"]["target"], lmtPrice=o["target"])
                            except Exception as e:
                                self.ibkr_alert("עדכון היעד אחרי הכניסה", f"{type(e).__name__}: {e}", o["symbol"], now)
                        self._live_mail(o, "open", now)
                    elif par and par["status"] == "Inactive" and not par["filled"]:
                        # (IBKR refused it: a permission, the money, the price... said at once)
                        o.update(status="skipped", why="אינטראקטיב דחתה את ההזמנה")
                        self.ibkr_alert("אינטראקטיב דחתה הזמנת כניסה", par.get("why") or "Inactive", o["symbol"], now)
                    elif par and par["status"] in ("Cancelled", "ApiCancelled") and not par["filled"]:
                        o.update(status="skipped", why=("לא הגיעה למחיר הכניסה עד הסגירה" if o.get("order")
                                                        else "המניה פתחה מעל התקרה (לא רודפים)" if o["ceiling"] else par["status"]))
                    continue
                for leg, why in (("stop", "stop"), ("target", "target"), ("exit", o.get("exit_why") or "time")):
                    x = known.get(o["ids"].get(leg)) if o["ids"].get(leg) else None
                    if x and x["status"] == "Filled":
                        exit_ = x["avg_price"]
                        why = ("trail" if o["stop"] > o["entry"] else "breakeven" if abs(o["stop"] - o["entry"]) < 0.01 else "stop") if leg == "stop" else why
                        fee = 2 * max(1.0, 0.005 * o["filled"])
                        o.update(status="closed", exit=round(exit_, 2), why=why, closed=now,
                                 pnl=round((exit_ - o["entry"]) * o["filled"] - fee, 2))
                        self._live_mail(o, "close", now)
                        break
        # 3. The stop as the simulation's, and the simulation's time exits.
        if book:
            sim_open = {x.get("id"): x for x in book["open"]}
            sim_closed = {x["id"]: x for x in book["trades"][-300:]}
            for o in [x for x in orders.values() if x["status"] == "open" and x.get("mode", "paper") == mode]:
                sid = f"{o['strategy']}:{o['symbol']}:{o.get('entry_day')}"
                s_t = sim_open.get(sid)
                try:
                    if s_t and s_t["stop"] > o["stop"] + 0.005:
                        if o.get("trail"):
                            # (IBKR trails it itself; when the simulation's stop is closer, so is the distance)
                            closer = round(s_t["best"] - s_t["stop"], 2) if s_t.get("best") else None
                            if closer and 0 < closer < o["trail"] - 0.005 and gw.modify(o["ids"]["stop"], auxPrice=closer):
                                o["trail"] = closer
                            o["stop"] = round(s_t["stop"], 2)
                        elif gw.modify(o["ids"]["stop"], auxPrice=round(s_t["stop"], 2)):
                            o["stop"] = round(s_t["stop"], 2)
                    elif sid in sim_closed and sim_closed[sid]["why"] in ("time", "open") and not o.get("exit_sent") and not session:
                        for leg in ("target", "stop"):
                            if o["ids"].get(leg):
                                gw.cancel(o["ids"][leg])
                        o["ids"]["exit"] = gw.sell_at_open(o["symbol"], o["filled"])
                        o["exit_sent"] = now
                except Exception as e:
                    self.ibkr_alert("ניהול טרייד פתוח (הסטופ / יציאה בזמן)", f"{type(e).__name__}: {e}", o["symbol"], now)
        if len(orders) > 400:
            for k in sorted(orders, key=lambda k: orders[k]["sent"])[:len(orders) - 400]:
                if orders[k]["status"] in ("closed", "skipped"):
                    orders.pop(k)
        self._set("ibkr", st)
        return st

    # ---------- the experiments lab (scalp.py): experiment 1, the in-play opening-range break ----------
    LAB_RETRY = 1800  # seconds between two tries of a day's recording

    def lab_due(self, now):
        """No minutes recorded yet, or a weekday's session closed (16:10 New York) and not recorded."""
        days = self.lab_bars.days()
        if not days:
            return True
        t = dt.datetime.fromtimestamp(now, NY)
        return t.weekday() < 5 and (t.hour, t.minute) >= (16, 10) and days[-1] < t.date().isoformat()

    def lab_tick(self, now=None, fetch=None, wait=False):
        """Records the closed days' minutes and runs the experiment again, in a thread of its own (a few
        hundred requests to the source: minutes, not the minute loop's time)."""
        now = now or time.time()
        if self._lab_busy or now - self.status.get("lab_tried", 0) < self.LAB_RETRY or not self.lab_due(now):
            return False
        self.status["lab_tried"] = now
        self._lab_busy = True

        def go():
            try:
                got = scalp.record(self.lab_bars, list(equity.UNIVERSE) + ["SPY", "QQQ"], now,
                                   fetch or scalp.fetch_minutes, pause=0 if fetch else 0.2)
                self.lab_run()
                self.status.update(lab_error=None, lab_recorded=got)
                log.warning("Atlas lab: %d days recorded, the experiment run again", len(got))
            except Exception as e:
                self.status["lab_error"] = f"{type(e).__name__}: {e}"[:300]
                log.warning("Atlas lab: %s", e)
            finally:
                self._lab_busy = False
        if wait:
            go()
        else:
            threading.Thread(target=go, daemon=True, name="atlas-lab").start()
        return True

    def lab_run(self):
        """Every variant over every recorded day, the board, the last day's scan and trades."""
        with self.eq.lock:
            prepared = self.eq.prepared
        days = self.lab_bars.days()
        atr = {}
        for d in days:  # (V7: the daily ATR as of the day before)
            for sym, p in prepared.items():
                i = p["index"].get(d)
                if i and p["atr"][i - 1]:
                    atr.setdefault(d, {})[sym] = p["atr"][i - 1]
        nr7 = {}
        for d in days:  # (V8: the stocks whose range the day before was the narrowest of 7 days)
            for sym, p in prepared.items():
                i = p["index"].get(d)
                if i and i >= 7:
                    rng = [p["h"][j] - p["l"][j] for j in range(i - 7, i)]
                    if rng[-1] <= min(rng):
                        nr7.setdefault(d, set()).add(sym)
        res, run = scalp.run_all(self.lab_bars, atr, symbols=set(equity.UNIVERSE), nr7=nr7)
        res2, run2 = intraday.run_all(self.lab_bars)
        scan, last = [], run[-1] if run else None
        if last:
            i = days.index(last)
            hist = [self.lab_bars.load(x) for x in days[max(0, i - scalp.BASE_DAYS):i]]
            net = scalp.scan(self.lab_bars.load(last), scalp.baselines(hist), scalp.VARIANTS["V1"])
            scan = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in net.values()]
        recent = sorted(([dict(t, variant=k) for k, ts in res.items() for t in ts if t["day"] == last] if last else []),
                        key=lambda t: (t["in"], t["variant"]))
        last2 = run2[-1] if run2 else None
        recent2 = sorted(([dict(t, variant=k) for k, ts in res2.items() for t in ts if t["day"] == last2] if last2 else []),
                         key=lambda t: (t["in"], t["variant"]))
        out = {"ts": time.time(), "board": scalp.board(res, run), "days": run, "recorded": days,
               "scan": scan, "last": last, "recent": recent[:200],
               # (experiment 2: the index ETFs' intraday strategies from the research, 28.10.2026)
               "board2": intraday.board(res2, run2), "days2": run2, "last2": last2, "recent2": recent2[:100]}
        self._set("lab", out)
        return out

    def lab_view(self):
        v = self._get("lab") or {}
        return dict(v, recorded=len(v.get("recorded") or []), first=(v.get("recorded") or [None])[0],
                    busy=self._lab_busy, error=self.status.get("lab_error"))

    # ---------- the streak study (streaks.py): do every strategy's wins and losses come in runs? ----------
    # Every strategy at the system's rules over the whole history (each as an account of its own, so one
    # doesn't crowd out another), every trade in R. The signal: what the trades that had already closed
    # said over the last 5 to 40 trading days (set before any data); a year of trades before a verdict.
    STREAK_LOOKBACKS = (5, 10, 20, 40)
    STREAK_NEED = 250

    def streaks_tick(self, wait=False):
        """Once a data version (new days), in a thread of its own (four backtests over the history)."""
        with self.eq.lock:
            prepared, version = self.eq.prepared, self.eq.version
        if (self._streaks_busy or not prepared.get("SPY") or self.status.get("streaks_version") == version
                or (self._get("streaks") or {}).get("version") == version):
            return False
        self._streaks_busy = True
        self.status["streaks_version"] = version  # (a failure waits for the next data, not a minute)

        def go():
            try:
                self.streaks_run(prepared, version)
                self.status["streaks_error"] = None
            except Exception as e:
                self.status["streaks_error"] = f"{type(e).__name__}: {e}"[:300]
                log.warning("Atlas streaks: %s", e)
            finally:
                self._streaks_busy = False
        if wait:
            go()
        else:
            threading.Thread(target=go, daemon=True, name="atlas-streaks").start()
        return True

    def streaks_run(self, prepared, version=None):
        index = prepared["SPY"]["index"]
        plans, port = self.plans(list(equity.STRATEGIES)), self.port()
        events = []
        for k, P in plans.items():
            for t in equity.portfolio(prepared, {k: P}, port)["trades"]:
                if t["why"] != "open" and t["r"] is not None and t["in"] in index and t["out"] in index:
                    events.append((index[t["in"]], index[t["out"]], t["r"], 1.0))
        out = streaks.study(*streaks.from_events(events), self.STREAK_LOOKBACKS, 1, self.STREAK_NEED, min_recent=5)
        out = dict(out, ts=time.time(), version=version, unit="days",
                   first=prepared["SPY"]["day"][0], last=prepared["SPY"]["day"][-1])
        self._set("streaks", out)
        return out

    # ---------- the size and the number open at once (sizing.py, the owner's questions of 29.10.2026) ----------
    # Every strategy in the simulation together, as one account of the size in the settings, over the whole
    # history; again with new data once a week, at once when the account's size, the strategies or the bound
    # change. An answer that held on the locked test becomes every strategy's (sizing_auto).
    SIZING_AGAIN = 7 * 86400

    def sizing_key(self):
        s = self.settings()
        kinds = sorted(k for k, on in s["strategies"].items() if on) or sorted(equity.STRATEGIES)
        return f"{s['account']:.0f}|{','.join(kinds)}|{s.get('sizing_dd', 20.0):g}"

    def sizing_tick(self, now=None, wait=False, apart=True):
        now = now or time.time()
        with self.eq.lock:
            prepared = self.eq.prepared
        last, key = self._get("sizing") or {}, self.sizing_key()
        tried = self.status.get("sizing_tried")  # (when, the key): a failure waits an hour before the same again
        if (self._sizing_busy or not prepared.get("SPY")
                or (last.get("key") == key and now - (last.get("ts") or 0) < self.SIZING_AGAIN)
                or (tried and tried[1] == key and 0 <= now - tried[0] < 3600)):
            return False
        self._sizing_busy = True
        self.status["sizing_tried"] = (now, key)
        s = self.settings()
        kinds = [k for k, on in s["strategies"].items() if on] or list(equity.STRATEGIES)
        plans, port = self.plans(kinds), self.port()

        def go():
            try:
                dd = (s.get("sizing_dd") or 20.0) / 100
                out = (sizing.study_apart(prepared, plans, port, dd) if apart
                       else sizing.study(prepared, plans, port, equity.portfolio, dd))
                out = dict(out, ts=now, key=key, first=prepared["SPY"]["day"][0], last=prepared["SPY"]["day"][-1])
                out["applied"] = self.sizing_apply(out)
                self._set("sizing", out)
                self.status["sizing_error"] = None
            except Exception as e:
                self.status["sizing_error"] = f"{type(e).__name__}: {e}"[:300]
                log.warning("Atlas sizing: %s", e)
            finally:
                self._sizing_busy = False
        if wait:
            go()
        else:
            threading.Thread(target=go, daemon=True, name="atlas-sizing").start()
        return True

    def sizing_apply(self, out):
        """The research's answers into the settings every strategy reads (the size, then the number open), when
        they held on the locked test and the owner leaves it to the system: what changed."""
        s = self.settings()
        if not s.get("sizing_auto", True):
            return None
        changed = {}
        A = out["size"].get("answer") if out["size"]["state"] == "found" else None
        if A:
            pos = round(A["max_position"] / 100) * 100
            if abs(s["risk_pct"] - A["risk_pct"]) > 1e-9:
                changed["risk_pct"] = (s["risk_pct"], A["risk_pct"])
                s["risk_pct"] = A["risk_pct"]
            if abs(s["max_position"] - pos) > 1:
                changed["max_position"] = (s["max_position"], pos)
                s["max_position"] = float(pos)
        B = out["slots"].get("answer") if out["slots"]["state"] == "found" else None
        if B and int(s["max_open"]) != B["max_open"]:
            changed["max_open"] = (int(s["max_open"]), B["max_open"])
            s["max_open"] = B["max_open"]
        if changed:
            self._set("settings", s)
            names = {"risk_pct": "סיכון לטרייד", "max_position": "פוזיציה מקסימלית", "max_open": "טריידים במקביל"}
            fmt = lambda k, v: f"{v:g}%" if k == "risk_pct" else f"${v:,.0f}" if k == "max_position" else f"{v}"
            self.note("sizing", "📐 המחקר עדכן את כל האסטרטגיות",
                      " · ".join(f"{names[k]}: {fmt(k, a)} ← {fmt(k, b)}" for k, (a, b) in changed.items()))
        return {k: list(v) for k, v in changed.items()} or None

    def sizing_view(self):
        v = self._get("sizing")
        if not v:
            return None
        return dict(v, busy=self._sizing_busy, error=self.status.get("sizing_error"), auto=self.settings().get("sizing_auto", True),
                    current=self.sizing_key() == v.get("key"))

    # ---------- the owner's hand on a trade (My trades' cards) ----------
    def _sim_open(self, trade_id):
        book = self.paper_book()
        t = next((x for x in (book or {}).get("open", []) if x.get("id") == trade_id), None)
        if t is None:
            raise ValueError("הטרייד הזה כבר לא פתוח בסימולציה")
        return book, t

    def _price_now(self, sym):
        q = self.quotes.get(sym)
        if q:
            return q["price"]
        with self.eq.lock:
            p = self.eq.prepared.get(sym)
        if not p:
            raise ValueError("אין מחיר עדכני למניה הזאת")
        return p["c"][-1]

    def _gateway(self):
        """The IB Gateway, connected, on a paper account; else why not (ValueError)."""
        from . import ibkr
        if not self.settings().get("ibkr_on"):
            raise ValueError("החיבור לאינטראקטיב כבוי: מדליקים אותו בטאב הזה, למטה")
        gw = self.ibkr or ibkr.IbGateway()
        self.ibkr = gw
        try:
            accounts = gw.connect()
        except Exception as e:
            raise ValueError(f"אין חיבור ל-IB Gateway: {type(e).__name__}: {e}"[:300])
        problem = ibkr.account_problem(accounts, self.settings().get("ibkr_mode", "paper"))
        if problem:
            raise ValueError(problem)
        return gw

    def _ib_qty(self, shares, price):
        """The shares to send: the simulation's on paper; on real money only live_size dollars' worth."""
        S = self.settings()
        if S.get("ibkr_mode") != "live":
            return int(shares)
        qty = min(int(shares), int(S["live_size"] // price)) if price > 0 else 0
        if qty < 1:
            raise ValueError(f"מניה אחת עולה יותר מ-${S['live_size']:,.0f} (הסכום לטרייד בכסף אמיתי)")
        return qty

    def _session(self, now):
        t = dt.datetime.fromtimestamp(now, NY)
        return t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0)

    def _send_bracket(self, gw, st, key, t, qty, now, why):
        """One bracket for qty shares of a simulation trade, at the market now during the session, at the
        next open otherwise; its stop and target the simulation's (the stop then follows the simulation's)."""
        entry = ("market", None) if self._session(now) else None
        trail = round(max(0.01, t.get("best", t["entry"]) - t["stop"]), 2)  # (from where the simulation's is now)
        ids = gw.place_bracket(t["symbol"], qty, None, t.get("target"), t["stop"], entry, trail=trail)
        st["orders"][key] = {"key": key, "strategy": t["strategy"], "symbol": t["symbol"], "qty": qty, "trail": trail,
                             "stop": round(t["stop"], 2), "order": None, "ceiling": None,
                             "target": t.get("target") and round(t["target"], 2), "rr": None, "ids": ids, "status": "sent",
                             "sent": now, "signal_day": t["day"], "entry_day": t["day"], "manual": why,
                             "mode": self.settings().get("ibkr_mode", "paper")}
        log.warning("IBKR (the owner's): %s %d shares (%s)", t["symbol"], qty, why)
        return st["orders"][key]

    def send_live(self, trade_id, now=None):
        """The owner: this open trade of the simulation into the IBKR paper account too, the same shares,
        stop and target (at the market now during the session, else at the next open)."""
        now = now or time.time()
        _, t = self._sim_open(trade_id)
        with self.ib_lock:
            gw = self._gateway()
            st = self._get("ibkr", {"orders": {}})
            mine = [o for o in st["orders"].values() if f"{o['strategy']}:{o['symbol']}:{o.get('entry_day')}" == trade_id
                    and o["status"] in ("sent", "open")]
            if mine:
                raise ValueError("הטרייד הזה כבר בחשבון של אינטראקטיב")
            qty = self._ib_qty(t["shares"], self._price_now(t["symbol"]))
            try:
                o = self._send_bracket(gw, st, f"manual:{trade_id}", t, qty, now, "live")
            except Exception as e:
                self.ibkr_alert("שליחה ידנית (הכנס למסחר חי)", f"{type(e).__name__}: {e}", t["symbol"], now)
                raise ValueError(f"אינטראקטיב: {type(e).__name__}: {e}"[:300])
            self._set("ibkr", st)
        return o

    def add_to(self, trade_id, pct, now=None):
        """The owner adds to an open trade of the simulation: a quarter, a half or the same again of its
        shares, at the price now, the stop and target as they are; and in the IBKR account too when the
        trade is there (a bracket of its own for the shares added)."""
        now = now or time.time()
        pct = float(pct)
        if pct not in (0.25, 0.5, 1.0):
            raise ValueError("מוסיפים 25%, 50% או 100%")
        with self.lock:
            book, t = self._sim_open(trade_id)
            P = equity.plan_for(t["strategy"])
            price = self._price_now(t["symbol"]) * (1 + P["slippage"])
            if price <= t["stop"]:
                raise ValueError("המחיר עכשיו מתחת לסטופ: לא מוסיפים")
            add = max(1, int(t["shares"] * pct))
            fee = equity._fee(P, add)
            if add * price + fee > book["cash"]:
                raise ValueError(f"אין מספיק מזומן בסימולציה (${book['cash']:,.0f})")
            book["cash"] -= add * price + fee
            t.setdefault("dist", equity.trail_distance(t))  # (the stop keeps trailing at the distance it started with)
            t["entry"] = (t["entry"] * t["shares"] + price * add) / (t["shares"] + add)
            t["shares"] += add
            t["fees"] = t.get("fees", 0) + fee
            t["risk"] = t["entry"] - t["stop"]
            t["risk_usd"] = max(0.0, t["risk"]) * t["shares"]
            t.setdefault("adds", []).append({"ts": now, "shares": add, "price": round(price, 2), "pct": pct})
            self._set("paper", book)
        live = None
        st = self._get("ibkr", {"orders": {}})
        linked = [o for o in st["orders"].values() if f"{o['strategy']}:{o['symbol']}:{o.get('entry_day')}" == trade_id
                  and o["status"] in ("sent", "open") and o.get("mode", "paper") == self.settings().get("ibkr_mode", "paper")]
        if linked:
            with self.ib_lock:
                try:
                    gw = self._gateway()
                    st = self._get("ibkr", {"orders": {}})
                    # (real money: the same share of what is in the account, not of the simulation's)
                    ib_add = max(1, int(sum(o.get("filled") or o["qty"] for o in linked) * pct)) \
                        if self.settings().get("ibkr_mode") == "live" else add
                    live = self._send_bracket(gw, st, f"add:{trade_id}:{int(now)}", t, ib_add, now, f"add {pct:.0%}")
                    self._set("ibkr", st)
                except Exception as e:
                    self.ibkr_alert("הוספה לטרייד (25/50/100%)", str(e), t["symbol"], now)
                    live = {"error": str(e)[:300]}
        return {"added": add, "price": round(price, 2), "shares": t["shares"], "entry": round(t["entry"], 2), "live": live}

    def close_live(self, symbol, now=None):
        """The owner closes a position in the account: its resting stops and targets cancelled, all its
        shares sold at the market now (at the next open outside the session); the system's trades in it
        close at that sale's price once it fills (the minute's look)."""
        now = now or time.time()
        with self.ib_lock:
            gw = self._gateway()
            mode = self.settings().get("ibkr_mode", "paper")
            try:
                done = gw.close_position(symbol, self._session(now))
            except ValueError:
                raise
            except Exception as e:
                self.ibkr_alert("סגירת פוזיציה", f"{type(e).__name__}: {e}", symbol, now)
                raise ValueError(f"אינטראקטיב: {type(e).__name__}: {e}"[:300])
            st = self._get("ibkr", {"orders": {}})
            for o in st["orders"].values():
                if o["symbol"] == symbol and o["status"] == "open" and o.get("mode", "paper") == mode:
                    o["ids"]["exit"], o["exit_sent"], o["exit_why"] = done["order"], now, "manual"
            self._set("ibkr", st)
            self._read_account(gw, mode, now)
        self.note("ibkr", f"📤 נשלחה סגירה: {symbol}", f"{done['shares']:g} מניות · {'עכשיו' if self._session(now) else 'בפתיחה'}", now=now)
        log.warning("IBKR (the owner's): close %s, %g shares", symbol, done["shares"])
        return {**done, "at": "now" if self._session(now) else "open"}

    STOP_TRAILS = (1.0, 2.0, 3.0, 4.0, 5.0)  # percent: the trailing stops offered for a position without one

    def set_stop(self, symbol, kind, pct, now=None):
        """The owner puts a stop on a position that has none (his own buys, 28.10.2026): fixed, pct percent
        under the price now; or trailing pct percent (1-5) under its best price from now on. At IBKR, good
        till cancelled, on all the position's shares."""
        now = now or time.time()
        pct = float(pct)
        if kind == "trail" and pct not in self.STOP_TRAILS:
            raise ValueError("סטופ נגרר: 1%, 2%, 3%, 4% או 5%")
        if kind == "fixed" and not 0.1 <= pct <= 50:
            raise ValueError("סטופ: בין 0.1% ל-50% מתחת למחיר")
        if kind not in ("fixed", "trail"):
            raise ValueError("bad request")
        with self.ib_lock:
            gw = self._gateway()
            mode = self.settings().get("ibkr_mode", "paper")
            acc = self._read_account(gw, mode, now) or {}
            pos = next((p for p in acc.get("positions") or [] if p["symbol"] == symbol), None)
            if pos is None:
                raise ValueError(f"אין פוזיציה ב-{symbol} בחשבון")
            if pos.get("stop"):
                raise ValueError(f"כבר יש סטופ על {symbol} (${pos['stop']:,.2f})")
            q = self.quotes.get(symbol)
            price = q["price"] if q and now - q.get("ts", 0) <= 120 else pos["price"]
            stop = round(price * (1 - pct / 100), 2)
            try:
                oid = gw.place_stop(symbol, pos["shares"], stop, pct if kind == "trail" else None)
            except Exception as e:
                self.ibkr_alert("הצבת סטופ", f"{type(e).__name__}: {e}", symbol, now)
                raise ValueError(f"אינטראקטיב: {type(e).__name__}: {e}"[:300])
            self._read_account(gw, mode, now)
        what = f"סטופ נגרר {pct:g}%" if kind == "trail" else f"סטופ ב-${stop:,.2f} ({pct:g}% מתחת)"
        self.note("ibkr", f"🛡️ הוצב סטופ: {symbol}", what, now=now)
        log.warning("IBKR (the owner's): stop on %s: %s", symbol, what)
        return {"order": oid, "stop": stop, "kind": kind, "pct": pct}

    TEST_HOLD = 25  # seconds the test share is held

    def test_round_trip(self, symbol="META", now=None, hold=None, wait=True):
        """The owner's check of the whole way (28.10.2026): one share bought at the market, sold the same way
        `hold` seconds after it filled, through the same record, mails and screens as every trade (strategy
        "test"). During the session only. Runs in the background; its steps are in status["ibkr_test"]."""
        now = now or time.time()
        symbol = (symbol or "META").upper().strip()
        if not re.fullmatch(r"[A-Z.]{1,8}", symbol):
            raise ValueError("סימבול לא תקין")
        if not self._session(now):
            raise ValueError("הבדיקה רק בשעות המסחר (9:30–16:00 שעון ניו יורק): מחוץ להן הוראת שוק לא מתבצעת")
        T = self.status.get("ibkr_test")
        if T and T.get("step") not in ("done", "error"):
            raise ValueError("בדיקה אחרת עוד רצה")
        gw = self._gateway()
        hold = self.TEST_HOLD if hold is None else hold
        T = self.status["ibkr_test"] = {"symbol": symbol, "step": "buy_sent", "started": now, "hold": hold}
        run = lambda: self._test_run(gw, symbol, T, hold)
        if wait:
            run()
        else:
            threading.Thread(target=run, daemon=True, name="ibkr-test").start()
        return T

    def _test_fill(self, gw, oid, T, limit=30):
        for _ in range(limit * 2):
            x = (gw.orders() or {}).get(oid)
            if x and x["status"] == "Filled":
                return x["avg_price"]
            if x and x["status"] in ("Cancelled", "ApiCancelled", "Inactive"):
                raise ValueError(f"אינטראקטיב: {x['status']} {x.get('why') or ''}".strip())
            time.sleep(0.5)
        raise ValueError(f"לא התמלא תוך {limit} שניות")

    def _test_run(self, gw, symbol, T, hold):
        mode = self.settings().get("ibkr_mode", "paper")
        key = f"test:{symbol}:{int(T['started'])}"
        try:
            with self.ib_lock:
                oid = gw.market(symbol, "BUY", 1)
                st = self._get("ibkr", {"orders": {}})
                st["orders"][key] = {"key": key, "strategy": "test", "symbol": symbol, "qty": 1, "stop": 0.0, "order": None,
                                     "ceiling": None, "target": None, "rr": None, "ids": {"parent": oid}, "status": "sent",
                                     "sent": time.time(), "signal_day": None, "mode": mode, "manual": "test"}
                self._set("ibkr", st)
            T.update(buy_order=oid)
            price = self._test_fill(gw, oid, T)
            now = time.time()
            with self.ib_lock:
                st = self._get("ibkr", {"orders": {}})
                o = st["orders"][key]
                if o["status"] == "sent":
                    o.update(status="open", entry=round(price, 2), filled=1, entry_day=dt.datetime.fromtimestamp(now, NY).date().isoformat())
                    self._set("ibkr", st)
                    self._live_mail(o, "open", now)
            T.update(step="holding", bought=round(price, 2), bought_at=now)
            time.sleep(hold)
            with self.ib_lock:
                sid = gw.market(symbol, "SELL", 1)
                st = self._get("ibkr", {"orders": {}})
                o = st["orders"][key]
                o["ids"]["exit"], o["exit_sent"], o["exit_why"] = sid, time.time(), "test"
                self._set("ibkr", st)
            T.update(step="sell_sent", sell_order=sid)
            exit_ = self._test_fill(gw, sid, T)
            now = time.time()
            with self.ib_lock:
                st = self._get("ibkr", {"orders": {}})
                o = st["orders"][key]
                if o["status"] == "open":
                    o.update(status="closed", exit=round(exit_, 2), why="test", closed=now,
                             pnl=round(exit_ - o["entry"] - 2 * 1.0, 2))  # (IBKR's least commission, each way)
                    self._set("ibkr", st)
                    self._live_mail(o, "close", now)
                pnl = o.get("pnl")
            T.update(step="done", sold=round(exit_, 2), sold_at=now, pnl=pnl)
            self.note("ibkr", f"✅ בדיקת מערכת: {symbol}", f"נקנתה ב-${price:,.2f}, נמכרה ב-${exit_:,.2f}", "good", now=now)
            log.warning("IBKR test: %s bought %.2f sold %.2f", symbol, price, exit_)
        except Exception as e:
            T.update(step="error", error=f"{type(e).__name__}: {e}"[:300])
            self.ibkr_alert("בדיקת מערכת (קנייה ומכירה של מניה אחת)", str(e), symbol)

    def add_live(self, symbol, pct, now=None):
        """The owner adds to a position in the account: a quarter, a half or the same again of its shares,
        at the market now (at the next open outside the session), with a stop of its own where the
        position's stands (and the target, when it has one); followed like every order from then on."""
        now = now or time.time()
        pct = float(pct)
        if pct not in (0.25, 0.5, 1.0):
            raise ValueError("מוסיפים 25%, 50% או 100%")
        with self.ib_lock:
            gw = self._gateway()
            mode = self.settings().get("ibkr_mode", "paper")
            acc = self._read_account(gw, mode, now) or {}
            pos = next((p for p in acc.get("positions") or [] if p["symbol"] == symbol), None)
            if pos is None:
                raise ValueError(f"אין פוזיציה ב-{symbol} בחשבון")
            st = self._get("ibkr", {"orders": {}})
            mine = [o for o in st["orders"].values() if o["symbol"] == symbol and o["status"] == "open"
                    and o.get("mode", "paper") == mode]
            stop = pos.get("stop") or max((o["stop"] for o in mine), default=None)
            price = pos.get("price") or self._price_now(symbol)
            if not stop:
                raise ValueError("אין סטופ על הפוזיציה הזאת: לא מוסיפים בלי סטופ")
            if price <= stop:
                raise ValueError("המחיר עכשיו מתחת לסטופ: לא מוסיפים")
            add = max(1, int(pos["shares"] * pct))
            base = mine[0] if mine else {}
            t = {"symbol": symbol, "strategy": base.get("strategy") or "manual", "stop": stop, "entry": price, "best": price,
                 "target": base.get("target"), "day": dt.datetime.fromtimestamp(now, NY).date().isoformat()}
            try:
                o = self._send_bracket(gw, st, f"add:{mode}:{symbol}:{int(now)}", t, add, now, f"add {pct:.0%}")
            except Exception as e:
                self.ibkr_alert("הגדלת פוזיציה (25/50/100%)", f"{type(e).__name__}: {e}", symbol, now)
                raise ValueError(f"אינטראקטיב: {type(e).__name__}: {e}"[:300])
            self._set("ibkr", st)
        self.note("ibkr", f"📤 נשלחה הגדלה: {symbol}", f"+{add} מניות ({pct:.0%}) · סטופ ${stop:,.2f}", now=now)
        return {"added": add, "stop": round(stop, 2), "at": "now" if self._session(now) else "open", "order": o["ids"]}

    def decide_signal(self, key, yes, now=None):
        """The owner, on approval: this signal of the simulation goes to the IBKR account (at the next look,
        within the minute, if the session hasn't opened yet) or not."""
        book = self.paper_book() or {}
        keys = {f"{k}:{sym}:{book.get('last')}" for k, sym, *_ in self.live_signals(book)}
        if key not in keys:
            raise ValueError("האות הזה כבר לא ממתין (הסימולציה כבר נכנסה או ויתרה)")
        with self.ib_lock:
            st = self._get("ibkr", {"orders": {}})
            for k in ("approved", "declined"):
                st[k] = [x for x in st.get(k, []) if x != key and x.rsplit(":", 1)[-1] >= str(book.get("last"))]
            st["approved" if yes else "declined"].append(key)
            self._set("ibkr", st)
        if yes:
            threading.Thread(target=self.ibkr_tick, args=(now,), daemon=True, name="atlas-ibkr-now").start()
        return {"ok": True}

    def live_signals(self, book=None):
        """The signals real money takes at the next open (the owner, 28.10.2026: Atlas' book in the account is
        its own): the simulation's own first, then the rest of the day's signals, best first, as many as the
        account's room for Atlas (its own limit, else the simulation's; its own positions and the orders on
        their way count). Not bound by the simulation's room: a full simulation no longer starves real money."""
        book = book if book is not None else (self.paper_book() or {})
        if not book.get("last"):
            return []
        with self.eq.lock:
            prepared = self.eq.prepared
        on = [k for k, st in self.lanes()["live"].items() if st == "on"]  # (the strategies real money runs)
        if not prepared.get("SPY") or book["last"] not in prepared["SPY"]["index"]:
            return [tuple(x) for x in book.get("pending", []) if x[0] in on]
        S = self.settings()
        orders = self._get("ibkr", {"orders": {}})["orders"].values()
        mode = S.get("ibkr_mode", "paper")
        mine = [o for o in orders if o.get("mode", "paper") == mode and o["status"] in ("open", "sent")]
        held = {o["symbol"] for o in mine}  # (the mentor's too: never two hands on one stock)
        room = int(S.get("live_max_open") or self.port()["max_open"]) - len({o["symbol"] for o in mine if o.get("strategy") != "mentor"})
        sim = [tuple(x) for x in book.get("pending", []) if x[1] not in held and x[0] in on]
        rest = equity.signals_for(prepared, book["last"], self.plans(on), self.port(), held | {x[1] for x in sim}) if on else []
        keys = {f"{o['strategy']}:{o['symbol']}:{book['last']}" for o in orders}  # (already decided: keep them listed)
        out = sim + rest
        return [x for x in out[:max(0, room)]] + [x for x in out[max(0, room):] if f"{x[0]}:{x[1]}:{book['last']}" in keys]

    def proposals(self):
        """The signals for the next open real money takes (live_signals), each with what the owner decided."""
        book = self.paper_book() or {}
        st = self._get("ibkr", {"orders": {}})
        out = []
        for kind, sym, stop, *rest in self.live_signals(book):
            key = f"{kind}:{sym}:{book.get('last')}"
            ceiling, order = (list(rest) + [None, None])[:2]
            o = st["orders"].get(key)
            out.append({"key": key, "strategy": kind, "symbol": sym, "stop": round(stop, 2), "ceiling": ceiling and round(ceiling, 2),
                        "order": order, "sent": bool(o), "status": o and o["status"],
                        "decision": "yes" if key in st.get("approved", []) else "no" if key in st.get("declined", []) else None})
        return out

    def _read_account(self, gw, mode, now):
        """The real account's figures (ibkr.IbGateway.account), kept for the page, with its value over time
        (a reading an hour, for the week's and the month's change)."""
        from . import ibkr
        try:
            acc = gw.account()
        except Exception as e:
            log.warning("IBKR account: %s", e)
            return None
        if acc.get("account") and ibkr.account_problem([acc["account"]], mode):
            return None  # (not the kind of account picked: never shown as it)
        acc = dict(acc, ts=now, mode=mode)
        self.status["ibkr_acct"] = acc
        self.status["ibkr_ts"] = now  # (an answer from the account: the connection is up now)
        if acc.get("net_liq") is not None:
            hist = self._get("ibkr_values", {})
            pts = hist.setdefault(mode, [])
            if not pts or now - pts[-1][0] >= 3600 or pts[-1][2] != acc["account"]:
                pts.append([now, round(acc["net_liq"], 2), acc["account"]])
                del pts[:-5000]
                self._set("ibkr_values", hist)
        return acc

    def _live_guard(self, acc, orders, now):
        """Why real money takes no new trade now (None: it can): the owner's daily loss and open limits."""
        s = self.settings()
        loss, most = s.get("live_daily_loss") or 0, s.get("live_max_open") or 0
        # (Atlas' own book: the owner's own positions and losses in the account are never its limit)
        L = self.live_book(now)
        mine = (L or {}).get("atlas")
        today = mine["today"] if mine else None
        if loss and today is not None and today <= -loss:
            day = dt.datetime.fromtimestamp(now, NY).date().isoformat()
            if self.status.get("ibkr_loss_day") != day:
                self.status["ibkr_loss_day"] = day
                self.ibkr_alert("הפסד יומי מקסימלי", f"היום ${-today:,.2f} בהפסד בטריידים של אטלס (המגבלה ${loss:,.0f}):"
                                " לא נכנסים לטריידים חדשים עד מחר. הפתוחים ממשיכים עם הסטופ שלהם.", now=now)
            return f"הגעת להפסד היומי המקסימלי (${loss:,.0f})"
        if most:
            held = len((mine or {}).get("positions") or []) + sum(
                1 for o in orders.values() if o.get("mode") == "live" and o["status"] == "sent" and o.get("strategy") != "mentor")
            if held >= most:
                return f"כבר {held} פוזיציות של אטלס בחשבון (המגבלה {most})"
        return None

    CONNECTED_WITHIN = 180  # seconds: the last good reading of the account, for "connected"

    def live_book(self, now=None):
        """The real account, for the page while connected with real money: what the strip, the day's summary
        and the dashboard's lists show then (the simulation's figures stay in the simulation tab). None
        when not connected to the real account (or its last reading is over 10 minutes old)."""
        now = now or time.time()
        s, acc = self.settings(), self.status.get("ibkr_acct")
        if not (s.get("ibkr_on") and s.get("ibkr_mode") == "live"):
            return None
        # (with real money on, the dashboard is the real account's even when the connection is down: its last
        # reading, marked as not connected, never the simulation's figures in its place)
        connected = bool(acc and acc.get("mode") == "live" and now - acc.get("ts", 0) <= self.CONNECTED_WITHIN
                         and not self.status.get("ibkr_error"))
        state = {"connected": connected, "checked": acc and acc.get("ts"), "error": self.status.get("ibkr_error"),
                 "every": self.IBKR_EVERY}
        if not (acc and acc.get("mode") == "live" and acc.get("net_liq") is not None):
            return dict(state, never=True)
        value = acc["net_liq"]
        pts = [p for p in self._get("ibkr_values", {}).get("live", []) if p[2] == acc["account"]]

        def since(days):  # (the value then: the last reading at or before, else the first there is)
            if not pts:
                return None
            cut = now - days * 86400
            then = next((p[1] for p in reversed(pts) if p[0] <= cut), pts[0][1])
            return value - then
        orders = [o for o in self._get("ibkr", {"orders": {}})["orders"].values() if o.get("mode") == "live"]
        by_symbol = {o["symbol"]: o for o in sorted(orders, key=lambda o: o["sent"]) if o["status"] == "open"}
        today = dt.datetime.fromtimestamp(now, NY).date().isoformat()
        day_of = lambda ts: dt.datetime.fromtimestamp(ts, NY).date().isoformat() if ts else None
        positions, moved = [], 0.0
        avg = lambda xs: sum((x.get("entry") or 0) * (x.get("filled") or 0) for x in xs) / sum(x.get("filled") or 0 for x in xs) \
            if sum(x.get("filled") or 0 for x in xs) else None
        for p in acc.get("positions") or []:
            q = self.quotes.get(p["symbol"])
            prev = q and q.get("prev_close")
            if q and q.get("source") == "ibkr" and now - q.get("ts", 0) <= 60 and p["shares"]:
                # (IBKR's account figures move every few minutes; its price stream, every second)
                moved += (q["price"] - p["price"]) * p["shares"]
                p = {**p, "price": q["price"], "value": q["price"] * p["shares"],
                     "unrealized": (q["price"] - p["avg_cost"]) * p["shares"]}
            mine = [x for x in orders if x["symbol"] == p["symbol"] and x["status"] == "open"]
            # (three hands in one account: Atlas' orders, the mentor's (mentor.py) and the owner's own buys)
            atl = [x for x in mine if x.get("strategy") != "mentor"]
            men = [x for x in mine if x.get("strategy") == "mentor"]
            o = max(atl or men, key=lambda x: x["sent"]) if mine else None
            stop = p.get("stop") or (o and o.get("stop"))
            ours_a, ours_m = sum(x.get("filled") or 0 for x in atl), sum(x.get("filled") or 0 for x in men)
            ours = ours_a + ours_m
            sys_sh = min(ours_a, p["shares"])
            men_sh = min(ours_m, max(0.0, p["shares"] - sys_sh))
            positions.append({**p, "strategy": o and o["strategy"], "day": o and o.get("entry_day"), "stop": stop,
                              "prev_close": prev, "sys_shares": sys_sh, "sys_cost": avg(atl), "man_shares": p["shares"] - sys_sh - men_sh,
                              "atlas_strategy": atl and max(atl, key=lambda x: x["sent"])["strategy"],
                              "atlas_day": atl and max(atl, key=lambda x: x["sent"]).get("entry_day"),
                              "atlas_stop": max((x["stop"] for x in atl), default=None),
                              "men_shares": men_sh, "men_cost": avg(men), "men_stop": max((x["stop"] for x in men), default=None),
                              "men_day": men and max(men, key=lambda x: x["sent"]).get("entry_day"),
                              "men_target": next((x.get("target") for x in men if x.get("target")), None),
                              "men_units": round(sum(x.get("units") or 0 for x in men), 3),
                              # (the stock's own move today, from yesterday's close; and who bought the shares)
                              "day_pct": p["price"] / prev - 1 if prev else None,
                              "day_usd": (p["price"] - prev) * p["shares"] if prev else None,
                              "source": ("mentor" if not ours_a else "system" if not ours_m else "mixed")
                              if ours >= p["shares"] - 1e-6 and ours else "mixed" if ours else "manual",
                              "risk": (max(0.0, (p["avg_cost"] - stop) * p["shares"]) if stop else abs(p["value"])),
                              # (what the system bought, against what IBKR says the account holds)
                              "ours": ours,
                              "closing": any(x.get("exit_sent") for x in mine),
                              "adding": sum(1 for x in orders if x["symbol"] == p["symbol"] and x["status"] == "sent")})
        closed = [o for o in orders if o["status"] == "closed" and day_of(o.get("closed")) == today and o.get("strategy") != "mentor"]
        entered = [o for o in orders if o["status"] in ("open", "closed") and o.get("entry_day") == today and o.get("strategy") != "mentor"]
        first = pts[0][0] if pts else now
        value += moved
        books = self._split_books(positions, orders, now, today, day_of)
        return {**state, **books, "account": acc["account"], "ts": acc["ts"], "value": value, "cash": acc.get("cash"),
                "buying_power": acc.get("buying_power"), "today": acc.get("daily"),
                "today_base": value - (acc.get("daily") or 0.0),
                "week": since(7), "month": since(30), "total": value - pts[0][1] if pts else 0.0,
                "since": day_of(first), "open_risk": sum(p["risk"] for p in positions), "positions": positions,
                "entered": len(entered), "entered_open": sum(1 for o in entered if o["status"] == "open"),
                "closed": [{k: o.get(k) for k in ("symbol", "strategy", "entry", "exit", "filled", "pnl", "why")} for o in closed],
                "realized": books["atlas"]["realized_today"],
                "floating_today": sum(p["unrealized"] for p in books["atlas"]["positions"] if p["day"] == today),
                "waiting": sum(1 for o in orders if o["status"] == "sent" and o.get("strategy") != "mentor"),
                "limits": {"daily_loss": s.get("live_daily_loss") or 0, "max_open": s.get("live_max_open") or 0}}

    def _split_books(self, positions, orders, now, today, day_of):
        """The account in three (the owner, 28.10.2026: he buys in it too, and the mentor's signals are traded in it):
        Atlas' own book (the shares its orders bought, at its own entries, its own closed trades), the mentor's
        (the same, of mentor.py's orders) and the owner's (the rest). Atlas' limits, its money and its risk are
        its book's only; the owner's positions and the mentor's aren't its business."""
        SH = {"atlas": "sys_shares", "mentor": "men_shares", "manual": "man_shares"}

        def part(kind):
            rows = []
            for p in positions:
                sh = p[SH[kind]]
                if sh <= 1e-9:
                    continue
                cost = {"atlas": p["sys_cost"], "mentor": p["men_cost"]}.get(kind) or p["avg_cost"]
                stop = {"atlas": p.get("atlas_stop"), "mentor": p.get("men_stop")}.get(kind) or p.get("stop")
                day = {"atlas": p.get("atlas_day"), "mentor": p.get("men_day")}.get(kind)
                prev = p.get("prev_close")
                base = cost if kind != "manual" and day == today else prev  # (bought today: its day starts at the entry)
                rows.append({**p, "shares": sh, "avg_cost": cost, "value": p["price"] * sh, "unrealized": (p["price"] - cost) * sh,
                             "day_usd": (p["price"] - base) * sh if base else None,
                             "day_pct": p["price"] / base - 1 if base else None, "stop": stop, "day": day,
                             "source": {"atlas": "system", "mentor": "mentor"}.get(kind, "manual"),
                             "risk": max(0.0, (cost - stop) * sh) if stop else p["price"] * sh,
                             "strategy": {"atlas": p.get("atlas_strategy"), "mentor": "mentor"}.get(kind),
                             "target": p.get("men_target") if kind == "mentor" else None,
                             "units": p.get("men_units") if kind == "mentor" else None})
            return rows

        def book(kind):
            rows = part(kind)
            own = [o for o in orders if (o.get("strategy") == "mentor") == (kind == "mentor")]
            closed = [o for o in own if o["status"] == "closed" and o.get("closed")]
            realized = lambda days: sum(o.get("pnl") or 0 for o in closed if now - o["closed"] <= days * 86400)
            open_pnl = sum(p["unrealized"] for p in rows)
            shut = [o for o in closed if day_of(o["closed"]) == today]
            today_closed = sum(o.get("pnl") or 0 for o in shut)
            moves = [p["day_usd"] for p in rows if p["day_usd"] is not None]
            entered = [o for o in own if o["status"] in ("open", "closed") and o.get("entry_day") == today]
            wins = [o for o in closed if (o.get("pnl") or 0) > 0]
            return {"positions": rows, "value": sum(p["value"] for p in rows), "open_pnl": open_pnl,
                    "today": today_closed + sum(moves), "realized_today": today_closed,
                    "week": realized(7) + open_pnl, "month": realized(30) + open_pnl,
                    "total": sum(o.get("pnl") or 0 for o in closed) + open_pnl,
                    "open_risk": sum(p["risk"] for p in rows), "trades": len(closed),
                    "win_rate": len(wins) / len(closed) if closed else None,
                    "closed": [{k: o.get(k) for k in ("symbol", "strategy", "entry", "exit", "filled", "pnl", "why")} for o in shut],
                    "entered": len(entered), "entered_open": sum(1 for o in entered if o["status"] == "open"),
                    "waiting": sum(1 for o in own if o["status"] == "sent")}
        theirs = part("manual")
        manual = {"positions": theirs, "value": sum(p["value"] for p in theirs),
                  "open_pnl": sum(p["unrealized"] for p in theirs),
                  "today": sum(p["day_usd"] or 0 for p in theirs), "today_known": all(p["day_usd"] is not None for p in theirs),
                  "open_risk": sum(p["risk"] for p in theirs)}
        return {"atlas": book("atlas"), "mentor": book("mentor"), "manual": manual}

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
                         "mode": o.get("mode", "paper"),
                         "sim": sim and {"entry": round(sim["entry"], 2), "exit": sim.get("exit"), "pnl": sim.get("pnl"),
                                         "shares": sim.get("shares"), "why": sim.get("why")}})
        return {"on": self.settings().get("ibkr_on"), "mode": self.settings().get("ibkr_mode", "paper"),
                "auto": self.settings().get("ibkr_auto", True), "proposals": self.proposals(),
                "rr_override": self.settings().get("rr_override"),
                "alerts": [a for a in self._get("ibkr_alerts", []) if a["ts"] > max(time.time() - 86400, self._get("ibkr_alerts_cleared", 0))][-8:][::-1],
                "live_size": self.settings().get("live_size"), "account": self.status.get("ibkr_account"),
                "error": self.status.get("ibkr_error"), "checked": self.status.get("ibkr_ts"),
                "down_since": self.status.get("ibkr_down_since"), "reconnect": self.status.get("reconnect"),
                "reconnect_every": self.RECONNECT_EVERY,
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
        if self.market_open(now) and now - self.status.get("quotes_ts", 0) >= 60:
            self.status["quotes_ts"] = now
            try:
                self.intraday_entries(now)
            except Exception as e:
                log.warning("Atlas entries during the day: %s", e)
            self.refresh_quotes()
            try:
                self.intraday_stops(now)
            except Exception as e:
                log.warning("Atlas stops during the day: %s", e)
        if now - self.status.get("news_ts", 0) >= 600:
            self.status["news_ts"] = now
            try:
                self.refresh_news()
                self.news_summary(now)
            except Exception as e:
                self.status["news_error"] = f"{type(e).__name__}: {e}"[:160]
        try:
            self.lab_tick(now)
        except Exception as e:
            self.status["lab_error"] = f"{type(e).__name__}: {e}"[:300]
        self.streaks_tick()
        try:
            self.sizing_tick(now)
        except Exception as e:
            log.warning("Atlas sizing: %s", e)
        if now - ((self._get("audit") or {}).get("ts") or 0) >= 20 * 3600:  # (the calculations' audit, once a day)
            try:
                self.run_audit(now)
            except Exception as e:
                log.warning("Atlas audit: %s", e)
        try:
            self.mentor.tick(now)  # (the stale ones let go, even with the gateway down)
        except Exception as e:
            log.warning("Mentor: %s", e)
        try:
            self.warm_backtest()
        except Exception as e:
            log.warning("Atlas backtest ahead: %s", e)
        try:
            self._auto_reconnect(now)
        except Exception as e:
            log.warning("Atlas reconnect: %s", e)
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

    LIVE_EVERY = 1  # seconds: the real account and the prices, while IBKR answers (the owner, 28.10.2026: every second)

    def live_tick(self, now=None):
        """Every few seconds while the gateway is connected: the real account read again, and during the
        session the prices of every position and every open trade of the simulation in real time (IBKR's
        stream: nothing is fetched), the simulation's stops moved on them. Nothing without the gateway:
        Yahoo's prices stay on the minute's pass."""
        now = now or time.time()
        gw = self.ibkr
        if gw is None or not gw.connected():
            return False
        S = self.settings()
        acc = self._read_account(gw, S.get("ibkr_mode", "paper"), now) if S.get("ibkr_on") else None
        if not self.market_open(now):
            return True
        book = self.paper_book() or {}
        symbols = {t["symbol"] for t in book.get("open", [])} | {p["symbol"] for p in (acc or {}).get("positions") or []} \
            | self.mentor.watch()
        got = 0
        for sym in sorted(symbols):
            try:
                q = gw.quote(sym, now)
            except Exception as e:
                self.status["ibkr_data_error"] = f"{sym}: {type(e).__name__}: {e}"[:160]
                continue
            if q:
                self.quotes[sym] = q
                got += 1
        if got:
            self.status.update(quotes_source="ibkr", quotes_source_ts=now)
            try:
                self.intraday_stops(now)
            except Exception as e:
                log.warning("Atlas stops (real time): %s", e)
            try:
                self.mentor.tick(now)  # (the mentor's "above 240" / "below 3.8", on the second's prices)
            except Exception as e:
                log.warning("Mentor triggers: %s", e)
        return True

    def live_loop(self, stop):
        while not stop.is_set():
            try:
                self.live_tick()
            except Exception as e:
                log.warning("Atlas live loop: %s", e)
            stop.wait(self.LIVE_EVERY)

    def livelab_loop(self, stop):
        while not stop.is_set():
            try:
                self.livelab.tick()
            except Exception as e:
                log.warning("Atlas live lab: %s", e)
            stop.wait(1)

    def start(self, stop):
        threading.Thread(target=self.run, args=(stop,), daemon=True, name="atlas").start()
        threading.Thread(target=self.livelab_loop, args=(stop,), daemon=True, name="atlas-livelab").start()
        threading.Thread(target=self.live_loop, args=(stop,), daemon=True, name="atlas-live").start()
        threading.Thread(target=self.learn_loop, args=(stop,), daemon=True, name="atlas-brain").start()

    # ---------- the strategies' lanes: lab -> simulation -> real money (the owner, 29.10.2026) ----------
    @staticmethod
    def _summ(rows):
        """rows: (pnl, paid) -> {n, wins, pnl, paid, roi, pess} (pess: lab.money_weighted's, from 10 trades)."""
        from .lab import money_weighted
        n, paid = len(rows), sum(p for _, p in rows)
        roi, pess = money_weighted(rows, 1.645) if n and paid > 0 else (None, None)
        return {"n": n, "wins": sum(1 for x, _ in rows if x > 0), "pnl": round(sum(x for x, _ in rows), 2), "paid": round(paid, 2),
                "roi": roi, "pess": pess if n >= 10 else None}

    def lane_view(self):
        """Each strategy where it is (the lab, the simulation, real money) with its own figures in each, from its
        first trade there, and the lab's on the same days beside them (the backtest of the same rules)."""
        book = self.paper_book() or {"open": [], "trades": []}
        orders = [o for o in self._get("ibkr", {"orders": {}})["orders"].values() if o.get("mode") == "live"]
        tests, L = self.strategy_tests(), self.lanes()
        plans = self.plans(list(equity.STRATEGIES))
        day = lambda ts: dt.datetime.fromtimestamp(ts, NY).date().isoformat()
        out = []
        for k, v in equity.STRATEGIES.items():
            closed = [t for t in book.get("trades", []) if t["strategy"] == k and t.get("why") != "open"]
            opened = [t for t in book.get("open", []) if t["strategy"] == k]
            days = [t["in"] for t in closed] + [t["day"] for t in opened]
            sim_since = min(days) if days else book.get("started")
            mine = [o for o in orders if o["strategy"] == k and o["status"] in ("open", "closed")]
            live_since = day(min(o["sent"] for o in mine)) if mine else None
            lite = (tests.get(k) or {}).get("lite") or []
            lab = lambda since: self._summ([(p, c) for d, p, c in lite if since and d >= since])
            P = plans.get(k) or {}
            out.append({"key": k, **{x: v[x] for x in ("name", "by", "rule")}, "sim_state": L["sim"][k], "live_state": L["live"][k],
                        "params": {x: P.get(x) for x in ("entry", "stop_atr", "max_stop_pct", "rr", "breakeven_r", "trail_r", "trail_atr", "max_hold")},
                        "sim": {**self._summ([(t["pnl"], t["entry"] * t["shares"]) for t in closed]), "open": len(opened), "since": sim_since},
                        "live": {**self._summ([(o.get("pnl") or 0, (o.get("entry") or 0) * (o.get("filled") or 0)) for o in mine if o["status"] == "closed"]),
                                 "open": sum(1 for o in mine if o["status"] == "open"), "since": live_since},
                        "lab": {"history": self._summ([(p, c) for _, p, c in lite]), "same_sim": lab(sim_since), "same_live": lab(live_since),
                                "test": (tests.get(k) or {}).get("new")}})
        # The intraday variants (experiment 1, livelab.LIVE): their simulation is the scanner's own record from the day
        # the owner moved them there; their real money, the real orders of livelab.Real; the lab, the recorded
        # history (and the days since in real time), in R ($30 risked a trade).
        hist = {r["variant"]: r for r in ((self._get("lab") or {}).get("board") or [])}
        fwd = {r["variant"]: r for r in ((self._get("livelab_board") or {}).get("rows") or [])}
        since = self.settings().get("lanes_since") or {}
        real = [o for o in self._get("livelab_real", {"orders": {}})["orders"].values()]
        today_rows = (self._get("livelab") or {}).get("rows") or {}
        for k in livelab.LIVE:
            h, f = hist.get(k) or {}, fwd.get(k) or {}
            sim_since = (since.get("sim") or {}).get(k)
            simr = self.livelab.day_rows(k, sim_since) if L["sim"][k] != "out" else []
            mine = [o for o in real if o.get("variant") == k and o["status"] in ("open", "closed")]
            r_of = lambda rows: (sum(r for r in rows) / len(rows)) if rows else None
            out.append({"key": k, "kind": "intraday", "name": scalp.NAMES[k], "by": "ניסוי 1: פריצת הפתיחה (תוך-יומי)", "rule": scalp.NAMES[k],
                        "sim_state": L["sim"][k], "live_state": L["live"][k], "params": {},
                        "sim": {**self._summ([(t["pnl"], t["entry"] * t["shares"]) for t in simr]), "r": r_of([t["r"] for t in simr]),
                                "open": sum(1 for t in today_rows.get(k, []) if t.get("open")) if L["sim"][k] != "out" else 0, "since": sim_since},
                        "live": {**self._summ([(o.get("pnl") or 0, (o.get("entry") or 0) * o.get("qty", 0)) for o in mine if o["status"] == "closed"]),
                                 "r": r_of([o.get("r_actual") or 0 for o in mine if o["status"] == "closed"]),
                                 "open": sum(1 for o in mine if o["status"] == "open"), "since": (since.get("live") or {}).get(k)},
                        "lab": {"history": {"n": h.get("trades") or 0, "wins": round((h.get("win_rate") or 0) * (h.get("trades") or 0)),
                                            "pnl": h.get("pnl") or 0, "roi": None, "pess": None, "r": h.get("r_per_trade"),
                                            "z": h.get("day_z"), "vs": h.get("vs_placebo"), "status": h.get("status"), "next": h.get("next")},
                                "fwd": {"n": f.get("trades") or 0, "pnl": f.get("pnl") or 0, "r": f.get("r_per_trade"), "vs": f.get("vs_placebo"),
                                        "status": f.get("status")}}})
        return out

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
            mine = self.settings().get("rr_override") is not None
            system = self.plans(list(equity.STRATEGIES), system=True) if mine else {}
            for k, P in plans.items():
                r = equity.portfolio(prepared, {k: P}, port)
                out[k] = {x: r["split"][x] for x in ("old", "val_stats", "new", "robust", "val", "cut")}
                # (every trade, light: the strategy cards set the lab beside the simulation and real money)
                out[k]["lite"] = [(t["in"], round(t["pnl"], 2), round(t["entry"] * t["shares"], 2)) for t in r["trades"]]
                if mine:  # (the owner's target against the system's own, on the same history: what his choice is worth)
                    out[k]["system"] = {x: equity.portfolio(prepared, {k: system[k]}, port)["split"][x] for x in ("old", "new")}
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
                                                                        if k not in ("quotes_ts", "ibkr_tried", "lab_tried", "streaks_version")}},
            "picture": equity.picture(prepared), "indices": equity.indices(prepared),
            "technical": equity.technical(spy) if spy and len(spy["c"]) > 260 else None,
            "outlook": equity.outlook(spy, prepared.get("^VIX")) if spy else None,
            "spy_chart": equity.chart(spy) if spy else None,
            "strategies": [{"key": k, **{x: v[x] for x in ("name", "by", "rule")}, "on": self.settings()["strategies"].get(k, True),
                            "brain": {x: (brain.get(k) or {}).get(x) for x in ("params", "old", "val", "tried", "kept", "since")},
                            "test": tests.get(k),
                            "history": ((brain.get(k) or {}).get("history") or [])[-10:]} for k, v in equity.STRATEGIES.items()],
            "learned": sorted((h for b in brain.values() for h in (b.get("history") or [])), key=lambda h: -h["ts"])[:20],
            "paper": self.paper_view(), "live": self.live_book(), "mentor": self.mentor.view(), "livelab": self.livelab.view(), "lanes": self.lane_view(), "quiet": self._quiet_safe(), "audit": self._get("audit"),
            "news": sorted(({**n, "important": important(n)} for n in news), key=lambda n: (not n["important"], -(n.get("ts") or 0)))[:40],
            "news_summary": self._get("news_summary"), "review": (self._get("reviews", []) or [None])[-1],
            "names": NAMES, "stages": equity.STAGES, "defaults": equity.DEFAULTS, "market_open": self.market_open(),
            "mails": self.mail_status(), "ibkr": self.ibkr_view(), "lab": self.lab_view(),
            "streaks": self._get("streaks"), "sizing": self.sizing_view(), "notes": self.notes_view(),
        }

    # The backtest tab's answer, ready before it is asked (the owner, 28.10.2026: it was slow to come up):
    # each answer is kept by everything it depends on (the data, the rules the brain has now, the request),
    # and the tab's first one (the system's rules, every strategy that is on) is worked out in the
    # background whenever any of those changes, so opening the tab takes no time.
    BT_KEEP = 8

    def _bt_key(self, kinds, mode, params):
        kinds = sorted(k for k in kinds or [] if k in equity.STRATEGIES) or sorted(k for k, on in self.settings()["strategies"].items() if on)
        return json.dumps([self.eq.version, kinds, mode, params or {}, self.plans(kinds), self.port()], sort_keys=True, default=str)

    def backtest(self, kinds, mode="system", params=None):
        key = self._bt_key(kinds, mode, params)
        cache = self.__dict__.setdefault("_bt_cache", {})
        if key in cache:
            return cache[key]
        out = self._backtest(kinds, mode, params)
        cache[key] = out
        while len(cache) > self.BT_KEEP:
            cache.pop(next(iter(cache)))
        return out

    def warm_backtest(self):
        """The tab's first answer, worked out ahead (in its own thread; once per change)."""
        if self.__dict__.get("_bt_warming") or not self.eq.prepared.get("SPY"):
            return False
        try:
            if self._bt_key([], "system", {}) in self.__dict__.get("_bt_cache", {}):
                return False
        except Exception:
            return False
        self._bt_warming = True

        def go():
            try:
                self.backtest([], "system", {})
            except Exception as e:
                log.warning("Atlas backtest ahead: %s", e)
            finally:
                self._bt_warming = False
        threading.Thread(target=go, daemon=True, name="atlas-bt-warm").start()
        return True

    def _backtest(self, kinds, mode="system", params=None):
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
