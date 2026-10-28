"""The mentor's signals (the owner, 28.10.2026): Raz Gamliel's messages in his Discord's four main channels,
read, understood and traded in the owner's IBKR account, in a book of their own (beside Atlas' and the owner's).

- parse(): a message into what it asks (pure, no I/O). Two kinds of signals:
  * the detailed one ("ביטקוין שורט / כניסה מיידית / סיכון 0.25% מהתיק / STOP 83200 / TP 65600"): its own
    stop, target and risk (a share of the account), taken as he says;
  * the quick one ("VRT מעל 240 אכנס חצי כמות", "ZETA הפחתתי לכמות קטנה", "XLAB מתחת 3.8 אסגור"): a size in
    words (full, half, a third, small, double) of the owner's own size for a trade, the owner's risk and a
    stop of the owner's (mentor_stop_pct), at a price when he names one.
  The rest (what he watches, what he holds, what happened) is information: recorded, never traded. A message
  that names a stock and nothing clear about it waits for the owner.
- Mentor: the messages coming in (the inbox, the paste box), each signal kept with what was done about it,
  the price triggers watched every second, and the orders themselves (in Atlas' IBKR state, strategy
  "mentor", so fills, stops, mails and the dashboard follow them as every order).
"""

import datetime as dt
import hashlib
import logging
import os
import re
import threading
import time

log = logging.getLogger("polybot")

# Futures and crypto he trades in his futures channel: recognised, not traded yet (stocks only for now).
ASSETS = {"ביטקוין": "BTC", "ביטקויין": "BTC", "ביטכוין": "BTC", "אתריום": "ETH", "את'ריום": "ETH", "זהב": "GOLD",
          "כסף מתכת": "SILVER", "נסדק": "NASDAQ", "נאסדק": "NASDAQ", 'נאסד"ק': "NASDAQ", "נסדאק": "NASDAQ",
          "אס אנד פי": "SPX", "אסנדפי": "SPX", "דאו": "DOW", "ראסל": "RUSSELL", "נפט": "OIL"}
FUTURES = {"BTC", "ETH", "GOLD", "SILVER", "NASDAQ", "SPX", "DOW", "RUSSELL", "OIL", "ES", "MES", "NQ", "MNQ", "YM",
           "MYM", "RTY", "M2K", "GC", "MGC", "SI", "CL", "MCL", "MBT", "XAUUSD", "BTCUSD", "US30", "NAS100", "SPX500"}
NOT_TICKERS = {"STOP", "TP", "SL", "EVERYONE", "HERE", "ATH", "ATL", "EMA", "SMA", "MA", "VWAP", "RSI", "MACD", "ATR",
               "AM", "PM", "OK", "LOL", "THE", "AND", "OR", "IF", "IN", "OUT", "LONG", "SHORT", "BUY", "SELL", "RISK",
               "TARGET", "ENTRY", "USD", "NYSE", "SEC", "FOMC", "CPI", "PPI", "FED", "ETF", "IPO", "CEO", "AI", "TPS",
               "RR", "R", "BE", "TO", "AT", "HOD", "LOD", "PT"}
NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
SIZES = (("שני שליש", 2 / 3), ("כמות כפולה", 2.0), ("כפולה", 2.0), ("כמות מלאה", 1.0), ("מלאה", 1.0),
         ("חצי", 0.5), ("שליש", 1 / 3), ("שלי ש", 1 / 3), ("רבע", 0.25), ("קטנה", 0.25), ("קטן", 0.25))
INFO = re.compile(r"למעקב|ראיתם|מחזיק רק|כרגע מחזיק|שימו לב|על הפרק|התרחיש היחיד|חשוב לציין|לא המלצה|ללמידה")
EXIT = re.compile(r"אסגור|סוגר|סגרתי|יצאתי|יוצא מ|אצא|יציאה מלאה|מוכר הכל|מכרתי הכל|מימשתי הכל|מממש הכל|אממש הכל|סגירה מלאה")
REDUCE = re.compile(r"מפחית|הפחתתי|אפחית|מוריד כמות|הורדתי|מקטין|הקטנתי|צמצמתי|מצמצם|מימשתי|מממש|אממש|מימוש חלקי|לקחתי רווח")
ENTER = re.compile(r"נכנסתי|אכנס|נכנס|בפנים|כניסה|קניתי|קונה|אקנה|הוספתי|מוסיף|אוסיף|לונג|שורט|כמות")
ABOVE = re.compile(r"(?:מעל|פריצה של|פריצת|אם יפרוץ את|אם יפרוץ|אם תפרוץ|מעבר ל)\s*ל?-?\s*\$?" + NUM)
BELOW = re.compile(r"(?:מתחת|אם לא תחזיק|אם לא יחזיק|שבירה של|שבירת|אם ישבור|אם תשבור|אם ירד מתחת|אם תרד מתחת)\s*ל?-?\s*\$?" + NUM)
STOP = re.compile(r"(?:\bSTOP\b|\bSL\b|סטופ)\s*:?\s*-?\s*" + NUM + r"\s*(נקודות|נק'?)?", re.I)
TP = re.compile(r"(?:\bTP\d?\b|טייק(?: פרופיט)?|יעד)\s*:?\s*-?\s*" + NUM + r"\s*(נקודות|נק'?)?", re.I)
RISK = re.compile(r"סיכון\s*(?:של\s*)?" + NUM + r"\s*%|" + NUM + r"\s*%\s*מהתיק")


def _num(s):
    return float(s.replace(",", ""))


def _clean(text):
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"<[@#:][^>]*>|@everyone|@here|@\w+", " ", text)
    return text


def tickers(text):
    """The stocks (and the futures he names in Hebrew) a message names, in order. A ticker of one or two
    letters counts only in capitals (his "be" is a word as often as Bloom Energy)."""
    out = []
    for word, sym in ASSETS.items():
        if word in text:
            out.append((text.index(word), sym))
    for m in re.finditer(r"(?<![A-Za-z$])\$?([A-Za-z]{1,5})(?![A-Za-z])", text):
        w = m.group(1)
        if len(w) <= 2 and not w.isupper():
            continue
        if w.upper() in NOT_TICKERS and not (w.isupper() and w.upper() in ("BE",)):
            continue
        out.append((m.start(), w.upper()))
    seen, res = set(), []
    for _, s in sorted(out):
        if s not in seen:
            seen.add(s)
            res.append(s)
    return res


def size_of(text):
    for word, f in SIZES:
        if word in text:
            return f
    return None


def parse(text, context=None):
    """One message into what it asks: {"kind": enter / reduce / exit / unclear / info, "symbol", "asset":
    stock / future, "side": long / short, "size" (of the owner's size for a trade; None: not said),
    "trigger": {"dir": above / below, "price"} or None, "stop", "stop_pts", "tp", "tp_pts", "risk_pct",
    "detailed"}. context: the symbol of his last message (a message without one is about it)."""
    t = _clean(text)
    syms = tickers(t)
    out = {"kind": "info", "symbol": None, "asset": "stock", "side": "long", "size": size_of(t), "trigger": None,
           "stop": None, "stop_pts": None, "tp": None, "tp_pts": None, "risk_pct": None, "detailed": False}
    m = STOP.search(t)
    if m:
        out["stop_pts" if m.group(2) else "stop"] = _num(m.group(1))
    m = TP.search(t)
    if m:
        out["tp_pts" if m.group(2) else "tp"] = _num(m.group(1))
    m = RISK.search(t)
    if m:
        out["risk_pct"] = _num(m.group(1) or m.group(2))
    if "שורט" in t:
        out["side"] = "short"
    m = ABOVE.search(t)
    if m:
        out["trigger"] = {"dir": "above", "price": _num(m.group(1))}
    m = BELOW.search(t)
    if m and not out["trigger"]:
        out["trigger"] = {"dir": "below", "price": _num(m.group(1))}
    out["detailed"] = bool((out["stop"] or out["stop_pts"]) and (out["risk_pct"] or "כניסה" in t))
    action = "exit" if EXIT.search(t) else "reduce" if REDUCE.search(t) else "enter" if ENTER.search(t) else None
    if INFO.search(t) and not out["detailed"]:
        return out
    if not syms and context and action:
        syms = [context]
    if not syms:
        return out
    out["symbol"] = syms[0]
    out["asset"] = "future" if syms[0] in FUTURES else "stock"
    if len(syms) > 1 and not out["detailed"]:
        out.update(kind="unclear", symbols=syms)
        return out
    out["kind"] = action or ("enter" if out["detailed"] else "unclear")
    return out


def describe(p):
    """What a parsed message means, in a line of Hebrew (for the page and the notes)."""
    if p["kind"] == "info":
        return "מידע / עדכון: לא איתות"
    sym = p.get("symbol") or "?"
    size = {2.0: "כמות כפולה", 1.0: "כמות מלאה", 0.5: "חצי כמות", 0.25: "כמות קטנה"}.get(p.get("size")) or (
        "שליש כמות" if p.get("size") and abs(p["size"] - 1 / 3) < 1e-6 else "שני שליש" if p.get("size") and abs(p["size"] - 2 / 3) < 1e-6 else None)
    trig = p.get("trigger")
    when = f" כשהמחיר {'עובר את ' if trig['dir'] == 'above' else 'יורד מתחת ל-'}{trig['price']:g}" if trig else ""
    if p["kind"] == "unclear":
        return f"{', '.join(p.get('symbols') or [sym])}: לא ברור מה לעשות (ממתין להחלטה שלך)"
    if p["kind"] == "exit":
        return f"יציאה מלאה מ-{sym}{when}"
    if p["kind"] == "reduce":
        return f"הקטנה של {sym}{' ל' + size if size else ' (לחצי)'}{when}"
    side = "שורט" if p["side"] == "short" else "לונג"
    bits = [f"{side} {sym}", size or ("לפי הסיכון שלו" if p.get("risk_pct") else "כמות מלאה")]
    if p.get("risk_pct"):
        bits.append(f"סיכון {p['risk_pct']:g}% מהתיק")
    if p.get("stop") or p.get("stop_pts"):
        bits.append(f"סטופ {p['stop']:g}" if p.get("stop") else f"סטופ {p['stop_pts']:g} נקודות")
    if p.get("tp") or p.get("tp_pts"):
        bits.append(f"יעד {p['tp']:g}" if p.get("tp") else f"יעד {p['tp_pts']:g} נקודות")
    return " · ".join(bits) + when


class Mentor:
    """The mentor's signals in the owner's account, through Atlas' IBKR connection and records."""
    KEEP = 300  # messages kept
    WAIT = 3600  # seconds a signal waits for the owner's yes (then it's stale: the price has moved)
    TRIGGER_DAYS = 3  # a "מעל 240 אכנס" stays armed this many days
    CONTEXT = 1800  # seconds his last symbol is what a message without one is about
    MAX_UNITS = 2.0  # (double, at most)

    def __init__(self, atlas):
        self.a = atlas
        self.lock = threading.Lock()

    def _state(self):
        return self.a._get("mentor", {"msgs": [], "signals": {}})

    def _save(self, st):
        st["msgs"] = st["msgs"][-self.KEEP:]
        if len(st["signals"]) > self.KEEP:
            for k in sorted(st["signals"], key=lambda k: st["signals"][k]["ts"])[:len(st["signals"]) - self.KEEP]:
                st["signals"].pop(k)
        self.a._set("mentor", st)

    # ---------- in ----------
    def receive(self, text, author=None, source="inbox", msg_id=None, now=None, run=True):
        """A message from the mentor's channels: recorded, understood, and (by the settings) traded at once,
        armed on its price, or left for the owner. The signal's record."""
        now = now or time.time()
        text = (text or "").strip()
        if not text:
            raise ValueError("הודעה ריקה")
        S = self.a.settings()
        who = (S.get("mentor_author") or "").strip()
        msg_id = msg_id or hashlib.sha1(f"{author}|{text}".encode()).hexdigest()[:16]
        with self.lock:
            st = self._state()
            if any(m["id"] == msg_id and now - m["ts"] < 3600 for m in st["msgs"]):
                return {"duplicate": True, "id": msg_id}
            last = next((m for m in reversed(st["msgs"]) if m.get("symbol") and now - m["ts"] <= self.CONTEXT), None)
            p = parse(text, last and last["symbol"])
            msg = {"id": msg_id, "ts": now, "author": author, "text": text[:2000], "source": source, "kind": p["kind"],
                   "symbol": p.get("symbol")}
            st["msgs"].append(msg)
            sig = {**p, "id": msg_id, "ts": now, "text": text[:600], "what": describe(p), "status": "info", "why": None}
            if author and who and who.lower() not in author.lower() and source == "inbox":
                sig.update(status="ignored", why=f"לא של {who}")
            elif p["kind"] == "info":
                pass
            elif p["asset"] == "future" or p["side"] == "short":
                sig.update(status="unsupported", why="חוזים / שורט: עוד לא נכנסים אוטומטית (רק מניות בלונג)")
            elif not S.get("mentor_on", True):
                sig.update(status="off", why="המסחר לפי המנטור כבוי בהגדרות")
            elif p["kind"] == "unclear" or not S.get("mentor_auto", True):
                sig.update(status="waiting", why=None if p["kind"] != "unclear" else "לא ברור מה לעשות")
            elif p["trigger"]:
                sig.update(status="armed")
            else:
                sig.update(status="go")
            st["signals"][msg_id] = sig
            self._save(st)
        if sig["status"] not in ("info", "ignored"):
            self.a.note("mentor", f"🎓 המנטור: {sig.get('symbol') or ''}", f"{sig['what']}" + (f" · {sig['why']}" if sig.get("why") else ""), now=now)
        if sig["status"] == "go" and run:
            threading.Thread(target=self.execute, args=(msg_id,), daemon=True, name="mentor-go").start()
        return sig

    def decide(self, sig_id, yes, size=None, now=None):
        """The owner on a signal that waits: do it (at the size he picks, for one that wasn't clear) or not."""
        now = now or time.time()
        with self.lock:
            st = self._state()
            sig = st["signals"].get(sig_id)
            if not sig or sig["status"] not in ("waiting", "armed", "unsupported", "off"):
                raise ValueError("האיתות הזה כבר לא ממתין")
            if not yes:
                sig.update(status="declined", why="ויתרת", decided=now)
                self._save(st)
                return sig
            if sig["status"] == "unsupported":
                raise ValueError("עוד אי אפשר לבצע את זה אוטומטית (חוזים / שורט)")
            if sig["kind"] == "unclear":
                sig.update(kind="enter", size=float(size or 1.0))
                sig["what"] = describe(sig)
            sig.update(status="armed" if sig.get("trigger") else "go", decided=now)
            self._save(st)
        if sig["status"] == "go":
            return self.execute(sig_id, now)
        return sig

    # ---------- the account ----------
    def _orders(self, st_ib, sym, mode):
        return [o for o in st_ib["orders"].values() if o.get("strategy") == "mentor" and o["symbol"] == sym
                and o.get("mode", "paper") == mode and o["status"] in ("sent", "open") and not o.get("exit_sent")]  # (being sold: gone)

    def units(self, sym, st_ib=None, mode=None):
        st_ib = st_ib or self.a._get("ibkr", {"orders": {}})
        mode = mode or self.a.settings().get("ibkr_mode", "paper")
        return sum(o.get("units") or 0 for o in self._orders(st_ib, sym, mode))

    def _price(self, gw, sym, now):
        try:
            q = gw.quote(sym, now)
        except Exception:
            q = None
        if q:
            self.a.quotes[sym] = q
            return q["price"]
        q = self.a.quotes.get(sym)
        if q and now - q.get("ts", 0) <= 300:
            return q["price"]
        raise ValueError(f"אין מחיר עדכני ל-{sym}")

    def execute(self, sig_id, now=None):
        """Do a signal now (its price, if it had one, was reached): the record says what was done and why."""
        now = now or time.time()
        with self.lock:
            st = self._state()
            sig = st["signals"].get(sig_id)
            if not sig or sig["status"] not in ("go", "armed"):
                return sig
            sig["status"] = "running"
            self._save(st)
        try:
            done = self._do(sig, now)
            sig.update(status=done.pop("status", "done"), why=done.pop("why", None), result=done, done=now)
        except Exception as e:
            sig.update(status="error", why=f"{e}"[:300], done=now)
            self.a.ibkr_alert("איתות של המנטור", str(e), sig.get("symbol"), now)
        with self.lock:
            st = self._state()
            st["signals"][sig_id] = sig
            self._save(st)
        icon = {"done": "✅", "skipped": "⏭️", "error": "⚠️"}.get(sig["status"], "•")
        self.a.note("mentor", f"{icon} המנטור · {sig.get('symbol')}", f"{sig['what']}" + (f" · {sig['why']}" if sig.get("why") else ""),
                    "bad" if sig["status"] == "error" else "info", now=now)
        return sig

    def _do(self, sig, now):
        a = self.a
        S = a.settings()
        mode = S.get("ibkr_mode", "paper")
        sym = sig["symbol"]
        with a.ib_lock:
            gw = a._gateway()
            st_ib = a._get("ibkr", {"orders": {}})
            mine = self._orders(st_ib, sym, mode)
            have = sum(o.get("units") or 0 for o in mine)
            if sig["kind"] == "enter":
                out = self._enter(gw, st_ib, sig, have, mode, now)
            else:
                if not mine:
                    return {"status": "skipped", "why": f"אין לנו פוזיציה של המנטור ב-{sym}"}
                keep = 0.0 if sig["kind"] == "exit" else (sig["size"] if sig.get("size") is not None else have / 2)
                if keep >= have - 1e-9:
                    return {"status": "skipped", "why": "כבר בכמות הזאת או פחות"}
                out = self._sell(gw, st_ib, mine, 1 - keep / have, now, sig)
            a._set("ibkr", st_ib)
            a._read_account(gw, mode, now)
        return out

    def _enter(self, gw, st_ib, sig, have, mode, now):
        a, S, sym = self.a, self.a.settings(), sig["symbol"]
        want = min(self.MAX_UNITS, sig["size"] if sig.get("size") is not None else 1.0)
        add = want - have
        if add <= 1e-9:
            return {"status": "skipped", "why": "כבר בפנים בכמות הזאת"}
        price = self._price(gw, sym, now)
        size, risk_cap = S.get("live_size") or 0, S.get("live_risk") or 0
        if sig.get("stop") or sig.get("stop_pts"):  # (his own stop, and his risk of the account when he gave one)
            stop = sig["stop"] if sig.get("stop") else price - sig["stop_pts"]
            if stop >= price:
                return {"status": "skipped", "why": f"הסטופ שלו (${stop:,.2f}) מעל המחיר עכשיו (${price:,.2f})"}
            if sig.get("risk_pct"):
                acc = a.status.get("ibkr_acct") or {}
                worth = acc.get("net_liq") or 0
                if not worth:
                    return {"status": "error", "why": "אין שווי חשבון לחישוב הסיכון"}
                qty = int(worth * sig["risk_pct"] / 100 / (price - stop))
                qty = min(qty, int(min(worth, 50_000.0) // price))  # (a typo's guard)
            else:
                qty = int(size * add // price)
                if risk_cap:
                    qty = min(qty, int(risk_cap * add // (price - stop)))
        else:
            stop = round(price * (1 - (S.get("mentor_stop_pct") or 6.0) / 100), 2)
            qty = int(size * add // price)
            if risk_cap:  # (the owner's most a trade may lose, for the size he gave)
                qty = min(qty, int(risk_cap * add // (price - stop)))
        if qty < 1:
            return {"status": "skipped", "why": f"פחות ממניה אחת: מניה עולה ${price:,.2f}, הסכום לטרייד ${size:,.0f}"
                    + (f", הסיכון המקסימלי ${risk_cap:,.0f}" if risk_cap else "")}
        tp = sig.get("tp") or (price + sig["tp_pts"] if sig.get("tp_pts") else None)
        if tp and tp <= price:
            tp = None
        trail = round(price - stop, 2) if S.get("mentor_trail") else None
        entry = ("market", None) if a._session(now) else None
        ids = gw.place_bracket(sym, qty, None, tp, stop, entry, trail=trail)
        key = f"mentor:{sym}:{sig['id']}"
        st_ib["orders"][key] = {"key": key, "strategy": "mentor", "symbol": sym, "qty": qty, "stop": round(stop, 2),
                                "stop0": round(stop, 2), "order": None, "ceiling": None, "target": tp and round(tp, 2), "rr": None,
                                "trail": trail, "ids": ids, "status": "sent", "sent": now, "units": round(add, 4),
                                "signal_day": dt.datetime.fromtimestamp(now, a_ny()).date().isoformat(), "mode": mode,
                                "manual": "mentor", "signal": sig["id"]}
        log.warning("Mentor: %s %d shares, stop %.2f%s", sym, qty, stop, f", target {tp:.2f}" if tp else "")
        return {"order": key, "qty": qty, "price": round(price, 2), "stop": round(stop, 2), "target": tp and round(tp, 2),
                "at": "now" if entry else "open"}

    def _sell(self, gw, st_ib, mine, part, now, sig):
        """part of the mentor's shares in a symbol sold (the latest first); the stops resting on what is left
        made smaller, so nothing is ever sold twice."""
        a = self.a
        session = a._session(now)
        sold, cancelled = 0, 0
        for o in [x for x in mine if x["status"] == "sent"]:
            if part >= 1 - 1e-9:  # (out before it came in: the entry cancelled)
                gw.cancel(o["ids"]["parent"])
                o.update(status="skipped", why="המנטור יצא לפני שנכנסנו")
                cancelled += 1
        held = [x for x in mine if x["status"] == "open" and (x.get("filled") or 0) > 0]
        total = sum(x["filled"] for x in held)
        left = int(round(total * part)) if part < 1 - 1e-9 else total
        if left < 1 and not cancelled:
            return {"status": "skipped", "why": "פחות ממניה אחת למכור"}
        for o in sorted(held, key=lambda x: -x["sent"]):
            if left <= 0:
                break
            take = min(o["filled"], left)
            sid = gw.market(o["symbol"], "SELL", take) if session else gw.sell_at_open(o["symbol"], take)
            if take >= o["filled"]:
                for leg in ("target", "stop"):
                    if o["ids"].get(leg):
                        gw.cancel(o["ids"][leg])
                o["ids"]["exit"], o["exit_sent"], o["exit_why"] = sid, now, "mentor"
            else:  # (part of this order: its own record for the shares sold, the rest's stop made smaller)
                rest = o["filled"] - take
                for leg in ("target", "stop"):
                    if o["ids"].get(leg):
                        gw.modify(o["ids"][leg], totalQuantity=float(rest))
                k = f"{o['key']}:part{int(now)}"
                share = take / o["filled"]
                st_ib["orders"][k] = {**o, "key": k, "qty": take, "filled": take, "units": round((o.get("units") or 0) * share, 4),
                                      "risk_usd": round((o.get("risk_usd") or 0) * share, 2),
                                      "ids": {"parent": None, "exit": sid}, "exit_sent": now, "exit_why": "mentor"}
                o.update(filled=rest, qty=rest, units=round((o.get("units") or 0) * (1 - share), 4),
                         risk_usd=round((o.get("risk_usd") or 0) * (1 - share), 2))
            sold += take
            left -= take
        log.warning("Mentor: %s sold %d shares (%s)", sig["symbol"], sold, sig["kind"])
        return {"sold": sold, "cancelled": cancelled, "at": "now" if session else "open"}

    # ---------- every second ----------
    def watch(self):
        """The symbols whose prices the armed signals wait on."""
        return {s["symbol"] for s in self._state()["signals"].values() if s["status"] == "armed" and s.get("symbol")}

    def tick(self, now=None):
        """The armed signals against the prices now (during the session): each reached, done; the stale ones
        (waiting too long for the owner, or armed too many days) let go."""
        now = now or time.time()
        due, changed = [], False
        with self.lock:
            st = self._state()
            for s in st["signals"].values():
                if s["status"] == "waiting" and now - s["ts"] > self.WAIT:
                    s.update(status="expired", why="לא אושר תוך שעה")
                    changed = True
                elif s["status"] == "armed":
                    if s["kind"] == "enter" and now - s["ts"] > self.TRIGGER_DAYS * 86400:
                        s.update(status="expired", why=f"המחיר לא הגיע תוך {self.TRIGGER_DAYS} ימים")
                        changed = True
                        continue
                    q = self.a.quotes.get(s["symbol"])
                    if not (q and now - q.get("ts", 0) <= 30 and self.a._session(now)):
                        continue
                    T = s["trigger"]
                    if (T["dir"] == "above" and q["price"] > T["price"]) or (T["dir"] == "below" and q["price"] < T["price"]):
                        due.append(s["id"])
            if changed:
                self._save(st)
        for sid in due:
            self.execute(sid, now)
        return due

    # ---------- the page ----------
    def view(self, live=None):
        st = self._state()
        sigs = sorted(st["signals"].values(), key=lambda s: -s["ts"])
        return {"signals": sigs[:60], "waiting": [s for s in sigs if s["status"] == "waiting"],
                "armed": [s for s in sigs if s["status"] == "armed"], "last": st["msgs"][-1]["ts"] if st["msgs"] else None,
                "count": len(st["msgs"]), "inbox": bool(os.environ.get("MENTOR_INBOX_TOKEN"))}


def a_ny():
    from .atlas import NY
    return NY
