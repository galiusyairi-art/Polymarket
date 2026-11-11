"""The day's watchlist (the owner, 07.11.2026: "the system should always research, say, a hundred recommended stocks it
finds on the internet and in analysts' recommendations, and know to enter on the same day: there are lots of chances").

Each trading morning, before the live lab gets ready (livelab.PREP, 9:00 New York), Claude searches the web for the US
stocks with a fresh reason to move up today: analysts' upgrades, new Buy ratings and raised price targets (the firm
named), earnings beats and raised guidance, company news, pre-market gaps on volume. Every one with its source; nothing
without one. The list goes into the live lab's candidates, and its own intraday variant (scalp V9: the base rules on
this list only) is measured on it, beside the scanner's: whether the list is worth anything is the record's to say.

Only what the owner allowed is spent: one research a day (a failed one is tried again after RETRY), and none without
the server's ANTHROPIC_API_KEY."""
import datetime as dt
import json
import logging
import os
import re
import threading
import time
from zoneinfo import ZoneInfo

log = logging.getLogger("polybot")

MODEL = "claude-opus-5-5"
SIZE = 100  # stocks at most
START, END = (7, 0), (15, 0)  # New York: the research runs from START (a day without one yet), never after END
RETRY = 1800  # seconds after a failed research before the next try
MAX_CONTINUE = 6  # the server's search loop paused this many times at most ("pause_turn")
# What it costs (the owner, 07.11.2026: "write in the control room what it costs a day exactly, and in total"): the API's
# own usage of each call priced at the list prices (research.PRICES for the tokens; a web search $10 a thousand, a web
# fetch free beyond its tokens). Every research is kept with its cost, the failed ones too: they are paid as well.
SEARCH_PRICE = 10.0 / 1000
KEEP = 400  # researches kept in the ledger
COST_STEP = 10.0  # a notification and a mail each time the total passes another this many dollars (the owner, 10.11.2026)

KINDS = ("upgrade", "target", "earnings", "news", "premarket", "momentum")
SYMBOL = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z])?$")
NY = ZoneInfo("America/New_York")

SYSTEM = ("You are the research desk of an intraday US-stock trading system. You search the web carefully and report only "
          "what you found a source for. You never invent a stock, a rating, a price target or a number.")

PROMPT = """Today is {day} (New York time, {clock}). Build today's watchlist: up to {n} US-listed common stocks with a fresh,
specific reason to move UP today or in the next few sessions. Search the web for:
- analyst upgrades, new coverage at Buy / Outperform / Overweight, and raised price targets in the last 3 days (name the firm);
- earnings beats or raised guidance reported in the last 2 days or before today's open;
- significant company news in the last 2 days (contracts, approvals, products, deals, buybacks);
- today's pre-market gainers on heavy volume.
Rules: NYSE / NASDAQ common stocks only, price at least $10, average daily volume at least 1 million shares. No ETFs, no OTC,
no SPACs, no warrants. Prefer liquid, well-known names. Aim for at least 40 and at most {n}; fewer real ones beat more weak ones.

Answer with ONLY a JSON array inside one ```json block, best first, each item:
{{"symbol": "NVDA", "kind": "upgrade|target|earnings|news|premarket|momentum",
  "why": "one short sentence in Hebrew: what happened and who said it", "source": "the outlet or the analyst firm",
  "url": "the page you read", "target": price target in dollars or null}}"""


def ny(now):
    return dt.datetime.fromtimestamp(now, NY)


def parse(text, size=SIZE):
    """The model's answer -> the list, checked: a symbol each (as exchanges write them), once, a known kind, a reason."""
    m = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text or "", re.S)
    raw = m[-1] if m else None
    if raw is None:
        i, j = (text or "").find("["), (text or "").rfind("]")
        raw = text[i:j + 1] if 0 <= i < j else "[]"
    try:
        items = json.loads(raw)
    except ValueError:
        return []
    out, seen = [], set()
    for x in items if isinstance(items, list) else []:
        if not isinstance(x, dict):
            continue
        sym = str(x.get("symbol") or "").strip().upper().replace("/", ".")
        if not SYMBOL.match(sym) or sym in seen:
            continue
        why = str(x.get("why") or "").strip()[:240]
        if not why:
            continue  # (nothing without a reason)
        kind = str(x.get("kind") or "news").strip().lower()
        try:
            target = float(x["target"]) if x.get("target") not in (None, "") else None
        except (TypeError, ValueError):
            target = None
        seen.add(sym)
        out.append({"symbol": sym.replace(".", "-") if "." in sym else sym, "kind": kind if kind in KINDS else "news",
                    "why": why, "source": str(x.get("source") or "")[:80], "url": str(x.get("url") or "")[:300],
                    "target": target})
        if len(out) >= size:
            break
    return out


# The research's level (the owner, 11.11.2026: "it drinks money: it must come down to a few dollars a day"): the model,
# and how much of the web it may read. Most of a research's cost is its tokens: each search's results go back in with
# every turn after it, so the searches' number and the pages fetched whole are what weigh; then the model's price
# (Haiku is a fortieth of Opus's, Sonnet half). "max" is the level it ran at till then.
TIERS = {
    "eco": {"model": "claude-haiku-5-5", "max_tokens": 8000,
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 12}]},
    "balanced": {"model": "claude-sonnet-5-5", "max_tokens": 12000,
                 "tools": [{"type": "web_search_20260209", "name": "web_search", "max_uses": 20}]},
    "max": {"model": "claude-opus-5-5", "max_tokens": 16000,
            "tools": [{"type": "web_search_20260209", "name": "web_search"}, {"type": "web_fetch_20260209", "name": "web_fetch"}]},
}
MAX_TRIES = 2  # researches a day at most (a failed one is paid too: 09-10.11.2026 had 4 and 14)
REFORMAT_MODEL = "claude-haiku-5-5"  # (an answer that came without its list: put in order, cheaply, not searched again)


def reformat(text, client=None):
    """An answer that has the stocks but not in the asked form, put into it by a small model without the web (cents, not
    a new research). (text, used)."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    from .research import cost_of
    msg = client.messages.create(model=REFORMAT_MODEL, max_tokens=8000, messages=[{"role": "user", "content":
        "Turn the research below into ONLY a JSON array inside one ```json block, each item {\"symbol\", \"kind\", \"why\" "
        "(Hebrew), \"source\", \"url\", \"target\"}; keep only stocks that have a source in it, invent nothing.\n\n" + text[-60000:]}])
    u = getattr(msg, "usage", None)
    out = "".join(getattr(b, "text", "") or "" for b in msg.content if getattr(b, "type", "") == "text")
    return out, {"searches": 0, "input": getattr(u, "input_tokens", 0) or 0, "output": getattr(u, "output_tokens", 0) or 0,
                 "cost": cost_of(REFORMAT_MODEL, u) if u is not None else 0.0}


def ask(day, clock, size=SIZE, client=None, tier="max"):
    """One research: Claude with the web (run on Anthropic's side), at the given level (TIERS), its server loop resumed
    while it pauses. (text, {"searches", "input", "output", "cost"})."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    T = TIERS.get(tier) or TIERS["balanced"]
    MODEL, tools = T["model"], T["tools"]
    user = {"role": "user", "content": PROMPT.format(day=day, clock=clock, n=size)}
    content, used = [], {"searches": 0, "input": 0, "output": 0, "cost": 0.0}
    for _ in range(MAX_CONTINUE + 1):
        messages = [user] + ([{"role": "assistant", "content": content}] if content else [])
        with client.messages.stream(model=MODEL, max_tokens=T["max_tokens"], system=SYSTEM, tools=tools, messages=messages) as stream:
            resp = stream.get_final_message()
        content = list(content) + list(resp.content)
        u = getattr(resp, "usage", None)
        if u is not None:
            from .research import cost_of
            used["input"] += getattr(u, "input_tokens", 0) or 0
            used["output"] += getattr(u, "output_tokens", 0) or 0
            stu = getattr(u, "server_tool_use", None)
            n = (getattr(stu, "web_search_requests", 0) or 0) if stu is not None else 0
            used["searches"] += n
            used["cost"] += cost_of(MODEL, u) + n * SEARCH_PRICE
        if resp.stop_reason != "pause_turn":
            break
    text = "".join(getattr(b, "text", "") or "" for b in content if getattr(b, "type", "") == "text")
    return text, used


class Watchlist:
    """The day's list in Atlas' store ("watchlist"): {"day", "ts", "items", "error", "used", "took"}."""

    def __init__(self, atlas, asker=None, reformatter=None):
        self.a = atlas
        self.asker = asker or (lambda day, clock: ask(day, clock, tier=self.tier()))
        self.reformatter = reformatter or reformat
        self.running = False

    def tier(self):
        return self.a.settings().get("research_tier") or "balanced"

    def over_budget(self, now=None):
        """Today's research (the mornings' and the trade ticket's) at or past the owner's daily cap (0: none)."""
        cap = float(self.a.settings().get("research_cap") or 0)
        return bool(cap) and self.costs(now)["today"] >= cap

    def get(self):
        return self.a._get("watchlist") or {}

    def symbols(self, day=None):
        """Today's symbols (yesterday's while today's is not in yet: a list a day old beats none)."""
        W = self.get()
        return [x["symbol"] for x in W.get("items") or []]

    def due(self, now):
        t = ny(now)
        if t.weekday() >= 5 or not (START <= (t.hour, t.minute) < END) or self.running:
            return False
        if not self.a.settings().get("watch_on", True) or not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        W = self.get()
        if W.get("day") == t.date().isoformat() and W.get("items"):
            return False
        if W.get("tries_day") == t.date().isoformat() and (W.get("tries") or 0) >= MAX_TRIES:
            return False  # (two a day at most: a failed one is paid too)
        if self.over_budget(now):
            return False
        return now - (W.get("tried") or 0) >= RETRY

    def run(self, now=None):
        now = now or time.time()
        t = ny(now)
        day = t.date().isoformat()
        self.running = True
        W = self.get()
        tries = (W.get("tries") or 0) + 1 if W.get("tries_day") == day else 1
        W = dict(W, tried=now, tries=tries, tries_day=day)
        self.a._set("watchlist", W)
        used, ok = None, False
        try:
            text, used = self.asker(day, t.strftime("%H:%M"))
            items = parse(text)
            if not items and text.strip():
                # (the stocks came, not in the asked form: put in order for cents, never a whole research again)
                try:
                    text2, used2 = self.reformatter(text)
                    items = parse(text2)
                    used = {k: (used or {}).get(k, 0) + used2.get(k, 0) for k in ("searches", "input", "output", "cost")}
                except Exception as e:
                    log.warning("Watchlist reformat: %s", e)
            if not items:  # (the answer's end kept with the error: what it said instead, for the research map)
                raise ValueError(f"לא חזרה רשימה עם מקורות · סוף התשובה: {' '.join(text.split())[-160:] or 'ריק'}")
            W = {"day": day, "ts": time.time(), "items": items, "error": None, "used": used, "took": round(time.time() - now),
                 "tried": now, "tries": tries, "tries_day": day, "tier": self.tier()}
            ok = True
            log.warning("Watchlist: %d stocks for %s (%s searches)", len(items), day, used.get("searches"))
        except Exception as e:
            W["error"] = f"{type(e).__name__}: {e}"[:300]
            log.warning("Watchlist: %s", W["error"])
        finally:
            self.running = False
        self.a._set("watchlist", W)
        self.record(now, day, used, ok)
        return W

    def record(self, now, day, used, ok):
        """One research into the ledger, with what it cost (a call that failed before any answer cost nothing)."""
        used = used or {}
        L = self.a._get("watchlist_costs") or []
        L.append({"ts": now, "day": day, "ok": bool(ok), "cost": round(float(used.get("cost") or 0.0), 4),
                  "searches": int(used.get("searches") or 0), "input": int(used.get("input") or 0), "output": int(used.get("output") or 0)})
        self.a._set("watchlist_costs", L[-KEEP:])
        tot = self.a._get("watchlist_total") or {"cost": 0.0, "runs": 0, "since": now}
        before = tot["cost"]
        tot.update(cost=round(tot["cost"] + L[-1]["cost"], 4), runs=tot["runs"] + 1)
        self.a._set("watchlist_total", tot)
        # (the owner, 10.11.2026: "every time the research's cost passes another $10, a notification and a mail")
        step = int(tot["cost"] // COST_STEP)
        if step > int(before // COST_STEP):
            C = self.costs(now)
            m = {"ts": now, "step": step * COST_STEP, "total": tot["cost"], "runs": tot["runs"], "since": tot.get("since"),
                 "avg": C.get("avg"), "today": C.get("today")}
            q = self.a._get("cost_mails") or []
            self.a._set("cost_mails", (q + [m])[-10:])
            self.a.note("cost", f"💰 עלות מחקר המניות עברה ${step * COST_STEP:,.0f}",
                        f"סה\"כ ${tot['cost']:,.2f} מאז שהתחיל, ב-{tot['runs']} מחקרים", "info", now)

    def costs(self, now=None):
        """For the control room: today's cost, the last days', the average a trading day, and the total since the first."""
        now = now or time.time()
        L = self.a._get("watchlist_costs") or []
        tot = self.a._get("watchlist_total") or {"cost": 0.0, "runs": 0, "since": None}
        by = {}
        for x in L:
            d = by.setdefault(x["day"], {"day": x["day"], "cost": 0.0, "runs": 0, "searches": 0, "ok": False})
            d["cost"] += x["cost"]
            d["runs"] += 1
            d["searches"] += x["searches"]
            d["ok"] = d["ok"] or x["ok"]
        days = sorted(by.values(), key=lambda d: d["day"], reverse=True)
        today = ny(now).date().isoformat()
        return {"today": by.get(today, {"cost": 0.0, "runs": 0, "searches": 0})["cost"], "days": days[:14],
                "avg": sum(d["cost"] for d in days) / len(days) if days else None, "total": tot["cost"], "runs": tot["runs"],
                "since": tot.get("since"), "price": {"search": SEARCH_PRICE * 1000},
                "tier": self.tier(), "cap": float(self.a.settings().get("research_cap") or 0)}

    def tick(self, now=None):
        now = now or time.time()
        if self.due(now):
            threading.Thread(target=self.run, args=(now,), daemon=True, name="atlas-watchlist").start()
            return True
        return False
