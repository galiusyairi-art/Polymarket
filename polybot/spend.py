"""What the research costs (the owner, 07.11.2026: "write in the control room what it costs a day exactly, and in total").

Every call to Claude that Atlas pays for is kept here with what it cost: the trade ticket's deep pass on a stock
(ticket.py) and the deep review of the strategies (deepreview.py). The cost is the API's own usage of each call priced
at the list prices (research.PRICES for the tokens; a web search $10 a thousand, a web fetch free beyond its tokens).
A call that failed is kept too: it is paid as well. From the ledger come the cost card of the control room (today, the
last days, an average trading day, the total), the owner's cap a day (settings["research_cap"]) and the notification and the
mail each time the total passes another COST_STEP dollars.

The ledger began with the morning's research of a hundred stocks (watchlist.py, taken down on 23.11.2026 by the owner:
"take it down completely"). Its keys in Atlas' store are still "watchlist_costs" and "watchlist_total": the names stay,
so what was spent till then goes on being counted in the total."""
import datetime as dt
import time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

SEARCH_PRICE = 10.0 / 1000  # dollars a web search
KEEP = 400  # calls kept in the ledger
COST_STEP = 10.0  # a notification and a mail each time the total passes another this many dollars (the owner, 10.11.2026)


class Ledger:
    """The calls and their cost in Atlas' store: "watchlist_costs" (the last KEEP, one a call: {"ts", "day", "ok", "cost",
    "searches", "input", "output"}) and "watchlist_total" ({"cost", "runs", "since"}: everything since the first)."""

    def __init__(self, atlas):
        self.a = atlas

    def over_budget(self, now=None):
        """Today's research at or past the owner's daily cap (0: none)."""
        cap = float(self.a.settings().get("research_cap") or 0)
        return bool(cap) and self.costs(now)["today"] >= cap

    def record(self, now, day, used, ok):
        """One call into the ledger, with what it cost (a call that failed before any answer cost nothing)."""
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
            self.a.note("cost", f"💰 עלות המחקר עברה ${step * COST_STEP:,.0f}",
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
        today = dt.datetime.fromtimestamp(now, NY).date().isoformat()
        S = self.a.settings()
        return {"today": by.get(today, {"cost": 0.0, "runs": 0, "searches": 0})["cost"], "days": days[:14],
                "avg": sum(d["cost"] for d in days) / len(days) if days else None, "total": tot["cost"], "runs": tot["runs"],
                "since": tot.get("since"), "price": {"search": SEARCH_PRICE * 1000},
                "tier": S.get("research_tier") or "balanced", "cap": float(S.get("research_cap") or 0)}
