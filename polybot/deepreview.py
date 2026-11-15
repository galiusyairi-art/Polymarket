"""The deep review of the strategies (the owner, 15.11.2026: "put the most advanced model on, have it go over all the
strategies, polish them and make better ones: a one-off that comes to help, sharpen and improve the ones there are").

One run a press, for each bot apart, by Claude's most capable model (Fable 5.1), on the server, where everything is:
every strategy's rules and its results (Atlas: the history's three parts, the training, the validation and the locked
test the brain never sees, and the simulation's and real money's own; Polybot: the lab's board, every rule and market
with its record), what the brain already tried. It returns a report, what is weak in each and why, and its
suggestions. None of them trades: Atlas' are run at once over the five years (its training, validation and locked test,
beside the rules now) and the owner applies one by his click; Polybot's go into the lab as rules of their own, measured
there like every other, and move on only by the lab's rules and the owner's push. Its cost is counted with the research's."""
import json
import logging
import re
import time

log = logging.getLogger("polybot")

MODEL = "claude-fable-5-1"
MAX_TOKENS = 32000

SYSTEM = ("You are a senior quantitative trading researcher reviewing a live automated trading system for its owner. "
          "You are rigorous and sceptical: you know backtests overfit, small samples lie, and a rule that only works "
          "on the training years is worthless. You judge by the out-of-sample and locked-test numbers first, the "
          "number of trades second, and you never invent a number that isn't in the data you were given.")


def ask(prompt, client=None):
    """(text, used) of one long pass (streaming: the answer is long)."""
    from .research import cost_of
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    with client.messages.stream(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                                messages=[{"role": "user", "content": prompt}]) as stream:
        msg = stream.get_final_message()
    if getattr(msg, "stop_reason", "") == "refusal":
        raise ValueError("המודל סירב לענות")
    u = getattr(msg, "usage", None)
    text = "".join(getattr(b, "text", "") or "" for b in msg.content if getattr(b, "type", "") == "text")
    return text, {"searches": 0, "input": getattr(u, "input_tokens", 0) or 0, "output": getattr(u, "output_tokens", 0) or 0,
                  "cost": cost_of(MODEL, u) if u is not None else 0.0}


def parse(text):
    m = re.search(r"```json\s*(\{.*\})\s*```", text or "", re.S) or re.search(r"(\{.*\})", text or "", re.S)
    if not m:
        raise ValueError("לא חזר דוח")
    return json.loads(m.group(1))


def _stats(s):
    """A backtest part's figures, light."""
    s = s or {}
    r = lambda x, d=2: None if x is None else round(x, d)
    return {"trades": s.get("trades"), "win_rate": r(s.get("win_rate"), 3), "expectancy_usd": r(s.get("expectancy")),
            "profit_factor": r(s.get("profit_factor")), "avg_r": r(s.get("avg_r"), 3)}


# ---------- Atlas ----------
ATLAS_PROMPT = """Review the swing strategies of "Atlas", a US-stock trading system (about 100 liquid stocks, daily bars, 5
years). Each strategy is fixed code with parameters; the system's "brain" already searches the parameters one or two at a
time and keeps a change only if it is better on training AND holds on validation. The locked test (the last ~20% of the
years) is never used for choosing. Results are per strategy as its own account (1% risk a trade, money rules below).

THE DATA (JSON):
{data}

What to do:
1. For each strategy: is it working (locked test and validation first), what is weak, and why (too few trades, edge only
   in training, losses come from where...).
2. Suggest parameter changes for existing strategies ("tweaks"): ONLY parameter keys listed in that strategy's
   "search_space", with values from its list. At most 3 per strategy, only where you have a reason.
3. Suggest new strategies ("new"): each must be an existing strategy's code with a different parameter set (a "base" and
   "params" from its search space), with a clear idea of what it catches. At most 5.
Everything you suggest will be backtested on the same years before anyone uses it; prefer few, well-reasoned ideas.

Answer with ONLY one JSON object inside a ```json block:
{{"report": "the overall verdict in Hebrew, 4-8 sentences",
  "strategies": [{{"key": "...", "verdict": "keep|improve|drop", "weak": "Hebrew, short", "why": "Hebrew, short"}}],
  "tweaks": [{{"strategy": "key", "params": {{"param": value}}, "why": "Hebrew"}}],
  "new": [{{"name": "Hebrew name", "base": "key", "params": {{"param": value}}, "why": "Hebrew"}}]}}"""


def atlas_data(A):
    from . import equity
    from .atlas import SEARCH
    tests, plans, brain, L = A.strategy_tests(), A.plans(list(equity.STRATEGIES)), A.brain(), A.lanes()
    out = []
    for k, v in equity.STRATEGIES.items():
        T = tests.get(k) or {}
        space = {p: list(SEARCH[p]) for p in SEARCH if p in equity.DEFAULTS or p in v["params"]}
        if k.endswith("_id"):
            space.pop("entry", None)
        out.append({"key": k, "name": v["name"], "by": v.get("by"), "rule": v.get("rule"),
                    "params_now": {p: plans[k].get(p) for p in space}, "search_space": space,
                    "in_simulation": L["sim"].get(k), "with_real_money": L["live"].get(k),
                    "training": _stats(T.get("old")), "validation": _stats(T.get("val_stats")), "locked_test": _stats(T.get("new")),
                    "brain": {"tried": (brain.get(k) or {}).get("tried"), "kept": (brain.get(k) or {}).get("kept"),
                              "recent": [{"change": h.get("change"), "kept": h.get("kept"), "why": h.get("why")}
                                         for h in ((brain.get(k) or {}).get("history") or [])[-6:]]}})
    port = A.port()
    return {"strategies": out, "money_rules": {x: port.get(x) for x in ("account", "risk_pct", "max_position", "max_open", "regime")}}


def atlas_test(A, base, change):
    """A suggestion run over the years at once: its training, validation and locked test, beside the rules now."""
    from . import equity
    with A.eq.lock:
        prepared = A.eq.prepared
    port = A.port()
    money = {x: port[x] for x in ("account", "risk_pct", "max_position", "max_open", "regime")}
    now = dict((A.brain().get(base) or {}).get("params") or {})
    cand = {**now, **change}
    r = equity.backtest(prepared, base, {**cand, **money})
    return {"params": cand, "training": _stats(r["split"]["old"]), "validation": _stats(r["split"]["val_stats"]),
            "locked_test": _stats(r["split"]["new"])}


def atlas_clean(A, raw):
    """Only what the code can run: known strategies, keys of their search space, values from its lists."""
    from . import equity
    from .atlas import SEARCH
    ok = lambda k: k in equity.STRATEGIES

    def params(base, P):
        space = {p for p in SEARCH if p in equity.DEFAULTS or p in equity.STRATEGIES[base]["params"]}
        if base.endswith("_id"):
            space.discard("entry")
        return {p: v for p, v in (P or {}).items() if p in space and v in SEARCH[p]}
    out = []
    for t in raw.get("tweaks") or []:
        if ok(t.get("strategy")) and params(t["strategy"], t.get("params")):
            out.append({"kind": "tweak", "base": t["strategy"], "name": equity.STRATEGIES[t["strategy"]]["name"],
                        "change": params(t["strategy"], t["params"]), "why": str(t.get("why") or "")[:400]})
    for t in raw.get("new") or []:
        if ok(t.get("base")) and params(t["base"], t.get("params")):
            out.append({"kind": "new", "base": t["base"], "name": str(t.get("name") or "")[:80],
                        "change": params(t["base"], t["params"]), "why": str(t.get("why") or "")[:400]})
    return out[:15]


# ---------- Polybot ----------
POLY_PROMPT = """Review the strategies of "Polybot", which trades Polymarket's 5- and 15-minute "up or down" crypto markets
(BTC, ETH...). Every strategy is a rule; the lab runs hundreds of them side by side on every round, each with its record.
A rule's key: resolution|min_edge|min_seconds_left-max_seconds_left, then optional parts: |r<min reward-to-risk ratio>,
|b<price low>-<price high> in cents, |c<min chance %>, |q (a maker filled by its place in the queue), |e<entry style>,
|g<gate>. Example keys from the lab are in the data: copy their exact form.

THE DATA (JSON):
{data}

What to do:
1. The overall verdict: what kind of rule works, what doesn't, on which markets, and why (pessimistic ROI and number of
   bets first: a rule with few bets proves nothing).
2. For the rules running in the simulation or with real money: keep, improve or drop, and why.
3. Suggest new rules for the lab ("new"): up to 12, each a key in the exact form above for a market of the data
   (asset and minutes), each a reasoned change of a rule that shows something (a narrower window, a price band, a
   ratio...). They will be measured in the lab like every other rule before any money touches them.

Answer with ONLY one JSON object inside a ```json block:
{{"report": "the overall verdict in Hebrew, 4-8 sentences",
  "strategies": [{{"key": "asset|minutes|rule", "verdict": "keep|improve|drop", "weak": "Hebrew, short", "why": "Hebrew, short"}}],
  "new": [{{"asset": "btc", "minutes": 5, "key": "...", "why": "Hebrew"}}]}}"""


def poly_data(E):
    board = (getattr(E, "lab", None) or {}).get("board") or []
    keep = ("asset", "minutes", "variant", "bets", "wins", "win_rate", "pnl", "roi", "roi_pessimistic", "status")
    rows = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items() if k in keep} for b in board]
    rows.sort(key=lambda b: -(b.get("roi_pessimistic") if b.get("roi_pessimistic") is not None else -9))
    return {"lab_top": rows[:80], "lab_bottom": rows[-20:], "lab_rules": len(rows),
            "in_simulation": E.strategies_for("sim"), "with_real_money": E.strategies_for("live"),
            "markets": sorted({(b["asset"], b["minutes"]) for b in board} or {(t.asset, t.minutes) for t in E.trackers})}


def poly_clean(E, raw):
    """Only keys the lab can run, on markets it has, not there already."""
    from . import lab
    markets = {(t.asset, t.minutes) for t in E.trackers}
    have = set(getattr(E, "current_variants", None) or ())
    out = []
    for x in raw.get("new") or []:
        try:
            a, m, k = str(x.get("asset", "")).lower(), int(x.get("minutes")), str(x.get("key", ""))
            v = lab.Variant.from_key(k)
            if v.key != k:
                k = v.key
        except Exception:
            continue
        if (a, m) in markets and (a, m, k) not in have and all((a, m, k) != (y["asset"], y["minutes"], y["key"]) for y in out):
            out.append({"asset": a, "minutes": m, "key": k, "why": str(x.get("why") or "")[:400]})
    return out[:12]


def poly_add(E, items, now=None):
    """The new rules into the lab (the bridge's way: always run, whatever the lab's focus)."""
    now = now or time.time()
    B = E.lab_bridge()
    e = B.setdefault("deep|fable", {"track": "deep", "code": "fable", "since": now, "keys": []})
    have = {tuple(k) for k in e["keys"]}
    for x in items:
        if (x["asset"], x["minutes"], x["key"]) not in have:
            e["keys"].append([x["asset"], x["minutes"], x["key"]])
    E.store.set_setting("lab_bridge", json.dumps(B))
    E.__dict__.pop("_hub_cache", None)
    if getattr(E, "current_variants", None) is not None:
        E._refresh_lab_variants()
    return len(e["keys"])
