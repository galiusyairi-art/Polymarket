"""Atlas: the best trade size for the account, and the best number of trades open at once (the owner's
questions of 29.10.2026: "a billion trades at once isn't sensible, but nine and then nothing more until they
close isn't good either; only the research can tell").

Every strategy in use together, as one account of the size set in the settings (the minimum commission and
the cash make a small account differ from a large one), over the whole history, one setting at a time:

  - the size: the risk a trade (0.25% to 2% of the account, the owner's rule's range) and the largest
    position (5% to 25% of it), at the number open now;
  - the number open: 3 to 30, at the size the first study chose.

Each is chosen on the first 80% of the history (the training and the validation) and judged on the last 20%,
the locked test that chose nothing. The choice: the most a year, among the settings whose worst fall stayed
under two thirds of DD_LIMIT (the future's falls are deeper than the past's), each setting's return averaged
with its neighbours' (a peak that stands alone is luck, not an answer; a neighbour that fell too far counts as
nothing). An answer holds if the locked test
made money and fell less than DD_LIMIT; then it becomes every strategy's (atlas.py, sizing_apply).
"""

import math

RISKS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)  # % of the account at risk a trade
POSITIONS = (5, 10, 15, 20, 25)  # the largest position, % of the account
SLOTS = (3, 5, 8, 10, 12, 15, 20, 25, 30)  # trades open at once
DD_LIMIT = 0.20  # the most the account may fall from its peak
CHOOSE = 0.8  # the part of the history the choice is made on (the rest: the locked test)


def _part(curve, start, end):
    return [(d, v) for d, v in curve if start <= d < end]


def measure(curve, trades, account, start="", end="9999"):
    """A stretch of the account's curve: the return a year (on the account), the worst fall from a peak, the
    daily Sharpe a year, and the trades opened in it."""
    c = _part(curve, start, end)
    if len(c) < 20:
        return None
    base = c[0][1]
    peak, dd = base, 0.0
    rets = []
    for (_, a), (_, b) in zip(c, c[1:]):
        rets.append((b - a) / account)
    for _, v in c:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak if peak else 0.0)
    years = len(c) / 252
    gain = (c[-1][1] - base) / account
    mean = sum(rets) / len(rets) if rets else 0.0
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) if len(rets) > 1 else 0.0
    n = [t for t in trades if start <= t["in"] < end]
    return {"a_year": gain / years if years else None, "gain": gain, "dd": dd,
            "sharpe": mean / sd * math.sqrt(252) if sd else None, "trades": len(n),
            "win": sum(1 for t in n if t["pnl"] > 0) / len(n) if n else None, "days": len(c)}


def run(prepared, plans, port, portfolio):
    """One setting: its whole-history account (portfolio: equity.portfolio), the choice's stretch and the locked
    test's."""
    r = portfolio(prepared, {k: {**P, **_money(port)} for k, P in plans.items()}, port)
    curve, trades = r["curve"], r["trades"]
    days = [d for d, _ in curve]
    if len(days) < 60:
        return None
    cut = days[int(len(days) * CHOOSE)]
    return {"choose": measure(curve, trades, port["account"], end=cut), "test": measure(curve, trades, port["account"], start=cut),
            "all": measure(curve, trades, port["account"]), "cut": cut,
            "skipped": len(r.get("skipped") or [])}


def _money(port):
    return {x: port[x] for x in ("account", "risk_pct", "max_position", "max_open")}


def _safe(m, dd):
    return m and m["dd"] <= dd * 2 / 3 and m["a_year"] is not None


def pick(rows, neighbours, dd=DD_LIMIT):
    """The setting to take: the most a year on the choice's stretch, with its neighbours (neighbours(i): the
    indexes beside row i), among the safe ones. (index or None, the smoothed returns)"""
    smooth = []
    for i, r in enumerate(rows):
        m = r["res"] and r["res"]["choose"]
        if not _safe(m, dd):
            smooth.append(None)
            continue
        # (a neighbour that fell too far counts as nothing: a setting beside the edge is a step from it)
        near = [(rows[j]["res"]["choose"]["a_year"] if _safe(rows[j]["res"] and rows[j]["res"]["choose"], dd) else 0.0)
                for j in neighbours(i)]
        smooth.append((m["a_year"] + sum(near)) / (1 + len(near)))
    ok = [i for i, s in enumerate(smooth) if s is not None and s > 0]
    return (max(ok, key=lambda i: smooth[i]) if ok else None), smooth


def verdict(row, dd=DD_LIMIT):
    if row is None:
        return "nothing"
    t = row["res"]["test"]
    if not t or t["trades"] < 10:
        return "promising"
    return "found" if t["gain"] > 0 and t["dd"] < dd else "nothing"


def _row(res, **k):
    return dict(k, res=res)


def study_size(prepared, plans, port, portfolio, dd=DD_LIMIT):
    rows = []
    for r in RISKS:
        for p in POSITIONS:
            P = dict(port, risk_pct=r, max_position=port["account"] * p / 100)
            rows.append(_row(run(prepared, plans, P, portfolio), risk_pct=r, position_pct=p))
    at = {(x["risk_pct"], x["position_pct"]): i for i, x in enumerate(rows)}

    def near(i):
        a, b = RISKS.index(rows[i]["risk_pct"]), POSITIONS.index(rows[i]["position_pct"])
        return [at[(RISKS[x], POSITIONS[y])] for x, y in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1))
                if 0 <= x < len(RISKS) and 0 <= y < len(POSITIONS)]
    best, smooth = pick(rows, near, dd)
    now = next((i for i, x in enumerate(rows) if x["risk_pct"] == port["risk_pct"]
                and abs(x["position_pct"] - port["max_position"] / port["account"] * 100) < 1e-6), None)
    return {"rows": [dict(x, smooth=s) for x, s in zip(rows, smooth)], "best": best, "now": now,
            "state": verdict(rows[best] if best is not None else None),
            "answer": {"risk_pct": rows[best]["risk_pct"], "max_position": port["account"] * rows[best]["position_pct"] / 100,
                       "position_pct": rows[best]["position_pct"]} if best is not None else None}


def study_slots(prepared, plans, port, portfolio, dd=DD_LIMIT):
    rows = [_row(run(prepared, plans, dict(port, max_open=n), portfolio), max_open=n) for n in SLOTS]
    best, smooth = pick(rows, lambda i: [j for j in (i - 1, i + 1) if 0 <= j < len(rows)], dd)
    now = next((i for i, x in enumerate(rows) if x["max_open"] == int(port["max_open"])), None)
    return {"rows": [dict(x, smooth=s) for x, s in zip(rows, smooth)], "best": best, "now": now,
            "state": verdict(rows[best] if best is not None else None),
            "answer": {"max_open": rows[best]["max_open"]} if best is not None else None}


def study(prepared, plans, port, portfolio, dd=DD_LIMIT):
    """Both, the size first (at the number open now), then the number open at the size it chose (if it chose).
    dd: the most the account may fall (DD_LIMIT, or the owner's)."""
    size = study_size(prepared, plans, port, portfolio, dd)
    at = dict(port)
    if size["answer"] and size["state"] == "found":
        at.update(risk_pct=size["answer"]["risk_pct"], max_position=size["answer"]["max_position"])
    slots = study_slots(prepared, plans, at, portfolio, dd)
    return {"size": size, "slots": slots, "account": port["account"], "strategies": sorted(plans),
            "dd_limit": dd, "choose": CHOOSE, "risks": RISKS, "positions": POSITIONS}


def _go(conn, prepared, plans, port, dd):
    """The study in a process of its own, at the lowest priority (the trading never waits behind it)."""
    import os
    try:
        os.nice(15)
    except Exception:
        pass
    from . import equity
    try:
        conn.send(("ok", study(prepared, plans, port, equity.portfolio, dd)))
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}"[:300]))
    finally:
        conn.close()


def study_apart(prepared, plans, port, dd=DD_LIMIT, timeout=3600):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_go, args=(b, prepared, plans, port, dd), daemon=True, name="atlas-sizing")
    p.start()
    b.close()
    try:
        if not a.poll(timeout):
            raise TimeoutError("the sizing study took too long")
        what, out = a.recv()
    finally:
        p.join(5)
        if p.is_alive():
            p.terminate()
    if what != "ok":
        raise RuntimeError(out)
    return out
