"""Atlas's live lab (the owner, 28.10.2026): experiment 1 in real time, on the whole market, every trading day.

Step 1, in the simulation: from 9:00 New York the candidates are the hundred stocks of Atlas plus what IBKR's
own scanner finds across every US stock (the most active, the biggest gainers, price $10 and up); their last
weeks of 5-minute bars give each one's normal opening volume and range, its daily range and ATR. At 9:35 the
scanner runs again, the first five minutes of every candidate are read, and the day's nets are picked by
the experiment's own rules (scalp.scan); their minutes are followed from then on as IBKR draws them, and
every few seconds each variant runs on the day so far, with the same code as the history's (scalp.run_day):
nothing is tuned for "live", so what it records here is what the variant does on days it never saw, in real
time. After the close the day is closed, the placebo drawn, the day's minutes kept, and the live board
(the forward test) updated.

Step 2, in the IBKR paper account (its own gateway, beside the real one's): the variants the owner picks send
their entries as real orders the moment the model enters (a limit at the trigger plus the model's own
allowance), with the model's stop resting at IBKR and raised as the model's rises; the model's own exits
(time, stall, the close) go at the market. Every paper trade is set beside the model's same trade: what the
fill cost against the model's (the real slippage), what was missed, and the R each made. That gap is the
answer to "does the lab's money survive real fills?".

The way to real money (stages(), shown per variant; the switch itself is the owner's): the history, then the
forward test (a checkpoint reached, ahead of the placebo, positive), then the paper account (enough filled
trades, positive after real fills, the gap with the model small).
"""

import datetime as dt
import logging
import os
import threading
import time

from . import equity, scalp

log = logging.getLogger("polybot")
NY = scalp.NY

SCANS = ("HOT_BY_VOLUME", "TOP_PERC_GAIN", "MOST_ACTIVE")
SCAN_ROWS = 30  # each scanner's rows
LIVE = tuple(k for k in scalp.VARIANTS if k not in ("V0", "V6"))  # (V0: drawn after the close; V6 needs every stock's minutes)
PREP = 9 * 60  # 9:00: the candidates and their history
SCAN = scalp.ORB_END  # 9:35
END = 16 * 60 + 1  # 16:01: the day closed
FULL_BY = scalp.ORB_END + 1  # a day joined later than this is kept, not counted on the board
CHUNK = 25  # symbols a request to the gateway (other calls go between them)
EVERY = 5  # seconds between two runs of the variants


def ny(now):
    return dt.datetime.fromtimestamp(now, NY)


def minute(now):
    t = ny(now)
    return t.hour * 60 + t.minute


def split_days(bars):
    """[(ts, o, h, l, c, v)] -> {day: [bars]} (New York's days)."""
    out = {}
    for b in bars:
        out.setdefault(scalp.day_of(b[0]), []).append(tuple(b))
    return out


def daily_view(hist, today):
    """From each symbol's 5-minute bars of the weeks before today: its daily ATR (14 days, as of yesterday)
    and whether yesterday's range was the narrowest of 7 (NR7). ({symbol: atr}, {symbols})."""
    atr, nr7 = {}, set()
    for sym, bars in hist.items():
        days = [(d, b) for d, b in sorted(split_days(bars).items()) if d < today]
        if len(days) < 8:
            continue
        hl = [(max(x[2] for x in b), min(x[3] for x in b), b[-1][4]) for _, b in days]
        trs = [hl[i][0] - hl[i][1] if i == 0 else max(hl[i][0], hl[i - 1][2]) - min(hl[i][1], hl[i - 1][2]) for i in range(len(hl))]
        atr[sym] = sum(trs[-14:]) / len(trs[-14:])
        rng = [h - lo for h, lo, _ in hl[-7:]]
        if rng[-1] <= min(rng):
            nr7.add(sym)
    return atr, nr7


class LiveLab:
    def __init__(self, atlas):
        self.a = atlas
        self.d = {}  # today's working state (in memory: the day's history is rebuilt after a restart)
        self.lock = threading.Lock()
        self.paper = Paper(self)
        self.bars = scalp.Bars(os.path.join(os.path.dirname(os.path.abspath(atlas.store.path)), "atlas_minutes_ibkr"))

    # ---------- the day ----------
    def tick(self, now=None):
        now = now or time.time()
        t = ny(now)
        S = self.a.settings()
        if not S.get("livelab_on", True) or t.weekday() >= 5:
            return None
        gw = self.a.ibkr
        if gw is None or not gw.connected():
            self._status("no_gateway", now)
            return None
        today, m = t.date().isoformat(), minute(now)
        with self.lock:
            if self.d.get("day") != today:
                self.d = {"day": today, "phase": "idle", "errors": []}
            D = self.d
            try:
                if D["phase"] == "idle" and PREP <= m < END:
                    self.prep(gw, now)
                if D["phase"] == "ready" and m >= SCAN and (m > SCAN or t.second >= 3) and m < END:
                    self.scan(gw, now)
                if D["phase"] == "live":
                    if m >= END:
                        self.finish(gw, now)
                    elif now - D.get("ran", 0) >= EVERY:
                        self.run(gw, now)
            except Exception as e:
                D["errors"] = (D.get("errors") or [])[-4:] + [f"{type(e).__name__}: {e}"[:300]]
                log.warning("Live lab: %s", e)
                if D["phase"] in ("idle", "ready") and len(D["errors"]) >= 5:
                    D["phase"] = "failed"
            self._save(now)
        return self.d.get("phase")

    def _chunks(self, gw, symbols, duration, size):
        out = {}
        symbols = list(symbols)
        for i in range(0, len(symbols), CHUNK):
            out.update(gw.bars_many(symbols[i:i + CHUNK], duration, size))
        return out

    def _scanner(self, gw, D):
        found = {}
        for code in SCANS:
            try:
                found[code] = gw.scanner(code, SCAN_ROWS)
            except Exception as e:
                D["scanner_error"] = f"{code}: {type(e).__name__}: {e}"[:200]
        return found

    def prep(self, gw, now):
        """The candidates (Atlas' hundred + the scanner's) and their weeks before: the normal opening volume
        and range (scalp.baselines), the daily ATR and NR7."""
        D = self.d
        D["phase"] = "prep"
        found = self._scanner(gw, D)
        extra = [s for code in SCANS for s in found.get(code, []) if s not in equity.UNIVERSE]
        cands = list(dict.fromkeys(list(equity.UNIVERSE) + extra))
        hist = self._chunks(gw, cands + ["SPY"], "25 D", "5 mins")
        self._history(hist, D["day"])
        D.update(phase="ready", scanner={k: v[:SCAN_ROWS] for k, v in found.items()}, candidates=sorted(hist),
                 prepared=now)
        log.warning("Live lab: %d candidates (%d from the scanner), %d with a baseline", len(cands), len(extra), len(D["base"]))

    def _history(self, hist, today):
        D = self.d
        days = {}
        for sym, bars in hist.items():
            for d, b in split_days(bars).items():
                if d < today:
                    days.setdefault(d, {})[sym] = b
        history = [days[d] for d in sorted(days)]
        base = scalp.baselines(history)
        atr, nr7 = daily_view(hist, today)
        D["base"] = {**D.get("base", {}), **base}
        D["atr"] = {**D.get("atr", {}), **atr}
        D["nr7"] = sorted(set(D.get("nr7", [])) | (nr7 & set(equity.UNIVERSE)))

    def scan(self, gw, now):
        """9:35: the scanner again (the session's own movers), the first five minutes of every candidate,
        each variant's net (scalp.scan), and the nets' minutes followed from now on."""
        D = self.d
        found = self._scanner(gw, D)
        new = [s for code in SCANS for s in found.get(code, []) if s not in D.get("candidates", []) and s not in equity.UNIVERSE]
        if new:
            self._history(self._chunks(gw, new, "25 D", "5 mins"), D["day"])
        for code, syms in found.items():
            D.setdefault("scanner", {})[code] = list(dict.fromkeys(D.get("scanner", {}).get(code, []) + syms))[:SCAN_ROWS * 2]
        cands = sorted(set(D.get("candidates", [])) | set(new))
        today = {s: [b for b in bars if scalp.day_of(b[0]) == D["day"]] for s, bars in self._chunks(gw, cands, "1 D", "1 min").items()}
        net, nets = set(), {}
        for k in LIVE:
            P = scalp.VARIANTS[k]
            got = scalp.scan(today, D["base"], P, set(equity.UNIVERSE) if P["universe_only"] else None, set(D["nr7"]))
            nets[k] = sorted(got)
            net |= set(got)
            if k == "V1":
                D["net"] = [{x: (round(v, 2) if isinstance(v, float) else v) for x, v in i.items()} for i in got.values()]
        follow = sorted(net | {"SPY"})
        D["follow"] = gw.follow(follow)
        D.update(phase="live", joined=minute(now), nets=nets, candidates=cands, scanned=now,
                 full=minute(now) <= FULL_BY)
        log.warning("Live lab: %d in the nets, following %d", len(net), len(D["follow"]))

    def _today(self, gw, now, closed_only):
        D, m = self.d, minute(now)
        day = {}
        for sym, bars in gw.followed().items():
            got = [b for b in bars if scalp.day_of(b[0]) == D["day"] and scalp.OPEN <= scalp.minute_of(b[0]) < 960]
            if closed_only:
                got = [b for b in got if scalp.minute_of(b[0]) < m]
            if got:
                day[sym] = got
        return day

    def _variant(self, k, day, partial, placebo_n=None):
        D, P = self.d, scalp.VARIANTS[k]
        return scalp.run_day(day, D["base"], P, D.get("atr"), set(equity.UNIVERSE) if P["universe_only"] else None,
                             seed=D["day"], placebo_n=placebo_n, nr7=set(D.get("nr7") or ()), partial=partial)

    def run(self, gw, now):
        """Every variant on the day so far (closed minutes: what the record keeps), and for the paper account's
        variants on the minute still forming too (its entries go the moment the model's would)."""
        D = self.d
        day = self._today(gw, now, True)
        D["rows"] = {k: self._variant(k, day, True) for k in LIVE} if day else {}
        D["ran"] = now
        picked = [k for k in (self.a.settings().get("livelab_paper") or []) if k in LIVE]
        if picked and self.paper.ready():
            live = self._today(gw, now, False)
            prov = {k: self._variant(k, live, True) for k in picked}
            self.paper.sync(D["day"], D["rows"], prov, now)

    def finish(self, gw, now):
        """After the close: the day's final run (the placebo drawn on the base's count, as in the history),
        kept with the day's minutes, the board again, the subscriptions stopped."""
        D = self.d
        day = self._today(gw, now, False)
        if not day.get("SPY"):  # (a holiday, or nothing came: not a day of the record)
            D.update(phase="done", finished=now, rows={})
            try:
                gw.follow([])
            except Exception:
                pass
            return
        final = {}
        n_base = None
        for k in ("V1",) + tuple(x for x in LIVE if x != "V1") + ("V0",):
            final[k] = self._variant(k, day, False, placebo_n=n_base)
            if k == "V1":
                n_base = len(final[k])
        self.a._set(f"livelab_day:{D['day']}", final)
        idx = [x for x in self.a._get("livelab_days", []) if x["day"] != D["day"]]
        idx.append({"day": D["day"], "full": bool(D.get("full")), "joined": D.get("joined"), "net": len(D.get("net") or []),
                    "follow": len(D.get("follow") or [])})
        self.a._set("livelab_days", sorted(idx, key=lambda x: x["day"])[-500:])
        if day:
            self.bars.save(D["day"], {s: [list(b) for b in bars] for s, bars in day.items()})
        self.a._set("livelab_board", self.board())
        try:
            gw.follow([])
        except Exception as e:
            log.warning("Live lab: unfollow: %s", e)
        self.paper.end_of_day(now)
        D.update(phase="done", rows=final, finished=now)

    def board(self):
        days = [x["day"] for x in self.a._get("livelab_days", []) if x.get("full")]
        res = {k: [] for k in ("V0",) + LIVE}
        for d in days:
            for k, rows in (self.a._get(f"livelab_day:{d}") or {}).items():
                if k in res:
                    res[k] += [dict(r, day=d) for r in rows]
        return {"rows": scalp.board(res, days), "days": days, "ts": time.time()}

    # ---------- the page ----------
    def _status(self, phase, now):
        if (self.a._get("livelab") or {}).get("phase") != phase:
            self.a._set("livelab", {"phase": phase, "ts": now})

    def _save(self, now):
        D = self.d
        if now - D.get("saved", 0) < 4 and D.get("phase") == D.get("saved_phase"):
            return
        D["saved"], D["saved_phase"] = now, D.get("phase")
        self.a._set("livelab", {k: D.get(k) for k in ("day", "phase", "joined", "full", "net", "nets", "follow", "scanner",
                                                      "scanner_error", "errors", "rows", "prepared", "scanned", "finished")}
                    | {"candidates": len(D.get("candidates") or []), "base": len(D.get("base") or {}),
                       "nr7": D.get("nr7"), "ts": now})

    def stages(self, board, paper):
        """Each variant's way to real money: the history (experiment 1's board), the forward test here, the
        paper account. {variant: {"history", "live", "paper", "ready"}} (each: "pass" / "fail" / "wait")."""
        hist = {r["variant"]: r for r in ((self.a._get("lab") or {}).get("board") or [])}
        live = {r["variant"]: r for r in (board or {}).get("rows") or []}
        out = {}
        for k in LIVE:
            h, L, p = hist.get(k) or {}, live.get(k) or {}, (paper or {}).get(k) or {}
            hs = "pass" if (h.get("vs_placebo") or 0) >= 2 and (h.get("r_per_trade") or 0) > 0 and h.get("status") != "collecting" \
                else "fail" if h.get("status") != "collecting" and h.get("trades") else "wait"
            ls = "wait" if L.get("status", "collecting") == "collecting" else \
                "pass" if (L.get("vs_placebo") or 0) >= 2 and (L.get("r_per_trade") or 0) > 0 else "fail"
            ps = "wait" if (p.get("closed") or 0) < self.PAPER_NEED else \
                "pass" if (p.get("r_actual") or 0) > 0 and (p.get("gap_r") or 0) < 0.5 * max(1e-9, abs(p.get("r_model") or 0)) else "fail"
            out[k] = {"history": hs, "live": ls, "paper": ps, "ready": hs == ls == ps == "pass"}
        return out

    PAPER_NEED = 40  # paper trades closed before its verdict

    def view(self):
        v = self.a._get("livelab") or {}
        board = self.a._get("livelab_board") or {"rows": [], "days": []}
        paper = self.paper.view()
        return {**v, "board": board, "days_index": self.a._get("livelab_days", [])[-60:], "paper": paper,
                "stages": self.stages(board, paper.get("by_variant")), "names": {k: scalp.NAMES[k] for k in ("V0",) + LIVE},
                "live_variants": list(LIVE), "picked": self.a.settings().get("livelab_paper") or []}


class Paper:
    """Step 2: the picked variants' trades as real orders in the IBKR paper account (a gateway of its own)."""
    CLIENT_ID = 9
    MISS_AFTER = 90  # seconds an entry may wait at its limit (the price ran away: missed)
    OUT_AFTER = 120  # seconds the model's stop may have been hit before ours is sold at the market
    LAST_CALL = 15 * 60 + 57  # anything still open is sold

    def __init__(self, lab):
        self.lab, self.a = lab, lab.a
        self.gw = None
        self.error = None

    @staticmethod
    def host():
        if os.environ.get("IBKR_PAPER_HOST"):
            return os.environ["IBKR_PAPER_HOST"]
        app = os.environ.get("FLY_APP_NAME")
        return f"{app}-ibgw-paper.internal" if app else None

    def ready(self):
        """The paper gateway connected to a paper account (DU...); else why not in self.error."""
        from . import ibkr
        if self.gw is None:
            h = self.host()
            if not h:
                self.error = "אין שרת Paper (IBKR_PAPER_HOST)"
                return False
            self.gw = ibkr.IbGateway(host=h, port=os.environ.get("IBKR_PAPER_PORT") or ibkr.PORT, client_id=self.CLIENT_ID)
        try:
            accounts = self.gw.connect()
        except Exception as e:
            self.error = f"אין חיבור לשרת ה-Paper ({self.gw.host}): {type(e).__name__}: {e}"[:300]
            return False
        problem = ibkr.account_problem(accounts, "paper")
        self.error = problem
        self.account = accounts[0] if accounts else None
        return problem is None

    def _st(self):
        return self.a._get("livelab_paper", {"orders": {}})

    def sync(self, day, closed_rows, prov_rows, now):
        """The model's trades (prov_rows: with the minute forming; closed_rows: the record's) into the paper
        account and the paper account's fills beside them."""
        gw, st = self.gw, self._st()
        orders, m = st["orders"], minute(now)
        # 1. new entries: the moment the model is in
        for k, rows in prov_rows.items():
            for r in rows:
                key = f"{day}:{k}:{r['symbol']}:{r['in']}"
                if not r.get("open") or r.get("exiting") or key in orders or m >= self.LAST_CALL - 5:
                    continue
                limit = round(r["entry"] + max(0.02, 0.0005 * r["entry"]), 2)
                try:
                    ids = gw.place_bracket(r["symbol"], int(r["shares"]), None, None, r["stop"], ("limit", limit))
                except Exception as e:
                    orders[key] = {"key": key, "variant": k, "symbol": r["symbol"], "status": "error", "why": str(e)[:200], "sent": now, "day": day}
                    continue
                orders[key] = {"key": key, "variant": k, "day": day, "symbol": r["symbol"], "qty": int(r["shares"]),
                               "model_entry": r["entry"], "limit": limit, "stop": r["stop"], "ids": ids, "status": "sent", "sent": now}
        # 2. where each stands
        mine = [o for o in orders.values() if o.get("day") == day and o["status"] in ("sent", "open")]
        if mine:
            known = gw.orders()
            model = {f"{day}:{k}:{r['symbol']}:{r['in']}": r for k, rows in closed_rows.items() for r in rows}
            model.update({f"{day}:{k}:{r['symbol']}:{r['in']}": r for k, rows in prov_rows.items() for r in rows
                          if not r.get("open") and r.get("why") in ("stop", "trail")})
            for o in mine:
                par = known.get(o["ids"]["parent"])
                if o["status"] == "sent":
                    if par and par["status"] == "Filled":
                        o.update(status="open", entry=round(par["avg_price"], 4), filled_at=now,
                                 slip=round(par["avg_price"] - o["model_entry"], 4))
                    elif par and par["status"] == "Inactive":
                        o.update(status="rejected", why=par.get("why") or "Inactive")
                    elif now - o["sent"] > self.MISS_AFTER:
                        gw.cancel(o["ids"]["parent"])
                        o.update(status="missed", why="המחיר ברח מעל הלימיט")
                    continue
                stop_leg = known.get(o["ids"].get("stop"))
                exit_leg = known.get(o["ids"].get("exit")) if o["ids"].get("exit") else None
                done = exit_leg if exit_leg and exit_leg["status"] == "Filled" else stop_leg if stop_leg and stop_leg["status"] == "Filled" else None
                if done:
                    px = done["avg_price"]
                    fees = 2 * scalp.commission(o["qty"], px) + scalp.sell_fees(o["qty"], px)
                    pnl = (px - o["entry"]) * o["qty"] - fees
                    r = model.get(o["key"]) or {}
                    o.update(status="closed", exit=round(px, 4), pnl=round(pnl, 2), r_actual=round(pnl / scalp.R, 3),
                             r_model=r.get("r") if not r.get("open") else None, why=r.get("why") or o.get("why"), closed=now)
                    continue
                r = model.get(o["key"])
                if r and not r.get("open"):  # (the model is out)
                    if r["why"] in ("stop", "trail") and not o.get("exit_sent"):
                        if now - (o.get("model_out_seen") or now) > self.OUT_AFTER:
                            self._sell(gw, o, now, "stop")
                        o.setdefault("model_out_seen", now)
                    elif not o.get("exit_sent"):
                        self._sell(gw, o, now, r["why"])
                elif r and r.get("stop") and r["stop"] > o["stop"] + 0.005 and o["ids"].get("stop"):
                    if gw.modify(o["ids"]["stop"], auxPrice=float(r["stop"])):
                        o["stop"] = r["stop"]
                if m >= self.LAST_CALL and not o.get("exit_sent"):
                    self._sell(gw, o, now, "close")
        self.a._set("livelab_paper", st)

    def _sell(self, gw, o, now, why):
        if o["ids"].get("stop"):
            gw.cancel(o["ids"]["stop"])
        o["ids"]["exit"] = gw.market(o["symbol"], "SELL", o["qty"])
        o.update(exit_sent=now, why=why)

    def end_of_day(self, now):
        st = self._st()
        for o in st["orders"].values():
            if o["status"] == "sent":
                o.update(status="missed", why="לא התמלא עד הסגירה")
        keep = sorted(st["orders"].values(), key=lambda o: o["sent"])[-3000:]
        st["orders"] = {o["key"]: o for o in keep}
        self.a._set("livelab_paper", st)

    def view(self):
        orders = list(self._st()["orders"].values())
        by = {}
        for o in orders:
            x = by.setdefault(o["variant"], {"sent": 0, "filled": 0, "missed": 0, "closed": 0, "slip": [], "r_actual": [], "r_model": []})
            x["sent"] += 1
            if o["status"] in ("open", "closed"):
                x["filled"] += 1
                x["slip"].append(o.get("slip") or 0.0)
            if o["status"] == "missed":
                x["missed"] += 1
            if o["status"] == "closed":
                x["closed"] += 1
                x["r_actual"].append(o.get("r_actual") or 0.0)
                if o.get("r_model") is not None:
                    x["r_model"].append(o["r_model"])
        avg = lambda xs: sum(xs) / len(xs) if xs else None
        out = {}
        for k, x in by.items():
            ra, rm = avg(x["r_actual"]), avg(x["r_model"])
            out[k] = {"sent": x["sent"], "filled": x["filled"], "missed": x["missed"], "closed": x["closed"],
                      "slip": avg(x["slip"]), "r_actual": ra, "r_model": rm,
                      "gap_r": (rm - ra) if ra is not None and rm is not None else None,
                      "pnl": round(sum(o.get("pnl") or 0 for o in orders if o["variant"] == k), 2)}
        return {"host": self.host(), "error": self.error, "account": getattr(self, "account", None),
                "by_variant": out, "recent": sorted(orders, key=lambda o: -o["sent"])[:40]}
