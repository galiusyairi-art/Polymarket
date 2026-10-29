"""Experiment E (the panels, 29.10.2026): does Binance's futures price move before its spot price, the fast
source the strategies read now? If it does, it is an earlier signal, for free.

Every coin's three prices as they arrive here (the arrival time: the same way from Binance's servers for spot
and futures, so the comparison is fair): Binance spot, Binance USD-M futures and Coinbase. A move starts an
event: spot moving MOVE or more within a second. Its moment is when spot crossed half of it; a source's is
when it crossed half of the same move (from its own price a second before). The lead: how much sooner the
futures (and Coinbase) got there. Read only: it never trades.
"""

import collections
import statistics
import threading
import time

MOVE = 0.0004  # 4 basis points within a second: a move worth racing for
LOOK = 1.0
AFTER = 2.0  # seconds after the move a source may still catch up (later: it didn't follow)
COOLDOWN = 5.0
KEEP = 3000
SOURCES = ("binance_fut", "coinbase")


class LeadWatch:
    def __init__(self, events=None):
        self.lock = threading.Lock()
        self.recent = {}  # (source, asset) -> deque of (arrival, price)
        self.events = collections.deque(events or [], maxlen=KEEP)
        self.pending = []  # events waiting for the sources to catch up
        self.last = {}

    def add(self, source, asset, price, now=None):
        now = now or time.time()
        with self.lock:
            d = self.recent.setdefault((source, asset), collections.deque())
            d.append((now, price))
            while d and d[0][0] < now - 10:
                d.popleft()
            self._settle(now)
            if source == "binance":
                self._detect(asset, now)

    @staticmethod
    def _at(d, t):
        """The price at or before t."""
        p = None
        for ts, x in d:
            if ts > t:
                break
            p = x
        return p

    @staticmethod
    def _cross(d, ref, sign, t0, t1):
        """When a source's price first went half the move past ref (sign: +1 up, -1 down), within [t0, t1]."""
        half = ref * (1 + sign * MOVE / 2)
        for ts, x in d:
            if ts < t0:
                continue
            if ts > t1:
                break
            if (x - half) * sign >= 0:
                return ts
        return None

    def _detect(self, asset, now):
        d = self.recent[("binance", asset)]
        p0 = self._at(d, now - LOOK)
        if not p0 or now - self.last.get(asset, 0) < COOLDOWN:
            return
        move = d[-1][1] / p0 - 1
        if abs(move) < MOVE:
            return
        sign = 1 if move > 0 else -1
        t_spot = self._cross(d, p0, sign, now - LOOK, now)
        if t_spot is None:
            return
        self.last[asset] = now
        self.pending.append({"asset": asset, "ts": now, "spot": t_spot, "sign": sign, "start": now - LOOK})

    def _settle(self, now):
        keep = []
        for e in self.pending:
            if now - e["ts"] < AFTER:
                keep.append(e)
                continue
            out = {"asset": e["asset"], "ts": round(e["ts"], 3)}
            for s in SOURCES:
                d = self.recent.get((s, e["asset"]))
                ref = self._at(d, e["start"]) if d else None
                t = self._cross(d, ref, e["sign"], e["start"], e["ts"] + AFTER) if ref else None
                out[s] = round((e["spot"] - t) * 1000, 1) if t is not None else None  # ms sooner than spot (+: first)
            self.events.append(out)
        self.pending = keep

    def view(self):
        with self.lock:
            ev = list(self.events)
        out = {"events": len(ev), "move_bp": MOVE * 1e4}
        for s in SOURCES:
            xs = [e[s] for e in ev if e.get(s) is not None]
            out[s] = {"n": len(xs), "followed": len(xs) / len(ev) if ev else None,
                      "median": statistics.median(xs) if xs else None,
                      "p25": sorted(xs)[len(xs) // 4] if xs else None, "p75": sorted(xs)[3 * len(xs) // 4] if xs else None,
                      "first": sum(1 for x in xs if x >= 20) / len(xs) if xs else None}  # (20 ms or more before spot)
        F = out["binance_fut"]
        out["state"] = ("collecting" if F["n"] < 200 else "found" if (F["median"] or 0) >= 20 and (F["first"] or 0) >= 0.5
                        else "nothing" if (F["median"] or 0) < 5 else "promising")
        out["progress"] = min(1.0, F["n"] / 200)
        return out

    def dump(self):
        with self.lock:
            return list(self.events)
