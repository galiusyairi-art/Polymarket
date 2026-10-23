"""The control center's measurements: is every part of the system working, how fast does it
react, and does what it sees match Polymarket?

Everything here is measured, not assumed:
- reaction time: from an update arriving to the decision on it, every evaluation;
- event-loop delay: how late the 4-per-second heartbeat wakes up (a backlog shows here first);
- feed ages and update rates;
- an independent check every minute: our live order book against Polymarket's own REST
  answer for the same market, and our clock against Polymarket's;
- restarts, warnings, CPU, memory and disk.
"""
import collections
import json
import logging
import os
import resource
import threading
import time

from . import feeds

log = logging.getLogger("polybot")


def _pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class RecentLog(logging.Handler):
    """Keeps the latest warnings for the control center."""

    def __init__(self, size=50):
        super().__init__(logging.WARNING)
        self.items = collections.deque(maxlen=size)

    def emit(self, record):
        self.items.append({"ts": record.created, "message": record.getMessage()})


class Health:
    WINDOW = 300  # seconds of history behind the latency figures

    def __init__(self, engine):
        self.engine = engine
        self.latency = collections.deque()  # (ts, seconds from update to decision)
        self.loop_lag = collections.deque()  # (ts, seconds the heartbeat woke up late)
        self.counts = collections.Counter()  # update counts since the last rate snapshot
        self.rates = {}
        self._rate_at = time.time()
        self.book_checks = collections.deque(maxlen=24 * 60)  # one entry per market per minute
        self.clock_offset = None
        self.fast_last = {}  # source -> asset -> when its latest price arrived
        self.log = RecentLog()
        logging.getLogger("polybot").addHandler(self.log)
        self.lock = threading.Lock()
        starts = json.loads(engine.store.get_setting("starts", "[]"))
        starts = [t for t in starts if t > time.time() - 7 * 86400] + [time.time()]
        engine.store.set_setting("starts", json.dumps(starts))
        self.starts = starts

    # Recorded from the event loop (cheap).
    def record_latency(self, now, seconds):
        self.latency.append((now, seconds))
        while self.latency and self.latency[0][0] < now - self.WINDOW:
            self.latency.popleft()

    def record_loop_lag(self, now, seconds):
        self.loop_lag.append((now, seconds))
        while self.loop_lag and self.loop_lag[0][0] < now - self.WINDOW:
            self.loop_lag.popleft()

    def count(self, kind):
        self.counts[kind] += 1

    def fast_seen(self, source, asset):
        self.fast_last.setdefault(source, {})[asset] = time.time()

    # Every minute, from a worker thread.
    def check(self):
        now = time.time()
        span = now - self._rate_at
        # Swap in a fresh counter rather than reading one the event loop is still adding to.
        counts, self.counts = self.counts, collections.Counter()
        self._rate_at = now
        if span > 0:
            self.rates = {k: v / span for k, v in list(counts.items())}
        self._check_books(now)
        self._check_clock()

    def _check_books(self, now):
        """Our streamed book vs Polymarket's REST book for each main market, read back to back."""
        engine = self.engine
        if not engine.books:
            return
        for t in engine.main_trackers:
            market = t.market
            if not market:
                continue
            token = market["up_token"]
            # The REST call takes a few hundred ms and the market can move meanwhile, so our
            # view is read just before and just after it, and the closer one is compared.
            before = engine.books.get(token)
            try:
                bids, asks = feeds.order_book(token)
            except Exception as e:
                log.warning("Book cross-check failed for %s: %s", t.key, e)
                continue
            after = engine.books.get(token)
            rest = (max((p for p, _ in bids), default=None), min((p for p, _ in asks), default=None))
            best = None
            for ours in (before, after):
                if not ours:
                    continue
                top = (max((p for p, _ in ours[0]), default=None), min((p for p, _ in ours[1]), default=None))
                gaps = [abs(a - b) for a, b in zip(rest, top) if a is not None and b is not None]
                if gaps and (best is None or max(gaps) < best[0]):
                    best = (max(gaps), top)
            if best:
                self.book_checks.append({"ts": now, "market": t.key, "gap": best[0],
                                         "ours": list(best[1]), "polymarket": list(rest)})

    def _check_clock(self):
        """Polymarket's server clock against ours (it reports whole seconds, so ±0.5s)."""
        try:
            t0 = time.time()
            server = float(feeds.get_json(f"{feeds.CLOB}/time"))
            t1 = time.time()
            self.clock_offset = server + 0.5 - (t0 + t1) / 2
        except Exception as e:
            log.warning("Clock check failed: %s", e)

    # For the dashboard.
    def report(self, now):
        engine = self.engine
        # The event loop appends to these while the web thread reads them: copy first
        # (list() of a deque is a single step), then iterate the copies.
        lat = [s for _, s in list(self.latency)]
        lag = [s for _, s in list(self.loop_lag)]
        day = [c for c in list(self.book_checks) if c["ts"] > now - 86400]
        within = sum(1 for c in day if c["gap"] <= 0.011)
        prices = engine.prices
        books = engine.books
        snaps = [t.snapshot for t in engine.main_trackers]
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu = usage.ru_utime + usage.ru_stime
        try:
            db_mb = os.path.getsize(engine.cfg.db_path) / 1e6
        except OSError:
            db_mb = None
        checks = []

        def check(name, ok, value, detail):
            checks.append({"name": name, "ok": ok, "value": value, "detail": detail})

        price_age = now - prices.last_price if prices and prices.last_price else None
        check("price", price_age is not None and price_age < 5, price_age,
              "מחירי Chainlink מגיעים (בדרך כלל כל שנייה)")
        sources = {}
        for source, by_asset in list(self.fast_last.items()):
            by_asset = dict(by_asset)
            sources[source] = {"age": now - max(by_asset.values()), "assets": len(by_asset),
                               "rate": self.rates.get(source)}
        direct = [s["age"] for name, s in sources.items() if name != "relay"]
        fast_age = min(direct) if direct else (sources["relay"]["age"] if "relay" in sources else None)
        check("fast", fast_age is not None and fast_age < 5, fast_age,
              "מחירי בורסות מגיעים (Binance ו-Coinbase ישירות, ובגיבוי דרך פולימרקט)")
        book_age = max((s.get("book_age") or 0 for s in snaps), default=None) if snaps else None
        check("books", book_age is not None and book_age < 3, book_age, "ספרי ההזמנות של פולימרקט עדכניים")
        p95 = _pct(lat, 0.95)
        check("reaction", p95 is not None and p95 < 0.25, p95, "זמן תגובה: מעדכון שמגיע עד החלטה")
        lag95 = _pct(lag, 0.95)
        check("loop", lag95 is not None and lag95 < 0.2, lag95, "המערכת לא עמוסה (אין תור של עדכונים)")
        match = within / len(day) if day else None
        # Judged from 20 checks on (about 7 minutes); before that a single moving market sways it.
        check("match", len(day) < 20 or match >= 0.9, match,
              "ספר ההזמנות שלנו זהה לזה של פולימרקט" + (" (צובר בדיקות)" if len(day) < 20 else ""))
        check("clock", self.clock_offset is not None and abs(self.clock_offset) < 1.5, self.clock_offset,
              "השעון שלנו מסונכרן עם פולימרקט")
        alerts = engine._alerts(now)
        check("alerts", not alerts, len(alerts), "אין התראות פתוחות")
        return {
            "ok": all(c["ok"] for c in checks),
            "checks": checks,
            "reaction": {"p50": _pct(lat, 0.5), "p95": p95, "max": max(lat, default=None), "count": len(lat)},
            "loop_lag": {"p50": _pct(lag, 0.5), "p95": lag95, "max": max(lag, default=None)},
            "rates": dict(self.rates),
            "sources": sources,
            "exchanges": [{"name": x.name, "connected": x.connected, "error": x.last_error,
                           "reconnects": x.reconnects} for x in engine.exchanges],
            "book_match": {"checks": len(day), "within_1c": within, "rate": match,
                           "max_gap": max((c["gap"] for c in day), default=None), "recent": list(day)[-6:]},
            "clock_offset": self.clock_offset,
            "uptime": now - engine.started_at,
            "restarts_24h": sum(1 for t in self.starts if t > now - 86400) - 1,
            "reconnects": {"prices": prices.reconnects if prices else 0, "books": books.reconnects if books else 0},
            "cpu_percent": cpu / max(1.0, now - engine.started_at) * 100,
            "memory_mb": usage.ru_maxrss / 1024,
            "db_mb": db_mb,
            "warnings": list(self.log.items)[-15:],  # list() first: the log handler appends from any thread
        }
