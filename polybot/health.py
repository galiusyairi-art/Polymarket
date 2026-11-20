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
        # Python's garbage collector stops every thread while it runs: each pause measured (an order
        # signalled in one waits for it), shown on the speed card.
        self.gc_pauses = collections.deque(maxlen=5000)  # (ts, generation, seconds)
        self._gc_started = None
        starts = json.loads(engine.store.get_setting("starts", "[]"))
        starts = [t for t in starts if t > time.time() - 7 * 86400] + [time.time()]
        engine.store.set_setting("starts", json.dumps(starts))
        self.starts = starts
        self.start_log = self._note_start(engine.store, time.time())
        self.cpu = {"ts": None, "total": None, "threads": [], "helpers": None}  # the last minute's processor, by thread
        self._cpu_last = None
        self.decided = 0.0  # seconds, in all, the trading loop spent deciding (Engine._evaluate)
        self.books_spent = lambda: 0.0  # (the engine's: the order books' stream, the same)

    START_LOG = 40

    @staticmethod
    def _note_start(store, now):
        """This start, with why the run before it ended, kept with the last ones (the heart of the system shows them).
        Why: a mark the run before left as it went down (a deploy or a stop, the guard with its reason); none and the
        version changed: a new version; none at all: it was ended from outside with no word (out of memory, a crash)."""
        from .engine import VERSION
        try:
            log_ = json.loads(store.get_setting("start_log") or "[]")
        except ValueError:
            log_ = []
        try:
            mark = json.loads(store.get_setting("exit_mark") or "null")
        except ValueError:
            mark = None
        prev = log_[-1] if log_ else None
        if mark and now - (mark.get("ts") or 0) < 900:
            why, detail = mark.get("why") or "stop", mark.get("detail")
        elif prev is None:
            why, detail = "first", None
        elif prev.get("version") != VERSION:
            why, detail = "version", None
        else:
            why, detail = "killed", None
        store.set_setting("exit_mark", None)
        try:
            last = json.loads(store.get_setting("run_last") or "null")
        except ValueError:
            last = None
        store.set_setting("run_last", None)
        log_ = (log_ + [{"ts": now, "why": why, "detail": detail, "version": VERSION,
                         "ran": round(now - prev["ts"]) if prev else None,
                         "last": last if last and prev and last.get("ts", 0) > prev["ts"] else None}])[-Health.START_LOG:]
        store.set_setting("start_log", json.dumps(log_))
        return log_

    def _check_cpu(self, now):
        """The processor's last minute, by thread (and the helper processes together): who is busy."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        kids = resource.getrusage(resource.RUSAGE_CHILDREN)
        cur = {"ts": now, "total": usage.ru_utime + usage.ru_stime, "helpers": kids.ru_utime + kids.ru_stime, "threads": {},
               "decide": self.decided, "books": self.books_spent()}
        for t in threading.enumerate():
            try:
                cur["threads"][t.ident] = (t.name, time.clock_gettime(time.pthread_getcpuclockid(t.ident)))
            except Exception:
                continue
        last, self._cpu_last = self._cpu_last, cur
        if not last or now - last["ts"] <= 0:
            return
        span = now - last["ts"]
        by = collections.Counter()
        for ident, (name, secs) in cur["threads"].items():
            before = last["threads"].get(ident)
            by[name] += max(0.0, secs - (before[1] if before and before[0] == name else 0.0))
        self.cpu = {"ts": now, "total": (cur["total"] - last["total"]) / span,
                    "helpers": (cur["helpers"] - last["helpers"]) / span,
                    "threads": [[n, s / span] for n, s in by.most_common(8) if s / span >= 0.005],
                    # the trading loop's own parts, of the one processor (20.11.2026: it had 45% of it): deciding,
                    # reading and applying the order books' messages; what is left of its share is the receiving of
                    # them (the decryption and the stream's framing, in the libraries)
                    "loop": {"decide": max(0.0, cur["decide"] - last.get("decide", 0.0)) / span,
                             "books": max(0.0, cur["books"] - last.get("books", 0.0)) / span}}

    @staticmethod
    def memory():
        """(this process now, its peak, the machine's whole and what is still free) in MB; None where unknown."""
        now_mb = total = free = None
        try:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS"):
                    now_mb = int(line.split()[1]) / 1024
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    total = int(line.split()[1]) / 1024
                elif line.startswith("MemAvailable"):
                    free = int(line.split()[1]) / 1024
        except (OSError, ValueError, IndexError):
            pass
        return {"now": now_mb, "peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "machine": total, "free": free}

    @staticmethod
    def trim():
        """The memory the C library holds free, given back to the machine (glibc's malloc_trim): what the process had
        before and after, in MB, and the milliseconds it took; None where there is no such call. 20.11.2026: the bot ended
        after eight hours at 1.65 GB with 120 MB left on the machine, while its own objects came to about a tenth of
        that. The rest was room freed after each large piece of work (a learning pass, a scan of every market, a page)
        and kept by the library, a little more each time; one call gave a third of the process back in 6 ms."""
        try:
            import ctypes
            fn = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            return None
        before = Health.memory()["now"]
        t0 = time.perf_counter()
        fn(0)
        return {"ts": time.time(), "before": before, "after": Health.memory()["now"], "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def watch_gc(self):
        import gc

        def callback(phase, info):
            if phase == "start":
                self._gc_started = time.perf_counter()
            elif self._gc_started is not None:
                self.gc_pauses.append((time.time(), info.get("generation"), time.perf_counter() - self._gc_started))
                self._gc_started = None
        if callback not in gc.callbacks:
            gc.callbacks.append(callback)

    def gc_view(self, now=None):
        """The collector's pauses over the last hour: how many, the longest, and the full ones."""
        now = now or time.time()
        last = [(g, s) for ts, g, s in list(self.gc_pauses) if ts > now - 3600]
        return {"n": len(last), "max": max((s for _, s in last), default=None),
                "full": sum(1 for g, _ in last if g == 2), "full_max": max((s for g, s in last if g == 2), default=None)}

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
        try:
            self._check_cpu(now)
            # (kept for the next start: a run that ends with no word leaves how it stood a minute before)
            m = self.memory()
            self.engine.store.set_setting("run_last", json.dumps({
                "ts": now, "memory": round(m["now"] or 0), "free": round(m["free"] or 0), "machine": round(m["machine"] or 0),
                "cpu": round((self.cpu.get("total") or 0) * 100),
                "busiest": (self.cpu.get("threads") or [[None, 0]])[0][0]}))
        except Exception as e:
            log.warning("Processor by thread: %s", e)
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
        # The markets the simulation trades; with none running (all paused, say), every market the
        # system follows, by the freshest book: the stream is alive if any book is fresh.
        book_age = max((s.get("book_age") or 0 for s in snaps), default=None) if snaps else min(
            (t.snapshot.get("book_age") for t in engine.trackers if t.market and t.snapshot.get("book_age") is not None),
            default=None)
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
        D = getattr(engine, "disk", None) or {}
        room = (D["free"] + D["reusable"]) / D["total"] if D.get("total") else None
        check("disk", room is None or not D.get("short"), room,
              "יש מקום בדיסק של השרת" + (f" (שומר {D['days']} ימים של הקלטות)" if D.get("days") else ""))
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
            "db_mb": db_mb, "disk": D,
            "warnings": list(self.log.items)[-15:],  # list() first: the log handler appends from any thread
        }
