"""The dashboard's data, ready before it is asked (the owner, 06.11.2026: "every time I open Polybot: 'the server isn't
answering', and it takes forever to load"). Engine.state() reads every trade there is (its summaries, the strategies'
results, the lab, the dashboard); on the server's one processor, beside the trading, it came to more than the page's
10 seconds. Here a thread works it out in the background while someone is looking (and right after one of the
owner's clicks), keeps it as JSON and gzip, and every ask is answered at once with the latest."""
import gzip
import json
import logging
import threading
import time

log = logging.getLogger("polybot")


class StateCache:
    MIN_GAP = 2.0  # seconds between two passes at least
    LOAD = 1.5  # ...and this many times the last pass's time (the trading keeps most of the processor)
    IDLE = 120  # seconds without an ask before it stops working it out
    FIRST_WAIT = 25  # seconds the very first ask (after a start) may wait for the first pass
    FRESH_WAIT = 10  # seconds an ask after a click waits for a pass that saw it

    def __init__(self, engine, clock=time.time):
        self.engine, self.clock = engine, clock
        self.snap = None  # {"ts", "raw", "gz", "took"}
        self.asked = 0.0
        self.touched = 0.0  # the owner's last click (POST): the next ask wants a pass started after it
        self._wake = threading.Event()
        self._done = threading.Condition()
        self._thread = None
        self.error = None

    def build(self):
        t0 = self.clock()
        data = self.engine.state()
        data["state_took"] = round(self.clock() - t0, 2)
        raw = json.dumps(data).encode()
        snap = {"ts": t0, "raw": raw, "gz": gzip.compress(raw, compresslevel=5), "took": self.clock() - t0}
        with self._done:
            self.snap = snap
            self.error = None
            self._done.notify_all()
        return snap

    def _run(self):
        while True:
            if self.clock() - self.asked > self.IDLE:
                self._wake.wait(5)
                self._wake.clear()
                continue
            try:
                snap = self.build()
                gap = max(self.MIN_GAP, snap["took"] * self.LOAD)
            except Exception as e:
                log.exception("Dashboard state (in the background) failed")
                with self._done:
                    self.error = f"{type(e).__name__}: {e}"
                    self._done.notify_all()
                gap = 5.0
            self._wake.wait(gap)
            self._wake.clear()

    def _start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True, name="state-cache")
            self._thread.start()

    def touch(self):
        """One of the owner's clicks: the next ask gets a pass that saw it."""
        self.touched = self.clock()
        self._wake.set()

    def get(self):
        """The latest snapshot, or None while the very first one is still being worked out (after a start)."""
        self.asked = self.clock()
        self._start()
        self._wake.set() if self.snap is None else None
        with self._done:
            if self.snap is None:
                self._done.wait_for(lambda: self.snap is not None or self.error is not None, self.FIRST_WAIT)
            elif self.snap["ts"] < self.touched:
                self._done.wait_for(lambda: self.snap["ts"] >= self.touched, self.FRESH_WAIT)
            return self.snap
