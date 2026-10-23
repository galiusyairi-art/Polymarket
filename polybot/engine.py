"""The engine: follow each market's price and Polymarket order books, re-evaluate on every update,
log the paper bets of the main strategy and of every lab variant, and hand the main strategy's
signals to the live trader.

A market is an asset in windows of a given length (BTC 15-minute, SOL 5-minute, ...). The main
strategy runs on the markets in cfg.assets x cfg.window_minutes; the lab runs on every market.

Two modes:
- streaming (default): Chainlink prices and order books arrive over websockets and every
  update is evaluated within milliseconds.
- REST (--rest, and the tests): poll Coinbase and the order books once a second.
"""
import asyncio
import dataclasses
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import feeds, lab, research, stats
from .live import LiveTrader
from .periods import period_starts
from .model import VolatilityEstimator, fair_up_probability, fair_up_probability_twap, taker_fee_per_share
from .paper import find_bet, settle_pnl
from .store import Store, connect

log = logging.getLogger("polybot")


def _version():
    """The running commit, so the dashboard shows which version is live."""
    commit = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("RAILWAY_GIT_COMMIT_SHA")
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2,
                                    cwd=os.path.dirname(__file__)).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = ""
    return commit[:7]


VERSION = _version()


def _best(levels, pick):
    return pick((p for p, _ in levels), default=None)


class Tracker:
    """Follows one market through its consecutive windows."""

    def __init__(self, asset, cfg, store, api=feeds, live=None, source_name="coinbase", dispatch=None,
                 lookup=None, main=True, variants=(), record_seconds=1.0):
        # cfg here is the market's own config: cfg.window_minutes is this market's window length.
        self.asset, self.cfg, self.store, self.api, self.live = asset, cfg, store, api, live
        self.minutes = cfg.window_minutes
        self.main = main
        self.variants = list(variants)
        self.record_seconds = record_seconds
        self.source_name = source_name
        # How slow work (HTTP lookups, live orders) runs: inline by default, on a
        # thread pool when streaming so the event loop never blocks.
        self.dispatch = dispatch or (lambda fn, *args: fn(*args))
        self.lookup = lookup
        self.vol = VolatilityEstimator(cfg.vol_lookback_seconds)  # Chainlink: what the market resolves on
        self.fast = VolatilityEstimator(cfg.vol_lookback_seconds)  # Binance: ~1s fresher, less smoothed
        self.price = self.price_ts = None
        self.window = None
        self.market = None
        self.market_cfg = cfg  # cfg with this market's own fee schedule
        self.next_market = None  # the next window's market, fetched ahead of time
        self.opens = {"twap": None, "point": None}
        self.bets_this_window = 0
        self.lab_done = set()  # variant keys that already bet in this window
        self.last_lookup = self.last_prefetch = 0.0
        self.last_record = 0.0
        self.snapshot = {"asset": asset, "minutes": self.minutes}
        self._set_market_cfg(None)

    @property
    def key(self):
        return f"{self.asset}-{self.minutes}m"

    @property
    def window_seconds(self):
        return self.minutes * 60

    @property
    def open_price(self):
        return self.opens[self.cfg.resolution]

    # REST mode: one poll per second.
    def step(self, now):
        self.on_price(now, self.api.spot_price(self.asset))
        if self.market is None and now - self.last_lookup >= 10:
            self._lookup(self.window)
        up = down = None
        if self.market:
            up = (*self.api.order_book(self.market["up_token"]), now)
            down = (*self.api.order_book(self.market["down_token"]), now)
        self.evaluate(now, up, down, record=True)

    def on_price(self, ts, price):
        """A new price, stamped with its source's time. Rolls the window when it crosses a boundary."""
        if self.price_ts is not None and ts <= self.price_ts:
            return
        self.price, self.price_ts = price, ts
        self.vol.add(ts, price)
        start = feeds.window_start(ts, self.minutes)
        if start != self.window:
            self._new_window(start, ts, price)

    def on_fast(self, ts, price):
        self.fast.add(ts, price)

    def nowcast(self, now):
        """(price, info_ts): the Chainlink price brought up to the present with Binance's move since.

        Chainlink arrives ~1.2s late and the order book doesn't wait for it. Binance's
        level differs from Chainlink's (USDT vs USD, aggregation), so only its return
        since the Chainlink timestamp is used, never its level. Falls back to the raw
        Chainlink price when the fast feed is missing or stale.
        """
        if self.fast.samples:
            fast_ts, fast_now = self.fast.samples[-1]
            fast_then = self.fast.price_at(self.price_ts)
            if fast_then and fast_ts > self.price_ts and now - fast_ts <= self.cfg.max_price_age:
                return self.price * fast_now / fast_then, fast_ts
        return self.price, self.price_ts

    def volatility(self):
        """Per-second volatility, from the fast feed when it has enough history: Chainlink's
        aggregated prices move more smoothly than the market, which understates risk."""
        for series in (self.fast, self.vol):
            if series.history_seconds() >= self.cfg.vol_min_seconds:
                return series.per_sqrt_second()
        return None

    def _fair(self, resolution, price, info_ts, seconds_left, vol, end, now):
        open_price = self.opens[resolution]
        if not (vol and open_price):
            return None
        if resolution == "twap":
            L = self.cfg.twap_seconds
            realized = None
            if seconds_left < L:
                start = end - L
                realized = self.vol.twap(start, now)
                # The last stretch since the latest Chainlink value is better
                # estimated by the nowcast than by holding the stale value.
                gap_from = max(start, self.price_ts)
                if realized is not None and now > gap_from and info_ts > self.price_ts:
                    realized += (price - self.price) * (now - gap_from) / (now - start)
            return fair_up_probability_twap(price, open_price, seconds_left, vol, L, realized)
        return fair_up_probability(price, open_price, seconds_left, vol)

    def evaluate(self, now, up_book, down_book, record=False):
        """Recompute fair probabilities against the current books; bet where the edge is there.

        up_book / down_book: (bids, asks, ts) or None.
        """
        cfg = self.cfg
        if self.price is None or self.window is None:
            return
        end = self.window + self.window_seconds
        seconds_left = end - now
        price, info_ts = self.nowcast(now)
        price_age = max(0.0, now - info_ts)
        # Clamped: Polymarket's clock can run a few tens of milliseconds ahead of ours.
        book_age = max(0.0, now - up_book[2], now - down_book[2]) if up_book and down_book else None
        vol = self.volatility()
        fair = {r: self._fair(r, price, info_ts, seconds_left, vol, end, now) for r in ("twap", "point")}
        fair_up = fair[cfg.resolution]

        up_bids, up_asks = (up_book[0], up_book[1]) if up_book else ([], [])
        down_bids, down_asks = (down_book[0], down_book[1]) if down_book else ([], [])
        tick = dict(
            ts=now, asset=self.asset, minutes=self.minutes, window_start=self.window, price=price,
            open_price=self.open_price, seconds_left=seconds_left, vol=vol, fair_up=fair_up,
            up_bid=_best(up_bids, max), up_ask=_best(up_asks, min),
            down_bid=_best(down_bids, max), down_ask=_best(down_asks, min),
            price_age=price_age, book_age=book_age,
        )
        if record or now - self.last_record >= self.record_seconds:
            self.store.add_tick(**tick)
            self.last_record = now
        fresh = price_age <= cfg.max_price_age and book_age is not None and book_age <= cfg.max_book_age
        # Past the window's end the next window's first Chainlink price hasn't arrived
        # yet (it comes ~1.2s late): show the old window as closed, never negative.
        self.snapshot = dict(tick, seconds_left=max(0.0, seconds_left), rolling=seconds_left <= 0,
                             market_found=bool(self.market), warming_up=vol is None,
                             joined_midway=self.open_price is None, bets_this_window=self.bets_this_window,
                             fresh=fresh)
        if not (self.market and fresh):
            return

        if (self.main and fair_up is not None and self.bets_this_window < cfg.max_bets_per_window
                and cfg.min_seconds_left <= seconds_left <= cfg.max_seconds_left):
            fill = find_bet(fair_up, up_asks, down_asks, self.market_cfg)
            if fill:
                self.bets_this_window += 1
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill,
                                   price_age=price_age, book_age=book_age)
                log.info("PAPER BET %s %s: fair %.1f%%, paid %.1fc/share (+fee), edge %.1f pts, %.1fs left,"
                         " data age price %.2fs / book %.2fs", self.key.upper(), fill.side.upper(),
                         fill.fair_prob * 100, fill.avg_price * 100, fill.edge * 100, seconds_left, price_age,
                         book_age)
                if self.live:
                    self.dispatch(self.live.maybe_trade, now, self.asset, self.minutes, self.window, seconds_left,
                                  fill, self.market, self.market_cfg)

        self._run_lab(now, seconds_left, fair, up_asks, down_asks, price_age, book_age)

    def _run_lab(self, now, seconds_left, fair, up_asks, down_asks, price_age, book_age):
        if not self.variants:
            return
        best_up = _best(up_asks, min)
        best_down = _best(down_asks, min)
        mc = self.market_cfg
        for v in self.variants:
            if v.key in self.lab_done or not (v.min_seconds_left <= seconds_left <= v.max_seconds_left):
                continue
            p = fair[v.resolution]
            if p is None:
                continue
            # Cheap check on the best prices before walking the books.
            edges = [prob - ask - taker_fee_per_share(ask, mc.fee_rate, mc.fee_exponent)
                     for prob, ask in ((p, best_up), (1 - p, best_down)) if ask is not None]
            if not edges or max(edges) < v.min_edge:
                continue
            fill = find_bet(p, up_asks, down_asks, self.lab_cfgs[v.key])
            if fill:
                self.lab_done.add(v.key)
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill, mode="lab",
                                   variant=v.key, price_age=price_age, book_age=book_age)

    def _new_window(self, start, ts, price):
        L = self.cfg.twap_seconds
        if self.window is not None:
            end = self.window + self.window_seconds
            self.store.upsert_window(self.asset, self.minutes, self.window, close_twap=self.vol.twap(end - L, end),
                                     close_point=price if ts - end < 2 else None)
        # Opening value, both readings of the rules: the average of the minute
        # before the window, and the price at its first second. Without history
        # reaching back far enough (just started) the window is recorded, never bet on.
        self.opens = {"twap": self.vol.twap(start - L, start), "point": price if ts - start < 2 else None}
        self.window, self.bets_this_window, self.lab_done = start, 0, set()
        self.market = None
        self._set_market_cfg(None)
        slug = feeds.market_slug(self.cfg, self.asset, start)
        self.store.upsert_window(self.asset, self.minutes, start, slug=slug, found=0, open_price=self.open_price,
                                 open_twap=self.opens["twap"], open_point=self.opens["point"],
                                 open_source=self.source_name)
        if self.next_market and self.next_market["slug"] == slug:
            self._use_market(start, self.next_market)
            self.next_market = None
        elif self.lookup:
            self.lookup(self, start)
        else:
            self._lookup(start)

    def _lookup(self, start):
        self.last_lookup = time.time()
        slug = feeds.market_slug(self.cfg, self.asset, start)
        market = self.api.find_market(slug)
        if not market:
            log.warning("No Polymarket market found for %s (will retry)", slug)
        elif self.window == start:
            self._use_market(start, market)

    def prefetch_due(self, now):
        """Start of the next window if its market should be fetched now, else None."""
        nxt = self.window + self.window_seconds if self.window is not None else None
        if nxt is None or now - self.last_prefetch < 10:
            return None
        if self.next_market and self.next_market["slug"] == feeds.market_slug(self.cfg, self.asset, nxt):
            return None
        return nxt

    def prefetch(self, start):
        self.last_prefetch = time.time()
        market = self.api.find_market(feeds.market_slug(self.cfg, self.asset, start))
        if market:
            self.next_market = market

    def _set_market_cfg(self, market):
        fee_rate = market.get("fee_rate") if market else None
        self.market_cfg = dataclasses.replace(
            self.cfg, fee_rate=self.cfg.fee_rate if fee_rate is None else fee_rate,
            fee_exponent=market.get("fee_exponent", self.cfg.fee_exponent) if market else self.cfg.fee_exponent)
        self.lab_cfgs = {v.key: dataclasses.replace(self.market_cfg, min_edge=v.min_edge) for v in self.variants}

    def _use_market(self, start, market):
        self._set_market_cfg(market)
        self.market = market
        self.store.upsert_window(self.asset, self.minutes, start, found=1)


def settle(store, cfg, now, api=feeds):
    """Fill in winners for finished windows (all markets, last 2 days) and settle their bets."""
    shortest = min((cfg.window_minutes, *cfg.lab_minutes))
    for asset, minutes, start, slug in store.unresolved_windows(now - shortest * 60 - 60, now - 2 * 86400):
        if start + minutes * 60 + 60 > now:
            continue
        winner = api.market_winner(slug)
        if not winner:
            continue
        store.upsert_window(asset, minutes, start, winner=winner)
        for bet_id, side, shares, cost, fee in store.unsettled_bets(asset, minutes, start):
            store.settle_bet(bet_id, winner, settle_pnl(side, shares, cost, fee, winner))


def markets_for(cfg, with_lab=True):
    """(asset, minutes, is_main) for every market to follow."""
    main = {(a, cfg.window_minutes) for a in cfg.assets}
    lab_markets = {(a, m) for a in cfg.lab_assets for m in cfg.lab_minutes} if with_lab else set()
    return sorted((a, m, (a, m) in main) for a, m in main | lab_markets)


class Engine:
    """Runs the trackers in the background and keeps the slow analysis fresh for the dashboard."""

    def __init__(self, cfg, api=feeds, env=None, streaming=True, with_lab=True):
        self.cfg, self.api, self.streaming = cfg, api, streaming
        self.store = Store(cfg.db_path)
        self.live = LiveTrader(cfg, self.store, env=env)
        self.pool = ThreadPoolExecutor(8, thread_name_prefix="work")
        self.trackers = []
        for asset, minutes, is_main in markets_for(cfg, with_lab):
            market_cfg = dataclasses.replace(cfg, window_minutes=minutes)
            self.trackers.append(Tracker(
                asset, market_cfg, self.store, api=api, live=self.live if is_main else None,
                source_name="chainlink" if streaming else "coinbase",
                dispatch=self._dispatch if streaming else None,
                lookup=self._lookup_async if streaming else None,
                main=is_main, variants=lab.variants_for(minutes) if with_lab else (),
                record_seconds=1.0 if is_main else cfg.lab_record_seconds,
            ))
        self.main_trackers = [t for t in self.trackers if t.main]
        self.current_variants = {(t.asset, t.minutes, v.key) for t in self.trackers for v in t.variants}
        self.by_asset = {}
        for t in self.trackers:
            self.by_asset.setdefault(t.asset, []).append(t)
        self.token_owner = {}
        self.prices = self.books = None
        self.errors = {}
        self._last_logged = {}
        self.started_at = time.time()
        self.analysis = None
        self.readiness = None
        self.lab = None
        self.research_running = False
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._stream_thread if self.streaming else self._rest_loop, daemon=True,
                         name="engine").start()
        for seconds, fn in ((30, self._settle), (60, self.refresh_lab), (300, self.refresh_analysis),
                            (300, self._daily)):
            threading.Thread(target=self._every, args=(seconds, fn), daemon=True, name=fn.__name__).start()

    def stop(self):
        self._stop.set()

    def _every(self, seconds, fn):
        while not self._stop.is_set():
            try:
                fn()
            except Exception as e:
                log.warning("%s: %s", fn.__name__, e)
            self._stop.wait(seconds)

    def _settle(self):
        settle(self.store, self.cfg, time.time(), api=self.api)

    def _guard(self, key, fn, *args):
        try:
            fn(*args)
            self.errors.pop(key, None)
        except Exception as e:
            self.errors[key] = f"{type(e).__name__}: {e}"
            # Network hiccups are normal; log each market's error at most once a minute.
            if time.time() - self._last_logged.get(key, 0) > 60:
                log.warning("%s: %s", key.upper(), e)
                self._last_logged[key] = time.time()

    def _dispatch(self, fn, *args):
        self.pool.submit(self._guard, "live", fn, *args)

    def _lookup_async(self, tracker, start):
        tracker.last_lookup = time.time()
        self.pool.submit(self._guard, tracker.key, tracker._lookup, start)

    # REST mode
    def _rest_loop(self):
        with ThreadPoolExecutor(len(self.trackers)) as pool:
            while not self._stop.is_set():
                now = time.time()
                futures = [pool.submit(self._guard, t.key, t.step, now) for t in self.trackers]
                for f in futures:
                    f.result()
                self._stop.wait(max(0.0, self.cfg.poll_seconds - (time.time() - now)))

    # Streaming mode
    def _stream_thread(self):
        asyncio.run(self._stream_main())

    async def _stream_main(self):
        self.prices = feeds.ChainlinkStream({a: f"{a}/usd" for a in self.by_asset}, self._on_price,
                                            {a: f"{a}usdt" for a in self.by_asset}, self._on_fast)
        self.books = feeds.BookStream(self._on_book)
        self._dirty, self._wake = set(), asyncio.Event()
        await asyncio.gather(self.prices.run(), self.books.run(), self._heartbeat(), self._evaluator())

    # Updates are applied the moment they arrive (cheap); evaluating a market against
    # them is not cheap and is batched instead. A burst of a hundred book changes then
    # costs one evaluation on the newest state, not a hundred on ever staler ones, so
    # the event loop never builds a backlog.
    def _mark(self, tracker):
        self._dirty.add(tracker)
        self._wake.set()

    async def _evaluator(self):
        while not self._stop.is_set():
            await self._wake.wait()
            self._wake.clear()
            await asyncio.sleep(0.02)  # let the rest of the burst arrive
            dirty, self._dirty = self._dirty, set()
            now = time.time()
            for tracker in dirty:
                self._evaluate(tracker, now)

    def _on_price(self, asset, ts, price):
        for tracker in self.by_asset.get(asset, ()):
            self._guard(tracker.key, tracker.on_price, ts, price)
            self._mark(tracker)

    def _on_fast(self, asset, ts, price):
        for tracker in self.by_asset.get(asset, ()):
            tracker.on_fast(ts, price)
            self._mark(tracker)

    def _on_book(self, token):
        tracker = self.token_owner.get(token)
        if tracker:
            self._mark(tracker)

    def _evaluate(self, tracker, now=None):
        market = tracker.market
        up = self.books.get(market["up_token"]) if market else None
        down = self.books.get(market["down_token"]) if market else None
        self._guard(tracker.key, tracker.evaluate, now or time.time(), up, down)

    async def _heartbeat(self):
        """Four times a second: keep time-based numbers moving, fetch upcoming markets, track subscriptions."""
        while not self._stop.is_set():
            now = time.time()
            tokens, owners = set(), {}
            for t in self.trackers:
                self._evaluate(t, now)
                if t.window is not None and t.market is None and now - t.last_lookup >= 10:
                    self._lookup_async(t, t.window)
                nxt = t.prefetch_due(now)
                if nxt is not None:
                    t.last_prefetch = now
                    self.pool.submit(self._guard, t.key, t.prefetch, nxt)
                # Subscribe to the current and the next window, so the switch at the
                # window boundary needs no new connection.
                for m in (t.market, t.next_market):
                    if m:
                        for token in (m["up_token"], m["down_token"]):
                            tokens.add(token)
                            owners[token] = t
            self.token_owner = owners
            self.books.set_tokens(tokens)
            await asyncio.sleep(0.25)

    # Analysis
    def refresh_analysis(self):
        """The slow numbers, every few minutes: prediction accuracy, what-if, calibration."""
        db = connect(self.cfg.db_path)
        try:
            self.analysis = dict(stats.analysis(db, self.cfg), calibration=stats.calibration(db))
        finally:
            db.close()
        self.store.prune_ticks(time.time() - self.cfg.keep_ticks_days * 86400)

    def refresh_lab(self):
        """Every minute: the lab ranking and the readiness checks (both cheap)."""
        db = connect(self.cfg.db_path)
        try:
            self.lab = lab.leaderboard(db, self.cfg, self.current_variants)
            first = db.execute("SELECT MIN(ts) FROM ticks").fetchone()[0]
            last = db.execute("SELECT MAX(ts) FROM ticks").fetchone()[0]
            hours = (last - first) / 3600 if first else 0.0
            readiness = stats.readiness(self.cfg, hours, stats.feed_agreement(db, self.cfg.resolution),
                                        stats.bet_summary(db, "paper"))
            self.readiness = dict(readiness, computed_at=time.time())
            self.live.set_readiness(self.readiness["ready"])
        finally:
            db.close()

    def _daily(self):
        """Once a day, after research_hour_utc, ask Claude for the research summary."""
        if not research.configured():
            return
        now = time.gmtime()
        today = time.strftime("%Y-%m-%d", now)
        if now.tm_hour >= self.cfg.research_hour_utc and self.store.get_setting("research_day") != today:
            self.store.set_setting("research_day", today)
            self.run_research()

    def run_research(self):
        if self.research_running:
            raise ValueError("a summary is already being written")
        if not research.configured():
            raise ValueError("ANTHROPIC_API_KEY is not set on the server")
        self.research_running = True
        try:
            if self.lab is None:
                self.refresh_lab()
            db = connect(self.cfg.db_path)
            try:
                feed = {m: stats.feed_agreement(db, m) for m in ("twap", "point")}
            finally:
                db.close()
            hours = self.analysis["hours"] if self.analysis else (time.time() - self.started_at) / 3600
            research.run(self.store, self.lab, self.readiness, feed, hours)
        finally:
            self.research_running = False

    def start_research(self):
        """Kick off a summary in the background (from the dashboard button)."""
        if self.research_running:
            raise ValueError("a summary is already being written")
        if not research.configured():
            raise ValueError("ANTHROPIC_API_KEY is not set on the server")
        self.pool.submit(self._guard, "research", self.run_research)

    def _dashboard(self, db, starts):
        """Per period (day / week / month): trades and P&L, simulated and real, and the strategy mix."""
        out = {}
        for period, since in starts.items():
            trades = {}
            for mode in ("paper", "live"):
                s = stats.bet_summary(db, mode, since=since)
                s.pop("rows")
                trades[mode] = s
            out[period] = {"since": since, "trades": trades,
                           "strategies": stats.strategy_mix(db, self.cfg, since, self.current_variants)}
        return {
            "periods": out,
            "strategies_total": sum(len(t.variants) for t in self.trackers) + 1,
            "thresholds": {"min_bets": self.cfg.dash_min_bets, "hot": self.cfg.dash_hot_roi,
                           "losing": self.cfg.dash_losing_roi},
            "timezone": self.cfg.timezone,
        }

    def state(self):
        """Everything the dashboard shows, as one JSON-ready dict."""
        now = time.time()
        db = connect(self.cfg.db_path)
        try:
            paper = stats.bet_summary(db, "paper")
            live_total = stats.bet_summary(db, "live")
            starts = period_starts(now, self.cfg.timezone)
            live_today = stats.bet_summary(db, "live", since=starts["day"])
            for s in (paper, live_total, live_today):
                s.pop("rows")
            snaps = [t.snapshot for t in self.trackers]
            price_ages = [s.get("price_age") for s in snaps]
            book_ages = [s.get("book_age") for s in snaps]
            sources = {
                "price": {"ok": all(a is not None and a < 5 for a in price_ages),
                          "name": "Chainlink" if self.streaming else "Coinbase"},
                "polymarket": {"ok": all(a is not None and a < 15 for a in book_ages),
                               "lag": self.books.lag if self.books else None},
            }
            lab_state = None
            if self.lab is not None:
                lab_state = dict(self.lab, board=self.lab["board"][:40])
            return {
                "now": now,
                "uptime": now - self.started_at,
                "streaming": self.streaming,
                "version": VERSION,
                "config": {"assets": list(self.cfg.assets), "window_minutes": self.cfg.window_minutes,
                           "min_edge": self.cfg.min_edge, "stake": self.cfg.stake_usd,
                           "min_seconds_left": self.cfg.min_seconds_left,
                           "max_seconds_left": self.cfg.max_seconds_left, "resolution": self.cfg.resolution},
                "sources": sources,
                "errors": self.errors,
                "assets": [t.snapshot for t in self.main_trackers],
                "paper": {"summary": paper,
                          "feed": {m: stats.feed_agreement(db, m) for m in ("twap", "point")},
                          "bets": stats.recent_bets(db, "paper", 20)},
                "analysis": self.analysis,
                "readiness": self.readiness,
                "live": {"status": self.live.status(), "total": live_total, "today": live_today,
                         "bets": stats.recent_bets(db, "live")},
                "dashboard": self._dashboard(db, starts),
                "lab": {
                    "results": lab_state,
                    "markets_followed": len(self.trackers),
                    "variants": sum(len(t.variants) for t in self.trackers),
                    "min_bets": self.cfg.lab_min_bets,
                    "research": research.latest(db),
                    "research_configured": research.configured(),
                    "research_running": self.research_running,
                    "research_hour_utc": self.cfg.research_hour_utc,
                },
            }
        finally:
            db.close()
