"""The engine: follow each asset's price and Polymarket order books, re-evaluate on every update,
log the paper bets the strategy would make, and hand the same signals to the live trader.

Two modes:
- streaming (default): Chainlink prices and order books arrive over websockets and every
  update is evaluated within milliseconds.
- REST (--rest, and the tests): poll Coinbase and the order books once a second.
"""
import asyncio
import dataclasses
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import feeds, stats
from .live import LiveTrader, start_of_today
from .model import VolatilityEstimator, fair_up_probability, fair_up_probability_twap
from .paper import find_bet, settle_pnl
from .store import Store, connect

log = logging.getLogger("polybot")


def _best(levels, pick):
    return pick((p for p, _ in levels), default=None)


class Tracker:
    """Follows one asset through its consecutive windows."""

    def __init__(self, asset, cfg, store, api=feeds, live=None, source_name="coinbase", dispatch=None,
                 lookup=None):
        self.asset, self.cfg, self.store, self.api, self.live = asset, cfg, store, api, live
        self.source_name = source_name
        # How slow work (HTTP lookups, live orders) runs: inline by default, on a
        # thread pool when streaming so the event loop never blocks.
        self.dispatch = dispatch or (lambda fn, *args: fn(*args))
        self.lookup = lookup
        self.vol = VolatilityEstimator(cfg.vol_lookback_seconds)
        self.price = self.price_ts = None
        self.window = None
        self.market = None
        self.market_cfg = cfg  # cfg with this market's own fee schedule
        self.next_market = None  # the next window's market, fetched ahead of time
        self.open_price = None
        self.bets_this_window = 0
        self.last_lookup = self.last_prefetch = 0.0
        self.last_record = 0.0
        self.snapshot = {"asset": asset}

    @property
    def window_seconds(self):
        return self.cfg.window_minutes * 60

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
        start = feeds.window_start(ts, self.cfg.window_minutes)
        if start != self.window:
            self._new_window(start, ts, price)

    def evaluate(self, now, up_book, down_book, record=False):
        """Recompute the fair probability against the current books; bet if the edge is there.

        up_book / down_book: (bids, asks, ts) or None.
        """
        cfg = self.cfg
        if self.price is None or self.window is None:
            return
        end = self.window + self.window_seconds
        seconds_left = end - now
        price_age = max(0.0, now - self.price_ts)
        book_age = max(now - up_book[2], now - down_book[2]) if up_book and down_book else None
        vol = self.vol.per_sqrt_second() if self.vol.history_seconds() >= cfg.vol_min_seconds else None

        fair_up = None
        if vol and self.open_price:
            if cfg.resolution == "twap":
                realized = self.vol.twap(end - cfg.twap_seconds, now) if seconds_left < cfg.twap_seconds else None
                fair_up = fair_up_probability_twap(self.price, self.open_price, seconds_left, vol,
                                                   cfg.twap_seconds, realized)
            else:
                fair_up = fair_up_probability(self.price, self.open_price, seconds_left, vol)

        up_bids, up_asks = (up_book[0], up_book[1]) if up_book else ([], [])
        down_bids, down_asks = (down_book[0], down_book[1]) if down_book else ([], [])
        tick = dict(
            ts=now, asset=self.asset, window_start=self.window, price=self.price, open_price=self.open_price,
            seconds_left=seconds_left, vol=vol, fair_up=fair_up,
            up_bid=_best(up_bids, max), up_ask=_best(up_asks, min),
            down_bid=_best(down_bids, max), down_ask=_best(down_asks, min),
            price_age=price_age, book_age=book_age,
        )
        if record or now - self.last_record >= 1.0:
            self.store.add_tick(**tick)
            self.last_record = now
        fresh = price_age <= cfg.max_price_age and book_age is not None and book_age <= cfg.max_book_age
        self.snapshot = dict(tick, market_found=bool(self.market), warming_up=vol is None,
                             joined_midway=self.open_price is None, bets_this_window=self.bets_this_window,
                             fresh=fresh)

        if not (fair_up is not None and self.market and fresh
                and self.bets_this_window < cfg.max_bets_per_window
                and cfg.min_seconds_left <= seconds_left <= cfg.max_seconds_left):
            return
        fill = find_bet(fair_up, up_asks, down_asks, self.market_cfg)
        if not fill:
            return
        self.bets_this_window += 1
        self.store.add_bet(now, self.asset, self.window, seconds_left, fill, price_age=price_age, book_age=book_age)
        log.info("PAPER BET %s %s: fair %.1f%%, paid %.1fc/share (+fee), edge %.1f pts, %.1fs left,"
                 " data age price %.2fs / book %.2fs", self.asset.upper(), fill.side.upper(), fill.fair_prob * 100,
                 fill.avg_price * 100, fill.edge * 100, seconds_left, price_age, book_age)
        if self.live:
            self.dispatch(self.live.maybe_trade, now, self.asset, self.window, seconds_left, fill, self.market,
                          self.market_cfg)

    def _new_window(self, start, ts, price):
        L = self.cfg.twap_seconds
        if self.window is not None:
            end = self.window + self.window_seconds
            self.store.upsert_window(self.asset, self.window, close_twap=self.vol.twap(end - L, end),
                                     close_point=price if ts - end < 2 else None)
        # Opening value, both readings of the rules: the average of the minute
        # before the window, and the price at its first second. Without history
        # reaching back far enough (just started) the window is recorded, never bet on.
        open_twap = self.vol.twap(start - L, start)
        open_point = price if ts - start < 2 else None
        self.open_price = open_twap if self.cfg.resolution == "twap" else open_point
        self.window, self.bets_this_window = start, 0
        self.market, self.market_cfg = None, self.cfg
        slug = feeds.market_slug(self.cfg, self.asset, start)
        self.store.upsert_window(self.asset, start, slug=slug, found=0, open_price=self.open_price,
                                 open_twap=open_twap, open_point=open_point, open_source=self.source_name)
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

    def _use_market(self, start, market):
        fee_rate = market.get("fee_rate")
        self.market_cfg = dataclasses.replace(
            self.cfg, fee_rate=self.cfg.fee_rate if fee_rate is None else fee_rate,
            fee_exponent=market.get("fee_exponent", self.cfg.fee_exponent))
        self.market = market
        self.store.upsert_window(self.asset, start, found=1)
        if market.get("price_to_beat") and self.open_price is not None:
            # Polymarket's own opening price is what the market resolves against.
            self.open_price = market["price_to_beat"]
            self.store.upsert_window(self.asset, start, open_price=self.open_price, open_source="polymarket")


def settle(store, cfg, now, api=feeds):
    """Fill in winners for finished windows and settle their bets, paper and live."""
    for asset, start, slug in store.unresolved_windows(now - cfg.window_minutes * 60 - 60):
        winner = api.market_winner(slug)
        if not winner:
            continue
        store.upsert_window(asset, start, winner=winner)
        for bet_id, side, shares, cost, fee in store.unsettled_bets(asset, start):
            pnl = settle_pnl(side, shares, cost, fee, winner)
            store.settle_bet(bet_id, winner, pnl)
            log.info("SETTLED %s %s bet: winner %s, P&L $%.2f", asset.upper(), side.upper(), winner.upper(), pnl)


class Engine:
    """Runs the trackers in the background and keeps the slow analysis fresh for the dashboard."""

    def __init__(self, cfg, api=feeds, env=None, streaming=True):
        self.cfg, self.api, self.streaming = cfg, api, streaming
        self.store = Store(cfg.db_path)
        self.live = LiveTrader(cfg, self.store, env=env)
        self.pool = ThreadPoolExecutor(8, thread_name_prefix="work")
        self.trackers = [
            Tracker(a, cfg, self.store, api=api, live=self.live,
                    source_name="chainlink" if streaming else "coinbase",
                    dispatch=self._dispatch if streaming else None,
                    lookup=self._lookup_async if streaming else None)
            for a in cfg.assets
        ]
        self.by_asset = {t.asset: t for t in self.trackers}
        self.token_owner = {}
        self.prices = self.books = None
        self.errors = {}
        self._last_logged = {}
        self.started_at = time.time()
        self.analysis = None
        self.readiness = None
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._stream_thread if self.streaming else self._rest_loop, daemon=True,
                         name="engine").start()
        threading.Thread(target=self._every, args=(30, self._settle), daemon=True, name="settle").start()
        threading.Thread(target=self._every, args=(60, self.refresh_analysis), daemon=True,
                         name="analysis").start()

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

    def _guard(self, asset, fn, *args):
        try:
            fn(*args)
            self.errors.pop(asset, None)
        except Exception as e:
            self.errors[asset] = f"{type(e).__name__}: {e}"
            # Network hiccups are normal; log each asset's error at most once a minute.
            if time.time() - self._last_logged.get(asset, 0) > 60:
                log.warning("%s: %s", asset.upper(), e)
                self._last_logged[asset] = time.time()

    def _dispatch(self, fn, *args):
        self.pool.submit(self._guard, "live", fn, *args)

    def _lookup_async(self, tracker, start):
        tracker.last_lookup = time.time()
        self.pool.submit(self._guard, tracker.asset, tracker._lookup, start)

    # REST mode
    def _rest_loop(self):
        with ThreadPoolExecutor(len(self.trackers)) as pool:
            while not self._stop.is_set():
                now = time.time()
                futures = [pool.submit(self._guard, t.asset, t.step, now) for t in self.trackers]
                for f in futures:
                    f.result()
                self._stop.wait(max(0.0, self.cfg.poll_seconds - (time.time() - now)))

    # Streaming mode
    def _stream_thread(self):
        asyncio.run(self._stream_main())

    async def _stream_main(self):
        self.prices = feeds.ChainlinkStream({a: f"{a}/usd" for a in self.cfg.assets}, self._on_price)
        self.books = feeds.BookStream(self._on_book)
        await asyncio.gather(self.prices.run(), self.books.run(), self._heartbeat())

    def _on_price(self, asset, ts, price):
        tracker = self.by_asset.get(asset)
        if tracker:
            self._guard(asset, tracker.on_price, ts, price)
            self._evaluate(tracker)

    def _on_book(self, token):
        tracker = self.token_owner.get(token)
        if tracker:
            self._evaluate(tracker)

    def _evaluate(self, tracker, now=None):
        market = tracker.market
        up = self.books.get(market["up_token"]) if market else None
        down = self.books.get(market["down_token"]) if market else None
        self._guard(tracker.asset, tracker.evaluate, now or time.time(), up, down)

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
                    self.pool.submit(self._guard, t.asset, t.prefetch, nxt)
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

    def refresh_analysis(self):
        db = connect(self.cfg.db_path)
        try:
            self.analysis = stats.analysis(db, self.cfg)
            paper = stats.bet_summary(db, "paper")
            recent = stats.recent_roi(paper["returns"], self.cfg.ready_recent_bets)
            self.readiness = stats.readiness(self.cfg, self.analysis["hours"],
                                             stats.feed_agreement(db, self.cfg.resolution), paper, recent)
            self.live.set_readiness(self.readiness["ready"])
        finally:
            db.close()

    def state(self):
        """Everything the dashboard shows, as one JSON-ready dict."""
        now = time.time()
        db = connect(self.cfg.db_path)
        try:
            paper = stats.bet_summary(db, "paper")
            live_total = stats.bet_summary(db, "live")
            live_today = stats.bet_summary(db, "live", since=start_of_today())
            for s in (paper, live_total, live_today):
                s.pop("returns")
            snaps = [t.snapshot for t in self.trackers]
            price_ages = [s.get("price_age") for s in snaps]
            book_ages = [s.get("book_age") for s in snaps]
            sources = {
                "price": {"ok": all(a is not None and a < 5 for a in price_ages),
                          "name": "Chainlink" if self.streaming else "Coinbase"},
                "polymarket": {"ok": all(a is not None and a < 15 for a in book_ages)},
            }
            return {
                "now": now,
                "uptime": now - self.started_at,
                "streaming": self.streaming,
                "config": {"assets": list(self.cfg.assets), "window_minutes": self.cfg.window_minutes,
                           "min_edge": self.cfg.min_edge, "stake": self.cfg.stake_usd,
                           "min_seconds_left": self.cfg.min_seconds_left,
                           "max_seconds_left": self.cfg.max_seconds_left, "resolution": self.cfg.resolution},
                "sources": sources,
                "errors": self.errors,
                "assets": snaps,
                "paper": {"summary": paper,
                          "feed": {m: stats.feed_agreement(db, m) for m in ("twap", "point")},
                          "bets": stats.recent_bets(db, "paper", 20)},
                "analysis": self.analysis,
                "readiness": self.readiness,
                "live": {"status": self.live.status(), "total": live_total, "today": live_today,
                         "bets": stats.recent_bets(db, "live")},
            }
        finally:
            db.close()
