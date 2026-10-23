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
import collections
import dataclasses
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import json

from . import calibration, feeds, lab, research, riskreward, stats
from .live import LiveTrader
from .health import Health
from .notify import Notifier
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
                 lookup=None, main=True, variants=(), record_seconds=1.0, main_rule=None):
        # cfg here is the market's own config: cfg.window_minutes is this market's window length.
        self.asset, self.cfg, self.store, self.api, self.live = asset, cfg, store, api, live
        self.minutes = cfg.window_minutes
        # The main strategy's rule on this market (a lab.Variant), or None if the market is
        # lab-only. By default the configured rule; the owner can promote a proven lab rule.
        if main and main_rule is None:
            main_rule = lab.Variant(cfg.min_edge, cfg.min_seconds_left, cfg.max_seconds_left, cfg.resolution)
        self.main_rule = main_rule
        self.calib_k = 1.0  # set by the engine from calibration.calibrate
        self.min_ratio = 0.0  # risk/reward filter on the main strategy, set by the engine
        self.variants = list(variants)
        self.record_seconds = record_seconds
        self.source_name = source_name
        # How slow work (HTTP lookups, live orders) runs: inline by default, on a
        # thread pool when streaming so the event loop never blocks.
        self.dispatch = dispatch or (lambda fn, *args: fn(*args))
        self.lookup = lookup
        self.vol = VolatilityEstimator(cfg.vol_lookback_seconds)  # Chainlink: what the market resolves on
        self.fast_recent = {}  # source -> the last ~15s of its prices, for the nowcast
        self.fast_vol = {}  # source -> one-second samples, for volatility
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
    def main(self):
        return self.main_rule is not None

    def set_main_rule(self, rule, record_seconds):
        self.main_rule, self.record_seconds = rule, record_seconds
        self._set_market_cfg(self.market)

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

    # Fast prices: straight from Binance and Coinbase (tens of ms), and Binance through
    # Polymarket's relay (about a second, the fallback). Preference order for volatility.
    FAST_SOURCES = ("binance", "coinbase", "relay")

    def on_fast(self, ts, price, source="relay"):
        recent = self.fast_recent.setdefault(source, collections.deque())
        if recent and ts <= recent[-1][0]:
            return
        recent.append((ts, price))
        while recent[0][0] < ts - 15:
            recent.popleft()
        # Volatility from one-second samples: finer sampling mostly measures bid/ask noise.
        series = self.fast_vol.setdefault(source, VolatilityEstimator(self.cfg.vol_lookback_seconds))
        if not series.samples or ts - series.samples[-1][0] >= 1.0:
            series.add(ts, price)

    def nowcast(self, now):
        """(price, info_ts): the Chainlink price brought up to the present.

        Chainlink arrives ~1.2s late and the order book doesn't wait for it. Each fresh fast
        source says how much the market moved since the Chainlink timestamp; their average move
        is applied. Only moves are used, never levels (Binance quotes in USDT, and every venue's
        level differs a little from Chainlink's aggregate). Falls back to the raw Chainlink
        price when no fast source is fresh.
        """
        moves, newest = [], None
        for recent in self.fast_recent.values():
            if not recent:
                continue
            ts_now, p_now = recent[-1]
            if ts_now <= self.price_ts or now - ts_now > self.cfg.max_price_age:
                continue
            p_then = next((p for ts, p in reversed(recent) if ts <= self.price_ts), None)
            if p_then:
                moves.append(p_now / p_then)
                newest = max(newest or 0.0, ts_now)
        if moves:
            return self.price * sum(moves) / len(moves), newest
        return self.price, self.price_ts

    def volatility(self):
        """Per-second volatility from the best fast source with enough history: Chainlink's
        aggregated prices move more smoothly than the market, which understates risk."""
        for source in self.FAST_SOURCES:
            series = self.fast_vol.get(source)
            if series and series.history_seconds() >= self.cfg.vol_min_seconds:
                return series.per_sqrt_second()
        if self.vol.history_seconds() >= self.cfg.vol_min_seconds:
            return self.vol.per_sqrt_second()
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
        raw = {r: self._fair(r, price, info_ts, seconds_left, vol, end, now) for r in ("twap", "point")}
        # Decisions use calibrated probabilities; the raw ones are stored with each bet, so
        # calibration is always fitted on what the model said, never on its own corrections.
        fair = {r: calibration.apply(p, self.calib_k) for r, p in raw.items()}
        rule = self.main_rule
        resolution = rule.resolution if rule else cfg.resolution
        fair_up = fair[resolution]

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
                             fresh=fresh, fair_up_raw=raw[resolution],
                             rule={"min_edge": rule.min_edge, "min_seconds_left": rule.min_seconds_left,
                                   "max_seconds_left": rule.max_seconds_left, "key": rule.key,
                                   "text": lab.describe(rule.key)} if rule else None)
        if not (self.market and fresh):
            return

        if (rule and fair_up is not None and self.bets_this_window < cfg.max_bets_per_window
                and rule.min_seconds_left <= seconds_left <= rule.max_seconds_left):
            fill = find_bet(fair_up, up_asks, down_asks, self.main_cfg)
            if fill:
                self.bets_this_window += 1
                side_raw = raw[resolution] if fill.side == "up" else 1 - raw[resolution]
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill,
                                   variant=rule.key, price_age=price_age, book_age=book_age, fair_raw=side_raw)
                log.info("PAPER BET %s %s: fair %.1f%%, paid %.1fc/share (+fee), edge %.1f pts, %.1fs left,"
                         " data age price %.2fs / book %.2fs", self.key.upper(), fill.side.upper(),
                         fill.fair_prob * 100, fill.avg_price * 100, fill.edge * 100, seconds_left, price_age,
                         book_age)
                if self.live:
                    self.dispatch(self.live.maybe_trade, now, self.asset, self.minutes, self.window, seconds_left,
                                  fill, self.market, self.main_cfg)

        self._run_lab(now, seconds_left, fair, raw, up_asks, down_asks, price_age, book_age)

    def _run_lab(self, now, seconds_left, fair, raw, up_asks, down_asks, price_age, book_age):
        if not self.variants:
            return
        best_up = _best(up_asks, min)
        best_down = _best(down_asks, min)
        mc = self.market_cfg
        cfgs = self.lab_cfgs  # read once: the engine may swap in a new set of variants
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
            v_cfg = cfgs.get(v.key)
            if v_cfg is None:
                continue
            fill = find_bet(p, up_asks, down_asks, v_cfg)
            if fill:
                self.lab_done.add(v.key)
                side_raw = raw[v.resolution] if fill.side == "up" else 1 - raw[v.resolution]
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill, mode="lab",
                                   variant=v.key, price_age=price_age, book_age=book_age, fair_raw=side_raw)

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
        key = (self.asset, self.minutes, start)
        # After a restart mid-window, what the previous run recorded still holds: the opening
        # values it measured, and the bets it already placed (so none is placed twice).
        stored = self.store._x("SELECT open_twap, open_point FROM windows"
                               " WHERE asset = ? AND minutes = ? AND window_start = ?", key)
        if stored:
            self.opens = {"twap": self.opens["twap"] if self.opens["twap"] is not None else stored[0][0],
                          "point": self.opens["point"] if self.opens["point"] is not None else stored[0][1]}
        prior = self.store._x("SELECT mode, variant FROM bets WHERE asset = ? AND minutes = ? AND window_start = ?", key)
        self.window = start
        self.bets_this_window = sum(1 for mode, _ in prior if mode == "paper")
        self.lab_done = {variant for mode, variant in prior if mode == "lab"}
        self.market = None
        self._set_market_cfg(None)
        slug = feeds.market_slug(self.cfg, self.asset, start)
        fields = {"slug": slug, "open_source": self.source_name}
        if not stored:
            fields["found"] = 0
        # Never overwrite a recorded value with an unknown one.
        fields.update({k: v for k, v in (("open_price", self.open_price), ("open_twap", self.opens["twap"]),
                                         ("open_point", self.opens["point"])) if v is not None})
        self.store.upsert_window(self.asset, self.minutes, start, **fields)
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
        self.main_cfg = (dataclasses.replace(self.market_cfg, min_edge=self.main_rule.min_edge,
                                             min_ratio=self.min_ratio)
                         if self.main_rule else self.market_cfg)

    def set_min_ratio(self, ratio):
        if ratio != self.min_ratio:
            self.min_ratio = ratio
            self._set_market_cfg(self.market)

    def set_variants(self, variants):
        """Swap the lab variants (the search near winners adds and retires some)."""
        cfgs = {v.key: dataclasses.replace(self.market_cfg, min_edge=v.min_edge) for v in variants}
        self.lab_cfgs = {**self.lab_cfgs, **cfgs}  # a superset first, so a running pass never misses one
        self.variants = list(variants)
        self.lab_cfgs = cfgs

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


def markets_for(cfg, with_lab=True, main_markets=None):
    """(asset, minutes, is_main) for every market to follow."""
    main = set(main_markets) if main_markets is not None else {(a, cfg.window_minutes) for a in cfg.assets}
    lab_markets = {(a, m) for a in cfg.lab_assets for m in cfg.lab_minutes} if with_lab else set()
    return sorted((a, m, (a, m) in main) for a, m in main | lab_markets)


def _key(asset, minutes, variant=None):
    return f"{asset}|{minutes}" + (f"|{variant}" if variant else "")


class Engine:
    """Runs the trackers in the background and keeps the slow analysis fresh for the dashboard."""

    def __init__(self, cfg, api=feeds, env=None, streaming=True, with_lab=True):
        self.cfg, self.api, self.streaming = cfg, api, streaming
        self.store = Store(cfg.db_path)
        self.live = LiveTrader(cfg, self.store, env=env)
        self.pool = ThreadPoolExecutor(8, thread_name_prefix="work")
        self.with_lab = with_lab
        self.trackers = []
        main = self.main_strategies()
        for asset, minutes, _ in markets_for(cfg, with_lab, [(m["asset"], m["minutes"]) for m in main]):
            market_cfg = dataclasses.replace(cfg, window_minutes=minutes)
            self.trackers.append(Tracker(
                asset, market_cfg, self.store, api=api, live=self.live,
                source_name="chainlink" if streaming else "coinbase",
                dispatch=self._dispatch if streaming else None,
                lookup=self._lookup_async if streaming else None,
                main=False, variants=self._lab_variants(asset, minutes) if with_lab else (),
                record_seconds=cfg.lab_record_seconds,
            ))
        self._apply_main(main)
        self._refresh_variant_set()
        self.calibration = {"k": 1.0, "applied": False, "samples": 0, "fitted": None,
                            "min_samples": cfg.calib_min_samples}
        self._last_explore = 0.0
        self.rr_table = []
        self.rr_auto = (0.0, "waiting")
        self.rr_effective = 0.0
        self._apply_risk_reward()
        self.by_asset = {}
        for t in self.trackers:
            self.by_asset.setdefault(t.asset, []).append(t)
        self.token_owner = {}
        self.prices = self.books = None
        self.exchanges = []
        self.errors = {}
        self._last_logged = {}
        self.started_at = time.time()
        self.analysis = None
        self.readiness = None
        self.lab = None
        self.research_running = False
        self.notifier = Notifier(cfg, self.store, self, env=env)
        self.health = Health(self)
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._stream_thread if self.streaming else self._rest_loop, daemon=True,
                         name="engine").start()
        for seconds, fn in ((30, self._settle), (60, self.refresh_lab), (300, self.refresh_analysis),
                            (300, self._daily), (10, self.notifier.tick), (300, self._refresh_balance),
                            (60, self.health.check)):
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
                                            {a: f"{a}usdt" for a in self.by_asset},
                                            lambda asset, ts, price: self._on_fast("relay", asset, ts, price))
        self.exchanges = []
        if self.cfg.exchange_feeds:
            self.exchanges = [
                feeds.BinanceStream({a: feeds.BINANCE_SYMBOLS[a] for a in self.by_asset if a in feeds.BINANCE_SYMBOLS},
                                    self._on_fast),
                feeds.CoinbaseStream({a: feeds.COINBASE_SYMBOLS[a] for a in self.by_asset if a in feeds.COINBASE_SYMBOLS},
                                     self._on_fast),
            ]
        self.books = feeds.BookStream(self._on_book)
        self._dirty, self._wake = {}, asyncio.Event()
        await asyncio.gather(self.prices.run(), self.books.run(), self._heartbeat(), self._evaluator(),
                             *(x.run() for x in self.exchanges))

    # Updates are applied the moment they arrive (cheap); evaluating a market against
    # them is not cheap and is batched instead. A burst of a hundred book changes then
    # costs one evaluation on the newest state, not a hundred on ever staler ones, so
    # the event loop never builds a backlog.
    def _mark(self, tracker):
        self._dirty.setdefault(tracker, time.time())  # when the oldest unhandled update arrived
        self._wake.set()

    async def _evaluator(self):
        while not self._stop.is_set():
            await self._wake.wait()
            self._wake.clear()
            await asyncio.sleep(0.02)  # let the rest of the burst arrive
            dirty, self._dirty = self._dirty, {}
            now = time.time()
            for tracker, arrived in dirty.items():
                self._evaluate(tracker, now)
                done = time.time()
                self.health.record_latency(done, done - arrived)

    def _on_price(self, asset, ts, price):
        self.health.count("chainlink")
        for tracker in self.by_asset.get(asset, ()):
            self._guard(tracker.key, tracker.on_price, ts, price)
            self._mark(tracker)

    def _on_fast(self, source, asset, ts, price):
        self.health.count(source)
        self.health.fast_seen(source, asset)
        for tracker in self.by_asset.get(asset, ()):
            tracker.on_fast(ts, price, source)
            self._mark(tracker)

    def _on_book(self, token):
        self.health.count("book")
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
            for stream in self.exchanges:
                stream.flush()
            before = time.time()
            await asyncio.sleep(0.25)
            after = time.time()
            # Waking late means the loop was busy: the first sign of a backlog.
            self.health.record_loop_lag(after, max(0.0, after - before - 0.25))

    # The main strategy: which rule trades which market. Starts as the configured rule on
    # cfg.assets x cfg.window_minutes; the owner can promote proven lab rules into it.
    def _json_setting(self, key, default):
        try:
            return json.loads(self.store.get_setting(key) or "null") or default
        except ValueError:
            return default

    def main_strategies(self):
        default = [{"asset": a, "minutes": self.cfg.window_minutes, "since": 0.0,
                    "variant": lab.Variant(self.cfg.min_edge, self.cfg.min_seconds_left, self.cfg.max_seconds_left,
                                           self.cfg.resolution).key} for a in self.cfg.assets]
        return self._json_setting("main_strategies", default)

    def _apply_main(self, main):
        rules = {(m["asset"], m["minutes"]): lab.Variant.from_key(m["variant"]) for m in main}
        for t in self.trackers:
            rule = rules.get((t.asset, t.minutes))
            t.set_main_rule(rule, 1.0 if rule else self.cfg.lab_record_seconds)
        self.main_trackers = [t for t in self.trackers if t.main]

    def promote(self, asset, minutes, variant):
        """Make a proven lab rule the main strategy on its market (the owner's decision)."""
        row = next((b for b in (self.lab or {}).get("board", [])
                    if (b["asset"], b["minutes"], b["variant"]) == (asset, minutes, variant)), None)
        if row is None or row["status"] != "proven":
            raise ValueError("אפשר לקדם רק אסטרטגיה מוכחת")
        if not any((t.asset, t.minutes) == (asset, minutes) for t in self.trackers):
            raise ValueError("השוק הזה לא במעקב")
        main = [m for m in self.main_strategies() if (m["asset"], m["minutes"]) != (asset, minutes)]
        main.append({"asset": asset, "minutes": minutes, "variant": variant, "since": time.time()})
        self._save_main(main)

    def demote(self, asset, minutes):
        main = [m for m in self.main_strategies() if (m["asset"], m["minutes"]) != (asset, minutes)]
        if not main:
            raise ValueError("חייבת להישאר לפחות אסטרטגיה ראשית אחת")
        self._save_main(main)

    # The risk/reward filter on the main strategy (the owner's setting, or the data's choice).
    def risk_reward(self):
        s = self._json_setting("risk_reward", {})
        mode = s.get("mode") if s.get("mode") in riskreward.MODES else "off"
        return {"mode": mode, "ratio": float(s.get("ratio") or 2.0)}

    def set_risk_reward(self, mode, ratio=None):
        if mode not in riskreward.MODES:
            raise ValueError("מצב לא מוכר")
        current = self.risk_reward()
        if ratio is not None:
            ratio = float(ratio)
            if not 0 < ratio <= 20:
                raise ValueError("היחס צריך להיות גדול מ-0 ועד פי 20")
            current["ratio"] = ratio
        current["mode"] = mode
        self.store.set_setting("risk_reward", json.dumps(current))
        self._apply_risk_reward()
        log.warning("Risk/reward filter: %s (ratio now %.2f)", mode, self.rr_effective)
        self.refresh_lab()

    def _apply_risk_reward(self):
        s = self.risk_reward()
        ratio = {"manual": s["ratio"], "auto": self.rr_auto[0]}.get(s["mode"], 0.0)
        self.rr_effective = ratio
        for t in self.trackers:
            t.set_min_ratio(ratio)

    def _save_main(self, main):
        self.store.set_setting("main_strategies", json.dumps(main))
        # Readiness for real money is judged afresh on the new main strategy's own bets.
        self.store.set_setting("main_changed_at", str(time.time()))
        self._apply_main(main)
        log.warning("Main strategy is now: %s", ", ".join(f"{m['asset']}-{m['minutes']}m {m['variant']}" for m in main))
        self.refresh_lab()

    # The lab's variants: the fixed grid plus what the search near winners added.
    def _lab_variants(self, asset, minutes):
        extra = self._json_setting("lab_dynamic", {}).get(_key(asset, minutes), [])
        base = lab.variants_for(minutes)
        keys = {v.key for v in base}
        return base + [lab.Variant.from_key(k) for k in extra if k not in keys]

    def _refresh_variant_set(self):
        self.current_variants = {(t.asset, t.minutes, v.key) for t in self.trackers for v in t.variants}
        dyn = self._json_setting("lab_dynamic", {})
        self.dynamic_variants = {(*k.split("|"), v) for k, vs in dyn.items() for v in vs}
        self.dynamic_variants = {(a, int(m), v) for a, m, v in self.dynamic_variants}

    def explore(self, board, now):
        """Search near winners: add neighbours of the leading variants, retire added ones that lose."""
        if not self.with_lab or now - self._last_explore < self.cfg.explore_every_seconds:
            return
        self._last_explore = now
        dyn = self._json_setting("lab_dynamic", {})
        changed = False
        by_key = {(b["asset"], b["minutes"], b["variant"]): b for b in board}
        for (asset, minutes, variant) in list(self.dynamic_variants):
            b = by_key.get((asset, minutes, variant))
            if b and b["status"] in ("losing", "failed"):
                dyn[_key(asset, minutes)].remove(variant)
                changed = True
        total = sum(len(v) for v in dyn.values())
        leaders = [b for b in board if b["status"] in ("proven", "candidate", "promising")
                   and b["bets"] >= self.cfg.lab_min_bets]
        leaders.sort(key=lambda b: -(b["roi_pessimistic"] if b["roi_pessimistic"] is not None else b["roi"]))
        for b in leaders:
            market = _key(b["asset"], b["minutes"])
            have = {v for (a, m, v) in self.current_variants if (a, m) == (b["asset"], b["minutes"])}
            for near in lab.neighbours(b["variant"], b["minutes"]):
                if near in have:
                    continue
                if total >= self.cfg.explore_total or len(dyn.get(market, [])) >= self.cfg.explore_per_market:
                    break
                dyn.setdefault(market, []).append(near)
                have.add(near)
                total += 1
                changed = True
        if changed:
            self.store.set_setting("lab_dynamic", json.dumps(dyn))
            for t in self.trackers:
                t.set_variants(self._lab_variants(t.asset, t.minutes))
            self._refresh_variant_set()
            log.warning("Lab search: now %d added strategies near the leaders", total)

    # Analysis
    def refresh_analysis(self):
        """The slow numbers, every few minutes: prediction accuracy, what-if, calibration."""
        db = connect(self.cfg.db_path)
        try:
            markets = [(t.asset, t.minutes) for t in self.main_trackers]
            self.calibration = calibration.calibrate(db, self.cfg)
            self.analysis = dict(stats.analysis(db, self.cfg, markets), calibration=stats.calibration(db),
                                 auto_calibration=self.calibration)
            self.rr_table = riskreward.table(db, self.cfg)
            self.rr_auto = riskreward.choose(self.rr_table, self.cfg.rr_min_bets)
        finally:
            db.close()
        self._apply_risk_reward()
        for t in self.trackers:
            t.calib_k = self.calibration["k"]
        self.store.prune_ticks(time.time() - self.cfg.keep_ticks_days * 86400)

    def refresh_lab(self):
        """Every minute: the lab ranking (with its stages and search) and the readiness checks."""
        now = time.time()
        candidates = {tuple(k.split("|", 2)): ts for k, ts in self._json_setting("lab_candidates", {}).items()}
        candidates = {(a, int(m), v): ts for (a, m, v), ts in candidates.items()}
        db = connect(self.cfg.db_path)
        try:
            board = lab.leaderboard(db, self.cfg, self.current_variants, candidates, self.dynamic_variants)
            fresh = [b for b in board["board"] if b["new_candidate"]]
            if fresh:
                stored = self._json_setting("lab_candidates", {})
                for b in fresh:
                    stored[_key(b["asset"], b["minutes"], b["variant"])] = now
                    b["candidate_since"], b["new_candidate"] = now, False
                self.store.set_setting("lab_candidates", json.dumps(stored))
                log.warning("Lab: %d strategies became candidates; they are now tested on new data", len(fresh))
            self.lab = board
            first = db.execute("SELECT MIN(ts) FROM ticks").fetchone()[0]
            last = db.execute("SELECT MAX(ts) FROM ticks").fetchone()[0]
            hours = (last - first) / 3600 if first else 0.0
            since = float(self.store.get_setting("main_changed_at", "0"))
            readiness = stats.readiness(self.cfg, hours, stats.feed_agreement(db, self.cfg.resolution),
                                        stats.bet_summary(db, "paper", since=since, min_ratio=self.rr_effective))
            self.readiness = dict(readiness, computed_at=now, since=since)
            self.live.set_readiness(self.readiness["ready"])
        finally:
            db.close()
        self.explore(board["board"], now)

    def _refresh_balance(self):
        self.live.balance()

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

    def _positions(self, db, now):
        """Open paper and lab bets, grouped into one card per market window, with what
        Polymarket shows for it right now: both prices, the price to beat, the live
        price and its path through the window, and our entry."""
        rows = db.execute(
            "SELECT mode, variant, ts, asset, minutes, window_start, side, seconds_left, fair_prob, avg_price,"
            " shares, cost + fee AS paid FROM bets WHERE mode IN ('paper', 'lab', 'live') AND status = 'filled'"
            " AND pnl IS NULL AND ts > ? ORDER BY ts",
            (now - 3 * 3600,),
        ).fetchall()
        trackers = {(t.asset, t.minutes): t for t in self.trackers}
        cards = {}
        for r in rows:
            key = (r["mode"], r["asset"], r["minutes"], r["window_start"])
            card = cards.get(key)
            if card is None:
                t = trackers.get((r["asset"], r["minutes"]))
                if t is None:
                    continue
                start, length = r["window_start"], r["minutes"] * 60
                card = {"mode": r["mode"], "asset": r["asset"], "minutes": r["minutes"], "window_start": start,
                        "end": start + length, "live": t.window == start, "bets": []}
                if card["live"]:
                    snap = t.snapshot
                    card.update({k: snap.get(k) for k in ("price", "open_price", "seconds_left", "fair_up",
                                                         "up_bid", "up_ask", "down_bid", "down_ask")})
                else:
                    w = db.execute("SELECT open_twap, close_twap, winner FROM windows"
                                   " WHERE asset = ? AND minutes = ? AND window_start = ?",
                                   (r["asset"], r["minutes"], start)).fetchone()
                    card.update({"open_price": w["open_twap"] if w else None,
                                 "price": w["close_twap"] if w else None, "seconds_left": 0})
                # The Chainlink path through the window, thinned to at most ~120 points. After
                # a restart the in-memory history is gone, so fall back to the stored ticks.
                path = [(ts, p) for ts, p in t.vol.samples if start - 5 <= ts <= start + length]
                if len(path) < 10:
                    path = [tuple(row) for row in db.execute(
                        "SELECT ts, price FROM ticks WHERE asset = ? AND minutes = ? AND window_start = ?"
                        " AND price IS NOT NULL ORDER BY ts", (r["asset"], r["minutes"], start))]
                step = max(1, len(path) // 120)
                card["series"] = [(round(ts - start, 1), p) for ts, p in path[::step]]
                cards[key] = card
            card["bets"].append({k: r[k] for k in ("variant", "ts", "side", "seconds_left", "fair_prob",
                                                   "avg_price", "shares", "paid")})
        out = {"paper": [], "lab": [], "live": []}
        for card in sorted(cards.values(), key=lambda c: (not c["live"], c["end"], c["asset"])):
            for b in card["bets"]:
                if b["variant"] != "main":
                    b["rule"] = lab.describe(b["variant"])
            out[card["mode"]].append(card)
        return out

    def _alerts(self, now):
        """Problems the owner must see at a glance, in plain words, for the red banner."""
        alerts = []
        settled_in = now - self.started_at > 30  # streams need a few seconds after a start
        if self.streaming and self.prices and settled_in:
            last = self.prices.last_price
            if last is None or now - last > 30:
                alerts.append(f"מחירי Chainlink לא מגיעים כבר {int(now - (last or self.started_at))} שניות. "
                              "המערכת מתחברת מחדש לבד, ובינתיים לא מהמרת.")
        if self.streaming and self.books and settled_in:
            last = self.books.last_message
            if self.books.tokens and (last is None or now - last > 30):
                alerts.append(f"ספרי ההזמנות של פולימרקט לא מתעדכנים כבר {int(now - (last or self.started_at))} שניות. "
                              "המערכת מתחברת מחדש לבד, ובינתיים לא מהמרת.")
        if os.environ.get("RENDER") and not os.path.abspath(self.cfg.db_path).startswith("/data/"):
            alerts.append("הנתונים נשמרים בדיסק זמני ויימחקו בהפעלה הבאה של השרת. "
                          "ב-Render, POLYBOT_DB צריך להיות בתוך /data.")
        return alerts

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
                "alerts": self._alerts(now),
                "control": self.health.report(now),
                "health": {
                    "data_since": db.execute("SELECT MIN(ts) FROM ticks").fetchone()[0],
                    "price_reconnects": self.prices.reconnects if self.prices else 0,
                    "book_reconnects": self.books.reconnects if self.books else 0,
                    "last_stream_error": (self.prices.last_error if self.prices else None),
                },
                "config": {"assets": list(self.cfg.assets), "window_minutes": self.cfg.window_minutes,
                           "min_edge": self.cfg.min_edge, "stake": self.cfg.stake_usd,
                           "min_seconds_left": self.cfg.min_seconds_left,
                           "max_seconds_left": self.cfg.max_seconds_left, "resolution": self.cfg.resolution},
                "sources": sources,
                "errors": self.errors,
                "assets": [t.snapshot for t in self.main_trackers],
                "main": [dict(m, rule=lab.describe(m["variant"])) for m in self.main_strategies()],
                "auto_calibration": self.calibration,
                "risk_reward": dict(self.risk_reward(), effective=self.rr_effective,
                                    max_price=riskreward.max_price(self.rr_effective),
                                    auto={"ratio": self.rr_auto[0], "reason": self.rr_auto[1]},
                                    table=self.rr_table, min_bets=self.cfg.rr_min_bets),
                "paper": {"summary": paper,
                          "feed": {m: stats.feed_agreement(db, m) for m in ("twap", "point")},
                          "bets": stats.recent_bets(db, "paper", 20)},
                "analysis": self.analysis,
                "readiness": self.readiness,
                "notify": self.notifier.status(),
                "live": {"status": self.live.status(), "total": live_total, "today": live_today,
                         "bets": stats.recent_bets(db, "live")},
                "dashboard": self._dashboard(db, starts),
                "positions": self._positions(db, now),
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
