"""The engine: every second, record price, fair probability and Polymarket's order book for each
asset, log the paper bets the strategy would make, and hand the same signals to the live trader."""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import feeds, stats
from .live import LiveTrader, start_of_today
from .model import VolatilityEstimator, fair_up_probability
from .paper import find_bet, settle_pnl
from .store import Store, connect

log = logging.getLogger("polybot")


class Tracker:
    """Follows one asset through its consecutive windows."""

    def __init__(self, asset, cfg, store, api=feeds, live=None):
        self.asset, self.cfg, self.store, self.api, self.live = asset, cfg, store, api, live
        self.vol = VolatilityEstimator(cfg.vol_lookback_seconds)
        self.window = None
        self.market = None
        self.open_price = None
        self.bets_this_window = 0
        self.last_lookup = 0.0
        self.last_ok = {"price": None, "polymarket": None}
        self.snapshot = {"asset": asset}

    def step(self, now):
        cfg = self.cfg
        price = self.api.spot_price(self.asset)
        self.last_ok["price"] = now
        self.vol.add(now, price)

        start = feeds.window_start(now, cfg.window_minutes)
        if start != self.window:
            self._new_window(start, now, price)
        if self.market is None and now - self.last_lookup >= 10:
            self._lookup(start)

        seconds_left = start + cfg.window_minutes * 60 - now
        vol = self.vol.per_sqrt_second() if self.vol.history_seconds() >= cfg.vol_min_seconds else None
        fair_up = None
        if vol and self.open_price:
            fair_up = fair_up_probability(price, self.open_price, seconds_left, vol)

        up_bids = up_asks = down_bids = down_asks = []
        if self.market:
            up_bids, up_asks = self.api.order_book(self.market["up_token"])
            down_bids, down_asks = self.api.order_book(self.market["down_token"])
            self.last_ok["polymarket"] = now

        tick = dict(
            ts=now, asset=self.asset, window_start=start, price=price, open_price=self.open_price,
            seconds_left=seconds_left, vol=vol, fair_up=fair_up,
            up_bid=max((p for p, _ in up_bids), default=None),
            up_ask=min((p for p, _ in up_asks), default=None),
            down_bid=max((p for p, _ in down_bids), default=None),
            down_ask=min((p for p, _ in down_asks), default=None),
        )
        self.store.add_tick(**tick)
        self.snapshot = dict(tick, market_found=bool(self.market), warming_up=vol is None,
                             joined_midway=self.open_price is None, bets_this_window=self.bets_this_window)

        if (
            fair_up is not None
            and self.market
            and self.bets_this_window < cfg.max_bets_per_window
            and cfg.min_seconds_left <= seconds_left <= cfg.max_seconds_left
        ):
            fill = find_bet(fair_up, up_asks, down_asks, cfg)
            if fill:
                self.store.add_bet(now, self.asset, start, seconds_left, fill)
                self.bets_this_window += 1
                log.info(
                    "PAPER BET %s %s: fair %.0f%%, paid %.1fc/share (+fee), edge %.1f pts, %ds left",
                    self.asset.upper(), fill.side.upper(), fill.fair_prob * 100, fill.avg_price * 100,
                    fill.edge * 100, seconds_left,
                )
                if self.live:
                    self.live.maybe_trade(now, self.asset, start, seconds_left, fill, self.market)

    def _new_window(self, start, now, price):
        if self.window is not None:
            self.store.upsert_window(self.asset, self.window, our_end_price=price)
        self.window, self.market, self.bets_this_window = start, None, 0
        # Joining mid-window we don't know the opening price, so that window is
        # recorded but never bet on.
        self.open_price = price if now - start <= 3 else None
        self.store.upsert_window(self.asset, start, slug=feeds.market_slug(self.cfg, self.asset, start),
                                 found=0, open_price=self.open_price, open_source="coinbase")
        self._lookup(start)

    def _lookup(self, start):
        self.last_lookup = time.time()
        slug = feeds.market_slug(self.cfg, self.asset, start)
        self.market = self.api.find_market(slug)
        if not self.market:
            log.warning("No Polymarket market found for %s (will retry)", slug)
            return
        self.last_ok["polymarket"] = time.time()
        self.store.upsert_window(self.asset, start, found=1)
        if self.market["price_to_beat"] and self.open_price is not None:
            # Polymarket's own opening price is what the market resolves against.
            self.open_price = self.market["price_to_beat"]
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
    """Runs the trackers in a background thread and keeps the slow analysis fresh for the dashboard."""

    def __init__(self, cfg, api=feeds, env=None):
        self.cfg, self.api = cfg, api
        self.store = Store(cfg.db_path)
        self.live = LiveTrader(cfg, self.store, env=env)
        self.trackers = [Tracker(a, cfg, self.store, api=api, live=self.live) for a in cfg.assets]
        self.errors = {}
        self.started_at = time.time()
        self.analysis = None
        self.readiness = None
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="engine").start()
        threading.Thread(target=self._analysis_loop, daemon=True, name="analysis").start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        last_settle = 0.0
        last_logged = {}
        with ThreadPoolExecutor(len(self.trackers)) as pool:
            while not self._stop.is_set():
                now = time.time()
                for tracker, future in [(t, pool.submit(t.step, now)) for t in self.trackers]:
                    try:
                        future.result()
                        self.errors.pop(tracker.asset, None)
                    except Exception as e:
                        self.errors[tracker.asset] = f"{type(e).__name__}: {e}"
                        # Network hiccups are normal; log each asset's error at most once a minute.
                        if now - last_logged.get(tracker.asset, 0) > 60:
                            log.warning("%s: %s", tracker.asset.upper(), e)
                            last_logged[tracker.asset] = now
                if now - last_settle > 30:
                    try:
                        settle(self.store, self.cfg, now, api=self.api)
                    except Exception as e:
                        log.warning("settle: %s", e)
                    last_settle = now
                self._stop.wait(max(0.0, self.cfg.poll_seconds - (time.time() - now)))

    def _analysis_loop(self):
        while not self._stop.is_set():
            try:
                self.refresh_analysis()
            except Exception as e:
                log.warning("analysis: %s", e)
            self._stop.wait(60)

    def refresh_analysis(self):
        db = connect(self.cfg.db_path)
        try:
            self.analysis = stats.analysis(db, self.cfg)
            paper = stats.bet_summary(db, "paper")
            recent = stats.recent_roi(paper["returns"], self.cfg.ready_recent_bets)
            self.readiness = stats.readiness(self.cfg, self.analysis["hours"], stats.feed_agreement(db),
                                             paper, recent)
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
            sources = {}
            for name in ("price", "polymarket"):
                times = [t.last_ok[name] for t in self.trackers if t.last_ok[name]]
                age = now - min(times) if len(times) == len(self.trackers) else None
                sources[name] = {"ok": age is not None and age < 15, "age": age}
            return {
                "now": now,
                "uptime": now - self.started_at,
                "config": {"assets": list(self.cfg.assets), "window_minutes": self.cfg.window_minutes,
                           "min_edge": self.cfg.min_edge, "stake": self.cfg.stake_usd,
                           "min_seconds_left": self.cfg.min_seconds_left,
                           "max_seconds_left": self.cfg.max_seconds_left},
                "sources": sources,
                "errors": self.errors,
                "assets": [t.snapshot for t in self.trackers],
                "paper": {"summary": paper, "feed": stats.feed_agreement(db), "bets": stats.recent_bets(db, "paper", 20)},
                "analysis": self.analysis,
                "readiness": self.readiness,
                "live": {"status": self.live.status(), "total": live_total, "today": live_today,
                         "bets": stats.recent_bets(db, "live")},
            }
        finally:
            db.close()
