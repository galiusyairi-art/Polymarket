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
import functools
import gc
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import json
import math

from . import calibration, export, feeds, fills, lab, lead, marketcheck, race, refine, research, riskreward, rrlab, predict, stakes, stats, stocks, streaks, wallets
from .live import LiveTrader, reject_reason
from .health import Health
from .notify import Notifier
from .periods import period_starts
from .model import (VolatilityEstimator, fair_up_probability, fair_up_probability_twap, norm_cdf, t3_cdf,
                    taker_fee_per_share)
from .atlas import Atlas
from .atlas_mail import AtlasMail
from .equity import Equities, EquityStore, move_out_of
from .lagwatch import LagWatch
from .paper import Fill, buy_shares, find_bet, find_lead, find_lock, find_lottery, settle_pnl, simulate_buy
from .store import Store, connect

log = logging.getLogger("polybot")
MARKET_CHECK_GAP = 900  # seconds between market checks
STREAKS_GAP = 3600  # seconds between streak studies (streaks.py)


def _version():
    """The version number shown in the header (polybot/VERSION: 1.1, 1.11, 1.12, ...)."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _commit():
    """The running commit, for when the exact build matters."""
    commit = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("RAILWAY_GIT_COMMIT_SHA")
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2,
                                    cwd=os.path.dirname(__file__)).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = ""
    return commit[:7]


VERSION = _version()
COMMIT = _commit()


def _best(levels, pick):
    return pick((p for p, _ in levels), default=None)


@functools.lru_cache(maxsize=4096)
def _rule(key):
    try:
        return lab.Variant.from_key(key)
    except Exception:
        return None


def _entry(key):
    """A rule's entry from the research (lab.Variant.entry): (kind, number) or (None, 0)."""
    v = _rule(key)
    return lab.entry_of(v) if v else (None, 0)


def _lab_only(variant, mode="sim"):
    """Both-sides rules run in the lab only for now: the simulation and real money don't trade them yet."""
    if lab.family(variant) == "pair":
        raise ValueError("אסטרטגיית שני הצדדים רצה בינתיים רק במעבדה")


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
        # A market can run several rules side by side (the simulation's, and separately real money's):
        # each bets at most once a window. The first one drives the dashboard's reading of the market.
        self.main_rules = [main_rule] if main_rule else []
        self.calib_k = 1.0  # set by the engine from calibration.calibrate
        self.blend = None  # (a, b): the learned mix of model and market price, when it earned its place
        self.lagwatch = None  # the engine's "who moves first" meter (lagwatch.py), when it runs
        # Whether the simulation's and real money's decisions use that mix (the owner's choice). The
        # lab never does: it measures our own model, which a mix leaning on the market would hide.
        self.blend_sim = self.blend_live = True
        self.min_ratio = self.max_ratio = 0.0  # risk/reward range on the main strategy, set by the engine
        self.live_rules = []  # real money's own rules on this market (none: real money doesn't trade it)
        self.on_skip = None  # the engine's record of why a signal didn't become a trade
        self.sim_on = True  # the owner can switch the crypto strategy's simulation off
        self.sim_stake = cfg.stake_usd  # dollars per simulated main-strategy bet (the owner's setting)
        self.sim_copies = self.live_copies = 1  # entries into each approved trade (the money strip)
        self.can_open = None  # the engine's check: room under the open-trade and daily-loss limits
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
        self.recorded_vol = self._recorded_vol(asset, cfg, store)  # the run before a restart
        self.price = self.price_ts = None
        self.window = None
        self.market = None
        self.market_cfg = cfg  # cfg with this market's own fee schedule
        self.next_market = None  # the next window's market, fetched ahead of time
        self.opens = {"twap": None, "point": None}
        self.pair_legs = {}
        self.open_guesses = {}
        self._why_live_at = {}  # rule -> when real money's reasons for not trading were last looked at  # window start -> the price to beat shown when the opening wasn't measured
        self.main_done = set()  # rule keys the simulation bet with in this window
        self.live_done = set()  # rule keys real money traded (or tried) in this window
        self.lab_done = set()  # variant keys done for this window (bet as many times as they may)
        # Variants that may enter several times a window: key -> (entries so far, when the last was).
        self.lab_entries, self.main_entries, self.live_entries = {}, {}, {}
        self.lab_votes = {}  # rule key -> the side it entered in this window (the consensus rules count them)
        self.good = set()  # the lab's good rules on this market (promising and up): set by the engine
        self.maker_orders = {}
        self.sim_maker_orders = {}  # the simulation's maker strategies' resting orders in this window
        self.live_maker_want = {}  # real money's maker strategies: where each wants its order now...
        self.live_maker_have = {}  # ...and the real order resting on Polymarket (engine thread and orders share it)
        self._maker_lock = threading.Lock()
        self.lab_orders = []  # the lab's orders on their way to Polymarket (see lab_fill_delay)
        self.lab_pending = []  # entries from the research, still open: intents kept, bids standing (_pending_run)
        # The research's gates (lab.Variant.gate): the lab's recent trades winning (set by the engine); this round's
        # trades signed, the last minute (the buying flow); the price in 5-second buckets, an hour (breakouts); our
        # model fed by Binance's futures, when a rule here reads it.
        self.hot = lambda: False
        self.flow = collections.deque()
        self.buckets = collections.deque()
        self.alt = None
        self._uses_f = (None, False)
        self.live_intents = {}  # real money's refused orders kept open (entry "i"): rule key -> {"until", "steady", "window"}
        self.last_lookup = self.last_prefetch = 0.0
        self.last_record = 0.0
        self.snapshot = {"asset": asset, "minutes": self.minutes}
        self._set_market_cfg(None)

    @property
    def main(self):
        return bool(self.main_rules)

    @property
    def main_rule(self):
        return self.main_rules[0] if self.main_rules else None

    @property
    def live_rule(self):
        return self.live_rules[0] if self.live_rules else None

    @property
    def bets_this_window(self):
        return len(self.main_done)

    def set_main_rule(self, rule, record_seconds):
        self.set_main_rules([rule] if rule else [], record_seconds)

    def set_main_rules(self, rules, record_seconds):
        self.main_rules, self.record_seconds = list(rules), record_seconds
        self._set_market_cfg(self.market)

    def _rule_cfg(self, rule):
        """The simulation's sizing and filters for one of this market's rules (a rule that enters
        several times a window spends its share of the stake each time)."""
        return dataclasses.replace(self.market_cfg, min_edge=rule.min_edge, min_ratio=max(rule.ratio, self.min_ratio),
                                   max_ratio=self.max_ratio, stake_usd=(rule.stake or self.sim_stake) / lab.entries(rule))

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
        b = int(ts // 5) * 5
        if self.buckets and self.buckets[-1][0] == b:
            _, hi, lo, _ = self.buckets[-1]
            self.buckets[-1] = (b, max(hi, price), min(lo, price), price)
        else:
            self.buckets.append((b, price, price, price))
            while self.buckets and self.buckets[0][0] < ts - 3700:
                self.buckets.popleft()
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

    def nowcast(self, now, only=None):
        """(price, info_ts): the Chainlink price brought up to the present.

        Chainlink arrives ~1.2s late and the order book doesn't wait for it. Each fresh fast
        source says how much the market moved since the Chainlink timestamp; their average move
        is applied. Only moves are used, never levels (Binance quotes in USDT, and every venue's
        level differs a little from Chainlink's aggregate). Falls back to the raw Chainlink
        price when no fast source is fresh.
        """
        moves, newest = [], None
        for source, recent in self.fast_recent.items():
            # (Binance's futures: only for the rules that read them, gate "f"; never in our model's own price)
            if not recent or (source != only if only else source == "binance_fut"):
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
        if only:  # (no fresh futures price: our model's own)
            return self.nowcast(now)
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
        # Just restarted: the prices recorded before the restart stand in until the new ones
        # cover enough time, as long as the restart was short (not prices from hours ago).
        rec = self.recorded_vol
        if (rec and self.price_ts is not None and rec.samples[-1][0] >= self.price_ts - self.RECORDED_GAP
                and rec.history_seconds() >= self.cfg.vol_min_seconds):
            return rec.per_sqrt_second()
        return None

    RECORDED_GAP = 600  # the longest restart whose recorded prices still stand in

    @staticmethod
    def _recorded_vol(asset, cfg, store):
        """The prices recorded in the last lookback period (a sample a second), or None."""
        try:
            rows = store._x("SELECT ts, price FROM ticks WHERE asset = ? AND ts > ? AND price > 0 ORDER BY ts",
                            (asset, time.time() - cfg.vol_lookback_seconds))
        except Exception:
            return None
        series = VolatilityEstimator(cfg.vol_lookback_seconds)
        for ts, price in rows:
            if not series.samples or ts - series.samples[-1][0] >= 1.0:
                series.add(ts, price)
        return series if len(series.samples) >= 2 else None

    def _fair(self, resolution, price, info_ts, seconds_left, vol, end, now):
        # "twapt": the same TWAP reading, priced with fat tails (Student's t) instead of the normal curve.
        fat = resolution == "twapt"
        resolution = "twap" if fat else resolution
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
            return fair_up_probability_twap(price, open_price, seconds_left, vol, L, realized,
                                            cdf=t3_cdf if fat else norm_cdf)
        return fair_up_probability(price, open_price, seconds_left, vol)

    def evaluate(self, now, up_book, down_book, record=False):
        """Recompute fair probabilities against the current books; bet where the edge is there.

        up_book / down_book: (bids, asks, ts) or None.
        """
        cfg = self.cfg
        self._bids = {"up": up_book[0] if up_book else [], "down": down_book[0] if down_book else []}
        if self.price is None or self.window is None:
            return
        end = self.window + self.window_seconds
        seconds_left = end - now
        price, info_ts = self.nowcast(now)
        price_age = max(0.0, now - info_ts)
        # Clamped: Polymarket's clock can run a few tens of milliseconds ahead of ours.
        book_age = max(0.0, now - up_book[2], now - down_book[2]) if up_book and down_book else None
        vol = self.volatility()
        raw = {r: self._fair(r, price, info_ts, seconds_left, vol, end, now) for r in ("twap", "point", "twapt")}
        # Decisions use calibrated probabilities; the raw ones are stored with each bet, so
        # calibration is always fitted on what the model said, never on its own corrections.
        up_mid = ((_best(up_book[0], max) + _best(up_book[1], min)) / 2
                  if up_book and up_book[0] and up_book[1] else None)
        cal = {r: calibration.apply(p, self.calib_k) for r, p in raw.items()}
        blended = ({r: (calibration.blend(p, up_mid, *self.blend) if p is not None else None) for r, p in raw.items()}
                   if self.blend and up_mid is not None else None)
        fair = blended if blended and self.blend_sim else cal  # the simulation's view (and the dashboard's)
        fair_live = blended if blended and self.blend_live else cal
        self.alt = None
        if self.uses_futures():  # (gate "f": the same model, fed by Binance's futures)
            pf, info_f = self.nowcast(now, only="binance_fut")
            raw_f = {r: self._fair(r, pf, info_f, seconds_left, vol, end, now) for r in ("twap", "point", "twapt")}
            cal_f = {r: calibration.apply(p, self.calib_k) for r, p in raw_f.items()}
            bl_f = ({r: (calibration.blend(p, up_mid, *self.blend) if p is not None else None) for r, p in raw_f.items()}
                    if self.blend and up_mid is not None else None)
            self.alt = {"cal": cal_f, "fair": bl_f if bl_f and self.blend_sim else cal_f,
                        "live": bl_f if bl_f and self.blend_live else cal_f, "raw": raw_f}
        rule = self.main_rule
        resolution = lab.reading(rule) if rule else cfg.resolution  # (a lottery or several-entries rule reads twap)
        fair_up = fair[resolution]

        up_bids, up_asks = (up_book[0], up_book[1]) if up_book else ([], [])
        down_bids, down_asks = (down_book[0], down_book[1]) if down_book else ([], [])
        tick = dict(
            ts=now, asset=self.asset, minutes=self.minutes, window_start=self.window, price=price,
            open_price=self.open_price, seconds_left=seconds_left, vol=vol, fair_up=fair_up,
            up_bid=_best(up_bids, max), up_ask=_best(up_asks, min),
            down_bid=_best(down_bids, max), down_ask=_best(down_asks, min),
            price_age=price_age, book_age=book_age, fair_raw=raw[resolution],
        )
        # (the second's recording is a write to the database: after real money's rules, never before them)
        due = record or now - self.last_record >= self.record_seconds
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
        # For the control center: the best edge on offer right now (after the fee) against what the
        # rule needs, whether or not it is enough to bet.
        if rule and fair_up is not None:
            mc = self.main_cfg
            options = [(prob - ask - taker_fee_per_share(ask, mc.fee_rate, mc.fee_exponent), side)
                       for side, prob, ask in (("up", fair_up, self.snapshot.get("up_ask")),
                                               ("down", 1 - fair_up, self.snapshot.get("down_ask"))) if ask is not None]
            if options:
                edge, side = max(options)
                self.snapshot.update(edge=edge, edge_side=side, edge_need=rule.min_edge,
                                     # how many dollars the book offers right now with the rule's edge
                                     depth_usd=stakes.depth_now(up_asks if side == "up" else down_asks,
                                                                fair_up if side == "up" else 1 - fair_up,
                                                                rule.min_edge, mc))
        if not (self.market and fresh):
            if due:
                self.store.add_tick(**tick)
                self.last_record = now
            return
        if self.lagwatch is not None:  # (who moves first: read only, see lagwatch.py)
            self.lagwatch.observe(self.key, self.asset, self.minutes, now, cal.get("twap"), tick["up_bid"], tick["up_ask"],
                                  tick["down_bid"], tick["down_ask"], self.market_cfg.fee_rate, self.market_cfg.fee_exponent)

        # Real money, the simulation and the lab each run whatever happens to the others: a fault in
        # one (shown in the red banner) must never keep the others from trading.
        failed = None
        for part in (self._live_rules_run, self._sim_rules_run, self._run_lab):
            try:
                part(now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age, fair, fair_live)
            except Exception as e:
                failed = failed or e
            if due and part == self._live_rules_run:
                self.store.add_tick(**tick)
                self.last_record = now
        if failed:
            raise failed

    def _sim_rules_run(self, now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age, fair, fair_live):
        for r in self.main_rules:
            if lab.is_maker(r.key):
                # A maker strategy rests an order and waits for a seller, as in the lab (not a buy at
                # the offer: its min_edge is the most it pays a share, or the edge under the model).
                if self.sim_on and r.key not in self.main_done:
                    self._maker(r, now, seconds_left, cal.get("twap"), raw.get("twap"), up_asks, down_asks, price_age,
                                book_age, mode="paper", stake=self._rule_cfg(r).stake_usd)
                continue
            view, rview = self._view(r, fair, "fair"), self._view(r, raw, "raw")
            if view.get(lab.reading(r)) is not None:
                self._sim_bet(now, seconds_left, r, view[lab.reading(r)], rview[lab.reading(r)], up_asks, down_asks,
                              price_age, book_age)

    def _live_rules_run(self, now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age, fair, fair_live):
        if self.live:
            for r in self.live_rules:
                if lab.is_maker(r.key):
                    self._live_maker(r, now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age)
                    continue
                view = self._view(r, fair_live, "live")
                if lab.entry_of(r)[0] == "s":
                    self._live_stand(r, now, seconds_left, view, up_asks, down_asks)
                    continue
                if view.get(lab.reading(r)) is not None:
                    self._live_bet(now, seconds_left, r, view[lab.reading(r)], up_asks, down_asks,
                                   plain=self._view(r, cal, "cal").get(lab.reading(r)))  # the lab's view: no mix with the market

    @staticmethod
    def _decide(v, p, up_asks, down_asks, cfg):
        """A rule's bet on these books (None: no bet), with the rule's own filters (its price group and
        least chance: each side on its own, the better of those that pass)."""
        if not (v.price_hi or v.min_chance):
            return Tracker._decide_plain(v, p, up_asks, down_asks, cfg)
        fills = [f for f in (Tracker._decide_plain(v, p, up_asks, [], cfg), Tracker._decide_plain(v, p, [], down_asks, cfg))
                 if f is not None and v.passes(f.avg_price, f.fair_prob)]
        return max(fills, key=lambda f: f.edge * f.shares, default=None)

    @staticmethod
    def _decide_plain(v, p, up_asks, down_asks, cfg):
        """A rule's bet on these books (None: no bet): a lottery ticket, or the usual edge rule.
        The risk/reward range in cfg applies to both."""
        if lab.is_end(v):
            fill = find_lead(p, up_asks, down_asks, v.price_lo, min(v.price_hi, lab.END_CAP), cfg)
            if fill and v.resolution == "endm" and fill.edge < 0:
                return None  # (our model gives the side less than it costs)
            return fill
        if lab.is_lead(v):
            return find_lead(p, up_asks, down_asks, lab.lead_least(v), v.min_edge, cfg)
        if lab.is_lock(v):
            return find_lock(p, up_asks, down_asks, lab.lock_least(v), v.min_edge, lab.LOCK_EDGE, cfg)
        if lab.is_lotto(v):
            fill = find_lottery(p, up_asks, down_asks, v.min_edge, lab.lotto_min_fair(v), cfg)
            if fill and not riskreward.passes_range(fill.total_paid / fill.shares, cfg.min_ratio, cfg.max_ratio):
                return None
            return fill
        return find_bet(p, up_asks, down_asks, cfg)

    def uses_futures(self):
        """Does a rule here read our model fed by the futures (gate "f")? Counted again when the rules change."""
        key = (id(self.variants), id(self.main_rules), id(self.live_rules))
        if self._uses_f[0] != key:
            rules = list(self.variants or ()) + list(self.main_rules or ()) + list(self.live_rules or ())
            self._uses_f = (key, any(getattr(v, "gate", "") == "f" for v in rules))
        return self._uses_f[1]

    def _view(self, v, view, which):
        """A rule's probabilities: our model's (`view`), or with gate "f" the one fed by the futures."""
        if getattr(v, "gate", "") == "f" and self.alt:
            return self.alt[which]
        return view

    def _feature(self, name, now):
        """The prediction study's signals now (predict.py): the coin's move over the last minute, the round's
        buying flow over the last minute, a breakout of the last 15 or 60 minutes. None: no history."""
        if name == "mom":
            back = next((p for ts, p in reversed(self.vol.samples) if ts <= now - 60), None)
            return self.price / back - 1 if back and self.price else None
        if name == "flow":
            return sum(x for ts, x in self.flow if ts >= now - 60)
        if name in ("brk15", "brk60"):
            return predict.breakout(list(self.buckets), now, 15 if name == "brk15" else 60)
        return None

    def _gate_ok(self, v, side, now):
        """The research's gate on a rule (lab.Variant.gate): may it enter this side now?"""
        g = getattr(v, "gate", "") or ""
        if not g or g == "f":
            return True
        if g == "k":
            return bool(self.hot())
        if g.startswith("p"):
            x = self._feature(g[1:], now)
            return bool(x) and (x > 0) == (side == "up")
        return False

    @staticmethod
    def _due(entries, done, v, now):
        """May this rule enter (again) in this window: not done, and the gap since its last entry passed."""
        if v.key in done:
            return False
        count, last = entries.get(v.key, (0, 0.0))
        return count == 0 or now - last >= lab.ENTRY_GAP

    @staticmethod
    def _entered(entries, done, v, now):
        count = entries.get(v.key, (0, 0.0))[0] + 1
        entries[v.key] = (count, now)
        if count >= lab.entries(v):
            done.add(v.key)

    def _sim_bet(self, now, seconds_left, rule, fair_up, fair_raw, up_asks, down_asks, price_age, book_age):
        """The simulation's bet on this market, under the simulation's settings."""
        in_time = rule.min_seconds_left <= seconds_left <= rule.max_seconds_left
        if in_time and self.sim_on and self._due(self.main_entries, self.main_done, rule, now):
            rcfg = self._rule_cfg(rule)
            fill = self._decide(rule, fair_up, up_asks, down_asks, rcfg)
            if fill is not None and not self._gate_ok(rule, fill.side, now):
                return  # (the research's gate: not now)
            if fill is None:
                if not (lab.is_lotto(rule) or lab.is_lock(rule)):
                    self._why_not(rule, fair_up, up_asks, down_asks, rcfg)
            elif self.can_open and not self.can_open("crypto", fill.total_paid, fill.side, self.window):
                self._skipped(rule, "limits")  # the owner's limits: open trades, same direction, today's losses
                fill = None
            if fill:
                self._entered(self.main_entries, self.main_done, rule, now)
                side_raw = fair_raw if fill.side == "up" else 1 - fair_raw
                # Exactly as the lab fills (lab_fill_delay): the order meets the book when it arrives.
                self._send(now, "paper", rule.key, fill, seconds_left, price_age, book_age, side_raw, rcfg.stake_usd)
                for extra in self._copies(rule, fill, up_asks if fill.side == "up" else down_asks, rcfg, self.sim_copies):
                    if self.can_open and not self.can_open("crypto", extra.total_paid):
                        break
                    self._send(now, "paper", rule.key, extra, seconds_left, price_age, book_age, side_raw, rcfg.stake_usd)
                log.info("PAPER BET %s %s: fair %.1f%%, paid %.1fc/share (+fee), edge %.1f pts, %.1fs left,"
                         " data age price %.2fs / book %.2fs", self.key.upper(), fill.side.upper(),
                         fill.fair_prob * 100, fill.avg_price * 100, fill.edge * 100, seconds_left, price_age,
                         book_age)

    # More entries into the same approved trade (the owner's "entries per trade"): each one the same
    # amount, at nearly the same price as the first (at most COPY_SLIP more a share), and only while
    # the rule's own conditions still hold. In the simulation they take the next shares of the same
    # book, as real orders would; a copy that can't be filled whole isn't taken (nor any after it).
    COPY_SLIP = 0.01

    def _copies(self, rule, fill, asks, cfg, copies):
        if copies <= 1 or lab.is_lotto(rule) or lab.is_lock(rule) or lab.is_end(rule) or rule.resolution == "twapm" or not asks:
            return []
        out, prev = [], (fill.shares, fill.cost, fill.fee)
        for j in range(2, copies + 1):
            shares, cost, fee, top = simulate_buy(asks, cfg.stake_usd * j, fill.fair_prob, cfg)
            if cost + fee < cfg.stake_usd * j * 0.999:
                break
            d_sh, d_cost, d_fee = shares - prev[0], cost - prev[1], fee - prev[2]
            if d_sh <= 0 or d_cost / d_sh > fill.avg_price + self.COPY_SLIP + 1e-9:
                break
            out.append(Fill(side=fill.side, fair_prob=fill.fair_prob, best_ask=fill.best_ask, avg_price=d_cost / d_sh,
                            shares=d_sh, cost=d_cost, fee=d_fee, edge=fill.fair_prob - (d_cost + d_fee) / d_sh, limit_price=top))
            prev = (shares, cost, fee)
        return out

    @staticmethod
    def _own_room(rule):
        """Entry "m" from the research: the cents this rule may pay over the price seen (0: real money's own room)."""
        kind, n = lab.entry_of(rule)
        return n if kind == "m" else 0

    def _live_orders(self, now, seconds_left, fill, cfg, rule, stake, copies, round_stake=None, until=None):
        """Real money: the order, then the extra entries as their own fill-or-kill orders, each
        allowed at most COPY_SLIP above the first price and never past the rule's edge."""
        # (the room, when it is on: only for the rules that buy on an edge, which bounds it; a favourite or a
        # lottery rule buys its price band, and stays at the price it saw)
        plain = not (lab.is_lotto(rule) or lab.is_lock(rule) or lab.is_end(rule) or lab.is_lead(rule))
        first = self.live.maybe_trade(now, self.asset, self.minutes, self.window, seconds_left, fill, self.market, cfg,
                                      "crypto", None, rule.key, stake, round_stake=round_stake,
                                      room=(rule.min_edge, rule.ratio, self._own_room(rule)) if plain else None)
        kind, n = lab.entry_of(rule)
        if first == "rejected" and kind == "i" and (until or now + n) > time.time():
            # (entry "i": refused, the intent stays open until its time, see _live_bet)
            self.live_intents[rule.key] = {"until": until or now + n, "steady": None, "window": self.window}
        if first != "filled" or copies <= 1 or lab.is_lotto(rule) or lab.is_lock(rule) or lab.is_end(rule) or rule.resolution == "twapm":
            return
        tick = (self.market or {}).get("tick_size") or 0.01
        price = fill.limit_price + self.COPY_SLIP
        while price > fill.limit_price + 1e-9:
            per = price + taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
            if fill.fair_prob - per >= rule.min_edge and (not rule.ratio or 1 - per >= rule.ratio * per):
                break
            price -= tick
        price = max(price, fill.limit_price)
        extra = dataclasses.replace(fill, limit_price=round(price, 6))
        for _ in range(copies - 1):
            if self.live.maybe_trade(now, self.asset, self.minutes, self.window, seconds_left, extra, self.market, cfg,
                                     "crypto", None, rule.key, stake, copy=True, round_stake=round_stake) != "filled":
                break

    # ---------- real money's maker strategies: a real resting order, as the lab and the simulation run it ----------
    MAKER_CHECK = 2.0  # seconds between two looks at a resting order
    MAKER_RETRY = 5.0  # after Polymarket refused one (a post-only order that would have crossed, say)

    def _live_maker(self, v, now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age):
        """Where the order should rest (the lab's own rule, real money's amount), then the real one
        made to match: placed, moved (cancelled and placed again), checked for sellers, cancelled when
        the rule's time is over. Once a round, like the lab. The calls to Polymarket run in the
        background (dispatch), one at a time a strategy."""
        live = self.live
        if (not live.trades("crypto") or live.approval()) or v.key in self.live_done:
            self._live_maker_drop(v.key, now)  # (real money off, or the owner approves each trade: nothing rests)
            return
        stake = min(v.stake or live.order_cfg(self.market_cfg).stake_usd, self.cfg.live_max_stake_usd)
        self._maker(v, now, seconds_left, cal.get("twap"), raw.get("twap"), up_asks, down_asks, price_age, book_age,
                    mode="live", stake=stake)
        self._live_maker_sync(v, now, seconds_left, stake)

    def _live_stand(self, v, now, seconds_left, view, up_asks, down_asks):
        """Entry "s" with real money: at the signal, a real bid n cents under the offer, resting STAND_FOR seconds
        or until our model no longer sees the edge at it, through real money's resting orders (the makers')."""
        live = self.live
        n = lab.entry_of(v)[1]
        want = self.live_maker_want.get(v.key)
        p = view.get(lab.reading(v))
        stake = min(v.stake or live.order_cfg(self.market_cfg).stake_usd, self.cfg.live_max_stake_usd)
        if want is None:
            if (not live.trades("crypto") or live.approval() or v.key in self.live_done or p is None
                    or not (v.min_seconds_left <= seconds_left <= v.max_seconds_left)):
                if v.key in self.live_maker_have:
                    self._live_maker_drop(v.key, now)
                return
            cfg = dataclasses.replace(self.market_cfg, min_edge=v.min_edge)
            lcfg = live.order_cfg(cfg)
            lcfg = dataclasses.replace(lcfg, stake_usd=stake, min_ratio=max(lcfg.min_ratio, v.ratio))
            fill = self._decide(v, p, up_asks, down_asks, lcfg)
            if fill is None or not self._gate_ok(v, fill.side, now):
                return
            bid = round(fill.best_ask - n / 100, 2)
            if bid <= 0.01:
                return
            self.live_maker_want[v.key] = {"side": fill.side, "price": bid, "fair": fill.fair_prob, "raw": None,
                                           "shares": stake / bid, "live_at": now, "until": now + self.STAND_FOR}
            self._entered(self.live_entries, self.live_done, v, now)  # (once a round, as the lab)
        else:
            prob = (p if want["side"] == "up" else 1 - p) if p is not None else None
            if (now > want["until"] or seconds_left < v.min_seconds_left
                    or (prob is not None and prob - want["price"] < v.min_edge - 1e-9)):
                self.live_maker_want.pop(v.key, None)  # (its time is over, or the edge is gone: cancelled)
        self._live_maker_sync(v, now, seconds_left, stake)

    def _live_maker_sync(self, v, now, seconds_left, stake):
        """The real resting order made to match where the rule wants it (live_maker_want): placed, moved
        (cancelled and placed again), checked for sellers, cancelled when not wanted any more."""
        live = self.live
        want = self.live_maker_want.get(v.key)
        job = None
        with self._maker_lock:
            have = self.live_maker_have.get(v.key)
            if have and have.get("busy"):
                return
            if have and have.get("order_id"):
                if (not want or (want["side"], want["price"]) != (have["side"], have["price"])
                        or have["window"] != self.window):
                    have["busy"] = True  # moved or over: cancel (what sellers took by then still counts)
                    job = self._live_maker_close
                elif now - have.get("checked", 0) >= self.MAKER_CHECK:
                    have.update(busy=True, checked=now)
                    job = self._live_maker_check
        if job:
            self.dispatch(job, v.key, have, now, seconds_left)
            return
        if have and have.get("order_id"):
            return
        with self._maker_lock:
            if not want or (have and now < have.get("retry_at", 0)):
                return
            reason = live.blocked("crypto", want["side"], self.window, stake=stake)
            if reason:
                live.note_skip(self.asset, self.minutes, want["side"], reason)
                return
            shares = int(want["shares"] * 100) / 100
            least = (self.market or {}).get("min_order_size") or 0
            if shares < least:
                bigger = math.ceil(least * 1.01 * 100) / 100
                if bigger * want["price"] > stake + 1e-9:
                    live.note_skip(self.asset, self.minutes, want["side"],
                                   f"פולימרקט מקבל רק מ-{least:g} מניות: ${least * want['price']:.2f}, יותר מהסכום לטרייד (${stake:g})")
                    return
                shares = bigger
            token = self.market["up_token"] if want["side"] == "up" else self.market["down_token"]
            have = {"busy": True, "side": want["side"], "price": want["price"], "shares": shares, "fair": want["fair"],
                    "token": token, "window": self.window, "stake": stake, "checked": now}
            self.live_maker_have[v.key] = have
        self.dispatch(self._live_maker_place, v.key, have, now)

    def _live_maker_place(self, key, have, now):
        oid, error = self.live.place_maker(have["token"], have["price"], have["shares"])
        with self._maker_lock:
            if oid:
                have.update(order_id=oid, busy=False, placed=now)
                late = have["window"] != self.window  # (the round ended while it was on its way)
            else:
                have.update(busy=False, order_id=None, retry_at=time.time() + self.MAKER_RETRY)
                self.live.last_error = f"{time.strftime('%H:%M:%S')} maker {key}: {error}"
                self.live.note_skip(self.asset, self.minutes, have["side"], f"פולימרקט לא קיבל את ההצעה: {error}"[:200])
                return
        if late:
            self._live_maker_drop(key, now)

    def _live_maker_check(self, key, have, now, seconds_left):
        """Did sellers take any of it? Any shares taken: the rest is cancelled and the trade recorded
        (once a round, as the lab)."""
        try:
            matched, status = self.live.maker_matched(have["order_id"])
        except Exception as e:
            with self._maker_lock:
                have["busy"] = False
            self.live.last_error = f"{time.strftime('%H:%M:%S')} maker check {key}: {type(e).__name__}: {e}"[:300]
            return
        if matched > 0 or status in ("canceled", "cancelled", "matched"):
            self._live_maker_close(key, have, now, seconds_left)
            return
        with self._maker_lock:
            have["busy"] = False

    def _live_maker_close(self, key, have, now, seconds_left):
        """Cancel what still rests, then record what sellers took (if anything)."""
        oid = have.get("order_id")
        if oid:
            self.live.cancel_maker(oid)
            try:
                matched, _ = self.live.maker_matched(oid)
            except Exception as e:
                matched = 0.0
                self.live.last_error = f"{time.strftime('%H:%M:%S')} maker {key} after cancel: {type(e).__name__}: {e}"[:300]
            if matched > 0:
                self.live.record_maker_fill(time.time(), self.asset, self.minutes, have["window"], seconds_left,
                                            have["side"], have["price"], matched, have["fair"], oid, key, have["stake"])
                if have["window"] == self.window:
                    self.live_done.add(key)
        with self._maker_lock:
            if self.live_maker_have.get(key) is have:
                self.live_maker_have.pop(key, None)

    def _live_maker_drop(self, key, now):
        with self._maker_lock:
            have = self.live_maker_have.get(key)
            if not have or have.get("busy") or not have.get("order_id"):
                return
            have["busy"] = True
        self.live_maker_want.pop(key, None)
        self.dispatch(self._live_maker_close, key, have, now, 0.0)

    def _live_bet(self, now, seconds_left, rule, fair_up, up_asks, down_asks, plain=None):
        """Real money's trade on this market: its own rule and its own settings (stake,
        risk/reward, caps). The simulation's limits, or the simulation being off, never hold it
        back or let it through. In the approval mode nothing runs here: the owner's click buys."""
        if self.live and self.live.approval():
            return
        intent = self.live_intents.get(rule.key)
        if intent is not None and (intent["window"] != self.window or now > intent["until"]
                                   or seconds_left < rule.min_seconds_left):
            self.live_intents.pop(rule.key, None)
            intent = None
        if intent is None and not (rule.min_seconds_left <= seconds_left <= rule.max_seconds_left
                                   and self._due(self.live_entries, self.live_done, rule, now)):
            return
        cfg = dataclasses.replace(self.market_cfg, min_edge=rule.min_edge)
        lcfg = self.live.order_cfg(cfg)
        round_stake = rule.stake or lcfg.stake_usd  # this strategy's amount a round, or real money's
        lcfg = dataclasses.replace(lcfg, stake_usd=round_stake / lab.entries(rule),
                                   min_ratio=max(lcfg.min_ratio, rule.ratio))  # the rule's filter, as in the lab
        live_fill = self._decide(rule, fair_up, up_asks, down_asks, lcfg)
        if live_fill and not self._gate_ok(rule, live_fill.side, now):
            return  # (the research's gate: not now)
        if intent is not None:
            # Entry "i": the order refused stays open; it goes again at an offer still within the edge that
            # stayed STEADY (as the lab), until the intent's time is over.
            if not live_fill or self.live.blocked("crypto", live_fill.side, self.window, stake=round_stake):
                intent["steady"] = None
                return
            if intent["steady"] is None or abs(intent["steady"][0] - live_fill.limit_price) > 1e-9:
                intent["steady"] = (live_fill.limit_price, now)
                return
            if now - intent["steady"][1] < self.STEADY - 1e-9:
                return
            self.live_intents.pop(rule.key, None)
            self.dispatch(self._live_orders, now, seconds_left, live_fill, cfg, rule, lcfg.stake_usd, 1, round_stake,
                          intent["until"])
            return
        reason = self.live.blocked("crypto", live_fill.side, self.window, stake=round_stake) if live_fill else None
        if reason:
            if self.live.trades("crypto"):  # (with real money off there is nothing to explain)
                self.live.note_skip(self.asset, self.minutes, live_fill.side, reason)
        elif live_fill:
            self._entered(self.live_entries, self.live_done, rule, now)
            self.dispatch(self._live_orders, now, seconds_left, live_fill, cfg, rule, lcfg.stake_usd, self.live_copies,
                          round_stake)
        elif now - self._why_live_at.get(rule.key, 0.0) >= 1.0:
            # (at most once a second a rule: this runs on every book update, and the reasons are
            # the same from one update to the next)
            self._why_live_at[rule.key] = now
            if self.live.trades("crypto"):
                self._live_why_not(rule, fair_up, plain, up_asks, down_asks, lcfg)

    def _live_why_not(self, rule, fair_up, plain, up_asks, down_asks, lcfg):
        """No trade under real money's settings: when the rule would have bet without one of them
        (the lab's view), say which one held it back. No edge at all says nothing (the normal case)."""
        fill = self._decide(rule, plain, up_asks, down_asks, lcfg) if plain is not None and plain != fair_up else None
        if fill:
            self.live.note_skip(self.asset, self.minutes, fill.side,
                                "השקלול עם מחיר השוק (הגנה של הכסף האמיתי) ביטל את הפער. מכבים אותו במתג \"בדיוק כמו המעבדה\"")
            return
        loose = dataclasses.replace(lcfg, min_ratio=rule.ratio, max_ratio=0.0)
        if (lcfg.min_ratio, getattr(lcfg, "max_ratio", 0.0)) != (loose.min_ratio, 0.0):
            fill = self._decide(rule, fair_up, up_asks, down_asks, loose)
            if fill:
                self.live.note_skip(self.asset, self.minutes, fill.side,
                                    f"{fill.avg_price * 100:.0f}¢ למניה: מחוץ לטווח סיכוי / סיכון שהוגדר לכסף האמיתי")
                return
        partial = self._decide(rule, fair_up, up_asks, down_asks, dataclasses.replace(lcfg, full_fill_only=False))
        if partial and partial.total_paid < lcfg.stake_usd * 0.999:
            self.live.note_skip(self.asset, self.minutes, partial.side,
                                f"בספר ההזמנות יש רק ${partial.total_paid:.2f} במחיר טוב, פחות מ-${lcfg.stake_usd:.2f} לכניסה")

    # Why a rule with an edge didn't bet: the whole stake wasn't on offer at good prices. (No edge
    # at all is the normal case and isn't counted; a rule's own risk/reward filter is part of it.)
    def _why_not(self, rule, fair_up, up_asks, down_asks, rcfg):
        if not self.on_skip:
            return
        if self._decide(rule, fair_up, up_asks, down_asks, dataclasses.replace(rcfg, full_fill_only=False)) is not None:
            self._skipped(rule, "full")

    def _queue_at(self, side, price):
        """A new resting order's place in the queue: the shares resting at its price on that side
        already (they trade before it), and what traded there since (on_market_trade)."""
        ahead = sum(size for p, size in (getattr(self, "_bids", {}) or {}).get(side, []) if abs(p - price) < 1e-9)
        return {"queue": float(ahead), "traded": 0.0}

    def on_market_trade(self, side, price, size, taker, ts):
        """Anyone's trade in this market: a seller taking bids at our resting price or under it moves
        the lab's queue twins ("|q") forward. Every trade counts in the buying flow (gate "p flow")."""
        if size:
            self.flow.append((ts, size if (side == "up") == (taker == "BUY") else -size))
            while self.flow and self.flow[0][0] < ts - 60:
                self.flow.popleft()
        if taker != "SELL":
            return
        for key, o in list(self.maker_orders.items()):
            if key.endswith("|q") and o["side"] == side and price <= o["price"] + 1e-9 and ts >= o["live_at"]:
                o["traded"] = o.get("traded", 0.0) + size

    def _skipped(self, rule, reason):
        if self.on_skip:
            self.on_skip(self, rule, reason)

    def _maker(self, v, now, seconds_left, p, p_raw, up_asks, down_asks, price_age, book_age, mode="lab", stake=None):
        """A resting buy order at our own price (simulated). Filled only when a seller comes down
        through it: the best ask drops below our price (sellers took the whole level, with whoever
        was in line there before us). No taker fee. Like a real order, it rests only from
        lab_fill_delay after it was sent, and a moved order is a new one (sent again).
        The lab's (mode "lab") and the simulation's maker strategies ("paper": its amount, and the
        owner's limits checked when it fills) alike, each with its own orders."""
        sim, live = mode == "paper", mode == "live"
        orders, done = ((self.sim_maker_orders, self.main_done) if sim else (self.live_maker_want, self.live_done) if live
                        else (self.maker_orders, self.lab_done))
        stake = stake or self.cfg.stake_usd
        order = orders.get(v.key)
        asks = {"up": up_asks, "down": down_asks}
        # (real money: here only where the order should rest; Polymarket's sellers fill it, see _live_maker)
        if order and not live and now >= order["live_at"]:
            ask = _best(asks[order["side"]], min)
            # Sellers broke through it, or (by the queue, v.fill "q") the shares resting at its price
            # before it, and then its own, traded.
            queued = v.fill == "q" and order.get("traded", 0.0) >= order.get("queue", 0.0) + order["shares"] - 1e-9
            if ask is not None and ask < order["price"] - 1e-9 or queued:
                ask = ask if ask is not None else order["price"]
                price, shares = order["price"], order["shares"]
                orders.pop(v.key, None)
                if sim and self.can_open and not self.can_open("crypto", shares * price, order["side"], self.window):
                    done.add(v.key)
                    self._skipped(v, "limits")
                    return
                fill = Fill(side=order["side"], fair_prob=order["fair"], best_ask=ask, avg_price=price, shares=shares,
                            cost=shares * price, fee=0.0, edge=order["fair"] - price, limit_price=price)
                done.add(v.key)
                if sim:
                    self._entered(self.main_entries, self.main_done, v, now)
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill, mode=mode,
                                   variant=v.key, price_age=price_age, book_age=book_age, fair_raw=order["raw"],
                                   stake=stake if sim else self.market_cfg.stake_usd)
                if sim:
                    log.info("PAPER MAKER FILL %s %s %s: %.1fc/share, %.0f shares, %.1fs left", self.key.upper(), v.key,
                             fill.side.upper(), price * 100, shares, seconds_left)
                return
        if p is None or not (v.min_seconds_left <= seconds_left <= v.max_seconds_left):
            orders.pop(v.key, None)
            return
        tick = (self.market or {}).get("tick_size") or 0.01
        if lab.is_fav(v) or lab.is_zone(v):
            # The market's leader, as a maker (no model): a resting buy on the side priced at least
            # fav_least, a tick under its offer and never above the cap (min_edge); or a zone's (the
            # maker check's finding): on the side priced in its range. It stays where it is once placed
            # (filled only if sellers come down through it). The same in the lab, the simulation and
            # real money: whatever filters it has are part of the rule itself.
            if order:
                return
            best = None
            for side, side_asks in (("up", up_asks), ("down", down_asks)):
                ask = _best(side_asks, min)
                if ask is None:
                    continue
                if lab.is_zone(v):
                    lo, hi = lab.zone_band(v)
                    price = round(ask - tick, 4)
                    if not lo - 1e-9 <= price < hi - 1e-9:
                        continue
                else:
                    if ask < lab.fav_least(v):
                        continue
                    price = round(min(v.min_edge, ask - tick), 4)
                    if price < lab.fav_least(v) - tick:
                        continue
                if best is None or ask > best[1]:
                    best = (side, ask, price)
            if best is None:
                return
            side, _, price = best
            prob = (p if side == "up" else 1 - p) if p is not None else 0.0
            raw_side = (p_raw if side == "up" else 1 - p_raw) if p_raw is not None else None
            orders[v.key] = {"side": side, "price": price, "fair": prob, "raw": raw_side,
                             "shares": stake / price, "live_at": now + self.cfg.lab_fill_delay, **self._queue_at(side, price)}
            return
        if order:
            # A resting order stays where it is while it still has its edge; it doesn't chase
            # the offer down (it would never be filled). Only a fair price that moved against
            # it pulls it lower.
            prob = p if order["side"] == "up" else 1 - p
            highest = round(math.floor((prob - v.min_edge) / tick + 1e-9) * tick, 4)
            if order["price"] <= highest:
                order["fair"] = prob
                return
        best = None
        for side, prob, side_asks in (("up", p, up_asks), ("down", 1 - p, down_asks)):
            target = math.floor((prob - v.min_edge) / tick + 1e-9) * tick
            ask = _best(side_asks, min)
            if ask is not None:
                target = min(target, ask - tick)  # resting: never crossing into a taker order
            if target >= tick and (best is None or prob - target > best[1]):
                best = (side, prob - target, round(target, 4), prob)
        if best is None:
            orders.pop(v.key, None)
            return
        side, _, price, prob = best
        raw_side = p_raw if side == "up" else (1 - p_raw if p_raw is not None else None)
        if order and (order["side"], order["price"]) == (side, price):
            order.update(fair=prob, raw=raw_side)  # the same order, still resting
            return
        orders[v.key] = {"side": side, "price": price, "fair": prob, "raw": raw_side,
                         "shares": stake / price, "live_at": now + self.cfg.lab_fill_delay, **self._queue_at(side, price)}

    def _run_lab(self, now, seconds_left, fair, raw, up_asks, down_asks, price_age, book_age, *_):
        if self.lab_orders:
            self._arrive(now, up_asks, down_asks)
        if self.lab_pending:
            self._pending_run(now, seconds_left, fair, up_asks, down_asks)
        if not self.variants:
            return
        best_up = _best(up_asks, min)
        best_down = _best(down_asks, min)
        mc = self.market_cfg
        cfgs = self.lab_cfgs  # read once: the engine may swap in a new set of variants
        for v in self.variants:
            if lab.is_pair(v):
                self._pair(v, now, seconds_left, fair, raw, (up_asks, down_asks), (best_up, best_down), price_age,
                           book_age, cfgs.get(v.key))
                continue
            if lab.is_maker(v.key) and v.key not in self.lab_done:
                self._maker(v, now, seconds_left, fair["twap"], raw["twap"], up_asks, down_asks, price_age, book_age)
                continue
            if lab.is_cons(v):
                self._consensus(v, now, seconds_left, fair, raw, up_asks, down_asks, price_age, book_age, cfgs.get(v.key))
                continue
            if not (v.min_seconds_left <= seconds_left <= v.max_seconds_left) or not self._due(
                    self.lab_entries, self.lab_done, v, now):
                continue
            p = self._view(v, fair, "cal").get(lab.reading(v))
            if p is None:
                continue
            # Cheap check on the best prices before walking the books.
            if lab.is_end(v):
                if not any(ask is not None and v.price_lo <= ask <= min(v.price_hi, lab.END_CAP) for ask in (best_up, best_down)):
                    continue
            elif lab.is_lead(v):
                if not any(ask is not None and lab.lead_least(v) <= ask <= v.min_edge for ask in (best_up, best_down)):
                    continue
            elif lab.is_lock(v):
                if not any(ask is not None and ask <= v.min_edge and prob >= lab.lock_least(v)
                           for prob, ask in ((p, best_up), (1 - p, best_down))):
                    continue
            elif lab.is_lotto(v):
                if not any(ask is not None and ask <= v.min_edge for ask in (best_up, best_down)):
                    continue
            else:
                edges = [prob - ask - taker_fee_per_share(ask, mc.fee_rate, mc.fee_exponent)
                         for prob, ask in ((p, best_up), (1 - p, best_down)) if ask is not None]
                if not edges or max(edges) < v.min_edge:
                    continue
            v_cfg = cfgs.get(v.key)
            if v_cfg is None:
                continue
            fill = self._decide(v, p, up_asks, down_asks, v_cfg)
            if fill and not self._gate_ok(v, fill.side, now):
                fill = None
            if fill:
                self._entered(self.lab_entries, self.lab_done, v, now)
                self.lab_votes.setdefault(v.key, fill.side)
                r = raw[lab.reading(v)]
                side_raw = None if r is None else (r if fill.side == "up" else 1 - r)
                self._send(now, "lab", v.key, fill, seconds_left, price_age, book_age, side_raw, v_cfg.stake_usd)

    def _consensus(self, v, now, seconds_left, fair, raw, up_asks, down_asks, price_age, book_age, v_cfg):
        """A consensus rule (lab.CONS_RULES): the side enough of the lab's rules agreed on in this round, bought at
        no more than its cap a share, the whole amount or nothing."""
        if v_cfg is None or not (v.min_seconds_left <= seconds_left <= v.max_seconds_left) or not self._due(
                self.lab_entries, self.lab_done, v, now):
            return
        side = lab.consensus_side(v, self.lab_votes, self.good)
        p = fair.get("twap")
        if side is None or p is None:
            return
        fill = find_lottery(p, up_asks if side == "up" else [], down_asks if side == "down" else [], v.min_edge, 0.0, v_cfg)
        if fill:
            self._entered(self.lab_entries, self.lab_done, v, now)
            r = raw.get("twap")
            self._send(now, "lab", v.key, fill, seconds_left, price_age, book_age,
                       None if r is None else (r if side == "up" else 1 - r), v_cfg.stake_usd)

    def _send(self, now, mode, variant, fill, seconds_left, price_age, book_age, fair_raw, stake):
        """An order of the lab or the simulation, on its way to the book like a real one: filled (or
        not) when it arrives, lab_fill_delay later (see _arrive). Both alike, so the simulation runs
        a strategy exactly as the lab measured it."""
        order = {"due": now + self.cfg.lab_fill_delay, "ts": now, "mode": mode, "variant": variant, "fill": fill,
                 "seconds_left": seconds_left, "price_age": price_age, "book_age": book_age,
                 "fair_raw": fair_raw, "stake": stake}
        kind, n = _entry(variant)
        if kind == "s":  # (standing in the book instead of taking: a bid n cents under the offer seen)
            bid = round(fill.best_ask - n / 100, 2)
            if bid > 0.01:
                self.lab_pending.append({"kind": "s", "order": order, "bid": bid, "shares": (fill.cost + fill.fee) / bid,
                                         "live": now + self.cfg.lab_fill_delay, "until": now + self.STAND_FOR})
            return
        if self.cfg.lab_fill_delay > 0:
            self.lab_orders.append(order)
        else:
            self._lab_record(order, fill, "filled")

    def _arrive(self, now, up_asks, down_asks):
        """The lab's and the simulation's orders that reached Polymarket by now meet the book as it is
        now: like a real money's order (fill-and-kill, since 28.10.2026), the shares at the price seen or
        better, as many as are left; none left, refused (recorded as such, so the results carry the
        orders a real one would have lost). The simulation's orders arriving together take the book one
        after the other (a rule's extra entries, as real orders would); each lab rule is its own
        experiment, on the whole book."""
        due = [o for o in self.lab_orders if o["due"] <= now]
        if not due:
            return
        self.lab_orders = [o for o in self.lab_orders if o["due"] > now]
        mc = self.market_cfg
        taken = {}  # (side, price) -> shares the simulation's orders took in this pass
        for o in due:
            f = o["fill"]
            asks = up_asks if f.side == "up" else down_asks
            sim = o.get("mode") == "paper"
            kind, n = _entry(o["variant"])
            limit = f.limit_price
            if kind == "m":  # (up to n cents over the price seen, never past the rule's edge)
                limit = self._room_limit(_rule(o["variant"]), f.fair_prob, f.limit_price, n)
            got = cost = fee = top = 0.0
            for price, size in sorted(asks or ()):
                if price > limit + 1e-9 or got >= f.shares - 1e-9:
                    break
                if sim:
                    size = max(0.0, size - taken.get((f.side, price), 0.0))
                take = min(size, f.shares - got)
                if sim and take > 0:
                    taken[(f.side, price)] = taken.get((f.side, price), 0.0) + take
                got, cost, top = got + take, cost + take * price, price
                fee += take * taker_fee_per_share(price, mc.fee_rate, mc.fee_exponent)
            if got < self.LEAST_FILL:
                if kind == "i":  # (the intent kept open n seconds: _pending_run)
                    self.lab_pending.append({"kind": "i", "order": o, "until": o["ts"] + n, "steady": None})
                    continue
                self._lab_record(o, f, "rejected")
                self._fill_test(o, f, asks, mc)
                continue
            per = (cost + fee) / got
            filled = Fill(side=f.side, fair_prob=f.fair_prob, best_ask=_best(asks, min), avg_price=cost / got, shares=got,
                          cost=cost, fee=fee, edge=f.fair_prob - per, limit_price=top, depth=getattr(f, "depth", None))
            self._lab_record(o, filled, "filled")

    STAND_FOR = 30.0  # seconds a standing bid (entry "s") rests (race.BID_FOR)
    STEADY = 1.0  # an intent buys at an offer that stayed this long (race.STEADY)

    def _per(self, price):
        mc = self.market_cfg
        return price + taker_fee_per_share(price, mc.fee_rate, mc.fee_exponent)

    def _worth(self, v, prob, price, fee=True):
        """Is a share at `price` still within the rule's edge and filters, at our model's chance `prob`?"""
        per = self._per(price) if fee else price
        return (prob - per >= v.min_edge - 1e-9 and (not v.ratio or 1 - per >= v.ratio * per - 1e-12)
                and v.passes(price, prob))

    def _room_limit(self, v, prob, seen, cents):
        """Entry "m": the highest price up to `cents` over the price seen that still keeps the rule's edge."""
        tick = (self.market or {}).get("tick_size") or 0.01
        p = round(seen + cents / 100, 4)
        while p > seen + 1e-9 and not (v and self._worth(v, prob, p)):
            p = round(p - tick, 4)
        return max(p, seen)

    def _pending_run(self, now, seconds_left, fair, up_asks, down_asks):
        """The research's entries still open: an intent buys at the first offer within the rule's edge (our model's
        chance now) that stayed STEADY; a standing bid fills when sellers go through it, and is cancelled when our
        model no longer sees the edge at it. Both end with the round or their time."""
        keep = []
        for x in self.lab_pending:
            o, f = x["order"], x["order"]["fill"]
            v = _rule(o["variant"])
            p = self._view(v, fair, "cal").get(lab.reading(v)) if v else None
            prob = None if p is None else (p if f.side == "up" else 1 - p)
            asks = up_asks if f.side == "up" else down_asks
            ask = _best(asks, min)
            if now > x["until"] or (v and seconds_left < v.min_seconds_left):
                self._lab_record(o, f, "rejected")
                continue
            if x["kind"] == "i":
                if prob is None or ask is None or not self._worth(v, prob, ask):
                    x["steady"] = None
                    keep.append(x)
                    continue
                if x["steady"] is None or abs(x["steady"][0] - ask) > 1e-9:
                    x["steady"] = (ask, now)
                if now - x["steady"][1] < self.STEADY - 1e-9:
                    keep.append(x)
                    continue
                got = cost = fee = 0.0
                for price, size in sorted(asks):
                    if price > ask + 1e-9 or got >= f.shares - 1e-9:
                        break
                    take = min(size, f.shares - got)
                    got, cost = got + take, cost + take * price
                    fee += take * (self._per(price) - price)
                if got < self.LEAST_FILL:
                    keep.append(x)
                    continue
                per = (cost + fee) / got
                self._lab_record(o, Fill(side=f.side, fair_prob=prob, best_ask=ask, avg_price=cost / got, shares=got, cost=cost,
                                         fee=fee, edge=prob - per, limit_price=ask), "filled")
            else:  # "s": a resting bid
                if now < x["live"]:
                    keep.append(x)
                    continue
                if prob is not None and not self._worth(v, prob, x["bid"], fee=False):
                    self._lab_record(o, f, "rejected")  # (cancelled: our model no longer sees the edge at it)
                    continue
                if ask is not None and ask < x["bid"] - 1e-9:
                    shares, bid = x["shares"], x["bid"]
                    self._lab_record(o, Fill(side=f.side, fair_prob=prob if prob is not None else f.fair_prob, best_ask=ask,
                                             avg_price=bid, shares=shares, cost=shares * bid, fee=0.0,
                                             edge=(prob if prob is not None else f.fair_prob) - bid, limit_price=bid), "filled")
                    continue
                keep.append(x)
        self.lab_pending = keep

    FILL_ROOM = (0.01, 0.02, 0.03, 0.05)  # the room tried on a refused order (fills.py)
    LEAST_FILL = 1.0  # shares: less left at the price is a refusal (a crumb, not a trade)

    def _fill_test(self, order, f, asks, mc):
        """A refused order: what was left at its price when it arrived, and what 1, 2, 3 and 5 cents more
        would have filled (all of it, fill-or-kill, the cheapest first) and cost."""
        try:
            levels = sorted(asks or ())
            left = sum(size for price, size in levels if price <= f.limit_price + 1e-9)
            out = []
            for room in self.FILL_ROOM:
                got = cost = fee = 0.0
                for price, size in levels:
                    if price > f.limit_price + room + 1e-9 or got >= f.shares - 1e-9:
                        break
                    take = min(size, f.shares - got)
                    got, cost = got + take, cost + take * price
                    fee += take * taker_fee_per_share(price, mc.fee_rate, mc.fee_exponent)
                out += [cost, fee] if got >= f.shares * 0.999 else [None, None]
            self.store.add_fill_test((order["ts"], self.asset, self.minutes, self.window, order.get("mode", "lab"),
                                      order["variant"], f.side, f.shares, f.limit_price, left, *out, 1))
        except Exception as e:  # (a study's record never stops the trading)
            log.warning("Fill test: %s", e)

    def _lab_record(self, order, fill, status):
        self.store.add_bet(order["ts"], self.asset, self.minutes, self.window, order["seconds_left"], fill,
                           mode=order.get("mode", "lab"),
                           variant=order["variant"], status=status, price_age=order["price_age"],
                           book_age=order["book_age"], fair_raw=order["fair_raw"], stake=order["stake"])

    def _recorded_twap(self, t0, t1):
        """The minute's average from the prices recorded each second (ticks), or None without them."""
        series = VolatilityEstimator(t1 - t0 + 1200)
        for ts, price in self.store._x("SELECT ts, price FROM ticks WHERE asset = ? AND ts BETWEEN ? AND ?"
                                       " AND price > 0 ORDER BY ts", (self.asset, t0 - 600, t1)):
            series.add(ts, price)
        return series.twap(t0, t1)

    def _backfill_closes(self, start):
        """After a restart: rounds of the last two hours that closed while the bot was down (or
        just restarting) have no closing average, so their cards can't say how they ended until
        Polymarket rules. It is taken from the prices recorded until the restart."""
        L = self.cfg.twap_seconds
        try:
            missing = self.store._x("SELECT window_start FROM windows WHERE asset = ? AND minutes = ?"
                                    " AND close_twap IS NULL AND window_start >= ? AND window_start + ? <= ?",
                                    (self.asset, self.minutes, start - 7200, self.window_seconds, start))
            for (ws,) in missing:
                end = ws + self.window_seconds
                close = self._recorded_twap(end - L, end)
                if close is not None:
                    self.store.upsert_window(self.asset, self.minutes, ws, close_twap=close)
        except Exception as e:
            log.warning("Closing prices after a restart (%s): %s", self.key, e)

    def _pair(self, v, now, seconds_left, fair, raw, asks, best, price_age, book_age, v_cfg):
        """A both-sides rule (see lab.py): the cheap side first, then the same number of shares of
        the other side once the two together cost at most the rule's target. asks: the (up, down)
        books; best: their lowest asks."""
        if v_cfg is None or v.key in self.lab_done:
            return
        books, best_ask = dict(zip(("up", "down"), asks)), dict(zip(("up", "down"), best))
        leg = self.pair_legs.get(v.key)
        if leg is None:
            if not (v.min_seconds_left <= seconds_left <= v.max_seconds_left):
                return
            shares = v_cfg.stake_usd  # each pair pays $1 a share at the close: a $10 stake is 10 pairs
            best = None
            for side in ("up", "down"):
                ask = best_ask[side]
                got = buy_shares(books[side], shares, v.min_edge, v_cfg) if ask is not None and ask <= v.min_edge else None
                if got and (best is None or got[0] + got[1] < best[1][0] + best[1][1]):
                    best = (side, got)
            if best is None:
                return
            side, got = best
            self._pair_leg(v, side, shares, got, now, seconds_left, fair, raw, books, price_age, book_age, v_cfg)
            leg = self.pair_legs[v.key] = (side, shares, (got[0] + got[1]) / shares)
        side, shares, per_share = leg
        if seconds_left < lab.PAIR_LAST:
            return
        other = "down" if side == "up" else "up"
        room = lab.pair_target(v) - per_share
        ask = best_ask[other]
        if ask is None or ask > room:
            return
        got = buy_shares(books[other], shares, room, v_cfg)
        if got:
            self._pair_leg(v, other, shares, got, now, seconds_left, fair, raw, books, price_age, book_age, v_cfg)
            self.lab_done.add(v.key)

    def _pair_leg(self, v, side, shares, got, now, seconds_left, fair, raw, books, price_age, book_age, v_cfg):
        cost, fee, top, depth = got
        p, r = fair.get("twap"), raw.get("twap")
        prob = None if p is None else (p if side == "up" else 1 - p)
        fill = Fill(side=side, fair_prob=prob, best_ask=_best(books[side], min), avg_price=cost / shares, shares=shares,
                    cost=cost, fee=fee, edge=None if prob is None else prob - (cost + fee) / shares, limit_price=top,
                    depth=depth)
        self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill, mode="lab", variant=v.key,
                           price_age=price_age, book_age=book_age,
                           fair_raw=None if r is None else (r if side == "up" else 1 - r), stake=v_cfg.stake_usd)

    def _new_window(self, start, ts, price):
        L = self.cfg.twap_seconds
        if self.window is not None:
            end = self.window + self.window_seconds
            close = self.vol.twap(end - L, end)
            if close is None:  # started again just before the close: the prices recorded before that
                close = self._recorded_twap(end - L, end)
            self.store.upsert_window(self.asset, self.minutes, self.window, close_twap=close,
                                     close_point=price if ts - end < 2 else None)
        else:
            self._backfill_closes(start)
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
        prior = self.store._x("SELECT mode, variant, side, shares, cost + fee FROM bets"
                              " WHERE asset = ? AND minutes = ? AND window_start = ? ORDER BY id", key)
        self.window = start
        self.main_entries, self.live_entries, self.lab_entries = {}, {}, {}
        self.main_done, self.live_done, self.lab_done = set(), set(), set()
        self.lab_votes = {}
        self.lab_orders = []  # (a round that closed before its order arrived: it never filled)
        for x in self.lab_pending:  # (the research's entries still open when the round ended: not filled)
            self._lab_record(x["order"], x["order"]["fill"], "rejected")
        self.lab_pending = []
        self.pair_legs = {}  # both-sides rule -> its first leg in this window: (side, shares, cost a share)
        for mode, variant, side, shares, paid in prior:
            if mode == "lab" and variant.startswith("pair"):
                leg = self.pair_legs.get(variant)
                if leg is None and shares:
                    self.pair_legs[variant] = (side, shares, paid / shares)
                elif leg and side != leg[0]:
                    self.lab_done.add(variant)
                continue
            if mode == "lab" and side in ("up", "down"):
                self.lab_votes.setdefault(variant, side)
            entries, done = {"paper": (self.main_entries, self.main_done), "live": (self.live_entries, self.live_done),
                             "lab": (self.lab_entries, self.lab_done)}.get(mode, (None, None))
            if entries is None:
                continue
            try:
                v = lab.Variant.from_key(variant)
            except ValueError:
                done.add(variant)  # (an old real-money row, recorded before rules were)
                continue
            self._entered(entries, done, v, 0.0)
        self.maker_orders = {}  # maker variant -> its resting order in this window
        self.sim_maker_orders = {}
        self.live_maker_want = {}
        for key in list(self.live_maker_have):  # (real orders of the round that ended: cancelled, what was taken recorded)
            self._live_maker_drop(key, time.time())
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
        self.lab_cfgs = {v.key: self._lab_cfg(v) for v in self.variants}
        self.main_cfg = (dataclasses.replace(self.market_cfg, min_edge=self.main_rule.min_edge,
                                             min_ratio=self.min_ratio, max_ratio=self.max_ratio,
                                             stake_usd=self.sim_stake)
                         if self.main_rule else self.market_cfg)

    def _lab_cfg(self, v):
        stake = lab.LOCK_STAKE if lab.is_lock(v) else lab.END_STAKE if lab.is_end(v) else self.market_cfg.stake_usd / lab.entries(v)  # (small: see lab.py)
        return dataclasses.replace(self.market_cfg, min_edge=v.min_edge, min_ratio=v.min_ratio, stake_usd=stake)

    def set_min_ratio(self, ratio, max_ratio=0.0):
        if (ratio, max_ratio) != (self.min_ratio, self.max_ratio):
            self.min_ratio, self.max_ratio = ratio, max_ratio
            self._set_market_cfg(self.market)

    def set_variants(self, variants):
        """Swap the lab variants (the search near winners adds and retires some)."""
        cfgs = {v.key: self._lab_cfg(v) for v in variants}
        self.lab_cfgs = {**self.lab_cfgs, **cfgs}  # a superset first, so a running pass never misses one
        self.variants = list(variants)
        self.lab_cfgs = cfgs

    def _use_market(self, start, market):
        self._set_market_cfg(market)
        self.market = market
        self.store.upsert_window(self.asset, self.minutes, start, found=1)


SETTLE_AFTER = 15  # seconds after a window closes before asking Polymarket for its winner


def settle(store, cfg, now, api=feeds, gate=None):
    """Fill in winners for finished windows (all markets, last 2 days) and settle their bets.
    gate(asset, minutes, start, slug, now): False holds a decided round back (it is published later,
    everything about it at once). Returns how many bets it settled."""
    shortest = min((cfg.window_minutes, *cfg.lab_minutes))
    settled = 0
    for asset, minutes, start, slug in store.unresolved_windows(now - shortest * 60 - SETTLE_AFTER, now - 2 * 86400):
        if start + minutes * 60 + SETTLE_AFTER > now:
            continue
        winner = api.market_winner(slug)
        if not winner or (gate is not None and not gate(asset, minutes, start, slug, now)):
            continue
        store.upsert_window(asset, minutes, start, winner=winner)
        for bet_id, side, shares, cost, fee in store.unsettled_bets(asset, minutes, start):
            store.settle_bet(bet_id, winner, settle_pnl(side, shares, cost, fee, winner))
            settled += 1
    return settled


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
        self.live.exact = self.lab_exact()["live"]
        self.pool = ThreadPoolExecutor(8, thread_name_prefix="work")
        # Real orders on threads of their own: never waiting behind the lookups and studies in the pool;
        # and a strategy's order to buy its own lane, never behind a resting order's place / move / cancel.
        self.live_pool = ThreadPoolExecutor(2, thread_name_prefix="live")
        self.taker_pool = ThreadPoolExecutor(1, thread_name_prefix="taker")
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
        # Why the simulation's rules didn't bet when they had an edge: (ts, asset, minutes, rule, reason).
        self.skips = collections.deque(maxlen=20000)
        self._skip_seen = set()
        for t in self.trackers:
            t.on_skip = self._on_skip
        if self.store.get_setting("live_main") is None:
            # Real money starts on what the simulation runs, then moves only when the owner pushes to it.
            self.store.set_setting("live_main", json.dumps(main))
            self.store.set_setting("live_changed_at", self.store.get_setting("main_changed_at", "0"))
        self._apply_live_main(self.live_main())
        self._apply_blend_use()
        self._refresh_variant_set()
        self.calibration = {"k": 1.0, "applied": False, "samples": 0, "fitted": None,
                            "min_samples": cfg.calib_min_samples}
        self._last_explore = 0.0
        self.market_blend = {"a": 1.0, "b": 0.0, "applied": False, "windows": 0, "samples": 0,
                             "min_windows": cfg.blend_min_windows}
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
        self.stocks = stocks.StockLab(cfg, self.store) if (with_lab and cfg.stocks_lab) else None
        if self.stocks:
            self.stocks.count_since = float(self.count_since().get("stocks") or 0)
        self._restore_candidates()
        # Since when the lab fills like a real order (lab_fill_delay): only its trades prove anything.
        self.real_fills_since = float(self.store.get_setting("lab_real_fills_since") or 0) or None
        if self.real_fills_since is None and cfg.lab_fill_delay > 0:
            self.real_fills_since = time.time()
            self.store.set_setting("lab_real_fills_since", str(self.real_fills_since))
        if self.stocks:
            self.stocks.live = self.live
            if streaming:
                self.stocks.dispatch = self._dispatch
        self._apply_sim()
        self.notifier = Notifier(cfg, self.store, self, env=env)
        self.health = Health(self)
        self._stop = threading.Event()
        self._learn_lock = threading.Lock()
        self._learn_queued = threading.Semaphore(1)
        self._lab_queued = threading.Semaphore(1)
        self.learning = {}
        self.stake_curve = {}
        self._mix_cache = {}  # period -> (since, when counted, the dashboard's strategy mix)
        # The market check (marketcheck.py): where the price itself is wrong. The last one is kept in the
        # database, so a restart (every deploy is one) shows it at once instead of an empty table.
        self.market_check = self._json_setting("market_check_last", None)
        # The streak study (streaks.py): do every strategy's wins and losses come in runs? Kept the same way.
        self.streaks = self._json_setting("streaks_last", None)
        # Predicting the move (predict.py): each decided round's readings kept for good, studied again often.
        self.predict = self._json_setting("predict_last", None)
        # Why orders don't go through, and whether 1-2 cents of room would pay (fills.py).
        self.fills = self._json_setting("fills_last", None)
        R = self.room_status()
        self.live.room = 0 if R.get("stopped") else int(R.get("cents") or 0)  # (the room as it was decided last)
        C = self.cheap_room_status()
        self.live.cheap_room = 0 if C.get("off") or C.get("stopped") else self.CHEAP_ROOM
        # Who moves first (lagwatch.py): Polymarket's prices after a jump of the price itself, and the
        # other coins'. Its events are kept in the database every few minutes.
        self.lagwatch = LagWatch(self._json_setting("lagwatch_events", []))
        # The race for the shares (race.py, lead.py, wallets.py: the panels of 29.10.2026), kept the same way.
        self.leads = lead.LeadWatch(self._json_setting("lead_events", []))
        self.race = self._json_setting("race_last", None)
        self.wallets = self._json_setting("wallets_last", None)
        self.hot_now = False
        for t in self.trackers:
            t.lagwatch = self.lagwatch
            t.hot = lambda: self.hot_now
        # The stocks area (equity.py): its own thread, and its own database file next to this one, so
        # nothing there ever writes to Polymarket's data.
        eq_store = EquityStore(os.path.join(os.path.dirname(cfg.db_path) or ".", "equity.sqlite"))
        move_out_of(self.store, eq_store)
        self.equities = Equities(eq_store)
        self.atlas = Atlas(self.equities)  # Atlas, the stocks bot (atlas.py): its simulation, brain, news and reviews
        self.notifier.atlas_mail = AtlasMail(self.notifier, self.atlas)  # its mails, on the notifier's loop
        self._trade_buf = []  # the markets' trades waiting to be written (_flush_trades)
        self.trade_side_check = {k: 0 for k in ("BUY", "BUY_above", "BUY_below", "SELL", "SELL_above", "SELL_below")}
        self.rr_research = {}
        self._migrate_money()
        self._owner_amounts_once()
        self._filters_into_strategies_once()
        self._panels_once()
        self._apply_money()

    def start(self):
        if self.streaming:
            # The order's thread waits for Python's lock at most 1 ms behind another thread, not 5; and
            # everything loaded so far (settings, markets, the lab's rules) is left out of the garbage
            # collector's full passes, which otherwise stop every thread to walk it all again.
            sys.setswitchinterval(0.001)
            gc.freeze()
            self.health.watch_gc()
        if self.streaming and self.cfg.equities:
            self.equities.start(self._stop)
            self.atlas.start(self._stop)
        threading.Thread(target=self._stream_thread if self.streaming else self._rest_loop, daemon=True,
                         name="engine").start()
        if self.streaming and self.cfg.exchange_feeds:
            threading.Thread(target=self._lead_thread, daemon=True, name="lead-feed").start()
        # Learning runs the moment trades are settled (_settle); these timers are only the fallback.
        for seconds, fn in ((10, self._settle), (60, self._lab_tick), (300, self._analysis_tick),
                            (300, self._daily), (5, self.notifier.tick), (30, self._refresh_wallet),
                            (60, self.health.check), (MARKET_CHECK_GAP, self._market_check_tick), (STREAKS_GAP, self._streaks_tick), (120, self._predict_tick), (300, self._ping), (self.AUTO_COPY_GAP, self._auto_copy_tick), (5, self._warm_live), (300, self._save_lagwatch),
                            (5, self._flush_trades), (600, self._refine_tick), (300, self._race_tick), (600, self._wallets_tick), (self.BRIDGE_EVERY, self._bridge_tick), (60, self._hot_tick),
                            (300, self._save_leads)):
            threading.Thread(target=self._every, args=(seconds, fn), daemon=True, name=fn.__name__).start()
        if self.stocks:
            for seconds, fn in ((300, self.stocks.refresh_markets), (60, self.stocks.settle)):
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

    # One moment for a round's result (the owner, 28.10.2026): its trades settle, their P&L counts, the
    # close mail goes and the wallet shown changes, all together. A round with real money waits until
    # Polymarket has decided it AND the wallet read after that has it resolved (its position redeemable
    # or gone): later, but never one before the other. PUBLISH_MAX_WAIT: never longer than this.
    PUBLISH_MAX_WAIT = 600

    def _live_waiting(self):
        return self.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live' AND status = 'filled' AND pnl IS NULL")[0][0]

    def _publish_gate(self, asset, minutes, start, slug, now):
        if not self.live.configured() or not self.store._x(
                "SELECT COUNT(*) FROM bets WHERE mode = 'live' AND status = 'filled' AND pnl IS NULL AND asset = ?"
                " AND minutes = ? AND window_start = ?", (asset, minutes, start))[0][0]:
            return True
        first = self._decided_at.setdefault((asset, minutes, start), now)
        if now - first >= self.PUBLISH_MAX_WAIT:
            log.warning("Round %s %s %s published without the wallet's word (%d s)", asset, minutes, start, now - first)
            return self._published(asset, minutes, start)
        # (without the market's name on the positions: one reading a minute after the decision)
        settle_by = first + (0 if self.live.wallet_slugs is not None else 60)
        if now < settle_by:
            return False
        W = self.live.wallet
        if not W or W.get("ts", 0) <= settle_by or W.get("error"):
            self.live.refresh_wallet()
            W = self.live.wallet
        if not W or W.get("ts", 0) <= settle_by or W.get("error"):
            return False
        P = self.live.wallet_slugs
        if P is not None and any(not redeemable and value > 0.005 for redeemable, value in P.get(slug, [])):
            return False  # (Polymarket's wallet doesn't have it resolved yet)
        return self._published(asset, minutes, start)

    def _published(self, asset, minutes, start):
        self._decided_at.pop((asset, minutes, start), None)
        self._wallet_publish = True
        return True

    def _settle(self):
        now = time.time()
        self.__dict__.setdefault("_decided_at", {})
        self._wallet_publish = False
        n = settle(self.store, self.cfg, now, api=self.api, gate=self._publish_gate)
        if self._wallet_publish:
            self.live.wallet_shown = self.live.wallet  # (the reading that had the round resolved)
        self.learning["settle"] = {"ts": now, "bets": n}
        if n:
            self.learning["last_settled"] = {"ts": now, "bets": n}
            self.learn_now("settle")
            try:
                self.notifier.tick()  # (the close mail now, with the card and the wallet)
            except Exception as e:
                log.warning("Close mail: %s", e)

    def _refresh_wallet(self):
        """Every half minute: the wallet read; shown at once only while no real trade waits for its result
        (one that does changes the wallet shown when it is published)."""
        self.live.refresh_wallet()
        if self.live.wallet and not self.live.wallet.get("error") and not self._live_waiting():
            self.live.wallet_shown = self.live.wallet
        elif self.live.wallet_shown is None:
            self.live.wallet_shown = self.live.wallet

    # Learning: everything the brain knows is recomputed from all the data the moment new trades
    # are settled (and on a timer as a fallback). One run at a time; a request while one runs
    # queues exactly one more, so the newest trades are always learned from.
    LEARN_GAP = 60  # seconds: settlements come every few seconds, a learning pass takes a few
    LAB_GAP = 180  # seconds: the lab's table (tens of seconds of the processor over millions of trades)

    def learn_now(self, reason):
        if not self._learn_queued.acquire(blocking=False):
            return  # one is already waiting and will see these trades too
        threading.Thread(target=self._learn, args=(reason,), daemon=True, name="learn").start()

    def _learn(self, reason):
        # At most one pass a minute: one waiting here sees every trade settled meanwhile.
        wait = self.LEARN_GAP - (time.time() - getattr(self, "_learned_at", 0.0)) if self.streaming else 0
        if wait > 0:
            time.sleep(wait)
        with self._learn_lock:
            self._learned_at = time.time()
            self._learn_queued.release()
            started = time.time()
            # (the lab's table is a pass over every lab trade: at most once in LAB_GAP, by the tick too)
            lab_due = not self.streaming or time.time() - getattr(self, "_lab_at", 0.0) >= self.LAB_GAP
            for fn in (self.refresh_analysis, self.refresh_lab) if lab_due else (self.refresh_analysis,):
                try:
                    fn()
                except Exception as e:
                    log.warning("%s: %s", fn.__name__, e)
            self.learning["learned"] = {"ts": time.time(), "took": time.time() - started, "reason": reason}

    def _on_trade(self, token, ts, price, size, taker):
        """A trade in one of the markets followed (anyone's), for the maker check: kept in memory,
        written every few seconds (marketcheck.maker_study)."""
        # Checking our reading of Polymarket's "side" (the taker's: who took the price): a taker's buy
        # prints at the offer, above the middle of the book. Counted, and shown with the check.
        book = self.books.books.get(token) if self.books else None
        if book and book["bids"] and book["asks"] and taker in ("BUY", "SELL"):
            mid = (max(book["bids"]) + min(book["asks"])) / 2
            c = self.trade_side_check
            c[taker] += 1
            c[taker + "_above"] += price > mid + 1e-9
            c[taker + "_below"] += price < mid - 1e-9
        for t in self.trackers:
            m = t.market
            if m and token in (m.get("up_token"), m.get("down_token")) and t.window is not None:
                side = "up" if token == m["up_token"] else "down"
                self._trade_buf.append((ts, t.asset, t.minutes, t.window, side, price, size, taker, t.window + t.window_seconds - ts))
                t.on_market_trade(side, price, size, taker, ts)
                return

    def _flush_trades(self):
        rows, self._trade_buf = self._trade_buf, []
        self.store.add_trades(rows)

    def _save_lagwatch(self):
        self.store.set_setting("lagwatch_events", json.dumps(self.lagwatch.events()))

    # ---------- the race for the shares (the owner's request and the panels of 29.10.2026) ----------
    RACE_EVERY = 1800  # seconds between two studies of A-D (in a process of their own)
    WALLETS_EVERY = 6 * 3600

    # Gate "k" (the streaks): the lab's trades that closed over the study's best lookback, winning (20 at least).
    HOT_LEAST = 20

    def _hot_tick(self, now=None):
        now = now or time.time()
        L = (self.streaks or {}).get("best") or self.STREAK_LOOKBACKS[1]
        n, pnl = self.store._x("SELECT COUNT(*), COALESCE(SUM(pnl), 0) FROM bets WHERE mode = 'lab' AND pnl IS NOT NULL"
                               " AND ts >= ? AND window_start + minutes * 60 >= ?", (now - L - 3600, now - L))[0]
        self.hot_now = n >= self.HOT_LEAST and pnl > 0
        return self.hot_now

    def _on_futures(self, source, asset, ts, price):
        """Experiment E's feed (its own thread): measured always; into our model's other view (gate "f") on the
        main loop, only while a rule reads it."""
        self.leads.add(source, asset, price)
        loop = getattr(self, "_loop", None)
        if loop is not None and any(t.uses_futures() for t in self.by_asset.get(asset, ())):
            loop.call_soon_threadsafe(self._fut_to_trackers, asset, ts, price)

    def _fut_to_trackers(self, asset, ts, price):
        for t in self.by_asset.get(asset, ()):
            t.on_fast(ts, price, "binance_fut")

    def _save_leads(self):
        self.store.set_setting("lead_events", json.dumps(self.leads.dump()[-lead.KEEP:]))

    def _race_arrive(self):
        """Our real delay from the signal to the order reaching the book (half the answer's way back), once real
        orders were timed; else the lab's."""
        Sp = (self.fills or {}).get("speed") or {}
        if (Sp.get("orders") or 0) >= self.FAST_ORDERS and Sp.get("decide") is not None and Sp.get("answer") is not None:
            return round(Sp["decide"] + Sp["answer"] / 2, 3)
        return self.cfg.lab_fill_delay

    def _race_tick(self, now=None, apart=True):
        now = now or time.time()
        if now - ((self.race or {}).get("ts") or 0) < self.RACE_EVERY or getattr(self, "_racing", False):
            return False
        self._racing = True
        try:
            args = (self.cfg.db_path, now, self._race_arrive(), self.cfg.fee_rate, self.cfg.fee_exponent)
            self.race = race.study_apart(*args) if apart else race.study(*args)
            self.store.set_setting("race_last", json.dumps(self.race))
            self.__dict__.pop("_hub_cache", None)
            self._bridge_tick(now)  # (a finding into the lab at once)
        except Exception as e:
            log.warning("Race study: %s", e)
            self.race = dict(self.race or {}, ts=now, error=f"{type(e).__name__}: {e}"[:200])
        finally:
            self._racing = False
        return True

    def _wallets_tick(self, now=None, **kw):
        now = now or time.time()
        if now - ((self.wallets or {}).get("ts") or 0) < self.WALLETS_EVERY:
            return False
        cache = self._json_setting("wallet_conditions", {})
        db = connect(self.cfg.db_path)
        try:
            self.wallets = wallets.study(db, cache, **kw)
        except Exception as e:
            log.warning("Wallet study: %s", e)
            self.wallets = dict(self.wallets or {}, ts=now, errors=[f"{type(e).__name__}: {e}"[:200]])
        finally:
            db.close()
        self.store.set_setting("wallet_conditions", wallets.dumps_cache(cache))
        self.store.set_setting("wallets_last", json.dumps(self.wallets))
        self.__dict__.pop("_hub_cache", None)
        return True

    # The edge hub (the control center's top): the tests looking for an edge, each in plain words,
    # how far it is from an answer, and what it says now. Counted at most every MIX_GAP seconds.
    HUB_DAYS = 3  # the maker check's answer: this many days of rounds
    HUB_EVENTS = 50  # "who moves first": this many events in a group

    def hub_view(self):
        hit = self.__dict__.get("_hub_cache")
        if hit and time.time() - hit[0] < self.MIX_GAP:
            return hit[1]
        tracks = []
        # 1. Who moves first (read only, always running).
        W = self.lagwatch_view()
        G = W["groups"]
        found = [g for g in G if g["edge_found"]]
        most = max((g["events"] for g in G), default=0)
        best = max(G, key=lambda g: g["mark_mid_low"] if g["mark_mid_low"] is not None else -9, default=None)
        fast_found = [g for g in G if g.get("fast_edge")]
        fast_most = max((g.get("fast_n") or 0 for g in G), default=0)
        fast_best = max(G, key=lambda g: g["mark_fast_low"] if g.get("mark_fast_low") is not None else -9, default=None)
        tracks.append({"key": "speed", "running": True, "found": len(found), "progress": min(1.0, most / self.HUB_EVENTS),
                       "events": W["events"], "most": most, "need": self.HUB_EVENTS,
                       "fast_delay": W["settings"].get("fast_delay"), "fast_found": len(fast_found), "fast_most": fast_most,
                       "fast_best": fast_best and fast_best.get("fast_n") and {k: fast_best.get(k) for k in
                                                                              ("kind", "leader", "follower", "minutes", "fast_n", "mark_fast", "mark_fast_low")},
                       "best": best and {k: best[k] for k in ("kind", "leader", "follower", "minutes", "events", "mark_mid", "mark_mid_low")},
                       # (at our own speed, once measured, the question is open again until it has its events too)
                       "state": "found" if found or fast_found else "collecting" if most < self.HUB_EVENTS
                       or (W["settings"].get("fast_delay") is not None and fast_most < self.HUB_EVENTS) else "nothing"})
        # 2. The maker check (everyone's trades, as if we were the maker).
        K = (self.market_check or {}).get("maker") or {}
        kg = K.get("groups") or []
        kfound = [g for g in kg if g.get("edge_found")]
        days = K.get("days") or 0
        tracks.append({"key": "maker_check", "running": True, "found": len(kfound), "days": days, "need": self.HUB_DAYS,
                       "progress": min(1.0, days / self.HUB_DAYS), "rounds": K.get("rounds") or 0, "ts": (self.market_check or {}).get("ts"),
                       "best": kfound[0] if kfound else None,
                       "state": ("found" if kfound and days >= self.HUB_DAYS else "promising" if kfound else
                                 "collecting" if days < self.HUB_DAYS else "nothing")})
        # 3, 4. The lab's makers built on it: the resting order on the market's leader, and the one on
        # the model's price.
        board = (self.lab or {}).get("board") or []
        times = self.lab_times()
        order = {"proven": 0, "candidate": 1, "promising": 2, "collecting": 3, "failed": 4, "losing": 5}
        zones = [lab.Variant.from_key(k) for k in self._json_setting("lab_zones", [])]
        for key, test in (("zone_lab", lab.is_zone), ("fav_lab", lab.is_fav), ("twapm_lab", lambda v: v.resolution == "twapm")):
            grid = [v for m in (5, 15) for v in lab.variants_for(m) if test(v)] + [v for v in zones if test(v)]
            groups = sorted({lab.time_group(v.key) for v in grid})
            rows = []
            for b in board:
                try:
                    if test(lab.Variant.from_key(b["variant"])):
                        rows.append(b)
                except (ValueError, IndexError):
                    continue
            stages = {st: sum(1 for b in rows if b["status"] == st) for st in order}
            top = min(rows, key=lambda b: (order[b["status"]], -(b["roi_pessimistic"] if b["roi_pessimistic"] is not None else -9)), default=None)
            bets = sum(b["bets"] for b in rows)
            running = times is None or any(g in times for g in groups)
            progress, done = self._family_progress(rows)
            state = ("off" if not running else "found" if stages["proven"] else "promising" if stages["candidate"] else
                     "nothing" if rows and done == len(rows) else "collecting")
            tracks.append({"key": key, "running": running, "groups": groups, "rules": len(rows), "bets": bets, "stages": stages,
                           "progress": progress, "decided": done,
                           "best": top and {k: top[k] for k in ("asset", "minutes", "variant", "status", "bets", "roi", "roi_pessimistic", "oos_bets")},
                           "state": state})
        # 5. The owner's rule (26.10): the usual rule with its own filters (70-80c, 85% chance, the last 2 minutes).
        own = []
        for b in board:
            try:
                if lab.Variant.from_key(b["variant"]).min_chance:
                    own.append(b)
            except (ValueError, IndexError):
                continue
        ostages = {st: sum(1 for b in own if b["status"] == st) for st in order}
        obets = sum(b["bets"] for b in own)
        otop = min(own, key=lambda b: (order[b["status"]], -(b["roi_pessimistic"] if b["roi_pessimistic"] is not None else -9)), default=None)
        oprogress, odone = self._family_progress(own)
        tracks.append({"key": "owner_rule", "running": True, "rules": len(own), "bets": obets, "stages": ostages,
                       "progress": oprogress, "decided": odone,
                       "best": otop and {k: otop[k] for k in ("asset", "minutes", "variant", "status", "bets", "roi", "roi_pessimistic", "oos_bets")},
                       "state": ("found" if ostages["proven"] else "promising" if ostages["candidate"] else
                                 "nothing" if own and odone == len(own) else "collecting")})
        # 6. The end-of-round favourite (27.10): twelve rules, each over every coin together.
        E = (self.lab or {}).get("end_pooled") or []
        verdicts = {v: sum(1 for e in E if e["verdict"] == v) for v in ("proven", "stopped", "undecided", "collecting")}
        # (an answer for the family once every rule has its verdict: proven, or stopped under break-even)
        most = max((e["trades"] for e in E), default=0)
        etop = max(E, key=lambda e: ((e["lower"] or 0) - (e["break_even"] or 1)) if e["trades"] else -9, default=None)
        # (how far from an answer: a rule with a verdict counts whole, one still undecided by its share of
        # the way to its next checkpoint)
        eprog = sum(1.0 if e["verdict"] in ("proven", "stopped") else min(0.99, e["trades"] / (e.get("next") or lab.END_CHECKPOINTS[0]))
                    for e in E) / len(E) if E else 0.0
        tracks.append({"key": "end_fav", "running": True, "rules": len(E), "trades": sum(e["trades"] for e in E), "most": most,
                       "need": lab.END_CHECKPOINTS[0], "verdicts": verdicts, "progress": eprog,
                       "best": etop and {k: etop[k] for k in ("variant", "rule", "trades", "win_rate", "break_even", "lower", "verdict")},
                       "state": ("found" if verdicts["proven"] else "nothing" if E and verdicts["stopped"] == len(E)
                                 else "promising" if etop and etop["trades"] and (etop["win_rate"] or 0) > (etop["break_even"] or 1)
                                 and most >= lab.END_CHECKPOINTS[0] else "collecting")})
        # 7. Streaks (27.10): do every strategy's wins and losses come in runs? (streaks.py)
        S = self.streaks or {}
        row = next((r for r in S.get("rows") or [] if r["lookback"] == S.get("best")), None)
        tracks.append({"key": "streaks", "running": True, "state": S.get("state") or "collecting",
                       "progress": S.get("progress") or 0.0, "trades": S.get("trades") or 0,
                       "days": (S.get("span") or 0) / 86400, "need": self.STREAK_NEED / 86400, "best": row, "ts": S.get("ts")})
        # 8. Predicting the move (27.10): the coin's own move and who is buying, against the market's price.
        Pr = self.predict or {}
        tracks.append({"key": "predict", "running": True, "state": Pr.get("state") or "collecting",
                       "progress": Pr.get("progress") or 0.0, "rounds": Pr.get("rounds") or 0, "days": Pr.get("days") or 0.0,
                       "need": predict.NEED_DAYS, "ts": Pr.get("ts")})
        # 9, 10. Orders that don't go through (28.10): why, and whether 1-2 cents of room pays (fills.py).
        F = self.fills or {}
        W = F.get("why") or {}
        tracks.append({"key": "fill_why", "running": True, "state": W.get("state") or "collecting", "why": W,
                       "progress": F.get("why_progress") or 0.0})
        Sp = F.get("speed") or {}
        tracks.append({"key": "speed_tech", "running": True, "state": Sp.get("state") or "collecting", "speed": Sp,
                       "progress": Sp.get("progress") or 0.0, "gc": self.health.gc_view()})
        tracks.append({"key": "fill_room", "running": True, "state": F.get("room_state") or "collecting",
                       "room": F.get("room") or {}, "best": F.get("best"), "need": F.get("need") or fills.NEED,
                       "progress": F.get("room_progress") or 0.0,
                       "live_room": dict(self.room_status(), on=self.live.room, stop_n=self.ROOM_STOP_N),
                       "cheap": F.get("cheap") or {}, "cheap_below": self.live.CHEAP_BELOW,
                       "live_cheap": dict(self.cheap_room_status(), on=self.live.cheap_room, max=self.CHEAP_ROOM,
                                          stop_n=self.CHEAP_STOP_N)})
        # 11-16. The race for the shares (the panels of 29.10.2026): A-D (race.py), E (lead.py), F (wallets.py).
        Rc = self.race or {}
        for key in ("racing", "intent", "shading", "standing"):
            X = Rc.get(key) or {}
            tracks.append({"key": "race_" + key, "running": True, "state": X.get("state") or "collecting", "data": X,
                           "progress": X.get("progress") or 0.0, "ts": Rc.get("ts"), "error": X.get("error") or Rc.get("error"),
                           "need": race.NEED, "days": race.DAYS})
        Lv = self.leads.view()
        tracks.append({"key": "race_lead", "running": True, "state": Lv["state"], "data": Lv, "progress": Lv["progress"],
                       "connected": bool(getattr(getattr(self, "lead_stream", None), "connected", False))})
        Wl = self.wallets or {}
        tracks.append({"key": "race_wallets", "running": True, "state": Wl.get("state") or "collecting", "data": Wl,
                       "progress": Wl.get("progress") or 0.0})
        Bd = self.lab_bridge()
        for t in tracks:  # (what each finding added to the lab, by itself)
            t["bridge"] = [{"code": e["code"], "n": len(e["keys"]), "since": e["since"]} for e in Bd.values() if e["track"] == t["key"]]
        out = {"tracks": tracks, "ts": time.time()}
        self._hub_cache = (time.time(), out)
        return out

    # A lab family's way to an answer: a rule is decided once proven or failed, or once it passed two
    # checkpoints (2 x FIRST_CHECKPOINT real fills) without becoming a candidate (it had its chances);
    # one still going counts by its share of that way. Returns (progress 0-1, rules decided).
    def _family_progress(self, rows):
        if not rows:
            return 0.0, 0
        enough = 2 * lab.FIRST_CHECKPOINT
        done = sum(1 for b in rows if b["status"] in ("proven", "failed") or
                   (b["status"] in ("losing", "promising", "collecting") and b["bets"] >= enough))
        part = sum(1.0 if b["status"] in ("proven", "failed") or (b["status"] in ("losing", "promising", "collecting") and b["bets"] >= enough)
                   else min(0.99, b["bets"] / enough) for b in rows)
        return part / len(rows), done

    MAX_ZONES = 16

    def _maker_zones(self):
        """Each group where the maker check found an edge becomes a maker of its own in the lab (a zone):
        takers selling a side at p lose to whoever buys it at p; takers buying a side at p lose to
        whoever sells it, the same as buying the other side at 1 - p. Kept once found (newest first, at
        most MAX_ZONES), so the lab goes on measuring them."""
        groups = ((self.market_check or {}).get("maker") or {}).get("groups") or []
        found = []
        for g in groups:
            if not g.get("edge_found"):
                continue
            lo, hi = g["price_lo"], g["price_hi"]
            if g["taker"] == "BUY":
                lo, hi = round(1 - hi, 2), round(1 - lo, 2)
            if hi <= 0.02 or lo >= 0.99:
                continue
            found.append(lab.zone_key(lo, hi, g["time_lo"], g["time_hi"]))
        have = self._json_setting("lab_zones", [])
        zones = list(dict.fromkeys(found + have))[:self.MAX_ZONES]
        if zones != have:
            self.store.set_setting("lab_zones", json.dumps(zones))
            log.warning("Maker zones from the maker check: %s", ", ".join(zones))
            if getattr(self, "current_variants", None) is not None:
                self._refresh_lab_variants()

    def lagwatch_view(self):
        hit = self.__dict__.get("_lag_cache")
        if hit and time.time() - hit[0] < self.MIX_GAP:
            return hit[1]
        out = self.lagwatch.summary()
        self._lag_cache = (time.time(), out)
        return out

    def _market_check_tick(self):
        """Every quarter of an hour (the first a minute after a start, when the streams are up)."""
        if not getattr(self, "_market_check_ran", False):
            self._market_check_ran = True
            if self._stop.wait(60):  # (the first a minute after a start, when the streams are up)
                return
        self.refresh_market_check()

    def refresh_market_check(self):
        db = connect(self.cfg.db_path)
        try:
            started = time.time()
            out = dict(marketcheck.study(db, self.cfg), maker=marketcheck.maker_study(db, self.cfg))
            self.market_check = dict(out, ts=time.time(), took=time.time() - started)
            self._maker_zones()
        finally:
            db.close()
        self.store.set_setting("market_check_last", json.dumps(self.market_check))

    # The streak study: every lab trade of the last STREAK_DAYS days, every family together. The signal is
    # what the trades that had already ended said in the last half hour to four hours (set before any
    # data); a week of trades before a verdict.
    STREAK_DAYS = 30
    STREAK_LOOKBACKS = (1800, 3600, 7200, 14400)
    STREAK_NEED = 7 * 86400

    def _streaks_tick(self):
        if not getattr(self, "_streaks_ran", False):
            self._streaks_ran = True
            if self._stop.wait(120):  # (not on top of a start's other work)
                return
        self.refresh_streaks()

    def refresh_streaks(self, now=None):
        now = now or time.time()
        db = connect(self.cfg.db_path)
        try:
            since = (now - self.STREAK_DAYS * 86400,)
            lab_ = " FROM bets WHERE mode = 'lab' AND pnl IS NOT NULL AND cost + fee > 0 AND shares > 0 AND ts >= ?"
            # (summed in the database: the lab makes tens of thousands of trades a day; a minute's bucket
            # is finer than the study's, and a round's trades end with it: its last second, plus the few
            # the result takes to arrive)
            started = db.execute("SELECT CAST(ts / 60 AS INTEGER) * 60, SUM(pnl), SUM(cost + fee), COUNT(*)" + lab_
                                 + " GROUP BY 1", since).fetchall()
            ended = db.execute("SELECT window_start + minutes * 60 + 5, SUM(pnl), SUM(cost + fee), COUNT(*)" + lab_
                               + " GROUP BY 1", since).fetchall()
        finally:
            db.close()
        out = streaks.study([tuple(r) for r in started], [tuple(r) for r in ended], self.STREAK_LOOKBACKS, 600,
                            self.STREAK_NEED, min_recent=20)
        self.streaks = dict(out, ts=now, unit="seconds")
        self.store.set_setting("streaks_last", json.dumps(self.streaks))
        self.__dict__.pop("_hub_cache", None)

    FAST_ORDERS = 20  # real orders timed before "who moves first" is priced at our speed too
    PREDICT_GAP = 900  # seconds between two studies (the readings are taken every two minutes)

    def _predict_tick(self, now=None):
        now = now or time.time()
        db = connect(self.cfg.db_path)
        try:
            predict.record(db)
            if not self.predict or now - (self.predict.get("ts") or 0) >= self.PREDICT_GAP:
                self.predict = dict(predict.study(db, self.cfg.fee_rate, self.cfg.fee_exponent, since=now - 90 * 86400), ts=now)
                self.store.set_setting("predict_last", json.dumps(self.predict))
                self.fills = dict(fills.study(db, since=now - 30 * 86400), ts=now,
                                  speed=fills.speed(db, since=now - 14 * 86400, our_delay=self.cfg.lab_fill_delay))
                Sp = self.fills["speed"]
                # (who moves first, priced also at our own speed once real orders were timed: the signal to
                # the order reaching the book, about half its way there and back)
                if Sp["orders"] >= self.FAST_ORDERS and Sp["decide"] is not None and Sp["answer"] is not None:
                    self.lagwatch.fast_delay = round(Sp["decide"] + Sp["answer"] / 2, 2)
                self.store.set_setting("fills_last", json.dumps(self.fills))
                self._room_tick(now)
                self.__dict__.pop("_hub_cache", None)
        finally:
            db.close()

    # The room (the owner's decision of 28.10.2026): real money's strategies may pay 1-2 cents over the price
    # their strategy saw (live.py: never past the rule's edge or the risk/reward filter), only once the room
    # study (fills.py) found it pays on the simulation's own refused orders. 1 cent first; 2 once real
    # money's own trades that paid more made money and the study finds 2 cents pays too. If real money's
    # trades that paid more lose, the room stops for good (until the owner starts it again).
    ROOM_STOP_N = 20  # real trades that paid over the price seen, settled, before they can stop the room
    ROOM_UP_N = 30  # ... before 2 cents

    def room_status(self):
        return self._json_setting("live_room", {})

    def _room_live(self, since, cheap=None):
        """Real money's trades that paid over the price their strategy saw, settled: (how many, profit).
        cheap: True, only the cheap sides' (the cheap room's); False, all but them (while the cheap room is on,
        they are its to judge)."""
        part = {True: " AND seen < ?", False: " AND seen >= ?"}.get(cheap, "")
        n, pnl = self.store._x("SELECT COUNT(*), COALESCE(SUM(pnl), 0) FROM bets WHERE mode = 'live' AND status = 'filled'"
                               " AND pnl IS NOT NULL AND seen IS NOT NULL AND avg_price > seen + 1e-9 AND ts >= ?" + part,
                               (since,) + ((self.live.CHEAP_BELOW,) if part else ()))[0]
        return int(n), float(pnl)

    def _room_tick(self, now=None):
        now = now or time.time()
        R, F = self.room_status(), self.fills or {}
        before = dict(R)
        cents = 0
        if not R.get("stopped") and F.get("room_state") == "found":
            since = R.setdefault("since", now)
            n, pnl = self._room_live(since, False if self.live.cheap_room else None)
            R.update(n=n, pnl=pnl)
            if n >= self.ROOM_STOP_N and pnl < 0:
                R["stopped"] = {"ts": now, "n": n, "pnl": pnl}
                log.warning("Room stopped: %d real trades that paid more lost $%.2f", n, -pnl)
            else:
                two = (F.get("room") or {}).get("paper2") or {}
                up = n >= self.ROOM_UP_N and pnl > 0 and (two.get("z") or 0) >= fills.Z and (two.get("roi") or 0) > 0
                cents = 2 if R.get("cents") == 2 or up else 1
        R["cents"] = cents
        if R != before:
            self.store.set_setting("live_room", json.dumps(R))
        self.live.room = cents
        self._cheap_room_tick(now)
        return R

    # The cheap room (the owner's decision of 29.10.2026: "it may make less, as long as it makes money"): on a
    # side under 20 cents, whose win pays 4x and more, real money may pay up to 5 cents over the price seen,
    # with the same bounds as the room (live.py: never past the rule's edge or the risk/reward filter), on
    # from the start. It stops by itself when real money's own trades that paid more there lose after
    # CHEAP_STOP_N of them (a cheap side wins one time in four or five: fewer would be noise), or when the
    # study (fills.py) has enough of the simulation's refused cheap orders and 5 cents more lost on them.
    CHEAP_ROOM = 5
    CHEAP_STOP_N = 40

    def cheap_room_status(self):
        return self._json_setting("live_cheap_room", {})

    def _cheap_room_tick(self, now=None):
        now = now or time.time()
        C = self.cheap_room_status()
        before = dict(C)
        cents = 0
        if not C.get("off") and not C.get("stopped"):
            since = C.setdefault("since", now)
            n, pnl = self._room_live(since, True)
            C.update(n=n, pnl=pnl)
            S = (((self.fills or {}).get("cheap") or {}).get("rooms") or {}).get(str(self.CHEAP_ROOM)) or {}
            if n >= self.CHEAP_STOP_N and pnl < 0:
                C["stopped"] = {"ts": now, "why": "live", "n": n, "pnl": pnl}
            elif S.get("settled", 0) >= fills.NEED and (S.get("roi") or 0) <= 0:
                C["stopped"] = {"ts": now, "why": "study", "n": S["settled"], "pnl": S.get("pnl") or 0.0}
            else:
                cents = self.CHEAP_ROOM
            if C.get("stopped"):
                log.warning("Cheap room stopped: %s", C["stopped"])
        C["cents"] = cents
        if C != before:
            self.store.set_setting("live_cheap_room", json.dumps(C))
        self.live.cheap_room = cents
        return C

    def set_cheap_room(self, on):
        """The owner turns the cheap room on (again: its count starts over) or off."""
        self.store.set_setting("live_cheap_room", json.dumps({} if on else {"off": True}))
        self.__dict__.pop("_hub_cache", None)
        return self._cheap_room_tick()

    def restart_room(self):
        """The owner starts a stopped room again: its count starts over."""
        self.store.set_setting("live_room", json.dumps({}))
        self.__dict__.pop("_hub_cache", None)
        return self._room_tick()

    PING_HOST = "clob.polymarket.com"  # Polymarket's order book: where a real order goes

    def _ping(self):
        """The network's round trip to Polymarket's order book: a connection's opening (one trip there and
        back), the best of three, every few minutes (fills.speed)."""
        import socket
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            try:
                socket.create_connection((self.PING_HOST, 443), timeout=5).close()
            except OSError:
                continue
            ms = (time.perf_counter() - t0) * 1000
            best = ms if best is None else min(best, ms)
        if best is not None:
            self.store.add_ping(time.time(), self.PING_HOST, round(best, 1))

    # ---------- consensus (lab.CONS_RULES, the owner 29.10.2026): many of the lab's rules agreeing ----------
    GOOD = ("promising", "candidate", "proven")

    def _set_good(self, board):
        good = {}
        for b in board:
            if b["status"] in self.GOOD and not b["variant"].startswith("cons"):
                good.setdefault((b["asset"], b["minutes"]), set()).add(b["variant"])
        for t in self.trackers:
            t.good = good.get((t.asset, t.minutes), set())

    def consensus_view(self):
        """The consensus rules on the lab's board, beside the good rules' own pooled result (the bar to beat)."""
        board = (self.lab or {}).get("board") or []
        rows = [{k: b.get(k) for k in ("asset", "minutes", "variant", "rule", "bets", "wins", "pnl", "paid", "roi", "roi_pessimistic", "status", "next")}
                for b in board if b["variant"].startswith("cons")]
        good = [b for b in board if b["status"] in self.GOOD and not b["variant"].startswith("cons")]
        paid = sum(b.get("paid") or 0 for b in good)
        cons_paid = sum(r.get("paid") or 0 for r in rows)
        return {"rows": rows, "good_n": len(good), "good_bets": sum(b["bets"] for b in good),
                "good_roi": sum(b.get("pnl") or 0 for b in good) / paid if paid else None,
                "cons_roi": sum(r.get("pnl") or 0 for r in rows) / cons_paid if cons_paid else None,
                "cons_bets": sum(r.get("bets") or 0 for r in rows)}

    # ---------- the workshop (refine.py, the owner 29.10.2026): a winning strategy cloned into the lab ----------
    REFINE_STUDY_EVERY = 6 * 3600  # the replay again (new rounds recorded since)

    def refine_list(self):
        return self._json_setting("lab_refine", [])

    def refine_start(self, asset, minutes, key):
        """The strategy `key` on (asset, minutes), cloned into the lab with its versions (refine.candidates),
        and the replay study started. Where it runs (the simulation, real money) nothing changes."""
        minutes = int(minutes)
        lab.Variant.from_key(key)  # (a key the lab can read, or ValueError)
        if not any(t.asset == asset and t.minutes == minutes for t in self.trackers):
            raise ValueError("אין שוק כזה")
        wid = _key(asset, minutes, key)
        ws = [w for w in self.refine_list() if w["id"] != wid]
        ws.append({"id": wid, "asset": asset, "minutes": minutes, "base": key, "started": time.time(),
                   "variants": refine.candidates(key, minutes)})
        self.store.set_setting("lab_refine", json.dumps(ws))
        log.warning("Workshop: %s cloned into the lab with %d versions", wid, len(ws[-1]["variants"]))
        if getattr(self, "current_variants", None) is not None:
            self._refresh_lab_variants()
        threading.Thread(target=self.refine_study, args=(wid,), daemon=True, name="refine").start()
        return ws[-1]

    def refine_stop(self, wid):
        ws = [w for w in self.refine_list() if w["id"] != wid]
        self.store.set_setting("lab_refine", json.dumps(ws))
        self.store.set_setting(f"refine_study:{wid}", None)
        if getattr(self, "current_variants", None) is not None:
            self._refresh_lab_variants()
        return {"ok": True}

    def refine_study(self, wid):
        """The replay of the clone and its versions on the recorded seconds, and where real money missed."""
        w = next((x for x in self.refine_list() if x["id"] == wid), None)
        if w is None:
            return None
        args = (self.cfg.db_path, w["asset"], w["minutes"], w["base"], [c["key"] for c in w["variants"]],
                self.cfg.fee_rate, self.cfg.fee_exponent, time.time())
        out = None
        if self.streaming:  # (on the server: a process of its own, behind the trading: it reads two weeks of seconds)
            try:
                import concurrent.futures
                import multiprocessing
                with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                                            initializer=os.nice, initargs=(15,)) as proc:
                    out = proc.submit(refine.study, *args).result(timeout=1800)
            except Exception as e:
                log.warning("Workshop study in its own process: %s", e)
                return None
        if out is None:
            out = refine.study(*args)
        self.store.set_setting(f"refine_study:{wid}", json.dumps(out))
        log.warning("Workshop: %s replayed on %d rounds", wid, out["rounds"])
        return out

    # ---------- what the research learns, into the lab by itself (the owner, 29.10.2026) ----------
    # "Whatever the research finds that helps or improves goes at once into a new, cloned, better strategy."
    # A finding of the research center that changes how a strategy enters becomes a clone of every taker rule
    # running in the simulation or with real money, with that entry (lab.Variant.entry), in the lab: from
    # there the same stages as every strategy, and to real money only on the owner's word. The findings that
    # don't change a strategy (a score, a measurement, the wallets) say so on their card.
    BRIDGE_EVERY = 600

    def bridge_findings(self):
        """The research's findings that change how a strategy enters: [(the card, the entry)]."""
        out = []
        R = self.race or {}
        I = R.get("intent") or {}
        if I.get("state") == "found" and I.get("best"):
            out.append(("race_intent", f"i{int(I['best'])}"))
        for b in (R.get("shading") or {}).get("bands") or []:
            if b.get("state") == "found" and b.get("best") and ("race_shading", f"m{int(b['best'])}") not in out:
                out.append(("race_shading", f"m{int(b['best'])}"))
        D = R.get("standing") or {}
        if D.get("state") == "found" and D.get("best") is not None:
            out.append(("race_standing", f"s{round(float(D['best']) * 100)}"))
        F = self.fills or {}
        if F.get("room_state") == "found" and F.get("best"):
            out.append(("fill_room", f"m{int(F['best'])}"))
        # The gates (lab.Variant.gate, "g..."): our model fed by the futures, the streaks, the prediction's signal.
        if self.leads.view().get("state") == "found":
            out.append(("race_lead", "gf"))
        if (self.streaks or {}).get("state") == "found":
            out.append(("streaks", "gk"))
        P = self.predict or {}
        if P.get("state") == "found" and (P.get("best") or {}).get("feature"):
            out.append(("predict", "gp" + P["best"]["feature"]))
        return out

    def _bridge_bases(self):
        """The strategies a finding is cloned from: every taker rule running in the simulation or with real money."""
        out = {}
        for mode in ("sim", "live"):
            for m in self.strategies_for(mode):
                v = _rule(m["variant"])
                if v and v.resolution in lab.TAKERS and not v.entry and not v.gate:
                    out[(m["asset"], int(m["minutes"]), m["variant"])] = v
        return out

    def lab_bridge(self):
        return self._json_setting("lab_bridge", {})

    def _bridge_keys(self, asset, minutes):
        return [k for e in self.lab_bridge().values() for a, m, k in e["keys"] if (a, int(m)) == (asset, minutes)]

    def _bridge_tick(self, now=None):
        now = now or time.time()
        B = self.lab_bridge()
        before = json.dumps(B, sort_keys=True)
        bases = self._bridge_bases()
        for track, code in self.bridge_findings():
            e = B.setdefault(f"{track}|{code}", {"track": track, "code": code, "since": now, "keys": []})
            have = {tuple(k) for k in e["keys"]}
            for (a, m, _), v in bases.items():
                change = {"gate": code[1:]} if code.startswith("g") else {"entry": code}
                clone = dataclasses.replace(v, filter_ratio=0.0, stake=0.0, **change).key
                if (a, m, clone) not in have:
                    e["keys"].append([a, m, clone])
                    have.add((a, m, clone))
        if json.dumps(B, sort_keys=True) != before:
            self.store.set_setting("lab_bridge", json.dumps(B))
            log.warning("Research -> lab: %s", {k: len(e["keys"]) for k, e in B.items()})
            self.__dict__.pop("_hub_cache", None)
            if getattr(self, "current_variants", None) is not None:
                self._refresh_lab_variants()
        return B

    def _refine_keys(self, asset, minutes):
        return [k for w in self.refine_list() if (w["asset"], w["minutes"]) == (asset, minutes)
                for k in [w["base"]] + [c["key"] for c in w["variants"]]]

    def _refine_seed(self):
        """Once (the owner, 29.10.2026): Bitcoin's 5-minute strategy with the 12-point edge that runs with real
        money, cloned into the workshop."""
        if self.store.get_setting("refine_seeded"):
            return
        for m in self.live_main():
            if m["asset"] == "btc" and int(m["minutes"]) == 5 and m["variant"].startswith("twap|0.12|"):
                self.refine_start("btc", 5, m["variant"])
                self.store.set_setting("refine_seeded", "1")
                break

    def _refine_tick(self, now=None):
        now = now or time.time()
        try:
            self._refine_seed()
        except Exception as e:
            log.warning("Workshop seed: %s", e)
        for w in self.refine_list():
            st = self._json_setting(f"refine_study:{w['id']}", None) or {}
            if now - (st.get("ts") or 0) >= self.REFINE_STUDY_EVERY and not getattr(self, "_refining", False):
                self._refining = True
                try:
                    self.refine_study(w["id"])
                except Exception as e:
                    log.warning("Workshop study %s: %s", w["id"], e)
                finally:
                    self._refining = False

    def refine_view(self):
        board = {(b["asset"], b["minutes"], b["variant"]): b for b in ((self.lab or {}).get("board") or [])}
        out = []
        for w in self.refine_list():
            st = self._json_setting(f"refine_study:{w['id']}", None) or {}
            J = st.get("judge") or {}
            rows = []
            for c in [{"key": w["base"], "goal": "base", "what": "השכפול: אותה אסטרטגיה בדיוק"}] + w["variants"]:
                b = board.get((w["asset"], w["minutes"], c["key"])) or {}
                rows.append({**c, "rule": lab.describe(c["key"]),
                             "lab": {k: b.get(k) for k in ("bets", "wins", "pnl", "roi", "roi_pessimistic", "status", "next")} if b else None,
                             "replay": J.get(c["key"])})
            out.append({"id": w["id"], "asset": w["asset"], "minutes": w["minutes"], "base": w["base"], "started": w["started"],
                        "rule": lab.describe(w["base"]), "rows": rows,
                        "study": {k: st.get(k) for k in ("ts", "rounds", "first", "last", "fidelity", "misses", "since")}})
        return out

    # The research center's finished cards the owner put away (an X): only hidden, their studies go on.
    def research_hidden(self):
        return self._json_setting("research_hidden", [])

    def hide_research(self, key, hidden=True):
        if not isinstance(key, str) or not 0 < len(key) <= 80:
            raise ValueError("bad request")
        keys = [k for k in self.research_hidden() if k != key] + ([key] if hidden else [])
        self.store.set_setting("research_hidden", json.dumps(keys))
        return keys

    def _analysis_tick(self):
        with self._learn_lock:
            self.refresh_analysis()

    def _lab_tick(self):
        with self._learn_lock:
            if time.time() - getattr(self, "_lab_at", 0.0) >= self.LAB_GAP:
                self.refresh_lab()

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

    EXCHANGE_GAP = 0.02  # seconds: the fastest an exchange's price is passed on for one coin

    def _dispatch(self, fn, *args):
        pool = self.taker_pool if getattr(fn, "__name__", "") == "_live_orders" else self.live_pool
        pool.submit(self._guard, "live", fn, *args)

    def _warm_live(self):
        """Every few seconds: the markets real money may order on (its strategies', and the simulation's,
        which the owner can copy), their details loaded ahead (LiveTrader.warm)."""
        if not self.live.armed():
            return
        keys = {(m["asset"], m["minutes"]) for m in self.live_main() + self.main_strategies()}
        tokens = [tok for t in self.trackers if (t.asset, t.minutes) in keys for m in (t.market, t.next_market) if m
                  for tok in (m.get("up_token"), m.get("down_token")) if tok]
        self.live.warm(tokens)

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

    def _lead_thread(self):
        """Experiment E's feed (Binance's futures, hundreds of messages a second, measured only): on a loop of its
        own, so reading it never puts a message in front of a book update the strategies are waiting for."""
        self.lead_stream = feeds.BinanceFuturesStream(
            {a: feeds.BINANCE_SYMBOLS[a] for a in self.by_asset if a in feeds.BINANCE_SYMBOLS},
            self._on_futures, min_interval=self.EXCHANGE_GAP)
        asyncio.run(self.lead_stream.run())

    async def _stream_main(self):
        self._loop = asyncio.get_running_loop()
        self.prices = feeds.ChainlinkStream({a: f"{a}/usd" for a in self.by_asset}, self._on_price,
                                            {a: f"{a}usdt" for a in self.by_asset},
                                            lambda asset, ts, price: self._on_fast("relay", asset, ts, price))
        self.exchanges = []
        if self.cfg.exchange_feeds:
            self.exchanges = [
                # (each coin's price passed on every 20 ms at most: in a fast move the nowcast, which the
                # strategies read, stays at most 20 ms behind the exchange, not 50)
                feeds.BinanceStream({a: feeds.BINANCE_SYMBOLS[a] for a in self.by_asset if a in feeds.BINANCE_SYMBOLS},
                                    self._on_fast, min_interval=self.EXCHANGE_GAP),
                feeds.CoinbaseStream({a: feeds.COINBASE_SYMBOLS[a] for a in self.by_asset if a in feeds.COINBASE_SYMBOLS},
                                     self._on_fast, min_interval=self.EXCHANGE_GAP),
            ]
        self.books = feeds.BookStream(self._on_book, self._on_trade)
        self._dirty, self._wake = {}, asyncio.Event()
        extra = []
        if self.stocks:
            self.stocks.stream = stocks.EquityStream(self.stocks.on_price)
            extra.append(self.stocks.stream.run())
        await asyncio.gather(self.prices.run(), self.books.run(), self._heartbeat(), self._evaluator(),
                             *(x.run() for x in self.exchanges), *extra)

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
            # Real money's markets at once: the wait for a burst is for the simulation and the lab, never
            # for an order racing others to the same shares (one more pass on them if the burst goes on).
            for tracker in [t for t in self._dirty if t.live_rules]:
                arrived = self._dirty.pop(tracker)
                self._evaluate(tracker, time.time())
                done = time.time()
                self.health.record_latency(done, done - arrived)
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
        if source in ("binance", "coinbase"):
            self.leads.add(source, asset, price)
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
            if self.stocks:
                tokens |= self.stocks.tokens(now)
                self._guard("stocks", self.stocks.evaluate, now, self.books.get)
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

    def base_strategies(self):
        """The original strategies the system started with (the "iron" ones): the configured
        rule on each configured market."""
        return [{"asset": a, "minutes": self.cfg.window_minutes, "since": 0.0,
                 "variant": lab.Variant(self.cfg.min_edge, self.cfg.min_seconds_left, self.cfg.max_seconds_left,
                                        self.cfg.resolution).key} for a in self.cfg.assets]

    def _list_setting(self, key):
        """A stored list, or None when never stored. (An empty list is a real choice: every
        strategy taken out. _json_setting would read it as "never stored".)"""
        raw = self.store.get_setting(key)
        try:
            value = json.loads(raw) if raw else None
        except ValueError:
            return None
        return value if isinstance(value, list) else None

    def main_strategies(self):
        stored = self._list_setting("main_strategies")
        return self.base_strategies() if stored is None else stored

    def _rules_by_market(self, main, mode="sim"):
        """The rules that trade, by market (a paused one stays listed but doesn't trade), each with
        the risk/reward filter in force there: the owner's for the whole mode (money strip), or
        the one set on this strategy (by the research or by the owner)."""
        over = self._money_overrides()[mode].get("rr") if hasattr(self, "store") else None
        exact = self.lab_exact()[mode] if hasattr(self, "store") else False
        sim = {}
        if mode == "live" and hasattr(self, "store") and self.live_mirror():
            # (as the simulation: its filter, for every strategy that runs there too)
            over = self._money_overrides()["sim"].get("rr")
            sim = {(x["asset"], x["minutes"], x["variant"]): x for x in self.main_strategies()}
        rules = {}
        for m in main:
            if not m.get("paused"):
                src = sim.get((m["asset"], m["minutes"], m["variant"]), m)
                ratio = float(over) if over is not None else float(src.get("ratio") or 0.0)
                if exact:
                    ratio = 0.0  # ("exactly like the lab": the rule the lab ran, no filter put on it)
                v = dataclasses.replace(lab.Variant.from_key(m["variant"]), filter_ratio=ratio,
                                        stake=self._strategy_stake(mode, m) if hasattr(self, "store") else 0.0)
                rules.setdefault((m["asset"], m["minutes"]), []).append(v)
        return rules

    # Managing what runs, beyond adding and removing: pause and resume one rule (it stays listed),
    # bring the original strategies back, and give a strategy a name of the owner's.
    def pause_strategy(self, asset, minutes, variant, mode="sim", paused=True):
        main = self.strategies_for(mode)
        hit = [m for m in main if (m["asset"], m["minutes"], m["variant"]) == (asset, minutes, variant)]
        if not hit:
            raise ValueError("האסטרטגיה הזאת לא רצה שם")
        for m in hit:
            m["paused"] = bool(paused)
        self._save_version(main, f"{'הושהתה' if paused else 'חזרה לפעול'}: {asset.upper()} {minutes}m: {lab.describe(variant)}",
                           "pause" if paused else "resume", mode=mode)
        self._save_for(mode, main, reset_readiness=False)

    def restore_base(self, mode="sim"):
        """Bring the original strategies back where they are missing or paused."""
        main = self.strategies_for(mode)
        changed = []
        for b in self.base_strategies():
            key = (b["asset"], b["minutes"], b["variant"])
            found = [m for m in main if (m["asset"], m["minutes"], m["variant"]) == key]
            self._set_deleted(key, False)
            if not found:
                main.append(dict(b, since=time.time()))
                self._set_removed(mode, key, False)
                changed.append(b)
            elif any(m.get("paused") for m in found):
                for m in found:
                    m["paused"] = False
                changed.append(b)
        if not changed:
            raise ValueError("האסטרטגיות הבסיסיות כבר רצות")
        self._save_version(main, "החזרת האסטרטגיות הבסיסיות", "restore", mode=mode)
        self._save_for(mode, main)

    def strategy_lab(self):
        """The lab's numbers for every strategy the owner manages (connected, taken out, or an
        original one), wherever it ranks: the trading tabs show them like the lab does."""
        keys = {(m["asset"], m["minutes"], m["variant"]) for mode in ("sim", "live") for m in self.strategies_for(mode)}
        keys |= {tuple(k) for ks in self.removed_strategies().values() for k in ks}
        keys |= {(b["asset"], b["minutes"], b["variant"]) for b in self.base_strategies()}
        fields = ("bets", "wins", "pnl", "paid", "roi", "roi_pessimistic", "roi_optimistic", "status", "next",
                  "oos_bets", "oos_roi", "oos_pessimistic", "candidate_since", "rule", "family")
        return {_key(b["asset"], b["minutes"], b["variant"]): {k: b.get(k) for k in fields}
                for b in (self.lab or {}).get("board", []) if (b["asset"], b["minutes"], b["variant"]) in keys}

    def strategy_names(self):
        return self._json_setting("strategy_names", {})

    def set_strategy_name(self, asset, minutes, variant, name):
        names = self.strategy_names()
        name = str(name or "").strip()[:40]
        key = f"{asset}|{int(minutes)}|{variant}"
        if name:
            names[key] = name
        else:
            names.pop(key, None)
        self.store.set_setting("strategy_names", json.dumps(names, ensure_ascii=False))

    # Counting afresh: the simulation's or the lab's results (win rate, profit, return, and the lab's
    # stages) from a moment the owner picks, so strategies are judged on what happens from then on.
    # Nothing is deleted (the research on amounts and filters still uses every trade); going back
    # to all the data is one click.
    # Starting over, a page at a time: the simulation, and each part of the lab (crypto, stocks,
    # both sides, the last minute) counts its results from its own moment; the others keep theirs.
    COUNT_AREAS = {"sim": None, "lab": "main", "stocks": None, "pair": "pair", "end": "end", "maker": "maker"}

    def count_since(self):
        since = self._json_setting("count_since", {})
        if "lab" in since and not self.store.get_setting("count_since_pages"):
            # (once: "lab" used to mean the whole lab, so its other parts keep that moment)
            for area in ("stocks", "pair", "end", "maker"):
                since.setdefault(area, since["lab"])
            self.store.set_setting("count_since", json.dumps(since))
            self.store.set_setting("count_since_pages", "1")
        return since

    def set_count_since(self, area, fresh):
        if area not in self.COUNT_AREAS:
            raise ValueError("אזור לא מוכר")
        # Only the page's win rate, profit / loss and return per dollar count from then on; the
        # strategies keep every trade, their counts and their stages (the research goes on).
        since, fam = self.count_since(), self.COUNT_AREAS[area]
        if fresh:
            since[area] = time.time()
        else:
            since.pop(area, None)
        self.store.set_setting("count_since", json.dumps(since))
        if self.stocks:
            self.stocks.count_since = float(since.get("stocks") or 0)
        log.warning("Counting %s %s", area, "afresh from now" if fresh else "over all the data again")
        if fam:
            self._refresh_lab_soon()

    def _restore_candidates(self):
        """Once: starting over used to set a lab page's stages aside too. They come back, with the
        moment each strategy became a candidate (so its test on new data keeps its trades)."""
        held = [k for (k,) in self.store._x("SELECT key FROM settings WHERE key LIKE 'lab_candidates_before_%'")]
        if held:
            stored = self._json_setting("lab_candidates", {})
            for k in held:
                stored.update(self._json_setting(k, {}))
                self.store.set_setting(k, None)
            self.store.set_setting("lab_candidates", json.dumps(stored))
            log.warning("Lab: the stages set aside by starting over are back (%d settings)", len(held))

    def lab_since(self):
        """Each lab family's own starting moment, for the lab ranking."""
        since = self.count_since()
        return {fam: since.get(area) for area, fam in self.COUNT_AREAS.items() if fam}

    TWIN_GAP = 30  # seconds between counts of the same-period comparison

    def strategy_twins(self, db):
        """Each strategy that runs in the simulation or with real money, side by side with its lab
        twin over the same period (from its first trade there): the lab's figures on a card cover
        its whole history, including what got it picked, so only the same period says whether it
        stopped working (the lab loses too) or runs differently there (the lab still wins).
        Counted at most every TWIN_GAP seconds."""
        hit = self.__dict__.get("_twin_cache")
        if hit and time.time() - hit[0] < self.TWIN_GAP:
            return hit[1]
        out, cards = {"paper": {}, "live": {}}, self.card_since()
        for mode, db_mode in (("sim", "paper"), ("live", "live")):
            for a, m, v in {(x["asset"], x["minutes"], x["variant"]) for x in self.strategies_for(mode)}:
                first = db.execute("SELECT MIN(ts) FROM bets WHERE mode = ? AND asset = ? AND minutes = ? AND variant = ?"
                                   " AND status = 'filled'", (db_mode, a, m, v)).fetchone()[0]
                if first is None:
                    continue
                side = {who if who == "lab" else "here": self._twin_side(db, who, a, m, v, first) for who in ("lab", db_mode)}
                # (the card's own figures from the owner's reset, when he reset it: only what the card shows)
                reset = cards.get(db_mode, {}).get(_key(a, m, v))
                if reset and reset > first:
                    side.update(shown=self._twin_side(db, db_mode, a, m, v, reset), shown_since=reset)
                out[db_mode][_key(a, m, v)] = dict(side, since=first)
        self._twin_cache = (time.time(), out)
        return out

    def _twin_side(self, db, who, a, m, v, since):
        rows = [tuple(r) for r in db.execute(
            "SELECT pnl, cost + fee FROM bets WHERE mode = ? AND asset = ? AND minutes = ? AND variant = ?"
            " AND status = 'filled' AND pnl IS NOT NULL AND ts >= ? ORDER BY ts", (who, a, m, v, since))]
        pnl, paid = sum(x for x, _ in rows), sum(y for _, y in rows)
        roi, low = lab.money_weighted(rows, self.cfg.lab_z) if rows else (None, None)
        return {"settled": len(rows), "wins": sum(1 for x, _ in rows if x > 0), "pnl": pnl,
                "roi": pnl / paid if paid else None,
                # (like the lab's: with only a handful of trades a pessimistic figure means nothing)
                "roi_pessimistic": low if len(rows) >= self.cfg.dash_min_bets else None}

    # A strategy card's figures from now (the owner, 29.10.2026): its return, pessimistic return, win rate, trades
    # and profit start over on the card only. Nothing is deleted: every trade stays in the data, and everything
    # else (the comparison with the lab, the research, the exports) keeps using all of them.
    def card_since(self):
        return self._json_setting("card_since", {})

    def set_card_since(self, mode, asset, minutes, variant, fresh=True):
        db_mode = {"sim": "paper", "live": "live"}.get(mode)
        if not db_mode:
            raise ValueError("bad request")
        C = self.card_since()
        if fresh:
            C.setdefault(db_mode, {})[_key(asset, int(minutes), variant)] = time.time()
        else:
            C.get(db_mode, {}).pop(_key(asset, int(minutes), variant), None)
        self.store.set_setting("card_since", json.dumps(C))
        self.__dict__.pop("_twin_cache", None)
        return C

    PRICE_GROUPS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0001)

    CHANCE_GROUPS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001)

    def price_groups(self, mode, by="price"):
        groups = self._price_groups_raw(mode, by)
        return {"mode": mode, "by": by, "groups": groups, "trades": sum(g["trades"] for g in groups),
                "pnl": sum(g["pnl"] for g in groups)}

    def _price_groups_raw(self, mode, by="price"):
        """Every settled trade (a strategy's round: all its buys in that round, as its card shows it) by
        the price a share cost on average, or (by "chance") by our model's chance for the side when it
        went in: per group, how many trades, how many won, the money made, the average, the biggest win
        and loss, the price paid on average. mode: sim / live / lab (every lab strategy's)."""
        if mode not in ("sim", "live", "lab") or by not in ("price", "chance"):
            raise ValueError("bad request")
        db = connect(self.cfg.db_path)
        try:
            rows = db.execute("SELECT variant, asset, minutes, window_start, SUM(shares), SUM(cost), SUM(cost + fee), SUM(pnl),"
                              " SUM(pnl IS NULL), SUM(fair_prob * shares), SUM(fair_prob IS NULL) FROM bets WHERE mode = ?"
                              " AND status = 'filled' GROUP BY variant, asset, minutes, window_start",
                              ({"live": "live", "lab": "lab"}.get(mode, "paper"),)).fetchall()
        finally:
            db.close()
        edges = self.PRICE_GROUPS if by == "price" else self.CHANCE_GROUPS
        groups = [{"lo": lo, "hi": min(hi, 1.0), "trades": 0, "wins": 0, "pnl": 0.0, "paid": 0.0, "best": None, "worst": None,
                   "shares": 0.0, "cost": 0.0} for lo, hi in zip(edges, edges[1:])]
        for _v, _a, _m, _w, shares, cost, paid, pnl, open_, fair_sh, no_fair in rows:
            if open_ or not shares or pnl is None:
                continue  # (not settled yet)
            if by == "chance" and (no_fair or fair_sh is None):
                continue  # (no chance recorded)
            price = cost / shares if by == "price" else min(fair_sh / shares, 1.0)
            g = next(g for g, hi in zip(groups, edges[1:]) if price < hi)
            g["shares"] += shares
            g["cost"] += cost
            g["trades"] += 1
            g["wins"] += pnl > 0
            g["pnl"] += pnl
            g["paid"] += paid or 0
            g["best"] = pnl if g["best"] is None else max(g["best"], pnl)
            g["worst"] = pnl if g["worst"] is None else min(g["worst"], pnl)
        for g in groups:
            n = g["trades"]
            g.update(win_rate=g["wins"] / n if n else None, avg=g["pnl"] / n if n else None,
                     roi=g["pnl"] / g["paid"] if g["paid"] else None, avg_price=g["cost"] / g["shares"] if g["shares"] else None)
        return groups

    def strategy_diff(self, mode, asset, minutes, variant):
        """Why a strategy runs differently here than in the lab: from its first trade here, round by
        round, the rounds both entered, only here, only in the lab; in the rounds both entered, the
        price a share and the dollars a trade each paid, and what each made per dollar; what the rest
        made; and the settings that differ (the amount, the market mix, the extra entries, the lab's
        entry times)."""
        if mode not in ("sim", "live"):
            raise ValueError("bad request")
        db_mode, minutes = ("live" if mode == "live" else "paper"), int(minutes)
        db = connect(self.cfg.db_path)
        try:
            first = db.execute("SELECT MIN(ts) FROM bets WHERE mode = ? AND asset = ? AND minutes = ? AND variant = ?"
                               " AND status = 'filled'", (db_mode, asset, minutes, variant)).fetchone()[0]
            if first is None:
                raise ValueError("לאסטרטגיה אין עדיין טריידים כאן")
            rounds = {}
            for who in (db_mode, "lab"):
                for ws, status, shares, cost, paid, pnl in db.execute(
                        "SELECT window_start, status, shares, cost, cost + fee, pnl FROM bets WHERE mode = ? AND asset = ?"
                        " AND minutes = ? AND variant = ? AND ts >= ?", (who, asset, minutes, variant, first)):
                    r = rounds.setdefault(ws, {}).setdefault("lab" if who == "lab" else "here",
                                                             {"n": 0, "rejected": 0, "shares": 0.0, "cost": 0.0, "paid": 0.0,
                                                              "pnl": 0.0, "settled": 0})
                    if status != "filled":
                        r["rejected"] += 1
                        continue
                    r["n"] += 1
                    r["shares"] += shares or 0
                    r["cost"] += cost or 0
                    r["paid"] += paid or 0
                    if pnl is not None:
                        r["pnl"] += pnl
                        r["settled"] += 1
        finally:
            db.close()
        filled = lambda r, who: r.get(who) and r[who]["n"] > 0
        both = [r for r in rounds.values() if filled(r, "here") and filled(r, "lab")]
        only_here = [r for r in rounds.values() if filled(r, "here") and not filled(r, "lab")]
        only_lab = [r for r in rounds.values() if filled(r, "lab") and not filled(r, "here")]

        def summary(rs, who):
            n = sum(r[who]["n"] for r in rs)
            shares, cost, paid, pnl = (sum(r[who][k] for r in rs) for k in ("shares", "cost", "paid", "pnl"))
            settled_paid = sum(r[who]["paid"] for r in rs if r[who]["settled"])
            return {"rounds": len(rs), "trades": n, "price": cost / shares if shares else None,
                    "per_trade": paid / n if n else None, "pnl": pnl, "roi": pnl / settled_paid if settled_paid else None}
        rule = lab.Variant.from_key(variant)
        mine = next((m for m in self.strategies_for(mode) if (m["asset"], m["minutes"], m["variant"]) == (asset, minutes, variant)), {})
        t = next((t for t in self.trackers if (t.asset, t.minutes) == (asset, minutes)), None)
        times = self.lab_times()
        return {
            "since": first,
            "both": {"here": summary(both, "here"), "lab": summary(both, "lab")},
            "only_here": summary(only_here, "here"), "only_lab": summary(only_lab, "lab"),
            "rejected": {"here": sum(r["here"]["rejected"] for r in rounds.values() if r.get("here")),
                         "lab": sum(r["lab"]["rejected"] for r in rounds.values() if r.get("lab"))},
            "settings": {
                "stake_here": (mine.get("stake") or (self.live.stake() if mode == "live" else self.sim_limits()["stake"])),
                "stake_lab": self.cfg.stake_usd, "entries": lab.entries(rule),
                "blend": bool(t and t.blend and (t.blend_live if mode == "live" else t.blend_sim)),
                "copies": (t.live_copies if mode == "live" else t.sim_copies) if t else None,
                "lab_runs_it": times is None or lab.time_group(variant) in times,
            },
        }

    def strategy_results(self, db):
        """Each strategy's own record in the simulation and in real money, by (asset, minutes, rule),
        on every trade (starting over resets only the page's headline money figures)."""
        out = {}
        since = {"paper": 0, "live": 0}
        for mode in ("paper", "live"):
            out[mode] = {f"{a}|{m}|{v}": {"bets": n, "settled": s, "wins": w, "pnl": pnl, "paid": paid}
                         for a, m, v, n, s, w, pnl, paid in db.execute(
                             "SELECT asset, minutes, variant, COUNT(*), COUNT(pnl), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),"
                             " COALESCE(SUM(pnl), 0), COALESCE(SUM(CASE WHEN pnl IS NOT NULL THEN cost + fee END), 0)"
                             " FROM bets WHERE mode = ? AND status = 'filled' AND minutes > 0 AND ts >= ?"
                             " GROUP BY asset, minutes, variant", (mode, since[mode])).fetchall()}
        return out

    # The owner's own entries, from a running round's card ("add to this trade"), in the simulation
    # or with real money alike. Kept under their own strategy, so they are followed apart and never
    # counted in a rule's record (nor in the model's calibration: the owner chose them, not the model).
    MANUAL = "manual"
    MANUAL_MIN_LEFT = 5  # seconds: closer to the close than this, no entry

    def _manual_round(self, asset, minutes, window_start):
        """The running round the owner's card shows: (its market, now, seconds left), or why not."""
        now = time.time()
        t = next((t for t in self.trackers if t.asset == asset and t.minutes == int(minutes)), None)
        if t is None or t.window != int(window_start) or not t.market:
            raise ValueError("הסבב הזה כבר נסגר")
        return t, now, t.window + t.window_seconds - now

    def _manual_asks(self, t, side, now):
        token = t.market["up_token"] if side == "up" else t.market["down_token"]
        book = self.books.get(token) if self.books else (*t.api.order_book(token), now)
        return list((book[1] if book else None) or [])

    # The owner closing a running round early, from its card: everything held in it (both sides,
    # every strategy) sold back to the book at what buyers pay now, after the taker fee. The trades
    # keep their record with the profit or loss of the sale (winner "sold": the round's result no
    # longer settles them, nor do they teach the model's calibration).
    def _held(self, mode, t):
        rows = self.store._x("SELECT id, side, shares, cost + fee FROM bets WHERE mode = ? AND status = 'filled'"
                             " AND pnl IS NULL AND winner IS NULL AND asset = ? AND minutes = ? AND window_start = ?",
                             ("live" if mode == "live" else "paper", t.asset, t.minutes, t.window))
        held = {}
        for bet_id, side, shares, paid in rows:
            h = held.setdefault(side, {"ids": [], "shares": 0.0, "paid": 0.0, "by_id": {}})
            h["ids"].append(bet_id)
            h["shares"] += shares
            h["paid"] += paid
            h["by_id"][bet_id] = (shares, paid)
        return held

    def _sale(self, t, side, shares, now):
        """Selling `shares` of `side` to the bids now: (dollars after the fee, the lowest price taken,
        shares the book takes)."""
        token = t.market["up_token"] if side == "up" else t.market["down_token"]
        book = self.books.get(token) if self.books else (*t.api.order_book(token), now)
        got = gross = fee = 0.0
        low = None
        for price, size in sorted((book[0] if book else None) or [], reverse=True):
            if got >= shares - 1e-9:
                break
            take = min(size, shares - got)
            got, gross, low = got + take, gross + take * price, price
            fee += take * taker_fee_per_share(price, t.market_cfg.fee_rate, t.market_cfg.fee_exponent)
        return gross - fee, low, got

    def close_quote(self, mode, asset, minutes, window_start):
        if mode not in ("sim", "live"):
            raise ValueError("bad request")
        t, now, seconds_left = self._manual_round(asset, minutes, window_start)
        held, sides = self._held(mode, t), {}
        for side, h in held.items():
            value, low, takes = self._sale(t, side, h["shares"], now)
            sides[side] = {"shares": h["shares"], "paid": h["paid"], "value": value, "low": low,
                           "book_takes": takes, "whole": takes >= h["shares"] - 1e-6}
        paid = sum(h["paid"] for h in held.values())
        value = sum(x["value"] for x in sides.values())
        return {"now": now, "end": t.window + t.window_seconds, "seconds_left": seconds_left, "sides": sides,
                "paid": paid, "value": value, "pnl_now": value - paid,
                "if_up": held.get("up", {}).get("shares", 0.0) - paid, "if_down": held.get("down", {}).get("shares", 0.0) - paid,
                "open": seconds_left >= self.MANUAL_MIN_LEFT and bool(held) and all(x["whole"] for x in sides.values())}

    def close_round(self, mode, asset, minutes, window_start):
        if mode == "sim":
            t = self._manual_round(asset, minutes, window_start)[0]
            if self.mirror_of(t) and self._held_live(t) and self._held("live", t):
                try:
                    self.close_round("live", asset, minutes, window_start)
                except ValueError as e:
                    raise ValueError(f"בכסף אמיתי זה לא נסגר, ולכן גם בסימולציה לא: {e}")
        q = self.close_quote(mode, asset, minutes, window_start)
        if not q["sides"]:
            raise ValueError("אין מה לסגור בסבב הזה")
        if q["seconds_left"] < self.MANUAL_MIN_LEFT:
            raise ValueError(f"נשארו פחות מ-{self.MANUAL_MIN_LEFT} שניות לסבב")
        t, now, _ = self._manual_round(asset, minutes, window_start)
        held, total = self._held(mode, t), 0.0
        for side, h in held.items():
            x = q["sides"][side]
            if not x["whole"]:
                raise ValueError(f"בספר יש עכשיו קונים רק ל-{x['book_takes']:.1f} מתוך {h['shares']:.1f} המניות")
            if mode == "live":
                token = t.market["up_token"] if side == "up" else t.market["down_token"]
                status, got, sold, error = self.live.sell(token, h["shares"], x["low"])
                if status != "filled":
                    raise ValueError("פולימרקט לא ביצעה את המכירה: " + reject_reason(error))
            else:
                got = x["value"]
            for bet_id in h["ids"]:  # the sale's dollars shared by the shares each trade held
                shares, paid = h["by_id"][bet_id]
                self.store._x("UPDATE bets SET winner = 'sold', pnl = ? WHERE id = ?",
                              (got * shares / h["shares"] - paid, bet_id))
            total += got
        pnl = total - sum(h["paid"] for h in held.values())
        log.warning("CLOSED %s %s early: got $%.2f, %+.2f", mode.upper(), t.key.upper(), total, pnl)
        return {"ok": True, "got": total, "pnl": pnl}

    # The approval mode: a round of the simulation (what its strategies, and the owner's additions,
    # hold in it) offered on its card, and bought with real money on the owner's click, on the same
    # terms or not at all:
    # - the terms are the simulation's, read here from its trades (never taken from the page);
    # - the same trade: the shares of each side in the same proportion, so the round makes, per dollar,
    #   what the simulation's makes if Up wins and if Down wins. That table depends only on the round's
    #   average price a share (the card's "average price"), which must be within APPROVE_SLIP of the
    #   simulation's, dearer or cheaper (a round much cheaper now is not a bargain but another trade:
    #   the market moved against it);
    # - real money's amount, or more when a side would be under Polymarket's least order (shares and
    #   dollars): the smallest copy that keeps the proportion, said on the button before the click;
    # - every side or none: checked whole before anything is sent; the orders go one after the other
    #   (the one with the least room first), and if a later one fails, the ones in are sold back;
    # - what the owner confirmed must still be the simulation's terms when the order goes;
    # - once a round: a round already held with real money isn't offered again.
    APPROVE_SLIP = 0.02
    MIN_BUY_USD = 1.0  # Polymarket refuses a marketable buy under a dollar

    @staticmethod
    def _legs(side=None, price=None, legs=None):
        """[(side, price, weight)], one a side: the round's terms as the page saw them."""
        raw = legs if legs else [{"side": side, "price": price, "weight": 1}]
        out = {}
        for leg in raw:
            sd, pr, w = leg.get("side"), float(leg.get("price")), float(leg.get("weight") or 0)
            if sd not in ("up", "down") or not 0 < pr < 1 or w < 0:
                raise ValueError("bad request")
            out[sd] = (sd, pr, w)
        if not out or not sum(w for _, _, w in out.values()):
            raise ValueError("bad request")
        return sorted(out.values())

    def _sim_legs(self, t):
        """The simulation's terms for a running round: [(side, average price, dollars in it, shares)]."""
        rows = self.store._x("SELECT side, SUM(shares), SUM(cost), SUM(cost + fee) FROM bets WHERE mode = 'paper'"
                             " AND status = 'filled' AND pnl IS NULL AND winner IS NULL AND asset = ? AND minutes = ?"
                             " AND window_start = ? AND side IN ('up', 'down') GROUP BY side ORDER BY side",
                             (t.asset, t.minutes, t.window))
        legs = [(side, round(cost / shares, 4), paid, shares) for side, shares, cost, paid in rows if shares and shares > 0]
        if not legs:
            raise ValueError("לסימולציה אין כלום בסבב הזה")
        return legs

    @staticmethod
    def _same_terms(a, b):
        """Two sets of legs are the same trade: the same sides, each at the same price (to half a cent)
        and with the same share of the money (to 3 points)."""
        if [x[0] for x in a] != [x[0] for x in b]:
            return False
        wa, wb = sum(x[2] for x in a), sum(x[2] for x in b)
        return all(abs(x[1] - y[1]) <= 0.005 and abs(x[2] / wa - y[2] / wb) <= 0.03 for x, y in zip(a, b))

    def _held_live(self, t):
        return self.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live' AND status = 'filled' AND asset = ?"
                             " AND minutes = ? AND window_start = ?", (t.asset, t.minutes, t.window))[0][0]

    def _walk(self, t, side, shares, now):
        """Buying `shares` of `side` from the book now: (shares it has, cost, fee, the dearest price taken,
        the cheapest price on offer, dollars on offer up to that dearest price)."""
        book = sorted(self._manual_asks(t, side, now))
        got = cost = fee = 0.0
        top = None
        for price, size in book:
            if got >= shares - 1e-9:
                break
            take = min(size, shares - got)
            got, cost, top = got + take, cost + take * price, price
            fee += take * taker_fee_per_share(price, t.market_cfg.fee_rate, t.market_cfg.fee_exponent)
        best = book[0][0] if book else None
        room = sum(p * s for p, s in book if top is not None and p <= top + 1e-9)
        return got, cost, fee, top, best, room

    def _approve_plan(self, t, legs, now, seconds_left, ratio=None):
        """The real-money copy of the simulation's round, now: {"fills": [(fill, amount, room)] (the one
        with the least room first), "total", "avg", "sim_avg", "if": {side: what it makes if it wins},
        "legs": per side}, or why not. ratio: the copy's size as a share of the simulation's (taking
        over a round); by default real money's amount, grown to Polymarket's least orders."""
        if self._held_live(t):
            raise ValueError("כבר נכנסת לסבב הזה בכסף אמיתי")
        if seconds_left < self.MANUAL_MIN_LEFT:
            raise ValueError(f"נשארו פחות מ-{self.MANUAL_MIN_LEFT} שניות לסבב")
        sim_paid, sim_shares = sum(x[2] for x in legs), sum(x[3] for x in legs)
        sim_avg = sim_paid / sim_shares
        least = (t.market or {}).get("min_order_size", 0) or 0
        # The copy's size: real money's amount, grown so that every side is an order Polymarket takes.
        k = self.live.stake() / sim_paid if ratio is None else ratio
        for side, price, paid, shares in legs:
            best = min((p for p, _ in self._manual_asks(t, side, now)), default=None) or price
            need = max(least * 1.02 / shares, self.MIN_BUY_USD * 1.03 / (shares * max(best, 0.001)))
            if ratio is not None and k < need:
                raise ValueError(f"ב-{ratio:.0%} {'עלייה' if side == 'up' else 'ירידה'} יוצאת בכסף אמיתי פחות מהמינימום של"
                                 f" פולימרקט ($1 / {least:g} מניות): צריך לפחות {need:.0%}")
            k = max(k, need)
        out, rows, cost_all = [], [], 0.0
        for side, price, paid, shares in legs:
            want = shares * k
            got, cost, fee, top, best, room = self._walk(t, side, want, now)
            name = "עלייה" if side == "up" else "ירידה"
            if got < want - 1e-6:
                raise ValueError(f"{name}: בספר יש רק {got:.1f} מתוך {want:.1f} מניות")
            amount = math.ceil((cost + fee) * 100) / 100
            fill = Fill(side=side, fair_prob=0.0, best_ask=best, avg_price=cost / got, shares=got, cost=cost, fee=fee,
                        edge=0.0, limit_price=top)
            out.append((fill, amount, room / amount if amount else 0.0))
            rows.append({"side": side, "shares": got, "amount": amount, "price": cost / got, "best": best, "sim": price})
            cost_all += cost + fee
        total_shares = sum(f.shares for f, _, _ in out)
        avg = cost_all / total_shares
        made = {side: sum(f.shares for f, _, _ in out if f.side == side) - cost_all for side in ("up", "down")}
        if abs(avg - sim_avg) > self.APPROVE_SLIP + 1e-9:
            dearer = avg > sim_avg
            raise ValueError(f"המחיר הממוצע עכשיו {avg * 100:.1f}¢ ובסימולציה {sim_avg * 100:.1f}¢: "
                             + ("המחיר ברח" if dearer else "השוק זז נגד הטרייד, וזה כבר טרייד אחר"))
        return {"fills": sorted(out, key=lambda x: x[2]), "total": cost_all, "avg": avg, "sim_avg": sim_avg,
                "if": made, "legs": rows, "grown": k > self.live.stake() / sim_paid + 1e-9}

    def approve_real(self, asset, minutes, window_start, variant, side=None, price=None, legs=None):
        seen = self._legs(side, price, legs)
        if not self.live.armed():  # (on approval or automatic: the owner's click copies a round either way)
            raise ValueError("המסחר בכסף אמיתי כבוי")
        t, now, seconds_left = self._manual_round(asset, minutes, window_start)
        legs = self._sim_legs(t)
        if not self._same_terms(seen, [(sd, pr, w) for sd, pr, w, _ in legs]):
            raise ValueError("הטרייד בסימולציה השתנה מאז מה שאישרת (או שהדף ישן): שום דבר לא נקנה. בדוק את הכרטיס ולחץ שוב")
        plan = self._approve_plan(t, legs, now, seconds_left)
        return self._execute_plan(t, plan, variant, now, seconds_left)

    def _execute_plan(self, t, plan, variant, now, seconds_left):
        if not self.live.trades("crypto"):
            raise ValueError("המסחר בכסף אמיתי לא פתוח")
        done = []
        for fill, amount, _ in plan["fills"]:
            status = self.live.maybe_trade(now, t.asset, t.minutes, t.window, seconds_left, fill, t.market, t.market_cfg,
                                           "crypto", None, variant, amount, round_stake=amount, approved=True)
            if status != "filled":
                last = self.live.skips[-1] if self.live.skips else None
                why = ("פולימרקט דחתה את ההזמנה: " + reject_reason(self.live.last_error) if status is not None
                       else last["reason"] if last and last["ts"] >= now else "ההזמנה לא נשלחה")
                if done:  # a side is in and this one isn't: not the trade approved, so the side in is sold back
                    names = " ו".join("עלייה" if f.side == "up" else "ירידה" for f in done)
                    try:
                        back = self.close_round("live", t.asset, t.minutes, t.window)
                        undo = f"ולכן {names} נמכרה בחזרה מיד ({back['pnl']:+.2f}$)"
                    except ValueError as e:
                        undo = f"וניסיתי למכור בחזרה את {names} ולא הצלחתי ({e}): היא נשארה בכסף אמיתי, אפשר לסגור מהכרטיס"
                    log.warning("APPROVED %s: %s in, %s failed (%s); %s", t.key.upper(), names, fill.side, why, undo)
                    raise ValueError(f"{'עלייה' if fill.side == 'up' else 'ירידה'} לא נכנסה ({why}), {undo}")
                raise ValueError(why)
            done.append(fill)
            log.warning("APPROVED %s %s %s: $%.2f at %.1fc", t.key.upper(), variant, fill.side.upper(), amount, fill.avg_price * 100)
        return {"status": "filled", "shares": sum(f.shares for f in done), "paid": sum(f.total_paid for f in done),
                "avg_price": sum(f.cost for f in done) / sum(f.shares for f in done),
                "legs": [{"side": f.side, "shares": f.shares, "avg_price": f.avg_price, "paid": f.total_paid} for f in done]}

    def approve_check(self, items):
        """Whether each round offered on the simulation's cards can be copied with real money right now,
        on the same terms as the button (the page asks several times a second): {key: {"ok", "terms":
        the simulation's legs, "legs": per side, "total", "avg", "sim_avg", "if", "grown", "cash",
        "why"}}. Only the books held in memory are read: nothing is sent anywhere."""
        out = {}
        on = self.live.armed() and self.live.trades("crypto")
        cash = self.live._balance[1] if getattr(self.live, "_balance", None) else None  # (the last one read: no request here)
        for it in (items or [])[:20]:
            key = f"{it.get('asset')}|{it.get('minutes')}|{it.get('window_start')}"
            terms = []
            try:
                t, now, left = self._manual_round(str(it["asset"]), int(it["minutes"]), int(it["window_start"]))
                legs = self._sim_legs(t)
                terms = [{"side": sd, "price": pr, "weight": w} for sd, pr, w, _ in legs]
                sim_avg = sum(x[2] for x in legs) / sum(x[3] for x in legs)
                plan = self._approve_plan(t, legs, now, left)
            except (ValueError, KeyError, TypeError) as e:
                out[key] = {"ok": False, "why": str(e), "legs": [], "terms": terms,
                            "sim_avg": sim_avg if terms else None, "cash": cash}
                continue
            why = None if on else "המסחר בכסף אמיתי לא פתוח"
            if why is None and cash is not None and plan["total"] > cash:
                why = f"בארנק ${cash:.2f}, וההעתקה צריכה ${plan['total']:.2f}"
            out[key] = {"ok": why is None, "why": why, "terms": terms, "legs": plan["legs"], "total": plan["total"],
                        "avg": plan["avg"], "sim_avg": plan["sim_avg"], "if": plan["if"], "grown": plan["grown"], "cash": cash}
        return out

    # "Enter by itself as soon as it can" (the owner, 28.10.2026): a simulation round marked on its card is
    # copied to real money the moment the button's own check passes (the price back within APPROVE_SLIP of
    # the simulation's, the book holding the shares, the cash there), checked a few times a second on the
    # books held in memory; faster than a hand. It waits until the round is too near its close.
    AUTO_COPY_GAP = 0.3

    def auto_copies(self):
        now = time.time()
        return {k: v for k, v in self._json_setting("auto_copy", {}).items() if int(k.split("|")[2]) > now - 86400}

    def set_auto_copy(self, asset, minutes, window_start, variant, on):
        key = _key(asset, int(minutes), int(window_start))
        A = self.auto_copies()
        if on:
            if not self.live.armed():
                raise ValueError("המסחר בכסף אמיתי כבוי")
            self._manual_round(asset, int(minutes), int(window_start))  # (still running, or why not)
            A[key] = {"since": time.time(), "variant": str(variant or self.MANUAL), "status": "waiting"}
        else:
            A.pop(key, None)
        self.store.set_setting("auto_copy", json.dumps(A))
        return {k: dict(v, why=self.__dict__.get("_auto_why", {}).get(k)) for k, v in A.items()}

    def _auto_copy_tick(self):
        A = self.auto_copies()
        waiting = {k: v for k, v in A.items() if v["status"] == "waiting"}
        if not waiting:
            return
        why = self.__dict__.setdefault("_auto_why", {})
        changed = False
        for key, v in waiting.items():
            asset, minutes, start = key.split("|")
            try:
                t, now, left = self._manual_round(asset, int(minutes), int(start))
            except ValueError as e:
                v.update(status="expired", why=str(e))
                changed = True
                continue
            if self._held_live(t):
                v.update(status="in", why="כבר בכסף אמיתי")
                changed = True
                continue
            if left < self.MANUAL_MIN_LEFT or not self.live.armed():
                v.update(status="expired", why="הסבב נגמר לפני שהתאפשר" if left < self.MANUAL_MIN_LEFT else "המסחר בכסף אמיתי כובה")
                changed = True
                continue
            try:
                plan = self._approve_plan(t, self._sim_legs(t), now, left)
                cash = self.live._balance[1] if getattr(self.live, "_balance", None) else None
                if cash is not None and plan["total"] > cash:
                    raise ValueError(f"בארנק ${cash:.2f}, וההעתקה צריכה ${plan['total']:.2f}")
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
                why[key] = str(e)  # (still waiting: the reason it can't yet, shown on the card)
                continue
            try:
                done = self._execute_plan(t, plan, v["variant"], now, left)
                v.update(status="in", at=now, paid=done["paid"], avg=done["avg_price"])
                log.warning("AUTO COPY %s: in at %.1fc", key, done["avg_price"] * 100)
            except ValueError as e:
                v.update(status="failed", why=str(e))
            changed = True
        if changed:
            stored = self.auto_copies()
            stored.update({k: A[k] for k in waiting})
            self.store.set_setting("auto_copy", json.dumps(stored))

    # Taking over a round (the approval mode): from then on, every move the owner makes on the
    # simulation's card (adding to a side, closing) is made with real money at the same moment, at
    # `ratio` of the simulation's size: real money's first, and if it can't be done, the simulation's
    # isn't either. So what real money holds is exactly what the owner built, at the same prices.
    # What the round held before (the strategies' entries) comes along only if real money can copy it
    # on the same terms at that moment (copy=True); otherwise real money starts from the owner's moves.
    MIRROR_RATIOS = (1.0, 0.5, 0.25, 0.1, 0.05)

    def mirrors(self):
        now = time.time()
        out = self._json_setting("approval_mirror", {})
        return {k: v for k, v in out.items() if int(k.split("|")[2]) > now - 2 * 86400}

    def mirror_of(self, t):
        return self.mirrors().get(_key(t.asset, t.minutes, t.window))

    def take_over(self, asset, minutes, window_start, ratio, copy=True):
        ratio = float(ratio)
        if ratio not in self.MIRROR_RATIOS:
            raise ValueError("bad request")
        if not self.live.armed():
            raise ValueError("המסחר בכסף אמיתי כבוי")
        if not self.live.trades("crypto"):
            raise ValueError("המסחר בכסף אמיתי לא פתוח")
        t, now, seconds_left = self._manual_round(asset, minutes, window_start)
        if self.mirror_of(t):
            raise ValueError("הטרייד הזה כבר בשליטה שלך")
        if self._held_live(t):
            raise ValueError("כבר נכנסת לסבב הזה בכסף אמיתי")
        if seconds_left < self.MANUAL_MIN_LEFT:
            raise ValueError(f"נשארו פחות מ-{self.MANUAL_MIN_LEFT} שניות לסבב")
        done = None
        if copy:
            plan = self._approve_plan(t, self._sim_legs(t), now, seconds_left, ratio=ratio)
            done = self._execute_plan(t, plan, self.MANUAL, now, seconds_left)
        stored = self.mirrors()
        stored[_key(t.asset, t.minutes, t.window)] = {"ratio": ratio, "since": now, "copied": bool(copy)}
        self.store.set_setting("approval_mirror", json.dumps(stored))
        log.warning("TAKE OVER %s at %.0f%% (%s)", t.key.upper(), ratio * 100, "with the round" if copy else "from now")
        return {"ok": True, "ratio": ratio, "copied": done}

    def _buy_now(self, t, side, amount, now):
        """The fill `amount` dollars of `side` gets from the book now (the whole amount or none)."""
        asks = self._manual_asks(t, side, now)
        if not asks:
            raise ValueError("אין מוכרים בצד הזה עכשיו")
        fair_up = t.snapshot.get("fair_up")
        prob = None if fair_up is None else (fair_up if side == "up" else 1 - fair_up)
        cfg = dataclasses.replace(t.market_cfg, min_edge=-1.0, min_ratio=0.0, max_ratio=0.0, stake_usd=amount)
        shares, cost, fee, top = simulate_buy(asks, amount, prob if prob is not None else 1.0, cfg)
        if shares <= 0 or cost + fee < amount * 0.999:
            raise ValueError(f"בספר יש עכשיו רק ${cost + fee:.2f} בצד הזה, פחות מהסכום")
        return Fill(side=side, fair_prob=prob or 0.0, best_ask=min(p for p, _ in asks), avg_price=cost / shares, shares=shares,
                    cost=cost, fee=fee, edge=(prob or 0.0) - (cost + fee) / shares, limit_price=top), prob

    def _mirror_buy(self, t, m, side, amount, now, seconds_left):
        """Real money's part of an addition on a round taken over: `ratio` of it, now, or why not."""
        real = round(amount * m["ratio"], 2)
        least = (t.market or {}).get("min_order_size", 0) or 0
        fill, _ = self._buy_now(t, side, real, now)
        if real < self.MIN_BUY_USD or fill.shares < least:
            raise ValueError(f"בכסף אמיתי זה יוצא ${real:.2f} ({fill.shares:.1f} מניות), פחות מהמינימום של פולימרקט"
                             f" ($1 / {least:g} מניות): הגדל את הסכום")
        status = self.live.maybe_trade(now, t.asset, t.minutes, t.window, seconds_left, fill, t.market, t.market_cfg,
                                       "crypto", None, self.MANUAL, real, round_stake=real, approved=True)
        if status != "filled":
            last = self.live.skips[-1] if self.live.skips else None
            raise ValueError("בכסף אמיתי זה לא נכנס, ולכן גם בסימולציה לא: " + (
                "פולימרקט דחתה את ההזמנה: " + reject_reason(self.live.last_error) if status is not None
                else last["reason"] if last and last["ts"] >= now else "ההזמנה לא נשלחה"))
        log.warning("MIRROR %s %s: $%.2f real at %.1fc", t.key.upper(), side.upper(), real, fill.avg_price * 100)
        return fill

    MANUAL_OPTIONS = 5  # at most this many amounts offered a side

    def manual_quote(self, mode, asset, minutes, window_start):
        """What the owner can add to a running round right now, a side at a time: a few amounts the
        book can take whole (the strategy's usual amount and its multiples, and all that sits at the
        best price), each with its shares and price, and what the round as a whole then makes if
        Up wins and if Down wins (with what the round holds already). The page asks every second."""
        if mode not in ("sim", "live"):
            raise ValueError("bad request")
        t, now, seconds_left = self._manual_round(asset, minutes, window_start)
        db_mode = "live" if mode == "live" else "paper"
        held = {"up": 0.0, "down": 0.0}
        paid = 0.0
        for side, shares, cost in self.store._x(
                "SELECT side, shares, cost + fee FROM bets WHERE mode = ? AND status = 'filled' AND asset = ?"
                " AND minutes = ? AND window_start = ?", (db_mode, t.asset, t.minutes, t.window)):
            held[side] += shares
            paid += cost
        stake = float(self.money()[mode]["stake"]["value"] or 5.0)
        ceiling = self.cfg.live_max_stake_usd if mode == "live" else 10000
        least = t.market.get("min_order_size") or 0
        fair_up = t.snapshot.get("fair_up")
        cfg = dataclasses.replace(t.market_cfg, min_edge=-1.0, min_ratio=0.0, max_ratio=0.0)
        sides = {}
        for side in ("up", "down"):
            asks = sorted(self._manual_asks(t, side, now))
            options = []
            if asks:
                best = asks[0][0]
                at_best = asks[0][1] * (best + taker_fee_per_share(best, cfg.fee_rate, cfg.fee_exponent))
                amounts = sorted({round(stake * k, 2) for k in (1, 2, 4, 10)} | {math.floor(at_best * 100) / 100})
                for amount in amounts:
                    if amount <= 0 or amount > ceiling + 1e-9:
                        continue
                    shares, cost, fee, top = simulate_buy(asks, amount, 1.0, cfg)
                    if cost + fee < amount * 0.999 or shares < least:
                        continue  # the book can't take it whole, or under Polymarket's minimum
                    up, down = held["up"] + (shares if side == "up" else 0), held["down"] + (shares if side == "down" else 0)
                    options.append({"amount": amount, "shares": shares, "avg": (cost + fee) / shares, "top": top,
                                    "best": abs(amount - math.floor(at_best * 100) / 100) < 1e-9,
                                    "if_up": up - paid - amount, "if_down": down - paid - amount})
                options = options[:self.MANUAL_OPTIONS]
            prob = None if fair_up is None else (fair_up if side == "up" else 1 - fair_up)
            sides[side] = {"ask": asks[0][0] if asks else None, "chance": prob, "options": options}
        m = self.mirror_of(t) if mode == "sim" else None
        if m:  # (a round taken over: each option's real-money part, and whether Polymarket takes it)
            for side, x in sides.items():
                for o in x["options"]:
                    real = round(o["amount"] * m["ratio"], 2)
                    o["real"] = real
                    o["real_ok"] = real >= self.MIN_BUY_USD and o["shares"] * m["ratio"] >= least
        return {"now": now, "end": t.window + t.window_seconds, "seconds_left": seconds_left, "mirror": m,
                "open": seconds_left >= self.MANUAL_MIN_LEFT, "held": held, "paid": paid,
                "now_if_up": held["up"] - paid, "now_if_down": held["down"] - paid, "sides": sides}

    def manual_add(self, mode, asset, minutes, window_start, side, amount):
        """Buy `amount` dollars of `side` in this round now, at what the book offers (the whole
        amount or nothing, as a real order), whatever a rule's edge says: the owner decided. The
        owner's protections still hold (the simulation's limits, real money's caps and the day's
        loss limit)."""
        if mode not in ("sim", "live") or side not in ("up", "down"):
            raise ValueError("bad request")
        amount = float(amount)
        if not 0 < amount <= 10000:
            raise ValueError("סכום לא תקין")
        t, now, seconds_left = self._manual_round(asset, minutes, window_start)
        if seconds_left < self.MANUAL_MIN_LEFT:
            raise ValueError(f"נשארו פחות מ-{self.MANUAL_MIN_LEFT} שניות לסבב")
        asks = self._manual_asks(t, side, now)
        if not asks:
            raise ValueError("אין מוכרים בצד הזה עכשיו")
        fair_up, raw_up = t.snapshot.get("fair_up"), t.snapshot.get("fair_raw")
        if fair_up is None:
            raise ValueError("המודל עוד לא חישב סיכוי לסבב הזה")
        prob = fair_up if side == "up" else 1 - fair_up
        cfg = dataclasses.replace(t.market_cfg, min_edge=-1.0, min_ratio=0.0, max_ratio=0.0, stake_usd=amount)
        shares, cost, fee, top = simulate_buy(asks, amount, prob, cfg)
        if shares <= 0 or cost + fee < amount * 0.999:
            raise ValueError(f"בספר יש עכשיו רק ${cost + fee:.2f} בצד הזה, פחות מהסכום")
        fill = Fill(side=side, fair_prob=prob, best_ask=min(p for p, _ in asks), avg_price=cost / shares, shares=shares,
                    cost=cost, fee=fee, edge=prob - (cost + fee) / shares, limit_price=top)
        if mode == "sim":
            m = self.mirror_of(t)
            if not m and t.can_open and not t.can_open("crypto", fill.total_paid, side, t.window):
                raise ValueError("ההגבלות של הסימולציה לא מאפשרות עוד טרייד עכשיו (טריידים פתוחים / אותו כיוון / הפסד יומי)")
            if m:  # (a round taken over: real money first, at the same moment; if it can't, neither)
                if not self.live.approval() or not self.live.trades("crypto"):
                    raise ValueError("הטרייד בשליטה שלך, והמסחר בכסף אמיתי לא פתוח: שום דבר לא נקנה")
                self._mirror_buy(t, m, side, amount, now, seconds_left)
            self.store.add_bet(now, t.asset, t.minutes, t.window, seconds_left, fill, variant=self.MANUAL, stake=amount,
                               fair_raw=None if raw_up is None else (raw_up if side == "up" else 1 - raw_up))
            status = "filled"
        else:
            if not self.live.trades("crypto"):
                raise ValueError("המסחר בכסף אמיתי כבוי")
            status = self.live.maybe_trade(now, t.asset, t.minutes, t.window, seconds_left, fill, t.market, t.market_cfg,
                                           "crypto", None, self.MANUAL, amount, round_stake=amount)
            if status is None:
                last = self.live.skips[-1] if self.live.skips else None
                raise ValueError(last["reason"] if last and last["ts"] >= now else "ההזמנה לא נשלחה")
            if status != "filled":
                raise ValueError("פולימרקט דחתה את ההזמנה: " + reject_reason(self.live.last_error))
        log.warning("MANUAL %s %s %s %s: $%.2f at %.1fc", mode.upper(), t.key.upper(), side.upper(), status,
                    amount, fill.avg_price * 100)
        return {"status": status, "shares": fill.shares, "avg_price": fill.avg_price, "paid": fill.total_paid}

    def _apply_main(self, main):
        rules = self._rules_by_market(main, "sim")
        for t in self.trackers:
            mine = rules.get((t.asset, t.minutes), [])
            t.set_main_rules(mine, 1.0 if mine else self.cfg.lab_record_seconds)
        self.main_trackers = [t for t in self.trackers if t.main]

    # Real money has its own strategy: it starts as the simulation's and changes only when the
    # owner pushes to it (or switched on the automatic push for real money).
    def live_main(self):
        stored = self._list_setting("live_main")
        return self.main_strategies() if stored is None else stored

    def strategies_for(self, mode):
        return self.live_main() if mode == "live" else self.main_strategies()

    def _apply_live_main(self, main):
        rules = self._rules_by_market(main, "live")
        for t in self.trackers:
            t.live_rules = rules.get((t.asset, t.minutes), [])

    def _save_live_main(self, main, reset_readiness=True):
        self.store.set_setting("live_main", json.dumps(main))
        # Readiness for real money is judged afresh on the new strategy's own bets.
        if reset_readiness:
            self.store.set_setting("live_changed_at", str(time.time()))
        self._apply_live_main(main)
        log.warning("Real-money strategy is now: %s", ", ".join(f"{m['asset']}-{m['minutes']}m {m['variant']}" for m in main))
        self._refresh_lab_soon()

    def _refresh_lab_soon(self):
        """The lab's table and readiness after the owner changed strategies. On the running server
        in the background (a pass over every trade takes a second or two there), so the click
        answers at once; several changes in a row make one pass."""
        if not self.streaming:
            self.refresh_lab()
            return
        if not self._lab_queued.acquire(blocking=False):
            return  # one is already waiting and will see this change too

        def run():
            with self._learn_lock:
                self._lab_queued.release()
                try:
                    self.refresh_lab()
                except Exception as e:
                    log.warning("refresh_lab: %s", e)
        threading.Thread(target=run, daemon=True, name="lab-refresh").start()

    def _save_for(self, mode, main, reset_readiness=True):
        (self._save_live_main if mode == "live" else self._save_main)(main, reset_readiness)
        if getattr(self, "current_variants", None) is not None:
            self._refresh_lab_variants()  # the lab runs whatever the owner connects
        self._apply_money()  # the day's limit follows how much is connected

    def promote(self, asset, minutes, variant, mode="sim"):
        """Make a proven lab rule the strategy on its market (the owner's decision)."""
        self.push([(asset, minutes, variant)], mode=mode)

    def _weakest(self, main, asset, minutes, board, keep=None):
        """Of the rules running on a market, the one the lab rates lowest (pessimistic return)."""
        on = [m for m in main if (m["asset"], m["minutes"]) == (asset, minutes) and m["variant"] != keep]
        score = lambda m: (board.get((asset, minutes, m["variant"])) or {}).get("roi_pessimistic")
        return min(on, key=lambda m: -9 if score(m) is None else score(m), default=None)

    def push(self, changes, label_prefix="", mode="sim", replace=False, allow_unproven=False):
        """Add strategies to the simulation or to real money: each change puts a lab rule on its
        market, next to what runs there (replace: instead of it). A market runs at most
        max_rules_per_market; past that the weakest one makes room. Real money takes only proven
        rules; the simulation can take any, when the owner asks for it explicitly. Saved as a new
        version, so the one before it can always be restored."""
        board = {(b["asset"], b["minutes"], b["variant"]): b for b in (self.lab or {}).get("board", [])}
        if not changes:
            raise ValueError("אין שינוי לדחוף")
        main = self.strategies_for(mode)
        texts, stats = [], []
        for asset, minutes, variant in changes:
            _lab_only(variant, mode)
            row = board.get((asset, minutes, variant))
            if row is None or (row["status"] != "proven" and not (allow_unproven and mode == "sim")):
                raise ValueError("לכסף אמיתי אפשר לדחוף רק אסטרטגיה מוכחת" if mode == "live" else "אפשר לדחוף רק אסטרטגיה מוכחת")
            if not any((t.asset, t.minutes) == (asset, minutes) for t in self.trackers):
                raise ValueError("השוק הזה לא במעקב")
            if any((m["asset"], m["minutes"], m["variant"]) == (asset, minutes, variant) for m in main):
                continue  # already running there
            if replace:
                main = [m for m in main if (m["asset"], m["minutes"]) != (asset, minutes)]
            elif sum(1 for m in main if (m["asset"], m["minutes"]) == (asset, minutes)) >= self.max_rules():
                weakest = self._weakest(main, asset, minutes, board)
                main = [m for m in main if m is not weakest]
                texts.append(f"הוצאה (הכי חלשה בשוק): {asset.upper()} {minutes}m: {lab.describe(weakest['variant'])}")
            main.append({"asset": asset, "minutes": minutes, "variant": variant, "since": time.time()})
            self._set_removed(mode, (asset, minutes, variant), False)
            self._set_deleted((asset, minutes, variant), False)
            texts.append(f"{asset.upper()} {minutes}m: {lab.describe(variant)}")
            stats.append({k: row.get(k) for k in ("asset", "minutes", "variant", "bets", "roi", "roi_pessimistic",
                                                   "oos_bets", "oos_roi", "oos_pessimistic")})
        if not stats:
            raise ValueError("האסטרטגיה כבר רצה שם")
        self._save_version(main, label_prefix + "; ".join(texts), "push", stats, mode=mode)
        self._save_for(mode, main)

    def remove_strategy(self, asset, minutes, variant, mode="sim"):
        """Take one rule off the simulation or real money (the owner's decision, the original
        ones and the last one included). It stays in that mode's "taken out" list, to put back
        or delete. Already out: nothing to do (a second click, or a page a moment behind)."""
        main = self.strategies_for(mode)
        key = (asset, minutes, variant)
        left = [m for m in main if (m["asset"], m["minutes"], m["variant"]) != key]
        if len(left) == len(main):
            return
        self._set_removed(mode, key, True)
        self._save_version(left, f"הוסרה: {asset.upper()} {minutes}m: {lab.describe(variant)}", "remove", mode=mode)
        self._save_for(mode, left)

    # Taken-out strategies, per mode, so they can be put back; and the ones deleted for good,
    # which no list shows any more (until they are added from the lab again).
    def removed_strategies(self):
        s = self._json_setting("removed_strategies", {})
        return {m: [k for k in s.get(m, []) if isinstance(k, list) and len(k) == 3] for m in ("sim", "live")}

    def _set_removed(self, mode, key, on):
        s = self.removed_strategies()
        key = [key[0], int(key[1]), key[2]]
        s[mode] = [k for k in s[mode] if k != key] + ([key] if on else [])
        self.store.set_setting("removed_strategies", json.dumps(s))
        self.hide_removed(mode, key, False)  # a fresh removal shows in the trading tab again

    # Taken-out strategies the owner closed (✕) in a trading tab: still taken out (the settings list
    # them), just no longer shown there.
    def hidden_removed(self):
        s = self._json_setting("removed_hidden", {})
        return {m: [k for k in s.get(m, []) if isinstance(k, list) and len(k) == 3] for m in ("sim", "live")}

    def hide_removed(self, mode, key, on=True):
        s = self.hidden_removed()
        key = [key[0], int(key[1]), key[2]]
        s[mode] = [k for k in s[mode] if k != key] + ([key] if on else [])
        self.store.set_setting("removed_hidden", json.dumps(s))

    def deleted_strategies(self):
        return set(self._json_setting("deleted_strategies", []))

    def _set_deleted(self, key, on):
        deleted = self.deleted_strategies()
        (deleted.add if on else deleted.discard)(_key(*key))
        self.store.set_setting("deleted_strategies", json.dumps(sorted(deleted)))

    def undelete_strategy(self, asset, minutes, variant):
        """Bring a deleted strategy back into the lists (not connected anywhere)."""
        self._set_deleted((asset, int(minutes), variant), False)

    def max_rules(self):
        """At most this many strategies on one market, in each mode (the owner's number, or 3)."""
        return int(self.store.get_setting("max_rules_per_market") or self.cfg.max_rules_per_market)

    def set_max_rules(self, n):
        n = int(n)
        if not 1 <= n <= 20:
            raise ValueError("מקסימום אסטרטגיות בשוק: בין 1 ל-20")
        self.store.set_setting("max_rules_per_market", str(n))
        log.warning("At most %d strategies a market", n)

    def put_back_strategy(self, asset, minutes, variant, mode="sim"):
        """Put a strategy back into the simulation or real money: one taken out earlier, an
        original one, or one running in the other mode. The owner's call, proven or not."""
        key = (asset, int(minutes), variant)
        main = self.strategies_for(mode)
        if any((m["asset"], m["minutes"], m["variant"]) == key for m in main):
            return
        if not any((t.asset, t.minutes) == key[:2] for t in self.trackers):
            raise ValueError("השוק הזה לא במעקב")
        if sum(1 for m in main if (m["asset"], m["minutes"]) == key[:2]) >= self.max_rules():
            raise ValueError(f"כבר רצות {self.max_rules()} אסטרטגיות בשוק הזה. צריך להוציא אחת קודם.")
        lab.Variant.from_key(variant)  # a rule the system knows
        _lab_only(variant, mode)
        main.append({"asset": key[0], "minutes": key[1], "variant": variant, "since": time.time()})
        self._set_removed(mode, key, False)
        self._set_deleted(key, False)
        self._save_version(main, f"הוחזרה: {asset.upper()} {minutes}m: {lab.describe(variant)}", "putback", mode=mode)
        self._save_for(mode, main)

    def delete_strategy(self, asset, minutes, variant):
        """Delete a strategy for good: out of the simulation and real money, and off every list."""
        key = (asset, int(minutes), variant)
        for mode in ("sim", "live"):
            main = self.strategies_for(mode)
            left = [m for m in main if (m["asset"], m["minutes"], m["variant"]) != key]
            if len(left) != len(main):
                self._save_version(left, f"נמחקה: {asset.upper()} {minutes}m: {lab.describe(variant)}", "delete", mode=mode)
                self._save_for(mode, left)
            self._set_removed(mode, key, False)
        self._set_deleted(key, True)

    # When a lab strategy becomes proven: a mail, and for each of the simulation and real money
    # whose automatic push the owner switched on, a push (only if it beats what runs there).
    def auto_push(self, mode="sim"):
        return self.store.get_setting("live_auto_push" if mode == "live" else "lab_auto_push", "0") == "1"

    def set_auto_push(self, on, mode="sim"):
        self.store.set_setting("live_auto_push" if mode == "live" else "lab_auto_push", "1" if on else "0")
        log.warning("Automatic push of proven strategies (%s): %s", mode, "on" if on else "off")

    # A strategy that reaches stage 3 of 4 (a candidate: profitable even in the pessimistic case, now
    # tested on new data) goes into the simulation at once, when the owner has this on (the default),
    # and an email says so. Per market only the best new one, and only if it beats the weakest of
    # the market's strategies when the market is full (that one makes room). The lab-only kinds
    # (both sides) never go. Saved as a version, so the one before it can always be restored.
    def auto_candidates(self):
        # (off by default: a candidate is the luckiest of hundreds, not yet tested on new data)
        return self.store.get_setting("lab_auto_candidates", "0") == "1"

    def set_auto_candidates(self, on):
        self.store.set_setting("lab_auto_candidates", "1" if on else "0")
        log.warning("Candidates into the simulation automatically: %s", "on" if on else "off")

    def _on_candidates(self, fresh):
        if not fresh or not self.auto_candidates() or getattr(self, "notifier", None) is None:
            return
        board = {(b["asset"], b["minutes"], b["variant"]): b for b in (self.lab or {}).get("board", [])}
        running = self.strategies_for("sim")
        on = {(m["asset"], m["minutes"], m["variant"]) for m in running}
        tracked = {(t.asset, t.minutes) for t in self.trackers}
        best = {}
        for b in fresh:
            key = (b["asset"], b["minutes"])
            if b.get("family") == "pair" or (*key, b["variant"]) in on or key not in tracked:
                continue
            if key not in best or (b["roi_pessimistic"] or -9) > (best[key]["roi_pessimistic"] or -9):
                best[key] = b
        added = []
        for (asset, minutes), b in best.items():
            weakest = None
            if sum(1 for m in running if (m["asset"], m["minutes"]) == (asset, minutes)) >= self.max_rules():
                weakest = self._weakest(running, asset, minutes, board)
                current = board.get((asset, minutes, weakest["variant"])) if weakest else None
                if current and current["roi_pessimistic"] is not None and (b["roi_pessimistic"] or -9) <= current["roi_pessimistic"]:
                    continue  # no better than the weakest one already running there
            added.append(dict(b, replaced=lab.describe(weakest["variant"]) if weakest else None))
        if not added:
            return
        try:
            self.push([(b["asset"], b["minutes"], b["variant"]) for b in added],
                      label_prefix="הוספה אוטומטית (שלב 3 מתוך 4): ", mode="sim", allow_unproven=True)
        except Exception as e:
            log.warning("Automatic add of candidates failed: %s", e)
            return
        log.warning("Lab: %d candidates added to the simulation", len(added))
        self.notifier.lab_candidates(added)

    def _stocks_auto_push(self, board):
        """The stock lab's brain, as the crypto lab's: with the automatic push to the simulation on, the
        best proven rule becomes the stock strategy (when it isn't already)."""
        if not (self.stocks and self.auto_push("sim")):
            return
        best = next((b for b in board if b["status"] == "proven"), None)
        current = (self.stocks.main() or {}).get("variant")
        if best and best["variant"] != current and (
                current is None or best["roi_pessimistic"] > ((next((b for b in board if b["variant"] == current), {}) or {})
                                                                 .get("roi_pessimistic") or -9)):
            self.stocks.set_main(best["variant"])
            log.warning("Stocks lab: %s proven, pushed into the simulation", best["variant"])
            self._apply_sim()  # (the owner's switch for the stock strategy in the simulation stays as they set it)

    def _on_proven(self, board):
        if getattr(self, "notifier", None) is None:
            return  # still starting up: the next refresh looks again
        proven = {_key(b["asset"], b["minutes"], b["variant"]): b for b in board if b["status"] == "proven"}
        raw = self.store.get_setting("lab_proven_seen")  # (not _json_setting: an empty list is not "never")
        seen = json.loads(raw) if raw else None
        self.store.set_setting("lab_proven_seen", json.dumps(sorted(set(seen or []) | set(proven))))
        if seen is None:
            return  # first run: what is proven already was there before this feature
        new = [b for k, b in proven.items() if k not in set(seen)]
        if not new:
            return
        pushed = {}
        new_keys = {(b["asset"], b["minutes"], b["variant"]) for b in new}
        for mode in ("sim", "live"):
            if not self.auto_push(mode):
                continue
            changes = [(c["asset"], c["minutes"], c["variant"]) for c in self.recommendation(mode)["changes"]
                       if (c["asset"], c["minutes"], c["variant"]) in new_keys]
            if changes:
                try:
                    self.push(changes, label_prefix="דחיפה אוטומטית: ", mode=mode)
                    pushed[mode] = changes
                except Exception as e:
                    log.warning("Automatic push (%s) failed: %s", mode, e)
        log.warning("Lab: %d strategies proven; pushed: %s", len(new), {m: len(c) for m, c in pushed.items()} or "none")
        self.notifier.lab_proven(new, pushed)

    def demote(self, asset, minutes, mode="sim"):
        main = [m for m in self.strategies_for(mode) if (m["asset"], m["minutes"]) != (asset, minutes)]
        if not main:
            raise ValueError("חייבת להישאר לפחות אסטרטגיה ראשית אחת")
        self._save_version(main, f"{asset.upper()} {minutes}m הוסר מהאסטרטגיה", "remove", mode=mode)
        self._save_for(mode, main)

    # Versions of each strategy (the simulation's, real money's): a snapshot each time the owner
    # changes it, so any earlier one can be restored. The first change also keeps what ran before it.
    @staticmethod
    def _version_keys(mode):
        return ("live_strategy_versions", "live_strategy_current") if mode == "live" else \
            ("strategy_versions", "strategy_current")

    def strategy_versions(self, mode="sim"):
        return self._json_setting(self._version_keys(mode)[0], [])

    def current_version(self, mode="sim"):
        versions = self.strategy_versions(mode)
        current = self.store.get_setting(self._version_keys(mode)[1])
        return int(current) if current else (versions[-1]["id"] if versions else None)

    def _save_version(self, main, label, kind, stats=None, stocks_variant=None, mode="sim"):
        versions = self.strategy_versions(mode)
        live = mode == "live"
        stock_now = (self.stocks.main() or {}).get("variant") if self.stocks and not live else None
        if not versions:
            since = self.store.get_setting("live_changed_at" if live else "main_changed_at", "0")
            first = {"id": 1, "ts": float(since or 0) or None, "label": "האסטרטגיה המקורית", "kind": "original",
                     "main": self.strategies_for(mode)}
            versions.append(first if live else dict(first, stocks=stock_now))
        new_id = max(v["id"] for v in versions) + 1
        version = {"id": new_id, "ts": time.time(), "label": label, "kind": kind, "main": main, "stats": stats or []}
        if not live:
            # None: the stock strategy stays as it is; "": it was removed.
            version["stocks"] = stock_now if stocks_variant is None else (stocks_variant or None)
        versions.append(version)
        keys = self._version_keys(mode)
        self.store.set_setting(keys[0], json.dumps(versions[-60:]))
        self.store.set_setting(keys[1], str(new_id))

    def versions_view(self, mode="sim"):
        """The versions, newest first, each with the strategies it took out (compared with the one
        before it) and whether they are connected again, for "put this strategy back"."""
        versions = self.strategy_versions(mode)
        now = {(m["asset"], m["minutes"], m["variant"]) for m in self.strategies_for(mode)}
        keys = lambda v: {(m["asset"], m["minutes"], m["variant"]) for m in v.get("main") or []}
        out = []
        for i, v in enumerate(versions):
            gone = sorted(keys(versions[i - 1]) - keys(v)) if i else []
            out.append(dict(v, removed=[list(k) for k in gone], removed_back=bool(gone) and all(k in now for k in gone)))
        return out[::-1]

    def rollback(self, version_id, mode="sim"):
        """Go back to an earlier version of a strategy (the owner's decision). Everything is
        checked first, so a rollback is applied whole or not at all."""
        version = next((v for v in self.strategy_versions(mode) if v["id"] == int(version_id)), None)
        if version is None:
            raise ValueError("גרסה לא קיימת")
        tracked = {(t.asset, t.minutes) for t in self.trackers}
        main = [m for m in version["main"] if (m["asset"], m["minutes"]) in tracked]
        if not main:
            raise ValueError("השווקים של הגרסה הזאת לא במעקב כרגע")
        stock_target = version.get("stocks") or None
        if stock_target and stock_target not in {v.key for v in stocks.VARIANTS}:
            raise ValueError("אסטרטגיית המניות של הגרסה הזאת כבר לא קיימת")
        self.store.set_setting(self._version_keys(mode)[1], str(version["id"]))
        if (mode == "sim" and self.stocks and "stocks" in version
                and stock_target != (self.stocks.main() or {}).get("variant")):
            self.stocks.set_main(stock_target)  # unchanged: its track record keeps counting
        unchanged = sorted((m["asset"], m["minutes"], m["variant"]) for m in self.strategies_for(mode)) == \
            sorted((m["asset"], m["minutes"], m["variant"]) for m in main)
        if not unchanged:
            self._save_for(mode, [dict(m, since=time.time()) for m in main])
        log.warning("Strategy (%s) rolled back to version %s", mode, version["id"])

    def recommendation(self, mode="sim"):
        """Should the owner push a new strategy now (to the simulation or to real money)? From the
        lab's stages: only a rule proven on data it wasn't picked on, and better than what runs
        on its market there now."""
        board = (self.lab or {}).get("board", [])
        running = self.strategies_for(mode)
        rows = {(b["asset"], b["minutes"], b["variant"]): b for b in board}
        on = {(m["asset"], m["minutes"], m["variant"]) for m in running}
        full = lambda key: sum(1 for m in running if (m["asset"], m["minutes"]) == key) >= self.max_rules()
        # What adding a rule to a full market would take out: its weakest one.
        dropped = {}
        best = {}
        for b in board:
            key = (b["asset"], b["minutes"])
            if b["status"] != "proven" or (*key, b["variant"]) in on or b.get("family") == "pair":
                continue
            current = None
            if full(key):
                weakest = self._weakest(running, *key, rows)
                current = rows.get((*key, weakest["variant"]))
                if (current and current["roi_pessimistic"] is not None
                        and (b["roi_pessimistic"] or -1) <= current["roi_pessimistic"]):
                    continue  # no better than the weakest one already running there
                dropped[key] = weakest["variant"]
            if key not in best or (b["oos_pessimistic"] or -1) > (best[key][0]["oos_pessimistic"] or -1):
                best[key] = (b, current)
        changes = [{"asset": b["asset"], "minutes": b["minutes"], "variant": b["variant"], "text": b["rule"],
                    "from_variant": dropped.get((b["asset"], b["minutes"])),
                    "from_text": lab.describe(dropped[(b["asset"], b["minutes"])]) if (b["asset"], b["minutes"]) in dropped else None,
                    "current_roi": cur["roi"] if cur else None,
                    **{k: b[k] for k in ("bets", "roi", "roi_pessimistic", "roi_recent", "oos_bets", "oos_roi",
                                         "oos_pessimistic")}}
                   for b, cur in best.values()]
        changes.sort(key=lambda c: -(c["oos_pessimistic"] or -1))
        need = self.cfg.lab_oos_min_bets
        leaders = [b for b in board if b["status"] in ("candidate", "promising")]
        leader = leaders[0] if leaders else None
        if changes:
            top = changes[0]
            score = int(min(99, 75 + max(0.0, top["oos_pessimistic"] or 0) * 250))
            signals = [
                {"ok": True, "text": f"הוכחה על נתונים חדשים: {top['oos_bets']} הימורים"},
                {"ok": (top["oos_roi"] or 0) > 0, "text": f"תשואה על הנתונים החדשים: {(top['oos_roi'] or 0) * 100:+.1f}%"},
                {"ok": (top["oos_pessimistic"] or 0) > 0, "text": f"גם בתרחיש הפסימי: {(top['oos_pessimistic'] or 0) * 100:+.1f}%"},
                {"ok": top["current_roi"] is None or (top["roi"] or 0) > top["current_roi"],
                 "text": "טובה מהאסטרטגיה הנוכחית בשוק הזה" if top["current_roi"] is not None else "שוק חדש לאסטרטגיה הראשית"},
                {"ok": (top["roi_recent"] or 0) > 0, "text": "רווחית גם לאחרונה"},
            ]
            return {"verdict": "push", "score": score, "changes": changes, "signals": signals, "leader": None,
                    "need": need}
        if leader:
            progress = min(1.0, leader["oos_bets"] / need) if leader["status"] == "candidate" else 0.0
            positive = (leader["oos_roi"] or 0) > 0
            score = int(15 + (45 * progress if positive else 10 * progress)) if leader["status"] == "candidate" else 12
        else:
            progress, positive, score = 0.0, False, 4
        signals = [{"ok": False, "text": "אין עדיין אסטרטגיה מוכחת"}]
        if leader:
            signals.append({"ok": leader["status"] == "candidate",
                            "text": "המובילה בבדיקה על נתונים חדשים" if leader["status"] == "candidate"
                            else "המובילה עוד לא עברה לבדיקה על נתונים חדשים"})
            if leader["status"] == "candidate":
                signals.append({"ok": positive, "text": f"על הנתונים החדשים: {(leader['oos_roi'] or 0) * 100:+.1f}%"
                                                        f" ({leader['oos_bets']} מתוך {need} הימורים)"})
        signals.append({"ok": self.calibration.get("applied", False),
                        "text": "הכיול האוטומטי פעיל" if self.calibration.get("applied") else "הכיול האוטומטי עוד אוסף נתונים"})
        return {"verdict": "wait", "score": score, "changes": [], "signals": signals, "need": need,
                "leader": {k: leader[k] for k in ("asset", "minutes", "variant", "rule", "status", "bets", "roi",
                                                   "oos_bets", "oos_roi")} if leader else None,
                "progress": progress}

    # Which main strategies run in the simulation (the owner's choice). Real money mirrors
    # the simulation's signals, so a strategy that doesn't run here can't trade real money.
    def sim_strategies(self):
        has_stocks = bool(self.stocks and self.stocks.main())
        stored = self._json_setting("sim_strategies", {})
        return {"crypto": bool(stored.get("crypto", True)), "stocks": bool(stored.get("stocks", has_stocks)) and has_stocks}

    def set_sim_strategies(self, crypto, stocks_on, stocks_variant=None):
        if stocks_on and stocks_variant:
            self.push_stocks(stocks_variant)
        if stocks_on and not (self.stocks and self.stocks.main()):
            raise ValueError("צריך לבחור אסטרטגיית מניות")
        self.store.set_setting("sim_strategies", json.dumps({"crypto": bool(crypto), "stocks": bool(stocks_on)}))
        self._apply_sim()
        log.warning("Simulation runs: crypto %s, stocks %s", "on" if crypto else "off", "on" if stocks_on else "off")

    def _apply_sim(self):
        on, limits = self.sim_strategies(), self.sim_limits()
        for t in self.trackers:
            t.sim_on = on["crypto"]
            t.can_open = self.can_open
            if t.sim_stake != limits["stake"]:
                t.sim_stake = limits["stake"]
                t._set_market_cfg(t.market)
        if self.stocks:
            self.stocks.enabled = on["stocks"]
            self.stocks.can_open = self.can_open
            self.stocks.sim_stake = limits["stake"]

    # The simulation's amount and protections (the owner's settings; real money has its own).
    # The protections apply only while "exactly like the lab" is off (see lab_exact).
    def sim_limits(self):
        stored = self._json_setting("sim_limits", {})
        max_open = stored.get("max_open") or {}
        return {"stake": float(stored.get("stake") or self.cfg.stake_usd),
                "daily_loss": float(stored["daily_loss"]) if stored.get("daily_loss") else None,
                # Bitcoin, Ether and Solana move together: same-direction bets in the same window are
                # mostly one bet. Default 1; 0 or empty in the settings means no cap.
                "same_direction": int(stored["same_direction"]) if stored.get("same_direction") is not None else 1,
                "max_open": {n: (int(max_open[n]) if max_open.get(n) else None) for n in ("crypto", "stocks")}}

    def set_sim_limits(self, stake=None, daily_loss=None, max_open=None, same_direction=None):
        limits = self.sim_limits()
        if same_direction is not None:
            limits["same_direction"] = max(0, int(same_direction))
        if stake is not None:
            if not 1 <= float(stake) <= 10_000:
                raise ValueError("סכום לטרייד: בין $1 ל-$10,000")
            limits["stake"] = float(stake)
        if daily_loss is not None:
            limits["daily_loss"] = float(daily_loss) if float(daily_loss) > 0 else None
        if max_open is not None:
            for name, n in max_open.items():
                if name in limits["max_open"]:
                    limits["max_open"][name] = int(n) if n and int(n) > 0 else None
        self.store.set_setting("sim_limits", json.dumps(limits))
        self._apply_sim()

    def can_open(self, strategy, stake, side=None, window=None):
        """Room for one more simulated main-strategy trade: under the strategy's open-trade cap,
        under the same-direction cap for crypto windows, and under today's loss limit (losses
        so far plus everything still at stake). None of it while the simulation runs exactly like
        the lab. The day's loss limit never stops the simulation: passing it shows in red."""
        limits = self.sim_limits()
        exact = self.lab_exact()["sim"]
        which = stats.STRATEGY_SQL[strategy]
        if not exact and strategy == "crypto" and side and window is not None and limits["same_direction"]:
            same = self.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'paper' AND status = 'filled' AND minutes > 0"
                                 " AND window_start = ? AND side = ?", (window, side))[0][0]
            if same >= limits["same_direction"]:
                return False
        cap = None if exact else limits["max_open"].get(strategy)
        if cap is not None:
            open_now = self.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'paper' AND status = 'filled'"
                                     f" AND pnl IS NULL{which}")[0][0]
            if open_now >= cap:
                return False
        # (the day's loss limit doesn't stop the simulation: passing it only shows in red)
        return True

    def strategy_filter(self):
        """Which bets the summaries count: both main strategies, or only the one that runs."""
        on = self.sim_strategies()
        if on["crypto"] and not on["stocks"]:
            return "crypto"
        if on["stocks"] and not on["crypto"]:
            return "stocks"
        return None

    # The stock market's main strategy (pushed from the lab's stocks tab).
    def push_stocks(self, variant):
        if not self.stocks:
            raise ValueError("ניסוי המניות כבוי")
        if variant not in {v.key for v in stocks.VARIANTS}:
            raise ValueError("אסטרטגיה לא מוכרת")
        self._save_version(self.main_strategies(), f"מניות: {stocks.describe(variant)}", "push_stocks",
                           stocks_variant=variant)
        self.stocks.set_main(variant)
        on = self.sim_strategies()
        self.store.set_setting("sim_strategies", json.dumps({"crypto": on["crypto"], "stocks": True}))
        self._apply_sim()
        log.warning("Stock strategy is now %s", variant)

    def remove_stocks(self):
        if self.stocks and self.stocks.main():
            self._save_version(self.main_strategies(), "מניות: הוסרה", "remove_stocks", stocks_variant="")
            self.stocks.set_main(None)

    def mode(self):
        """The whole system follows one mode: real money when Live is on, else the simulation."""
        return "live" if self.live.armed() and not self.live.approval() else "paper"  # (approving: the simulation's view)

    def set_mode(self, live, confirmed=False, keep_daily=False, approval=False):
        if not confirmed:
            raise ValueError("צריך אישור")
        self.live.arm(bool(live), override=True, keep_daily=keep_daily, approval=bool(approval))

    # The money strip, per mode: the amount for each trade, the day's loss limit, and how many
    # entries go into each approved trade. Each has the system's value (from the data) and,
    # when the owner typed one, the owner's, which then rules until "back to the system's values".
    MONEY_KEYS = ("stake", "daily_loss", "copies", "rr")

    def _money_overrides(self):
        s = self._json_setting("money_override", {})
        return {m: {k: v for k, v in (s.get(m) or {}).items() if k in self.MONEY_KEYS and v is not None} for m in ("sim", "live")}

    def _owner_amounts_once(self):
        """The owner's choice of 26.10.2026, applied once: $5 a trade in both modes (the simulation as
        real money, so its numbers are the wallet's). Theirs to change afterwards, each mode on its own."""
        if self.store.get_setting("owner_amounts_20261026"):
            return
        self.store.set_setting("owner_amounts_20261026", "1")
        if not self.store._x("SELECT 1 FROM bets WHERE mode IN ('paper', 'live') LIMIT 1"):
            return  # (a new installation: its own defaults)
        over = self._money_overrides()
        for mode in ("sim", "live"):
            over[mode]["stake"] = 5.0
        self.store.set_setting("money_override", json.dumps(over))
        log.warning("Owner's amounts applied: $5 a trade (simulation and real money)")

    # The owner's choice of 26.10.2026: no layer of filters over the strategies (a price group, the most
    # a round may lose, a least chance, the model's agreement). Each strategy carries its own, tested in
    # the lab like the rest of it. What those filters left in the settings goes, once.
    OLD_FILTERS = ("price_band", "price_band_live", "price_band_proposal", "price_band_dismissed", "max_loss_sim",
                   "max_loss_live", "min_chance_sim", "min_chance_live", "model_agree", "brain_log")

    def _panels_once(self):
        """The expert panels of 27.10.2026, applied once: only proven strategies go into the
        simulation by themselves (no candidates), and no filter changes by itself on a strategy
        under test. Both the owner's to turn back on."""
        if self.store.get_setting("owner_panels_20261027"):
            return
        self.store.set_setting("owner_panels_20261027", "1")
        self.store.set_setting("lab_auto_candidates", "0")
        self.store.set_setting("rr_auto", json.dumps({"sim": False, "live": False}))
        log.warning("Panels' rules applied: no candidates into the simulation, no automatic filters")

    def _filters_into_strategies_once(self):
        if self.store.get_setting("owner_no_filters_20261026"):
            return
        self.store.set_setting("owner_no_filters_20261026", "1")
        for k in self.OLD_FILTERS:
            self.store._x("DELETE FROM settings WHERE key = ?", (k,))
        self.store._x("DELETE FROM settings WHERE key LIKE 'brain\\_%' ESCAPE '\\'")
        log.warning("The filters over the strategies are gone: each strategy carries its own")

    def _migrate_money(self):
        """Before the strip, the amounts in the settings were the owner's: they stay the owner's."""
        if self.store.get_setting("money_override") is not None:
            return
        stored = self._json_setting("sim_limits", {})
        sim = {k: float(stored[k]) for k in ("stake", "daily_loss") if stored.get(k)}
        live = {}
        if self.store.get_setting("live_stake") is not None:
            live["stake"] = float(self.store.get_setting("live_stake"))
        if self.store.get_setting("live_daily_loss_limit") is not None:
            live["daily_loss"] = float(self.store.get_setting("live_daily_loss_limit"))
        self.store.set_setting("money_override", json.dumps({"sim": sim, "live": live}))

    def money_system(self, mode):
        """The system's values: the stake that made the most money on the strategies running there
        (or the default until there is enough data); a day's limit of two full losing rounds of
        everything connected; one entry per trade."""
        live = mode == "live"
        best = (getattr(self, "stake_curve", {}).get(mode) or {}).get("best")  # (none yet while starting up)
        stake = float(best) if best else (self.cfg.live_stake_usd if live else self.cfg.stake_usd)
        if live:
            stake = max(1.0, min(stake, self.cfg.live_max_stake_usd))
        running = max(1, sum(1 for m in self.strategies_for(mode) if not m.get("paused")))
        daily = float(max(10, round(stake * running * 2)))
        return {"stake": stake, "daily_loss": daily, "copies": 1, "rr": None}  # rr None: each strategy's own

    def money(self):
        out, over = {}, self._money_overrides()
        for mode in ("sim", "live"):
            sysv = self.money_system(mode)
            out[mode] = {k: {"value": over[mode].get(k, sysv[k]), "system": sysv[k], "override": k in over[mode]}
                         for k in self.MONEY_KEYS}
        # Today so far: realized losses plus everything still at stake (what the day's limit counts).
        day = period_starts(time.time(), self.cfg.timezone)["day"]
        out["sim"]["today"] = self.store._x(
            "SELECT COALESCE(SUM(CASE WHEN pnl IS NULL THEN cost + fee ELSE -pnl END), 0) FROM bets"
            " WHERE mode = 'paper' AND status = 'filled' AND ts >= ?", (day,))[0][0]
        out["live"]["today"] = self.live.today_exposure()
        return out

    def set_money(self, mode, values, reset=False):
        """The owner's values (a key set to None goes back to the system's); reset: all of them."""
        mode = "live" if mode == "live" else "sim"
        allm = self._money_overrides()
        mine = {} if reset else dict(allm[mode])
        for k, v in (values or {}).items():
            if k not in self.MONEY_KEYS:
                continue
            if v is None or v == "":
                mine.pop(k, None)
                continue
            v = float(v)
            if k == "stake" and not 1 <= v <= (self.cfg.live_max_stake_usd if mode == "live" else 10_000):
                raise ValueError("סכום לטרייד מחוץ לטווח")
            if k == "daily_loss" and not (v == 0 and mode == "sim") and not 1 <= v <= 100_000:
                raise ValueError("הפסד יומי מחוץ לטווח")
            if k == "copies" and not 1 <= v <= 10:
                raise ValueError("כניסות לכל טרייד: בין 1 ל-10")
            if k == "rr" and not 0 <= v <= 3:
                raise ValueError("יחס סיכוי / סיכון: בין $0 ל-$30 על כל $10")
            mine[k] = int(v) if k == "copies" else v
        allm[mode] = mine
        self.store.set_setting("money_override", json.dumps(allm))
        self._apply_money()
        log.warning("Money (%s): %s", mode, "the system's values" if not mine else mine)

    def _apply_money(self):
        M = self.money()
        sim, live = M["sim"], M["live"]
        limits = self.sim_limits()
        limits["stake"] = float(sim["stake"]["value"])
        limits["daily_loss"] = float(sim["daily_loss"]["value"]) if sim["daily_loss"]["value"] else None
        self.store.set_setting("sim_limits", json.dumps(limits))
        self.store.set_setting("live_stake", float(live["stake"]["value"]))
        if live["daily_loss"]["value"]:
            self.store.set_setting("live_daily_loss_limit", float(live["daily_loss"]["value"]))
        for t in self.trackers:
            t.sim_copies, t.live_copies = int(sim["copies"]["value"]), int(live["copies"]["value"])
        self._apply_main(self.main_strategies())  # the strip's risk/reward, if the owner set one
        self._apply_live_main(self.live_main())
        self._apply_sim()

    # Risk/reward research, per strategy (see rrlab.py): on every learning pass each connected
    # strategy is studied on its own trades; where the research proves a better filter, the
    # simulation takes it on its own (the owner can switch that off), real money only when the
    # owner applies it or switches its own automatic choice on. The owner can always set a
    # strategy's filter, or one for a whole mode on the money strip.
    def rr_auto(self):
        s = self._json_setting("rr_auto", {})
        # (off by default: a filter changed by itself while a rule is under test mixes its evidence)
        return {"sim": bool(s.get("sim", False)), "live": bool(s.get("live", False))}

    def set_rr_auto(self, mode, on):
        s = self.rr_auto()
        s["live" if mode == "live" else "sim"] = bool(on)
        self.store.set_setting("rr_auto", json.dumps(s))
        self._apply_rr_research()

    def _study_rr(self, db):
        out = {}
        for mode in ("sim", "live"):
            entries = {(m["asset"], m["minutes"], m["variant"]): float(m.get("ratio") or 0.0) for m in self.strategies_for(mode)}
            out[mode] = rrlab.research(db, set(entries), entries)
        self.rr_research = out

    def _apply_rr_research(self):
        """The automatic part: a strategy whose research proves a better filter gets it, unless the
        owner set its filter or it changed fewer than rrlab.GAP trades ago."""
        auto, notes = self.rr_auto(), []
        for mode in ("sim", "live"):
            if not auto[mode] or self._money_overrides()[mode].get("rr") is not None:
                continue
            main, changed = self.strategies_for(mode), []
            for m in main:
                st = (getattr(self, "rr_research", {}).get(mode) or {}).get((m["asset"], m["minutes"], m["variant"]))
                if not st or st["verdict"] != "change" or m.get("ratio_src") == "owner":
                    continue
                if m.get("ratio_n") is not None and st["n"] - m["ratio_n"] < rrlab.GAP:
                    continue
                before = float(m.get("ratio") or 0.0)
                m.update(ratio=st["recommended"], ratio_src="system", ratio_n=st["n"], ratio_ts=time.time())
                tb, tn = st["test_best"], st["test_now"]
                why = (f"נבחר על {st['train_n']} הטריידים הישנים; על {st['test_n']} החדשים: "
                       f"{(tb['roi'] or 0) * 100:+.1f}% מול {(tn['roi'] or 0) * 100:+.1f}% בלי השינוי")
                changed.append(m)
                notes.append({"mode": mode, "asset": m["asset"], "minutes": m["minutes"], "rule": lab.describe(m["variant"]),
                              "before": before, "after": st["recommended"], "why": why})
            if changed:
                self._save_version(main, "יחס סיכוי / סיכון: " + "; ".join(f"{c['asset'].upper()} {c['minutes']}m" for c in changed),
                                   "ratio", mode=mode)
                self._save_for(mode, main, reset_readiness=False)
        if notes and getattr(self, "notifier", None):
            self.notifier.rr_changed(notes)
            log.warning("Risk/reward research changed %d filters", len(notes))

    def set_strategy_ratio(self, mode, asset, minutes, variant, ratio):
        """The owner's filter for one strategy where it runs (None: back to the research's choice)."""
        mode = "live" if mode == "live" else "sim"
        main = self.strategies_for(mode)
        hit = [m for m in main if (m["asset"], m["minutes"], m["variant"]) == (asset, int(minutes), variant)]
        if not hit:
            raise ValueError("האסטרטגיה הזאת לא מחוברת שם")
        st = (getattr(self, "rr_research", {}).get(mode) or {}).get((asset, int(minutes), variant)) or {}
        for m in hit:
            if ratio is None:
                m.update(ratio=st.get("recommended") or 0.0, ratio_src="system", ratio_n=st.get("n"), ratio_ts=time.time())
            else:
                if not 0 <= float(ratio) <= 3:
                    raise ValueError("יחס סיכוי / סיכון: בין $0 ל-$30 על כל $10")
                m.update(ratio=float(ratio), ratio_src="owner", ratio_ts=time.time())
        self._save_version(main, f"יחס סיכוי / סיכון ({'המערכת' if ratio is None else 'שלך'}): {asset.upper()} {minutes}m",
                           "ratio", mode=mode)
        self._save_for(mode, main, reset_readiness=False)

    # The amount a round, per strategy (see stakes.py): each connected strategy's settled lab trades
    # replayed at each amount against the order book they met; the amount with the most profit
    # (and enough trades) is the system's choice, taken on its own. The mode's amount on the money
    # strip is the ceiling for the system's choice (in real money the owner's amount is the most the
    # system may put on a trade); an amount the owner sets on a strategy is used as set.
    STAKE_GAP = 30  # new trades of a strategy between two changes of its amount

    def _study_stakes(self, db):
        out = {}
        for mode in ("sim", "live"):
            out[mode] = {}
            for m in self.strategies_for(mode):
                key = (m["asset"], m["minutes"], m["variant"])
                if key not in out[mode]:
                    out[mode][key] = stakes.curve(db, self.cfg, {key})
        self.stake_research = out

    # One amount for every trade (the owner's rule, on by default): every strategy trades the mode's
    # amount, not one of its own (the research's or one set for it), so a loss is always the same and
    # a win, inside the price group, always in the same range.
    def uniform_stake(self, mode="sim"):
        return self.store.get_setting(f"uniform_stake_{'live' if mode == 'live' else 'sim'}", "1") == "1"

    def set_uniform_stake(self, mode, on):
        self.store.set_setting(f"uniform_stake_{'live' if mode == 'live' else 'sim'}", "1" if on else "0")
        self._apply_main(self.main_strategies())
        self._apply_live_main(self.live_main())

    def _strategy_stake(self, mode, m):
        """The amount a round this strategy trades where it runs (0: the mode's amount)."""
        if self.uniform_stake(mode):
            return 0.0
        stake = float(m.get("stake") or 0.0)
        if not stake or m.get("stake_src") == "owner":
            return stake
        # The mode's amount (the owner's on the money strip, or the system's). Not money(): this
        # runs while the engine starts, before what money() reads is there.
        over = self._money_overrides()[mode].get("stake")
        ceiling = float(over) if over is not None else float(self.money_system(mode)["stake"] or 0.0)
        return min(stake, ceiling) if ceiling else stake

    def _apply_stake_research(self):
        for mode in ("sim", "live"):
            main, changed = self.strategies_for(mode), []
            for m in main:
                st = (getattr(self, "stake_research", {}).get(mode) or {}).get((m["asset"], m["minutes"], m["variant"]))
                if not st or st["best"] is None or m.get("stake_src") == "owner":
                    continue
                if m.get("stake") and m.get("stake_n") is not None and st["trades"] - m["stake_n"] < self.STAKE_GAP:
                    continue
                if float(m.get("stake") or 0) == float(st["best"]):
                    continue
                m.update(stake=float(st["best"]), stake_src="system", stake_n=st["trades"], stake_ts=time.time())
                changed.append(m)
            if changed:
                self._save_version(main, "סכום לטרייד (המערכת): " + "; ".join(
                    f"{c['asset'].upper()} {c['minutes']}m ${c['stake']:g}" for c in changed), "stake", mode=mode)
                self._save_for(mode, main, reset_readiness=False)
                log.warning("Stake research (%s) set %d amounts", mode, len(changed))

    def set_strategy_stake(self, mode, asset, minutes, variant, stake):
        """The owner's amount a round for one strategy where it runs (None: back to the system's)."""
        mode = "live" if mode == "live" else "sim"
        main = self.strategies_for(mode)
        hit = [m for m in main if (m["asset"], m["minutes"], m["variant"]) == (asset, int(minutes), variant)]
        if not hit:
            raise ValueError("האסטרטגיה הזאת לא מחוברת שם")
        st = (getattr(self, "stake_research", {}).get(mode) or {}).get((asset, int(minutes), variant)) or {}
        for m in hit:
            if stake is None:
                m.update(stake=float(st.get("best") or 0.0), stake_src="system", stake_n=st.get("trades"), stake_ts=time.time())
            else:
                most = self.cfg.live_max_stake_usd if mode == "live" else 10_000
                if not 1 <= float(stake) <= most:
                    raise ValueError(f"סכום לטרייד: בין $1 ל-${most:g}")
                m.update(stake=float(stake), stake_src="owner", stake_ts=time.time())
        self._save_version(main, f"סכום לטרייד ({'המערכת' if stake is None else 'שלך'}): {asset.upper()} {minutes}m",
                           "stake", mode=mode)
        self._save_for(mode, main, reset_readiness=False)

    def stake_view(self):
        """For the page: each connected strategy's amount study and the amount in force, by mode and key."""
        R = getattr(self, "stake_research", {}) or {}
        out = {}
        for mode in ("sim", "live"):
            ceiling = float(self.money()[mode]["stake"]["value"] or 0.0)
            out[mode] = {}
            for m in self.strategies_for(mode):
                key = (m["asset"], m["minutes"], m["variant"])
                out[mode][_key(*key)] = {"study": (R.get(mode) or {}).get(key), "now": self._strategy_stake(mode, m) or ceiling,
                                         "src": m.get("stake_src") if m.get("stake") else None, "ceiling": ceiling}
        return out

    def rr_view(self):
        """For the page: each connected strategy's study, by mode and key."""
        R = getattr(self, "rr_research", {}) or {}
        return {mode: {_key(a, m, v): st for (a, m, v), st in (R.get(mode) or {}).items()} for mode in ("sim", "live")}

    # The protections a mode adds on top of the lab's rules, while "exactly like the lab" is off:
    # the mix with the market price, the caps on same-direction and open trades, a daily loss limit.
    def set_protections(self, mode, same_direction=None, max_open=None, daily_loss=None, blend=None):
        if mode == "live":
            if max_open is not None:
                self.live.set_max_open(max_open)
            if same_direction is not None:
                self.store.set_setting("live_same_direction", str(max(0, int(same_direction))))
        else:
            self.set_sim_limits(max_open=max_open, same_direction=same_direction)
        if daily_loss is not None:
            self.set_money(mode, {"daily_loss": daily_loss})
        if blend is not None:
            self.set_blend_use(mode, blend)

    def save_trading(self, mode, body):
        """A trading tab of the settings (simulation or real money), saved as one: what runs, the
        amount, and the protections."""
        stocks_variant = str(body["stocks_variant"]) if body.get("stocks_variant") else None
        if mode == "sim":
            self.set_sim_strategies(bool(body.get("crypto")), bool(body.get("stocks")), stocks_variant)
            if body.get("stake"):
                self.set_money("sim", {"stake": body["stake"]})
        else:
            if stocks_variant:
                self.push_stocks(stocks_variant)
            if body.get("stake"):
                self.set_money("live", {"stake": body["stake"]})
            self.live.set_strategy("crypto", bool(body.get("crypto")), confirmed=True)
            self.live.set_strategy("stocks", bool(body.get("stocks")), confirmed=True)
            # Real money mirrors the simulation: what goes live also runs there.
            on = self.sim_strategies()
            self.set_sim_strategies(on["crypto"] or bool(body.get("crypto")), on["stocks"] or bool(body.get("stocks")))
        P = body.get("protections")
        if P:
            self.set_protections(mode, same_direction=P.get("same_direction"), max_open=P.get("max_open"),
                                 daily_loss=P.get("daily_loss"), blend=P.get("blend"))

    # Operating costs: hosting is a fixed monthly figure the owner sets; Claude is measured.
    def operating_costs(self):
        return {"hosting": float(self._json_setting("operating_costs", {}).get("hosting", self.cfg.hosting_usd_month))}

    def set_operating_costs(self, hosting):
        hosting = float(hosting)
        if not 0 <= hosting <= 10000:
            raise ValueError("סכום לא תקין")
        self.store.set_setting("operating_costs", json.dumps({"hosting": hosting}))

    def costs_since(self, db, since, now):
        """Operating costs over a period: hosting pro rata (a 30-day month), Claude as spent."""
        hosting = self.operating_costs()["hosting"] * max(0.0, now - since) / (30 * 86400)
        claude = research.spent_since(db, since)
        return {"hosting": hosting, "claude": claude, "total": hosting + claude}

    def _save_main(self, main, reset_readiness=True):
        self.store.set_setting("main_strategies", json.dumps(main))
        # Readiness for real money is judged afresh on the new main strategy's own bets.
        if reset_readiness:
            self.store.set_setting("main_changed_at", str(time.time()))
        self._apply_main(main)
        log.warning("Main strategy is now: %s", ", ".join(f"{m['asset']}-{m['minutes']}m {m['variant']}" for m in main))
        self._refresh_lab_soon()

    # The lab's variants: the fixed grid plus what the search near winners added.
    def _owner_keys(self):
        """Every strategy connected to the owner (the simulation or real money, paused included)."""
        return {(m["asset"], m["minutes"], m["variant"]) for mode in ("sim", "live") for m in self.strategies_for(mode)}

    def _lab_variants(self, asset, minutes, every_time=False):
        """What the lab runs on a market: the fixed grid, what its search added, and every
        strategy connected to the owner there, so the lab keeps measuring what the owner runs.
        Only the entry times the owner picked (lab_times), if any: those strategies keep trading in
        the simulation and with real money, but the lab doesn't run them meanwhile."""
        extra = [k for k in self._json_setting("lab_dynamic", {}).get(_key(asset, minutes), [])
                 if not k.startswith("lotto")]  # (the lottery tickets left the lab)
        mine = sorted(v for a, m, v in self._owner_keys() if (a, m) == (asset, minutes))
        refining = self._refine_keys(asset, minutes)  # (the workshop's: always, whatever the entry times picked)
        refining += self._bridge_keys(asset, minutes)  # (and the research's clones, the same way)
        base = lab.variants_for(minutes)
        times = None if every_time else self.lab_times()
        if times is not None:  # (the owner's pick holds for every strategy in the lab, theirs too)
            base = [v for v in base if lab.time_group(v.key) in times]
            extra = [k for k in extra if lab.time_group(k) in times]
            mine = [k for k in mine if lab.time_group(k) in times]
        # The maker check's findings, as makers of their own (zones), on the markets long enough for them.
        zones = [k for k in self._json_setting("lab_zones", []) if lab.Variant.from_key(k).max_seconds_left <= minutes * 60 - 30]
        if times is not None:
            zones = [k for k in zones if lab.time_group(k) in times]
        base = base + [lab.Variant.from_key(k) for k in zones]
        keys = {v.key for v in base}
        out = base + [lab.Variant.from_key(k) for k in extra if k not in keys]
        keys |= set(extra)
        out += [lab.Variant.from_key(k) for k in mine if k not in keys]
        keys |= set(mine)
        out += [lab.Variant.from_key(k) for k in dict.fromkeys(refining) if k not in keys]
        keys |= set(refining)
        # Every maker twice: filled when sellers break through it, and by its place in the queue ("|q").
        twins = [dataclasses.replace(v, fill="q") for v in out if lab.is_maker(v.key) and v.fill != "q"]
        return out + [v for v in twins if v.key not in keys]

    # The entry times the lab runs on: the system's choice (all of them) unless the owner picked some.
    def lab_times(self):
        """The owner's entry times (a set of lab.TIME_GROUPS keys), or None: the system's (all)."""
        picked = self._json_setting("lab_times", None)
        groups = {g for g, _, _ in lab.TIME_GROUPS}
        picked = {g for g in picked or () if g in groups}
        return picked or None

    def set_lab_times(self, picked):
        """picked: the entry times to run on; None, empty or all of them: back to the system's choice."""
        groups = [g for g, _, _ in lab.TIME_GROUPS]
        picked = [g for g in groups if g in set(picked or ())]
        self.store.set_setting("lab_times", json.dumps(picked) if picked and len(picked) < len(groups) else None)
        log.warning("Lab entry times: %s", ", ".join(picked) if picked and len(picked) < len(groups) else "all (the system's)")
        if getattr(self, "current_variants", None) is not None:
            self._refresh_lab_variants()
        self._refresh_lab_soon()

    def lab_times_view(self):
        times, now = self.lab_times(), time.time()
        cached = getattr(self, "_times_count", None)
        if cached is None or now - cached[0] > self.MIX_GAP:  # (how many a time has, counted twice a minute)
            every = [v.key for t in self.trackers for v in self._lab_variants(t.asset, t.minutes, every_time=True)] if self.with_lab else []
            self._times_count = cached = (now, collections.Counter(lab.time_group(k) for k in every))
        return {"picked": sorted(times) if times else None,
                "groups": [{"key": g, "label": label, "on": times is None or g in times,
                            "variants": cached[1].get(g, 0)}
                           for g, label, _ in lab.TIME_GROUPS]}

    def _refresh_lab_variants(self):
        if not self.with_lab:
            return
        for t in self.trackers:
            t.set_variants(self._lab_variants(t.asset, t.minutes))
        self._refresh_variant_set()

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
        mine = self._owner_keys()
        for (asset, minutes, variant) in list(self.dynamic_variants):
            b = by_key.get((asset, minutes, variant))
            if b and b["status"] in ("losing", "failed") and (asset, minutes, variant) not in mine:  # never the owner's
                dyn[_key(asset, minutes)].remove(variant)
                retired = self._json_setting("lab_retired", [])
                self.store.set_setting("lab_retired", json.dumps(retired + [_key(asset, minutes, variant)]))
                changed = True
        total = sum(1 for vs in dyn.values() for k in vs if "|r" not in k)
        leaders = [b for b in board if b["status"] in ("proven", "candidate", "promising")
                   and b["bets"] >= self.cfg.lab_min_bets]
        leaders.sort(key=lambda b: -(b["roi_pessimistic"] if b["roi_pessimistic"] is not None else b["roi"]))
        for b in leaders:
            market = _key(b["asset"], b["minutes"])
            have = {v for (a, m, v) in self.current_variants if (a, m) == (b["asset"], b["minutes"])}
            times = self.lab_times()
            for near in lab.neighbours(b["variant"], b["minutes"]):
                if near in have or (times is not None and lab.time_group(near) not in times):
                    continue
                if total >= self.cfg.explore_total or len(dyn.get(market, [])) >= self.cfg.explore_per_market:
                    break
                dyn.setdefault(market, []).append(near)
                have.add(near)
                total += 1
                changed = True
        # Risk/reward versions ($3 / $5 / $7 for every $10) of the owner's strategies and the leaders,
        # on their own budget. One retired for losing isn't brought back.
        retired = set(self._json_setting("lab_retired", []))
        ratio_total = sum(1 for vs in dyn.values() for k in vs if "|r" in k)
        own = [(m["asset"], m["minutes"], m["variant"]) for mode in ("sim", "live") for m in self.strategies_for(mode)]
        tracked = {(t.asset, t.minutes) for t in self.trackers}
        for asset, minutes, variant in own + [(b["asset"], b["minutes"], b["variant"]) for b in leaders]:
            if (asset, minutes) not in tracked:
                continue
            market = _key(asset, minutes)
            have = {v for (a, m, v) in self.current_variants if (a, m) == (asset, minutes)} | set(dyn.get(market, []))
            for rv in lab.ratio_versions(variant):
                if rv in have or _key(asset, minutes, rv) in retired:
                    continue
                if ratio_total >= self.cfg.explore_ratio_total:
                    break
                dyn.setdefault(market, []).append(rv)
                ratio_total += 1
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
            self.market_blend = calibration.market_blend(db, self.cfg)
            self.analysis = dict(stats.analysis(db, self.cfg, markets), calibration=stats.calibration(db),
                                 auto_calibration=self.calibration)
            self.stake_curve = {mode: stakes.curve(db, self.cfg, self._active_keys(mode)) for mode in ("sim", "live")}
            self._study_rr(db)
            self._study_stakes(db)
            counts = dict(db.execute("SELECT mode, COUNT(*) FROM bets WHERE winner IS NOT NULL GROUP BY mode").fetchall())
            last = db.execute("SELECT MAX(window_start + minutes * 60) FROM bets WHERE winner IS NOT NULL AND minutes > 0").fetchone()[0]
            self.learning["analysis"] = {"ts": time.time(), "settled": counts, "last_resolved": last}
        finally:
            db.close()
        self._apply_rr_research()
        self._apply_stake_research()
        self._apply_money()
        for t in self.trackers:
            t.calib_k = self.calibration["k"]
            t.blend = (self.market_blend["a"], self.market_blend["b"]) if self.market_blend["applied"] else None
        self._apply_blend_use()
        self.store.prune_ticks(time.time() - self.cfg.keep_ticks_days * 86400)

    def _lab_board(self, candidates):
        """lab.board_from: on the running server in a process of its own (it has the processor to
        itself there, and pages answer meanwhile); in tests, or if that process fails, right here."""
        args = (self.cfg.db_path, self.cfg, set(self.current_variants) if self.current_variants is not None else None,
                candidates, set(self.dynamic_variants), self.lab_since(), self.real_fills_since,
                period_starts(time.time(), self.cfg.timezone))
        if self.streaming:
            try:
                if getattr(self, "_lab_proc", None) is None:
                    import concurrent.futures
                    import multiprocessing
                    # (behind the server: it yields the processor; a new one each pass: its memory goes back)
                    self._lab_proc = concurrent.futures.ProcessPoolExecutor(
                        1, mp_context=multiprocessing.get_context("spawn"), initializer=os.nice, initargs=(10,),
                        max_tasks_per_child=1)
                return self._lab_proc.submit(lab.board_from, *args).result(timeout=600)
            except Exception as e:
                log.warning("lab board in its own process: %s; counted here", e)
                proc, self._lab_proc = getattr(self, "_lab_proc", None), None
                if proc is not None:
                    proc.shutdown(wait=False, cancel_futures=True)
        return lab.board_from(*args)

    def refresh_lab(self):
        """Every few minutes: the lab ranking (with its stages and search) and the readiness checks."""
        now = time.time()
        candidates = {tuple(k.split("|", 2)): ts for k, ts in self._json_setting("lab_candidates", {}).items()}
        candidates = {(a, int(m), v): ts for (a, m, v), ts in candidates.items()}
        self._lab_at = now
        board = self._lab_board(candidates)
        db = connect(self.cfg.db_path)
        try:
            fresh = [b for b in board["board"] if b["new_candidate"]]
            if fresh:
                stored = self._json_setting("lab_candidates", {})
                for b in fresh:
                    stored[_key(b["asset"], b["minutes"], b["variant"])] = now
                    b["candidate_since"], b["new_candidate"] = now, False
                self.store.set_setting("lab_candidates", json.dumps(stored))
                log.warning("Lab: %d strategies became candidates; they are now tested on new data", len(fresh))
            self.lab = board
            self._set_good(board["board"])
            self.learning["lab"] = {"ts": time.time(), "rules": len(board["board"])}
            self._on_proven(board["board"])
            self._on_candidates(fresh)
            first = self.recording_since(db)
            last = db.execute("SELECT MAX(ts) FROM ticks").fetchone()[0]
            hours = (last - first) / 3600 if first and last else 0.0
            live_rules = {(m["asset"], m["minutes"], m["variant"]) for m in self.live_main() if not m.get("paused")}
            if live_rules == {(m["asset"], m["minutes"], m["variant"]) for m in self.main_strategies() if not m.get("paused")}:
                since = float(self.store.get_setting("main_changed_at", "0"))
                summary = stats.bet_summary(db, "paper", since=since, strategy="crypto")
            else:
                # Real money runs a rule the simulation doesn't: judged on that rule's own lab bets.
                since = float(self.store.get_setting("live_changed_at", "0"))
                summary = stats.bet_summary(db, "lab" if self.with_lab else "paper", since=since,
                                            strategy="crypto", keys=live_rules)
            readiness = stats.readiness(self.cfg, hours, stats.feed_agreement(db, self.cfg.resolution), summary)
            self.readiness = dict(readiness, computed_at=now, since=since)
            self.live.set_readiness(self.readiness["ready"])
            if self.stocks:
                stock_ready = self.stocks.readiness(db, now)
                self.live.set_strategy_readiness("stocks", bool(stock_ready and stock_ready["ready"]))
        finally:
            db.close()
        self.explore(board["board"], now)

    def _active_keys(self, mode):
        """The strategies running (not paused) in the simulation or with real money."""
        rules = self.live_main() if mode == "live" else self.main_strategies()
        return {(m["asset"], m["minutes"], m["variant"]) for m in rules if not m.get("paused")}

    def pulse(self, now=None):
        """The live, light view the dashboard asks for every second: how fresh the data is, how
        fast the system reacts, what it learned last, and each market's chance right now."""
        now = now or time.time()
        lat = [s for ts, s in list(self.health.latency) if ts > now - 60]
        ordered = sorted(lat)
        q = lambda f: ordered[min(len(ordered) - 1, int(f * len(ordered)))] if ordered else None
        snaps = [(t, t.snapshot or {}) for t in self.main_trackers]
        markets = [{"asset": t.asset, "minutes": t.minutes, "seconds_left": s.get("seconds_left"),
                    "fair_up": s.get("fair_up"), "up_ask": s.get("up_ask"), "down_ask": s.get("down_ask"),
                    "edge": s.get("edge"), "edge_side": s.get("edge_side"), "edge_need": s.get("edge_need"),
                    "depth_usd": s.get("depth_usd"), "price_age": s.get("price_age"), "book_age": s.get("book_age")}
                   for t, s in snaps]
        last_price = self.prices.last_price if self.prices else None
        return {"now": now,
                "price_age": now - last_price if last_price else None,
                "book_age": max((m["book_age"] for m in markets if m["book_age"] is not None), default=None),
                "reaction": {"p50": q(0.5), "p95": q(0.95), "per_second": len(lat) / 60},
                "learning": self.learning,
                "stake": {mode: {"best": c.get("best"), "reason": c.get("reason"), "trades": c.get("trades")}
                          for mode, c in self.stake_curve.items()},
                "markets": markets}

    def recording_since(self, db):
        """When the recording began. Kept as a setting, so it survives moving to another server
        (only the last day of the price recording travels with an export)."""
        stored = float(self.store.get_setting("recording_since") or 0) or None
        first = db.execute("SELECT MIN(ts) FROM ticks").fetchone()[0]
        since = min(x for x in (stored, first) if x) if (stored or first) else None
        if since and since != stored:
            self.store.set_setting("recording_since", str(since))
        return since

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
                s = stats.bet_summary(db, mode, since=since, strategy=self.strategy_filter())
                s.pop("rows")
                trades[mode] = s
            out[period] = {"since": since, "trades": trades, "costs": self.costs_since(db, since, time.time()),
                           "strategies": self._strategy_mix(db, period, since)}
        return {
            "periods": out,
            "strategies_total": sum(len(t.variants) for t in self.trackers) + 1,
            "thresholds": {"min_bets": self.cfg.dash_min_bets, "hot": self.cfg.dash_hot_roi,
                           "losing": self.cfg.dash_losing_roi},
            "timezone": self.cfg.timezone,
        }

    MIX_GAP = 30  # seconds between counts of the lab's strategies on the dashboard

    def _strategy_mix(self, db, period, since):
        """The lab's strategies by how they did. It reads every settled lab trade (seconds, with millions),
        so on the running server it is counted with the lab's table, in its process (lab.board_from): the
        page takes that count, a few minutes old at most; a period that just began (midnight) shows the
        last count until the next one. In tests (and before the first count), here, every MIX_GAP."""
        counted = ((self.lab or {}).get("mix") or {}).get(period)
        if counted and (counted[0] == since or self.streaming):
            return counted[1]
        if self.streaming:
            return dict(stats.strategy_mix_empty(), counting=True)  # (the first count after a start: a minute or so)
        hit = self._mix_cache.get(period)
        if hit is None or hit[0] != since or time.time() - hit[1] > self.MIX_GAP:
            hit = self._mix_cache[period] = (since, time.time(), stats.strategy_mix(db, self.cfg, since, self.current_variants))
        return hit[2]

    def _tickers(self):
        """The dashboard's strip: Bitcoin, Ether and Solana now, with their path through the current window."""
        out, seen = [], set()
        for t in sorted(self.trackers, key=lambda t: (t.asset not in self.cfg.assets, -t.minutes)):
            if t.asset not in self.cfg.assets or t.asset in seen or t.window is None:
                continue
            seen.add(t.asset)
            start, length = t.window, t.window_seconds
            path = [(ts, p) for ts, p in list(t.vol.samples) if start - 5 <= ts <= start + length]
            step = max(1, len(path) // 120)
            # Joined mid-window (a restart): no opening price yet, so the first price seen stands in.
            ref = t.open_price if t.open_price is not None else (path[0][1] if path else None)
            out.append({"asset": t.asset, "minutes": t.minutes, "window_start": start, "end": start + length,
                        "price": t.snapshot.get("price") or t.price, "open_price": ref,
                        "ref_is_open": t.open_price is not None,
                        "series": [(round(ts - start, 1), p) for ts, p in path[::step]]})
        return out

    def _stocks_main_state(self, db, now, board):
        if not self.stocks:
            return None
        main = self.stocks.main()
        out = {"main": dict(main, text=stocks.describe(main["variant"])) if main else None,
               "readiness": self.stocks.readiness(db, now),
               "recommendation": self.stocks.recommendation(board)}
        self._stocks_auto_push(board)
        if main:
            summary = stats.bet_summary(db, "paper", since=main["since"], strategy="stocks")
            summary.pop("rows")
            out["summary"] = summary
        return out

    def _positions(self, db, now):
        """Open paper and lab bets, grouped into one card per market window, with what
        Polymarket shows for it right now: both prices, the price to beat, the live
        price and its path through the window, and our entry."""
        rows = db.execute(
            "SELECT mode, variant, ts, asset, minutes, window_start, side, seconds_left, fair_prob, avg_price,"
            " shares, cost + fee AS paid, market FROM bets WHERE mode IN ('paper', 'lab', 'live', 'stocks')"
            " AND status = 'filled' AND pnl IS NULL AND ts > ? ORDER BY ts",
            (now - 24 * 3600,),
        ).fetchall()
        trackers = {(t.asset, t.minutes): t for t in self.trackers}
        book = self.books.get if self.books else (lambda token: None)
        cards = {}
        for r in rows:
            if r["minutes"] == 0:  # the stock markets: one card per market
                if not self.stocks or not r["market"]:
                    continue
                key = (r["mode"], r["market"])
                if key not in cards:
                    cards[key] = self.stocks.card(r["mode"], r["market"], r["asset"], r["window_start"], now, book)
                cards[key]["bets"].append({k: r[k] for k in ("variant", "ts", "side", "seconds_left", "fair_prob",
                                                             "avg_price", "shares", "paid")})
                continue
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
                if card.get("open_price") is None:
                    card.update(open_price=self._open_guess(t, start, path), open_estimated=True)
                cards[key] = card
            card["bets"].append({k: r[k] for k in ("variant", "ts", "side", "seconds_left", "fair_prob",
                                                   "avg_price", "shares", "paid")})
        out = {"paper": [], "lab": [], "live": [], "stocks": []}
        for card in sorted(cards.values(), key=lambda c: (not c["live"], c["end"], c["asset"])):
            for b in card["bets"]:
                v = b["variant"] or "main"
                if v.startswith("stocks:"):
                    b["rule"] = stocks.describe(v.split(":", 1)[1])
                elif card["minutes"] == 0:
                    b["rule"] = stocks.describe(v) if "|" in v else v
                elif v != "main" and "|" in v:
                    b["rule"] = lab.describe(v)
                elif v == self.MANUAL:
                    b["rule"] = "הוספה ידנית שלך"
            out[card["mode"]].append(card)
        return out

    @staticmethod
    def _open_guess(t, start, path):
        """For a round whose opening average wasn't measured (the bot restarted in it), so its card
        still has a price to beat and a chart: Polymarket's own price to beat, else the average
        from the prices recorded before the restart, else the first price seen. Shown only, never
        traded on (the rules don't bet in such a round)."""
        guesses = t.open_guesses
        if start not in guesses:
            market = t.market if t.window == start else None
            guess = ((market or {}).get("price_to_beat") or t._recorded_twap(start - t.cfg.twap_seconds, start)
                     or (path[0][1] if path else None))
            if guess is None:
                return None
            if len(guesses) > 50:
                guesses.clear()
            guesses[start] = guess
        return guesses[start]

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

    def import_data(self, f):
        """An export ZIP from another server replaces this one's data (see export.read)."""
        return export.read(self.cfg, f, replace=True)

    def export(self, out, full=False):
        """Everything recorded, as a ZIP for research (see export.py)."""
        return export.write(self.cfg, out, version=VERSION, full=full,
                            extra={"simulation_strategy": self.main_strategies(), "real_money_strategy": self.live_main()})

    # "Exactly like the lab": a strategy trades in the simulation (and, if the owner switches it on,
    # with real money) just as it did in the lab, so what the lab proved is what runs. No mix with
    # the market price, no risk/reward filter, no caps on open or same-direction trades and no
    # daily loss limit; only the stake is the owner's. On by default for the simulation, off by
    # default for real money, where those protections show in its tab with their values.
    def lab_exact(self):
        s = self._json_setting("lab_exact", {})
        out = {"sim": bool(s.get("sim", True)), "live": bool(s.get("live", False))}
        if self.live_mirror():  # (real money as the simulation: its switch is the simulation's)
            out["live"] = out["sim"]
        return out

    # Real money as the simulation (the owner, 29.10.2026: "the simulation makes money; real money should do
    # exactly what it does"): a strategy that runs in both trades with real money under the simulation's own
    # rules: its filter, "exactly like the lab", the mix with the market price. Only the owner's money stays
    # his own: the amount, the day's loss limit.
    def live_mirror(self):
        return self.store.get_setting("live_mirror", "1") == "1"

    def set_live_mirror(self, on):
        self.store.set_setting("live_mirror", "1" if on else "0")
        self.live.exact = self.lab_exact()["live"]
        self._apply_blend_use()
        self._apply_live_main(self.live_main())
        log.warning("Real money as the simulation: %s", "on" if on else "off")
        return {"live_mirror": self.live_mirror()}

    def set_lab_exact(self, mode, on):
        raw = self._json_setting("lab_exact", {})  # (the owner's own values: real money's kept while it mirrors)
        s = {"sim": bool(raw.get("sim", True)), "live": bool(raw.get("live", False))}
        s["live" if mode == "live" else "sim"] = bool(on)
        self.store.set_setting("lab_exact", json.dumps(s))
        s = self.lab_exact()
        self.live.exact = s["live"]
        self._apply_live_main(self.live_main())
        self._apply_blend_use()
        self._apply_sim()
        log.warning("Exactly like the lab: simulation %s, real money %s", s["sim"], s["live"])

    def protections(self):
        """What each mode adds on top of the lab's own rules, with its value now (shown in the
        trading tabs, so nothing is applied out of sight)."""
        limits, use = self.sim_limits(), self.blend_use()
        return {"sim": {"same_direction": limits["same_direction"], "max_open": limits["max_open"],
                        "daily_loss": limits["daily_loss"], "blend": use["sim"]},
                "live": {"same_direction": self.live.same_direction(), "max_open": self.live.max_open(),
                         "daily_loss": self.live.daily_loss_limit(), "blend": use["live"]},
                "blend_learned": bool(self.market_blend.get("applied")), "exact": self.lab_exact(), "live_mirror": self.live_mirror()}

    # Whether the learned mix with the market price drives the simulation's and real money's decisions.
    def blend_use(self):
        s = self._json_setting("blend_use", {})
        out = {"sim": bool(s.get("sim", True)), "live": bool(s.get("live", True))}
        if self.live_mirror():
            out["live"] = out["sim"]
        return out

    def set_blend_use(self, mode, on):
        s = self.blend_use()
        s["live" if mode == "live" else "sim"] = bool(on)
        self.store.set_setting("blend_use", json.dumps(s))
        self._apply_blend_use()

    def _apply_blend_use(self):
        use, exact = self.blend_use(), self.lab_exact()
        use = {m: use[m] and not exact[m] for m in use}  # the lab decides on our model alone
        for t in self.trackers:
            t.blend_sim, t.blend_live = use["sim"], use["live"]

    def _on_skip(self, tracker, rule, reason):
        key = (tracker.asset, tracker.minutes, tracker.window, rule.key, reason)
        if key in self._skip_seen:
            return  # once per window, rule and reason
        if len(self._skip_seen) > 50000:
            self._skip_seen.clear()
        self._skip_seen.add(key)
        self.skips.append((time.time(), tracker.asset, tracker.minutes, rule.key, reason))

    def why_no_trades(self, db, now, hours=1.0):
        """The last hour of the simulation's main strategies: rounds that closed, trades made, and
        the times a rule had an edge but didn't bet, by reason."""
        since = now - hours * 3600
        markets = {(t.asset, t.minutes) for t in self.main_trackers}
        rounds = sum(1 for a, m, s in db.execute(
            "SELECT asset, minutes, window_start FROM windows WHERE window_start + minutes * 60 BETWEEN ? AND ?",
            (since, now)).fetchall() if (a, m) in markets)
        trades = db.execute("SELECT COUNT(*) FROM bets WHERE mode = 'paper' AND status = 'filled' AND minutes > 0 AND ts >= ?",
                            (since,)).fetchone()[0]
        reasons = {"full": 0, "limits": 0}
        for ts, asset, minutes, rule, reason in list(self.skips):
            if ts >= since and reason in reasons:
                reasons[reason] += 1
        return {"hours": hours, "rounds": rounds, "rules": sum(len(t.main_rules) for t in self.main_trackers),
                "trades": trades, "skipped": reasons, "stake": self.sim_limits()["stake"],
                "sim_on": self.sim_strategies()["crypto"]}

    def _trade_interval(self, db, now):
        """The average time between trades entered (the mode in use, the strategies that run),
        over the last 50 trades of the past week. None until there are a few."""
        ts = [r[0] for r in db.execute(
            "SELECT ts FROM bets WHERE mode = ? AND status = 'filled' AND ts >= ?" + stats.STRATEGY_SQL[self.strategy_filter()]
            + " ORDER BY ts DESC LIMIT 50", (self.mode(), now - 7 * 86400)).fetchall()]
        if len(ts) < 3:
            return None
        return {"seconds": (ts[0] - ts[-1]) / (len(ts) - 1), "trades": len(ts)}

    def state(self):
        """Everything the dashboard shows, as one JSON-ready dict."""
        now = time.time()
        db = connect(self.cfg.db_path)
        try:
            which = self.strategy_filter()
            paper = stats.bet_summary(db, "paper", since=self.count_since().get("sim"), strategy=which)
            paper_all = stats.bet_summary(db, "paper", strategy=which)  # (the trade count: always all of them)
            live_total = stats.bet_summary(db, "live", strategy=which)
            starts = period_starts(now, self.cfg.timezone)
            live_today = stats.bet_summary(db, "live", since=starts["day"], strategy=which)
            paper_today = stats.bet_summary(db, "paper", since=starts["day"], strategy=which)
            for s in (paper, paper_all, live_total, live_today, paper_today):
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
            stock_state = self.stocks.state(db, now) if self.stocks else None
            return {
                "now": now,
                "uptime": now - self.started_at,
                "streaming": self.streaming,
                "version": VERSION,
                "commit": COMMIT,
                "alerts": self._alerts(now),
                "control": self.health.report(now),
                "health": {
                    "data_since": self.recording_since(db),
                    "price_reconnects": self.prices.reconnects if self.prices else 0,
                    "book_reconnects": self.books.reconnects if self.books else 0,
                    "last_stream_error": (self.prices.last_error if self.prices else None),
                },
                "config": {"assets": list(self.cfg.assets), "window_minutes": self.cfg.window_minutes,
                           "min_edge": self.cfg.min_edge, "stake": self.cfg.stake_usd,
                           "min_seconds_left": self.cfg.min_seconds_left,
                           "max_seconds_left": self.cfg.max_seconds_left, "resolution": self.cfg.resolution,
                           "fee_rate": self.cfg.fee_rate, "fee_exponent": self.cfg.fee_exponent},
                "sources": sources,
                "errors": self.errors,
                "assets": [t.snapshot for t in self.main_trackers],
                "tickers": self._tickers(),
                "main": [dict(m, rule=lab.describe(m["variant"])) for m in self.main_strategies()],
                "mode": self.mode(),
                "trade_interval": self._trade_interval(db, now),
                "why_no_trades": self.why_no_trades(db, now),
                "last_trade_ts": db.execute("SELECT MAX(ts) FROM bets WHERE mode = ? AND status = 'filled'"
                                            + stats.STRATEGY_SQL[self.strategy_filter()],
                                            (self.mode(),)).fetchone()[0],
                "recommendation": self.recommendation(),
                "live_recommendation": self.recommendation("live"),
                "live_main": [dict(m, rule=lab.describe(m["variant"])) for m in self.live_main()], "refine": self.refine_view(), "consensus": self.consensus_view(),
                "versions": {"list": self.versions_view("sim"), "current": self.current_version()},
                "live_versions": {"list": self.versions_view("live"), "current": self.current_version("live")},
                "auto_push": {"sim": self.auto_push(), "live": self.auto_push("live")},
                "uniform_stake": {"sim": self.uniform_stake("sim"), "live": self.uniform_stake("live")},
                "auto_candidates": self.auto_candidates(),
                "lab_times": self.lab_times_view(),
                "approval_mirror": self.mirrors(),
                "lagwatch": self.lagwatch_view(),
                "hub": self.hub_view(), "streaks": self.streaks, "predict": self.predict,
                "research_hidden": self.research_hidden(),
                "auto_copy": {k: dict(v, why=v.get("why") or self.__dict__.get("_auto_why", {}).get(k)) for k, v in self.auto_copies().items()},
                "strategy_names": self.strategy_names(),
                "strategy_results": self.strategy_results(db),
                "strategy_twins": self.strategy_twins(db),
                "base_strategies": [f"{b['asset']}|{b['minutes']}|{b['variant']}" for b in self.base_strategies()],
                "removed_strategies": self.removed_strategies(),
                "removed_hidden": self.hidden_removed(),
                "deleted_strategies": sorted(self.deleted_strategies()),
                "strategy_lab": self.strategy_lab(),
                "strategy_rules": {_key(a, m, v): lab.describe(v) for a, m, v in
                                   self._owner_keys() | {tuple(k) for ks in self.removed_strategies().values() for k in ks}},
                "auto_calibration": self.calibration,
                "market_blend": self.market_blend,
                "blend_use": self.blend_use(),
                "operating_costs": dict(self.operating_costs(), claude_month=research.spent_since(db, starts["month"])),
                "stake_curve": self.stake_curve,
                "protections": self.protections(),
                "money": self.money(),
                "rr_research": self.rr_view(),
                "rr_auto": self.rr_auto(),
                "count_since": self.count_since(),
                "max_rules": self.max_rules(),
                "stake_research": self.stake_view(),
                "learning": self.learning,
                "paper": {"summary": paper, "summary_all": paper_all, "today": paper_today,
                          "feed": {m: stats.feed_agreement(db, m) for m in ("twap", "point")},
                          "bets": stats.recent_bets(db, "paper", 20, strategy="crypto"),
                          "bets_stocks": stats.recent_bets(db, "paper", 20, strategy="stocks")},
                "sim_strategies": self.sim_strategies(),
                "sim_limits": self.sim_limits(),
                "analysis": self.analysis,
                "readiness": self.readiness,
                "notify": self.notifier.status(),
                "live": {"status": self.live.status(), "total": live_total, "today": live_today,
                         "bets": [dict(b, why=reject_reason(b["error"]) if b["status"] != "filled" else "", would=stats.would_have(b))
                                  for b in stats.recent_bets(db, "live")]},
                "dashboard": self._dashboard(db, starts),
                "positions": self._positions(db, now),
                "stocks": stock_state,
                "stocks_main": self._stocks_main_state(db, now, stock_state["board"] if stock_state else []),
                # Trades still running: their window hasn't closed (closed ones only wait for the result).
                "open_counts": dict(db.execute(
                    # One per card on the page (a market's window), however many strategies entered it.
                    "SELECT mode, COUNT(DISTINCT asset || '|' || minutes || '|' || window_start) FROM bets"
                    " WHERE status = 'filled' AND pnl IS NULL AND"
                    " (CASE WHEN minutes > 0 THEN window_start + minutes * 60 ELSE window_start END) > ?"
                    " GROUP BY mode", (now,)).fetchall()),
                "lab": {
                    "results": lab_state,
                    "markets_followed": len(self.trackers),
                    "variants": sum(len(t.variants) for t in self.trackers),
                    "variants_by_family": {f: sum(1 for t in self.trackers for v in t.variants if lab.family(v.key) == f)
                                           for f in lab.FAMILIES},
                    "min_bets": self.cfg.lab_min_bets, "oos_min_bets": self.cfg.lab_oos_min_bets,
                    "research": research.latest(db),
                    "research_configured": research.configured(),
                    "research_running": self.research_running,
                    "research_hour_utc": self.cfg.research_hour_utc,
                },
            }
        finally:
            db.close()
