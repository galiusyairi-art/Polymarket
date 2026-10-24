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
import math

from . import calibration, export, feeds, lab, research, riskreward, stakes, stats, stocks
from .live import LiveTrader
from .health import Health
from .notify import Notifier
from .periods import period_starts
from .model import (VolatilityEstimator, fair_up_probability, fair_up_probability_twap, norm_cdf, t3_cdf,
                    taker_fee_per_share)
from .paper import Fill, find_bet, find_lottery, settle_pnl
from .store import Store, connect

log = logging.getLogger("polybot")


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
        # Whether the simulation's and real money's decisions use that mix (the owner's choice). The
        # lab never does: it measures our own model, which a mix leaning on the market would hide.
        self.blend_sim = self.blend_live = True
        self.min_ratio = self.max_ratio = 0.0  # risk/reward range on the main strategy, set by the engine
        self.live_rules = []  # real money's own rules on this market (none: real money doesn't trade it)
        self.on_skip = None  # the engine's record of why a signal didn't become a trade
        self.sim_on = True  # the owner can switch the crypto strategy's simulation off
        self.sim_stake = cfg.stake_usd  # dollars per simulated main-strategy bet (the owner's setting)
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
        self.price = self.price_ts = None
        self.window = None
        self.market = None
        self.market_cfg = cfg  # cfg with this market's own fee schedule
        self.next_market = None  # the next window's market, fetched ahead of time
        self.opens = {"twap": None, "point": None}
        self.main_done = set()  # rule keys the simulation bet with in this window
        self.live_done = set()  # rule keys real money traded (or tried) in this window
        self.lab_done = set()  # variant keys done for this window (bet as many times as they may)
        # Variants that may enter several times a window: key -> (entries so far, when the last was).
        self.lab_entries, self.main_entries, self.live_entries = {}, {}, {}
        self.maker_orders = {}
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
        return dataclasses.replace(self.market_cfg, min_edge=rule.min_edge, min_ratio=self.min_ratio,
                                   max_ratio=self.max_ratio, stake_usd=self.sim_stake / lab.entries(rule))

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
            return

        for r in self.main_rules:
            if fair.get(lab.reading(r)) is not None:
                self._sim_bet(now, seconds_left, r, fair[lab.reading(r)], raw[lab.reading(r)], up_asks, down_asks,
                              price_age, book_age)
        if self.live:
            for r in self.live_rules:
                if fair_live.get(lab.reading(r)) is not None:
                    self._live_bet(now, seconds_left, r, fair_live[lab.reading(r)], up_asks, down_asks)

        self._run_lab(now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age)

    @staticmethod
    def _decide(v, p, up_asks, down_asks, cfg):
        """A rule's bet on these books (None: no bet): a lottery ticket, or the usual edge rule.
        The risk/reward range in cfg applies to both."""
        if lab.is_lotto(v):
            fill = find_lottery(p, up_asks, down_asks, v.min_edge, lab.lotto_min_fair(v), cfg)
            if fill and not riskreward.passes_range(fill.total_paid / fill.shares, cfg.min_ratio, cfg.max_ratio):
                return None
            return fill
        return find_bet(p, up_asks, down_asks, cfg)

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
            if fill is None and not lab.is_lotto(rule):
                self._why_not(rule, fair_up, up_asks, down_asks, rcfg)
            elif self.can_open and not self.can_open("crypto", fill.total_paid, fill.side, self.window):
                self._skipped(rule, "limits")  # the owner's limits: open trades, same direction, today's losses
                fill = None
            if fill:
                self._entered(self.main_entries, self.main_done, rule, now)
                side_raw = fair_raw if fill.side == "up" else 1 - fair_raw
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill,
                                   variant=rule.key, price_age=price_age, book_age=book_age, fair_raw=side_raw,
                                   stake=rcfg.stake_usd)
                log.info("PAPER BET %s %s: fair %.1f%%, paid %.1fc/share (+fee), edge %.1f pts, %.1fs left,"
                         " data age price %.2fs / book %.2fs", self.key.upper(), fill.side.upper(),
                         fill.fair_prob * 100, fill.avg_price * 100, fill.edge * 100, seconds_left, price_age,
                         book_age)

    def _live_bet(self, now, seconds_left, rule, fair_up, up_asks, down_asks):
        """Real money's trade on this market: its own rule and its own settings (stake,
        risk/reward, caps). The simulation's limits, or the simulation being off, never hold it
        back or let it through."""
        if not (rule.min_seconds_left <= seconds_left <= rule.max_seconds_left
                and self._due(self.live_entries, self.live_done, rule, now)):
            return
        cfg = dataclasses.replace(self.market_cfg, min_edge=rule.min_edge)
        lcfg = self.live.order_cfg(cfg)
        lcfg = dataclasses.replace(lcfg, stake_usd=lcfg.stake_usd / lab.entries(rule))
        live_fill = self._decide(rule, fair_up, up_asks, down_asks, lcfg)
        if live_fill and self.live.blocked("crypto", live_fill.side, self.window) is None:
            self._entered(self.live_entries, self.live_done, rule, now)
            self.dispatch(self.live.maybe_trade, now, self.asset, self.minutes, self.window, seconds_left,
                          live_fill, self.market, cfg, "crypto", None, rule.key, lcfg.stake_usd)

    # Why a rule with an edge didn't bet: the whole stake wasn't on offer at good prices, or the
    # risk/reward range left it out. (No edge at all is the normal case and isn't counted.)
    def _why_not(self, rule, fair_up, up_asks, down_asks, rcfg):
        if not self.on_skip:
            return
        loose = dataclasses.replace(rcfg, full_fill_only=False, min_ratio=0.0, max_ratio=0.0)
        if find_bet(fair_up, up_asks, down_asks, loose) is None:
            return
        no_ratio = dataclasses.replace(rcfg, min_ratio=0.0, max_ratio=0.0)
        self._skipped(rule, "full" if find_bet(fair_up, up_asks, down_asks, no_ratio) is None else "ratio")

    def _skipped(self, rule, reason):
        if self.on_skip:
            self.on_skip(self, rule, reason)

    def _maker(self, v, now, seconds_left, p, p_raw, up_asks, down_asks, price_age, book_age):
        """A resting buy order at our own price (simulated). Filled only when a seller comes down
        through it: the best ask drops below our price. No taker fee."""
        order = self.maker_orders.get(v.key)
        asks = {"up": up_asks, "down": down_asks}
        if order:
            ask = _best(asks[order["side"]], min)
            if ask is not None and ask < order["price"] - 1e-9:
                price, shares = order["price"], order["shares"]
                fill = Fill(side=order["side"], fair_prob=order["fair"], best_ask=ask, avg_price=price, shares=shares,
                            cost=shares * price, fee=0.0, edge=order["fair"] - price, limit_price=price)
                self.lab_done.add(v.key)
                self.maker_orders.pop(v.key, None)
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill, mode="lab",
                                   variant=v.key, price_age=price_age, book_age=book_age, fair_raw=order["raw"],
                                   stake=self.market_cfg.stake_usd)
                return
        if p is None or not (v.min_seconds_left <= seconds_left <= v.max_seconds_left):
            self.maker_orders.pop(v.key, None)
            return
        tick = (self.market or {}).get("tick_size") or 0.01
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
            self.maker_orders.pop(v.key, None)
            return
        side, _, price, prob = best
        raw_side = p_raw if side == "up" else (1 - p_raw if p_raw is not None else None)
        self.maker_orders[v.key] = {"side": side, "price": price, "fair": prob, "raw": raw_side,
                                    "shares": self.cfg.stake_usd / price}

    def _run_lab(self, now, seconds_left, fair, raw, up_asks, down_asks, price_age, book_age):
        if not self.variants:
            return
        best_up = _best(up_asks, min)
        best_down = _best(down_asks, min)
        mc = self.market_cfg
        cfgs = self.lab_cfgs  # read once: the engine may swap in a new set of variants
        for v in self.variants:
            if v.resolution == "twapm" and v.key not in self.lab_done:
                self._maker(v, now, seconds_left, fair["twap"], raw["twap"], up_asks, down_asks, price_age, book_age)
                continue
            if not (v.min_seconds_left <= seconds_left <= v.max_seconds_left) or not self._due(
                    self.lab_entries, self.lab_done, v, now):
                continue
            p = fair.get(lab.reading(v))
            if p is None:
                continue
            # Cheap check on the best prices before walking the books.
            if lab.is_lotto(v):
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
            if fill:
                self._entered(self.lab_entries, self.lab_done, v, now)
                r = raw[lab.reading(v)]
                side_raw = None if r is None else (r if fill.side == "up" else 1 - r)
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill, mode="lab",
                                   variant=v.key, price_age=price_age, book_age=book_age, fair_raw=side_raw,
                                   stake=v_cfg.stake_usd)

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
        self.main_entries, self.live_entries, self.lab_entries = {}, {}, {}
        self.main_done, self.live_done, self.lab_done = set(), set(), set()
        for mode, variant in prior:
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
        return dataclasses.replace(self.market_cfg, min_edge=v.min_edge,
                                   stake_usd=self.market_cfg.stake_usd / lab.entries(v))

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


def settle(store, cfg, now, api=feeds):
    """Fill in winners for finished windows (all markets, last 2 days) and settle their bets.
    Returns how many bets it settled."""
    shortest = min((cfg.window_minutes, *cfg.lab_minutes))
    settled = 0
    for asset, minutes, start, slug in store.unresolved_windows(now - shortest * 60 - SETTLE_AFTER, now - 2 * 86400):
        if start + minutes * 60 + SETTLE_AFTER > now:
            continue
        winner = api.market_winner(slug)
        if not winner:
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
        self.stocks = stocks.StockLab(cfg, self.store) if (with_lab and cfg.stocks_lab) else None
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
        self.learning = {}
        self.stake_curve = {}

    def start(self):
        threading.Thread(target=self._stream_thread if self.streaming else self._rest_loop, daemon=True,
                         name="engine").start()
        # Learning runs the moment trades are settled (_settle); these timers are only the fallback.
        for seconds, fn in ((10, self._settle), (60, self._lab_tick), (300, self._analysis_tick),
                            (300, self._daily), (5, self.notifier.tick), (300, self._refresh_balance),
                            (60, self.health.check)):
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

    def _settle(self):
        now = time.time()
        n = settle(self.store, self.cfg, now, api=self.api)
        self.learning["settle"] = {"ts": now, "bets": n}
        if n:
            self.learning["last_settled"] = {"ts": now, "bets": n}
            self.learn_now("settle")

    # Learning: everything the brain knows is recomputed from all the data the moment new trades
    # are settled (and on a timer as a fallback). One run at a time; a request while one runs
    # queues exactly one more, so the newest trades are always learned from.
    def learn_now(self, reason):
        if not self._learn_queued.acquire(blocking=False):
            return  # one is already waiting and will see these trades too
        threading.Thread(target=self._learn, args=(reason,), daemon=True, name="learn").start()

    def _learn(self, reason):
        with self._learn_lock:
            self._learn_queued.release()
            started = time.time()
            for fn in (self.refresh_analysis, self.refresh_lab):
                try:
                    fn()
                except Exception as e:
                    log.warning("%s: %s", fn.__name__, e)
            self.learning["learned"] = {"ts": time.time(), "took": time.time() - started, "reason": reason}

    def _analysis_tick(self):
        with self._learn_lock:
            self.refresh_analysis()

    def _lab_tick(self):
        with self._learn_lock:
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

    def main_strategies(self):
        return self._json_setting("main_strategies", self.base_strategies())

    @staticmethod
    def _rules_by_market(main):
        """The rules that trade, by market (a paused one stays listed but doesn't trade)."""
        rules = {}
        for m in main:
            if not m.get("paused"):
                rules.setdefault((m["asset"], m["minutes"]), []).append(lab.Variant.from_key(m["variant"]))
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
            if not found:
                main.append(dict(b, since=time.time()))
                changed.append(b)
            elif any(m.get("paused") for m in found):
                for m in found:
                    m["paused"] = False
                changed.append(b)
        if not changed:
            raise ValueError("האסטרטגיות הבסיסיות כבר רצות")
        self._save_version(main, "החזרת האסטרטגיות הבסיסיות", "restore", mode=mode)
        self._save_for(mode, main)

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

    def strategy_results(self, db):
        """Each strategy's own record in the simulation and in real money, by (asset, minutes, rule)."""
        out = {}
        for mode in ("paper", "live"):
            out[mode] = {f"{a}|{m}|{v}": {"bets": n, "settled": s, "wins": w, "pnl": pnl, "paid": paid}
                         for a, m, v, n, s, w, pnl, paid in db.execute(
                             "SELECT asset, minutes, variant, COUNT(*), COUNT(pnl), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),"
                             " COALESCE(SUM(pnl), 0), COALESCE(SUM(CASE WHEN pnl IS NOT NULL THEN cost + fee END), 0)"
                             " FROM bets WHERE mode = ? AND status = 'filled' AND minutes > 0 GROUP BY asset, minutes, variant",
                             (mode,)).fetchall()}
        return out

    def _apply_main(self, main):
        rules = self._rules_by_market(main)
        for t in self.trackers:
            mine = rules.get((t.asset, t.minutes), [])
            t.set_main_rules(mine, 1.0 if mine else self.cfg.lab_record_seconds)
        self.main_trackers = [t for t in self.trackers if t.main]

    # Real money has its own strategy: it starts as the simulation's and changes only when the
    # owner pushes to it (or switched on the automatic push for real money).
    def live_main(self):
        return self._json_setting("live_main", None) or self.main_strategies()

    def strategies_for(self, mode):
        return self.live_main() if mode == "live" else self.main_strategies()

    def _apply_live_main(self, main):
        rules = self._rules_by_market(main)
        for t in self.trackers:
            t.live_rules = rules.get((t.asset, t.minutes), [])

    def _save_live_main(self, main, reset_readiness=True):
        self.store.set_setting("live_main", json.dumps(main))
        # Readiness for real money is judged afresh on the new strategy's own bets.
        if reset_readiness:
            self.store.set_setting("live_changed_at", str(time.time()))
        self._apply_live_main(main)
        log.warning("Real-money strategy is now: %s", ", ".join(f"{m['asset']}-{m['minutes']}m {m['variant']}" for m in main))
        self.refresh_lab()

    def _save_for(self, mode, main, reset_readiness=True):
        (self._save_live_main if mode == "live" else self._save_main)(main, reset_readiness)

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
            row = board.get((asset, minutes, variant))
            if row is None or (row["status"] != "proven" and not (allow_unproven and mode == "sim")):
                raise ValueError("לכסף אמיתי אפשר לדחוף רק אסטרטגיה מוכחת" if mode == "live" else "אפשר לדחוף רק אסטרטגיה מוכחת")
            if not any((t.asset, t.minutes) == (asset, minutes) for t in self.trackers):
                raise ValueError("השוק הזה לא במעקב")
            if any((m["asset"], m["minutes"], m["variant"]) == (asset, minutes, variant) for m in main):
                continue  # already running there
            if replace:
                main = [m for m in main if (m["asset"], m["minutes"]) != (asset, minutes)]
            elif sum(1 for m in main if (m["asset"], m["minutes"]) == (asset, minutes)) >= self.cfg.max_rules_per_market:
                weakest = self._weakest(main, asset, minutes, board)
                main = [m for m in main if m is not weakest]
                texts.append(f"הוצאה (הכי חלשה בשוק): {asset.upper()} {minutes}m: {lab.describe(weakest['variant'])}")
            main.append({"asset": asset, "minutes": minutes, "variant": variant, "since": time.time()})
            texts.append(f"{asset.upper()} {minutes}m: {lab.describe(variant)}")
            stats.append({k: row.get(k) for k in ("asset", "minutes", "variant", "bets", "roi", "roi_pessimistic",
                                                   "oos_bets", "oos_roi", "oos_pessimistic")})
        if not stats:
            raise ValueError("האסטרטגיה כבר רצה שם")
        self._save_version(main, label_prefix + "; ".join(texts), "push", stats, mode=mode)
        self._save_for(mode, main)

    def remove_strategy(self, asset, minutes, variant, mode="sim"):
        """Take one rule off a market (the owner's decision); at least one strategy stays."""
        main = self.strategies_for(mode)
        left = [m for m in main if (m["asset"], m["minutes"], m["variant"]) != (asset, minutes, variant)]
        if len(left) == len(main):
            raise ValueError("האסטרטגיה הזאת לא רצה שם")
        if not left:
            raise ValueError("חייבת להישאר לפחות אסטרטגיה אחת")
        self._save_version(left, f"הוסרה: {asset.upper()} {minutes}m: {lab.describe(variant)}", "remove", mode=mode)
        self._save_for(mode, left)

    # When a lab strategy becomes proven: a mail, and for each of the simulation and real money
    # whose automatic push the owner switched on, a push (only if it beats what runs there).
    def auto_push(self, mode="sim"):
        return self.store.get_setting("live_auto_push" if mode == "live" else "lab_auto_push", "0") == "1"

    def set_auto_push(self, on, mode="sim"):
        self.store.set_setting("live_auto_push" if mode == "live" else "lab_auto_push", "1" if on else "0")
        log.warning("Automatic push of proven strategies (%s): %s", mode, "on" if on else "off")

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
        full = lambda key: sum(1 for m in running if (m["asset"], m["minutes"]) == key) >= self.cfg.max_rules_per_market
        # What adding a rule to a full market would take out: its weakest one.
        dropped = {}
        best = {}
        for b in board:
            key = (b["asset"], b["minutes"])
            if b["status"] != "proven" or (*key, b["variant"]) in on:
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
            self.stocks.min_ratio, self.stocks.max_ratio = getattr(self, "rr_ranges", {}).get("stocks", (0.0, 0.0))
            self.stocks.enabled = on["stocks"]
            self.stocks.can_open = self.can_open
            self.stocks.sim_stake = limits["stake"]

    # The simulation's amounts and limits (the owner's settings; real money has its own).
    # Who sets the strategy's own dials (risk/reward, the mix with the market price, the caps on
    # open and same-direction trades): the brain, by default, or the owner's choices from the
    # Studio, once applied there. Amounts, the daily loss limit, and which strategies run are the
    # owner's either way.
    BRAIN = {"sim": {"rr": "auto", "blend": False, "same": 1}, "live": {"rr": "auto", "blend": True, "same": 1}}

    def control(self):
        return "studio" if self.store.get_setting("strategy_control") == "studio" else "brain"

    def set_control(self, control):
        if control not in ("brain", "studio"):
            raise ValueError("מצב לא מוכר")
        self.store.set_setting("strategy_control", control)
        self._apply_control()
        log.warning("Strategy dials managed by: %s", control)

    def _apply_control(self):
        self._apply_risk_reward()
        self._apply_blend_use()
        self._apply_sim()

    def studio_apply(self, body):
        """The Studio's "apply to the system": its dials for the simulation and real money become
        the ones in use, and the owner, not the brain, now manages them."""
        self.store.set_setting("strategy_control", "studio")
        for mode in ("sim", "live"):
            part = body.get(mode) or {}
            if "max_open" in part or "same" in part:
                if mode == "sim":
                    self.set_sim_limits(max_open=part.get("max_open"), same_direction=part.get("same"))
                else:
                    if "max_open" in part:
                        self.live.set_max_open(part["max_open"] or {"crypto": 0, "stocks": 0})
                    if part.get("same") is not None:
                        self.store.set_setting("live_same_direction", str(max(0, int(part["same"]))))
            if part.get("blend") is not None:
                self.set_blend_use(mode, part["blend"])
            rr = part.get("rr") or {}
            if rr.get("mode"):
                (self.set_live_risk_reward if mode == "live" else self.set_risk_reward)(rr["mode"], rr.get("ratio"))
        self._apply_control()
        log.warning("Studio applied to the system")

    def studio_values(self):
        """What the Studio holds (the owner's dials, stored even while the brain manages)."""
        raw = self._sim_limits_raw()
        return {"sim": {"rr": self.risk_reward(), "blend": self.blend_use()["sim"], "same": raw["same_direction"],
                        "max_open": raw["max_open"]},
                "live": {"rr": self.live_risk_reward(), "blend": self.blend_use()["live"],
                         "same": self.live.same_direction(raw=True), "max_open": self.live.max_open(raw=True)}}

    def sim_limits(self):
        limits = self._sim_limits_raw()
        if self.control() == "brain":
            limits.update(same_direction=self.BRAIN["sim"]["same"], max_open={"crypto": None, "stocks": None})
        return limits

    def _sim_limits_raw(self):
        stored = self._json_setting("sim_limits", {})
        max_open = stored.get("max_open") or {}
        return {"stake": float(stored.get("stake") or self.cfg.stake_usd),
                "daily_loss": float(stored["daily_loss"]) if stored.get("daily_loss") else None,
                # Bitcoin, Ether and Solana move together: same-direction bets in the same window are
                # mostly one bet. Default 1; 0 or empty in the settings means no cap.
                "same_direction": int(stored["same_direction"]) if stored.get("same_direction") is not None else 1,
                "max_open": {n: (int(max_open[n]) if max_open.get(n) else None) for n in ("crypto", "stocks")}}

    def set_sim_limits(self, stake=None, daily_loss=None, max_open=None, same_direction=None):
        limits = self._sim_limits_raw()
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
        so far plus everything still at stake). None of it while the simulation runs exactly
        like the lab."""
        if self.lab_exact()["sim"]:
            return True
        limits = self.sim_limits()
        which = stats.STRATEGY_SQL[strategy]
        if strategy == "crypto" and side and window is not None and limits["same_direction"]:
            same = self.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'paper' AND status = 'filled' AND minutes > 0"
                                 " AND window_start = ? AND side = ?", (window, side))[0][0]
            if same >= limits["same_direction"]:
                return False
        cap = limits["max_open"].get(strategy)
        if cap is not None:
            open_now = self.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'paper' AND status = 'filled'"
                                     f" AND pnl IS NULL{which}")[0][0]
            if open_now >= cap:
                return False
        if limits["daily_loss"] is not None:
            day = period_starts(time.time(), self.cfg.timezone)["day"]
            exposure = self.store._x(
                "SELECT COALESCE(SUM(CASE WHEN pnl IS NULL THEN cost + fee ELSE -pnl END), 0) FROM bets"
                " WHERE mode = 'paper' AND status = 'filled' AND ts >= ?", (day,))[0][0]
            if exposure + stake > limits["daily_loss"]:
                return False
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
        return "live" if self.live.armed() else "paper"

    def set_mode(self, live, confirmed=False):
        if not confirmed:
            raise ValueError("צריך אישור")
        self.live.arm(bool(live), override=True)

    # The risk/reward filter on the main strategy (the owner's setting, or the data's choice).
    def risk_reward(self):
        return self._rr_setting("risk_reward")

    def _rr_setting(self, key):
        """A risk/reward choice of the trading tabs: the brain decides (default), no filter, or the
        owner's minimum. A range once saved from the calculator reads as its minimum."""
        s = self._json_setting(key, {})
        mode = s.get("mode") if s.get("mode") in riskreward.MODES else "auto"
        ratio = float(s.get("ratio") or 0.5)
        if mode == "calc":
            calc = s.get("calc") or {}
            mode, ratio = ("manual", float(calc["min_ratio"])) if calc.get("min_ratio") else ("off", ratio)
        return {"mode": mode, "ratio": ratio}

    @staticmethod
    def _rr_ranges(s, auto):
        """{strategy: (min, max)} a risk/reward setting means (0: none)."""
        none = {"crypto": (0.0, 0.0), "stocks": (0.0, 0.0)}
        if s["mode"] == "manual":
            return {n: (s["ratio"], 0.0) for n in none}
        if s["mode"] == "auto":
            return {n: (auto, 0.0) for n in none}
        if s["mode"] == "calc":
            c = s["calc"]
            r = (float(c.get("min_ratio") or 0.0), float(c.get("max_ratio") or 0.0))
            return {n: (r if (c.get("strategies") or {}).get(n) else (0.0, 0.0)) for n in none}
        return none

    def save_calculator(self, targets, min_ratio, max_ratio, strategies):
        self.store.set_setting("strategy_control", "studio")  # the calculator is part of the Studio
        """The calculator's "save": its range becomes the risk/reward setting ("calculator") of
        the simulation, real money, or both, for the strategies ticked."""
        targets = [t for t in targets if t in ("sim", "live")]
        if not targets:
            raise ValueError("צריך לבחור סימולציה, כסף אמיתי, או שניהם")
        lo, hi = float(min_ratio or 0), float(max_ratio or 0)
        if not (0 <= lo <= 20 and 0 <= hi <= 20) or (hi and hi <= lo):
            raise ValueError("הטווח לא תקין: המקסימום צריך להיות גדול מהמינימום")
        strategies = {n: bool((strategies or {}).get(n)) for n in ("crypto", "stocks")}
        if not any(strategies.values()):
            raise ValueError("צריך לבחור לפחות אסטרטגיה אחת")
        calc = {"min_ratio": lo, "max_ratio": hi, "strategies": strategies, "saved_at": time.time()}
        for target in targets:
            key = "live_risk_reward" if target == "live" else "risk_reward"
            current = self._json_setting(key, {})
            current.update(mode="calc", calc=calc)
            self.store.set_setting(key, json.dumps(current))
        self._apply_risk_reward()
        log.warning("Calculator range saved for %s: %.2f-%s on %s", "+".join(targets), lo, hi or "none",
                    [n for n, on in strategies.items() if on])
        self.refresh_lab()

    def calc_trades(self, limit=500):
        """The calculator's data: the last settled simulated trades of the strategies that run."""
        db = connect(self.cfg.db_path)
        try:
            rows = db.execute(
                "SELECT ts, asset, minutes, side, cost + fee AS paid, shares, pnl, fair_prob FROM bets"
                " WHERE mode = 'paper' AND status = 'filled' AND pnl IS NOT NULL AND shares > 0 AND cost + fee > 0"
                " ORDER BY ts DESC LIMIT ?", (int(limit),)).fetchall()
        finally:
            db.close()
        return {"trades": [[r["ts"], "crypto" if r["minutes"] else "stocks", r["asset"], r["minutes"], r["side"],
                            round(r["paid"], 4), round(r["shares"], 4), round(r["pnl"], 4), r["fair_prob"]]
                           for r in reversed(rows)],
                "stake": self.sim_limits()["stake"], "strategies": ["crypto", "stocks"]}

    def set_risk_reward(self, mode, ratio=None):
        if mode not in riskreward.MODES:
            raise ValueError("מצב לא מוכר")
        current = self.risk_reward()
        if mode == "calc" and not current.get("calc"):
            raise ValueError("עוד לא נשמרו ערכים מהמחשבון")
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
        # Risk/reward is the trading tabs' choice in either management mode (the brain is one option there).
        exact = self.lab_exact()
        none = {"crypto": (0.0, 0.0), "stocks": (0.0, 0.0)}  # the lab has no risk/reward filter
        self.rr_ranges = none if exact["sim"] else self._rr_ranges(self.risk_reward(), self.rr_auto[0])
        self.rr_effective = self.rr_ranges["crypto"][0]
        for t in self.trackers:
            t.set_min_ratio(*self.rr_ranges["crypto"])
        if getattr(self, "stocks", None):  # not built yet on the first call at startup
            self.stocks.min_ratio, self.stocks.max_ratio = self.rr_ranges["stocks"]
        self.live.ratios = none if exact["live"] else self._rr_ranges(self.live_risk_reward(), self.rr_auto[0])
        self.live.min_ratio = self.live.ratios["crypto"][0]

    # Real money has its own risk/reward setting (same choices; checked again on each real order).
    def live_risk_reward(self):
        return self._rr_setting("live_risk_reward")

    def set_live_risk_reward(self, mode, ratio=None):
        if mode not in riskreward.MODES:
            raise ValueError("מצב לא מוכר")
        current = self.live_risk_reward()
        if mode == "calc" and not current.get("calc"):
            raise ValueError("עוד לא נשמרו ערכים מהמחשבון")
        if ratio is not None:
            if not 0 < float(ratio) <= 20:
                raise ValueError("היחס צריך להיות גדול מ-0 ועד פי 20")
            current["ratio"] = float(ratio)
        current["mode"] = mode
        self.store.set_setting("live_risk_reward", json.dumps(current))
        self._apply_risk_reward()

    def save_trading(self, mode, body):
        """The settings page's "simulation trading" / "real-money trading" tab, saved as one."""
        stocks_variant = str(body["stocks_variant"]) if body.get("stocks_variant") else None
        rr = body.get("risk_reward") or {}
        if body.get("blend") is not None:
            self.set_blend_use(mode, body["blend"])
        if mode == "sim":
            self.set_sim_strategies(bool(body.get("crypto")), bool(body.get("stocks")), stocks_variant)
            # (the caps are the Studio's now; a body without them leaves them as they are)
            self.set_sim_limits(body.get("stake"), body.get("daily_loss") or 0, body.get("max_open"), body.get("same_direction"))
            if rr.get("mode"):
                self.set_risk_reward(rr["mode"], rr.get("ratio"))
            return
        if stocks_variant:
            self.push_stocks(stocks_variant)
        self.live.update_settings(stake=float(body["stake"]) if body.get("stake") else None,
                                  daily_loss_limit=float(body["daily_loss"]) if body.get("daily_loss") else None)
        if "max_open" in body:
            self.live.set_max_open(body["max_open"] or {"crypto": 0, "stocks": 0})
        if body.get("same_direction") is not None:
            self.store.set_setting("live_same_direction", str(max(0, int(body["same_direction"]))))
        self.live.set_strategy("crypto", bool(body.get("crypto")), confirmed=True)
        self.live.set_strategy("stocks", bool(body.get("stocks")), confirmed=True)
        # Real money mirrors the simulation: what goes live also runs there.
        on = self.sim_strategies()
        self.set_sim_strategies(on["crypto"] or bool(body.get("crypto")), on["stocks"] or bool(body.get("stocks")))
        if rr.get("mode"):
            self.set_live_risk_reward(rr["mode"], rr.get("ratio"))

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
            self.market_blend = calibration.market_blend(db, self.cfg)
            self.analysis = dict(stats.analysis(db, self.cfg, markets), calibration=stats.calibration(db),
                                 auto_calibration=self.calibration)
            self.rr_table = riskreward.table(db, self.cfg)
            self.rr_auto = riskreward.choose(self.rr_table, self.cfg.rr_min_bets)
            self.stake_curve = {mode: stakes.curve(db, self.cfg, self._active_keys(mode)) for mode in ("sim", "live")}
            counts = dict(db.execute("SELECT mode, COUNT(*) FROM bets WHERE winner IS NOT NULL GROUP BY mode").fetchall())
            last = db.execute("SELECT MAX(window_start + minutes * 60) FROM bets WHERE winner IS NOT NULL AND minutes > 0").fetchone()[0]
            self.learning["analysis"] = {"ts": time.time(), "settled": counts, "last_resolved": last}
        finally:
            db.close()
        self._apply_risk_reward()
        for t in self.trackers:
            t.calib_k = self.calibration["k"]
            t.blend = (self.market_blend["a"], self.market_blend["b"]) if self.market_blend["applied"] else None
        self._apply_blend_use()
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
            self.learning["lab"] = {"ts": time.time(), "rules": len(board["board"])}
            self._on_proven(board["board"])
            first = db.execute("SELECT MIN(ts) FROM ticks").fetchone()[0]
            last = db.execute("SELECT MAX(ts) FROM ticks").fetchone()[0]
            hours = (last - first) / 3600 if first else 0.0
            live_rules = {(m["asset"], m["minutes"], m["variant"]) for m in self.live_main() if not m.get("paused")}
            if live_rules == {(m["asset"], m["minutes"], m["variant"]) for m in self.main_strategies() if not m.get("paused")}:
                since = float(self.store.get_setting("main_changed_at", "0"))
                summary = stats.bet_summary(db, "paper", since=since, min_ratio=self.rr_effective, strategy="crypto")
            else:
                # Real money runs a rule the simulation doesn't: judged on that rule's own lab bets.
                since = float(self.store.get_setting("live_changed_at", "0"))
                summary = stats.bet_summary(db, "lab" if self.with_lab else "paper", since=since,
                                            min_ratio=self.live.min_ratio, strategy="crypto", keys=live_rules)
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
                s = stats.bet_summary(db, mode, since=since, strategy=self.strategy_filter())
                s.pop("rows")
                trades[mode] = s
            out[period] = {"since": since, "trades": trades, "costs": self.costs_since(db, since, time.time()),
                           "strategies": stats.strategy_mix(db, self.cfg, since, self.current_variants)}
        return {
            "periods": out,
            "strategies_total": sum(len(t.variants) for t in self.trackers) + 1,
            "thresholds": {"min_bets": self.cfg.dash_min_bets, "hot": self.cfg.dash_hot_roi,
                           "losing": self.cfg.dash_losing_roi},
            "timezone": self.cfg.timezone,
        }

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
        return {"sim": bool(s.get("sim", True)), "live": bool(s.get("live", False))}

    def set_lab_exact(self, mode, on):
        s = self.lab_exact()
        s["live" if mode == "live" else "sim"] = bool(on)
        self.store.set_setting("lab_exact", json.dumps(s))
        self.live.exact = s["live"]
        self._apply_blend_use()
        self._apply_risk_reward()
        self._apply_sim()
        log.warning("Exactly like the lab: simulation %s, real money %s", s["sim"], s["live"])

    def protections(self):
        """What each mode adds on top of the lab's own rules, with its value now (shown in the
        trading tabs, so nothing is applied out of sight)."""
        limits, use = self.sim_limits(), (self.blend_use() if self.control() == "studio"
                                          else {m: self.BRAIN[m]["blend"] for m in ("sim", "live")})
        rr = lambda r: r[0] if r else 0.0
        return {"sim": {"same_direction": limits["same_direction"], "max_open": limits["max_open"],
                        "daily_loss": limits["daily_loss"], "blend": use["sim"],
                        "rr": rr(self._rr_ranges(self.risk_reward(), self.rr_auto[0])["crypto"])},
                "live": {"same_direction": self.live.same_direction(), "max_open": self.live.max_open(),
                         "daily_loss": self.live.daily_loss_limit(), "blend": use["live"],
                         "rr": rr(self._rr_ranges(self.live_risk_reward(), self.rr_auto[0])["crypto"])},
                "blend_learned": bool(self.market_blend.get("applied")), "exact": self.lab_exact()}

    # Whether the learned mix with the market price drives the simulation's and real money's decisions.
    def blend_use(self):
        s = self._json_setting("blend_use", {})
        return {"sim": bool(s.get("sim", True)), "live": bool(s.get("live", True))}

    def set_blend_use(self, mode, on):
        s = self.blend_use()
        s["live" if mode == "live" else "sim"] = bool(on)
        self.store.set_setting("blend_use", json.dumps(s))
        self._apply_blend_use()

    def _apply_blend_use(self):
        use = self.blend_use() if self.control() == "studio" else {m: self.BRAIN[m]["blend"] for m in ("sim", "live")}
        exact = self.lab_exact()
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
        reasons = {"full": 0, "ratio": 0, "limits": 0}
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
            paper = stats.bet_summary(db, "paper", strategy=which)
            live_total = stats.bet_summary(db, "live", strategy=which)
            starts = period_starts(now, self.cfg.timezone)
            live_today = stats.bet_summary(db, "live", since=starts["day"], strategy=which)
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
                "live_main": [dict(m, rule=lab.describe(m["variant"])) for m in self.live_main()],
                "versions": {"list": self.strategy_versions()[::-1], "current": self.current_version()},
                "live_versions": {"list": self.strategy_versions("live")[::-1], "current": self.current_version("live")},
                "auto_push": {"sim": self.auto_push(), "live": self.auto_push("live")},
                "strategy_names": self.strategy_names(),
                "strategy_results": self.strategy_results(db),
                "base_strategies": [f"{b['asset']}|{b['minutes']}|{b['variant']}" for b in self.base_strategies()],
                "auto_calibration": self.calibration,
                "market_blend": self.market_blend,
                "blend_use": self.blend_use(),
                "strategy_control": self.control(),
                "studio": self.studio_values(),
                "brain": {"sim": {"ratio": self.rr_ranges["crypto"][0] if self.control() == "brain" else None},
                          "rr_auto": {"ratio": self.rr_auto[0], "reason": self.rr_auto[1]}},
                "operating_costs": dict(self.operating_costs(), claude_month=research.spent_since(db, starts["month"])),
                "stake_curve": self.stake_curve,
                "protections": self.protections(),
                "learning": self.learning,
                "risk_reward": dict(self.risk_reward(), effective=self.rr_effective,
                                    max_price=riskreward.max_price(self.rr_effective),
                                    auto={"ratio": self.rr_auto[0], "reason": self.rr_auto[1]},
                                    table=self.rr_table, min_bets=self.cfg.rr_min_bets),
                "paper": {"summary": paper,
                          "feed": {m: stats.feed_agreement(db, m) for m in ("twap", "point")},
                          "bets": stats.recent_bets(db, "paper", 20, strategy="crypto"),
                          "bets_stocks": stats.recent_bets(db, "paper", 20, strategy="stocks")},
                "sim_strategies": self.sim_strategies(),
                "sim_limits": self.sim_limits(),
                "live_risk_reward": dict(self.live_risk_reward(), effective=self.live.min_ratio,
                                         max_price=riskreward.max_price(self.live.min_ratio)),
                "analysis": self.analysis,
                "readiness": self.readiness,
                "notify": self.notifier.status(),
                "live": {"status": self.live.status(), "total": live_total, "today": live_today,
                         "bets": stats.recent_bets(db, "live")},
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
                                           for f in ("main", "lotto")},
                    "min_bets": self.cfg.lab_min_bets, "oos_min_bets": self.cfg.lab_oos_min_bets,
                    "research": research.latest(db),
                    "research_configured": research.configured(),
                    "research_running": self.research_running,
                    "research_hour_utc": self.cfg.research_hour_utc,
                },
            }
        finally:
            db.close()
