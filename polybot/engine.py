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

from . import calibration, export, feeds, lab, research, riskreward, rrlab, stakes, stats, stocks
from .live import LiveTrader, reject_reason
from .health import Health
from .notify import Notifier
from .periods import period_starts
from .model import (VolatilityEstimator, fair_up_probability, fair_up_probability_twap, norm_cdf, t3_cdf,
                    taker_fee_per_share)
from .paper import Fill, buy_shares, find_bet, find_lock, find_lottery, settle_pnl, simulate_buy
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


def _lab_only(variant):
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

        # Real money, the simulation and the lab each run whatever happens to the others: a fault in
        # one (shown in the red banner) must never keep the others from trading.
        failed = None
        for part in (self._live_rules_run, self._sim_rules_run, self._run_lab):
            try:
                part(now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age, fair, fair_live)
            except Exception as e:
                failed = failed or e
        if failed:
            raise failed

    def _sim_rules_run(self, now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age, fair, fair_live):
        for r in self.main_rules:
            if fair.get(lab.reading(r)) is not None:
                self._sim_bet(now, seconds_left, r, fair[lab.reading(r)], raw[lab.reading(r)], up_asks, down_asks,
                              price_age, book_age)

    def _live_rules_run(self, now, seconds_left, cal, raw, up_asks, down_asks, price_age, book_age, fair, fair_live):
        if self.live:
            for r in self.live_rules:
                if fair_live.get(lab.reading(r)) is not None:
                    self._live_bet(now, seconds_left, r, fair_live[lab.reading(r)], up_asks, down_asks,
                                   plain=cal.get(lab.reading(r)))  # the lab's view: no mix with the market

    @staticmethod
    def _decide(v, p, up_asks, down_asks, cfg):
        """A rule's bet on these books (None: no bet): a lottery ticket, or the usual edge rule.
        The risk/reward range in cfg applies to both."""
        if lab.is_lock(v):
            return find_lock(p, up_asks, down_asks, lab.lock_least(v), v.min_edge, lab.LOCK_EDGE, cfg)
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
            if fill is None:
                if not (lab.is_lotto(rule) or lab.is_lock(rule)):
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
                for extra in self._copies(rule, fill, up_asks if fill.side == "up" else down_asks, rcfg, self.sim_copies):
                    if self.can_open and not self.can_open("crypto", extra.total_paid):
                        break
                    self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, extra,
                                       variant=rule.key, price_age=price_age, book_age=book_age, fair_raw=side_raw,
                                       stake=rcfg.stake_usd)
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
        if copies <= 1 or lab.is_lotto(rule) or lab.is_lock(rule) or rule.resolution == "twapm" or not asks:
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

    def _live_orders(self, now, seconds_left, fill, cfg, rule, stake, copies, round_stake=None):
        """Real money: the order, then the extra entries as their own fill-or-kill orders, each
        allowed at most COPY_SLIP above the first price and never past the rule's edge."""
        first = self.live.maybe_trade(now, self.asset, self.minutes, self.window, seconds_left, fill, self.market, cfg,
                                      "crypto", None, rule.key, stake, round_stake=round_stake)
        if first != "filled" or copies <= 1 or lab.is_lotto(rule) or lab.is_lock(rule) or rule.resolution == "twapm":
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

    def _live_bet(self, now, seconds_left, rule, fair_up, up_asks, down_asks, plain=None):
        """Real money's trade on this market: its own rule and its own settings (stake,
        risk/reward, caps). The simulation's limits, or the simulation being off, never hold it
        back or let it through."""
        if not (rule.min_seconds_left <= seconds_left <= rule.max_seconds_left
                and self._due(self.live_entries, self.live_done, rule, now)):
            return
        cfg = dataclasses.replace(self.market_cfg, min_edge=rule.min_edge)
        lcfg = self.live.order_cfg(cfg)
        round_stake = rule.stake or lcfg.stake_usd  # this strategy's amount a round, or real money's
        lcfg = dataclasses.replace(lcfg, stake_usd=round_stake / lab.entries(rule),
                                   min_ratio=max(lcfg.min_ratio, rule.ratio))  # the rule's filter, as in the lab
        live_fill = self._decide(rule, fair_up, up_asks, down_asks, lcfg)
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
        if find_bet(fair_up, up_asks, down_asks, dataclasses.replace(rcfg, full_fill_only=False)) is not None:
            self._skipped(rule, "full")

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

    def _run_lab(self, now, seconds_left, fair, raw, up_asks, down_asks, price_age, book_age, *_):
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
            if lab.is_lock(v):
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
            if fill:
                self._entered(self.lab_entries, self.lab_done, v, now)
                r = raw[lab.reading(v)]
                side_raw = None if r is None else (r if fill.side == "up" else 1 - r)
                self.store.add_bet(now, self.asset, self.minutes, self.window, seconds_left, fill, mode="lab",
                                   variant=v.key, price_age=price_age, book_age=book_age, fair_raw=side_raw,
                                   stake=v_cfg.stake_usd)

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
        self.pair_legs = {}  # both-sides rule -> its first leg in this window: (side, shares, cost a share)
        for mode, variant, side, shares, paid in prior:
            if mode == "lab" and variant.startswith("pair"):
                leg = self.pair_legs.get(variant)
                if leg is None and shares:
                    self.pair_legs[variant] = (side, shares, paid / shares)
                elif leg and side != leg[0]:
                    self.lab_done.add(variant)
                continue
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
        stake = lab.LOCK_STAKE if lab.is_lock(v) else self.market_cfg.stake_usd / lab.entries(v)  # (small: see lab.py)
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
            self.stocks.count_since = float(self.count_since().get("lab") or 0)
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
        self.rr_research = {}
        self._migrate_money()
        self._apply_money()

    def start(self):
        threading.Thread(target=self._stream_thread if self.streaming else self._rest_loop, daemon=True,
                         name="engine").start()
        # Learning runs the moment trades are settled (_settle); these timers are only the fallback.
        for seconds, fn in ((10, self._settle), (60, self._lab_tick), (300, self._analysis_tick),
                            (300, self._daily), (5, self.notifier.tick), (30, self.live.refresh_wallet),
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
    LEARN_GAP = 60  # seconds: settlements come every few seconds, a learning pass takes a few

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
        rules = {}
        for m in main:
            if not m.get("paused"):
                ratio = float(over) if over is not None else float(m.get("ratio") or 0.0)
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
    def count_since(self):
        return self._json_setting("count_since", {})

    def set_count_since(self, area, fresh):
        if area not in ("sim", "lab"):
            raise ValueError("אזור לא מוכר")
        since = self.count_since()
        if fresh:
            since[area] = time.time()
            if area == "lab":  # the stages start over too: candidates are picked on the new data
                self.store.set_setting("lab_candidates_before_count", json.dumps(self._json_setting("lab_candidates", {})))
                self.store.set_setting("lab_candidates", json.dumps({}))
        else:
            since.pop(area, None)
            if area == "lab" and self.store.get_setting("lab_candidates_before_count"):
                before = self._json_setting("lab_candidates_before_count", {})
                self.store.set_setting("lab_candidates", json.dumps({**before, **self._json_setting("lab_candidates", {})}))
        self.store.set_setting("count_since", json.dumps(since))
        if self.stocks:
            self.stocks.count_since = float(since.get("lab") or 0)
        log.warning("Counting %s %s", area, "afresh from now" if fresh else "over all the data again")
        if area == "lab":
            self._refresh_lab_soon()

    def strategy_results(self, db):
        """Each strategy's own record in the simulation and in real money, by (asset, minutes, rule)
        (the simulation's from where the owner started counting afresh, if they did)."""
        out = {}
        since = {"paper": self.count_since().get("sim") or 0, "live": 0}
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
        now = time.time()
        t = next((t for t in self.trackers if t.asset == asset and t.minutes == int(minutes)), None)
        if t is None or t.window != int(window_start) or not t.market:
            raise ValueError("הסבב הזה כבר נסגר")
        seconds_left = t.window + t.window_seconds - now
        if seconds_left < self.MANUAL_MIN_LEFT:
            raise ValueError(f"נשארו פחות מ-{self.MANUAL_MIN_LEFT} שניות לסבב")
        token = t.market["up_token"] if side == "up" else t.market["down_token"]
        book = self.books.get(token) if self.books else (*t.api.order_book(token), now)
        asks = (book[1] if book else None) or []
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
            if t.can_open and not t.can_open("crypto", fill.total_paid, side, t.window):
                raise ValueError("ההגבלות של הסימולציה לא מאפשרות עוד טרייד עכשיו (טריידים פתוחים / אותו כיוון / הפסד יומי)")
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
            _lab_only(variant)
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
        _lab_only(variant)
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
        return self.store.get_setting("lab_auto_candidates", "1") == "1"

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
        return "live" if self.live.armed() else "paper"

    def set_mode(self, live, confirmed=False, keep_daily=False):
        if not confirmed:
            raise ValueError("צריך אישור")
        self.live.arm(bool(live), override=True, keep_daily=keep_daily)

    # The money strip, per mode: the amount for each trade, the day's loss limit, and how many
    # entries go into each approved trade. Each has the system's value (from the data) and,
    # when the owner typed one, the owner's, which then rules until "back to the system's values".
    MONEY_KEYS = ("stake", "daily_loss", "copies", "rr")

    def _money_overrides(self):
        s = self._json_setting("money_override", {})
        return {m: {k: v for k, v in (s.get(m) or {}).items() if k in self.MONEY_KEYS and v is not None} for m in ("sim", "live")}

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
        return {"sim": bool(s.get("sim", True)), "live": bool(s.get("live", False))}

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

    def _strategy_stake(self, mode, m):
        """The amount a round this strategy trades where it runs (0: the mode's amount)."""
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

    def _lab_variants(self, asset, minutes):
        """What the lab runs on a market: the fixed grid, what its search added, and every
        strategy connected to the owner there, so the lab keeps measuring what the owner runs."""
        extra = [k for k in self._json_setting("lab_dynamic", {}).get(_key(asset, minutes), [])
                 if not k.startswith("lotto")]  # (the lottery tickets left the lab)
        mine = sorted(v for a, m, v in self._owner_keys() if (a, m) == (asset, minutes))
        base = lab.variants_for(minutes)
        keys = {v.key for v in base}
        out = base + [lab.Variant.from_key(k) for k in extra if k not in keys]
        keys |= set(extra)
        return out + [lab.Variant.from_key(k) for k in mine if k not in keys]

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
            for near in lab.neighbours(b["variant"], b["minutes"]):
                if near in have:
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

    def refresh_lab(self):
        """Every minute: the lab ranking (with its stages and search) and the readiness checks."""
        now = time.time()
        candidates = {tuple(k.split("|", 2)): ts for k, ts in self._json_setting("lab_candidates", {}).items()}
        candidates = {(a, int(m), v): ts for (a, m, v), ts in candidates.items()}
        db = connect(self.cfg.db_path)
        try:
            board = lab.leaderboard(db, self.cfg, self.current_variants, candidates, self.dynamic_variants,
                                    since=self.count_since().get("lab"))
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
        """The lab's strategies by how they did, counted at most every MIX_GAP seconds: it reads every
        settled lab trade, the slowest part of the page's refresh (nearly all of it on a server with
        days of lab trades), and on the server's one processor that time is taken from the trading.
        It moves only as rounds settle, so half a minute old reads the same."""
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
        return {"sim": bool(s.get("sim", True)), "live": bool(s.get("live", False))}

    def set_lab_exact(self, mode, on):
        s = self.lab_exact()
        s["live" if mode == "live" else "sim"] = bool(on)
        self.store.set_setting("lab_exact", json.dumps(s))
        self.live.exact = s["live"]
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
                "live_main": [dict(m, rule=lab.describe(m["variant"])) for m in self.live_main()],
                "versions": {"list": self.versions_view("sim"), "current": self.current_version()},
                "live_versions": {"list": self.versions_view("live"), "current": self.current_version("live")},
                "auto_push": {"sim": self.auto_push(), "live": self.auto_push("live")},
                "auto_candidates": self.auto_candidates(),
                "strategy_names": self.strategy_names(),
                "strategy_results": self.strategy_results(db),
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
                "paper": {"summary": paper,
                          "feed": {m: stats.feed_agreement(db, m) for m in ("twap", "point")},
                          "bets": stats.recent_bets(db, "paper", 20, strategy="crypto"),
                          "bets_stocks": stats.recent_bets(db, "paper", 20, strategy="stocks")},
                "sim_strategies": self.sim_strategies(),
                "sim_limits": self.sim_limits(),
                "analysis": self.analysis,
                "readiness": self.readiness,
                "notify": self.notifier.status(),
                "live": {"status": self.live.status(), "total": live_total, "today": live_today,
                         "bets": [dict(b, why=reject_reason(b["error"]) if b["status"] != "filled" else "")
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
                                           for f in ("main", "pair", "lock")},
                    "min_bets": self.cfg.lab_min_bets, "oos_min_bets": self.cfg.lab_oos_min_bets,
                    "research": research.latest(db),
                    "research_configured": research.configured(),
                    "research_running": self.research_running,
                    "research_hour_utc": self.cfg.research_hour_utc,
                },
            }
        finally:
            db.close()
