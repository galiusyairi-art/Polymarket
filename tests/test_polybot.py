from decimal import Decimal
import dataclasses
import json
import os
import random
import tempfile
import time
import types
import unittest

import datetime as dt

from polybot import feeds, hedge, lab
from polybot.config import Config as _Config


def Config(**kw):
    """The tests fill the lab's and the simulation's orders at once, unless a test sets the delay
    (the server waits lab_fill_delay, as a real order does)."""
    kw.setdefault("lab_fill_delay", 0.0)
    return _Config(**kw)
from polybot.model import VolatilityEstimator, fair_up_probability, fair_up_probability_twap, taker_fee_per_share
from polybot.paper import Fill, find_bet, settle_pnl, simulate_buy
from polybot.engine import Engine, Tracker, settle
from polybot.live import LiveTrader
from polybot.stats import readiness
from polybot.store import Store, connect


class ModelTest(unittest.TestCase):
    def test_at_open_price_is_a_coin_flip(self):
        self.assertAlmostEqual(fair_up_probability(100, 100, 600, 0.0005), 0.5)

    def test_more_likely_up_when_above_open_and_more_so_near_the_end(self):
        early = fair_up_probability(100.1, 100, 600, 0.0005)
        late = fair_up_probability(100.1, 100, 30, 0.0005)
        self.assertGreater(early, 0.5)
        self.assertGreater(late, early)
        self.assertAlmostEqual(fair_up_probability(99.9, 100, 30, 0.0005), 1 - late, places=3)

    def test_expired_window_is_decided(self):
        self.assertEqual(fair_up_probability(101, 100, 0, 0.0005), 1.0)
        self.assertEqual(fair_up_probability(100, 100, 0, 0.0005), 1.0)  # a tie counts as Up
        self.assertEqual(fair_up_probability(99, 100, 0, 0.0005), 0.0)

    def test_fee_is_largest_near_fifty_cents(self):
        self.assertAlmostEqual(taker_fee_per_share(0.5, 0.25, 2), 0.015625)
        # Polymarket's own example: 100 shares at 50c pay $1.56 at rate 0.0624; and real fills at 7%:
        # 79c cost 80.16c a share, 84c cost 84.94c (what Polymarket's activity shows: 80.2c, 84.9c).
        self.assertAlmostEqual(taker_fee_per_share(0.5, 0.0624, 1) * 100, 1.56)
        self.assertAlmostEqual(0.79 + taker_fee_per_share(0.79, 0.07, 1), 0.80161, places=5)
        self.assertAlmostEqual(0.84 + taker_fee_per_share(0.84, 0.07, 1), 0.84941, places=5)
        self.assertLess(taker_fee_per_share(0.95, 0.25, 2), taker_fee_per_share(0.5, 0.25, 2))

    def test_volatility_recovers_known_step_size(self):
        vol = VolatilityEstimator(1800)
        price = 100.0
        for t in range(600):
            price *= 1.001 if t % 2 else 1 / 1.001
            vol.add(float(t), price)
        self.assertAlmostEqual(vol.per_sqrt_second(), 0.001, places=5)
        self.assertEqual(vol.history_seconds(), 599)

    def test_twap_weights_by_time(self):
        vol = VolatilityEstimator(1800)
        vol.add(0.0, 100.0)
        vol.add(30.0, 110.0)
        vol.add(50.0, 120.0)
        self.assertAlmostEqual(vol.twap(0, 60), (100 * 30 + 110 * 20 + 120 * 10) / 60)
        self.assertAlmostEqual(vol.twap(40, 60), (110 * 10 + 120 * 10) / 20)
        self.assertIsNone(vol.twap(-60, 0))  # history doesn't reach back

    def test_twap_probability(self):
        # Long before the end it behaves like the spot model with a little less variance.
        self.assertAlmostEqual(fair_up_probability_twap(100, 100, 600, 0.0005, 60), 0.5)
        far = fair_up_probability_twap(100.1, 100, 600, 0.0005, 60)
        self.assertGreater(far, fair_up_probability(100.1, 100, 600, 0.0005))
        # Inside the last minute the average so far dominates: spot dipped below open
        # 5s before the end, but the minute's average is well above it.
        self.assertGreater(fair_up_probability_twap(99.9, 100, 5, 0.0005, 60, realized_avg=100.3), 0.99)
        self.assertEqual(fair_up_probability_twap(99, 100, 0, 0.0005, 60, realized_avg=100.1), 1.0)

    def test_volatility_forgets_old_samples(self):
        vol = VolatilityEstimator(60)
        for t in range(200):
            vol.add(float(t), 100.0)
        self.assertLessEqual(vol.history_seconds(), 60)


class PaperTest(unittest.TestCase):
    cfg = Config(min_edge=0.05, stake_usd=10.0, fee_rate=0.0)

    def test_buys_only_levels_with_enough_edge(self):
        asks = [(0.80, 5), (0.70, 5), (0.86, 100)]  # unsorted on purpose
        shares, cost, fee, _ = simulate_buy(asks, 100.0, 0.90, self.cfg)
        self.assertEqual(shares, 10)  # 0.86 leaves only 4 points of edge
        self.assertAlmostEqual(cost, 7.5)

    def test_stops_at_budget(self):
        shares, cost, fee, _ = simulate_buy([(0.50, 1000)], 10.0, 0.90, self.cfg)
        self.assertAlmostEqual(cost, 10.0)
        self.assertAlmostEqual(shares, 20.0)

    def test_budget_includes_fees(self):
        cfg = Config(min_edge=0.05, stake_usd=10.0)
        shares, cost, fee, _ = simulate_buy([(0.50, 1000)], 10.0, 0.90, cfg)
        self.assertGreater(fee, 0)
        self.assertAlmostEqual(cost + fee, 10.0)

    def test_picks_the_underpriced_side(self):
        fill = find_bet(0.20, [(0.30, 100)], [(0.70, 100)], self.cfg)
        self.assertEqual(fill.side, "down")
        self.assertAlmostEqual(fill.edge, 0.10)

    def test_no_bet_without_edge(self):
        self.assertIsNone(find_bet(0.5, [(0.52, 100)], [(0.50, 100)], self.cfg))

    def test_pnl(self):
        self.assertAlmostEqual(settle_pnl("up", 20, 10, 0.2, "up"), 9.8)
        self.assertAlmostEqual(settle_pnl("up", 20, 10, 0.2, "down"), -10.2)


class FakeApi:
    """Stands in for Coinbase and Polymarket."""

    def __init__(self):
        self.price = 100.0
        self.books = {"U": ([(0.49, 100)], [(0.51, 100)]), "D": ([(0.49, 100)], [(0.51, 100)])}
        self.winner = None
        self.price_to_beat = None

    def spot_price(self, asset):
        return self.price

    def find_market(self, slug):
        return {"slug": slug, "up_token": "U", "down_token": "D", "fee_rate": 0.07, "fee_exponent": 1.0,
                "min_order_size": 5, "price_to_beat": self.price_to_beat, "resolution_source": "test"}

    def order_book(self, token):
        return self.books[token]

    def market_winner(self, slug):
        return self.winner


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(assets=("btc",), vol_min_seconds=60, db_path=os.path.join(self.tmp.name, "t.sqlite"))
        self.store = Store(self.cfg.db_path)
        self.api = FakeApi()
        self.tracker = Tracker("btc", self.cfg, self.store, api=self.api)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def run_seconds(self, start, count, drift=0.0):
        for i in range(count):
            self.api.price *= 1.0005 if i % 2 else 1 / 1.0005
            self.api.price += drift
            self.tracker.step(start + i)

    def test_full_window_bets_and_settles(self):
        start = feeds.window_start(1_800_000_000, 15)
        self.run_seconds(start - 300, 300)  # warm up volatility before the window
        self.run_seconds(start, 3)  # window opens at ~100
        self.api.price = 101.0  # big jump while Polymarket still prices Up at 51c
        self.tracker.step(start + 600)
        bets = self.store.db.execute("SELECT side, fair_prob FROM bets").fetchall()
        self.assertEqual(len(bets), 1)
        self.assertEqual(bets[0][0], "up")
        # The jump itself raises measured volatility, so fair is well above 51c but not near 100%.
        self.assertGreater(bets[0][1], 0.7)

        self.tracker.step(start + 901)  # next window: closes out the old one
        self.api.winner = "up"
        settle(self.store, self.cfg, start + 2000, api=self.api)
        winner, pnl = self.store.db.execute("SELECT winner, pnl FROM bets").fetchone()
        self.assertEqual(winner, "up")
        self.assertGreater(pnl, 0)
        end_price, w = self.store.db.execute(
            "SELECT close_twap, winner FROM windows WHERE window_start = ?", (start,)).fetchone()
        self.assertGreater(end_price, 100.5)  # last minute's average, mostly at 101
        self.assertEqual(w, "up")

    def test_uses_the_markets_fee_schedule(self):
        start = feeds.window_start(1_800_000_000, 15)
        self.run_seconds(start - 120, 121)
        self.assertEqual((self.tracker.market_cfg.fee_rate, self.tracker.market_cfg.fee_exponent), (0.07, 1.0))

    def test_nowcast_moves_chainlink_by_the_fast_feeds_return(self):
        t = self.tracker
        t.on_fast(100.0, 50_000.0)
        t.on_price(100.0, 49_900.0)  # Chainlink level differs from Binance's
        t.on_fast(101.2, 50_500.0)  # Binance +1% since, Chainlink not updated yet
        price, info_ts = t.nowcast(101.3)
        self.assertAlmostEqual(price, 49_900.0 * 1.01)
        self.assertEqual(info_ts, 101.2)
        # Stale fast feed: fall back to Chainlink as is.
        price, info_ts = t.nowcast(101.2 + self.cfg.max_price_age + 1)
        self.assertEqual((price, info_ts), (49_900.0, 100.0))

    def test_volatility_prefers_the_fast_feed(self):
        t = self.tracker
        for i in range(400):
            t.on_price(float(i), 100.0)  # perfectly smooth Chainlink
            t.on_fast(float(i), 100.0 * (1.001 if i % 2 else 1.0))
        self.assertAlmostEqual(t.volatility(), 0.001, places=4)

    def test_never_bets_on_a_window_joined_midway(self):
        start = feeds.window_start(1_800_000_000, 15)
        self.run_seconds(start + 100, 300)
        self.api.price *= 1.05
        self.tracker.step(start + 600)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM bets").fetchone()[0], 0)

    def test_records_both_readings_of_the_open(self):
        start = feeds.window_start(1_800_000_000, 15)
        self.run_seconds(start - 120, 122)
        open_twap, open_point = self.store.db.execute(
            "SELECT open_twap, open_point FROM windows WHERE window_start = ?", (start,)).fetchone()
        self.assertIsNotNone(open_twap)
        self.assertIsNotNone(open_point)

    def test_lab_variants_bet_once_each_and_settle(self):
        tracker = Tracker("btc", self.cfg, self.store, api=self.api, main=False, variants=lab.variants_for(15))
        self.tracker = tracker
        start = feeds.window_start(1_800_000_000, 15)
        self.run_seconds(start - 300, 303)
        self.api.price = 101.0
        for t in (start + 500, start + 550, start + 870):  # several qualifying moments, same window
            tracker.step(t)
        rows = self.store.db.execute("SELECT mode, variant FROM bets").fetchall()
        self.assertTrue(rows)
        self.assertTrue(all(mode == "lab" for mode, _ in rows))
        keys = [v for _, v in rows]
        for k in set(keys):  # one bet per variant per window (several only for the several-entries kind)
            self.assertLessEqual(keys.count(k), lab.entries(lab.Variant.from_key(k)))
        # Variants only bet inside their own timing: none of the "last 60s" ones at 400s left.
        early = self.store.db.execute("SELECT variant FROM bets WHERE seconds_left > 300").fetchall()
        self.assertTrue(all(not v.endswith("-60") for (v,) in early))
        tracker.step(start + 901)
        self.api.winner = "up"
        settle(self.store, self.cfg, start + 2000, api=self.api)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM bets WHERE pnl IS NULL").fetchone()[0], 0)


class ReadinessTest(unittest.TestCase):
    cfg = Config()
    feed_ok = {"windows": 300, "agree": 297, "rate": 0.99}

    def paper(self, returns):
        paid = 10.0 * len(returns)
        pnl = sum(r * 10 for r in returns)
        return {"rows": [(r * 10, 10.0) for r in returns], "roi": pnl / paid if paid else None}

    def test_consistent_profit_passes(self):
        returns = [0.9, -1.0, 0.9, 0.9, -1.0, 0.9, 0.9] * 20  # ~36% ROI over 140 bets
        r = readiness(self.cfg, 80, self.feed_ok, self.paper(returns))
        self.assertTrue(r["ready"], [c for c in r["checks"] if not c["passed"]])

    def test_each_requirement_blocks(self):
        good = [0.9, -1.0, 0.9, 0.9, -1.0, 0.9, 0.9] * 20
        self.assertFalse(readiness(self.cfg, 10, self.feed_ok, self.paper(good))["ready"])
        bad_feed = {"windows": 300, "agree": 270, "rate": 0.90}
        self.assertFalse(readiness(self.cfg, 80, bad_feed, self.paper(good))["ready"])
        self.assertFalse(readiness(self.cfg, 80, self.feed_ok, self.paper(good[:50]))["ready"])
        # Profitable overall, but the latest 50 bets all lost: no longer working.
        fading = readiness(self.cfg, 80, self.feed_ok, self.paper(good * 3 + [-1.0] * 50))
        self.assertFalse(fading["ready"])
        self.assertEqual([c["id"] for c in fading["checks"] if not c["passed"]], ["recent"])

    def test_small_noisy_profit_is_called_luck(self):
        # Coin-flip-like results with a thin average profit: ROI passes but the lower bound doesn't.
        returns = [1.0, -1.0] * 60 + [0.8] * 2
        r = readiness(self.cfg, 80, self.feed_ok, self.paper(returns))
        luck = next(c for c in r["checks"] if c["id"] == "luck")
        self.assertFalse(luck["passed"])
        self.assertFalse(r["ready"])


class FakeClob:
    """Polymarket's SDK client as the live trader uses it: fill-or-kill buys that fill."""
    def __init__(self, response=None):
        self.orders = []
        self.response = response

    def place_market_order(self, **order):
        self.orders.append(order)
        if self.response is not None:
            return self.response
        spent = Decimal(str(order["amount"]))
        return types.SimpleNamespace(ok=True, order_id="0xabc", status="matched", making_amount=spent,
                                     taking_amount=spent / Decimal(str(order["max_price"])))

    def cancel_all(self):
        pass


class LiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(db_path=os.path.join(self.tmp.name, "t.sqlite"))
        self.store = Store(self.cfg.db_path)
        self.fill = find_bet(0.9, [(0.70, 100)], [(0.31, 100)], self.cfg)
        self.market = {"up_token": "U", "down_token": "D", "min_order_size": 5, "tick_size": 0.01}

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def trader(self, key="0xkey"):
        t = LiveTrader(self.cfg, self.store, env={"POLYMARKET_PRIVATE_KEY": key} if key else {})
        t.client = FakeClob()
        return t

    def live_bets(self):
        return self.store.db.execute("SELECT status, shares, cost FROM bets WHERE mode = 'live'").fetchall()

    def test_cannot_switch_on_before_readiness_or_without_wallet(self):
        t = self.trader()
        with self.assertRaises(ValueError):
            t.arm(True)
        t.set_readiness(True)
        t.arm(True)
        self.assertTrue(t.armed())
        with self.assertRaises(ValueError):
            no_wallet = self.trader(key=None)
            no_wallet.set_readiness(True)
            no_wallet.arm(True)

    def test_trades_only_when_ready_and_armed(self):
        t = self.trader()
        self.assertIsNone(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market))
        t.set_readiness(True)
        self.assertIsNone(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market))  # not armed
        t.arm(True)
        self.assertEqual(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market), "filled")
        order = t.client.orders[0]
        self.assertEqual((order["token_id"], order["side"], order["max_price"], order["order_type"]), ("U", "BUY", 0.70, "FAK"))  # (a strategy's own order: what there is)
        self.assertLessEqual(order["amount"], t.stake())
        status, shares, cost = self.live_bets()[0]
        self.assertEqual(status, "filled")
        self.assertAlmostEqual(cost, order["amount"], places=6)

    def test_a_refused_order_is_recorded_with_its_reason(self):
        t = self.trader()
        t.client = FakeClob(types.SimpleNamespace(ok=False, code="fok_not_filled", message="couldn't be fully filled"))
        t.set_readiness(True)
        t.arm(True)
        self.assertEqual(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market), "rejected")
        self.assertIn("fok_not_filled", t.last_error)
        self.assertEqual(self.live_bets()[0][0], "rejected")

    def test_skips_orders_under_minimum_size(self):
        t = self.trader()
        t.set_readiness(True)
        t.arm(True)
        t.update_settings(stake=1)  # $1 at 70c is ~1.4 shares, under the 5-share minimum
        self.assertIsNone(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market))
        self.assertEqual(self.live_bets(), [])

    def test_losing_readiness_switches_live_off(self):
        t = self.trader()
        t.set_readiness(True)
        t.arm(True)
        self.assertTrue(t.trades("crypto"))
        t.set_readiness(False)
        self.assertFalse(t.trades("crypto"))  # stops by itself; the switch stays where the owner put it

    def test_daily_limit_counts_open_bets(self):
        t = self.trader()
        t.update_settings(stake=5, daily_loss_limit=12)
        import time as _time
        for _ in range(2):
            self.store.add_bet(_time.time(), "btc", 15, 0, 300, self.fill, mode="live")  # 2 open bets, ~$10 at risk
        self.assertGreater(t.today_exposure(), 9)
        t.set_readiness(True)
        t.arm(True)
        self.assertIsNone(t.maybe_trade(_time.time(), "btc", 15, 0, 300, self.fill, self.market))

    def test_stake_limits(self):
        t = self.trader()
        with self.assertRaises(ValueError):
            t.update_settings(stake=self.cfg.live_max_stake_usd + 1)
        t.update_settings(stake=7)
        self.assertEqual(t.stake(), 7)


class EngineStateTest(unittest.TestCase):
    def test_state_is_json_ready(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            streaming=False, with_lab=False)
            start = feeds.window_start(1_800_000_000, 15)
            for i in range(5):
                engine.trackers[0].step(start + i)
            engine.refresh_analysis()
            engine.refresh_lab()
            state = json.loads(json.dumps(engine.state()))
            self.assertFalse(state["readiness"]["ready"])
            self.assertFalse(state["live"]["status"]["armed"])
            self.assertEqual(state["assets"][0]["price"], engine.trackers[0].snapshot["price"])
            engine.store.close()


class LabRankingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(db_path=os.path.join(self.tmp.name, "t.sqlite"))
        self.store = Store(self.cfg.db_path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add(self, asset, minutes, variant, results, t0=0.0):
        fill = find_bet(0.9, [(0.5, 1000)], [(0.5, 1000)], Config(fee_rate=0.0))
        for i, won in enumerate(results):
            self.store.add_bet(t0 + i, asset, minutes, int(t0) + i * 900, 100, fill, mode="lab", variant=variant)
            bet_id = self.store.db.execute("SELECT MAX(id) FROM bets").fetchone()[0]
            side = "up"
            self.store.settle_bet(bet_id, side if won else "down",
                                  settle_pnl(side, fill.shares, fill.cost, fill.fee, side if won else "down"))

    def board(self, candidates=None):
        db = connect(self.cfg.db_path)
        try:
            return lab.leaderboard(db, self.cfg, candidates=candidates)
        finally:
            db.close()

    def status(self, result):
        return {(b["asset"], b["minutes"]): b["status"] for b in result["board"]}

    def test_the_table_from_a_process_of_its_own_is_the_same(self):
        # (the running server counts the lab in a separate process, so pages answer meanwhile)
        import concurrent.futures
        import multiprocessing
        self.add("btc", 15, "twap|0.05|15-180", [True, False, True] * 10)
        self.add("eth", 5, "end|0.00|45-60|b90-95", [True] * 12)
        args = (self.cfg.db_path, self.cfg, None, {}, set(), None, None)
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            there = pool.submit(lab.board_from, *args).result(timeout=120)
        here = lab.board_from(*args)
        self.assertEqual(there, here)
        self.assertEqual(here["board"], self.board()["board"])
        self.assertIn("end_pooled", here)

    def test_stages_from_collecting_to_proven_on_new_data(self):
        strong = "twap|0.05|15-180"
        self.add("btc", 15, strong, [True, True, True, False] * 25)  # 75% wins at 50c, 100 trades: a checkpoint
        self.add("eth", 15, strong, [True] * 10)  # perfect but only 10 bets
        self.add("sol", 5, strong, [True, False] * 50)  # coin flip at 50c: no edge
        first = self.board()
        status = self.status(first)
        self.assertEqual(status[("btc", 15)], "candidate")
        self.assertTrue(next(b for b in first["board"] if b["asset"] == "btc")["new_candidate"])
        self.assertEqual(status[("eth", 15)], "collecting")
        self.assertNotIn(status[("sol", 5)], ("candidate", "proven"))
        self.assertEqual(first["picks"], [])  # nothing is proven on the data that picked it

        candidates = {("btc", 15, strong): 10_000.0}
        self.add("btc", 15, strong, [True, True, True, False] * 5, t0=10_000.0)  # 20 new bets: not yet enough
        self.assertEqual(self.status(self.board(candidates))[("btc", 15)], "candidate")
        self.add("btc", 15, strong, [True, True, True, False] * 15, t0=20_000.0)  # 80 new: before the first checkpoint
        self.assertEqual(self.status(self.board(candidates))[("btc", 15)], "candidate")
        self.add("btc", 15, strong, [True, True, True, False] * 5, t0=30_000.0)  # 100 new, still winning
        proven = self.board(candidates)
        self.assertEqual(self.status(proven)[("btc", 15)], "proven")
        self.assertEqual([(p["asset"], p["minutes"]) for p in proven["picks"]], [("btc", 15)])

    def test_candidate_that_loses_on_new_data_fails(self):
        v = "twap|0.05|15-180"
        self.add("btc", 15, v, [True, True, True, False] * 25)
        self.add("btc", 15, v, [False, False, True] * 34, t0=10_000.0)
        self.assertEqual(self.status(self.board({("btc", 15, v): 10_000.0}))[("btc", 15)], "failed")

    def test_picks_are_on_different_markets(self):
        keys = ("twap|0.05|15-180", "twap|0.08|15-180", "twap|0.03|15-180")
        candidates = {}
        for variant in keys:
            self.add("btc", 15, variant, [True, True, True, False] * 25, t0=10_000.0)
            candidates[("btc", 15, variant)] = 0.0
        self.add("eth", 5, "twap|0.05|10-120", [True, True, True, False] * 25, t0=10_000.0)
        candidates[("eth", 5, "twap|0.05|10-120")] = 0.0
        picks = self.board(candidates)["picks"]
        self.assertEqual(sorted((p["asset"], p["minutes"]) for p in picks), [("btc", 15), ("eth", 5)])

    def test_new_data_proves_only_on_fills_like_a_real_orders(self):
        v = "twap|0.05|15-180"
        self.add("btc", 15, v, [True, True, True, False] * 15)
        self.add("btc", 15, v, [True, True, True, False] * 15, t0=10_000.0)  # 60 new bets, but before real fills
        db = connect(self.cfg.db_path)
        try:
            board = lab.leaderboard(db, self.cfg, candidates={("btc", 15, v): 10_000.0}, real_since=50_000.0)
        finally:
            db.close()
        row = board["board"][0]
        self.assertEqual((row["status"], row["oos_bets"]), ("candidate", 0))

    def test_neighbours(self):
        near = lab.neighbours("twap|0.05|15-180", 15)
        self.assertIn("twap|0.04|15-180", near)
        self.assertIn("twap|0.07|15-180", near)
        self.assertIn("twap|0.05|15-120", near)
        self.assertIn("twap|0.05|15-270", near)
        self.assertNotIn("twap|0.05|15-180", near)
        self.assertTrue(all(lab.Variant.from_key(k).max_seconds_left <= 15 * 60 - 30 for k in near))

    def test_rule_descriptions(self):
        self.assertIn("5 נקודות", lab.describe("twap|0.05|15-180"))
        self.assertIn("3 הדקות האחרונות", lab.describe("twap|0.05|15-180"))
        self.assertIn("45 השניות האחרונות", lab.describe("point|0.12|10-45"))
        # 12 rules + 6 maker + 8 several-entries rules + 4 both-sides + 3 makers on the leader + 9 rules
        # with their own price group and least chance (the last minute's 8 and the 6 leaders at the end
        # no longer run), and the 4 consensus rules (29.10.2026)
        self.assertEqual(len(lab.variants_for(15)), 60)  # (+ the 14 lock rules, 30.10.2026)
        self.assertFalse(any(lab.is_lock(v) for v in lab.variants_for(15)))
        # The makers have their own tab: a resting order on the model's price, or on the market's leader.
        self.assertEqual(lab.family("twapm|0.05|15-600"), "maker")
        self.assertEqual(lab.family("fav60|0.70|180-300"), "maker")
        self.assertEqual(lab.family("twap3|0.05|15-600"), "main")
        self.assertEqual(lab.family("twap|0.05|15-600"), "main")
        self.assertFalse(any(lab.is_lotto(v) for v in lab.variants_for(15)))  # the lottery tickets are gone
        self.assertIn("שני הצדדים", lab.describe("pair94|0.35|120-885"))


class ResearchTest(unittest.TestCase):
    def test_stores_structured_summary_and_errors(self):
        from unittest import mock
        from polybot import research

        data = research.Research(headline="h", key_findings=["f"], top_picks=[], warnings=["w"],
                                 next_experiments=["e"])

        class FakeResponse:
            stop_reason = "end_turn"
            parsed_output = data
            model = "claude-opus-5-5"

            class usage:
                input_tokens, output_tokens = 10_000, 5_000
                cache_creation_input_tokens = cache_read_input_tokens = 0

        sent = {}

        class FakeClient:
            class beta:
                class messages:
                    @staticmethod
                    def parse(**kwargs):
                        sent.update(kwargs)
                        return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            lab_state = {"board": [], "markets": [], "settled_bets": 0}
            with mock.patch("anthropic.Anthropic", return_value=FakeClient()):
                research.run(store, lab_state, None, {}, 1.0)
            self.assertEqual(sent["model"], "claude-opus-5-5")
            self.assertIs(sent["output_format"], research.Research)
            db = connect(store.path)
            latest = research.latest(db)
            self.assertEqual(latest["data"]["headline"], "h")
            # $4 per million in, $20 per million out: 0.04 + 0.10
            self.assertAlmostEqual(research.spent_since(db, 0), 0.14)
            FakeResponse.stop_reason, FakeResponse.parsed_output = "refusal", None
            with mock.patch("anthropic.Anthropic", return_value=FakeClient()):
                with self.assertRaises(RuntimeError):
                    research.run(store, lab_state, None, {}, 1.0)
            self.assertIn("refusal", research.latest(db)["error"])
            db.close()
            store.close()


class FeedsTest(unittest.TestCase):
    def test_market_winner_asks_for_closed_markets_and_needs_a_decided_price(self):
        from unittest import mock
        replies = {
            "closed": [{"closed": True, "outcomes": '["Up", "Down"]', "outcomePrices": '["0", "1"]'}],
            "open": [],
        }
        seen = []

        def fake_get(url, timeout=3.0):
            seen.append(url)
            return replies["closed"] if "closed=true" in url else replies["open"]

        with mock.patch.object(feeds, "get_json", fake_get):
            self.assertEqual(feeds.market_winner("btc-updown-15m-1"), "down")
            self.assertIn("closed=true", seen[0])
            replies["closed"] = []
            self.assertIsNone(feeds.market_winner("btc-updown-15m-1"))


class WebAuthTest(unittest.TestCase):
    def setUp(self):
        import threading
        from http.server import ThreadingHTTPServer
        from polybot.web import make_handler

        class FakeEngine:
            def state(self):
                return {"ok": 1}

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), "  סיסמה-Secret1 "))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, path, data=None, cookie=None):
        import urllib.error
        import urllib.request

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        req = urllib.request.Request(self.base + path, data=data)
        if cookie:
            req.add_header("Cookie", cookie)
        try:
            with opener.open(req, timeout=5) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_login_flow_with_hebrew_password(self):
        import urllib.parse
        code, headers, _ = self.request("/")
        self.assertEqual((code, headers["Location"]), (303, "/login"))
        self.assertEqual(self.request("/api/state")[0], 401)
        self.assertEqual(self.request("/healthz")[0], 200)

        code, _, body = self.request("/login", urllib.parse.urlencode({"password": "wrong"}).encode())
        self.assertEqual(code, 401)
        self.assertIn("סיסמה שגויה".encode(), body)

        code, headers, _ = self.request("/login", urllib.parse.urlencode({"password": "סיסמה-Secret1"}).encode())
        self.assertEqual(code, 303)
        cookie = headers["Set-Cookie"].split(";")[0]
        self.assertNotIn("Secret1", cookie)
        code, _, body = self.request("/api/state", cookie=cookie)
        self.assertEqual((code, json.loads(body)), (200, {"ok": 1}))
        self.assertEqual(self.request("/api/state", cookie="polybot_session=forged")[0], 401)

    def test_the_page_is_kept_by_the_browser_and_checked_for_a_new_version(self):
        import urllib.error
        import urllib.parse
        import urllib.request
        code, headers, _ = self.request("/login", urllib.parse.urlencode({"password": "סיסמה-Secret1"}).encode())
        cookie = headers["Set-Cookie"].split(";")[0]
        code, headers, body = self.request("/", cookie=cookie)
        self.assertEqual(code, 200)
        self.assertIn(b"<html", body.lower())
        self.assertEqual(headers["Cache-Control"], "no-cache")
        req = urllib.request.Request(self.base + "/", headers={"Cookie": cookie, "If-None-Match": headers["ETag"]})
        with self.assertRaises(urllib.error.HTTPError) as same:  # (urllib reads a 304 as an error)
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(same.exception.code, 304)  # unchanged: nothing sent again

    def test_wrong_passwords_are_rate_limited(self):
        import urllib.parse
        data = urllib.parse.urlencode({"password": "nope"}).encode()
        codes = [self.request("/login", data)[0] for _ in range(11)]
        self.assertEqual(codes[:10], [401] * 10)
        self.assertEqual(codes[10], 429)


class WatchdogTest(unittest.TestCase):
    def test_silent_price_topic_is_closed_for_reconnect(self):
        import asyncio
        import time as _time

        class FakeWs:
            closed = False

            async def close(self):
                self.closed = True

        stream = feeds.ChainlinkStream({"btc": "btc/usd"}, lambda *a: None, stale_after=1.0)
        stream.last_price = _time.time() - 5  # Chainlink silent for 5s
        ws = FakeWs()
        asyncio.run(asyncio.wait_for(stream._watch(ws, "crypto_prices_chainlink"), 5))
        self.assertTrue(ws.closed)
        self.assertEqual(stream.reconnects, 1)
        self.assertIn("reconnecting", stream.last_error)

    def test_dead_price_feed_raises_a_plain_alert(self):
        import time as _time
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            with_lab=False)
            engine.started_at = _time.time() - 600
            engine.prices = feeds.ChainlinkStream({"btc": "btc/usd"}, lambda *a: None)
            engine.prices.last_price = _time.time() - 120
            alerts = engine._alerts(_time.time())
            engine.prices.last_price = _time.time()
            quiet = engine._alerts(_time.time())
            engine.store.close()
        self.assertTrue(any("Chainlink" in a and "120" in a for a in alerts))
        self.assertEqual([a for a in quiet if "Chainlink" in a], [])


class NotifyTest(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        self.tmp = tempfile.TemporaryDirectory()
        self.sent = []
        sent = self.sent

        class FakeSMTP:
            def __init__(self, host, port, timeout=None):
                self.host = host

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def starttls(self):
                pass

            def login(self, user, password):
                assert password == "abcdabcdabcdabcd"  # spaces from Gmail's display removed

            def send_message(self, msg):
                sent.append(msg)

        self.patch = mock.patch("smtplib.SMTP", FakeSMTP)
        self.patch.start()
        self.api = FakeApi()
        env = {"SMTP_USER": "bot@gmail.com", "SMTP_PASSWORD": "abcd abcd abcd abcd"}
        self.engine = Engine(Config(assets=("btc",), vol_min_seconds=60, db_path=os.path.join(self.tmp.name, "t.sqlite")),
                             api=self.api, env=env, streaming=False, with_lab=False)
        self.engine.notifier.update(email="owner@example.com", open=True, close=True)
        self.t = self.engine.trackers[0]
        self.start = feeds.window_start(1_800_000_000, 15)

    def tearDown(self):
        self.patch.stop()
        self.engine.store.close()
        self.tmp.cleanup()

    def open_a_trade(self):
        for i in range(-300, 3):
            self.api.price *= 1.0005 if i % 2 else 1 / 1.0005
            self.t.step(self.start + i)
        self.api.price = 101.0
        self.t.step(self.start + 600)

    def html(self, msg):
        return msg.get_body(("html",)).get_content()

    def test_open_mail_has_the_card_and_chart_then_close_mail_has_the_result(self):
        self.assertEqual(self.engine.notifier.host, "smtp.gmail.com")  # inferred from a Gmail address
        self.open_a_trade()
        self.engine.notifier.tick(self.start + 601)
        self.engine.notifier.tick(self.start + 620)
        self.assertEqual(len(self.sent), 0)  # waits for 30 quiet seconds
        self.engine.notifier.tick(self.start + 632)
        self.assertEqual(len(self.sent), 1)
        opened = self.sent[0]
        self.assertTrue(opened["Subject"].startswith("📈 ") or opened["Subject"].startswith("📉 "))
        self.assertNotIn("כסף אמיתי", opened["Subject"])  # simulation: no label
        self.assertEqual(opened["To"], "owner@example.com")
        body = self.html(opened)
        self.assertIn(">ביטקוין<", body)  # the card is titled by the coin alone
        self.assertIn("#1f8a4c", body)  # our side's button is green
        images = [p for p in opened.walk() if p.get_content_type() == "image/png"]
        self.assertEqual(len(images), 1)
        self.assertIn(images[0]["Content-ID"][1:-1], body)

        self.engine.notifier.tick(self.start + 700)
        self.assertEqual(len(self.sent), 1)  # nothing new, nothing re-sent

        self.t.step(self.start + 901)
        self.api.winner = "up"
        settle(self.engine.store, self.engine.cfg, self.start + 2000, api=self.api)
        self.engine.notifier.tick(self.start + 2001)
        self.engine.notifier.tick(self.start + 2032)
        self.assertEqual(len(self.sent), 2)
        closed = self.sent[1]
        self.assertTrue(closed["Subject"].startswith("✅ \u2066+$"))
        self.assertIn("#1f8a4c", self.html(closed))

    def test_a_real_money_trade_is_mailed_while_real_money_is_on(self):
        from unittest import mock
        self.open_a_trade()  # (also brings the market's window up)
        self.engine.notifier.tick(self.start + 601)
        self.engine.notifier.tick(self.start + 632)
        self.sent.clear()
        fill = types.SimpleNamespace(side="up", fair_prob=0.7, best_ask=0.5, avg_price=0.5, shares=10, cost=5.0, fee=0.05,
                                     edge=0.15, depth=None)
        self.engine.store.add_bet(self.start + 640, "btc", 15, self.start, 260, fill, mode="live", variant="twap|0.05|15-600")
        with mock.patch.object(self.engine.live, "armed", return_value=True):
            self.engine.notifier.tick(self.start + 641)
            self.engine.notifier.tick(self.start + 672)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("כסף אמיתי", self.sent[0]["Subject"])
        # "Every 5 trades" is for the simulation's many: a single real trade is still mailed.
        self.engine.notifier.update(open_every=5)
        self.sent.clear()
        self.engine.store.add_bet(self.start + 680, "btc", 15, self.start, 220, fill, mode="live", variant="twap|0.05|15-600")
        with mock.patch.object(self.engine.live, "armed", return_value=True):
            self.engine.notifier.tick(self.start + 681)
            self.engine.notifier.tick(self.start + 712)
        self.assertEqual(len(self.sent), 1)
        self.engine.notifier.update(open_every=1)
        # Real money switched off (a restart, the day's limit) before its trade was mailed: still mailed.
        self.sent.clear()
        self.engine.store.add_bet(self.start + 700, "btc", 15, self.start, 200, fill, mode="live", variant="twap|0.05|15-600")
        with mock.patch.object(self.engine.live, "armed", return_value=False):
            self.engine.notifier.tick(self.start + 701)
            self.engine.notifier.tick(self.start + 732)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("כסף אמיתי", self.sent[0]["Subject"])

    def test_each_real_trade_says_whether_its_mails_went_and_can_send_them_again(self):
        from unittest import mock
        self.open_a_trade()
        self.engine.notifier.tick(self.start + 601)
        self.engine.notifier.tick(self.start + 632)
        self.sent.clear()
        fill = types.SimpleNamespace(side="up", fair_prob=0.7, best_ask=0.5, avg_price=0.5, shares=10, cost=5.0, fee=0.05,
                                     edge=0.15, depth=None)
        self.engine.store.add_bet(self.start + 640, "btc", 15, self.start, 260, fill, mode="live", variant="twap|0.05|15-600")
        with mock.patch.object(self.engine.live, "armed", return_value=True):
            self.engine.notifier.tick(self.start + 641)
            self.engine.notifier.tick(self.start + 672)
        bet = self.engine.state()["live"]["bets"][0]
        self.assertTrue(json.loads(bet["mail_open"])["ok"])
        self.assertIsNone(bet["mail_close"])  # its result comes after the round is decided
        with self.assertRaises(ValueError):
            self.engine.notifier.resend(bet["id"], "close")
        self.sent.clear()
        self.engine.notifier.resend(bet["id"], "open")
        self.assertEqual(len(self.sent), 1)
        # With the closing mail switched off, the result is handled without a mail, and says so.
        self.engine.notifier.update(close_live=False)
        self.api.winner = "up"
        settle(self.engine.store, self.engine.cfg, self.start + 2000, api=self.api)
        with mock.patch.object(self.engine.live, "armed", return_value=True):
            self.engine.notifier.tick(self.start + 2001)
        why = json.loads(self.engine.state()["live"]["bets"][0]["mail_close"])
        self.assertFalse(why["ok"])
        self.assertIn("כבוי בהגדרות", why["why"])

    def test_a_refused_real_order_is_not_mailed_and_its_reason_is_on_the_page(self):
        from unittest import mock
        self.open_a_trade()
        self.engine.notifier.tick(self.start + 601)
        self.engine.notifier.tick(self.start + 632)
        self.sent.clear()
        fill = types.SimpleNamespace(side="up", fair_prob=0.7, best_ask=0.76, avg_price=0.76, shares=6.5, cost=4.94, fee=0.06,
                                     edge=0.1, depth=None)
        self.engine.store.add_bet(self.start + 640, "btc", 15, self.start, 260, fill, mode="live", status="rejected",
                                  error="fok_not_filled: order couldn't be fully filled. FOK orders are fully filled or killed.",
                                  variant="twap|0.05|15-600")
        with mock.patch.object(self.engine.live, "armed", return_value=True):
            self.engine.notifier.tick(self.start + 641)
            self.engine.notifier.tick(self.start + 672)
        self.assertEqual(self.sent, [])
        self.assertIn("המחיר זז", self.engine.state()["live"]["bets"][0]["why"])

    def test_switched_off_mails_are_skipped_and_never_flood_later(self):
        self.engine.notifier.update(open=False)
        self.open_a_trade()
        self.engine.notifier.tick(self.start + 601)
        self.engine.notifier.update(open=True)
        self.engine.notifier.tick(self.start + 602)
        self.assertEqual(self.sent, [])

    def test_daily_summary(self):
        self.open_a_trade()
        subject, body = self.engine.notifier.daily_mail(self.start + 700)
        self.assertTrue(subject.startswith("📊 סיכום היום: "))
        self.assertIn("כל הטריידים של היום", body)
        self.assertIn("טריידים נפתחו היום", body)
        self.assertIn("החשבון עוד לא מחובר", body)
        self.assertIn("מה המערכת למדה היום", body)
        self.assertIn("עוד חסר לכסף אמיתי" if self.engine.readiness else "אסטרטגיה", body)

    def test_per_trade_mails_are_off_by_default(self):
        fresh = self.engine.notifier
        fresh.store.set_setting("notify_open", None)
        fresh.store._x("DELETE FROM settings WHERE key IN ('notify_open', 'notify_close')")
        self.assertFalse(fresh.settings()["open"])
        self.assertFalse(fresh.settings()["close"])
        self.assertTrue(fresh.settings()["daily"])

    def test_bad_address_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.notifier.update(email="not-an-email")


class HealthTest(unittest.TestCase):
    def test_report_and_book_cross_check(self):
        from unittest import mock
        import time as _time
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            with_lab=False)
            t = engine.trackers[0]
            t.market = {"up_token": "U", "down_token": "D"}

            class Books:
                reconnects, last_message, tokens, lag = 0, None, set(), None

                def get(self, token):
                    return [(0.49, 10)], [(0.52, 10)], _time.time()

            engine.books = Books()
            with mock.patch.object(feeds, "order_book", return_value=([(0.49, 5)], [(0.51, 5)])), \
                 mock.patch.object(feeds, "get_json", return_value=str(int(_time.time()))):
                engine.health.check()
            now = _time.time()
            for i in range(10):
                engine.health.record_latency(now, 0.02 + i / 1000)
            report = engine.health.report(now)
            engine.store.close()
        match = report["book_match"]
        self.assertEqual(match["checks"], 1)
        self.assertAlmostEqual(match["max_gap"], 0.01)
        self.assertEqual(match["within_1c"], 1)
        self.assertAlmostEqual(report["reaction"]["p50"], 0.025, places=3)
        self.assertIsNotNone(report["clock_offset"])
        names = {c["name"] for c in report["checks"]}
        self.assertTrue({"price", "books", "reaction", "match", "clock", "alerts"} <= names)

    def test_problem_mail_after_two_minutes_and_recovery_mail(self):
        from unittest import mock
        sent = []

        class FakeSMTP:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *e): return False
            def starttls(self): pass
            def login(self, *a): pass
            def send_message(self, msg): sent.append(msg["Subject"])

        with tempfile.TemporaryDirectory() as tmp, mock.patch("smtplib.SMTP", FakeSMTP):
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(),
                            env={"SMTP_USER": "b@gmail.com", "SMTP_PASSWORD": "x"}, streaming=False, with_lab=False)
            engine.notifier.update(email="o@example.com")
            alerts = ["מחירי Chainlink לא מגיעים"]
            with mock.patch.object(engine, "_alerts", lambda now: alerts):
                engine.notifier._watch_problems(1000.0, True)
                engine.notifier._watch_problems(1060.0, True)
                self.assertEqual(sent, [])  # a one-minute blip mails nobody
                engine.notifier._watch_problems(1130.0, True)
                engine.notifier._watch_problems(1200.0, True)
                alerts.clear()
                engine.notifier._watch_problems(1300.0, True)
            engine.store.close()
        self.assertEqual(sent, ["⚠️ תקלה בבוט פולימרקט", "🟢 הבוט חזר לפעול"])


class ExchangeFeedTest(unittest.TestCase):
    def test_binance_book_ticker_mid(self):
        got = []
        stream = feeds.BinanceStream({"btc": "BTCUSDT"}, lambda *a: got.append(a), min_interval=0)
        self.assertIn("btcusdt@bookTicker", stream.url)
        stream._handle(json.dumps({"stream": "btcusdt@bookTicker", "data": {
            "u": 1, "s": "BTCUSDT", "b": "81000.00", "B": "1", "a": "81002.00", "A": "1"}}))
        self.assertEqual(got[0][0:2], ("binance", "btc"))
        self.assertAlmostEqual(got[0][3], 81001.0)

    def test_coinbase_ticker_mid_and_its_own_timestamp(self):
        got = []
        stream = feeds.CoinbaseStream({"eth": "ETH-USD"}, lambda *a: got.append(a), min_interval=0)
        self.assertEqual(stream.subscribe_messages(),
                         [{"type": "subscribe", "product_ids": ["ETH-USD"], "channels": ["ticker"]}])
        stream._handle(json.dumps({"type": "ticker", "product_id": "ETH-USD", "price": "2400.10",
                                   "best_bid": "2400.00", "best_ask": "2400.20", "time": "2026-10-23T21:57:43.500000Z"}))
        self.assertEqual(got[0][0:2], ("coinbase", "eth"))
        self.assertAlmostEqual(got[0][3], 2400.10)
        self.assertAlmostEqual(got[0][2] % 60, 43.5, places=3)
        stream._handle(json.dumps({"type": "error", "message": "Failed to subscribe", "reason": "HYPE-USD is not a valid product"}))
        self.assertIn("HYPE-USD", stream.last_error)

    def test_rate_limit_keeps_the_newest_price(self):
        got = []
        stream = feeds.BinanceStream({"btc": "BTCUSDT"}, lambda *a: got.append(a), min_interval=60)
        for mid in (100, 101, 102):
            stream._handle(json.dumps({"data": {"s": "BTCUSDT", "b": str(mid - 1), "a": str(mid + 1)}}))
        self.assertEqual([round(g[3]) for g in got], [100])  # 101 and 102 held back
        stream.last_tick["btc"] = 0  # the interval has passed
        stream.flush()
        self.assertEqual([round(g[3]) for g in got], [100, 102])

    def test_nowcast_averages_the_fresh_sources(self):
        cfg = Config()
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            t = Tracker("btc", cfg, store, api=FakeApi())
            t.on_fast(99.5, 50_000.0, "binance")
            t.on_fast(99.6, 49_990.0, "coinbase")
            t.on_price(100.0, 49_900.0)
            t.on_fast(100.8, 50_500.0, "binance")  # +1.0%
            t.on_fast(100.9, 50_239.9, "coinbase")  # +0.5%
            price, info_ts = t.nowcast(101.0)
            store.close()
        self.assertAlmostEqual(price, 49_900.0 * (1.01 + 1.005) / 2, places=1)
        self.assertEqual(info_ts, 100.9)


class LearningTest(unittest.TestCase):
    def test_calibration_recovers_an_overconfident_model(self):
        import random
        from statistics import NormalDist
        from polybot import calibration
        random.seed(7)
        nd = NormalDist()
        rows = []
        for _ in range(3000):
            z_true = random.gauss(0, 1)
            p_true = nd.cdf(z_true)
            p_model = nd.cdf(z_true / 0.7)  # the model's z-scores are 1/0.7 too large
            rows.append((p_model, random.random() < p_true, 1.0))
        k, loss_k, loss_1 = calibration.fit(rows)
        self.assertAlmostEqual(k, 0.7, delta=0.06)
        self.assertLess(loss_k, loss_1)
        self.assertAlmostEqual(calibration.apply(0.5, 0.7), 0.5)
        self.assertLess(calibration.apply(0.9, 0.7), 0.9)
        self.assertEqual(calibration.apply(0.9, 1.0), 0.9)

    def test_calibration_waits_for_enough_evidence(self):
        from polybot import calibration
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            db = connect(cfg.db_path)
            out = calibration.calibrate(db, cfg)
            db.close()
            store.close()
        self.assertEqual((out["k"], out["applied"]), (1.0, False))

    def test_promote_moves_a_proven_rule_into_the_main_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc", "xrp"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            self.assertEqual([t.key for t in engine.main_trackers], ["btc-15m"])
            with self.assertRaises(ValueError):
                engine.promote("xrp", 5, "twap|0.05|10-120")  # not proven
            engine.lab = {"board": [{"asset": "xrp", "minutes": 5, "variant": "twap|0.05|10-120", "status": "proven"}]}
            engine.refresh_lab = lambda: None
            engine.promote("xrp", 5, "twap|0.05|10-120")
            keys = sorted(t.key for t in engine.main_trackers)
            xrp = next(t for t in engine.trackers if t.key == "xrp-5m")
            self.assertEqual(keys, ["btc-15m", "xrp-5m"])
            self.assertEqual(xrp.main_rule.max_seconds_left, 120)
            self.assertAlmostEqual(xrp.main_cfg.min_edge, 0.05)
            self.assertEqual(xrp.record_seconds, 1.0)
            self.assertGreater(float(engine.store.get_setting("main_changed_at")), 0)
            # Survives a restart.
            engine.store.close()
            again = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            self.assertEqual(sorted(t.key for t in again.main_trackers), ["btc-15m", "xrp-5m"])
            again.refresh_lab = lambda: None
            again.demote("btc", 15)
            self.assertEqual([t.key for t in again.main_trackers], ["xrp-5m"])
            with self.assertRaises(ValueError):
                again.demote("xrp", 5)  # at least one must stay
            again.store.close()

    def test_explore_adds_neighbours_of_leaders_and_retires_losers(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",), lab_minutes=(15,))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.store.set_setting("lab_focus", "all"); engine._refresh_lab_variants()  # (the whole grid: the lab's focus on the locked pair is off)
            leader = {"asset": "btc", "minutes": 15, "variant": "twap|0.05|15-180", "status": "promising",
                      "bets": 40, "roi": 0.2, "roi_pessimistic": 0.05}
            engine.explore([leader], 10_000.0)
            added = {v for (a, m, v) in engine.dynamic_variants}
            self.assertIn("twap|0.04|15-180", added)
            btc = next(t for t in engine.trackers if t.key == "btc-15m")
            self.assertIn("twap|0.04|15-180", {v.key for v in btc.variants})
            self.assertIn("twap|0.04|15-180", btc.lab_cfgs)
            loser = {"asset": "btc", "minutes": 15, "variant": "twap|0.04|15-180", "status": "losing",
                     "bets": 40, "roi": -0.2, "roi_pessimistic": -0.3}
            engine.explore([loser], 10_000.0 + cfg.explore_every_seconds)
            self.assertNotIn("twap|0.04|15-180", {v for (a, m, v) in engine.dynamic_variants})
            engine.store.close()


class PageScriptTest(unittest.TestCase):
    def test_dashboard_script_parses(self):
        """A syntax error in the page's script blanks the whole dashboard; catch it here."""
        import re
        import shutil
        import subprocess
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        with open(os.path.join(os.path.dirname(__file__), "..", "polybot", "static", "index.html"),
                  encoding="utf-8") as f:
            html = f.read()
        scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
        self.assertTrue(scripts)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write("\n".join(scripts))
        try:
            result = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
        finally:
            os.unlink(f.name)
        self.assertEqual(result.returncode, 0, result.stderr)


class PositionsTest(unittest.TestCase):
    def test_open_bet_appears_as_a_live_card_with_its_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeApi()
            engine = Engine(Config(assets=("btc",), vol_min_seconds=60, db_path=os.path.join(tmp, "t.sqlite")),
                            api=api, env={}, streaming=False, with_lab=False)
            t = engine.trackers[0]
            start = feeds.window_start(1_800_000_000, 15)
            for i in range(-300, 3):
                api.price *= 1.0005 if i % 2 else 1 / 1.0005
                t.step(start + i)
            api.price = 101.0
            t.step(start + 600)
            db = connect(engine.cfg.db_path)
            cards = engine._positions(db, start + 601)
            db.close()
            engine.store.close()
        self.assertEqual(len(cards["paper"]), 1)
        card = cards["paper"][0]
        self.assertTrue(card["live"])
        self.assertEqual(card["bets"][0]["side"], "up")
        self.assertEqual(card["end"], start + 900)
        self.assertGreater(len(card["series"]), 2)
        self.assertLessEqual(len(card["series"]), 130)
        self.assertEqual(card["price"], t.snapshot["price"])

    def test_card_chart_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), vol_min_seconds=60, db_path=os.path.join(tmp, "t.sqlite"))
            api = FakeApi()
            engine = Engine(cfg, api=api, env={}, streaming=False, with_lab=False)
            t = engine.trackers[0]
            start = feeds.window_start(1_800_000_000, 15)
            for i in range(-300, 3):
                api.price *= 1.0005 if i % 2 else 1 / 1.0005
                t.step(start + i)
            api.price = 101.0
            for i in range(600, 620):
                t.step(start + i)
            engine.store.close()
            # A new process: empty memory, same database.
            restarted = Engine(cfg, api=api, env={}, streaming=False, with_lab=False)
            restarted.trackers[0].step(start + 625)
            db = connect(cfg.db_path)
            cards = restarted._positions(db, start + 626)
            db.close()
            restarted.store.close()
        self.assertGreater(len(cards["paper"][0]["series"]), 10)

    def test_restart_mid_window_keeps_the_open_and_never_bets_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), vol_min_seconds=60, db_path=os.path.join(tmp, "t.sqlite"))
            api = FakeApi()
            engine = Engine(cfg, api=api, env={}, streaming=False, with_lab=False)
            t = engine.trackers[0]
            start = feeds.window_start(1_800_000_000, 15)
            for i in range(-300, 3):
                api.price *= 1.0005 if i % 2 else 1 / 1.0005
                t.step(start + i)
            api.price = 101.0
            t.step(start + 600)
            open_before = t.open_price
            engine.store.close()
            restarted = Engine(cfg, api=api, env={}, streaming=False, with_lab=False)
            t2 = restarted.trackers[0]
            for i in range(-200, 0):  # warm volatility up again; history before the window is gone
                api.price = 101.0 * (1.0005 if i % 2 else 1 / 1.0005)
                t2.vol.add(start + 400 + i, api.price)
            api.price = 101.5
            t2.step(start + 650)
            t2.step(start + 700)
            bets = restarted.store.db.execute("SELECT COUNT(*) FROM bets WHERE mode = 'paper'").fetchone()[0]
            stored = restarted.store.db.execute("SELECT open_twap FROM windows WHERE window_start = ?", (start,)).fetchone()[0]
            restarted.store.close()
        self.assertAlmostEqual(t2.open_price, open_before)
        self.assertAlmostEqual(stored, open_before)
        self.assertEqual(bets, 1)


class DashboardTest(unittest.TestCase):
    def test_periods_follow_the_owners_time_zone_and_sunday_weeks(self):
        import datetime as dt
        from zoneinfo import ZoneInfo
        from polybot.periods import period_starts
        tz = ZoneInfo("Asia/Jerusalem")
        # Tuesday 2026-10-27 01:30 in Israel is still Monday 22:30 UTC.
        now = dt.datetime(2026, 10, 27, 1, 30, tzinfo=tz).timestamp()
        starts = {k: dt.datetime.fromtimestamp(v, tz) for k, v in period_starts(now, "Asia/Jerusalem").items()}
        self.assertEqual(starts["day"], dt.datetime(2026, 10, 27, tzinfo=tz))
        self.assertEqual(starts["week"], dt.datetime(2026, 10, 25, tzinfo=tz))  # Sunday
        self.assertEqual(starts["month"], dt.datetime(2026, 10, 1, tzinfo=tz))

    def test_strategy_mix_buckets(self):
        from polybot.stats import strategy_mix
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            fill = find_bet(0.9, [(0.5, 1000)], [(0.5, 1000)], Config(fee_rate=0.0))

            def add(variant, wins, losses, ts=100.0):
                for won in [True] * wins + [False] * losses:
                    store.add_bet(ts, "btc", 15, 0, 100, fill, mode="lab", variant=variant)
                    bet_id = store.db.execute("SELECT MAX(id) FROM bets").fetchone()[0]
                    store.settle_bet(bet_id, "up" if won else "down", 10.0 if won else -10.0)

            add("twap|0.03|15-60", 8, 2)  # +60%: hot
            add("twap|0.05|15-60", 5, 5)  # 0%: swinging
            add("twap|0.08|15-60", 2, 8)  # -60%: losing
            add("twap|0.12|15-60", 3, 0)  # too few
            add("twap|0.03|15-180", 10, 0, ts=1.0)  # before the period
            db = connect(cfg.db_path)
            mix = strategy_mix(db, cfg, since=50.0)
            db.close()
            store.close()
        self.assertEqual(mix["counts"], {"hot": 1, "swinging": 1, "losing": 1, "few": 1})
        self.assertEqual(mix["top"][0]["variant"], "twap|0.03|15-60")
        self.assertIn("נקודות", mix["top"][0]["rule"])
        self.assertEqual(mix["active"], 4)


class StatsTest(unittest.TestCase):
    def test_money_weighted_return_matches_dollars(self):
        # A $2 winner and a $10 loser: per-bet average would say +40%, the dollars say -67%.
        mean, lower = lab.money_weighted([(2.0, 2.0), (-10.0, 10.0)], 1.65)
        self.assertAlmostEqual(mean, -8.0 / 12.0)
        self.assertIsNone(lower)  # stakes this uneven count as ~1.4 bets: too few to judge
        mean, lower = lab.money_weighted([(10.0, 10.0), (-10.0, 10.0), (10.0, 10.0)], 1.65)
        self.assertAlmostEqual(mean, 1 / 3)
        self.assertLess(lower, mean)
        self.assertEqual(lab.money_weighted([(5.0, 10.0)], 1.65), (0.5, None))

    def test_calibration_buckets(self):
        from polybot.stats import calibration
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            for fair, won in ((0.75, True), (0.72, False), (0.95, True), (0.3, False)):
                fill = find_bet(fair, [(0.1, 100)], [(0.99, 1)], Config(fee_rate=0.0))
                store.add_bet(1.0, "btc", 15, 0, 100, fill, mode="lab", variant="v")
                bet_id = store.db.execute("SELECT MAX(id) FROM bets").fetchone()[0]
                store.settle_bet(bet_id, "up", 1.0 if won else -1.0)
            db = connect(store.path)
            cal = calibration(db)
            db.close()
            store.close()
        by = {b["lo"]: b for b in cal["buckets"]}
        self.assertEqual(by[0.7]["bets"], 2)
        self.assertAlmostEqual(by[0.7]["happened"], 0.5)
        self.assertAlmostEqual(by[0.7]["said"], 0.735)
        self.assertEqual(cal["bets"], 4)


if __name__ == "__main__":
    unittest.main()


class RiskRewardTest(unittest.TestCase):
    def test_filter_stops_buying_above_the_ratio_price(self):
        import dataclasses
        cfg = dataclasses.replace(Config(), fee_rate=0.0, min_edge=0.01, stake_usd=100.0, min_ratio=3.0)
        # 1:3 allows at most 25c per share: the 20c level is bought, the 30c level is not.
        shares, cost, fee, top = simulate_buy([(0.20, 10), (0.30, 100)], 100.0, 0.9, cfg)
        self.assertAlmostEqual(shares, 10)
        self.assertAlmostEqual(top, 0.20)
        self.assertIsNone(find_bet(0.9, [(0.30, 100)], [], cfg))
        self.assertIsNotNone(find_bet(0.9, [(0.30, 1000)], [], dataclasses.replace(cfg, min_ratio=0.0)))

    def test_all_or_nothing_like_a_real_order(self):
        import dataclasses
        cfg = dataclasses.replace(Config(), fee_rate=0.0, min_edge=0.05, stake_usd=50.0)
        # $50 at 40c needs 125 shares; only 20 are on offer with enough edge (the 80c level has none).
        thin = [(0.40, 20), (0.80, 1000)]
        self.assertIsNone(find_bet(0.7, thin, [], cfg))
        partial = find_bet(0.7, thin, [], dataclasses.replace(cfg, full_fill_only=False))
        self.assertAlmostEqual(partial.cost, 8.0)  # what it used to do: buy what it could
        full = find_bet(0.7, [(0.40, 20), (0.42, 1000)], [], cfg)
        self.assertAlmostEqual(full.cost + full.fee, 50.0)  # exactly the stake set

class StockLabTest(unittest.TestCase):
    CLOSE = 1793044800  # 2026-10-26 16:00 New York

    def market(self, slug="aapl-close-above-on-october-26-2026", title="$340", threshold="0", closed=False):
        return {"slug": slug.replace("-on-", "-340-on-") if title else slug, "endDate": "2026-10-26T20:00:00Z",
                "clobTokenIds": '["111", "222"]', "groupItemTitle": title, "groupItemThreshold": threshold,
                "closed": closed, "feeSchedule": {"rate": 0, "exponent": 1}}

    def test_parse_and_discover(self):
        from polybot import stocks
        m = stocks.parse_market("aapl-close-above-on-october-26-2026", self.market())
        self.assertEqual((m["symbol"], m["kind"], m["strike"], m["close"]), ("aapl", "close-above", 340.0, self.CLOSE))
        self.assertEqual((m["yes_token"], m["no_token"]), ("111", "222"))
        updown = stocks.parse_market("spy-up-or-down-on-october-26-2026", self.market(title="", threshold=None))
        self.assertIsNone(updown["strike"])
        self.assertIsNone(stocks.parse_market("spy-opens-up-or-down-on-october-26-2026", self.market()))
        self.assertIsNone(stocks.parse_market("aapl-close-above-on-october-26-2026", self.market(closed=True)))
        # The real Gamma fields: groupItemThreshold is the item's index, not its price.
        real = dict(self.market(), slug="aapl-close-above-345-on-october-26-2026", groupItemTitle="$345", groupItemThreshold="1")
        self.assertEqual(stocks.parse_market("aapl-close-above-on-october-26-2026", real)["strike"], 345.0)
        clash = dict(real, groupItemTitle="$350")
        self.assertIsNone(stocks.parse_market("aapl-close-above-on-october-26-2026", clash))
        calls = []

        def get(url, timeout=3.0):
            calls.append(url)
            return [{"slug": "aapl-close-above-on-october-26-2026", "markets": [self.market()]},
                    {"slug": "will-x-happen", "markets": [self.market()]}]
        found = stocks.discover(self.CLOSE - 3600, get)
        self.assertEqual(list(found), ["aapl-close-above-340-on-october-26-2026"])
        self.assertIn("tag_slug=stocks", calls[0])

    def lab(self, tmp):
        from polybot import stocks
        cfg = Config(db_path=os.path.join(tmp, "t.sqlite"), vol_min_seconds=60)
        store = Store(cfg.db_path)
        replies = {}
        lab_ = stocks.StockLab(cfg, store, get=lambda url, timeout=3.0: replies.get(url.split("slug=")[1].split("&")[0]))
        m = stocks.parse_market("aapl-close-above-on-october-26-2026", self.market())
        lab_.markets = {m["slug"]: m}
        return lab_, store, replies, m

    def test_model_bets_when_the_book_lags_and_settles(self):
        import random
        random.seed(3)
        with tempfile.TemporaryDirectory() as tmp:
            lab_, store, replies, m = self.lab(tmp)
            now = self.CLOSE - 240
            price = 345.0
            for t in range(int(now) - 600, int(now) + 1):
                price *= 1 + random.gauss(0, 0.0002)
                lab_.on_price("aapl", t, 345.0 + (price - 345.0) * 0.01)
            lab_.last["aapl"] = (now, 345.0, now)  # 4 minutes left, $5 above the line
            self.assertIn("111", lab_.tokens(now))
            books = {"111": ([(0.80, 100)], [(0.85, 100)], now), "222": ([(0.14, 100)], [(0.17, 100)], now)}
            lab_.evaluate(now, books.get)
            bets = store._x("SELECT variant, side, market FROM bets WHERE mode = 'stocks'")
            kinds = {v.split("|")[0] for v, _, _ in bets}
            self.assertEqual(kinds, {"model", "fav"})
            self.assertTrue(all(side == "up" and market == m["slug"] for _, side, market in bets))
            n = len(bets)
            lab_.evaluate(now + 1, books.get)  # each variant bets once per market
            self.assertEqual(len(store._x("SELECT id FROM bets WHERE mode = 'stocks'")), n)
            replies[m["slug"]] = [{"closed": True, "outcomePrices": '["1", "0"]'}]
            lab_.settle(self.CLOSE + 120)
            pnl = store._x("SELECT SUM(pnl), COUNT(*) FROM bets WHERE mode = 'stocks' AND winner = 'up'")[0]
            self.assertEqual(pnl[1], n)
            self.assertGreater(pnl[0], 0)
            db = connect(store.path)
            state = lab_.state(db, self.CLOSE + 120)
            db.close()
            self.assertEqual(state["settled"], n)
            self.assertEqual(len(state["board"]), len(__import__("polybot.stocks", fromlist=["x"]).VARIANTS))
            store.close()

    def test_no_bet_without_fresh_data_and_close_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab_, store, replies, m = self.lab(tmp)
            now = self.CLOSE - 240
            books = {"111": ([], [(0.50, 100)], now), "222": ([], [(0.50, 100)], now)}
            lab_.evaluate(now, books.get)  # no price at all
            self.assertEqual(store._x("SELECT COUNT(*) FROM bets")[0][0], 0)
            # Crossing 4pm records the last price before it as that day's close.
            lab_.on_price("spy", self.CLOSE - 1, 700.0)
            lab_.on_price("spy", self.CLOSE + 1, 701.0)
            self.assertEqual(lab_.prior_close("spy", self.CLOSE + 86400), 700.0)
            self.assertIsNone(lab_.prior_close("spy", self.CLOSE))
            from polybot import stocks
            again = stocks.StockLab(lab_.cfg, store)
            self.assertEqual(again.prior_close("spy", self.CLOSE + 86400), 700.0)  # survives a restart
            store.close()

    def test_equity_stream_parses_relay_messages(self):
        from polybot import stocks
        got = []
        stream = stocks.EquityStream(lambda *a: got.append(a))
        stream.handle(json.dumps({"topic": "equity_prices", "payload": {"symbol": "TSLA", "timestamp": 1792794858000,
                                                                        "value": 371.845}}))
        stream.handle(json.dumps({"topic": "crypto_prices", "payload": {"symbol": "btc", "timestamp": 1, "value": 1}}))
        stream.handle("PONG")
        self.assertEqual(got, [("tsla", 1792794858.0, 371.845)])

    def test_missing_previous_close_comes_from_pyth(self):
        from polybot import stocks
        with tempfile.TemporaryDirectory() as tmp:
            lab_, store, replies, m = self.lab(tmp)
            updown = stocks.parse_market("spy-up-or-down-on-october-26-2026", self.market(title="", threshold=None))
            updown["slug"] = "spy-up-or-down-on-october-26-2026"
            lab_.markets = {updown["slug"]: updown}
            friday = self.CLOSE - 3 * 86400  # Monday's previous session
            asked = []

            def get(url, timeout=3.0):
                asked.append(url)
                return {"s": "ok", "t": [friday - 120, friday - 60], "c": [699.0, 700.5]}
            lab_.get = get
            self.assertIsNone(lab_.line(updown))
            lab_.fill_missing_closes(self.CLOSE - 3600)
            self.assertEqual(lab_.line(updown), 700.5)
            self.assertIn("Equity.US.SPY/USD", asked[0])
            store.close()


class VersionTest(unittest.TestCase):
    def test_version_is_a_simple_number(self):
        import re
        from polybot import engine
        self.assertRegex(engine.VERSION, r"^1\.\d+$")


class StockCloseFallbackTest(unittest.TestCase):
    def test_404_falls_back_to_benchmarks_and_429_stops_quietly(self):
        import urllib.error
        from unittest import mock
        from polybot import stocks
        close = StockLabTest.CLOSE
        friday = close - 3 * 86400
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)

            def get(url, timeout=3.0):
                if "tradingview" in url:
                    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
                if "price_feeds" in url:
                    return [{"id": "abc", "attributes": {"base": "SPY", "quote_currency": "USD", "country": "US"}}]
                self.assertIn(f"/{friday - 1}?ids=abc", url)
                return {"parsed": [{"price": {"price": "70050000", "expo": -5}}]}
            lab_ = stocks.StockLab(cfg, store, get=get)
            m = stocks.parse_market("spy-up-or-down-on-october-26-2026",
                                    StockLabTest().market(title="", threshold=None))
            m["slug"] = "spy-up-or-down-on-october-26-2026"
            lab_.markets = {m["slug"]: m}
            with mock.patch("time.sleep"):
                lab_.fill_missing_closes(close - 3600)
            self.assertAlmostEqual(lab_.line(m), 700.5)
            self.assertEqual(lab_.missing_lines, [])

            busy = stocks.StockLab(cfg, store, get=lambda url, timeout=3.0: (_ for _ in ()).throw(
                urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)))
            busy.closes = {}
            busy.markets = lab_.markets
            with mock.patch("time.sleep"), self.assertNoLogs("polybot", level="WARNING"):
                busy.fill_missing_closes(close - 3600)
            self.assertEqual(busy.missing_lines, ["spy"])
            store.close()


class ModeAndVersionsTest(unittest.TestCase):
    def engine(self, tmp, env=None):
        cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc", "xrp"))
        engine = Engine(cfg, api=FakeApi(), env=env or {}, streaming=False)
        engine.refresh_lab = lambda: None
        return engine

    def test_switch_to_live_needs_a_wallet_and_a_confirmation_but_not_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.engine(tmp)
            with self.assertRaises(ValueError):
                engine.set_mode(True, confirmed=True)  # no wallet
            engine.store.close()
            engine = self.engine(tmp, env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64})
            with self.assertRaises(ValueError):
                engine.set_mode(True, confirmed=False)
            engine.set_mode(True, confirmed=True)  # readiness not passed: the owner's call
            self.assertEqual(engine.mode(), "live")
            engine.live.set_readiness(False)  # an override is not undone by the checks
            self.assertEqual(engine.mode(), "live")
            self.assertTrue(engine.live.trades("crypto"))
            engine.set_mode(False, confirmed=True)
            self.assertEqual(engine.mode(), "paper")
            engine.live.set_readiness(True)
            engine.set_mode(True, confirmed=True)
            engine.live.set_readiness(False)  # live because the checks passed: stops when they fail
            self.assertFalse(engine.live.trades("crypto"))
            engine.store.close()

    def test_push_saves_versions_and_rollback_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.engine(tmp)
            original = engine.main_strategies()
            self.assertEqual(engine.recommendation()["verdict"], "wait")
            row = {"asset": "xrp", "minutes": 5, "variant": "twap|0.05|10-120", "status": "proven", "rule": "r",
                   "bets": 80, "roi": 0.1, "roi_pessimistic": 0.02, "roi_recent": 0.05, "oos_bets": 35,
                   "oos_roi": 0.08, "oos_pessimistic": 0.01}
            engine.lab = {"board": [row]}
            rec = engine.recommendation()
            self.assertEqual(rec["verdict"], "push")
            self.assertEqual([(c["asset"], c["minutes"]) for c in rec["changes"]], [("xrp", 5)])
            engine.push([(c["asset"], c["minutes"], c["variant"]) for c in rec["changes"]])
            versions = engine.strategy_versions()
            self.assertEqual([v["id"] for v in versions], [1, 2])
            self.assertEqual(versions[0]["main"], original)
            self.assertEqual(engine.current_version(), 2)
            self.assertIn(("xrp", 5), {(t.asset, t.minutes) for t in engine.main_trackers})
            self.assertEqual(engine.recommendation()["verdict"], "wait")  # already the main strategy
            engine.rollback(1)
            self.assertEqual([(t.asset, t.minutes) for t in engine.main_trackers], [("btc", 15)])
            self.assertEqual(engine.current_version(), 1)
            self.assertEqual(len(engine.strategy_versions()), 2)  # rolling back keeps every version
            state = engine.state()
            self.assertEqual(state["versions"]["current"], 1)
            self.assertEqual(state["mode"], "paper")
            engine.store.close()


class MailBatchTest(unittest.TestCase):
    def test_each_new_trade_restarts_the_wait_up_to_a_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            n = engine.notifier
            add = lambda: engine.store._x(
                "INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                " VALUES ('paper', 'filled', 0, 'btc', 15, 0, 'up', 1, 1, 0)")
            cond = "notified & 1 = 0"
            self.assertFalse(n._quiet("open", cond, "paper", 0))  # nothing waiting
            add()
            self.assertFalse(n._quiet("open", cond, "paper", 100))
            self.assertFalse(n._quiet("open", cond, "paper", 125))
            add()  # a new trade at 125: the wait starts again
            self.assertFalse(n._quiet("open", cond, "paper", 126))
            self.assertFalse(n._quiet("open", cond, "paper", 150))
            self.assertTrue(n._quiet("open", cond, "paper", 157))
            for t in range(160, 400, 20):  # a steady stream still gets mailed after 5 minutes
                add()
                if n._quiet("open", cond, "paper", t):
                    break
            self.assertGreaterEqual(t, 400 - 20 - 20)
            engine.store.close()


class StockStrategyTest(unittest.TestCase):
    def test_pushed_stock_strategy_bets_in_the_simulation_and_settles(self):
        import random
        random.seed(5)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",),
                         vol_min_seconds=60)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.refresh_lab = lambda: None
            lab_ = engine.stocks
            m = __import__("polybot.stocks", fromlist=["x"]).parse_market(
                "aapl-close-above-on-october-26-2026", StockLabTest().market())
            lab_.markets = {m["slug"]: m}
            engine.push_stocks("fav|0.70|300")
            self.assertEqual(engine.strategy_versions()[-1]["stocks"], "fav|0.70|300")
            now = StockLabTest.CLOSE - 240
            # (pushed before the market's last minutes, whatever today's date: its close is a fixed day)
            engine.store.set_setting("stocks_main", json.dumps({"variant": "fav|0.70|300", "since": now - 3600}))
            for t in range(int(now) - 600, int(now) + 1):
                lab_.on_price("aapl", t, 345.0 + random.gauss(0, 0.01))
            lab_.last["aapl"] = (now, 345.0, now)
            books = {"111": ([(0.80, 100)], [(0.85, 100)], now), "222": ([(0.14, 100)], [(0.17, 100)], now)}
            lab_.evaluate(now, books.get)
            main = engine.store._x("SELECT variant, market, side FROM bets WHERE mode = 'paper' AND minutes = 0")
            self.assertEqual(main, [("stocks:fav|0.70|300", m["slug"], "up")])
            lab_.evaluate(now + 5, books.get)  # once per market
            self.assertEqual(len(engine.store._x("SELECT id FROM bets WHERE mode = 'paper' AND minutes = 0")), 1)
            db = connect(engine.store.path)
            cards = engine._positions(db, now)
            db.close()
            card = cards["paper"][0]
            self.assertEqual((card["asset"], card["kind"], card["open_price"], card["labels"]["up"]),
                             ("aapl", "close-above", 340.0, "כן"))
            self.assertEqual(len(cards["stocks"]), 1)  # the lab's own bets have their card too
            lab_.get = lambda url, timeout=3.0: [{"closed": True, "outcomePrices": '["1", "0"]'}]
            lab_.settle(StockLabTest.CLOSE + 120)
            pnl = engine.store._x("SELECT pnl FROM bets WHERE mode = 'paper' AND minutes = 0")[0][0]
            self.assertGreater(pnl, 0)
            # The crypto readiness never counts the stock strategy's bets.
            db = connect(engine.store.path)
            from polybot import stats
            self.assertEqual(stats.bet_summary(db, "paper", strategy="crypto")["settled"], 0)
            self.assertEqual(stats.bet_summary(db, "paper", strategy="stocks")["settled"], 1)
            state = engine.state()
            self.assertEqual(state["stocks_main"]["main"]["variant"], "fav|0.70|300")
            self.assertEqual(state["stocks_main"]["summary"]["settled"], 1)
            db.close()
            engine.rollback(1)  # the original had no stock strategy
            self.assertIsNone(lab_.main())
            engine.store.close()

    def test_live_strategy_switches(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            t = LiveTrader(cfg, store, env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64})
            self.assertEqual(t.strategies(), {"crypto": True, "stocks": False})
            self.assertFalse(t.trades("crypto"))  # Live is off
            t.arm(True, override=True)
            self.assertTrue(t.trades("crypto"))
            self.assertFalse(t.trades("stocks"))
            with self.assertRaises(ValueError):
                t.set_strategy("stocks", True)  # its checks haven't passed: needs the owner's confirmation
            t.set_strategy("stocks", True, confirmed=True)
            t.set_strategy("crypto", False)
            self.assertEqual((t.trades("crypto"), t.trades("stocks")), (False, True))
            store.close()


class LiveSafetyTest(unittest.TestCase):
    def test_no_real_order_before_this_run_has_judged_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            key = {"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64}
            t = LiveTrader(cfg, store, env=key)
            t.set_readiness(True)
            t.arm(True)  # armed because the checks passed
            again = LiveTrader(cfg, store, env=key)  # a restart: readiness not judged yet
            self.assertTrue(again.armed())
            self.assertFalse(again.trades("crypto"))
            self.assertIsNone(again.maybe_trade(0, "btc", 15, 0, 300, None, {}))
            again.set_readiness(True)
            self.assertTrue(again.trades("crypto"))
            store.close()

    def test_stock_strategy_needs_its_own_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            t = LiveTrader(cfg, store, env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64})
            t.set_readiness(True)
            t.set_strategy_readiness("stocks", True)
            t.set_strategy("stocks", True)
            t.arm(True)
            self.assertTrue(t.trades("stocks"))
            t.set_strategy_readiness("stocks", False)
            self.assertFalse(t.trades("stocks"))
            self.assertTrue(t.trades("crypto"))  # one strategy failing doesn't stop the other
            store.close()


class WalletConnectTest(unittest.TestCase):
    def test_check_connection_reads_the_balance_and_the_account_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            t = LiveTrader(cfg, store, env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64, "POLYMARKET_FUNDER": "0x" + "a" * 40})
            client = types.SimpleNamespace(
                wallet_type="POLY_PROXY", wallet="0x" + "a" * 40,
                get_balance_allowance=lambda asset_type: types.SimpleNamespace(balance=61_680_000, allowances={"x": 5}))
            t._make_client = lambda: client
            result = t.check_connection()
            self.assertEqual((result["balance"], result["wallet_type"], result["approved"]), (61.68, "POLY_PROXY", True))
            self.assertEqual(t.wallet_type, "POLY_PROXY")
            self.assertEqual(t.status()["balance"], 61.68)

            def wrong_account():
                raise RuntimeError("Wallet 0xaa does not match the expected Deposit Wallet 0xbb for this signer")
            t._make_client = wrong_account
            with self.assertRaises(ValueError) as caught:
                t.check_connection()
            self.assertIn("לא שייכים לאותו חשבון", str(caught.exception))
            store.close()


class SimStrategiesTest(unittest.TestCase):
    def test_switches_gate_the_bets_and_the_summaries_follow(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.refresh_lab = lambda: None
            self.assertEqual(engine.sim_strategies(), {"crypto": True, "stocks": False})
            with self.assertRaises(ValueError):
                engine.set_sim_strategies(True, True)  # stocks need a strategy
            engine.set_sim_strategies(True, True, "fav|0.70|300")
            self.assertEqual(engine.sim_strategies(), {"crypto": True, "stocks": True})
            self.assertTrue(engine.stocks.enabled and engine.main_trackers[0].sim_on)
            add = lambda minutes, pnl: engine.store._x(
                "INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl)"
                " VALUES ('paper', 'filled', 1, 'x', ?, 0, 'up', 20, 10, 0, ?)", (minutes, pnl))
            for _ in range(3):
                add(0, 5.0)   # three stock trades
            for _ in range(2):
                add(15, -2.0)  # two crypto trades
            self.assertEqual(engine.state()["paper"]["summary"]["settled"], 5)
            self.assertAlmostEqual(engine.state()["paper"]["summary"]["pnl"], 11.0)
            engine.set_sim_strategies(False, True)
            self.assertFalse(engine.main_trackers[0].sim_on)
            self.assertEqual(engine.state()["paper"]["summary"]["settled"], 3)  # only what runs is counted
            engine.set_sim_strategies(True, False)
            self.assertFalse(engine.stocks.enabled)
            self.assertEqual(engine.state()["paper"]["summary"]["settled"], 2)
            engine.store.close()


class MailRulesTest(unittest.TestCase):
    setUp, tearDown, open_a_trade, html = NotifyTest.setUp, NotifyTest.tearDown, NotifyTest.open_a_trade, NotifyTest.html

    def local(self, ts, minutes=0):
        import datetime as dt
        from zoneinfo import ZoneInfo
        return (dt.datetime.fromtimestamp(ts, ZoneInfo(self.engine.cfg.timezone)) + dt.timedelta(minutes=minutes)).strftime("%H:%M")

    def test_subject_shows_chance_and_what_a_win_pays_against_a_loss(self):
        self.open_a_trade()
        self.engine.notifier.tick(self.start + 601)
        self.engine.notifier.tick(self.start + 632)
        subject = self.sent[0]["Subject"]
        self.assertIn("%", subject)
        self.assertIn(" || ", subject)
        self.assertRegex(subject, r"\+\$[\d.]+⁩ / ⁦-\$[\d.]+")

    def test_quiet_hours_hold_mails_and_send_one_summary_after(self):
        n = self.engine.notifier
        n.update(quiet=True, quiet_from=self.local(self.start, -60), quiet_to=self.local(self.start, 60))
        self.open_a_trade()
        n.tick(self.start + 601)
        n.tick(self.start + 700)
        self.assertEqual(self.sent, [])  # quiet
        n.tick(self.start + 3 * 3600)  # the quiet hours are over
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.sent[0]["Subject"].startswith("🌙"))
        n.tick(self.start + 3 * 3600 + 60)
        self.assertEqual(len(self.sent), 1)  # nothing sent twice

    def test_the_nights_summary_is_one_mail_for_real_money_and_the_simulation(self):
        # (the owner, 30.10.2026: three mails of the night at once)
        n = self.engine.notifier
        n.update(quiet=True, quiet_from=self.local(self.start, -60), quiet_to=self.local(self.start, 60))
        self.open_a_trade()
        rows = self.engine.store._x("SELECT ts, asset, minutes, window_start, side, shares, cost, fee FROM bets WHERE mode = 'paper'")
        if not rows:
            self.skipTest("no simulated trade")
        ts, asset, minutes, ws, side, shares, cost, fee = rows[0]
        self.engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee, avg_price,"
                             " fair_prob, variant) VALUES ('live', 'filled', ?, ?, ?, ?, ?, ?, ?, ?, 0.5, 0.6, 'main')",
                             (ts, asset, minutes, ws, side, shares, cost, fee))
        n.tick(self.start + 601)
        n.tick(self.start + 3 * 3600)  # the quiet hours are over
        night = [m for m in self.sent if m["Subject"].startswith("🌙")]
        self.assertEqual(len(night), 1)
        self.assertIn("כסף אמיתי", night[0]["Subject"])
        self.assertIn("סימולציה", night[0]["Subject"])

    def test_after_the_quiet_hours_every_trade_is_mailed_again(self):
        # The quiet hours' end once left the text "None" behind: from then on every tick took the
        # waiting trades, failed on it, and no trade mail went out again (simulation and real money).
        n = self.engine.notifier
        n.update(quiet=True, quiet_from=self.local(self.start, -60), quiet_to=self.local(self.start, 60))
        n.tick(self.start + 30)
        n.tick(self.start + 3 * 3600)  # over
        self.assertIsNone(self.engine.store.get_setting("notify_quiet_since"))
        n.update(quiet=False)
        self.sent.clear()
        self.open_a_trade()
        n.tick(self.start + 601)
        n.tick(self.start + 632)
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.sent[0]["Subject"].startswith(("📈", "📉")))

    def test_a_database_with_the_stuck_value_heals(self):
        self.engine.store._x("INSERT OR REPLACE INTO settings VALUES ('notify_quiet_since', 'None')")
        self.open_a_trade()
        n = self.engine.notifier
        n.tick(self.start + 601)  # the one summary of what waited
        n.tick(self.start + 632)
        n.tick(self.start + 700)
        self.assertEqual(len(self.sent), 1)
        self.assertIsNone(self.engine.store.get_setting("notify_quiet_since"))
        self.sent.clear()
        self.t.step(self.start + 901)
        self.api.winner = "up"
        settle(self.engine.store, self.engine.cfg, self.start + 2000, api=self.api)
        n.tick(self.start + 2001)
        n.tick(self.start + 2032)
        self.assertEqual(len(self.sent), 1)  # and the result mail goes out as usual
        self.assertTrue(self.sent[0]["Subject"].startswith(("✅", "❌")))

    def test_every_n_trades_waits_for_n(self):
        self.engine.notifier.update(open_every=2)
        self.open_a_trade()
        self.engine.notifier.tick(self.start + 601)
        self.engine.notifier.tick(self.start + 900)
        self.assertEqual(self.sent, [])  # one trade, two needed

    def test_a_win_comes_with_confetti(self):
        self.open_a_trade()
        self.engine.notifier.tick(self.start + 601)
        self.engine.notifier.tick(self.start + 632)
        self.t.step(self.start + 901)
        self.api.winner = "up"
        settle(self.engine.store, self.engine.cfg, self.start + 2000, api=self.api)
        self.engine.notifier.tick(self.start + 2001)
        self.engine.notifier.tick(self.start + 2032)
        closed = self.sent[-1]
        gifs = [p for p in closed.walk() if p.get_content_type() == "image/gif"]
        self.assertEqual(len(gifs), 1 if closed["Subject"].startswith("✅") else 0)


class SimLimitsTest(unittest.TestCase):
    def test_open_trade_cap_daily_loss_and_stake(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.set_lab_exact("sim", False)  # these check the simulation's own filters
            self.assertTrue(engine.can_open("crypto", 10))
            engine.set_sim_limits(stake=50, daily_loss=60, max_open={"crypto": 1, "stocks": 0})
            self.assertEqual(engine.main_trackers[0].main_cfg.stake_usd, 50)
            self.assertEqual(engine.stocks.sim_stake, 50)
            import time as _t
            engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                            " VALUES ('paper', 'filled', ?, 'btc', 15, 0, 'up', 80, 50, 0)", (_t.time(),))
            self.assertFalse(engine.can_open("crypto", 50))  # one open, cap 1
            self.assertTrue(engine.can_open("stocks", 50))  # past the $60 a day: red on the page, the simulation goes on
            self.assertTrue(engine.money()["sim"]["today"] >= 50)
            engine.set_sim_limits(daily_loss=0, max_open={"crypto": 0, "stocks": 0})
            self.assertTrue(engine.can_open("crypto", 50))
            engine.store.close()


class MarketBlendTest(unittest.TestCase):
    def test_fit_finds_the_weights_that_generated_the_data(self):
        import math, random
        from polybot import calibration
        random.seed(11)
        samples = []
        for _ in range(6000):
            x, y = random.gauss(0, 1.5), random.gauss(0, 1.5)
            p = 1 / (1 + math.exp(-(0.4 * x + 0.7 * y)))
            samples.append((x, y, random.random() < p))
        a, b = calibration.fit_blend(samples)
        self.assertAlmostEqual(a, 0.4, delta=0.08)
        self.assertAlmostEqual(b, 0.7, delta=0.08)
        self.assertAlmostEqual(calibration.blend(0.5, 0.5, a, b), 0.5)

    def test_blend_applies_only_when_it_wins_on_unseen_windows(self):
        import random
        from polybot import calibration
        random.seed(3)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=os.path.join(tmp, "t.sqlite"), blend_min_windows=100)
            store = Store(cfg.db_path)
            # The market knows; the model is noise: the mix should lean on the market, and apply.
            for w in range(400):
                truth = random.random()
                market = min(0.97, max(0.03, truth + random.gauss(0, 0.05)))
                model = random.random()
                won = random.random() < truth
                store.upsert_window("btc", 15, w * 900, winner="up" if won else "down")
                store.add_tick(ts=w * 900 + 600, asset="btc", minutes=15, window_start=w * 900, price=1, open_price=1,
                               seconds_left=300, vol=0.001, fair_up=model, up_bid=market - 0.01, up_ask=market + 0.01,
                               down_bid=None, down_ask=None, price_age=0, book_age=0, fair_raw=model)
            db = connect(cfg.db_path)
            out = calibration.market_blend(db, cfg)
            db.close()
            store.close()
        self.assertTrue(out["applied"])
        self.assertLess(out["a"], 0.3)
        self.assertGreater(out["b"], 0.6)
        self.assertLess(out["loss_blend"], out["loss_model"])


class TradingSettingsTest(unittest.TestCase):
    def test_live_tab_saves_what_runs_the_stake_and_its_protections(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64}, streaming=False)
            engine.refresh_lab = lambda: None
            engine.set_live_mirror(False)  # (real money's own rules: its protections apply)
            engine.save_trading("live", {"crypto": True, "stocks": True, "stocks_variant": "fav|0.70|300", "stake": 7,
                                         "protections": {"daily_loss": 30, "max_open": {"crypto": 2, "stocks": 1},
                                                         "same_direction": 2, "blend": False}})
            L = engine.live
            self.assertEqual((L.stake(), L.daily_loss_limit(), L.max_open(), L.same_direction()),
                             (7, 30, {"crypto": 2, "stocks": 1}, 2))
            self.assertEqual(engine.blend_use()["live"], False)
            self.assertEqual(L.strategies(), {"crypto": True, "stocks": True})
            self.assertEqual(engine.sim_strategies(), {"crypto": True, "stocks": True})  # live implies simulation
            self.assertEqual(engine.protections()["live"]["daily_loss"], 30)
            engine.store.close()

class SameDirectionTest(unittest.TestCase):
    def test_one_same_direction_bet_per_window_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.set_lab_exact("sim", False)  # these check the simulation's own filters
            self.assertTrue(engine.can_open("crypto", 10, "up", 900))
            engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                            " VALUES ('paper', 'filled', 1, 'eth', 15, 900, 'up', 20, 10, 0)")
            self.assertFalse(engine.can_open("crypto", 10, "up", 900))  # ETH already up in this window
            self.assertTrue(engine.can_open("crypto", 10, "down", 900))
            self.assertTrue(engine.can_open("crypto", 10, "up", 1800))
            engine.set_sim_limits(same_direction=0)
            self.assertTrue(engine.can_open("crypto", 10, "up", 900))
            engine.store.close()


class MakerTest(unittest.TestCase):
    def test_resting_order_fills_only_when_a_seller_comes_through_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            t = next(t for t in engine.trackers if t.key == "btc-15m")
            t.window, t.market = 900, {"tick_size": 0.01}
            v = lab.Variant(0.05, 15, 600, "twapm")
            # Fair 70% for Up, best offer 60c: rests one tick under the offer, at 59c.
            t._maker(v, 1000, 300, 0.70, 0.70, [(0.60, 100)], [(0.42, 100)], 0.1, 0.1)
            self.assertEqual((t.maker_orders[v.key]["side"], t.maker_orders[v.key]["price"]), ("up", 0.59))
            t._maker(v, 1001, 299, 0.70, 0.70, [(0.59, 100)], [(0.42, 100)], 0.1, 0.1)  # touched, not through
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets")[0][0], 0)
            t._maker(v, 1002, 298, 0.70, 0.70, [(0.58, 100)], [(0.43, 100)], 0.1, 0.1)  # a seller came through
            row = engine.store._x("SELECT side, avg_price, fee, variant FROM bets")[0]
            self.assertEqual(tuple(row), ("up", 0.59, 0.0, v.key))
            self.assertIn(v.key, t.lab_done)
            engine.store.close()

    def test_a_seller_before_the_order_rests_doesnt_fill_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",), lab_fill_delay=0.5)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            t = next(t for t in engine.trackers if t.key == "btc-15m")
            t.window, t.market = 900, {"tick_size": 0.01}
            v = lab.Variant(0.05, 15, 600, "twapm")
            t._maker(v, 1000.0, 300, 0.70, 0.70, [(0.60, 100)], [(0.42, 100)], 0.1, 0.1)  # sent at 59c
            t._maker(v, 1000.3, 300, 0.70, 0.70, [(0.58, 100)], [(0.43, 100)], 0.1, 0.1)  # still on its way
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets")[0][0], 0)
            self.assertNotIn(v.key, t.lab_done)
            engine.store.close()


class SettingsTakeEffectTest(unittest.TestCase):
    """Every settings tab, saved through the web server with the body the page sends, changes
    what the system actually does, and reads back the same after the page reloads."""

    def setUp(self):
        import threading
        from http.server import ThreadingHTTPServer
        from polybot.web import make_handler

        self.tmp = tempfile.TemporaryDirectory()
        cfg = Config(assets=("btc",), db_path=os.path.join(self.tmp.name, "t.sqlite"), lab_assets=("btc",))
        self.engine = Engine(cfg, api=FakeApi(), env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64}, streaming=False)
        self.engine.live.client = FakeClob()
        self.t = next(t for t in self.engine.trackers if t.key == "btc-15m")
        self.t.window, self.t.market = 900, {"up_token": "U", "down_token": "D", "tick_size": 0.01, "min_order_size": 1}
        self.t._set_market_cfg(self.t.market)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.engine))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.engine.store.close()
        self.tmp.cleanup()

    def post(self, path, body):
        import urllib.request
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            return json.loads(r.read())

    def state(self):
        import urllib.request
        with urllib.request.urlopen(self.base + "/api/state", timeout=5) as r:
            return json.loads(r.read())

    def signal(self, price, window=900):
        """A strong signal for Up at this price; returns (paper bets, real orders) placed."""
        t, rule = self.t, self.t.main_rule
        if t.window != window:
            t.window, t.main_done, t.live_done, t.main_entries, t.live_entries = window, set(), set(), {}, {}
        before = len(self.engine.live.client.orders)
        papers = self.engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'paper'")[0][0]
        now, left, up, down = time.time(), rule.min_seconds_left + 1, [(price, 10_000)], [(0.99, 10)]
        t._sim_bet(now, left, rule, 0.97, 0.97, up, down, 0.1, 0.1)
        t._live_bet(now, left, t.live_rule, 0.97, up, down)
        return (self.engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'paper'")[0][0] - papers,
                len(self.engine.live.client.orders) - before)

    def sim_body(self, **over):
        P = {"daily_loss": over.pop("daily_loss", 0), "max_open": over.pop("max_open", {"crypto": 0, "stocks": 0}),
             "same_direction": over.pop("same_direction", 0), "blend": False}
        body = {"confirmed": True, "crypto": True, "stocks": False, "stocks_variant": None, "stake": 25, "protections": P}
        body.update(over)
        return body

    def live_body(self, **over):
        P = {"daily_loss": over.pop("daily_loss", 30), "max_open": over.pop("max_open", {"crypto": 0, "stocks": 0}),
             "same_direction": over.pop("same_direction", 0), "blend": False}
        body = {"confirmed": True, "crypto": True, "stocks": False, "stocks_variant": None, "stake": 7, "protections": P}
        body.update(over)
        return body

    def test_simulation_tab(self):
        self.engine.set_lab_exact("sim", False)  # these check the simulation's own filters
        self.post("/api/trading/sim", self.sim_body(crypto=False))
        self.assertEqual(self.signal(0.55), (0, 0))  # crypto switched off: no simulated bet
        self.post("/api/trading/sim", self.sim_body())
        self.assertEqual(self.signal(0.55), (1, 0))
        stake = self.engine.store._x("SELECT cost + fee FROM bets WHERE mode = 'paper'")[0][0]
        self.assertAlmostEqual(stake, 25, delta=0.6)  # the stake set on the page

        # Open-trade cap, same direction, daily loss.
        self.post("/api/trading/sim", self.sim_body(max_open={"crypto": 1, "stocks": 0}))
        self.assertEqual(self.signal(0.55, window=3600), (0, 0))  # 1 already open
        self.post("/api/trading/sim", self.sim_body(same_direction=1))
        self.engine.store._x("UPDATE bets SET pnl = 0")  # nothing open any more
        self.engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl)"
                             " VALUES ('paper', 'filled', ?, 'eth', 15, 4500, 'up', 20, 10, 0, 0)", (time.time(),))
        self.assertEqual(self.signal(0.55, window=4500), (0, 0))  # ETH already bet Up in this window
        self.post("/api/trading/sim", self.sim_body(daily_loss=20))
        self.assertEqual(self.signal(0.55, window=5400), (1, 0))  # past the $20 a day: shown in red, the simulation goes on

        # What the page shows after a reload is what was saved.
        self.post("/api/trading/sim", self.sim_body(stake=40, daily_loss=90, max_open={"crypto": 3, "stocks": 2},
                                                    same_direction=2))
        s = self.state()
        self.assertEqual((s["sim_limits"]["stake"], s["sim_limits"]["daily_loss"], s["sim_limits"]["same_direction"]),
                         (40, 90, 2))
        self.assertEqual(s["sim_limits"]["max_open"], {"crypto": 3, "stocks": 2})
        self.assertEqual(s["protections"]["sim"]["daily_loss"], 90)

    def test_real_money_enters_each_trade_as_many_times_as_set(self):
        live = self.engine.live
        self.post("/api/trading/live", self.live_body(daily_loss=1000))
        self.post("/api/money", {"mode": "live", "values": {"copies": 3, "round_cap": 0}})  # (no cap a round: 3 x $7)
        live.arm(True, override=True)
        self.assertEqual(self.signal(0.55)[1], 3)  # three orders of the same stake
        prices = [o["max_price"] for o in live.client.orders[-3:]]
        self.assertTrue(all(p <= prices[0] + 0.01 + 1e-9 for p in prices))  # at most a cent above the first
        self.assertEqual(self.engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live'")[0][0], 3)

    def test_real_money_tab_follows_its_own_settings_only(self):
        live = self.engine.live
        self.engine.set_live_mirror(False)  # (real money's own rules, not the simulation's)
        self.post("/api/trading/live", self.live_body())
        self.assertEqual((live.stake(), live.daily_loss_limit()), (7, 30))
        live.arm(True, override=True)  # the owner confirmed (readiness is re-judged on every save)

        # The simulation's own limits don't hold real money back: here the simulation is off.
        self.post("/api/trading/sim", self.sim_body(crypto=False))
        self.assertEqual(self.signal(0.55), (0, 1))
        self.assertLessEqual(live.client.orders[-1]["amount"], 7)  # real money's own stake
        self.assertEqual(self.signal(0.55), (0, 0))  # one per window

        self.post("/api/trading/sim", self.sim_body())
        # Its open-trade cap and daily limit.
        self.post("/api/trading/live", self.live_body(max_open={"crypto": 1, "stocks": 0}))
        self.assertEqual(self.signal(0.55, window=2700), (1, 0))  # one real trade already open
        self.post("/api/trading/live", self.live_body(daily_loss=10))
        self.assertEqual(self.signal(0.55, window=3600), (1, 0))  # $7 open + $7 more > $10
        self.assertFalse(live.armed())  # reaching it switches real money off...
        self.assertEqual(live.stopped()["reason"], "daily_loss")
        live.arm(True, override=True)  # ...until the owner switches it back on
        # Strategy unticked for real money: no orders at all.
        self.post("/api/trading/live", self.live_body(crypto=False, daily_loss=100))
        self.assertEqual(self.signal(0.55, window=4500), (1, 0))
        self.post("/api/trading/live", self.live_body(daily_loss=100))
        self.assertEqual(self.signal(0.55, window=5400), (1, 1))

        s = self.state()["live"]["status"]
        self.assertEqual((s["stake"], s["daily_loss_limit"], s["strategies"]["crypto"]), (7, 100, True))

    def test_general_tab_mail_and_costs(self):
        body = {"email": "me@example.com", "open": True, "close": False, "daily": True, "problems": True, "lab": False,
                "open_every": 3, "close_every": 1, "quiet": True, "quiet_from": "02:00", "quiet_to": "10:00",
                "quiet_problems": True}
        self.post("/api/notify/settings", body)
        n = self.state()["notify"]
        for key, value in body.items():
            self.assertEqual(n[key], value, key)
        self.post("/api/costs/settings", {"hosting": 25})
        self.assertEqual(self.state()["operating_costs"]["hosting"], 25)


class ProvenStrategyTest(unittest.TestCase):
    """A lab strategy that proves itself: a mail, and with the switch on, a push into the simulation
    (never while real money trades crypto)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = Config(assets=("btc",), db_path=os.path.join(self.tmp.name, "t.sqlite"), lab_assets=("btc",))
        self.engine = Engine(cfg, api=FakeApi(), env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64,
                                                      "SMTP_USER": "a@gmail.com", "SMTP_PASSWORD": "x"}, streaming=False)
        self.engine.notifier.update(email="me@example.com")
        self.sent = []
        self.engine.notifier._safe_send = lambda subject, body, images=(): self.sent.append((subject, body))
        self.engine.notifier._maybe_daily = lambda now: None  # the daily summary, due late in the evening, isn't tested here
        if self.engine.notifier.atlas_mail is not None:  # (nor Atlas's, at 23:30: a run after it would count it)
            self.engine.notifier.atlas_mail.maybe_daily = lambda *a, **k: None
        self.row = {"asset": "btc", "minutes": 15, "variant": "twap|0.08|15-180", "rule": "rule", "status": "proven",
                    "bets": 90, "roi": 0.12, "roi_pessimistic": 0.02, "roi_recent": 0.1,
                    "oos_bets": 40, "oos_roi": 0.15, "oos_pessimistic": 0.03}

    def tearDown(self):
        self.engine.store.close()
        self.tmp.cleanup()

    def prove(self):
        self.engine.lab = {"board": [self.row]}
        self.engine._on_proven([self.row])
        self.engine.notifier.tick(time.time())

    def main_variant(self):
        return [m["variant"] for m in self.engine.main_strategies() if (m["asset"], m["minutes"]) == ("btc", 15)]

    def test_first_run_is_silent_then_a_new_proven_strategy_is_mailed(self):
        self.engine._on_proven([])  # what was proven before the feature counts as seen
        before = self.main_variant()
        self.prove()
        self.assertEqual(len(self.sent), 1)
        self.assertIn("הוכחה", self.sent[0][0])
        self.assertEqual(self.main_variant(), before)  # the switch is off by default: nothing pushed
        self.prove()
        self.assertEqual(len(self.sent), 1)  # mailed once

    def test_switch_on_pushes_into_the_simulation(self):
        self.engine._on_proven([])
        self.engine.set_auto_push(True)
        self.prove()
        self.assertIn("twap|0.08|15-180", self.main_variant())
        self.assertIn("נדחפה", self.sent[0][0])
        self.assertTrue(self.engine.strategy_versions()[-1]["label"].startswith("דחיפה אוטומטית"))

    def live_variant(self):
        return [m["variant"] for m in self.engine.live_main() if (m["asset"], m["minutes"]) == ("btc", 15)]

    def test_simulation_and_real_money_each_follow_their_own_switch(self):
        self.engine._on_proven([])
        self.engine.set_auto_push(True)  # the simulation's switch only
        before = self.live_variant()
        self.prove()
        self.assertIn("twap|0.08|15-180", self.main_variant())
        self.assertEqual(self.live_variant(), before)  # real money didn't move
        self.engine.lab = {"board": [self.row]}  # (the push recomputed the board from the empty test database)
        self.assertEqual(self.engine.recommendation("live")["changes"][0]["variant"], "twap|0.08|15-180")
        self.assertIn("לסימולציה", self.sent[0][1])
        # Pushed by the owner to real money: a version of its own, and the trackers trade it.
        self.engine.push([("btc", 15, "twap|0.08|15-180")], mode="live")
        self.assertIn("twap|0.08|15-180", self.live_variant())
        self.assertEqual(self.engine.strategy_versions("live")[-1]["kind"], "push")
        t = next(t for t in self.engine.trackers if (t.asset, t.minutes) == ("btc", 15))
        self.assertIn("twap|0.08|15-180", [r.key for r in t.live_rules])  # next to the one already there
        self.engine.rollback(1, mode="live")
        self.assertEqual(self.live_variant(), before)
        self.assertIn("twap|0.08|15-180", self.main_variant())  # the simulation's stays

    def test_real_money_switch_pushes_to_real_money(self):
        self.engine._on_proven([])
        self.engine.set_auto_push(True, "live")
        before = self.main_variant()
        self.prove()
        self.assertIn("twap|0.08|15-180", self.live_variant())
        self.assertEqual(self.main_variant(), before)
        self.assertIn("לכסף אמיתי", self.sent[0][1])

    def test_no_mail_when_switched_off(self):
        self.engine._on_proven([])
        self.engine.notifier.update(lab=False)
        self.prove()
        self.assertEqual(self.sent, [])


class ExportTest(unittest.TestCase):
    def test_zip_has_every_trade_exactly_and_the_download_works(self):
        import csv as _csv, io as _io, threading, urllib.request, zipfile
        from http.server import ThreadingHTTPServer
        from polybot.web import make_handler
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            price = 0.1 + 0.2  # 0.30000000000000004: must come back exactly
            fill = find_bet(0.9, [(price, 100)], [], dataclasses.replace(cfg, fee_rate=0.0, min_edge=0.01))
            engine.store.add_tick(ts=time.time(), asset="btc", minutes=15, window_start=900, price=1, open_price=1, seconds_left=300, vol=0.001, fair_up=0.5, up_bid=0.4, up_ask=0.5, down_bid=0.4, down_ask=0.5, price_age=0, book_age=0, fair_raw=0.5)
            engine.store.add_bet(1.5, "btc", 15, 900, 300, fill, variant="twap|0.05|15-600", stake=10.0, market="m-1")
            engine.store.add_bet(2.5, "eth", 15, 900, 300, fill, mode="lab", variant="x")
            engine.store.set_setting("notify_email", "me@example.com")
            buf = _io.BytesIO()
            counts = engine.export(buf)
            self.assertEqual(counts["bets"], 2)
            z = zipfile.ZipFile(buf)
            self.assertEqual(set(z.namelist()), {"bets.csv", "windows.csv", "research.csv", "settings.json", "manifest.json", "README.txt"})
            rows = list(_csv.DictReader(_io.TextIOWrapper(z.open("bets.csv"), encoding="utf-8")))
            self.assertEqual(float(rows[0]["avg_price"]), fill.avg_price)
            self.assertEqual(float(rows[0]["shares"]), fill.shares)  # no rounding
            self.assertEqual((rows[0]["mode"], rows[0]["stake"], rows[0]["market"], rows[0]["pnl"]), ("paper", "10.0", "m-1", ""))
            self.assertEqual(rows[1]["mode"], "lab")
            self.assertEqual(json.loads(z.read("settings.json"))["notify_email"], "me@example.com")
            self.assertNotIn("ticks.csv", z.namelist())
            # The full one adds the recording; both download through the web server.
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/export?what=full", timeout=10) as r:
                    self.assertIn("attachment", r.headers["Content-Disposition"])
                    full = zipfile.ZipFile(_io.BytesIO(r.read()))
                    self.assertEqual(len(full.read("ticks.csv").decode().strip().splitlines()), 2)  # header + the recent tick
            finally:
                server.shutdown()
                server.server_close()
            engine.store.close()


class PortfolioTest(unittest.TestCase):
    """Several strategies on one market, each betting once a window; and why a rule with an edge
    didn't bet."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = Config(assets=("btc",), db_path=os.path.join(self.tmp.name, "t.sqlite"), lab_assets=("btc",))
        self.engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
        self.t = next(t for t in self.engine.trackers if t.key == "btc-15m")
        self.t.window, self.t.market = 900, {"up_token": "U", "down_token": "D", "tick_size": 0.01, "min_order_size": 1}
        self.t._set_market_cfg(self.t.market)
        rows = [{"asset": "btc", "minutes": 15, "variant": v, "rule": v, "status": st, "roi_pessimistic": p,
                 "bets": 50, "roi": 0.1, "roi_recent": 0.1, "oos_bets": 30, "oos_roi": 0.1, "oos_pessimistic": 0.02}
                for v, st, p in (("twap|0.08|15-180", "proven", 0.03), ("twap|0.03|15-60", "proven", 0.05),
                                 ("twap|0.12|15-600", "proven", 0.04), ("twapt|0.05|15-600", "promising", -0.1))]
        self.engine.lab = {"board": rows}
        self.engine.refresh_lab = lambda: None  # keep the test board

    def tearDown(self):
        self.engine.store.close()
        self.tmp.cleanup()

    def keys(self, mode="sim"):
        return [m["variant"] for m in self.engine.strategies_for(mode) if m["asset"] == "btc"]

    def test_add_alongside_up_to_three_then_the_weakest_makes_room(self):
        self.engine.push([("btc", 15, "twap|0.08|15-180")])
        self.assertEqual(self.keys(), ["twap|0.05|15-600", "twap|0.08|15-180"])
        self.assertEqual([r.key for r in self.t.main_rules], self.keys())
        self.engine.push([("btc", 15, "twap|0.03|15-60")])
        self.engine.push([("btc", 15, "twap|0.12|15-600")])  # a fourth: the default (no lab score) goes
        self.assertEqual(sorted(self.keys()), ["twap|0.03|15-60", "twap|0.08|15-180", "twap|0.12|15-600"])
        with self.assertRaises(ValueError):
            self.engine.push([("btc", 15, "twap|0.08|15-180")])  # already running
        with self.assertRaises(ValueError):
            self.engine.push([("btc", 15, "twapt|0.05|15-600")], mode="live", allow_unproven=True)  # real money: proven only
        self.engine.push([("btc", 15, "twapt|0.05|15-600")], allow_unproven=True, replace=True)  # an experiment, alone
        self.assertEqual(self.keys(), ["twapt|0.05|15-600"])
        self.engine.push([("btc", 15, "twap|0.08|15-180")])
        self.engine.remove_strategy("btc", 15, "twapt|0.05|15-600")
        self.assertEqual(self.keys(), ["twap|0.08|15-180"])
        self.engine.remove_strategy("btc", 15, "twap|0.08|15-180")  # the last one too: the owner's call
        self.assertEqual(self.engine.main_strategies(), [])  # empty stays empty (not "back to the start")
        self.assertEqual(self.t.main_rules, [])

    def test_each_rule_bets_once_a_window(self):
        self.engine.push([("btc", 15, "twap|0.03|15-60")])
        self.engine.set_sim_limits(same_direction=0)
        rules = self.t.main_rules
        for _ in range(3):
            for r in rules:
                self.t._sim_bet(time.time(), 30, r, 0.9, 0.9, [(0.55, 10_000)], [(0.99, 10)], 0.1, 0.1)
        got = self.engine.store._x("SELECT variant FROM bets WHERE mode = 'paper'")
        self.assertEqual(sorted(v for (v,) in got), sorted(r.key for r in rules))

    def test_why_a_rule_with_an_edge_did_not_bet(self):
        self.engine.set_lab_exact("sim", False)  # these check the simulation's own filters
        self.engine.set_sim_limits(stake=50, same_direction=0)
        rule = self.t.main_rule
        # 20 shares at 55c is $11: an edge, but not the whole $50.
        self.t._sim_bet(time.time(), 30, rule, 0.9, 0.9, [(0.55, 20), (0.99, 1000)], [], 0.1, 0.1)
        self.t._sim_bet(time.time(), 30, rule, 0.9, 0.9, [(0.55, 20), (0.99, 1000)], [], 0.1, 0.1)  # counted once a window
        own = lab.Variant.from_key(rule.key + "|r1.0")  # its own filter leaves 55c out: part of the rule, no skip
        self.t._sim_bet(time.time(), 30, own, 0.9, 0.9, [(0.55, 10_000)], [], 0.1, 0.1)
        from polybot.store import connect
        db = connect(self.engine.cfg.db_path)
        why = self.engine.why_no_trades(db, time.time())
        db.close()
        self.assertEqual(why["skipped"], {"full": 1, "limits": 0})
        self.assertEqual(why["trades"], 0)


class LastMinuteTest(unittest.TestCase):
    def test_buys_the_side_the_locked_average_favours_only_when_the_market_sells_it_cheap(self):
        from polybot.paper import find_lock
        cfg = dataclasses.replace(Config(), fee_rate=0.0, stake_usd=5.0)
        fill = find_lock(0.80, [(0.40, 1000)], [(0.62, 1000)], 0.75, 0.50, 0.02, cfg)  # the model 80%, sold at 40c
        self.assertEqual((fill.side, fill.avg_price), ("up", 0.40))
        self.assertAlmostEqual(fill.cost, 5.0)
        self.assertIsNone(find_lock(0.70, [(0.40, 1000)], [], 0.75, 0.50, 0.02, cfg))  # the model under 75%
        self.assertIsNone(find_lock(0.80, [(0.55, 1000)], [], 0.75, 0.50, 0.02, cfg))  # dearer than the cap
        self.assertIsNone(find_lock(0.80, [(0.40, 5)], [], 0.75, 0.50, 0.02, cfg))  # $2 on offer, not the whole $5

    def test_the_lab_runs_it_in_its_last_seconds_with_five_dollars(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_fill_delay=0, lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            t = next(t for t in engine.trackers if t.key == "btc-15m")
            t.window, t.market = 900, {"tick_size": 0.01, "fee_rate": 0.0}
            t.variants = [lab.Variant(0.50, 3, 60, "lock75")]
            t._set_market_cfg(t.market)
            fair = {"twap": 0.85, "point": 0.85, "twapt": 0.85}
            t._run_lab(1000, 120, fair, fair, [(0.40, 1000)], [(0.62, 1000)], 0.1, 0.1)  # two minutes left: not yet
            t._run_lab(1001, 30, fair, fair, [(0.40, 1000)], [(0.62, 1000)], 0.1, 0.1)
            rows = engine.store._x("SELECT side, cost + fee, variant FROM bets WHERE mode = 'lab'")
            self.assertEqual(rows, [("up", 5.0, "lock75|0.50|3-60")])
            self.assertEqual(lab.family("lock75|0.50|3-60"), "main")  # (its tab is gone: an old trade shows with the rest)
            engine.store.close()


class BothSidesTest(unittest.TestCase):
    """The both-sides rule: the cheap side first, then the same shares of the other side once the
    two cost at most the target together; a round with both legs makes money whoever wins."""

    def _setup(self, tmp, v):
        cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
        engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
        t = next(t for t in engine.trackers if t.key == "btc-15m")
        t.window, t.market = 900, {"tick_size": 0.01, "fee_rate": 0.0}
        t.variants = [v]
        t._set_market_cfg(t.market)
        return engine, t

    def legs(self, engine):
        return engine.store._x("SELECT side, shares, cost + fee, ts FROM bets WHERE mode = 'lab' ORDER BY id")

    def test_first_the_cheap_side_then_the_other_when_the_pair_is_cheap_enough(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = lab.Variant(0.35, 120, 885, "pair94")
            engine, t = self._setup(tmp, v)
            fair = {"twap": 0.3, "point": 0.3, "twapt": 0.3}
            t._run_lab(1000, 600, fair, fair, [(0.30, 1000)], [(0.72, 1000)], 0.1, 0.1)  # up at 30c: first leg
            t._run_lab(1010, 590, fair, fair, [(0.30, 1000)], [(0.66, 1000)], 0.1, 0.1)  # 30 + 66 = 96 > 94: wait
            t._run_lab(1020, 580, fair, fair, [(0.20, 1000)], [(0.70, 1000)], 0.1, 0.1)  # no second first leg
            t._run_lab(1030, 570, fair, fair, [(0.30, 1000)], [(0.62, 1000)], 0.1, 0.1)  # 30 + 62 = 92: second leg
            t._run_lab(1040, 560, fair, fair, [(0.10, 1000)], [(0.10, 1000)], 0.1, 0.1)  # done for this round
            legs = self.legs(engine)
            self.assertEqual([(r[0], r[1], r[3]) for r in legs], [("up", 10, 1000), ("down", 10, 1030)])
            self.assertAlmostEqual(sum(r[2] for r in legs), 9.2)  # 10 pairs for $9.20, paying $10 whoever wins
            engine.store.close()

    def test_first_leg_only_in_its_time_and_second_not_at_the_last_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._setup(tmp, lab.Variant(0.35, 120, 885, "pair94"))
            fair = {"twap": 0.5, "point": 0.5, "twapt": 0.5}
            t._run_lab(1000, 100, fair, fair, [(0.30, 1000)], [(0.72, 1000)], 0.1, 0.1)  # under 2 minutes left
            self.assertEqual(self.legs(engine), [])
            t._run_lab(1010, 200, fair, fair, [(0.30, 1000)], [(0.72, 1000)], 0.1, 0.1)
            t._run_lab(1300, 3, fair, fair, [(0.30, 1000)], [(0.50, 1000)], 0.1, 0.1)  # 3s before the close
            self.assertEqual([r[0] for r in self.legs(engine)], ["up"])
            t._run_lab(1290, 10, fair, fair, [(0.95, 1000)], [(0.30, 5)], 0.1, 0.1)  # 5 shares on offer, not 10
            self.assertEqual([r[0] for r in self.legs(engine)], ["up"])
            engine.store.close()

    def test_a_restart_remembers_the_first_leg(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = lab.Variant(0.35, 120, 885, "pair94")
            engine, t = self._setup(tmp, v)
            fair = {"twap": 0.5, "point": 0.5, "twapt": 0.5}
            t._run_lab(1000, 600, fair, fair, [(0.30, 1000)], [(0.72, 1000)], 0.1, 0.1)
            t.window = None
            t._new_window(900, 1001, 100.0)  # what a restart does: the window's bets read back
            t.market = {"tick_size": 0.01}
            self.assertEqual(t.pair_legs[v.key][0], "up")
            t._run_lab(1010, 590, fair, fair, [(0.25, 1000)], [(0.60, 1000)], 0.1, 0.1)
            self.assertEqual([r[0] for r in self.legs(engine)], ["up", "down"])
            engine.store.close()

    def test_the_board_counts_a_round_once_and_the_locked_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            key = "pair94|0.35|120-885"
            rows = [(900, "up", 3.0, 7.0), (900, "down", 6.2, -6.2),  # locked: +0.8 together
                    (1800, "up", 3.0, -3.0)]  # only the first leg, and it lost
            for ws, side, paid, pnl in rows:
                store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost,"
                         " fee, pnl, winner) VALUES ('lab', ?, 'filled', ?, 'btc', 15, ?, ?, 10, ?, 0, ?, 'up')",
                         (key, ws + 100, ws, side, paid, pnl))
            db = connect(cfg.db_path)
            board = lab.leaderboard(db, cfg)
            row = board["board"][0]
            self.assertEqual((row["family"], row["bets"], row["wins"], row["locked"]), ("pair", 2, 1, 1))
            self.assertAlmostEqual(row["pnl"], 0.8 - 3.0)
            self.assertIn("pair", board["families"])  # (its own tab again: the locked pair, 30.10.2026)
            db.close()
            store.close()

    def test_it_runs_in_the_lab_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            with self.assertRaises(ValueError):
                engine.put_back_strategy("btc", 15, "pair94|0.35|120-885")
            engine.store.close()


class LotteryAndEntriesTest(unittest.TestCase):
    def test_lottery_ticket_buys_only_cheap_shares_the_model_believes_in(self):
        cfg = dataclasses.replace(Config(), fee_rate=0.0, stake_usd=10.0)
        from polybot.paper import find_lottery
        fill = find_lottery(0.15, [(0.01, 5000)], [(0.98, 5000)], 0.05, 0.10, cfg)
        self.assertEqual((fill.side, fill.avg_price), ("up", 0.01))
        self.assertAlmostEqual(fill.shares, 1000)
        self.assertIsNone(find_lottery(0.05, [(0.01, 5000)], [(0.98, 5000)], 0.05, 0.10, cfg))  # the model says 5%
        self.assertIsNone(find_lottery(0.15, [(0.06, 5000)], [], 0.05, 0.0, cfg))  # dearer than the cap
        self.assertIsNone(find_lottery(0.15, [(0.01, 100)], [], 0.05, 0.0, cfg))  # $1 on offer, not the whole $10

    def test_several_entries_spaced_out_with_a_third_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            t = next(t for t in engine.trackers if t.key == "btc-15m")
            t.window, t.market = 900, {"tick_size": 0.01}
            t.variants = [lab.Variant(0.05, 15, 600, "twap3")]
            t._set_market_cfg(t.market)
            fair = {"twap": 0.9, "point": 0.9, "twapt": 0.9}
            for now in (1000, 1005, 1030, 1060, 1100):  # 1005 is too soon after 1000; the fourth is one too many
                t._run_lab(now, 300, fair, fair, [(0.55, 10_000)], [(0.99, 10)], 0.1, 0.1)
            rows = engine.store._x("SELECT ts, cost + fee FROM bets WHERE mode = 'lab'")
            self.assertEqual([r[0] for r in rows], [1000, 1030, 1060])
            self.assertTrue(all(abs(r[1] - 10 / 3) < 1e-6 for r in rows))
            engine.store.close()

    def test_a_market_led_by_a_new_kind_of_rule_still_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            api = FakeApi()
            store = Store(cfg.db_path)
            t = Tracker("btc", cfg, store, api=api, main_rule=lab.Variant(0.05, 15, 600, "twap3"))
            start = feeds.window_start(1_800_000_000, 15)
            for i in range(300):
                api.price *= 1.0005 if i % 2 else 1 / 1.0005
                t.step(start - 300 + i)
            for i in range(3):
                t.step(start + i)
            api.price = 101.0
            t.step(start + 600)  # used to fail: the dashboard reading looked up "twap3" as a model reading
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM bets WHERE mode = 'paper'").fetchone()[0], 1)
            store.close()

    def test_a_mix_all_market_stops_the_simulation_but_never_the_lab(self):
        def run(blend_sim):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_fill_delay=0)
                api = FakeApi()
                store = Store(cfg.db_path)
                t = Tracker("btc", cfg, store, api=api, variants=[lab.Variant(0.05, 15, 600, "twap")])
                start = feeds.window_start(1_800_000_000, 15)
                for i in range(300):
                    api.price *= 1.0005 if i % 2 else 1 / 1.0005
                    t.step(start - 300 + i)
                for i in range(3):
                    t.step(start + i)
                t.blend, t.blend_sim = (0.0, 1.0), blend_sim  # learned: 0% our model, 100% the market price
                api.price = 101.0
                t.step(start + 600)
                out = dict(store.db.execute("SELECT mode, COUNT(*) FROM bets GROUP BY mode").fetchall())
                store.close()
                return out
        self.assertEqual(run(True), {"lab": 1})  # the simulation sees no edge at all; the lab still measures the model
        self.assertEqual(run(False), {"lab": 1, "paper": 1})  # the owner switched the mix off for the simulation

    def test_the_lab_counted_once_per_round(self):
        rows = [{"asset": "btc", "minutes": 5, "window_start": 0, "side": "up", "pnl": 990.0, "paid": 10.0}] * 29 + \
               [{"asset": "btc", "minutes": 5, "window_start": 300, "side": "up", "pnl": -10.0, "paid": 10.0}] * 5
        out = lab.per_round(rows)
        self.assertEqual(out["rounds"], 2)
        self.assertAlmostEqual(out["pnl"], 980.0)  # +990 once, -10 once: not 29 times the win



class StateShapeTest(unittest.TestCase):
    def test_no_key_appears_twice_in_the_dashboard_state(self):
        # A repeated key in a dict literal silently replaces the first (it once hid the control
        # center's data behind the strategy's manager): none in engine.py.
        import ast
        tree = ast.parse(open(os.path.join(os.path.dirname(__file__), "..", "polybot", "engine.py"), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                dupes = {k for k in keys if keys.count(k) > 1}
                self.assertFalse(dupes, f"repeated keys at line {node.lineno}: {dupes}")


class ManageStrategiesTest(unittest.TestCase):
    def test_pause_resume_restore_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc", "eth"), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.refresh_lab = lambda: None
            base = engine.base_strategies()[0]
            btc = next(t for t in engine.trackers if t.asset == "btc")
            engine.pause_strategy("btc", 15, base["variant"])
            self.assertEqual(btc.main_rules, [])  # paused: listed, not trading
            self.assertTrue(next(m for m in engine.main_strategies() if m["asset"] == "btc")["paused"])
            self.assertEqual(len(engine.live_main()), 2)  # real money's own list is untouched
            engine.pause_strategy("btc", 15, base["variant"], paused=False)
            self.assertEqual([r.key for r in btc.main_rules], [base["variant"]])
            engine.remove_strategy("eth", 15, base["variant"])
            engine.pause_strategy("btc", 15, base["variant"])
            engine.restore_base()  # the original ones back: eth re-added, btc resumed
            self.assertEqual(sorted((m["asset"], bool(m.get("paused"))) for m in engine.main_strategies()),
                             [("btc", False), ("eth", False)])
            with self.assertRaises(ValueError):
                engine.restore_base()
            engine.set_strategy_name("btc", 15, base["variant"], "  הבסיסית של ביטקוין ")
            self.assertEqual(engine.strategy_names(), {f"btc|15|{base['variant']}": "הבסיסית של ביטקוין"})
            engine.set_strategy_name("btc", 15, base["variant"], "")
            self.assertEqual(engine.strategy_names(), {})
            engine.store.close()


class StakeAndLearningTest(unittest.TestCase):
    def test_a_bet_keeps_the_book_its_rule_would_buy(self):
        cfg = Config(min_edge=0.05, fee_rate=0.0)
        fill = find_bet(0.70, [(0.50, 10), (0.60, 10), (0.66, 10)], [(0.40, 10)], cfg, stake=4)
        self.assertEqual(fill.side, "up")
        self.assertEqual(fill.depth, [(0.50, 10), (0.60, 10)])  # 0.66 leaves less than the 5-point edge
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            store.add_bet(1.0, "btc", 15, 0, 100, fill, mode="lab", variant="twap|0.05|15-600")
            self.assertEqual(json.loads(store._x("SELECT book FROM bets")[0][0]), [[0.5, 10], [0.6, 10]])
            store.close()

    def test_the_stake_curve_finds_where_profit_peaks(self):
        from polybot import stakes
        cfg = Config(fee_rate=0.0)
        # Cheap shares only for the first $5; then dear ones that lose on average.
        levels = [(0.50, 10), (0.95, 1000)]
        self.assertAlmostEqual(stakes.replay(levels, 5, "up", "up", cfg)[0], 5.0)  # 10 shares win $10
        self.assertIsNone(stakes.replay([(0.5, 2)], 5, "up", "up", cfg))  # the book can't take it all
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            for i in range(60):  # wins 60% of the time
                fill = find_bet(0.9, levels, [], Config(min_edge=-1, fee_rate=0.0), stake=5)
                fill.depth = levels
                store.add_bet(float(i), "btc", 15, i, 100, fill, mode="lab", variant="twap|0.05|15-600")
                winner = "up" if i % 5 < 3 else "down"
                store.settle_bet(i + 1, winner, 0.0)
            db = connect(store.path)
            curve = stakes.curve(db, cfg, {("btc", 15, "twap|0.05|15-600")})
            self.assertEqual(curve["trades"], 60)
            self.assertEqual(curve["best"], 5)  # past $5 every dollar buys 95¢ shares that win 60%
            self.assertEqual(stakes.curve(db, cfg, set())["best"], None)
            store.close()

    def test_learning_runs_the_moment_trades_settle(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            api = FakeApi()
            engine = Engine(cfg, api=api, env={}, streaming=False, with_lab=False)
            ran = []
            engine.refresh_analysis = lambda: ran.append("analysis")
            engine.refresh_lab = lambda: ran.append("lab")
            engine._settle()
            self.assertEqual(ran, [])  # nothing settled: nothing new to learn
            start = int(time.time()) - 15 * 60 - 60
            engine.store.upsert_window("btc", 15, start, slug="s", found=1)
            fill = find_bet(0.9, [(0.5, 100)], [], Config(min_edge=0.05, fee_rate=0.0), stake=10)
            engine.store.add_bet(time.time(), "btc", 15, start, 100, fill)
            api.winner = "up"
            engine._settle()
            for _ in range(100):
                if "learned" in engine.learning:
                    break
                time.sleep(0.02)
            self.assertEqual(ran, ["analysis", "lab"])
            self.assertEqual(engine.learning["last_settled"]["bets"], 1)
            self.assertEqual(engine.learning["learned"]["reason"], "settle")
            pulse = engine.pulse()
            json.dumps(pulse)
            self.assertEqual({"price_age", "book_age", "reaction", "learning", "stake", "markets"} <= set(pulse), True)
            engine.store.close()

    def test_the_simulation_runs_exactly_like_the_lab_and_real_money_can_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            t = engine.trackers[0]
            self.assertEqual(engine.lab_exact(), {"sim": True, "live": True})  # (real money as the simulation)
            engine.set_live_mirror(False)
            self.assertEqual(engine.lab_exact(), {"sim": True, "live": False})
            engine.set_sim_limits(max_open={"crypto": 1}, same_direction=1)
            engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                            " VALUES ('paper', 'filled', ?, 'eth', 15, 0, 'up', 20, 10, 0)", (time.time(),))
            engine.set_blend_use("sim", True)
            self.assertFalse(t.blend_sim)  # our model only, like the lab
            self.assertTrue(engine.can_open("crypto", 100, "up", 0))  # no caps (one open, same direction)
            self.assertEqual(engine.protections()["sim"]["same_direction"], 1)  # what would apply, still shown
            engine.set_sim_limits(daily_loss=50)
            self.assertTrue(engine.can_open("crypto", 100, "up", 0))  # the day's limit never stops the simulation (red only)
            engine.set_lab_exact("sim", False)
            self.assertTrue(t.blend_sim)
            self.assertFalse(engine.can_open("crypto", 100, "up", 0))
            # Real money: its protections apply until the owner switches them off.
            engine.live.trades = lambda strategy: True
            engine.set_money("live", {"daily_loss": 1000})
            engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                            " VALUES ('live', 'filled', ?, 'eth', 15, 0, 'up', 20, 10, 0)", (time.time(),))
            self.assertIsNotNone(engine.live.blocked("crypto", "up", 0))  # one Up already in this window
            engine.set_lab_exact("live", True)
            self.assertIsNone(engine.live.blocked("crypto", "up", 0))
            engine.store.close()


class TakeOutPutBackDeleteTest(unittest.TestCase):
    def test_originals_can_go_come_back_and_be_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc", "eth"), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.refresh_lab = lambda: None
            v = engine.base_strategies()[0]["variant"]
            engine.remove_strategy("btc", 15, v)
            engine.remove_strategy("btc", 15, v)  # a second click: nothing more to do
            engine.remove_strategy("eth", 15, v)
            self.assertEqual(engine.main_strategies(), [])
            self.assertEqual(len(engine.live_main()), 2)  # real money keeps its own list
            self.assertEqual(engine.removed_strategies()["sim"], [["btc", 15, v], ["eth", 15, v]])
            engine.remove_strategy("btc", 15, v, mode="live")
            engine.remove_strategy("eth", 15, v, mode="live")
            self.assertEqual(engine.live_main(), [])  # empty: never falls back to the simulation's list
            engine.put_back_strategy("btc", 15, v)
            self.assertEqual([m["asset"] for m in engine.main_strategies()], ["btc"])
            self.assertEqual(engine.removed_strategies()["sim"], [["eth", 15, v]])
            engine.delete_strategy("btc", 15, v)  # gone everywhere and off the lists
            self.assertEqual(engine.main_strategies(), [])
            self.assertIn(f"btc|15|{v}", engine.deleted_strategies())
            self.assertEqual(engine.removed_strategies()["live"], [["eth", 15, v]])
            engine.restore_base()  # the originals come back, deleted ones included
            self.assertEqual(sorted(m["asset"] for m in engine.main_strategies()), ["btc", "eth"])
            self.assertEqual(engine.deleted_strategies(), set())
            engine.store.close()


class RatioVariantsTest(unittest.TestCase):
    def test_a_rule_carries_its_own_risk_reward_filter(self):
        v = lab.Variant.from_key("twap|0.05|15-600|r0.7")
        self.assertEqual((v.min_edge, v.min_ratio, v.key), (0.05, 0.7, "twap|0.05|15-600|r0.7"))
        self.assertEqual(lab.Variant.from_key("twap|0.05|15-600").min_ratio, 0.0)
        self.assertIn("$7", lab.describe(v.key))
        self.assertEqual(lab.ratio_versions("twap|0.05|15-600"),
                         ["twap|0.05|15-600|r0.3", "twap|0.05|15-600|r0.5", "twap|0.05|15-600|r0.7"])
        self.assertEqual(lab.ratio_versions("lotto0|0.02|15-60"), [])  # tickets are capped already
        self.assertEqual(lab.ratio_versions(v.key), [])
        self.assertTrue(all(k.endswith("|r0.7") for k in lab.neighbours(v.key, 15)))

    def test_lab_simulation_and_real_money_apply_it_alike(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",), fee_rate=0.0)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.set_lab_focus(False)  # (the whole grid: the lab's focus on the locked pair is off)
            engine.refresh_lab = lambda: None
            t = next(x for x in engine.trackers if x.asset == "btc" and x.minutes == 15)
            rule = lab.Variant.from_key("twap|0.05|15-600|r0.7")  # buys up to ~59c
            self.assertEqual(t._lab_cfg(rule).min_ratio, 0.7)
            self.assertEqual(t._rule_cfg(rule).min_ratio, 0.7)
            self.assertIsNone(t._decide(rule, 0.9, [(0.70, 100)], [], t._rule_cfg(rule)))  # 70c: too dear
            self.assertIsNotNone(t._decide(rule, 0.9, [(0.55, 100)], [], t._rule_cfg(rule)))
            # The search adds the filtered versions of the owner's own strategies.
            engine._last_explore = 0
            engine.explore([], time.time())
            added = engine._json_setting("lab_dynamic", {}).get("btc|15", [])
            base = engine.base_strategies()[0]["variant"]
            self.assertEqual(sorted(added), sorted(lab.ratio_versions(base)))
            self.assertIn(base + "|r0.5", {v.key for v in t.variants})
            engine.store.close()


class MoneyStripTest(unittest.TestCase):
    def test_system_values_owner_values_and_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc", "eth"), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64}, streaming=False, with_lab=False)
            M = engine.money()["live"]
            self.assertEqual((M["stake"]["value"], M["daily_loss"]["value"], M["copies"]["value"]), (5.0, 20.0, 1))
            self.assertFalse(any(M[k]["override"] for k in ("stake", "daily_loss", "copies")))
            engine.stake_curve = {"live": {"best": 15}, "sim": {"best": None}}  # the data found a better amount
            engine._apply_money()
            self.assertEqual((engine.live.stake(), engine.live.daily_loss_limit()), (15.0, 60.0))
            engine.set_money("live", {"stake": 8, "copies": 3})
            self.assertEqual(engine.live.stake(), 8.0)
            self.assertEqual(engine.trackers[0].live_copies, 3)
            self.assertTrue(engine.money()["live"]["stake"]["override"])
            engine.set_money("live", {}, reset=True)
            self.assertEqual((engine.live.stake(), engine.trackers[0].live_copies), (15.0, 1))
            with self.assertRaises(ValueError):
                engine.set_money("live", {"copies": 20})
            engine.store.close()

    def test_old_amounts_stay_the_owners(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            store.set_setting("sim_limits", json.dumps({"stake": 4}))
            store.set_setting("live_stake", 6)
            store.close()
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            M = engine.money()
            self.assertEqual((M["sim"]["stake"]["value"], M["sim"]["stake"]["override"]), (4.0, True))
            self.assertEqual((M["live"]["stake"]["value"], M["live"]["daily_loss"]["override"]), (6.0, False))
            engine.store.close()

    def test_extra_entries_take_the_next_shares_at_nearly_the_same_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), fee_rate=0.0)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.set_money("sim", {"stake": 10, "copies": 3})
            t = engine.trackers[0]
            t.window, t.market = 900, {"tick_size": 0.01}
            t._set_market_cfg(t.market)
            rule = t.main_rule
            # $10 at 50c, $10 more at 50.5c, then only 70c: two entries, not three.
            t._sim_bet(time.time(), rule.min_seconds_left + 1, rule, 0.9, 0.9, [(0.50, 20), (0.505, 20), (0.70, 1000)], [], 0.1, 0.1)
            rows = engine.store._x("SELECT avg_price, cost FROM bets WHERE mode = 'paper' ORDER BY id")
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[0][0], 0.50)
            self.assertAlmostEqual(rows[1][0], 0.505, places=3)
            engine.store.close()


class PutBackFromVersionsTest(unittest.TestCase):
    def test_a_removal_version_names_what_it_took_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc", "sol"), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.refresh_lab = lambda: None
            v = engine.base_strategies()[0]["variant"]
            engine.remove_strategy("sol", 15, v)
            top = engine.versions_view("sim")[0]
            self.assertEqual((top["kind"], top["removed"], top["removed_back"]), ("remove", [["sol", 15, v]], False))
            engine.put_back_strategy("sol", 15, v, "sim")  # back where it was taken from
            self.assertIn("sol", [m["asset"] for m in engine.main_strategies()])
            self.assertTrue(next(x for x in engine.versions_view("sim") if x["kind"] == "remove")["removed_back"])
            self.assertEqual(len(engine.live_main()), 2)  # real money untouched
            engine.store.close()


class LabKeepsTheOwnersStrategiesTest(unittest.TestCase):
    def test_a_connected_strategy_always_runs_in_the_lab(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.refresh_lab = lambda: None
            odd = "twap3|0.03|15-600"  # not on the lab's own grid (an old search addition, say)
            t = next(x for x in engine.trackers if x.key == "btc-15m")
            self.assertNotIn(odd, {v.key for v in t.variants})
            engine.put_back_strategy("btc", 15, odd, "sim")
            self.assertIn(odd, {v.key for v in t.variants})  # the lab measures what the owner runs
            # The search never retires one of the owner's strategies, even a losing one.
            engine.store.set_setting("lab_dynamic", json.dumps({"btc|15": [odd]}))
            engine._refresh_variant_set()
            engine._last_explore = 0
            engine.explore([{"asset": "btc", "minutes": 15, "variant": odd, "status": "losing", "bets": 99,
                             "roi": -0.2, "roi_pessimistic": -0.3}], time.time())
            self.assertIn(odd, engine._json_setting("lab_dynamic", {}).get("btc|15", []))
            self.assertIn("3 כניסות", engine.state()["strategy_rules"][f"btc|15|{odd}"])
            engine.store.close()


class RiskRewardResearchTest(unittest.TestCase):
    @staticmethod
    def trades(n, dear_wins):
        """Cheap entries (55c) win 70%; dear ones (90c) win `dear_wins` of the time. Oldest first."""
        out = []
        for i in range(n):
            if i % 2:
                won = (i // 2) % 10 < 7
                out.append((0.55, (10 / 0.55 - 10) if won else -10, 10))
            else:
                won = (i // 2) % 10 < round(dear_wins * 10)
                out.append((0.90, (10 / 0.90 - 10) if won else -10, 10))
        return out

    def test_a_filter_that_helps_on_old_and_new_trades_is_recommended(self):
        from polybot import rrlab
        st = rrlab.study(self.trades(120, 0.80))  # 90c needs 90% to break even: the dear ones lose
        self.assertEqual(st["verdict"], "change")
        self.assertGreaterEqual(st["recommended"], 0.2)  # anything up to 83c leaves the 90c ones out
        self.assertLessEqual(1 / (1 + st["recommended"]), 0.9)
        self.assertEqual(rrlab.study(self.trades(10, 0.8))["verdict"], "waiting")

    def test_a_filter_that_only_looked_good_on_the_old_trades_is_not_taken(self):
        from polybot import rrlab
        old = self.trades(84, 0.80)  # dear entries lose on the older 70%...
        new = [(0.90, 10 / 0.90 - 10, 10)] * 36  # ...and all win on the newer 30%
        st = rrlab.study(old + new)
        self.assertNotEqual(st["verdict"], "change")
        self.assertEqual(st["recommended"], 0.0)

    def test_the_simulation_takes_it_real_money_waits_the_owner_decides(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.refresh_lab = lambda: None
            engine.set_rr_auto("sim", True)  # (off by default since the panels: the owner's to turn on)
            engine.set_lab_exact("sim", False)  # ("exactly like the lab" runs the rule without a filter)
            v = engine.base_strategies()[0]["variant"]
            for i, (per, pnl, paid) in enumerate(self.trades(120, 0.80)):
                engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, variant, shares, cost, fee, pnl, winner)"
                                " VALUES ('lab', 'filled', ?, 'btc', 15, ?, 'up', ?, ?, ?, 0, ?, 'up')", (i, i * 900, v, paid / per, paid, pnl))
            from polybot.store import connect
            db = connect(cfg.db_path)
            engine._study_rr(db)
            db.close()
            engine._apply_rr_research()
            sim, live = engine.main_strategies()[0], engine.live_main()[0]
            self.assertGreater(sim["ratio"], 0)
            self.assertEqual(sim["ratio_src"], "system")
            self.assertFalse(live.get("ratio"))  # real money: a recommendation until the owner applies it
            self.assertEqual(engine.trackers[0].main_rules[0].filter_ratio, sim["ratio"])
            self.assertEqual(engine.trackers[0].main_rules[0].key, v)  # same strategy, same record
            engine.set_strategy_ratio("sim", "btc", 15, v, 0.0)  # the owner's own choice: no filter
            engine._apply_rr_research()
            self.assertEqual((engine.main_strategies()[0]["ratio"], engine.main_strategies()[0]["ratio_src"]), (0.0, "owner"))
            engine.set_money("sim", {"rr": 0.7})  # one filter for the whole simulation, from the strip
            self.assertEqual(engine.trackers[0].main_rules[0].filter_ratio, 0.7)
            engine.set_money("sim", {}, reset=True)
            self.assertEqual(engine.trackers[0].main_rules[0].filter_ratio, 0.0)
            engine.store.close()


class MoveToAnotherMachineTest(unittest.TestCase):
    def test_an_export_carries_on_on_a_new_machine_with_real_money_off(self):
        from polybot import export
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "old.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.refresh_lab = lambda: None
            v = engine.base_strategies()[0]["variant"]
            engine.set_strategy_name("btc", 15, v, "שלי")
            engine.set_money("sim", {"stake": 7, "copies": 2})
            engine.store.set_setting("live_armed", "1")
            engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, variant, shares, cost, fee, pnl, book)"
                            " VALUES ('lab', 'filled', 1, 'btc', 15, 0, 'up', ?, 20, 10, 0.1, 9.9, '[[0.5,20]]')", (v,))
            zip_path = os.path.join(tmp, "export.zip")
            export.write(cfg, zip_path)
            engine.store.close()
            new = Config(assets=("btc",), db_path=os.path.join(tmp, "new.sqlite"))
            Store(new.db_path).close()
            counts = export.read(new, zip_path)
            self.assertEqual(counts["bets"], 1)
            again = Engine(new, api=FakeApi(), env={}, streaming=False, with_lab=False)
            self.assertEqual(again.strategy_names(), {f"btc|15|{v}": "שלי"})
            self.assertEqual((again.money()["sim"]["stake"]["value"], again.trackers[0].sim_copies), (7.0, 2))
            self.assertFalse(again.live.armed())  # real money arrives switched off
            row = again.store._x("SELECT pnl, book FROM bets")[0]
            self.assertAlmostEqual(float(row[0]), 9.9)
            self.assertEqual(json.loads(row[1]), [[0.5, 20]])
            again.store.close()

    def test_a_regional_refusal_switches_real_money_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            live = LiveTrader(cfg, Store(cfg.db_path), env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64})

            class Refusing(FakeClob):
                def place_market_order(self, **order):
                    raise RuntimeError("RequestRejectedError: Trading restricted in your region, please refer to available regions")
            live.client = Refusing()
            live.set_readiness(True)
            live.arm(True)
            fill = types.SimpleNamespace(side="up", fair_prob=0.9, best_ask=0.5, avg_price=0.5, shares=20, cost=10, fee=0,
                                         edge=0.4, limit_price=0.5)
            live.maybe_trade(time.time(), "btc", 15, 0, 300, fill, {"up_token": "u", "down_token": "d", "tick_size": 0.01}, cfg)
            self.assertFalse(live.armed())
            self.assertEqual(live.stopped()["reason"], "region")
            live.store.close()


class RestartKeepsTradingTest(unittest.TestCase):
    """After a short restart the prices recorded just before it measure volatility, so trading goes
    on at once instead of after five minutes of new prices; after a long stop it waits as before."""

    def _tracker(self, store, cfg):
        return Tracker("btc", cfg, store, api=FakeApi())

    def test_short_restart_trades_at_once_and_a_long_one_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            now = time.time()
            for i in range(600):  # ten minutes of prices, a second apart, up to a minute ago
                store.add_tick(ts=now - 660 + i, asset="btc", minutes=15, window_start=0, price=100 + (i % 7) * 0.05,
                               open_price=None, seconds_left=None, vol=None, fair_up=None, up_bid=None, up_ask=None,
                               down_bid=None, down_ask=None, price_age=None, book_age=None)
            t = self._tracker(store, cfg)
            self.assertIsNone(t.volatility())  # no new price yet
            t.on_price(now, 100.1)
            self.assertIsNotNone(t.volatility())  # the recorded prices stand in
            later = self._tracker(store, cfg)
            later.on_price(now + Tracker.RECORDED_GAP + 120, 100.1)  # back after a long stop
            self.assertIsNone(later.volatility())
            store.close()


class RestartAtTheCloseTest(unittest.TestCase):
    def test_a_round_that_closed_during_a_restart_gets_its_closing_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            ws = feeds.window_start(time.time(), 15) - 900  # the round before this one
            store.upsert_window("btc", 15, ws, open_twap=100.0)
            for i in range(900):  # recorded each second; the last minute at 101
                store.add_tick(ts=ws + i, asset="btc", minutes=15, window_start=ws, price=101.0 if i >= 840 else 100.0,
                               open_price=None, seconds_left=None, vol=None, fair_up=None, up_bid=None, up_ask=None,
                               down_bid=None, down_ask=None, price_age=None, book_age=None)
            t = Tracker("btc", cfg, store, api=FakeApi())  # started again after the close
            t.on_price(ws + 905, 102.0)
            close = store._x("SELECT close_twap FROM windows WHERE asset = 'btc' AND minutes = 15 AND window_start = ?", (ws,))
            self.assertAlmostEqual(close[0][0], 101.0, places=1)
            store.close()


class RestartInARoundShowsItsChartTest(unittest.TestCase):
    def test_the_card_of_a_round_joined_midway_has_a_price_to_beat(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            t = next(t for t in engine.trackers if t.key == "btc-5m")
            now = time.time()
            start = feeds.window_start(now, 5)
            for i in range(120):  # recorded before the restart, up to the round's start
                engine.store.add_tick(ts=start - 119 + i, asset="btc", minutes=5, window_start=start - 300,
                                      price=100.0, open_price=None, seconds_left=None, vol=None, fair_up=None,
                                      up_bid=None, up_ask=None, down_bid=None, down_ask=None, price_age=None, book_age=None)
            t.window = start  # joined midway: no opening measured
            fill = types.SimpleNamespace(side="up", fair_prob=None, best_ask=0.2, avg_price=0.2, shares=10, cost=2.0,
                                         fee=0.0, edge=None, depth=None)
            engine.store.add_bet(now, "btc", 5, start, 100, fill, mode="lab", variant="pair85|0.20|60-285", stake=10)
            db = connect(cfg.db_path)
            card = engine._positions(db, now)["lab"][0]
            self.assertTrue(card["open_estimated"])
            self.assertAlmostEqual(card["open_price"], 100.0)
            db.close()
            engine.store.close()


class OneFaultDoesNotStopTheRestTest(unittest.TestCase):
    def test_a_lottery_rule_without_a_ticket_is_no_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            rule = lab.Variant(0.05, 15, 120, "lotto0")
            t = Tracker("btc", cfg, store, api=FakeApi(), main_rule=rule)
            t.window = 900
            t.can_open = lambda *a: True  # the engine's limits are set (that is when it used to fail)
            t._sim_bet(1000, 60, rule, 0.5, 0.5, [(0.50, 1000)], [(0.50, 1000)], 0.1, 0.1)  # nothing at 5c
            self.assertEqual(store._x("SELECT COUNT(*) FROM bets")[0][0], 0)
            store.close()

    def test_a_fault_in_the_simulation_still_lets_real_money_and_the_lab_run(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            t = Tracker("btc", cfg, store, api=FakeApi())
            t.price, t.price_ts, t.window, t.market = 100.0, 1000.0, 900, {"tick_size": 0.01}
            ran = []
            with mock.patch.object(t, "_sim_rules_run", side_effect=RuntimeError("boom")), \
                    mock.patch.object(t, "_live_rules_run", side_effect=lambda *a: ran.append("live")), \
                    mock.patch.object(t, "_run_lab", side_effect=lambda *a: ran.append("lab")):
                with self.assertRaises(RuntimeError):  # still reported
                    t.evaluate(1000.0, ([(0.4, 10)], [(0.5, 10)], 1000.0), ([(0.4, 10)], [(0.5, 10)], 1000.0))
            self.assertEqual(ran, ["live", "lab"])
            store.close()


class LiveSaysWhyNotTest(unittest.TestCase):
    def test_a_signal_under_polymarkets_minimum_is_shown_with_its_reason(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            live = engine.live
            fill = types.SimpleNamespace(side="up", fair_prob=0.8, best_ask=0.6, avg_price=0.6, shares=3, cost=1.8,
                                         fee=0.0, edge=0.2, limit_price=0.6)
            market = {"tick_size": 0.01, "min_order_size": 5, "up_token": "u", "down_token": "d"}
            with mock.patch.object(live, "blocked", return_value=None), mock.patch.object(live, "stake", return_value=2.0):
                for _ in range(3):  # the same signal at three prices in a row: one line
                    self.assertIsNone(live._trade(1000, "btc", 15, 900, 100, fill, market, cfg, None, stake=2.0))
            skips = live.status()["skips"]
            self.assertEqual(len(skips), 1)  # $2 can't buy 5 shares at 60c: skipped, and said why
            self.assertIn("פולימרקט מקבל רק מ-5", skips[0]["reason"])
            self.assertEqual(skips[0]["n"], 3)
            engine.store.close()

    def test_an_order_whose_shares_cost_under_a_dollar_grows_to_one_or_is_skipped(self):
        # Polymarket's dollar is the shares' price alone: "$1.00" with the fee was $0.93 of shares, refused (30.10.2026)
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            live, orders = engine.live, []
            fill = types.SimpleNamespace(side="up", fair_prob=0.8, best_ask=0.5, avg_price=0.5, shares=2, cost=0.98,
                                         fee=0.02, edge=0.2, limit_price=0.5)
            market = {"tick_size": 0.01, "min_order_size": 1, "up_token": "u", "down_token": "d"}

            def place(**kw):
                orders.append(kw["amount"])
                return types.SimpleNamespace(ok=True, order_id="x", status="matched", taking_amount=kw["amount"] / 0.5,
                                             making_amount=kw["amount"])
            with mock.patch.object(live, "blocked", return_value=None), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=place)):
                self.assertEqual(live._trade(1000, "btc", 5, 900, 100, fill, market, cfg, None, "crypto", "k", 1.0, round_stake=5),
                                 "filled")  # room in the round: it grows to a dollar's shares
                self.assertIsNone(live._trade(1001, "btc", 5, 1800, 100, fill, market, cfg, None, "crypto", "k2", 1.0,
                                              round_stake=1.0))  # no room: skipped, never sent
            self.assertEqual(len(orders), 1)
            self.assertGreaterEqual(orders[0], 1.0)
            self.assertIn("רק מ-$1", live.status()["skips"][0]["reason"])
            engine.store.close()

    def test_a_round_holds_at_most_the_cap_by_the_portfolio_tiers_for_all_strategies_together(self):
        # The owner, 30.10.2026: up to $500 → $15 a round, $500-750 → $30, from $750 → $50
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            streaming=False, with_lab=False)
            live = engine.live
            self.assertEqual(engine.money()["live"]["round_cap"]["value"], 15)  # not read yet: the smallest
            for total, cap in ((120, 15), (500, 15), (500.01, 30), (749, 30), (750, 50), (4000, 50)):
                live.wallet = {"total": total}
                self.assertEqual(engine.money()["live"]["round_cap"]["system"], cap, total)
            live.wallet = {"total": 300}
            self.assertIsNone(engine.money()["sim"]["round_cap"]["value"])  # the simulation: no cap
            orders = []

            def place(**kw):
                orders.append(kw["amount"])
                return types.SimpleNamespace(ok=True, order_id="x", status="matched", taking_amount=kw["amount"] / 0.5,
                                             making_amount=kw["amount"])
            fill = types.SimpleNamespace(side="down", fair_prob=0.8, best_ask=0.5, avg_price=0.5, shares=20, cost=10,
                                         fee=0.0, edge=0.3, limit_price=0.5)
            market = {"tick_size": 0.01, "min_order_size": 5, "up_token": "u", "down_token": "d"}
            cfg = dataclasses.replace(engine.cfg, fee_rate=0.0)
            with mock.patch.object(live, "blocked", return_value=None), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=place)):
                self.assertEqual(live._trade(1000, "btc", 5, 900, 100, fill, market, cfg, None, "crypto", "a", 10, round_stake=10), "filled")
                self.assertIsNone(live._trade(1001, "btc", 5, 900, 99, fill, market, cfg, None, "crypto", "b", 10, round_stake=10))  # $20 > $15
                self.assertIn("תקרה לסבב", live.status()["skips"][-1]["reason"])
                self.assertEqual(live._trade(1002, "btc", 5, 1200, 99, fill, market, cfg, None, "crypto", "b", 10, round_stake=10), "filled")
                walked = Fill(side="down", fair_prob=0.8, best_ask=0.5, avg_price=0.5, shares=20, cost=10, fee=0.0, edge=0.3,
                              limit_price=0.5)  # the owner's own buy in the same round: their decision, not capped
                self.assertEqual(live._trade(1003, "btc", 5, 900, 98, walked, market, cfg, None, "crypto", "manual", 10,
                                             round_stake=10), "filled")
                engine.set_money("live", {"round_cap": 0})  # no cap
                self.assertEqual(live._trade(1004, "btc", 5, 900, 97, fill, market, cfg, None, "crypto", "c", 10, round_stake=10), "filled")
            self.assertEqual(len(orders), 4)
            with self.assertRaises(ValueError):
                engine.set_money("sim", {"round_cap": 20})
            engine.store.close()

    def test_a_split_entry_under_the_minimum_grows_to_it_within_the_trade_amount(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dataclasses.replace(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), fee_rate=0.0)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            live = engine.live
            fill = types.SimpleNamespace(side="up", fair_prob=0.8, best_ask=0.6, avg_price=0.6, shares=3, cost=1.8,
                                         fee=0.0, edge=0.2, limit_price=0.6)
            market = {"tick_size": 0.01, "min_order_size": 5, "up_token": "u", "down_token": "d"}
            orders = []

            def place(**kw):
                orders.append(kw["amount"])
                return types.SimpleNamespace(ok=True, order_id="x", status="matched", taking_amount=kw["amount"] / 0.6,
                                             making_amount=kw["amount"])
            client = types.SimpleNamespace(place_market_order=place)
            with mock.patch.object(live, "blocked", return_value=None), mock.patch.object(live, "stake", return_value=5.0), \
                    mock.patch.object(live, "_client", return_value=client):
                # a third of $5 ($1.67) buys 2.8 shares at 60c: the entry grows to 5.05 shares ($3.03)
                self.assertEqual(live._trade(1000, "btc", 5, 900, 100, fill, market, cfg, None, "crypto", "k", 5 / 3), "filled")
                # the round has $1.97 left of its $5: the next entry can't grow to 5 shares, and is skipped
                self.assertIsNone(live._trade(1001, "btc", 5, 900, 99, fill, market, cfg, None, "crypto", "k", 5 / 3))
            self.assertEqual(orders, [3.03])
            self.assertIn("יותר מהסכום לטרייד", live.status()["skips"][0]["reason"])
            engine.store.close()


class AddToATradeTest(unittest.TestCase):
    """The owner's own entry from a running round's card: the same in the simulation and with real
    money, bought from the book now, and kept under its own strategy ("manual")."""

    def _engine(self, tmp):
        cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
        api = FakeApi()
        api.books["U"] = ([(0.84, 100)], [(0.85, 100)])
        engine = Engine(cfg, api=api, env={}, streaming=False, with_lab=False)
        t = next(t for t in engine.trackers if t.minutes == 15)
        now = time.time()
        t.window = int(now // 900 * 900) if now % 900 < 840 else int(now // 900 * 900) + 900
        t.market = api.find_market("btc-updown-15m")
        t.snapshot = dict(t.snapshot, fair_up=0.9, fair_raw=0.88)
        return engine, t

    def test_the_simulation_buys_at_the_price_now_under_its_own_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            done = engine.manual_add("sim", "btc", 15, t.window, "up", 10)
            self.assertEqual(done["status"], "filled")
            self.assertAlmostEqual(done["avg_price"], 0.85)
            self.assertAlmostEqual(done["paid"], 10, places=2)
            row = engine.store._x("SELECT mode, variant, side, fair_prob FROM bets")[0]
            self.assertEqual(tuple(row), ("paper", "manual", "up", 0.9))
            self.assertIn("btc|15|manual", engine.state()["strategy_results"]["paper"])
            with self.assertRaises(ValueError):  # a round that already closed
                engine.manual_add("sim", "btc", 15, t.window - 900, "up", 10)
            engine.store.close()

    def test_the_offer_lists_amounts_the_book_takes_whole_with_the_round_either_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            engine.api.books["D"] = ([], [])  # nobody selling Down right now
            q = engine.manual_quote("sim", "btc", 15, t.window)
            up = q["sides"]["up"]["options"]
            self.assertTrue(1 <= len(up) <= engine.MANUAL_OPTIONS)
            self.assertEqual([o["amount"] for o in up], sorted(o["amount"] for o in up))
            self.assertTrue(any(o["best"] for o in up))  # all that sits at the best price
            self.assertTrue(all(o["shares"] >= 5 for o in up))
            self.assertEqual(q["sides"]["down"]["options"], [])
            first = up[0]
            self.assertAlmostEqual(first["if_up"], first["shares"] - first["amount"])
            self.assertAlmostEqual(first["if_down"], -first["amount"])
            engine.manual_add("sim", "btc", 15, t.window, "up", 10)
            q = engine.manual_quote("sim", "btc", 15, t.window)  # what the round holds counts in
            self.assertAlmostEqual(q["now_if_down"], -10, places=2)
            o = q["sides"]["up"]["options"][0]
            self.assertAlmostEqual(o["if_down"], -10 - o["amount"], places=2)
            engine.store.close()

    def test_a_buy_that_locks_both_sides_is_offered_and_bought_only_while_it_still_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            engine.api.books["D"] = ([], [(0.30, 1000)])
            self.assertEqual(engine.manual_quote("sim", "btc", 15, t.window)["locks"], [])  # nothing held: no lock
            engine.manual_add("sim", "btc", 15, t.window, "up", 10)  # 11.76 Up for $10
            engine.api.books["D"] = ([], [(0.05, 1000)])  # Down went cheap: a little of it wins either way
            q = engine.manual_quote("sim", "btc", 15, t.window)
            self.assertAlmostEqual(q["paid_side"]["up"], 10, places=2)
            self.assertTrue(q["locks"])
            for o in q["locks"]:
                self.assertEqual(o["side"], "down")
                self.assertGreater(o["if_up"], 0)
                self.assertGreater(o["if_down"], 0)
            self.assertEqual(q["locks"][0]["sure"], max(o["sure"] for o in q["locks"]))
            with self.assertRaises(ValueError):  # too much Down: if Up wins it loses, so nothing is bought
                engine.manual_add("sim", "btc", 15, t.window, "down", 5, lock=True)
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets")[0][0], 1)
            best = q["locks"][0]
            engine.manual_add("sim", "btc", 15, t.window, "down", best["amount"], lock=True)
            q = engine.manual_quote("sim", "btc", 15, t.window)
            self.assertGreater(q["now_if_up"], 0)
            self.assertGreater(q["now_if_down"], 0)
            self.assertTrue(all(o["if_up"] > 0 and o["if_down"] > 0 for o in q["locks"]))  # what still improves it
            engine.store.close()

    def test_a_locked_round_is_offered_only_what_betters_it_and_a_pair_buys_in_one_click(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)  # Up offered at 85c, and the model says Up (90%) leads
            engine.manual_add("sim", "btc", 15, t.window, "up", 10)
            engine.api.books["D"] = ([], [(0.05, 1000)])
            engine.manual_add("sim", "btc", 15, t.window, "down", 1.5, lock=True)
            engine.api.books["D"] = ([], [(0.16, 1000)])  # Down back up: 85c + 16c is more than the $1 a pair pays
            q = engine.manual_quote("sim", "btc", 15, t.window)
            self.assertGreater(min(q["now_if_up"], q["now_if_down"]), 0)  # locked
            leads = [o for o in q["locks"] if o["kind"] == "lead"]
            self.assertTrue(leads)
            for o in q["locks"]:  # never a buy that only raises the side behind
                self.assertGreater(o["if_up"], 0)
                self.assertGreater(o["if_down"], 0)
                self.assertGreater(o["if_up"], q["now_if_up"])  # Up leads: it always wins more
            self.assertFalse([o for o in q["locks"] if o["kind"] == "pair"])
            engine.api.books["U"] = ([], [(0.50, 100)])
            engine.api.books["D"] = ([], [(0.05, 1000)])  # Up at 50c and Down at 5c: a pair pays $1 for ~57c
            q = engine.manual_quote("sim", "btc", 15, t.window)
            pair = next(o for o in q["locks"] if o["kind"] == "pair")
            self.assertGreater(pair["if_up"], q["now_if_up"])
            self.assertGreater(pair["if_down"], q["now_if_down"])
            done = engine.manual_pair("sim", "btc", 15, t.window, pair["up_amount"], pair["down_amount"], pair["first"])
            self.assertEqual(done["status"], "filled")
            after = engine.manual_quote("sim", "btc", 15, t.window)
            self.assertAlmostEqual(after["now_if_up"], pair["if_up"], delta=0.2)
            self.assertAlmostEqual(after["now_if_down"], pair["if_down"], delta=0.2)
            engine.store.close()

    def test_a_buy_ending_on_a_crumb_of_a_far_dearer_level_is_not_offered_and_the_owners_buy_is_sized_by_the_book(self):
        # 30.10.2026: $1 of Down was 92 shares at 1c and a cent's worth at 99c; real money read it as "1 share at 99c"
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            engine.manual_add("sim", "btc", 15, t.window, "up", 10)
            engine.api.books["D"] = ([], [(0.01, 92), (0.99, 100)])
            q = engine.manual_quote("sim", "btc", 15, t.window)
            self.assertEqual(q["locks"], [])  # every lock of a dollar's shares needs the 99c crumb
            with self.assertRaises(ValueError):
                engine.manual_add("sim", "btc", 15, t.window, "down", 1.0, lock=True)
            live, orders = engine.live, []

            def place(**kw):
                orders.append((kw["amount"], kw["max_price"]))
                return types.SimpleNamespace(ok=True, order_id="x", status="matched", taking_amount=150,
                                             making_amount=kw["amount"])
            fill = Fill(side="down", fair_prob=0.1, best_ask=0.01, avg_price=0.0107, shares=150, cost=1.5, fee=0.1,
                        edge=0.0, limit_price=0.02)
            market = {"tick_size": 0.01, "min_order_size": 5, "up_token": "u", "down_token": "d"}
            with mock.patch.object(live, "blocked", return_value=None), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=place)):
                self.assertEqual(live._trade(1000, "btc", 15, t.window, 100, fill, market, t.market_cfg, None, "crypto",
                                             "manual", 1.6, round_stake=1.6), "filled")
            self.assertEqual(orders, [(1.5, 0.02)])  # the dollars of the walk, capped at its last level
            engine.store.close()

    def test_real_money_sends_the_order_even_outside_its_risk_reward_range(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            live, orders = engine.live, []

            def place(**kw):
                orders.append((kw["amount"], kw["max_price"]))
                return types.SimpleNamespace(ok=True, order_id="x", status="matched", taking_amount=kw["amount"] / 0.85,
                                             making_amount=kw["amount"])
            live.ratios = {"crypto": (1.0, 0.0)}  # real money's filter: wins must pay at least what's risked
            with mock.patch.object(live, "trades", return_value=True), mock.patch.object(live, "blocked", return_value=None), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=place)):
                done = engine.manual_add("live", "btc", 15, t.window, "up", 10)
            self.assertEqual(done["status"], "filled")
            self.assertEqual(orders[0][1], 0.85)
            self.assertEqual(engine.store._x("SELECT mode, variant, status FROM bets")[0][:], ("live", "manual", "filled"))
            with mock.patch.object(live, "trades", return_value=False):
                with self.assertRaises(ValueError):  # real money off: nothing is sent
                    engine.manual_add("live", "btc", 15, t.window, "up", 10)
            engine.store.close()


class CloseATradeTest(unittest.TestCase):
    """The owner closing a running round early: what it holds sold to the bids now, after the fee."""
    _engine = AddToATradeTest._engine

    def test_the_simulation_sells_at_the_bids_and_the_trades_keep_the_sale(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)  # Up offered at 85c
            engine.manual_add("sim", "btc", 15, t.window, "up", 10)
            engine.api.books["U"] = ([(0.70, 100)], [(0.72, 100)])  # it went against us: buyers pay 70c now
            q = engine.close_quote("sim", "btc", 15, t.window)
            shares = q["sides"]["up"]["shares"]
            fee = shares * 0.07 * 0.70 * 0.30
            self.assertAlmostEqual(q["value"], shares * 0.70 - fee, places=6)
            self.assertAlmostEqual(q["pnl_now"], q["value"] - q["paid"], places=6)
            self.assertAlmostEqual(q["if_down"], -q["paid"])  # waiting, and it loses: all of it
            self.assertTrue(q["open"])
            done = engine.close_round("sim", "btc", 15, t.window)
            self.assertAlmostEqual(done["pnl"], q["pnl_now"], places=6)
            row = engine.store._x("SELECT winner, pnl FROM bets")[0]
            self.assertEqual(row[0], "sold")
            self.assertAlmostEqual(row[1], q["pnl_now"], places=6)
            self.assertEqual(engine.store.unsettled_bets("btc", 15, t.window), [])  # the result won't settle it again
            with self.assertRaises(ValueError):
                engine.close_round("sim", "btc", 15, t.window)  # nothing left to close
            engine.store.close()

    def test_real_money_sends_a_sell_order_and_a_thin_book_stops_it(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            fill = types.SimpleNamespace(side="up", fair_prob=0.8, best_ask=0.85, avg_price=0.85, shares=10, cost=8.5, fee=0.1,
                                         edge=0.0, depth=None)
            engine.store.add_bet(time.time(), "btc", 15, t.window, 300, fill, mode="live", variant="manual")
            engine.api.books["U"] = ([(0.90, 4)], [(0.92, 100)])  # buyers for only 4 of the 10 shares
            self.assertFalse(engine.close_quote("live", "btc", 15, t.window)["open"])
            with self.assertRaises(ValueError):
                engine.close_round("live", "btc", 15, t.window)
            engine.api.books["U"] = ([(0.90, 100)], [(0.92, 100)])
            with mock.patch.object(engine.live, "sell", return_value=("filled", 8.95, 10.0, None)) as sell:
                done = engine.close_round("live", "btc", 15, t.window)
            self.assertEqual(sell.call_args[0][1:], (10.0, 0.9))  # all 10 shares, no lower than 90c
            self.assertAlmostEqual(done["pnl"], 8.95 - 8.6)
            engine.store.close()


class ApprovalModeTest(unittest.TestCase):
    """Real money on in the owner's approval mode: nothing buys by itself; the owner's click buys the
    simulation's round on the same terms, or nothing."""
    _engine = AddToATradeTest._engine

    def _arm(self, engine):
        from unittest import mock
        with mock.patch.object(engine.live, "configured", return_value=True):
            engine.set_mode(True, confirmed=True, approval=True)

    @staticmethod
    def _place(orders):
        def place(**kw):
            orders.append(kw)
            return types.SimpleNamespace(ok=True, order_id="x", status="matched", taking_amount=kw["amount"] / kw["max_price"],
                                         making_amount=kw["amount"])
        return place

    def test_nothing_buys_by_itself_and_the_owners_click_does_without_the_guards(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            live, orders = engine.live, []
            engine.manual_add("sim", "btc", 15, t.window, "up", 10)  # the simulation: Up at 85c
            self._arm(engine)
            self.assertTrue(live.approval())
            self.assertEqual(engine.mode(), "paper")  # the simulation's view (and its mails)
            fill = types.SimpleNamespace(side="up", fair_prob=0.9, best_ask=0.85, avg_price=0.85, shares=10, cost=8.5,
                                         fee=0.1, edge=0.05, limit_price=0.85)
            with mock.patch.object(live, "trades", return_value=True), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=self._place(orders))):
                self.assertIsNone(live.maybe_trade(time.time(), "btc", 15, t.window, 300, fill, t.market))  # a rule alone: no
                live.store.set_setting("live_daily_loss_limit", "1")  # (a guard that would stop an automatic order)
                with self.assertRaises(ValueError):  # not the simulation's terms (85c): nothing
                    engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "up", 0.80)
                with self.assertRaises(ValueError):  # the simulation holds Up: Down is another trade
                    engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "down", 0.85)
                done = engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "up", 0.85)
                with self.assertRaises(ValueError):  # once a round
                    engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "up", 0.85)
            self.assertEqual(done["status"], "filled")
            self.assertEqual(len(orders), 1)
            self.assertLessEqual(orders[0]["max_price"], 0.87)
            row = engine.store._x("SELECT mode, variant, side FROM bets WHERE mode = 'live'")[0]
            self.assertEqual(tuple(row), ("live", "twap|0.05|15-600", "up"))
            engine.set_mode(False, confirmed=True)
            self.assertFalse(live.approval())
            engine.store.close()

    def test_a_price_far_off_either_way_is_another_trade(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            live, orders = engine.live, []
            engine.manual_add("sim", "btc", 15, t.window, "up", 10)  # Up at 85c
            self._arm(engine)
            key = f"btc|15|{t.window}"
            item = {"asset": "btc", "minutes": 15, "window_start": t.window}
            self.assertFalse(engine.approve_check([item])[key]["ok"])  # real money is off
            with mock.patch.object(live, "trades", return_value=True), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=self._place(orders))):
                ok = engine.approve_check([item])[key]
                self.assertTrue(ok["ok"])
                self.assertEqual(ok["terms"], [{"side": "up", "price": 0.85, "weight": ok["terms"][0]["weight"]}])
                t.api.books["U"] = ([(0.88, 100)], [(0.90, 100)])  # dearer than 85c + 2c
                self.assertIn("המחיר ברח", engine.approve_check([item])[key]["why"])
                # Much cheaper (the market went against it, as Up at 2c instead of 19c): not a bargain, another trade.
                t.api.books["U"] = ([(0.01, 100)], [(0.02, 5000)])
                cheap = engine.approve_check([item])[key]
                self.assertFalse(cheap["ok"])
                self.assertIn("טרייד אחר", cheap["why"])
                with self.assertRaises(ValueError):
                    engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "up", 0.85)
                self.assertEqual(orders, [])
                gone = engine.approve_check([dict(item, window_start=t.window - 900)])
                self.assertFalse(gone[f"btc|15|{t.window - 900}"]["ok"])  # a round that closed
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live'")[0][0], 0)
            engine.store.close()

    def _both(self, tmp):
        """The simulation holds a round on both sides: Up at 85c, then Down at 2c."""
        engine, t = self._engine(tmp)
        engine.books = None
        t.api.books["D"] = ([(0.01, 100)], [(0.02, 5000)])
        engine.manual_add("sim", "btc", 15, t.window, "up", 60)
        engine.manual_add("sim", "btc", 15, t.window, "down", 5)
        self._arm(engine)
        legs = [{"side": "up", "price": 0.85, "weight": 60}, {"side": "down", "price": 0.02, "weight": 5}]
        return engine, t, legs

    def test_a_round_held_on_both_sides_is_bought_on_both_or_not_at_all(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t, legs = self._both(tmp)
            live, orders = engine.live, []
            item = {"asset": "btc", "minutes": 15, "window_start": t.window}
            with mock.patch.object(live, "trades", return_value=True), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=self._place(orders))):
                check = engine.approve_check([item])[f"btc|15|{t.window}"]
                self.assertTrue(check["ok"])
                amounts = {x["side"]: x["amount"] for x in check["legs"]}
                self.assertAlmostEqual(amounts["up"] / amounts["down"], 12, delta=0.6)  # the simulation's split
                # The $5 of Down at real money's amount would be under Polymarket's $1: the copy grows, same split.
                self.assertTrue(check["grown"])
                self.assertGreaterEqual(amounts["down"], 1.0)
                self.assertGreater(check["if"]["up"], 0)
                self.assertGreater(check["if"]["down"], 0)  # held either way, as in the simulation
                # The small side a little cheaper now (1c for 2c): the same trade, a hair better.
                t.api.books["D"] = ([(0.005, 100)], [(0.01, 9000)])
                self.assertTrue(engine.approve_check([item])[f"btc|15|{t.window}"]["ok"])
                with self.assertRaises(ValueError):  # one side only: not this round's trade
                    engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "up", 0.85)
                t.api.books["D"] = ([(0.97, 100)], [(0.98, 5000)])  # Down ran: no side of it is bought
                self.assertFalse(engine.approve_check([item])[f"btc|15|{t.window}"]["ok"])
                with self.assertRaises(ValueError):
                    engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", legs=legs)
                self.assertEqual(orders, [])
                t.api.books["D"] = ([(0.01, 100)], [(0.02, 5000)])
                done = engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", legs=legs)
            self.assertEqual(sorted(x["side"] for x in done["legs"]), ["down", "up"])
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live'")[0][0], 2)
            engine.store.close()

    def test_if_the_second_side_fails_the_first_is_sold_back(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t, legs = self._both(tmp)
            live, orders, sold = engine.live, [], []

            def place(**kw):
                orders.append(kw)
                if len(orders) == 2:  # the second side: the book moved in between
                    return types.SimpleNamespace(ok=False, status=None, code="400", message="not filled")
                return types.SimpleNamespace(ok=True, order_id="x", status="matched", taking_amount=kw["amount"] / kw["max_price"],
                                             making_amount=kw["amount"])
            with mock.patch.object(live, "trades", return_value=True), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=place)), \
                    mock.patch.object(live, "sell", side_effect=lambda token, shares, low: sold.append(token) or ("filled", shares * low, shares, None)):
                with self.assertRaises(ValueError) as e:
                    engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", legs=legs)
            self.assertIn("נמכרה בחזרה", str(e.exception))
            self.assertEqual(len(sold), 1)
            open_live = engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live' AND status = 'filled' AND winner IS NULL")[0][0]
            self.assertEqual(open_live, 0)  # nothing left of a trade that wasn't the one approved
            engine.store.close()


class TakeOverARoundTest(unittest.TestCase):
    """The owner takes over a simulation round: every move on its card is made with real money too, at
    the same moment, or not at all, so real money holds exactly the trade the owner built."""
    _engine = AddToATradeTest._engine
    _arm = ApprovalModeTest._arm
    _place = staticmethod(ApprovalModeTest._place)

    def test_every_addition_goes_to_both_or_neither(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            engine.books = None
            live, orders = engine.live, []
            engine.manual_add("sim", "btc", 15, t.window, "up", 50)  # what the round held before
            self._arm(engine)
            with mock.patch.object(live, "trades", return_value=True), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=self._place(orders))):
                with self.assertRaises(ValueError):  # only the offered sizes
                    engine.take_over("btc", 15, t.window, 0.33)
                done = engine.take_over("btc", 15, t.window, 0.1, copy=True)
                self.assertAlmostEqual(done["copied"]["paid"], 5, delta=0.2)  # the round's $50, at 10%
                self.assertIn(f"btc|15|{t.window}", engine.state()["approval_mirror"])
                q = engine.manual_quote("sim", "btc", 15, t.window)
                self.assertEqual(q["mirror"]["ratio"], 0.1)
                self.assertTrue(all("real" in o for x in q["sides"].values() for o in x["options"]))
                # An addition: real money first, at 10%, the same moment and book.
                t.api.books["D"] = ([(0.10, 1000)], [(0.12, 1000)])
                engine.manual_add("sim", "btc", 15, t.window, "down", 30)
                self.assertAlmostEqual(orders[-1]["amount"], 3, delta=0.25)  # (the dollars sent; the fee on top makes $3)
                # Too small for Polymarket at 10% ($0.50): neither is bought.
                n = engine.store._x("SELECT COUNT(*) FROM bets")[0][0]
                with self.assertRaises(ValueError):
                    engine.manual_add("sim", "btc", 15, t.window, "down", 5)
                self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets")[0][0], n)
            # Real money refuses: the simulation doesn't buy either.
            with mock.patch.object(live, "trades", return_value=True), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=lambda **kw:
                        types.SimpleNamespace(ok=False, status=None, code="400", message="fok_not_filled"))):
                with self.assertRaises(ValueError) as e:
                    engine.manual_add("sim", "btc", 15, t.window, "down", 30)
            self.assertIn("גם בסימולציה לא", str(e.exception))
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets WHERE status = 'filled'")[0][0], n)
            real = engine.store._x("SELECT side, ROUND(SUM(cost + fee), 1) FROM bets WHERE mode = 'live' AND status = 'filled' GROUP BY side")
            sim = engine.store._x("SELECT side, ROUND(SUM(cost + fee), 1) FROM bets WHERE mode = 'paper' GROUP BY side")
            for (s1, r), (s2, p) in zip(real, sim):
                self.assertEqual(s1, s2)
                self.assertAlmostEqual(r, p * 0.1, delta=0.3)  # the same trade, at 10%
            engine.store.close()

    def test_taking_over_from_now_and_closing_both(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            engine.books = None
            live, orders, sold = engine.live, [], []
            engine.manual_add("sim", "btc", 15, t.window, "up", 50)
            self._arm(engine)
            with mock.patch.object(live, "trades", return_value=True), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=self._place(orders))), \
                    mock.patch.object(live, "sell", side_effect=lambda token, shares, low: sold.append(token) or ("filled", shares * low, shares, None)):
                engine.take_over("btc", 15, t.window, 0.5, copy=False)  # what the strategies bought stays in the simulation
                self.assertEqual(orders, [])
                engine.manual_add("sim", "btc", 15, t.window, "up", 10)
                self.assertEqual(len(orders), 1)
                engine.close_round("sim", "btc", 15, t.window)  # closing the card closes real money's first
            self.assertEqual(sold, ["U"])
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets WHERE winner IS NULL AND status = 'filled'")[0][0], 0)
            engine.store.close()


class CandidatesGoToTheSimulationTest(unittest.TestCase):
    def _row(self, variant, pess, family="main", asset="btc"):
        return {"asset": asset, "minutes": 15, "variant": variant, "rule": lab.describe(variant), "status": "candidate",
                "roi": 0.2, "roi_pessimistic": pess, "bets": 120, "family": family, "new_candidate": True}

    def test_a_new_candidate_goes_in_and_an_email_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), max_rules_per_market=3)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            self.assertFalse(engine.auto_candidates())  # (off by default since the panels: the owner's to turn on)
            engine.set_auto_candidates(True)
            good, worse, pair = (self._row("twap|0.08|15-180", 0.02), self._row("twap|0.12|15-180", 0.01),
                                 self._row("pair85|0.20|120-885", 0.5, family="pair"))
            engine.lab = {"board": [good, worse, pair]}
            engine._on_candidates([good, worse, pair])
            sim = {(m["asset"], m["variant"]) for m in engine.strategies_for("sim")}
            self.assertIn(("btc", good["variant"]), sim)  # the best new one of the market
            self.assertNotIn(("btc", worse["variant"]), sim)
            self.assertNotIn(("btc", pair["variant"]), sim)  # the lab only
            pending = json.loads(engine.store.get_setting("notify_cand_pending"))
            self.assertEqual([p["rule"] for p in pending], [good["rule"]])
            engine.set_auto_candidates(False)
            later = self._row("twap|0.03|15-600", 0.05)
            engine.lab = {"board": [later]}
            engine._on_candidates([later])
            self.assertNotIn(("btc", later["variant"]), {(m["asset"], m["variant"]) for m in engine.strategies_for("sim")})
            engine.store.close()


class StakePerStrategyTest(unittest.TestCase):
    def test_the_research_sets_each_strategys_amount_under_the_modes_and_the_owner_can_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            for mode in ("sim", "live"):  # (each strategy's own amount: the one amount for all is off)
                engine.store.set_setting(f"uniform_stake_{mode}", "0")
            m = engine.strategies_for("sim")[0]
            key = (m["asset"], m["minutes"], m["variant"])
            engine.set_money("sim", {"stake": 20})  # the owner's amount: the ceiling for the system's
            engine.stake_research = {"sim": {key: {"best": 50, "trades": 100}}, "live": {}}
            engine._apply_stake_research()
            entry = next(x for x in engine.strategies_for("sim") if (x["asset"], x["minutes"], x["variant"]) == key)
            self.assertEqual((entry["stake"], entry["stake_src"]), (50, "system"))
            self.assertEqual(engine._strategy_stake("sim", entry), 20)  # never over the mode's amount
            rule = engine._rules_by_market(engine.strategies_for("sim"), "sim")[key[:2]][0]
            self.assertEqual(rule.stake, 20)
            engine.stake_research["sim"][key] = {"best": 10, "trades": 110}
            engine._apply_stake_research()  # only 10 new trades: no change yet
            entry = next(x for x in engine.strategies_for("sim") if (x["asset"], x["minutes"], x["variant"]) == key)
            self.assertEqual(entry["stake"], 50)
            engine.set_strategy_stake("sim", *key, 35)  # the owner's, used as set
            entry = next(x for x in engine.strategies_for("sim") if (x["asset"], x["minutes"], x["variant"]) == key)
            self.assertEqual(engine._strategy_stake("sim", entry), 35)
            engine.set_strategy_stake("sim", *key, None)  # back to the system's
            entry = next(x for x in engine.strategies_for("sim") if (x["asset"], x["minutes"], x["variant"]) == key)
            self.assertEqual((entry["stake_src"], engine._strategy_stake("sim", entry)), ("system", 10))
            engine.store.close()


class LiveSaysWhenItsSettingsHoldItBackTest(unittest.TestCase):
    def test_the_market_mix_that_wipes_out_the_edge_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dataclasses.replace(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), fee_rate=0.0)
            store = Store(cfg.db_path)
            notes = []
            live = types.SimpleNamespace(trades=lambda s: True, approval=lambda: False, order_cfg=lambda c: dataclasses.replace(c, stake_usd=5.0),
                                         blocked=lambda *a, **k: None, note_skip=lambda *a: notes.append(a))
            t = Tracker("btc", cfg, store, api=FakeApi(), live=live)
            t.window = 900
            rule = lab.Variant(0.05, 15, 600, "twap")
            # mixed with the market: 52% (no edge at 50c); the lab's own view: 70% (an edge)
            t._live_bet(1000, 300, rule, 0.52, [(0.50, 1000)], [(0.50, 1000)], plain=0.70)
            self.assertEqual(len(notes), 1)
            self.assertIn("השקלול עם מחיר השוק", notes[0][3])
            notes.clear()
            t._live_bet(1002, 298, rule, 0.52, [(0.50, 1000)], [(0.50, 1000)], plain=0.52)  # no edge either way
            self.assertEqual(notes, [])
            # It runs on every book update: the reasons are looked at once a second, not each time
            # (the check whether real money is on reads the database).
            calls = []
            live.trades = lambda s: calls.append(1) or True
            for i in range(500):
                t._live_bet(1010 + i / 1000, 290, rule, 0.52, [(0.50, 1000)], [(0.50, 1000)], plain=0.70)
            self.assertEqual(len(calls), 1)
            store.close()


class StartsWithAmountsSetTest(unittest.TestCase):
    def test_the_engine_starts_when_strategies_have_the_systems_amounts(self):
        # 1.71-1.75 crashed on every start once the research had set an amount: the amount's
        # ceiling read the money strip before the engine had what it needs for it.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            for mode in ("sim", "live"):  # (each strategy's own amount: the one amount for all is off)
                engine.store.set_setting(f"uniform_stake_{mode}", "0")
            m = engine.strategies_for("sim")[0]
            key = (m["asset"], m["minutes"], m["variant"])
            engine.stake_research = {"sim": {key: {"best": 30, "trades": 100}}, "live": {key: {"best": 7, "trades": 100}}}
            if not any((x["asset"], x["minutes"], x["variant"]) == key for x in engine.strategies_for("live")):
                engine.put_back_strategy(*key, mode="live")
            engine._apply_stake_research()
            engine.store.close()
            again = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)  # a restart
            rule = again._rules_by_market(again.strategies_for("live"), "live")[key[:2]][0]
            self.assertEqual(rule.stake, 5)  # the research said $7; real money's amount ($5) is the ceiling
            again.store.close()


class MaxRulesTest(unittest.TestCase):
    def test_a_full_market_takes_out_its_weakest_unless_the_owner_raises_the_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            first = engine.strategies_for("sim")[0]
            row = lambda v: {"asset": first["asset"], "minutes": first["minutes"], "variant": v, "status": "candidate",
                             "rule": lab.describe(v), "roi_pessimistic": 0.01}
            engine.lab = {"board": [row("twap|0.08|15-180"), row("twap|0.12|15-180")]}
            engine.set_max_rules(1)
            engine.push([(first["asset"], first["minutes"], "twap|0.08|15-180")], allow_unproven=True)
            on = lambda: sorted(m["variant"] for m in engine.strategies_for("sim") if m["asset"] == first["asset"])
            self.assertEqual(on(), ["twap|0.08|15-180"])  # full: the other one made room
            engine.set_max_rules(2)
            engine.lab = {"board": [row("twap|0.08|15-180"), row("twap|0.12|15-180")]}  # (the push recomputed it)
            engine.push([(first["asset"], first["minutes"], "twap|0.12|15-180")], allow_unproven=True)
            self.assertEqual(on(), ["twap|0.08|15-180", "twap|0.12|15-180"])  # room for both
            engine.store.close()


class CountAfreshTest(unittest.TestCase):
    def test_starting_over_resets_only_the_pages_money_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            now = time.time()
            for mode, ts in (("lab", now - 3600), ("lab", now + 60), ("paper", now - 3600)):
                engine.store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares,"
                                " cost, fee, pnl, winner) VALUES (?, 'twap|0.05|15-600', 'filled', ?, 'btc', 15, ?, 'up', 10,"
                                " 5, 0, 5, 'up')", (mode, ts, int(ts)))
            engine.store.set_setting("lab_candidates", json.dumps({"btc|15|twap|0.05|15-600": now - 100}))
            engine.set_count_since("lab", True)
            engine.set_count_since("sim", True)
            db = connect(cfg.db_path)
            board = lab.leaderboard(db, cfg, since=engine.lab_since())
            self.assertEqual(board["settled_bets"], 2)  # the strategy keeps every trade
            self.assertEqual(board["board"][0]["bets"], 2)
            self.assertEqual(board["families"]["main"]["pnl"], 5)  # the money: only since starting over
            self.assertEqual(board["families"]["main"]["per_round"]["rounds"], 1)
            self.assertIn("btc|15|twap|0.05|15-600", engine._json_setting("lab_candidates", {}))  # its stage stays
            state = engine.state()
            self.assertEqual(state["paper"]["summary"]["settled"], 0)  # the simulation's money: from now
            self.assertEqual(state["paper"]["summary_all"]["settled"], 1)  # its count: every trade
            self.assertEqual(state["strategy_results"]["paper"]["btc|15|twap|0.05|15-600"]["bets"], 1)
            engine.set_count_since("lab", False)
            self.assertNotIn("lab", engine.count_since())
            db.close()
            engine.store.close()

    def test_stages_set_aside_by_an_older_start_over_come_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            store.set_setting("lab_candidates", json.dumps({"btc|15|a": 5}))
            store.set_setting("lab_candidates_before_pair", json.dumps({"btc|15|pair85|0.20|120-885": 7}))
            store.close()
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            self.assertEqual(engine._json_setting("lab_candidates", {}), {"btc|15|a": 5, "btc|15|pair85|0.20|120-885": 7})
            self.assertIsNone(engine.store.get_setting("lab_candidates_before_pair"))
            engine.store.close()


class StartOverAPageAtATimeTest(unittest.TestCase):
    def test_starting_over_on_one_lab_page_leaves_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            now = time.time()
            main, end = "twap|0.05|15-600", "end|0.00|45-60|b90-95"
            for v, ts in ((main, now - 3600), (end, now - 3600), (end, now + 60)):
                engine.store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares,"
                                " cost, fee, pnl, winner) VALUES ('lab', ?, 'filled', ?, 'btc', 15, ?, 'up', 10, 5, 0, 5, 'up')",
                                (v, ts, int(ts)))
            engine.set_count_since("end", True)
            db = connect(cfg.db_path)
            fams = lab.leaderboard(db, cfg, since=engine.lab_since())["families"]
            self.assertEqual(fams["main"]["pnl"], 5)  # the crypto page: all its money
            self.assertEqual(fams["end"]["pnl"], 5)  # the end-of-round page: one trade of two, since starting over
            self.assertEqual(fams["end"]["settled_bets"], 2)  # ...but still both trades counted
            self.assertNotIn("lab", engine.count_since())
            db.close()
            engine.store.close()

    def test_an_older_lab_wide_start_carries_to_every_lab_page_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.store.set_setting("count_since", json.dumps({"lab": 123.0}))
            self.assertEqual(engine.lab_since(), {"main": 123.0, "pair": 123.0, "end": 123.0, "maker": 123.0})
            self.assertEqual(engine.count_since()["stocks"], 123.0)
            engine.set_count_since("maker", False)
            self.assertEqual(engine.lab_since()["maker"], None)  # and each is its own from then on
            engine.store.close()


class MakerCheckTest(unittest.TestCase):
    def test_trades_in_the_markets_are_read_and_kept_with_their_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            t = engine.trackers[0]
            t.window, t.market = 1_800_000_000, {"up_token": "U", "down_token": "D"}
            books = feeds.BookStream(lambda token: None, engine._on_trade)
            books._handle(json.dumps([{"event_type": "last_trade_price", "asset_id": "D", "price": "0.91", "size": "50",
                                       "side": "BUY", "timestamp": str((1_800_000_000 + 800) * 1000)},
                                      {"event_type": "last_trade_price", "asset_id": "X", "price": "0.5", "size": "1",
                                       "side": "SELL", "timestamp": "0"}]))  # (a market not followed: left out)
            engine._flush_trades()
            rows = engine.store._x("SELECT asset, minutes, window_start, outcome, price, size, taker, seconds_left FROM trades")
            self.assertEqual([tuple(r) for r in rows], [("btc", 15, 1_800_000_000, "down", 0.91, 50.0, "BUY", 100.0)])
            engine.store.close()

    def test_the_maker_side_of_each_trade_is_scored_on_the_result(self):
        from polybot import marketcheck
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            rows = []
            for i in range(100):  # takers buy the favourite at 90c a minute before the close; it wins 80% of the time
                ws = 1_800_000_000 + i * 900
                store._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', 15, ?, ?)",
                         (ws, "down" if i % 5 == 0 else "up"))
                rows += [(ws + 820, "btc", 15, ws, "up", 0.9, 10.0, "BUY", 80.0), (ws + 821, "btc", 15, ws, "up", 0.9, 10.0, "BUY", 79.0)]
            store.add_trades(rows)
            db = connect(cfg.db_path)
            r = marketcheck.maker_study(db, cfg)
            g = next(g for g in r["groups"] if g["taker"] == "BUY" and g["price_lo"] == 0.9 and g["time"] == "60-120")
            self.assertEqual((g["rounds"], g["trades"]), (100, 200))
            self.assertAlmostEqual(g["per_share"], 0.9 - 0.8)  # we sold at 90c what pays $1 only 80% of the time
            self.assertTrue(g["edge_found"])
            self.assertEqual(r["trades"], 200)
            db.close()
            store.close()


class SamePeriodTwinTest(unittest.TestCase):
    def test_a_strategy_is_compared_with_its_lab_twin_from_its_first_trade_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            m = engine.main_strategies()[0]
            a, mins, v = m["asset"], m["minutes"], m["variant"]
            add = lambda mode, ts, pnl: engine.store._x(
                "INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl, winner)"
                " VALUES (?, ?, 'filled', ?, ?, ?, ?, 'up', 10, 5, 0, ?, 'up')", (mode, v, ts, a, mins, int(ts), pnl))
            add("lab", 100, 5.0)  # before it came to the simulation: the history that picked it
            add("lab", 200, -5.0)
            add("paper", 200, -5.0)
            twin = engine.strategy_twins(connect(cfg.db_path))["paper"][f"{a}|{mins}|{v}"]
            self.assertEqual(twin["since"], 200)
            self.assertEqual((twin["lab"]["settled"], twin["lab"]["pnl"]), (1, -5.0))  # the same period only
            self.assertEqual((twin["here"]["settled"], twin["here"]["pnl"]), (1, -5.0))
            engine.store.close()


class CardResetTest(unittest.TestCase):
    """The owner (29.10.2026): a strategy card's figures start over on the card only; every trade stays in the data."""

    def test_the_card_counts_from_the_reset_and_everything_else_from_the_first_trade(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            m = engine.main_strategies()[0]
            a, mins, v = m["asset"], m["minutes"], m["variant"]
            add = lambda mode, ts, pnl: engine.store._x(
                "INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl, winner)"
                " VALUES (?, ?, 'filled', ?, ?, ?, ?, 'up', 10, 5, 0, ?, 'up')", (mode, v, ts, a, mins, int(ts), pnl))
            for ts, pnl in ((100, -5.0), (200, -5.0), (300, 5.0)):
                add("paper", ts, pnl)
                add("lab", ts, pnl)
            with mock.patch.object(time, "time", lambda: 250.0):
                engine.set_card_since("sim", a, mins, v)
            twin = engine.strategy_twins(connect(cfg.db_path))["paper"][f"{a}|{mins}|{v}"]
            self.assertEqual((twin["shown"]["settled"], twin["shown"]["pnl"], twin["shown_since"]), (1, 5.0, 250.0))  # the card
            self.assertEqual((twin["here"]["settled"], twin["here"]["pnl"]), (3, -5.0))  # the rest: every trade
            self.assertEqual(twin["lab"]["settled"], 3)
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'paper'")[0][0], 3)  # nothing deleted
            engine.set_card_since("sim", a, mins, v, fresh=False)
            self.assertNotIn("shown", engine.strategy_twins(connect(cfg.db_path))["paper"][f"{a}|{mins}|{v}"])
            with self.assertRaises(ValueError):
                engine.set_card_since("lab", a, mins, v)
            engine.store.close()


class RaceExperimentsTest(unittest.TestCase):
    """The race for the shares (the owner and three panels, 29.10.2026): A-D on what is recorded, E live, F wallets."""
    V = "twap|0.05|0-300"

    def _db(self, tmp):
        from polybot.store import Store
        return Store(os.path.join(tmp, "t.sqlite"))

    def test_a_the_simulation_racing(self):
        from polybot import race
        from polybot.store import connect
        with tempfile.TemporaryDirectory() as tmp:
            st = self._db(tmp)
            for i in range(4):
                ws = 1_800_000_000 + i * 300
                st._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, variant, shares, cost, fee, avg_price, pnl, book)"
                      " VALUES ('paper', 'filled', ?, 'btc', 5, ?, 'up', ?, 10, 5, 0.1, 0.5, ?, ?)",
                      (ws + 100, ws, self.V, 4.9 if i % 2 else -5.1, json.dumps([[0.5, 12]])))
                if i < 2:  # others bought 5 at our price 0.2s after the signal: only 2 left for our 10
                    st.add_trades([(ws + 100.2, "btc", 5, ws, "up", 0.5, 5.0, "BUY", 199.8)])
            db = connect(st.path)
            r = race.racing(db, 0, 0.5)
            self.assertEqual((r["now"]["n"], r["race"]["n"], r["lost"]), (4, 2, 2))
            self.assertAlmostEqual(r["race"]["pnl"], -5.1 + 4.9)
            self.assertEqual(r["rows"][0]["now"]["n"], 4)
            db.close()
            st.close()

    def test_b_the_intent_kept_open_buys_a_steady_offer_within_the_edge(self):
        from polybot import race
        from polybot.store import connect
        with tempfile.TemporaryDirectory() as tmp:
            st = self._db(tmp)
            ws = 1_800_000_000
            st._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', 5, ?, 'up')", (ws,))
            st.add_fill_test((ws + 100, "btc", 5, ws, "paper", self.V, "up", 10, 0.50, 0, None, None, None, None))
            for k, ask in enumerate((0.60, 0.53, 0.53, 0.53)):  # 0.60: past the edge; 0.53 held from the 2nd second
                st._x("INSERT INTO ticks (ts, asset, minutes, window_start, seconds_left, fair_up, up_ask, down_ask)"
                      " VALUES (?, 'btc', 5, ?, ?, 0.62, ?, 0.5)", (ws + 101 + k, ws, 199 - k, ask))
            db = connect(st.path)
            r = race.intent(db, 0, 0.0, 1.0)
            W5 = r["windows"]["5"]
            self.assertEqual((W5["tried"], W5["filled"]), (1, 1))
            self.assertAlmostEqual(W5["pnl"], 10 * (1 - 0.53))
            self.assertAlmostEqual(W5["paid_more"], 0.03)
            self.assertIsNone(race.intent_fill([(1, 199, 0.62, 0.53, 0.5)], "up", race._plain(self.V), 0.0, 1.0, 5))  # (a flicker)
            db.close()
            st.close()

    def test_c_the_room_by_price_and_d_standing_in_the_book(self):
        from polybot import race
        from polybot.store import connect
        with tempfile.TemporaryDirectory() as tmp:
            st = self._db(tmp)
            for i in range(40):
                ws = 1_800_000_000 + i * 300
                st._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', 5, ?, ?)", (ws, "up" if i % 2 else "down"))
                st.add_fill_test((ws + 100, "btc", 5, ws, "paper", self.V, "up", 10, 0.15, 0, None, None, 1.7, 0.05))  # cheap: +2c fills
                st._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, variant, shares, cost, fee, avg_price, pnl)"
                      " VALUES ('paper', 'rejected', ?, 'btc', 5, ?, 'up', ?, 10, 5, 0.1, 0.5, NULL)", (ws + 100, ws, self.V))
                st._x("INSERT INTO ticks (ts, asset, minutes, window_start, seconds_left, fair_up, up_ask, down_ask)"
                      " VALUES (?, 'btc', 5, ?, 190, 0.7, 0.55, 0.47)", (ws + 101, ws))
                st.add_trades([(ws + 110, "btc", 5, ws, "up", 0.47, 20.0, "SELL", 190.0)])  # sellers went under 0.48 and 0.49
            db = connect(st.path)
            C = race.shading(db, 0)
            cheap = C["bands"][0]
            self.assertEqual((cheap["refused"], cheap["rooms"]["2"]["n"], cheap["best"]), (40, 40, "2"))
            self.assertAlmostEqual(cheap["rooms"]["2"]["pnl"], 20 * 10 - 40 * 1.75)
            D = race.standing(db, 0)
            self.assertEqual({u: D["bids"][u]["n"] for u in D["bids"]}, {"0.0": 40, "0.01": 40, "0.02": 40})
            self.assertAlmostEqual(D["bids"]["0.02"]["pnl"], 20 * 10 - 40 * 4.8)  # (no fee for the maker)
            db.close()
            st.close()

    def test_e_the_futures_lead(self):
        from polybot import lead
        L = lead.LeadWatch()
        t = 1000.0
        for k in range(10):  # flat
            for src in ("binance", "binance_fut", "coinbase"):
                L.add(src, "btc", 100.0, now=t + k * 0.1)
        L.add("binance_fut", "btc", 100.05, now=t + 1.0)  # the futures first...
        L.add("coinbase", "btc", 100.05, now=t + 1.08)
        L.add("binance", "btc", 100.05, now=t + 1.05)  # ...spot 50 ms later
        L.add("binance", "btc", 100.05, now=t + 3.2)  # (the event settles once the sources had their time)
        v = L.view()
        self.assertEqual(v["events"], 1)
        self.assertAlmostEqual(v["binance_fut"]["median"], 50.0, places=0)
        self.assertAlmostEqual(v["coinbase"]["median"], -30.0, places=0)
        self.assertEqual(lead.LeadWatch(L.dump()).view()["events"], 1)  # (kept across a restart)

    def test_f_the_wallets_that_race_us(self):
        from polybot import wallets
        from polybot.store import connect
        with tempfile.TemporaryDirectory() as tmp:
            st = self._db(tmp)
            ws = int(time.time()) - 3600
            st._x("INSERT INTO windows (asset, minutes, window_start, slug, winner) VALUES ('btc', 5, ?, 'btc-5m-x', 'up')", (ws,))
            st.add_fill_test((ws + 100, "btc", 5, ws, "paper", self.V, "up", 10, 0.5, 0, None, None, None, None))
            trades = [{"proxyWallet": "0xAAA", "pseudonym": "fast-bot", "side": "BUY", "outcome": "Up", "size": 100, "price": 0.5, "timestamp": ws + 100},
                      {"proxyWallet": "0xbbb", "side": "BUY", "outcome": "Down", "size": 10, "price": 0.5, "timestamp": ws + 250}]
            db = connect(st.path)
            r = wallets.study(db, {}, sleep=lambda s: None, fetch_trades=lambda cid: trades, condition=lambda slug, cache: "0xcid")
            self.assertEqual((r["rounds"], r["wallets"]), (1, 2))
            self.assertEqual((r["racers"][0]["wallet"], r["racers"][0]["raced"], r["racers"][0]["pnl"]), ("0xaaa", 1, 50.0))
            self.assertEqual(r["top"][-1]["pnl"], -5.0)
            db.close()
            st.close()

    def test_the_engine_runs_them_and_the_hub_shows_six_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={}, streaming=False, with_lab=False)
            self.assertTrue(engine._race_tick(1_900_000_000.0, apart=False))
            self.assertFalse(engine._race_tick(1_900_000_000.0 + 60, apart=False))  # (half an hour between studies)
            self.assertEqual(set(engine.race) >= {"racing", "intent", "shading", "standing"}, True)
            self.assertTrue(engine._wallets_tick(1_900_000_000.0, sleep=lambda s: None, fetch_trades=lambda c: [], condition=lambda s, c: None))
            keys = [t["key"] for t in engine.hub_view()["tracks"]]
            self.assertTrue({"race_racing", "race_intent", "race_shading", "race_standing", "race_lead", "race_wallets"} <= set(keys))
            engine.store.close()


class LimitlessGoneTest(unittest.TestCase):
    def test_its_tables_are_dropped_from_an_old_database(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.sqlite")
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE lx_ticks (ts REAL)")
            db.execute("CREATE TABLE lx_trades (ts REAL)")
            db.commit()
            db.close()
            store = Store(path)
            names = {r[0] for r in store._x("SELECT name FROM sqlite_master WHERE type = 'table'")}
            self.assertFalse({"lx_ticks", "lx_trades", "lx_windows"} & names)
            store.close()


class LabFillsLikeARealOrderTest(unittest.TestCase):
    def _tracker(self, tmp):
        cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_fill_delay=0.5)
        engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
        t = engine.trackers[0]
        t.window = 1_800_000_000
        fill = Fill(side="up", fair_prob=0.8, best_ask=0.6, avg_price=0.6, shares=10, cost=6.0, fee=0.1, edge=0.19,
                    limit_price=0.6)
        t.lab_orders = [{"due": 100.5, "ts": 100.0, "variant": "twap|0.05|15-600", "fill": fill, "seconds_left": 200,
                         "price_age": 0, "book_age": 0, "fair_raw": 0.8, "stake": 6.0}]
        return engine, t

    def test_the_order_fills_only_if_the_book_still_has_it_when_it_arrives(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            t._arrive(100.2, [(0.6, 100)], [(0.41, 100)])
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets")[0][0], 0)  # still on its way
            t._arrive(100.6, [(0.59, 4), (0.6, 100)], [(0.41, 100)])  # there, and a little cheaper even
            row = engine.store._x("SELECT status, shares, avg_price, ts FROM bets")[0]
            self.assertEqual(row[0], "filled")
            self.assertAlmostEqual(row[1], 10)
            self.assertAlmostEqual(row[2], (4 * 0.59 + 6 * 0.6) / 10)
            self.assertEqual(row[3], 100.0)  # entered when the signal came
            engine.store.close()

    def test_the_simulation_fills_the_same_way_and_its_entries_take_the_book_in_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            first = dict(t.lab_orders[0], mode="paper")
            t.lab_orders = [first, dict(first)]  # a rule's two entries, 10 shares each, at most 60c
            t._arrive(100.6, [(0.6, 15)], [(0.41, 100)])  # only 15 shares at 60c when they arrive
            rows = engine.store._x("SELECT mode, status, shares FROM bets ORDER BY id")
            # (like real money's order since 28.10.2026: what there is, the second entry the 5 left)
            self.assertEqual([tuple(r) for r in rows], [("paper", "filled", 10.0), ("paper", "filled", 5.0)])
            engine.store.close()

    def test_a_book_that_moved_away_refuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            t._arrive(100.6, [(0.62, 100)], [(0.39, 100)])  # nothing left at 60c
            self.assertEqual(tuple(engine.store._x("SELECT status, pnl FROM bets")[0]), ("rejected", None))
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM fill_tests")[0][0], 1)  # (the room study's case)
            engine.store.close()

    def test_a_book_with_some_left_fills_what_there_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            t._arrive(100.6, [(0.6, 3), (0.62, 100)], [(0.39, 100)])  # only 3 shares left at 60c
            row = tuple(engine.store._x("SELECT status, shares, avg_price FROM bets")[0])
            self.assertEqual(row[:2], ("filled", 3.0))
            self.assertAlmostEqual(row[2], 0.6)  # at the price seen: never the 62c past it
            engine.store.close()


class MarketCheckTest(unittest.TestCase):
    def test_a_side_that_wins_more_often_than_it_costs_is_found(self):
        from polybot import marketcheck
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            # 400 rounds: Up offered at 90c with 20 seconds left, and it wins 98% of the time (it costs ~90.6c).
            for i in range(400):
                ws = 1_800_000_000 + i * 900
                winner = "down" if i % 50 == 0 else "up"
                store._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', 15, ?, ?)", (ws, winner))
                for left, up, down in ((20, 0.90, 0.11), (19, 0.91, 0.10), (400, 0.55, 0.46)):
                    store.add_tick(ts=ws + 900 - left, asset="btc", minutes=15, window_start=ws, price=1, open_price=1,
                                   seconds_left=left, vol=0.1, fair_up=0.5, up_bid=up - 0.01, up_ask=up,
                                   down_bid=down - 0.01, down_ask=down, price_age=0, book_age=0)
            db = connect(cfg.db_path)
            r = marketcheck.study(db, cfg)
            self.assertEqual(r["rounds"], 400)
            g = next(g for g in r["groups"] if g["time"] == "15-30" and g["price_lo"] == 0.9)
            self.assertEqual(g["n"], 400)  # one sample a round in a stretch: the first price seen in it
            self.assertAlmostEqual(g["win_rate"], 0.98)
            self.assertGreater(g["edge"], 0.05)
            self.assertTrue(g["edge_found"])
            lows = [x["edge_low"] for x in r["groups"] if x["edge_low"] is not None]
            self.assertEqual(lows, sorted(lows, reverse=True))  # the best pessimistic edge first
            cheap = next(g for g in r["groups"] if g["time"] == "15-30" and g["price_lo"] == 0.1)
            self.assertFalse(cheap["edge_found"])  # the 10c side wins 2%: no
            db.close()
            store.close()


class MoveKeepsTheRecordingTest(unittest.TestCase):
    def test_recording_start_and_the_last_day_of_prices_travel(self):
        from polybot import export
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "old.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            now = time.time()
            for ts in (now - 10 * 86400, now - 3600):  # a tick ten days ago, and one an hour ago
                engine.store._x("INSERT INTO ticks (ts, asset, minutes, window_start, price) VALUES (?, 'btc', 15, 0, 100)", (ts,))
            db = connect(cfg.db_path)
            self.assertAlmostEqual(engine.recording_since(db), now - 10 * 86400, places=0)  # stored as a setting
            db.close()
            zip_path = os.path.join(tmp, "full.zip")
            export.write(cfg, zip_path, full=True)  # only the last day of ticks
            engine.store.close()
            new = Config(assets=("btc",), db_path=os.path.join(tmp, "new.sqlite"))
            Store(new.db_path).close()
            counts = export.read(new, zip_path)
            self.assertEqual(counts["ticks"], 1)
            again = Engine(new, api=FakeApi(), env={}, streaming=False, with_lab=False)
            db = connect(new.db_path)
            self.assertAlmostEqual(again.recording_since(db), now - 10 * 86400, places=0)  # the hours of recording go on
            db.close()
            again.store.close()


class LabRunsOnTheOwnersEntryTimesTest(unittest.TestCase):
    """The owner picks the lab's entry times (say, only the last minute and the last 30 seconds)."""

    def test_short_windows_on_the_grid(self):
        for minutes in (5, 15):
            groups = {lab.time_group(v.key) for v in lab.variants_for(minutes)}
            self.assertIn("30", groups)
            self.assertIn("60", groups)
        self.assertEqual(lab.time_group("twap|0.05|5-30"), "30")
        self.assertEqual(lab.time_group("twap|0.05|10-45"), "60")
        self.assertEqual(lab.time_group("lock60|0.50|3-25"), "30")
        self.assertEqual(lab.time_group("twap|0.05|15-180"), "240")
        self.assertEqual(lab.time_group("pair80|0.15|450-885"), "more")
        # (a resting order needs time: none on the short windows)
        self.assertFalse(any(v.resolution == "twapm" and v.max_seconds_left < 120 for v in lab.variants_for(5)))

    def test_the_lab_runs_only_the_picked_times_the_owners_strategies_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.set_lab_focus(False)  # (the whole grid: the lab's focus on the locked pair is off)
            engine.refresh_lab = lambda: None
            t = next(x for x in engine.trackers if x.key == "btc-15m")
            every = len(t.variants)
            self.assertIsNone(engine.state()["lab_times"]["picked"])  # the system's: every time
            engine.set_lab_times(["30", "60"])
            self.assertEqual(engine.lab_times(), {"30", "60"})
            mine = {(m["variant"]) for m in engine.strategies_for("sim") if (m["asset"], m["minutes"]) == ("btc", 15)}
            self.assertTrue(t.variants)
            self.assertTrue(all(lab.time_group(v.key) in ("30", "60") for v in t.variants))  # nothing else, not even
            self.assertTrue(any(lab.time_group(k) not in ("30", "60") for k in mine))  # the owner's 10-minute strategy
            self.assertTrue(engine.strategies_for("sim"))  # (which still trades in the simulation)
            self.assertLess(len(t.variants), every)
            view = engine.state()["lab_times"]
            self.assertEqual(view["picked"], ["30", "60"])
            self.assertEqual([g["key"] for g in view["groups"] if g["on"]], ["30", "60"])
            self.assertTrue(all(g["variants"] > 0 for g in view["groups"] if g["key"] in ("30", "60")))
            # The search near winners stays within the picked times.
            engine._last_explore = 0
            engine.explore([{"asset": "btc", "minutes": 15, "variant": "twap|0.05|15-60", "status": "candidate",
                             "bets": 999, "roi": 0.2, "roi_pessimistic": 0.1}], time.time())
            added = [k for k in engine._json_setting("lab_dynamic", {}).get("btc|15", []) if "|r" not in k]
            self.assertTrue(all(lab.time_group(k) in ("30", "60") for k in added))
            # Back to the system's choice: every time again (picking all of them is the same).
            engine.set_lab_times(None)
            self.assertIsNone(engine.lab_times())
            self.assertGreaterEqual(len(t.variants), every)
            engine.set_lab_times([g for g, _, _ in lab.TIME_GROUPS])
            self.assertIsNone(engine.lab_times())
            engine.store.close()


class TheMarketCheckSurvivesARestartTest(unittest.TestCase):
    def test_the_last_check_shows_at_once_after_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            self.assertIsNone(engine.market_check)
            engine.refresh_market_check()
            ts = engine.market_check["ts"]
            engine.store.close()
            again = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)  # (a deploy restarts it)
            self.assertEqual(again.market_check["ts"], ts)
            again.store.close()


class TheLeaderAtTheEndTest(unittest.TestCase):
    """The leader at the very end: with 30 to 15 seconds left, the side the market leads, up to a cap."""

    def test_the_rule(self):
        from polybot.paper import find_lead
        v = lab.Variant.from_key("lead55|0.90|15-30")
        self.assertTrue(lab.is_lock(v) and lab.is_lead(v))
        self.assertEqual(lab.family(v.key), "main")  # (the last minute's tab is gone)
        self.assertEqual(lab.time_group(v.key), "30")
        self.assertIn("המוביל בסוף", lab.describe(v.key))
        cfg = Config(fee_rate=0.0, stake_usd=5.0)
        # Up leads at 78c: bought, whatever the model says (it is only recorded).
        f = find_lead(0.3, [(0.78, 100)], [(0.23, 100)], 0.55, 0.90, cfg)
        self.assertEqual(f.side, "up")
        self.assertAlmostEqual(f.avg_price, 0.78)
        self.assertIsNone(find_lead(0.9, [(0.93, 100)], [(0.08, 100)], 0.55, 0.90, cfg))  # over the cap
        self.assertIsNone(find_lead(0.5, [(0.52, 100)], [(0.49, 100)], 0.55, 0.90, cfg))  # no side leads enough
        self.assertIsNone(find_lead(0.9, [(0.78, 3)], [(0.23, 100)], 0.55, 0.90, cfg))  # not the whole $5 there
        self.assertEqual(Tracker._decide(v, 0.3, [(0.78, 100)], [(0.23, 100)], cfg).side, "up")  # how the lab decides it


class WhyAStrategyRunsDifferentlyTest(unittest.TestCase):
    def test_round_by_round_against_the_lab(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            v = "twap|0.05|15-600"
            now = time.time()

            def bet(mode, ws, price, paid, pnl, status="filled"):
                engine.store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, avg_price,"
                                " shares, cost, fee, pnl) VALUES (?, ?, ?, ?, 'btc', 15, ?, 'up', ?, ?, ?, 0, ?)",
                                (mode, v, status, now + ws, ws, price, paid / price, paid, pnl))
            for ws in (0, 900, 1800):  # both entered; the simulation paid more a share, with $100
                bet("paper", ws, 0.60, 100, -10)
                bet("lab", ws, 0.55, 10, 1)
            bet("paper", 2700, 0.5, 100, -100)  # only the simulation
            bet("lab", 3600, 0.5, 10, 5)  # only the lab
            bet("paper", 4500, 0.5, 100, None, status="rejected")
            D = engine.strategy_diff("sim", "btc", 15, v)
            self.assertEqual(D["both"]["here"]["rounds"], 3)
            self.assertAlmostEqual(D["both"]["here"]["price"], 0.60)
            self.assertAlmostEqual(D["both"]["lab"]["price"], 0.55)
            self.assertAlmostEqual(D["both"]["here"]["per_trade"], 100)
            self.assertEqual(D["only_here"]["rounds"], 1)
            self.assertEqual(D["only_lab"]["rounds"], 1)
            self.assertEqual(D["rejected"]["here"], 1)
            self.assertIn("stake_here", D["settings"])
            engine.store.close()


class MakerOnTheLeaderTest(unittest.TestCase):
    """A resting buy on the side the market leads, no model: filled only if sellers come through it."""

    def test_rests_under_the_leaders_offer_and_fills_when_swept(self):
        v = lab.Variant.from_key("fav60|0.70|180-300")
        self.assertTrue(lab.is_fav(v))
        self.assertEqual(lab.reading(v), "twap")
        self.assertIn("עושה שוק על המוביל", lab.describe(v.key))
        self.assertIn(v.key, {x.key for x in lab.variants_for(15)})
        self.assertNotIn(v.key, {x.key for x in lab.variants_for(5)})  # (a 5-minute round is too short for it)
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = AddToATradeTest._engine(AddToATradeTest(), tmp)
            now = time.time()
            t._maker(v, now, 250, 0.2, 0.2, [(0.66, 100)], [(0.36, 100)], 0, 0)  # Up leads at 66c
            order = t.maker_orders[v.key]
            self.assertEqual((order["side"], order["price"]), ("up", 0.65))
            t._maker(v, now + 1, 240, 0.2, 0.2, [(0.80, 100)], [(0.22, 100)], 0, 0)  # it stays where it is
            self.assertEqual(t.maker_orders[v.key]["price"], 0.65)
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets")[0][0], 0)
            t._maker(v, now + 2, 230, 0.2, 0.2, [(0.60, 100)], [(0.41, 100)], 0, 0)  # sellers came down through it
            row = engine.store._x("SELECT mode, side, avg_price, fee FROM bets")[0]
            self.assertEqual(tuple(row), ("lab", "up", 0.65, 0.0))  # no fee for a maker
            engine.store.close()

    def test_the_simulation_runs_a_maker_strategy_as_a_resting_order(self):
        # (the bug: the simulation sent it through the taker's rule, where its 70c cap read as a
        # 70-point edge, so it never traded; a model maker bought at the offer instead of resting)
        v = lab.Variant.from_key("fav60|0.70|180-300")
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = AddToATradeTest._engine(AddToATradeTest(), tmp)
            t.main_rules = [v]
            now = time.time()
            cal, raw = {"twap": 0.9}, {"twap": 0.9}  # (the model agrees with the leader)
            t._sim_rules_run(now, 250, cal, raw, [(0.66, 100)], [(0.36, 100)], 0, 0, cal, cal)
            self.assertEqual(t.sim_maker_orders[v.key]["price"], 0.65)
            self.assertNotIn(v.key, t.maker_orders)  # (the lab's own orders are apart)
            t._sim_rules_run(now + 2, 230, cal, raw, [(0.60, 100)], [(0.41, 100)], 0, 0, cal, cal)
            row = engine.store._x("SELECT mode, variant, side, avg_price, fee FROM bets")[0]
            self.assertEqual(tuple(row), ("paper", v.key, "up", 0.65, 0.0))
            t._sim_rules_run(now + 3, 220, cal, raw, [(0.66, 100)], [(0.36, 100)], 0, 0, cal, cal)
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets")[0][0], 1)  # once a round
            engine.store.close()

    def test_the_same_rule_in_the_simulation_as_in_the_lab(self):
        v = lab.Variant.from_key("fav60|0.97|180-300")
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = AddToATradeTest._engine(AddToATradeTest(), tmp)
            t.main_rules = [v]
            now = time.time()
            # Down leads at 90c: the order rests at 89c, whatever the model says (no layer over the rule).
            t._sim_rules_run(now, 250, {"twap": 0.29}, {"twap": 0.29}, [(0.11, 100)], [(0.90, 100)], 0, 0, {}, {})
            self.assertEqual(t.sim_maker_orders[v.key]["price"], 0.89)
            t._maker(v, now, 250, 0.29, 0.29, [(0.11, 100)], [(0.90, 100)], 0, 0)
            self.assertEqual(t.maker_orders[v.key]["price"], 0.89)
            engine.store.close()

    def test_no_leader_no_order(self):
        v = lab.Variant.from_key("fav60|0.70|180-300")
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = AddToATradeTest._engine(AddToATradeTest(), tmp)
            t._maker(v, time.time(), 250, 0.5, 0.5, [(0.52, 100)], [(0.50, 100)], 0, 0)
            self.assertNotIn(v.key, t.maker_orders)
            engine.store.close()


class MakersFromTheFindingsTest(unittest.TestCase):
    """The maker check's findings as makers of their own (zones), and every maker measured twice:
    filled when sellers break through it, and by its place in the queue."""

    def test_findings_become_zones(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = AddToATradeTest._engine(AddToATradeTest(), tmp)
            engine.set_lab_focus(False)  # (the whole grid: the lab's focus on the locked pair is off)
            g = lambda taker, lo, hi, t_lo, t_hi, found=True: {"edge_found": found, "taker": taker, "price_lo": lo,
                                                               "price_hi": hi, "time_lo": t_lo, "time_hi": t_hi}
            engine.market_check = {"maker": {"groups": [
                g("SELL", 0.6, 0.7, 180, 300),  # sellers of a side at 60-70c lose to its buyers: buy it there
                g("BUY", 0.3, 0.4, 120, 180),  # buyers of a side at 30-40c lose to its sellers: buy the other at 60-70c
                g("BUY", 0.0, 0.1, 15, 30, found=False)]}}
            engine._maker_zones()
            zones = engine._json_setting("lab_zones", [])
            self.assertEqual(zones, ["zone060070|0.70|180-300", "zone060070|0.70|120-180"])
            v = lab.Variant.from_key(zones[0])
            self.assertTrue(lab.is_zone(v) and lab.is_maker(v.key))
            self.assertEqual(lab.zone_band(v), (0.6, 0.7))
            self.assertIn("מהממצאים", lab.describe(v.key))
            keys = {x.key for x in engine._lab_variants("btc", 15)}
            self.assertIn(zones[0], keys)
            self.assertIn(zones[0] + "|q", keys)  # (its queue twin)
            self.assertIn("fav60|0.70|180-300|q", keys)
            self.assertNotIn(zones[0], {x.key for x in engine._lab_variants("btc", 5)})  # (300s doesn't fit a 5-minute round)
            engine.store.close()

    def test_a_zone_rests_and_the_queue_twin_fills_by_the_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = AddToATradeTest._engine(AddToATradeTest(), tmp)
            z = lab.Variant.from_key("zone060070|0.70|180-300")
            q = lab.Variant.from_key("zone060070|0.70|180-300|q")
            self.assertEqual(q.fill, "q")
            self.assertEqual(q.key, "zone060070|0.70|180-300|q")
            t._bids = {"up": [(0.65, 40.0)], "down": []}  # 40 shares already resting at 65c
            now = time.time()
            for v in (z, q):
                t._maker(v, now, 250, 0.5, 0.5, [(0.66, 100)], [(0.36, 100)], 0, 0)
            self.assertEqual((t.maker_orders[q.key]["price"], t.maker_orders[q.key]["queue"]), (0.65, 40.0))
            # Sellers take 40 + some shares at 65c: the queue twin is filled, the other (no break-through) isn't.
            shares = t.maker_orders[q.key]["shares"]
            t.on_market_trade("up", 0.65, 40 + shares, "SELL", now + 1)
            for v in (z, q):
                t._maker(v, now + 2, 240, 0.5, 0.5, [(0.66, 100)], [(0.36, 100)], 0, 0)
            rows = dict(engine.store._x("SELECT variant, avg_price FROM bets WHERE mode = 'lab'"))
            self.assertEqual(rows, {q.key: 0.65})
            # The ask breaks through: the other one too.
            t._maker(z, now + 3, 230, 0.5, 0.5, [(0.60, 100)], [(0.41, 100)], 0, 0)
            self.assertIn(z.key, dict(engine.store._x("SELECT variant, avg_price FROM bets WHERE mode = 'lab'")))
            engine.store.close()


class WhoMovesFirstTest(unittest.TestCase):
    """The live meter of ideas 2 and 3: after a jump of the model's chance, how Polymarket follows."""

    def test_a_jump_its_market_and_the_other_coin(self):
        from polybot.lagwatch import LagWatch
        w = LagWatch()
        t0 = 1000.0

        def feed(ts, key, asset, fair, mid):
            w.observe(key, asset, 15, ts, fair, mid - 0.01, mid + 0.01, 1 - mid - 0.01, 1 - mid + 0.01, 0.0, 1.0)
        ts = t0
        while ts < t0 + 20:
            jumped = ts >= t0 + 3
            # Bitcoin: the model jumps at +3s; its market follows 2 seconds later.
            feed(ts, "btc-15m", "btc", 0.60 if jumped else 0.50, 0.58 if ts >= t0 + 5 else 0.50)
            # Ether: its model follows a second after Bitcoin's jump, its market a second after that.
            feed(ts, "eth-15m", "eth", 0.55 if ts >= t0 + 4 else 0.50, 0.55 if ts >= t0 + 5 else 0.50)
            ts += 0.1
        S = w.summary()
        by = {(g["kind"], g["follower"]): g for g in S["groups"]}
        me = by[("self", "btc")]
        self.assertEqual(me["events"], 1)
        self.assertAlmostEqual(me["lag_median"], 2.0, delta=0.15)
        self.assertEqual(me["market_first"], 0)
        # Bought half a second after the jump at the stale offer (51c), worth 58c ten seconds later.
        self.assertAlmostEqual(me["mark_mid"], 0.07, delta=0.005)
        other = by[("cross", "eth")]
        self.assertAlmostEqual(other["fair_lag_median"], 1.0, delta=0.15)
        self.assertAlmostEqual(other["lag_median"], 2.0, delta=0.15)
        self.assertGreater(other["mark_mid"], 0)
        again = LagWatch(w.events())  # (kept across a restart)
        self.assertEqual(again.summary()["events"], S["events"])


class BooksCheckWithNothingRunningTest(unittest.TestCase):
    def test_the_books_are_measured_on_every_market_when_no_strategy_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            with_lab=False)
            engine.main_trackers = []  # (every simulation strategy paused)
            t = engine.trackers[0]
            t.market = {"up_token": "U", "down_token": "D"}
            t.snapshot = dict(t.snapshot, book_age=0.4)
            books = next(c for c in engine.health.report(time.time())["checks"] if c["name"] == "books")
            engine.store.close()
        self.assertTrue(books["ok"])
        self.assertAlmostEqual(books["value"], 0.4)


class TheEdgeHubTest(unittest.TestCase):
    def test_one_place_for_every_test_looking_for_an_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.refresh_lab = lambda: None
            row = lambda v, st, n: {"asset": "btc", "minutes": 15, "variant": v, "status": st, "bets": n, "roi": 0.1,
                                    "roi_pessimistic": 0.02, "oos_bets": 0}
            engine.lab = {"board": [row("fav60|0.70|180-300", "candidate", 80), row("fav90|0.97|30-120", "losing", 40),
                                    row("lead55|0.90|15-30", "collecting", 5), row("twap|0.05|15-600", "proven", 300)]}
            engine.market_check = {"maker": {"groups": [{"edge_found": True}], "days": 0.1, "rounds": 225}, "ts": time.time()}
            H = {t["key"]: t for t in engine.hub_view()["tracks"]}
            self.assertEqual(set(H), {"speed", "maker_check", "zone_lab", "fav_lab", "twapm_lab", "owner_rule", "end_fav",
                                      "streaks", "predict", "fill_why", "fill_room", "speed_tech", "race_racing", "race_intent",
                                      "race_shading", "race_standing", "race_lead", "race_wallets", "both_lock"})
            self.assertEqual(H["predict"]["state"], "collecting")
            self.assertEqual(H["streaks"]["state"], "collecting")
            self.assertEqual(H["maker_check"]["state"], "promising")  # found, but on a tenth of a day
            self.assertEqual(H["fav_lab"]["rules"], 2)  # (only its own rules)
            self.assertEqual(H["fav_lab"]["state"], "promising")
            self.assertEqual(H["fav_lab"]["best"]["variant"], "fav60|0.70|180-300")
            self.assertEqual(H["speed"]["state"], "collecting")
            engine._hub_cache = None
            engine.set_lab_times(["30", "60"])  # the maker's times not picked: it doesn't run
            H = {t["key"]: t for t in engine.hub_view()["tracks"]}
            self.assertEqual(H["fav_lab"]["state"], "off")
            self.assertEqual(H["fav_lab"]["groups"], ["120", "600"])
            self.assertEqual(H["twapm_lab"]["state"], "off")
            engine.store.close()


class StreaksTest(unittest.TestCase):
    """The owner's idea (27.10.2026): wins and losses come in runs across every strategy at once; trade
    only while the recent results are green. Tested on the trades already made, against a placebo."""

    @staticmethod
    def _trades(days, regimes, seed=1):
        rng, out, t, hot = random.Random(seed), [], 0, True
        while t < days * 86400:
            if regimes and rng.random() < 300 / 7200:
                hot = not hot
            for _ in range(rng.randint(3, 8)):
                win = rng.random() < ((0.62 if hot else 0.38) if regimes else 0.5)
                out.append((t, t + 300, 0.5 if win else -0.5, 0.5))
            t += 300
        return out

    def test_runs_show_up_and_chance_does_not(self):
        from polybot import streaks
        found = streaks.study(*streaks.from_events(self._trades(9, True)), (1800, 7200), 600, 7 * 86400, min_recent=20, shifts=120)
        self.assertEqual(found["state"], "found")
        best = next(r for r in found["rows"] if r["lookback"] == found["best"])
        self.assertGreater(best["hot"]["roi"], 0)
        self.assertLess(best["cold"]["roi"], 0)
        self.assertGreater(best["caught"], 0.5)  # most of the winning stretches' profit, made while hot
        self.assertGreater(best["avoided"], 0.5)
        none = streaks.study(*streaks.from_events(self._trades(9, False)), (1800, 7200), 600, 7 * 86400, min_recent=20, shifts=120)
        self.assertIn(none["state"], ("nothing", "collecting"))
        early = streaks.study(*streaks.from_events(self._trades(3, True)), (1800,), 600, 7 * 86400, min_recent=20, shifts=60)
        self.assertNotEqual(early["state"], "found")  # (three days of a week's need: not yet)
        self.assertLess(early["progress"], 0.5)

    def test_the_signal_never_knows_the_future(self):
        from polybot import streaks
        # Each bucket's trades end a second after they begin and all win or all lose together, by chance:
        # a signal that saw its own bucket would look perfect; one read at the bucket's start sees nothing.
        rng, ev = random.Random(5), []
        for b in range(2000):
            pnl = 1.0 if rng.random() < 0.5 else -1.0
            for k in range(5):
                ev.append((b * 600 + 60 + k, b * 600 + 61 + k, pnl, 1.0))
        r = streaks.study(*streaks.from_events(ev), (600,), 600, 86400, min_recent=1, shifts=60)["rows"][0]
        self.assertLess(abs(r["diff"]), 0.15)
        self.assertGreater(r["p"], 0.01)

    def test_the_engine_reads_the_labs_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.refresh_lab = lambda: None
            now = 1_800_000_000
            start = now - 9 * 86400
            for t0, _, pnl, paid in self._trades(9, True, seed=2):
                ws = start + t0
                engine.store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, cost, fee, shares, pnl)"
                                " VALUES ('lab', 'twap|0.05|15-200', 'filled', ?, 'btc', 5, ?, 'up', ?, 0, 1, ?)",
                                (ws + 60, ws, paid, pnl))
            engine.store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, cost, fee, shares, pnl)"
                            " VALUES ('paper', 'twap|0.05|15-200', 'filled', ?, 'btc', 5, ?, 'up', 1, 0, 1, 99)", (now - 100, now - 200))
            engine.refresh_streaks(now)
            self.assertEqual(engine.streaks["state"], "found")
            self.assertGreater(engine.streaks["trades"], 5000)
            H = {t["key"]: t for t in engine.hub_view()["tracks"]}
            self.assertEqual(H["streaks"]["state"], "found")
            self.assertGreater(H["streaks"]["days"], 8)
            # (kept: a restart shows it at once)
            again = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            self.assertEqual(again.streaks["state"], "found")
            again.store.close()
            engine.store.close()


class PredictTheMoveTest(unittest.TestCase):
    """The owner's question (27.10.2026): do the coin's move and who is buying say who wins, better than
    the market's own price? Each round's readings kept for good; scored against the market's price."""

    def test_a_rounds_readings(self):
        from polybot import predict
        ticks = [(180, 100.0, 0.48, 0.50, 0.50, 0.52), (120, 100.5, 0.55, 0.57, 0.43, 0.45), (60, 100.2, 0.5, 0.52, 0.48, 0.5)]
        trades = [(150, "up", 10, "BUY"), (130, "down", 4, "BUY"), (125, "down", 3, "SELL"), (100, "up", 99, "SELL")]
        r = predict.readings(ticks, trades, 5)
        self.assertAlmostEqual(r[120][4], 100.5 / 100.0 - 1)  # the coin's move over the minute before
        self.assertEqual(r[120][5], 10 - 4 + 3)  # buying Up, buying Down, selling Down (pushes Up)
        self.assertEqual(r[60][5], -99)
        self.assertNotIn(30, r)  # (no reading near 30 seconds left)

    def test_a_breakout(self):
        from polybot import predict
        hist = [(t, 100.0 + (t % 50) / 100, 99.5, 100.0) for t in range(0, 900, 5)]  # 15 minutes between 99.5 and 100.49
        up = hist + [(900, 101.0, 100.6, 101.0)]
        self.assertAlmostEqual(predict.breakout(up, 900, 15), 101.0 / max(p[1] for p in hist if 0 <= p[0] < 840) - 1)
        inside = hist + [(900, 100.2, 100.0, 100.2)]
        self.assertEqual(predict.breakout(inside, 900, 15), 0.0)
        down = hist + [(900, 99.0, 99.0, 99.0)]
        self.assertLess(predict.breakout(down, 900, 15), 0)
        self.assertIsNone(predict.breakout(up, 900, 60))  # (no hour of history)

    def test_who_is_buying_found_when_it_knows_more_than_the_price(self):
        from polybot import predict
        from polybot.store import Store, connect
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.sqlite")
            store = Store(path)
            rng = random.Random(4)
            start = 1_800_000_000
            for i in range(8 * 288):
                ws = start + i * 300
                push = rng.choice((1, -1))
                winner = ("up" if push > 0 else "down") if rng.random() < 0.7 else ("down" if push > 0 else "up")
                store._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', 5, ?, ?)", (ws, winner))
                for m in (180, 120, 90, 60, 30):
                    store._x("INSERT INTO ticks (ts, asset, minutes, window_start, price, seconds_left, up_bid, up_ask, down_bid, down_ask)"
                             " VALUES (?, 'btc', 5, ?, 100, ?, 0.49, 0.51, 0.49, 0.51)", (ws + 300 - m, ws, m))
                for m in (150, 75, 40):
                    store._x("INSERT INTO trades (ts, asset, minutes, window_start, outcome, price, size, taker, seconds_left)"
                             " VALUES (?, 'btc', 5, ?, 'up', 0.5, ?, 'BUY', ?)", (ws + 300 - m, ws, 5 + rng.random() * 20 if push > 0 else 0, m))
                    if push < 0:
                        store._x("INSERT INTO trades (ts, asset, minutes, window_start, outcome, price, size, taker, seconds_left)"
                                 " VALUES (?, 'btc', 5, ?, 'down', 0.5, ?, 'BUY', ?)", (ws + 300 - m, ws, 5 + rng.random() * 20, m))
            db = connect(path)
            self.assertEqual(predict.record(db, limit=1000, budget=60), 1000)
            predict.record(db, limit=5000, budget=60)
            self.assertEqual(predict.record(db), 0)  # (each round read once)
            r = predict.study(db)
            self.assertEqual(r["rounds"], 8 * 288)
            flow = [x for x in r["rows"] if x["feature"] == "flow"]
            self.assertTrue(all(x["edge"] > 0.15 for x in flow))
            self.assertEqual(r["state"], "found")
            self.assertFalse([x for x in r["rows"] if x["feature"] == "mom" and x["n"]])  # (the price never moved)
            # The raw recording goes (14 days); the readings stay.
            store._x("DELETE FROM ticks")
            self.assertEqual(predict.study(db)["rounds"], 8 * 288)
            db.close()
            store.close()

    def test_a_finished_card_put_away_and_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            self.assertEqual(engine.hide_research("p:speed"), ["p:speed"])
            self.assertEqual(engine.hide_research("a:streaks"), ["p:speed", "a:streaks"])
            self.assertEqual(engine.hide_research("p:speed", False), ["a:streaks"])
            with self.assertRaises(ValueError):
                engine.hide_research("")
            engine._predict_tick(now=1_800_000_000)
            H = {t["key"]: t for t in engine.hub_view()["tracks"]}
            self.assertEqual(H["predict"]["state"], "collecting")
            engine.store.close()


class OneMomentForAResultTest(unittest.TestCase):
    """The owner (28.10.2026): a real-money round's P&L, its close mail and the wallet shown change at one
    moment: once Polymarket decided it and the wallet read after that has it resolved."""

    def test_the_round_waits_for_the_wallet_then_everything_moves_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            api = FakeApi()
            engine = Engine(cfg, api=api, env={}, streaming=False)
            engine.learn_now = lambda reason: None  # (no learning thread on a closing database)
            start = int(time.time()) - 3600
            engine.store.upsert_window("btc", 5, start, slug="btc-updown-5m-x", found=1)
            engine.store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, cost, fee, shares)"
                            " VALUES ('live', 'twap|0.05|15-200', 'filled', ?, 'btc', 5, ?, 'up', 5, 0.1, 10)", (start + 100, start))
            api.winner = "up"
            engine.live.configured = lambda: True
            reads = []

            def read(open_=True):
                reads.append(time.time())
                engine.live.wallet = {"ts": time.time() + 0.001, "cash": 50.0, "positions": 10.0 if open_ else 0.0,
                                      "total": 60.0 if open_ else 50.0, "error": None}
                engine.live.wallet_slugs = {"btc-updown-5m-x": [(not open_, 10.0)]}
            engine.live.refresh_wallet = lambda: read(True)
            engine.live.wallet_shown = {"ts": 1, "cash": 45.0, "positions": 5.0, "total": 50.0}
            engine._settle()
            pnl = lambda: engine.store._x("SELECT pnl FROM bets WHERE mode = 'live'")[0][0]
            self.assertIsNone(pnl())  # Polymarket decided, the wallet still holds it open: nothing yet
            self.assertEqual(engine.live.status()["wallet"]["total"], 50.0)  # (the wallet shown waits too)
            engine._refresh_wallet()
            self.assertEqual(engine.live.status()["wallet"]["total"], 50.0)  # (a real trade waits: not shown yet)
            engine.live.refresh_wallet = lambda: read(False)
            engine.live.wallet["ts"] = 0  # (the next reading is fresh)
            engine._settle()
            self.assertAlmostEqual(pnl(), 10 - 5.1)  # now: the P&L,
            self.assertEqual(engine.live.status()["wallet"]["total"], 50.0)  # the wallet that has it,
            self.assertEqual(engine.live.status()["wallet"]["positions"], 0.0)
            # (no mail address here: the close is handled, at this same moment)
            self.assertTrue(engine.store._x("SELECT notified & 2 FROM bets WHERE mode = 'live'")[0][0])
            engine.store.close()

    def test_a_simulation_round_is_published_at_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            api = FakeApi()
            engine = Engine(cfg, api=api, env={}, streaming=False)
            engine.learn_now = lambda reason: None
            start = int(time.time()) - 3600
            engine.store.upsert_window("btc", 5, start, slug="s", found=1)
            engine.store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, cost, fee, shares)"
                            " VALUES ('paper', 'twap|0.05|15-200', 'filled', ?, 'btc', 5, ?, 'up', 5, 0.1, 10)", (start + 100, start))
            api.winner = "down"
            engine._settle()
            self.assertAlmostEqual(engine.store._x("SELECT pnl FROM bets")[0][0], -5.1)
            engine.store.close()


class OrdersThatDontGoThroughTest(unittest.TestCase):
    """The owner's questions (28.10.2026): why real orders are refused (the amount, or others faster),
    whether 1-2 cents of room would pay, and how fast we'd need to be."""

    def test_a_refused_order_is_kept_with_what_room_would_have_done(self):
        from polybot.paper import Fill
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            t = engine.trackers[0]
            t.window = 1_800_000_000
            f = Fill(side="up", fair_prob=0.7, best_ask=0.50, avg_price=0.50, shares=10, cost=5.0, fee=0.1, edge=0.1, limit_price=0.50)
            t._fill_test({"ts": 1.0, "mode": "paper", "variant": "v"}, f, [(0.50, 4.0), (0.51, 3.0), (0.52, 10.0)], t.market_cfg)
            row = engine.store._x("SELECT want, limit_price, left_at_limit, cost1, cost2 FROM fill_tests")[0]
            self.assertEqual(row[:3], (10, 0.50, 4.0))
            self.assertIsNone(row[3])  # +1c: 7 shares, not all 10
            self.assertAlmostEqual(row[4], 4 * 0.50 + 3 * 0.51 + 3 * 0.52)  # +2c: all of it, the cheapest first
            engine.store.close()

    def test_three_and_five_cents_and_the_cheap_sides(self):
        from polybot import fills
        from polybot.paper import Fill
        from polybot.store import connect
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            t = engine.trackers[0]
            t.window = 1_800_000_000
            engine.store._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', ?, ?, 'up')", (t.minutes, t.window))
            f = Fill(side="up", fair_prob=0.4, best_ask=0.15, avg_price=0.15, shares=10, cost=1.5, fee=0.05, edge=0.2, limit_price=0.15)
            t._fill_test({"ts": 1.0, "mode": "paper", "variant": "v"}, f, [(0.15, 2.0), (0.17, 3.0), (0.19, 10.0)], t.market_cfg)
            row = engine.store._x("SELECT cost1, cost2, cost3, cost5, wide FROM fill_tests")[0]
            self.assertEqual(row[:3], (None, None, None))  # +1, +2, +3c: not all 10
            self.assertAlmostEqual(row[3], 2 * 0.15 + 3 * 0.17 + 5 * 0.19)  # +5c: all of it
            self.assertEqual(row[4], 1)
            engine.store.add_fill_test((2.0, "btc", t.minutes, t.window, "paper", "v", "up", 10, 0.15, 0, 1.6, 0.05, 1.7, 0.05))  # (an old row)
            db = connect(cfg.db_path)
            r = fills.study(db)
            self.assertEqual((r["room"]["paper5"]["of"], r["room"]["paper5"]["would_fill"], r["room"]["paper1"]["of"]), (1, 1, 2))
            C = r["cheap"]
            self.assertEqual((C["refused"], C["won"], C["rooms"]["5"]["wins"], C["rooms"]["1"]["wins"]), (2, 2, 1, 1))
            self.assertGreater(C["rooms"]["5"]["pnl"], 0)
            db.close()
            engine.store.close()

    def test_why_room_and_speed(self):
        from polybot import fills
        from polybot.store import Store, connect
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.sqlite")
            store = Store(path)
            for i in range(80):
                ws = 1_800_000_000 + i * 300
                store._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', 5, ?, ?)", (ws, "up" if i % 3 else "down"))
                left = 0.0 if i < 60 else 3.0  # most: nothing left at our price (others were faster)
                store.add_fill_test((ws + 100, "btc", 5, ws, "paper", "v", "up", 10, 0.50, left, 5.1, 0.17, 5.2, 0.18))
                store._x("INSERT INTO trades (ts, asset, minutes, window_start, outcome, price, size, taker, seconds_left)"
                         " VALUES (?, 'btc', 5, ?, 'up', 0.50, 20, 'BUY', 190)", (ws + 100.3, ws))
            for i in range(30):
                store.add_order_timing((i, "btc", 5, "rejected", 0.004, 0.18))
            store.add_ping(5, "clob", 41.0)
            db = connect(path)
            r = fills.study(db)
            self.assertEqual(r["why"]["verdict"], "speed")
            self.assertEqual(r["why"]["gone"], 60)
            room = r["room"]["paper1"]
            self.assertEqual(room["settled"], 80)
            self.assertAlmostEqual(room["pnl"], sum(10 * (1 if i % 3 else 0) - 5.1 - 0.17 for i in range(80)))
            self.assertGreater(room["roi"], 0)
            self.assertEqual(r["room_state"], "promising")  # (80 settled of the 100 a verdict needs)
            S = fills.speed(db, our_delay=0.5)
            self.assertEqual(S["race_n"], 80)
            self.assertAlmostEqual(S["race_median"], 0.3, places=3)
            self.assertAlmostEqual(S["ours"], 0.184)
            self.assertEqual(S["ping"], 41.0)
            self.assertEqual(S["state"], "found")
            self.assertAlmostEqual(S["near_saved"], (41.0 - fills.NEAR_MS) / 2000)  # half the ping, less what's left there
            self.assertAlmostEqual(S["arrive"], 0.184 - 0.0205)
            self.assertEqual(S["near_saves"], 0.0)  # (the shares went 0.3s after the signal: after our order)
            db.close()
            store.close()


class EnterByItselfWhenItCanTest(unittest.TestCase):
    """The owner (28.10.2026): with real money on (automatic too), a simulation round can be copied by hand;
    and marked "enter by itself", it is copied the moment the button's check passes."""

    def test_waits_for_the_chance_then_enters(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.live.armed = lambda: True
            t = engine.trackers[0]
            t.window, t.market = int(time.time()) - 60, {"up_token": "U", "down_token": "D", "min_order_size": 5}
            calls = {"plan": 0, "done": 0}

            def plan(*a, **k):
                calls["plan"] += 1
                if calls["plan"] < 3:
                    raise ValueError("המחיר הממוצע עכשיו 60.0¢ ובסימולציה 55.0¢: המחיר ברח")
                return {"total": 5.0}
            engine._approve_plan = plan
            engine._sim_legs = lambda t: [("up", 0.55, 5.0, 9.0)]
            engine._execute_plan = lambda t, p, variant, now, left: calls.__setitem__("done", calls["done"] + 1) or {"paid": 5.0, "avg_price": 0.55}
            engine.set_auto_copy("btc", t.minutes, t.window, "twap|0.05|15-200", True)
            engine._auto_copy_tick()
            k = f"btc|{t.minutes}|{t.window}"
            self.assertEqual(engine.auto_copies()[k]["status"], "waiting")
            self.assertIn("ברח", engine._auto_why[k])  # (why not yet, on the card)
            engine._auto_copy_tick()
            engine._auto_copy_tick()  # the price came back: in, once
            engine._auto_copy_tick()
            self.assertEqual(calls["done"], 1)
            self.assertEqual(engine.auto_copies()[k]["status"], "in")
            # Too near the close: it stops waiting.
            t.window = int(time.time()) - t.window_seconds + 2
            engine.set_auto_copy("btc", t.minutes, t.window, "v", True)
            engine._auto_copy_tick()
            self.assertEqual(engine.auto_copies()[f"btc|{t.minutes}|{t.window}"]["status"], "expired")
            engine.store.close()


class FasterRealOrdersTest(unittest.TestCase):
    """Speed (28.10.2026): a round's market details loaded into Polymarket's SDK before any signal (an
    order then pays only its own trip), reloaded before the SDK forgets them, and the connection kept open."""

    def test_markets_loaded_ahead_and_the_connection_kept(self):
        from polybot.live import LiveTrader
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            live = LiveTrader(cfg, store, env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64})
            calls = []
            meta = types.SimpleNamespace(resolve_market=lambda ctx, token_id: calls.append(("load", token_id)),
                                         fetch_current_market=lambda ctx, token_id: calls.append(("reload", token_id)))
            clob = types.SimpleNamespace(get_json=lambda path: calls.append(("conn", path)))
            live.client = types.SimpleNamespace(_ctx=types.SimpleNamespace(order_metadata=meta, secure_clob=clob))
            self.assertEqual(live.warm(["U", "D"], now=1000), 0)  # real money off: nothing
            store.set_setting("live_armed", "1")
            self.assertEqual(live.warm(["U", "D"], now=1000), 2)
            self.assertEqual(calls, [("load", "U"), ("load", "D"), ("conn", "/time")])
            calls.clear()
            self.assertEqual(live.warm(["U", "D"], now=1010), 0)  # (loaded: nothing again, the connection too soon)
            self.assertEqual(calls, [])
            live.warm(["U", "D"], now=1000 + live.WARM_AGAIN + 1)
            self.assertEqual(calls[:2], [("reload", "U"), ("reload", "D")])  # before the SDK's ten minutes
            self.assertIn(("conn", "/time"), calls)
            store.close()


class WhoMovesFirstAtOurSpeedTest(unittest.TestCase):
    def test_once_orders_are_timed_the_jump_is_priced_at_our_speed_too(self):
        from polybot.lagwatch import LagWatch
        W = LagWatch()
        W.fast_delay = 0.2
        t0 = 1000.0
        # A coin's chance jumps; the other coin's book: cheap until 0.3 s after, dear from then on.
        for i in range(160):
            ts = t0 + i * 0.1
            fair = 0.5 if ts < t0 + 2 else 0.6
            W.observe("btc", "btc", 5, ts, fair, 0.49, 0.51, 0.49, 0.51, 0.07, 1.0)
            late = ts >= t0 + 2.3
            W.observe("eth", "eth", 5, ts, 0.5, 0.59 if late else 0.49, 0.61 if late else 0.51, 0.39 if late else 0.49,
                      0.41 if late else 0.51, 0.07, 1.0)
        cross = [e for e in W.events() if e["kind"] == "cross"]
        self.assertEqual(len(cross), 1)
        e = cross[0]
        self.assertEqual(e["fast_delay"], 0.2)
        self.assertGreater(e["mark_fast"], 0.05)  # in at 51c at our speed: worth ~60c later
        self.assertLess(e["mark_mid"], e["mark_fast"])  # (half a second: in at the dearer price)
        g = next(x for x in W.summary()["groups"] if x["kind"] == "cross")
        self.assertEqual(g["fast_n"], 1)
        self.assertEqual(W.summary()["settings"]["fast_delay"], 0.2)


class StocksBacktestTest(unittest.TestCase):
    """The stocks area: indicators, the market's picture, and the backtest's money rules."""

    @staticmethod
    def _bars(n, start=100.0, step=0.5, day0=dt.date(2025, 1, 1)):
        out, d, price = [], day0, start
        while len(out) < n:
            if d.weekday() < 5:
                out.append((d.isoformat(), price, price + 1, price - 1, price + 0.2, 1_000_000))
                price += step
            d += dt.timedelta(days=1)
        return out

    def test_indicators(self):
        from polybot import equity
        self.assertEqual(equity.sma([1, 2, 3, 4], 2), [None, 1.5, 2.5, 3.5])
        r = equity.rsi([1, 2, 3, 4, 5, 6], 2)
        self.assertEqual(r[-1], 100.0)  # only gains
        p = equity.prepare(self._bars(260))
        self.assertIsNotNone(p["sma200"][-1])
        self.assertAlmostEqual(p["atr"][-1], 2.0, delta=0.6)

    def test_the_money_rules(self):
        from unittest import mock
        from polybot import equity
        spy = self._bars(300)
        prepared = {"SPY": equity.prepare(spy)}
        # A stock: flat at 100, then a signal day; the next day opens at 100 and runs to 130.
        bars = [(d, 100.0, 101.0, 99.0, 100.0, 1e6) for d, *_ in spy[:250]]
        days = [d for d, *_ in spy[250:]]
        bars += [(days[0], 100.0, 101.0, 99.0, 100.0, 1e6)]
        price = 100.0
        for d in days[1:]:
            bars.append((d, price, price + 3, price - 0.5, price + 2.5, 1e6))
            price += 2.5
        prepared["AAA"] = equity.prepare(bars)
        sig_day = days[0]
        with mock.patch.object(equity, "signal", side_effect=lambda kind, p, i, P: 96.0 if p["day"][i] == sig_day and p is prepared["AAA"] else None):
            r = equity.backtest(prepared, "pullback", {"regime": "off", "rr": 3.0, "slippage": 0.0, "entry": "open"}, symbols=("AAA",))
        t = r["trades"][0]
        self.assertEqual(t["in"], days[1])  # the next day's open
        self.assertAlmostEqual(t["entry"], 100.0)
        # Risk $4 a share; 1% of $100K = $1,000 -> 250 shares, but at most $10K -> 100 shares.
        self.assertEqual(t["shares"], 100)
        self.assertEqual(t["why"], "target")  # 3R = $12 above: 112
        self.assertAlmostEqual(t["exit"], 112.0)
        self.assertAlmostEqual(t["r"], 3.0, delta=0.1)

    def test_a_winner_that_turns_never_loses(self):
        from unittest import mock
        from polybot import equity
        spy = self._bars(300)
        prepared = {"SPY": equity.prepare(spy)}
        days = [d for d, *_ in spy[250:]]
        bars = [(d, 100.0, 101.0, 99.0, 100.0, 1e6) for d, *_ in spy[:250]]
        bars.append((days[0], 100.0, 101.0, 99.0, 100.0, 1e6))
        bars.append((days[1], 100.0, 105.0, 99.5, 104.0, 1e6))  # +1.25R: the stop goes to the entry
        bars.append((days[2], 103.0, 103.5, 90.0, 91.0, 1e6))  # then a fall through it
        bars += [(d, 91.0, 92.0, 90.0, 91.0, 1e6) for d in days[3:]]
        prepared["AAA"] = equity.prepare(bars)
        with mock.patch.object(equity, "signal", side_effect=lambda kind, p, i, P: 96.0 if p["day"][i] == days[0] and p is prepared["AAA"] else None):
            r = equity.backtest(prepared, "pullback", {"regime": "off", "slippage": 0.0, "trail_atr": 0, "entry": "open"}, symbols=("AAA",))
        t = r["trades"][0]
        self.assertEqual(t["why"], "breakeven")
        self.assertAlmostEqual(t["exit"], 100.0)  # out at the entry, not at the $96 stop
        self.assertLess(abs(t["pnl"]), 3)  # (only the commissions)

    def test_the_picture_and_the_page(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            store = equity.EquityStore(os.path.join(tmp, "equity.sqlite"))
            fake = lambda sym, rng: self._bars(300, start=20 if sym == "^VIX" else 100, step=0.0 if sym == "^VIX" else 0.5)
            eq = equity.Equities(store, fetch=fake)
            with mock_sleep():
                eq.refresh()
            self.assertEqual(eq.status["symbols"], len(equity.UNIVERSE) + len(equity.BENCH))
            V = eq.view()
            P = V["picture"]
            self.assertEqual(P["primary"], "up")
            self.assertEqual(P["fear"], "normal")
            self.assertGreater(P["breadth"], 0.9)
            self.assertEqual(len(V["strategies"]), len(equity.STRATEGIES))  # (4, and the 3 entering mid-session)
            self.assertTrue(eq.data_ok())
            out = eq.backtest("breakout", {"risk_pct": 0.5, "stop_atr": 3, "vol_mult": 1.2, "regime": "both",  # (its own parameter too)
                                           "max_open": 10.0, "max_hold": 15.0})  # (counts as the page sends them)
            self.assertEqual(eq.backtest("fib", {"swing_days": 30.0})["params"]["swing_days"], 30)
            self.assertEqual(out["params"]["vol_mult"], 1.2)
            self.assertIn("stage", out)
            with self.assertRaises(ValueError):
                eq.backtest("nope")
            store.close()


import contextlib


@contextlib.contextmanager
def mock_sleep():
    from unittest import mock
    with mock.patch("polybot.equity.time.sleep"):
        yield


class StocksKeepToThemselvesTest(unittest.TestCase):
    def test_the_stocks_data_lives_in_its_own_file(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            main = Store(os.path.join(tmp, "polybot.sqlite"))
            # (as the first version left it: the bars inside Polymarket's database)
            main._x("CREATE TABLE eq_daily (symbol TEXT, day TEXT, o REAL, h REAL, l REAL, c REAL, v REAL, PRIMARY KEY (symbol, day))")
            main._x("INSERT INTO eq_daily VALUES ('AAPL', '2026-10-23', 1, 2, 0.5, 1.5, 100)")
            main.close()
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "polybot.sqlite")), api=FakeApi(), env={},
                            streaming=False, with_lab=False)
            self.assertTrue(os.path.exists(os.path.join(tmp, "equity.sqlite")))
            self.assertEqual(engine.equities.store._x("SELECT COUNT(*) FROM eq_daily")[0][0], 1)  # moved
            self.assertFalse(engine.store._x("SELECT name FROM sqlite_master WHERE name = 'eq_daily'"))  # and gone from there
            engine.equities.store.add_bars([("MSFT", "2026-10-23", 1, 2, 0.5, 1.5, 100)])
            self.assertFalse(engine.store._x("SELECT name FROM sqlite_master WHERE name = 'eq_daily'"))  # never back
            engine.store.close()


class AtlasTest(unittest.TestCase):
    """Atlas, the stocks bot: its simulation, its brain, its news, its settings and its calendar's backtest."""

    def _atlas(self, tmp, n=320):
        from polybot import atlas, equity
        import random as _r
        rnd = _r.Random(3)

        def fake(sym, rng):
            out, d, p = [], dt.date(2025, 1, 1), 100.0
            while len(out) < n:
                if d.weekday() < 5:
                    o = p * (1 + (rnd.uniform(0.04, 0.08) if rnd.random() < 0.02 else rnd.gauss(0, 0.004)))
                    p = max(5.0, o * (1 + rnd.gauss(0.0008, 0.015)))
                    out.append((d.isoformat(), o, max(o, p) * 1.006, min(o, p) * 0.994, p, rnd.randint(500_000, 2_000_000)))
                d += dt.timedelta(days=1)
            return out
        eq = equity.Equities(equity.EquityStore(os.path.join(tmp, "equity.sqlite")), fetch=fake)
        with mock_sleep():
            eq.refresh()
        return atlas.Atlas(eq), eq

    def test_the_simulation_trades_forward_and_keeps_its_book(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = self._atlas(tmp)
            spy = eq.prepared["SPY"]
            # Started on an older day (as if switched on then), it catches up a day at a time.
            book = __import__("polybot.equity", fromlist=["new_book"]).new_book(100_000)
            book.update(started=spy["day"][250], last=spy["day"][250])
            A._set("paper", book)
            A.paper_step()
            book = A.paper_book()
            self.assertEqual(book["last"], spy["day"][-1])
            self.assertTrue(book["trades"] or book["open"])
            v = A.paper_view()
            self.assertIn("today", v)
            self.assertAlmostEqual(v["value"], book["curve"][-1][1], delta=1)
            # Real money's rules hold: no trade risks more than 1% of the account, none is over $10K.
            for t in book["trades"]:
                self.assertLessEqual(t["shares"] * t["entry"], 10_000 + 1)
            A.paper_reset()
            self.assertEqual(A.paper_book()["started"], spy["day"][-1])  # (a fresh start, from today)

    def test_the_brain_keeps_only_what_holds_on_new_data(self):
        import random as _r
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = self._atlas(tmp)
            rnd = _r.Random(1)
            recs = [A.learn_once("pullback", rnd=rnd) for _ in range(12)]
            b = A.brain()["pullback"]
            self.assertEqual(b["tried"], 12)
            for r in recs:
                if r["kept"]:
                    self.assertGreater(r["new"], 0)  # never kept without making money on the newer data
            self.assertEqual(b["kept"] if "kept" in b else 0, sum(1 for r in recs if r["kept"]))
            plans = A.plans(["pullback"])
            for k, v in (b.get("params") or {}).items():
                self.assertEqual(plans["pullback"][k], v)  # the simulation uses what was learned

    def test_settings_news_and_the_calendar(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = self._atlas(tmp)
            with self.assertRaises(ValueError):
                A.set_settings({"risk_pct": 3})  # over the owner's 2%
            A.set_settings({"risk_pct": 0.5, "strategies": {"gap": False}})
            self.assertEqual(A.port()["risk_pct"], 0.5)
            self.assertNotIn("gap", A.plans())
            feed = lambda url: [{"title": "Fed holds rates steady", "link": url + "/1", "ts": time.time()},
                                {"title": "A new phone", "link": url + "/2", "ts": time.time() - 60}]
            news = A.refresh_news(fetch=feed)
            self.assertTrue(any(n["important"] for n in news))
            self.assertTrue(any(not n["important"] for n in news if n["source"] != "הפד"))
            out = A.backtest(["pullback", "breakout"])
            self.assertTrue({t["strategy"] for t in out["trades"]} <= {"pullback", "breakout"})
            custom = A.backtest(["breakout"], "custom", {"risk_pct": 1, "rr": 4, "max_open": 5.0})
            self.assertEqual(custom["plans"]["breakout"]["rr"], 4)
            V = A.view()
            for k in ("picture", "indices", "technical", "outlook", "spy_chart", "strategies", "paper", "news"):
                self.assertIn(k, V)
            self.assertEqual({x["symbol"] for x in V["indices"]}, set(__import__("polybot.equity", fromlist=["BENCH"]).BENCH))
            self.assertIsNotNone(V["outlook"]["d20"]["n"])


class StocksDataTrustTest(unittest.TestCase):
    """The data the backtest and the simulation stand on: only finished sessions, and a history that follows
    the source when it changes (a split)."""

    def test_only_finished_sessions(self):
        from polybot import equity
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        at = lambda h, m: dt.datetime(2026, 10, 23, h, m, tzinfo=ny).timestamp()
        self.assertFalse(equity.completed("2026-10-23", "AAPL", now=at(11, 0)))  # the session is still on
        self.assertTrue(equity.completed("2026-10-22", "AAPL", now=at(11, 0)))
        self.assertTrue(equity.completed("2026-10-23", "AAPL", now=at(16, 30)))  # closed
        self.assertFalse(equity.completed("2026-10-23", "BTC-USD", now=at(18, 0)))  # a coin's day ends at midnight UTC
        self.assertTrue(equity.completed("2026-10-22", "BTC-USD", now=at(18, 0)))

    def test_a_split_brings_the_whole_history_again(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            store = equity.EquityStore(os.path.join(tmp, "equity.sqlite"))
            days = [f"2026-09-{d:02d}" for d in range(1, 29)]
            split = {"on": False}

            def fake(sym, rng):
                k = 0.25 if split["on"] else 1.0  # a 4-for-1 split: the source divides all the past by 4
                rows = [(d, 400 * k, 404 * k, 396 * k, 400 * k, 1e6) for d in days]
                return rows if rng != "1mo" else rows[-10:]
            eq = equity.Equities(store, fetch=fake)
            with mock_sleep():
                eq.refresh()
            split["on"] = True
            with mock_sleep():
                eq.refresh()
            closes = {c for (c,) in store._x("SELECT DISTINCT c FROM eq_daily WHERE symbol = 'AAPL'")}
            self.assertEqual(closes, {100.0})  # no stale pre-split prices left: no fake 75% crash in the history
            self.assertIn("AAPL", eq.status.get("rewritten", []))
            store.close()


class ProfitByEntryPriceTest(unittest.TestCase):
    """The owner's table: every settled trade (a strategy's round, all its buys) by the price a share cost."""

    def test_trades_fall_in_their_price_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={})
            db = eng.store
            rows = [  # (variant, window, shares, cost, pnl): a 3-cent ticket that won, a 95-cent buy that won $1...
                ("a", 1, 100, 3.0, 96.0), ("a", 2, 10, 9.5, 0.4), ("b", 2, 10, 9.0, -9.2),
                ("a", 3, 10, 4.0, None),  # (not settled: left out)
            ]
            for v, w, sh, cost, pnl in rows:
                db._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl)"
                      " VALUES ('paper', ?, 'filled', 1, 'btc', 5, ?, 'up', ?, ?, 0.1, ?)", (v, w, sh, cost, pnl))
            # A second buy in the same round joins its trade: (9.5 + 0.5) / 20 shares = 50 cents.
            db._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl)"
                  " VALUES ('paper', 'a', 'filled', 1, 'btc', 5, 2, 'down', 10, 0.5, 0.1, 9.4)")
            r = eng.price_groups("sim")
            by = {(round(g["lo"], 2)): g for g in r["groups"]}
            self.assertEqual(r["trades"], 3)
            self.assertEqual(by[0.0]["trades"], 1)
            self.assertAlmostEqual(by[0.0]["pnl"], 96.0)
            self.assertEqual(by[0.5]["trades"], 1)
            self.assertAlmostEqual(by[0.5]["pnl"], 9.8)
            self.assertEqual(by[0.9]["trades"], 1)
            self.assertEqual(by[0.9]["wins"], 0)
            self.assertEqual(eng.price_groups("live")["trades"], 0)
            with self.assertRaises(ValueError):
                eng.price_groups("x")


class RealMakerOrdersTest(unittest.TestCase):
    """Real money's maker strategies: a real post-only order rests on Polymarket, is checked, moved,
    recorded when sellers take it, and cancelled when its time is over."""

    def _setup(self, tmp):
        from unittest import mock
        engine, t = AddToATradeTest._engine(AddToATradeTest(), tmp)
        book = {"orders": {}, "placed": [], "cancelled": []}

        class Clob:
            def place_limit_order(self, **kw):
                oid = f"o{len(book['placed']) + 1}"
                book["placed"].append(kw)
                book["orders"][oid] = {"matched": 0.0, "status": "live"}
                return types.SimpleNamespace(ok=True, order_id=oid, status="live")

            def get_order(self, order_id):
                o = book["orders"][order_id]
                return types.SimpleNamespace(size_matched=o["matched"], status=o["status"])

            def cancel_order(self, order_id):
                book["cancelled"].append(order_id)
                book["orders"][order_id]["status"] = "canceled"
        engine.live.client = Clob()
        patches = [mock.patch.object(engine.live, "trades", return_value=True),
                   mock.patch.object(engine.live, "approval", return_value=False)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        t.market["min_order_size"] = 5
        return engine, t, book

    def test_it_rests_fills_and_records_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t, book = self._setup(tmp)
            v = lab.Variant.from_key("fav60|0.70|180-300")
            t.live_rules = [v]
            cal, raw = {"twap": 0.9}, {"twap": 0.9}  # (the model agrees with the leader)
            now = time.time()
            t._live_rules_run(now, 250, cal, raw, [(0.66, 100)], [(0.36, 100)], 0, 0, cal, cal)
            self.assertEqual(len(book["placed"]), 1)
            order = book["placed"][0]
            self.assertEqual((order["side"], order["price"], order["post_only"]), ("BUY", "0.65", True))
            self.assertEqual(order["token_id"], t.market["up_token"])
            # Nobody sold yet: it rests; nothing recorded.
            t._live_rules_run(now + 3, 247, cal, raw, [(0.66, 100)], [(0.36, 100)], 0, 0, cal, cal)
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live'")[0][0], 0)
            # Sellers took 7 shares: the rest is cancelled, the 7 recorded at our price, no fee.
            book["orders"]["o1"]["matched"] = 7.0
            t._live_rules_run(now + 6, 244, cal, raw, [(0.66, 100)], [(0.36, 100)], 0, 0, cal, cal)
            self.assertEqual(book["cancelled"], ["o1"])
            row = engine.store._x("SELECT mode, status, variant, side, shares, avg_price, fee, order_id FROM bets")[0]
            self.assertEqual(tuple(row), ("live", "filled", v.key, "up", 7.0, 0.65, 0.0, "o1"))
            # Once a round.
            t._live_rules_run(now + 9, 241, cal, raw, [(0.66, 100)], [(0.36, 100)], 0, 0, cal, cal)
            self.assertEqual(len(book["placed"]), 1)
            engine.store.close()

    def test_moved_and_cancelled_when_its_time_is_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t, book = self._setup(tmp)
            v = lab.Variant.from_key("twapm|0.05|15-600")
            t.live_rules = [v]
            raw = {"twap": 0.7}
            now = time.time()
            # Up: the model's 70%, offered at 60c: it rests a tick under the offer (59c).
            t._live_rules_run(now, 300, {"twap": 0.7}, raw, [(0.60, 100)], [(0.42, 100)], 0, 0, {}, {})
            self.assertEqual((book["placed"][0]["price"], book["placed"][0]["token_id"]), ("0.59", t.market["up_token"]))
            # The model's chance fell to 60%: the order no longer has its edge; cancelled, and placed again
            # where it has it (Down now: 40%, offered at 30c, so 29c).
            t._live_rules_run(now + 1, 299, {"twap": 0.6}, raw, [(0.60, 100)], [(0.30, 100)], 0, 0, {}, {})
            self.assertEqual(book["cancelled"], ["o1"])
            t._live_rules_run(now + 2, 298, {"twap": 0.6}, raw, [(0.60, 100)], [(0.30, 100)], 0, 0, {}, {})
            self.assertEqual((book["placed"][1]["price"], book["placed"][1]["token_id"]), ("0.29", t.market["down_token"]))
            # Past its time: cancelled; nothing was taken, nothing recorded.
            t._live_rules_run(now + 3, 10, {"twap": 0.6}, raw, [(0.60, 100)], [(0.30, 100)], 0, 0, {}, {})
            self.assertEqual(book["cancelled"], ["o1", "o2"])
            self.assertEqual(engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live'")[0][0], 0)
            engine.store.close()

    def test_real_money_off_nothing_rests(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t, book = self._setup(tmp)
            v = lab.Variant.from_key("fav60|0.70|180-300")
            t.live_rules = [v]
            cal = {"twap": 0.9}
            t._live_rules_run(time.time(), 250, cal, cal, [(0.66, 100)], [(0.36, 100)], 0, 0, cal, cal)
            with mock.patch.object(engine.live, "trades", return_value=False):
                t._live_rules_run(time.time(), 249, cal, cal, [(0.66, 100)], [(0.36, 100)], 0, 0, cal, cal)
            self.assertEqual(book["cancelled"], ["o1"])
            engine.store.close()


class TheStocksLabsBrainTest(unittest.TestCase):
    """The stock lab's stages, as the crypto lab's: promising on what it has, then tested on new trades
    only, proven (and pushed into the simulation automatically, when that's on) or failed."""

    def test_stages_and_the_automatic_push(self):
        from polybot import stocks
        with tempfile.TemporaryDirectory() as tmp:
            eng = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                         streaming=False)
            S = eng.stocks
            key = "fav|0.70|900"
            add = lambda ts, pnl, n=1: [eng.store._x(
                "INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl)"
                " VALUES ('stocks', ?, 'filled', ?, 'aapl', 0, 1, 'up', 10, 8.0, 0, ?)", (key, ts + i * 0.001, pnl))
                for i in range(n)]
            add(100, 2.0, 40)  # 40 bets, each +25%: promising, so the test on new trades starts
            db = connect(eng.cfg.db_path)
            b = {r["variant"]: r for r in S.board(db, now=200)}[key]
            self.assertEqual(b["status"], "candidate")
            self.assertEqual(b["oos_bets"], 0)
            add(300, 2.0, eng.cfg.lab_oos_min_bets)  # the new trades hold up
            board = S.board(db, now=400)
            b = {r["variant"]: r for r in board}[key]
            self.assertEqual(b["status"], "proven")
            self.assertEqual(board[0]["variant"], key)
            self.assertIsNone(S.main())
            eng._stocks_auto_push(board)  # automatic push off: nothing changes
            self.assertIsNone(S.main())
            eng.set_auto_push(True)
            eng._stocks_auto_push(board)
            self.assertEqual(S.main()["variant"], key)
            self.assertEqual(S.recommendation(board)["verdict"], "current")
            db.close()
            eng.store.close()

    def test_a_card_charts_the_price_from_when_its_market_is_followed(self):
        from polybot import stocks
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            S = stocks.StockLab(Config(), store)
            close = 1_800_000_000.0
            for k in range(0, 3 * 3600, 5):  # a price every 5 seconds for the three hours before the close
                S.on_price("aapl", close - 3 * 3600 + k, 340 + k / 1000)
            c = S.card("lab", "aapl-up-or-down-on-october-26-2026", "aapl", close, close - 4600, lambda tok: None)
            self.assertGreater(len(c["series"]), 100)  # an hour and a quarter before the close: a chart already
            self.assertEqual(c["span"], stocks.WATCH_SECONDS)
            self.assertLess(c["series"][0][0], 60)  # (from the start of the window)
            store.close()

    def test_the_grid_is_wider(self):
        from polybot import stocks
        kinds = {(v.kind, v.level) for v in stocks.VARIANTS}
        self.assertIn(("fav", 0.9), kinds)
        self.assertIn(("model", 0.12), kinds)
        self.assertEqual(len(stocks.VARIANTS), 32)


class ThePanelsGateTest(unittest.TestCase):
    """The expert panels' rules (27.10.2026): a test that knows what each trade had to win, at fixed
    checkpoints only; only proven strategies into the simulation by themselves; the end-of-round
    favourite judged over every coin together."""

    def test_a_streak_at_97c_proves_little(self):
        # 100 wins in 100 trades at 97c (break-even 97.2%): about 1.7 standard errors, not 3.
        z = lab.price_z([(1.0, 0.972)] * 100)
        self.assertLess(z, 3.0)
        self.assertGreater(lab.price_z([(1.0, 0.5)] * 75 + [(0.0, 0.5)] * 25), 3.0)  # 75% at 50c: a real lead
        self.assertEqual(lab.checkpoints(450), [100, 200, 400])
        self.assertEqual(lab.next_checkpoint(450), 800)
        lo, up = lab.win_bounds(400, 400)
        self.assertGreater(lo, 0.99)
        self.assertEqual(up, 1.0)

    def test_the_end_of_round_favourite(self):
        cfg = Config()
        mine = [v for v in lab.variants_for(5) if lab.is_end(v)]
        self.assertEqual(len(mine), 12)
        self.assertEqual({lab.family(v.key) for v in mine}, {"end"})
        self.assertTrue(all(v.max_seconds_left <= 60 for v in mine))
        v = lab.Variant.from_key("end|0.00|15-30|b95-100")
        fill = Tracker._decide(v, 0.5, [(0.97, 500)], [(0.04, 500)], cfg)  # no model asked
        self.assertEqual((fill.side, round(fill.avg_price, 2)), ("up", 0.97))
        self.assertIsNone(Tracker._decide(v, 0.5, [(0.93, 500)], [(0.08, 500)], cfg))  # not in its group
        self.assertIsNone(Tracker._decide(v, 0.5, [(0.97, 2)], [(0.04, 500)], cfg))  # the whole $5 or nothing
        with_model = lab.Variant.from_key("endm|0.00|5-30|b95-100")
        self.assertIsNone(Tracker._decide(with_model, 0.90, [(0.97, 500)], [(0.04, 500)], cfg))  # the model: 90%
        self.assertEqual(Tracker._decide(with_model, 0.995, [(0.97, 500)], [(0.04, 500)], cfg).side, "up")

    def test_judged_over_every_coin_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = Store(os.path.join(tmp, "t.sqlite"))
            key = "end|0.00|15-30|b95-100"
            for i in range(400):  # 400 trades on four coins at 97c, every one won; and 40 orders the book refused
                coin = ("btc", "eth", "sol", "xrp")[i % 4]
                st._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl)"
                      " VALUES ('lab', ?, 'filled', ?, ?, 5, ?, 'up', 5.15, 5.0, 0.0, 0.15)", (key, i, coin, i * 300))
            for i in range(40):
                st._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                      " VALUES ('lab', ?, 'refused', ?, 'btc', 5, ?, 'up', 5.15, 5.0, 0.0)", (key, 1000 + i, i * 300))
            db = connect(os.path.join(tmp, "t.sqlite"))
            row = lab.end_pooled(db)[0]
            self.assertEqual((row["trades"], row["wins"], row["tried"], row["filled"]), (400, 400, 440, 400))
            self.assertAlmostEqual(row["break_even"], 5.0 / 5.15)
            self.assertEqual(row["verdict"], "proven")  # 400 of 400: the lower bound over 97.1%
            db.close()
            st.close()

    def test_only_proven_strategies_go_in_by_themselves(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = Store(os.path.join(tmp, "t.sqlite"))
            st.set_setting("lab_auto_candidates", "1")  # (an installation from before)
            st.set_setting("rr_auto", json.dumps({"sim": True}))
            st.close()
            eng = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                         streaming=False, with_lab=False)
            self.assertFalse(eng.auto_candidates())
            self.assertFalse(eng.rr_auto()["sim"])
            eng.set_auto_candidates(True)
            eng.store.close()
            again = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                           streaming=False, with_lab=False)
            self.assertTrue(again.auto_candidates())  # once: the owner's later choice stays
            again.store.close()


class TheStrategiesOwnFiltersTest(unittest.TestCase):
    """No layer of filters over the strategies: a strategy carries its own price group and least
    chance, tested in the lab like the rest of it, and the same wherever it runs."""

    def test_a_rule_with_its_own_filters(self):
        v = lab.Variant.from_key("twap|0.02|5-120|b70-80|c85")
        self.assertEqual((v.price_lo, v.price_hi, v.min_chance, v.max_seconds_left), (0.7, 0.8, 0.85, 120))
        self.assertEqual(v.key, "twap|0.02|5-120|b70-80|c85")
        self.assertTrue(v.passes(0.75, 0.9))
        self.assertFalse(v.passes(0.80, 0.9))   # (the group's top is out)
        self.assertFalse(v.passes(0.75, 0.84))  # under its least chance
        self.assertIn("70¢–80¢", lab.describe(v.key))
        self.assertIn("85%", lab.describe(v.key))
        # The owner's: on every market, the last two minutes, the group and the chance (one rule: its
        # neighbours only multiplied the tests).
        for minutes in (5, 15):
            mine = [x for x in lab.variants_for(minutes) if x.price_hi and x.resolution == "twap"]
            self.assertEqual(len(mine), 1)
            self.assertIn("twap|0.02|5-120|b70-80|c85", {x.key for x in mine})
            self.assertTrue(all(x.max_seconds_left <= 120 for x in mine))
        # Its neighbours keep its filters; no risk/reward versions on top of a price group.
        self.assertTrue(all("|b70-80|c85" in k for k in lab.neighbours(v.key, 5)))
        self.assertEqual(lab.ratio_versions(v.key), [])

    def test_the_rule_decides_with_them(self):
        cfg = Config()
        v = lab.Variant.from_key("twap|0.02|5-120|b70-80|c85")
        # Up at 75c with our chance at 90%: in its group and over its least chance.
        fill = Tracker._decide(v, 0.90, [(0.75, 500)], [(0.20, 500)], cfg)
        self.assertEqual((fill.side, round(fill.avg_price, 2)), ("up", 0.75))
        # Our chance 82%: under 85%, no trade.
        self.assertIsNone(Tracker._decide(v, 0.82, [(0.75, 500)], [(0.20, 500)], cfg))
        # Up at 2c (our chance 15%) is the better bet for a rule without filters; this one takes
        # Down at 75c (85%), the side in its group and over its least chance.
        plain = lab.Variant.from_key("twap|0.02|5-120")
        self.assertEqual(Tracker._decide(plain, 0.15, [(0.02, 500)], [(0.75, 500)], cfg).side, "up")
        self.assertEqual(Tracker._decide(v, 0.15, [(0.02, 500)], [(0.75, 500)], cfg).side, "down")

    def test_what_the_filters_left_goes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = Store(os.path.join(tmp, "t.sqlite"))
            for k, v in (("price_band", '{"lo": 0.7, "hi": 0.8}'), ("min_chance_sim", "0.85"), ("max_loss_live", "20"),
                         ("model_agree", "0.1"), ("brain_band_sim", "auto")):
                st.set_setting(k, v)
            st.close()
            eng = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                         streaming=False, with_lab=False)
            for k in ("price_band", "min_chance_sim", "max_loss_live", "model_agree", "brain_band_sim"):
                self.assertIsNone(eng.store.get_setting(k))
            self.assertNotIn("price_band", eng.state())
            eng.store.close()

    def test_the_owners_amounts_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = Store(os.path.join(tmp, "t.sqlite"))  # (an installation that traded already)
            st._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                  " VALUES ('paper', 'a', 'filled', 1, 'btc', 5, 1, 'up', 10, 5, 0)")
            st.close()
            eng = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                         streaming=False, with_lab=False)
            M = eng.money()
            self.assertEqual((M["sim"]["stake"]["value"], M["live"]["stake"]["value"]), (5.0, 5.0))
            eng.set_money("sim", {"stake": 7})
            eng.store.close()
            again = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                           streaming=False, with_lab=False)
            self.assertEqual(again.money()["sim"]["stake"]["value"], 7.0)  # once: the owner's later change stays
            again.store.close()

    def test_one_amount_for_every_trade(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                         streaming=False, with_lab=False)
            m = {"asset": "btc", "minutes": 15, "variant": "twap|0.05|15-600", "stake": 166.67, "stake_src": "system"}
            self.assertTrue(eng.uniform_stake("sim"))
            self.assertEqual(eng._strategy_stake("sim", m), 0.0)  # (0: the simulation's one amount)
            eng.set_uniform_stake("sim", False)
            self.assertGreater(eng._strategy_stake("sim", m), 0.0)
            eng.store.close()

    def test_groups_by_our_chance_and_the_least_chance(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = AddToATradeTest._engine(AddToATradeTest(), tmp)
            ins = lambda mode, ws, fair, cost, pnl: engine.store._x(
                "INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl, fair_prob)"
                " VALUES (?, 'a', 'filled', 1, 'btc', 5, ?, 'up', 10, ?, 0, ?, ?)", (mode, ws, cost, pnl, fair))
            ins("lab", 1, 0.34, 2.5, -2.5)
            ins("lab", 2, 0.97, 9.4, 0.6)
            ins("lab", 3, 0.96, 9.5, 0.5)
            r = engine.price_groups("lab", "chance")
            by = {round(g["lo"], 2): g for g in r["groups"] if g["trades"]}
            self.assertEqual(set(by), {0.0, 0.95})
            self.assertEqual((by[0.95]["trades"], by[0.95]["wins"]), (2, 2))
            self.assertAlmostEqual(by[0.95]["avg_price"], 0.945)
            engine.store.close()

class EnteringWhenThePriceSaysSoTest(unittest.TestCase):
    """Not at a fixed moment: each strategy's entry order the day after its signal (a buy-stop over the
    signal day's high, a limit under its close, or the open), the same in the backtest, during the day
    and at IBKR."""

    def test_where_an_order_fills(self):
        from polybot import equity
        self.assertEqual(equity.fill_price(None, 100, 105, 95), 100)  # the open
        self.assertEqual(equity.fill_price(["stop", 102], 100, 105, 95), 102)  # reached during the day
        self.assertEqual(equity.fill_price(["stop", 102], 103, 105, 101), 103)  # gapped over: at the open
        self.assertIsNone(equity.fill_price(["stop", 106], 100, 105, 95))  # never confirmed: no trade
        self.assertEqual(equity.fill_price(["limit", 97], 100, 105, 95), 97)
        self.assertEqual(equity.fill_price(["limit", 97], 96, 105, 95), 96)  # opened under it
        self.assertIsNone(equity.fill_price(["limit", 94], 100, 105, 95))
        self.assertEqual(equity.plan_for("breakout")["entry"], "stop")
        self.assertEqual(equity.plan_for("pullback")["entry"], "limit")

    def test_a_stop_never_farther_than_its_cap(self):
        from polybot import equity
        book = equity.new_book(100_000)
        book["pending"] = [("fib", "CRWD", 200.0, None, None)]  # the strategy's stop: 23% under the open
        P = equity.plan_for("fib", {"slippage": 0.0})
        t = equity.enter_pending(book, "2026-10-26", {"fib": P}, P, lambda s: (258.66, 260.0, 255.0))[0]
        self.assertAlmostEqual(t["stop"], 258.66 * 0.9)  # 10% at most
        self.assertAlmostEqual(t["target"], 258.66 + 3 * 25.866, places=2)  # (3 times the risk, from the capped stop)
        book["pending"] = [("fib", "CRWD", 200.0, None, None)]
        P0 = equity.plan_for("fib", {"slippage": 0.0, "max_stop_pct": 0})
        book["open"] = []
        self.assertEqual(equity.enter_pending(book, "2026-10-26", {"fib": P0}, P0, lambda s: (258.66, 260.0, 255.0))[0]["stop"], 200.0)

    def test_entered_during_the_day_as_the_close_would(self):
        from polybot import equity
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            book = equity.new_book(100_000)
            book.update(last="2026-10-23", curve=[("2026-10-23", 100_000)],
                        pending=[("breakout", "NVDA", 90.0, 110.0, ["stop", 101.0]), ("breakout", "AAPL", 90.0, 110.0, ["stop", 108.0])])
            A._set("paper", book)
            t0 = dt.datetime(2026, 10, 26, 9, 30, tzinfo=ny).timestamp()
            bars = {"NVDA": [(t0, 100.0, 102.5, 99.5, 102.0)], "AAPL": [(t0, 100.0, 102.5, 99.5, 102.0)]}
            fetch = lambda sym: {"price": bars[sym][-1][4], "ts": t0, "bars": bars[sym]}
            entered = A.intraday_entries(t0 + 3600, fetch=fetch)
            self.assertEqual([t["symbol"] for t in entered], ["NVDA"])  # through 101; AAPL not at 108 yet
            b = A.paper_book()
            self.assertEqual(b["open"][0]["day"], "2026-10-26")
            self.assertAlmostEqual(b["open"][0]["entry"], 101.0 * (1 + equity.DEFAULTS["slippage"]))
            self.assertEqual([x[1] for x in b["pending"]], ["AAPL"])  # (still waiting today)
            self.assertTrue(any(m["kind"] == "open" for m in A.peek_mails()))  # its mail goes now
            self.assertEqual(A.intraday_entries(t0 + 3700, fetch=fetch), [])  # once
            self.assertEqual(A.intraday_entries(dt.datetime(2026, 10, 26, 17, 0, tzinfo=ny).timestamp(), fetch=fetch), [])


class AtlasToIbkrPaperTest(unittest.TestCase):
    """Atlas' simulation as real orders in the owner's IBKR paper account (a fake gateway here)."""

    class Gateway:
        def __init__(self, accounts=("DU123456",)):
            self.accounts, self.host, self.port = list(accounts), "fake", 4006
            self.placed, self.state, self.held, self.modified, self.cancelled, self.sold = [], {}, {}, [], [], []
            self.next = 100

        def connect(self):
            return self.accounts

        def place_bracket(self, symbol, qty, limit, target, stop, entry=None, trail=None):
            ids = {"parent": self.next, "target": self.next + 1 if target else None, "stop": self.next + 2}
            self.next += 3
            self.placed.append((symbol, qty, limit, target, stop))
            self.trails = getattr(self, "trails", []) + [trail]
            for i in ids.values():
                if i:
                    self.state[i] = {"status": "Submitted", "filled": 0.0, "avg_price": 0.0}
            return ids

        def orders(self):
            return self.state

        def positions(self):
            return self.held

        def modify(self, oid, **kw):
            self.modified.append((oid, kw))
            return True

        def cancel(self, oid):
            self.cancelled.append(oid)
            return True

        def sell_at_open(self, symbol, qty):
            self.sold.append((symbol, qty))
            self.state[999] = {"status": "Submitted", "filled": 0.0, "avg_price": 0.0}
            return 999

    def test_signals_go_at_the_open_and_follow_the_simulation(self):
        from polybot import equity
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ny).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            book = equity.new_book(100_000)
            book.update(last="2026-10-26", pending=[("breakout", "NVDA", 100.0, 105.0)], curve=[("2026-10-26", 100_000)])
            A._set("paper", book)
            gw = self.Gateway()
            self.assertIsNone(A.ibkr_tick(evening, gw))  # switched off: nothing
            A.set_settings({"ibkr_on": True})
            A.ibkr_tick(evening, gw)
            sym, qty, limit, target, stop = gw.placed[0]
            self.assertEqual((sym, limit, stop), ("NVDA", 105.0, 100.0))
            self.assertEqual(qty, int(min(1000 / 5, 10_000 / 105)))  # 1% at risk, at most $10K, at the ceiling
            A.ibkr_tick(evening + 60, gw)
            self.assertEqual(len(gw.placed), 1)  # once a signal
            # The next morning it opened at 103: the target moves to the real entry's 3R... (rr of the plan)
            gw.state[100] = {"status": "Filled", "filled": qty, "avg_price": 103.0}
            morning = dt.datetime(2026, 10, 27, 9, 45, tzinfo=ny).timestamp()
            A.ibkr_tick(morning, gw)
            o = A._get("ibkr")["orders"]["breakout:NVDA:2026-10-26"]
            rr = A.plans(["breakout"])["breakout"]["rr"]
            self.assertEqual((o["status"], o["entry"], o["entry_day"]), ("open", 103.0, "2026-10-27"))
            self.assertIn((101, {"lmtPrice": round(103 + rr * 3, 2)}), gw.modified)
            # The stop is IBKR's own trailing stop, $5 under the best price (the signal's distance).
            self.assertEqual(gw.trails[0], 5.0)
            # The simulation's stop is closer to its best price ($3 under 106): so is IBKR's distance.
            book = A.paper_book()
            book["open"] = [{"id": "breakout:NVDA:2026-10-27", "stop": 103.0, "entry": 103.0, "best": 106.0}]
            A._set("paper", book)
            A.ibkr_tick(morning + 60, gw)
            self.assertIn((102, {"auxPrice": 3.0}), gw.modified)
            # The stop was hit: closed, its result recorded beside the simulation's.
            gw.state[102] = {"status": "Filled", "filled": qty, "avg_price": 103.0}
            A.ibkr_tick(morning + 120, gw)
            o = A._get("ibkr")["orders"]["breakout:NVDA:2026-10-26"]
            self.assertEqual((o["status"], o["why"]), ("closed", "breakeven"))
            self.assertEqual(A.ibkr_view()["rows"][0]["status"], "closed")

    def test_only_a_paper_account(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            book = equity.new_book(100_000)
            book.update(last="2026-10-26", pending=[("breakout", "NVDA", 100.0, 105.0)])
            A._set("paper", book)
            A.set_settings({"ibkr_on": True})
            gw = self.Gateway(accounts=("U765432",))  # a real-money account
            A.ibkr_tick(dt.datetime(2026, 10, 26, 21, 0, tzinfo=dt.timezone.utc).timestamp(), gw)
            self.assertEqual(gw.placed, [])
            self.assertIn("Paper", A.status["ibkr_error"])


class AtlasToIbkrRealMoneyTest(unittest.TestCase):
    """The owner's real IBKR account (28.10.2026): only after his confirmation, only when the gateway is
    logged into a real account, and only live_size dollars ($100) a trade."""

    def test_a_hundred_dollars_a_trade_in_the_real_account_only(self):
        from polybot import equity, ibkr
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ny).timestamp()
        self.assertIsNone(ibkr.account_problem(["U1234567"], "live"))
        self.assertIsNotNone(ibkr.account_problem(["DU123456"], "live"))
        self.assertIsNotNone(ibkr.account_problem(["U1234567"], "paper"))  # (a real account never on paper's settings)
        self.assertIsNone(ibkr.account_problem(["DU123456"], "paper"))
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            with self.assertRaises(ValueError):
                A.set_settings({"ibkr_mode": "live"})  # (not without the owner's confirmation)
            with self.assertRaises(ValueError):
                A.set_settings({"live_size": 500_000})  # (a typo's guard)
            A.set_settings({"ibkr_on": True, "ibkr_mode": "live", "confirm_live": True})
            book = equity.new_book(100_000)
            book.update(last="2026-10-26", pending=[("breakout", "NVDA", 30.0, 33.0), ("breakout", "BKNG", 4900.0, 5000.0)],
                        curve=[("2026-10-26", 100_000)])
            A._set("paper", book)
            paper_gw = AtlasToIbkrPaperTest.Gateway()
            A.ibkr_tick(evening, paper_gw)
            self.assertEqual(paper_gw.placed, [])  # real money picked, a paper login: nothing
            self.assertIn("Paper", A.status["ibkr_error"])
            gw = AtlasToIbkrPaperTest.Gateway(accounts=("U1234567",))
            A.ibkr_tick(evening, gw)
            self.assertEqual([(p[0], p[1]) for p in gw.placed], [("NVDA", int(100 // 33.0))])  # $100 worth, not the simulation's
            orders = A._get("ibkr")["orders"]
            self.assertEqual(orders["breakout:NVDA:2026-10-26"]["mode"], "live")
            self.assertEqual(orders["breakout:BKNG:2026-10-26"]["status"], "skipped")  # one share costs more than $100
            self.assertEqual(A.ibkr_view()["mode"], "live")
            # Back to paper: a real login sends nothing more.
            A.set_settings({"ibkr_mode": "paper"})
            book["pending"], book["last"] = [("breakout", "AMD", 20.0, 22.0)], "2026-10-27"
            A._set("paper", book)
            A.ibkr_tick(evening + 86400, gw)
            self.assertEqual(len(gw.placed), 1)


class AtlasModesHubTest(unittest.TestCase):
    """Atlas's own three modes (as Polymarket's): the simulation only, to IBKR on the owner's approval, or
    by itself; real money only with his confirmation."""

    def test_on_approval_a_signal_waits_for_the_owners_yes(self):
        from polybot import equity
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ny).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            book = equity.new_book(100_000)
            book.update(last="2026-10-26", pending=[("breakout", "NVDA", 100.0, 105.0), ("breakout", "AMD", 50.0, 52.0)],
                        curve=[("2026-10-26", 100_000)])
            A._set("paper", book)
            A.set_settings({"ibkr_on": True, "ibkr_auto": False})
            gw = AtlasToIbkrPaperTest.Gateway()
            A.ibkr = gw
            A.ibkr_tick(evening, gw)
            self.assertEqual(gw.placed, [])  # on approval: nothing by itself
            P = {x["symbol"]: x for x in A.ibkr_view()["proposals"]}
            self.assertEqual(set(P), {"NVDA", "AMD"})
            # (each as a row: the shares, the money in, what it can lose at the stop, 30.10.2026)
            N = P["NVDA"]
            self.assertEqual(N["est"], 105.0)
            self.assertGreater(N["qty"], 0)
            self.assertAlmostEqual(N["size"], N["qty"] * 105.0, places=2)
            self.assertAlmostEqual(N["max_loss"], (105.0 - N["stop"]) * N["qty"], places=2)
            self.assertFalse(A.ibkr_view()["auto"])
            A.decide_signal("breakout:NVDA:2026-10-26", True, now=evening)
            A.decide_signal("breakout:AMD:2026-10-26", False, now=evening)
            A.ibkr_tick(evening + 60, gw)
            self.assertEqual([p[0] for p in gw.placed], ["NVDA"])
            P = {x["symbol"]: x for x in A.ibkr_view()["proposals"]}
            self.assertTrue(P["NVDA"]["sent"])
            self.assertEqual(P["AMD"]["decision"], "no")
            with self.assertRaises(ValueError):
                A.decide_signal("breakout:TSLA:2026-10-26", True)
            # Real money: the connection on (or automatic) only with the owner's confirmation.
            A.set_settings({"ibkr_on": False, "ibkr_mode": "live", "confirm_live": True})
            with self.assertRaises(ValueError):
                A.set_settings({"ibkr_on": True})
            A.set_settings({"ibkr_on": True, "ibkr_auto": True, "confirm_live": True})
            self.assertTrue(A.settings()["ibkr_auto"])


class AtlasRealMoneyDayTest(unittest.TestCase):
    """A day of real money at IBKR, end to end (a fake gateway): the signal sent at the owner's size and
    target, the fill, the mails, the stop following the simulation's, the exit; every failure said with
    where; the simulation's own book untouched."""

    def test_a_real_money_day_end_to_end(self):
        import copy
        from polybot import atlas_mail, equity
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ny).timestamp()
        morning = dt.datetime(2026, 10, 27, 9, 45, tzinfo=ny).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            A.set_settings({"ibkr_on": True, "ibkr_mode": "live", "confirm_live": True, "live_size": 1000, "rr_override": 3})
            self.assertEqual(A.plans(["breakout"])["breakout"]["rr"], 3)  # the owner's target, for the simulation too
            self.assertNotEqual(A.plans(["breakout"], system=True)["breakout"]["rr"], None)
            book = equity.new_book(100_000)
            book.update(last="2026-10-26", pending=[("breakout", "NVDA", 95.0, 100.0), ("breakout", "AMD", 48.0, 50.0)],
                        curve=[("2026-10-26", 100_000)])
            A._set("paper", book)
            before = copy.deepcopy(A.paper_book())
            gw = AtlasToIbkrPaperTest.Gateway(accounts=("U1234567",))
            real_place = gw.place_bracket

            def place(symbol, *a, **k):
                if symbol == "AMD":
                    raise RuntimeError("Order rejected: no trading permissions")
                return real_place(symbol, *a, **k)
            gw.place_bracket = place
            A.ibkr_tick(evening, gw)
            sym, qty, limit, target, stop = gw.placed[0]
            self.assertEqual((sym, qty), ("NVDA", 10))  # $1,000 at the $100 ceiling
            self.assertEqual(target, 100 + 3 * 5)  # 1:3 from the owner
            alerts = A._get("ibkr_alerts")
            self.assertEqual((alerts[-1]["where"], alerts[-1]["symbol"]), ("שליחת הזמנת כניסה", "AMD"))
            self.assertIn("permissions", alerts[-1]["what"])
            A.ibkr_tick(evening + 60, gw)  # (the same failure again: counted, not said again)
            self.assertEqual(len(A._get("ibkr_alerts")), 1)
            self.assertEqual(A._get("ibkr_alerts")[-1]["n"], 2)
            self.assertEqual(len(A._get("alert_mails")), 1)
            # Filled the next morning at 101: open, the target from the real entry, the opening mail.
            gw.state[100] = {"status": "Filled", "filled": 10, "avg_price": 101.0}
            A.ibkr_tick(morning, gw)
            o = A._get("ibkr")["orders"]["breakout:NVDA:2026-10-26"]
            self.assertEqual((o["status"], o["entry"], o["risk_usd"]), ("open", 101.0, 60.0))
            self.assertIn((101, {"lmtPrice": round(101 + 3 * 6, 2)}), gw.modified)
            mails = [m for m in A.peek_mails() if m["mode"] == "live"]
            self.assertEqual([(m["kind"], m["trade"]["symbol"], m["trade"]["shares"]) for m in mails], [("open", "NVDA", 10)])
            # The simulation's stop rose: IBKR's trailing distance follows.
            b = A.paper_book()
            b["open"] = [{"id": "breakout:NVDA:2026-10-27", "stop": 104.0, "entry": 101.0, "best": 107.0}]
            A._set("paper", b)
            A.ibkr_tick(morning + 60, gw)
            self.assertIn((102, {"auxPrice": 3.0}), gw.modified)
            # The trailing stop filled at 104: closed, the result's mail.
            gw.state[102] = {"status": "Filled", "filled": 10, "avg_price": 104.0}
            A.ibkr_tick(morning + 120, gw)
            o = A._get("ibkr")["orders"]["breakout:NVDA:2026-10-26"]
            self.assertEqual((o["status"], o["why"], o["pnl"]), ("closed", "trail", round(3 * 10 - 2, 2)))
            close = [m for m in A.peek_mails() if m["mode"] == "live" and m["kind"] == "close"]
            self.assertAlmostEqual(close[0]["trade"]["r"], 28 / 60)
            # IBKR refused an order: said with where and IBKR's own words.
            A._set("paper", dict(A.paper_book(), pending=[("breakout", "MSFT", 95.0, 100.0)], last="2026-10-27"))
            A.ibkr_tick(evening + 86400, gw)
            pid = A._get("ibkr")["orders"]["breakout:MSFT:2026-10-27"]["ids"]["parent"]
            gw.state[pid] = {"status": "Inactive", "filled": 0.0, "avg_price": 0.0, "why": "Insufficient funds"}
            A.ibkr_tick(evening + 86400 + 60, gw)
            a = A._get("ibkr_alerts")[-1]
            self.assertEqual((a["where"], a["symbol"], a["what"]), ("אינטראקטיב דחתה הזמנת כניסה", "MSFT", "Insufficient funds"))
            self.assertEqual(A._get("ibkr")["orders"]["breakout:MSFT:2026-10-27"]["status"], "skipped")
            # The gateway down: said too.
            class Down(AtlasToIbkrPaperTest.Gateway):
                def connect(self):
                    raise ConnectionRefusedError("gateway not running")
            A.ibkr_tick(evening + 86400 + 120, Down())
            self.assertNotEqual(A._get("ibkr_alerts")[-1]["where"], "החיבור ל-IB Gateway")  # (a restart's minute: not yet)
            self.assertIn("IB Gateway", A.status["ibkr_error"])
            A.ibkr_tick(evening + 86400 + 120 + A.DOWN_GRACE, Down())
            self.assertEqual(A._get("ibkr_alerts")[-1]["where"], "החיבור ל-IB Gateway")
            shown = [a for a in A._get("ibkr_alerts") if a["ts"] > time.time() - 86400][::-1]
            self.assertEqual([a["where"] for a in A.ibkr_view()["alerts"]], [a["where"] for a in shown][:8])  # (the last day's, newest first)
            # The simulation's own trades are its own: nothing at IBKR touched its book's closed trades or cash.
            self.assertEqual(A.paper_book()["trades"], before["trades"])
            self.assertEqual(A.paper_book()["cash"], before["cash"])
            # The alert mail: where and what, in one mail.
            sent = []
            n = types.SimpleNamespace(url="http://x", _safe_send=lambda subj, body, images=(): sent.append((subj, body)))
            atlas_mail.AtlasMail.mail_alerts(types.SimpleNamespace(n=n), A._get("ibkr_alerts"))
            self.assertIn("שליחת הזמנת כניסה", sent[0][0])
            self.assertIn("Insufficient funds", sent[0][1])

    def test_only_the_owners_two_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            A.set_settings({"rr_override": 0})
            self.assertEqual(A.plans(["pullback"])["pullback"]["rr"], 0)  # no target: the trailing stop only
            A.set_settings({"rr_override": None})
            self.assertEqual(A.plans(["pullback"])["pullback"]["rr"], A.plans(["pullback"], system=True)["pullback"]["rr"])
            for bad in (0.5, 11, -1):
                with self.assertRaises(ValueError):
                    A.set_settings({"rr_override": bad})
            A.set_settings({"live_size": 5000})
            self.assertEqual(A.settings()["live_size"], 5000)
            self.assertEqual(A._ib_qty(500, 120.0), 500)  # paper: the simulation's shares
            A.set_settings({"ibkr_mode": "live", "confirm_live": True})
            self.assertEqual(A._ib_qty(500, 120.0), 41)  # real money: $5,000's worth, whatever the simulation holds
            with self.assertRaises(ValueError):
                A._ib_qty(500, 6000.0)  # one share costs more than a trade's amount
            A.set_settings({"rr_override": 4})
            tests = A.strategy_tests()
            self.assertIn("system", next(iter(tests.values())))  # his choice against the system's, on the same history


class AtlasBellTest(unittest.TestCase):
    def test_everything_said_in_one_place_and_the_count_goes_once_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            t0 = time.time()
            A.queue_mails("sim", [{"id": "x", "strategy": "breakout", "symbol": "NVDA", "entry": 100.0, "shares": 10}], [], t0)
            A.queue_mails("live", [], [{"id": "y", "strategy": "pullback", "symbol": "AMD", "pnl": -12.5, "r": -1.0}], t0 + 1)
            A.ibkr_alert("שליחת הזמנת כניסה", "no permissions", "MSFT", t0 + 2)
            V = A.view()["notes"]
            self.assertEqual(V["unread"], 3)
            self.assertEqual([x["kind"] for x in V["items"]], ["alert", "trade", "trade"])  # newest first
            self.assertIn("AMD", V["items"][1]["title"])
            self.assertEqual(V["items"][1]["level"], "bad")
            self.assertIn("כסף אמיתי", V["items"][1]["text"])
            A.notes_seen(t0 + 3)  # opened: the count goes, read or not
            self.assertEqual(A.notes_view()["unread"], 0)
            A.ibkr_alert("החיבור ל-IB Gateway", "down", now=t0 + 4)
            self.assertEqual(A.notes_view()["unread"], 1)


class AtlasBacktestAheadTest(unittest.TestCase):
    def test_the_tabs_first_answer_is_ready_before_it_is_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            calls = []
            real = A._backtest
            A._backtest = lambda *a, **k: calls.append(a) or real(*a, **k)
            self.assertTrue(A.warm_backtest())
            for _ in range(200):
                if not A.__dict__.get("_bt_warming"):
                    break
                time.sleep(0.05)
            self.assertEqual(len(calls), 1)
            first = A.backtest([], "system", {})
            self.assertEqual(len(calls), 1)  # (ready: not worked out again)
            self.assertIn("trades", first)
            self.assertFalse(A.warm_backtest())  # (nothing changed)
            A.set_settings({"rr_override": 4})  # the rules changed: worked out again
            A.backtest([], "system", {})
            self.assertEqual(len(calls), 2)


class AtlasStopRisesTest(unittest.TestCase):
    """The owner's rule: in every trade of every strategy the stop rises with the price from the first
    day and never comes down; no strategy may turn the trailing stop off."""

    def test_the_stop_follows_the_best_close_from_the_first_day(self):
        from polybot import equity
        days = ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"]
        bars = {"o": [100, 101, 104, 103], "h": [101, 104, 106, 104], "l": [99.5, 100.5, 103, 101.5],
                "c": [100, 103, 105, 102]}
        prepared = {"X": dict(bars, day=days, index={d: i for i, d in enumerate(days)}, atr=[2.0] * 4),
                    "SPY": dict(bars, day=days, index={d: i for i, d in enumerate(days)}, atr=[2.0] * 4)}
        for kind in equity.STRATEGIES:
            P = equity.plan_for(kind, {"trail_atr": 0.0, "breakeven_r": 0.0})
            self.assertGreater(P["trail_atr"], 0)  # (no strategy may turn it off)
            book = equity.new_book(100_000)
            t = {"id": f"{kind}:X:{days[0]}", "strategy": kind, "symbol": "X", "day": days[0], "entry": 100.0, "stop": 96.0,
                 "stop0": 96.0, "shares": 10, "target": None, "risk": 4.0, "risk_usd": 40.0, "fees": 1.0, "best": 100.0, "held": 0}
            book["open"] = [t]
            port = {"max_open": 10, "regime": "off", "account": 100_000, "risk_pct": 1, "max_position": 10_000}
            stops = []
            for d in days[1:]:
                equity.step(book, prepared, d, {kind: P}, port, signals=False, symbols=("X",))
                stops.append(book["open"][0]["stop"] if book["open"] else None)
            self.assertEqual(stops[0], 100.0)  # its high 104: 104 - 4, the distance it started with
            self.assertEqual(stops[1], 102.0)  # its high 106
            self.assertIsNone(stops[2])  # it came down to 102: out there, above the entry
            self.assertEqual(book["trades"][-1]["why"], "trail")
            self.assertAlmostEqual(book["trades"][-1]["exit"], round(102 * (1 - P["slippage"]), 2))

    def test_a_day_that_rose_and_fell_back_closed_it_on_the_way_down(self):
        from polybot import equity
        days = ["2026-10-01", "2026-10-02"]
        bars = {"o": [100, 101], "h": [101, 110], "l": [99.5, 100.5], "c": [100, 104]}
        prepared = {"X": dict(bars, day=days, index={d: i for i, d in enumerate(days)}, atr=[2.0] * 2),
                    "SPY": dict(bars, day=days, index={d: i for i, d in enumerate(days)}, atr=[2.0] * 2)}
        P = equity.plan_for("breakout")
        book = equity.new_book(100_000)
        book["open"] = [{"id": "breakout:X:2026-10-01", "strategy": "breakout", "symbol": "X", "day": days[0], "entry": 100.0,
                         "stop": 96.0, "stop0": 96.0, "shares": 10, "target": None, "risk": 4.0, "risk_usd": 40.0, "fees": 1.0,
                         "best": 100.0, "held": 0, "dist": 4.0}]
        equity.step(book, prepared, days[1], {"breakout": P}, {"max_open": 10, "regime": "off", "account": 100_000,
                                                               "risk_pct": 1, "max_position": 10_000}, signals=False, symbols=("X",))
        # Up to 110 (its stop up to 106) and back down to 104: out at 106 on the way down.
        self.assertEqual(book["open"], [])
        self.assertAlmostEqual(book["trades"][-1]["exit"], round(106 * (1 - P["slippage"]), 2))

    def test_during_the_day_with_every_price(self):
        from polybot import equity
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        noon = dt.datetime(2026, 10, 27, 12, 0, tzinfo=ny).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            book = equity.new_book(100_000)
            book.update(last="2026-10-26", open=[{"id": "breakout:X:2026-10-23", "strategy": "breakout", "symbol": "X",
                                                  "day": "2026-10-23", "entry": 100.0, "stop": 96.0, "stop0": 96.0, "shares": 10,
                                                  "target": None, "risk": 4.0, "risk_usd": 40.0, "fees": 1.0, "best": 100.0,
                                                  "held": 2, "dist": 4.0}])
            A._set("paper", book)
            t0 = dt.datetime(2026, 10, 27, 9, 30, tzinfo=ny).timestamp()
            A.quotes["X"] = {"price": 107.0, "ts": noon, "bars": [(t0, 100, 103, 99.8, 102), (t0 + 300, 102, 108, 101.5, 107)]}
            self.assertEqual(A.intraday_stops(noon), [])
            self.assertEqual(A.paper_book()["open"][0]["stop"], 104.0)  # its best so far 108: 108 - 4
            A.quotes["X"] = dict(A.quotes["X"], price=103.9)  # back down to it: out, there and then
            closed = A.intraday_stops(noon + 60)
            self.assertEqual([t["why"] for t in closed], ["trail"])
            self.assertEqual(A.paper_book()["open"], [])


class AtlasLabOpeningRangeTest(unittest.TestCase):
    """The experiments lab's first experiment (scalp.py): eight variants set before any data, run on
    minute bars in event time with costs, judged in R against a random-entry placebo."""

    @staticmethod
    def bars(date, paths):
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        out = {}
        for sym, (price_at, vol_at) in paths.items():
            rows = []
            for k in range(390):
                m = 570 + k
                ts = dt.datetime.fromisoformat(date).replace(hour=m // 60, minute=m % 60, tzinfo=ny).timestamp()
                p0, p1 = price_at(k), price_at(k + 1)
                rows.append((int(ts), p0, max(p0, p1) + 0.02, min(p0, p1) - 0.02, p1, vol_at(k)))
            out[sym] = rows
        return out

    def book(self, tmp, last_day):
        from polybot import scalp
        flat = lambda base: (lambda k: base * (1 + 0.004 * (((k // 3) % 4) - 1.5) / 1.5))
        b = scalp.Bars(tmp)
        days = ["2026-10-%02d" % d for d in (12, 13, 14, 15, 16, 19)]
        for d in days:
            b.save(d, self.bars(d, {"AAA": (flat(100), lambda k: 1000), "BBB": (flat(50), lambda k: 1000),
                                    "SPY": (flat(600), lambda k: 5000)}))
        b.save("2026-10-20", self.bars("2026-10-20", {"AAA": (last_day, lambda k: 5000 if k < 5 else 2000),
                                                      "BBB": (flat(50), lambda k: 1000), "SPY": (flat(600), lambda k: 5000)}))
        return b

    def test_an_in_play_stock_breaks_out_and_rides_the_rising_stop(self):
        from polybot import scalp

        def climb(k):  # in play: up from 9:36 to 10:10, then back down
            if k < 5:
                return 100 + 0.1 * k
            return 100.5 + 0.12 * (k - 5) if k < 40 else 100.5 + 0.12 * 35 - 0.15 * (k - 40)
        with tempfile.TemporaryDirectory() as tmp:
            res, run = scalp.run_all(self.book(tmp, climb), symbols={"AAA", "BBB"})
            self.assertEqual(run[-1], "2026-10-20")
            v1 = [t for t in res["V1"] if t["day"] == "2026-10-20"]
            self.assertEqual([t["symbol"] for t in v1], ["AAA"])  # (BBB: not in play)
            t = v1[0]
            self.assertEqual(t["why"], "trail")  # out on the stop that rose with it
            self.assertGreater(t["exit"], t["entry"] + 3)
            self.assertLessEqual(t["shares"] * (t["entry"] - t["stop0"]), scalp.R + 1e-6)  # $30 at risk at most
            self.assertGreater(t["cost"], 2.0)  # (two commissions and the fees paid)
            self.assertEqual(len([x for x in res["V0"] if x["day"] == "2026-10-20"]), 1)  # the placebo: as many trades
            self.assertGreater(len(res["V6"]), len(res["V1"]))  # (no scanner: the quiet stock too)
            rows = {r["variant"]: r for r in scalp.board(res, run)}
            self.assertEqual(rows["V1"]["status"], "collecting")
            self.assertEqual(rows["V1"]["next"], {"trades": 300, "days": 20})

    def test_a_trade_that_doesnt_move_is_out_after_ten_minutes(self):
        from polybot import scalp

        def stall(k):  # breaks out a little, then goes nowhere
            if k < 5:
                return 100 + 0.1 * k
            return 100.55 + 0.03 * ((k % 4) - 1.5)
        with tempfile.TemporaryDirectory() as tmp:
            res, run = scalp.run_all(self.book(tmp, stall), symbols={"AAA", "BBB"})
            v1 = [t for t in res["V1"] if t["day"] == "2026-10-20"]
            self.assertEqual([t["why"] for t in v1], ["time"])
            self.assertLessEqual(v1[0]["out"] - v1[0]["in"], 12)
            v4 = [t for t in res["V4"] if t["day"] == "2026-10-20"]  # (no time stop: held on)
            self.assertNotIn("time", [t["why"] for t in v4])

    def test_atlas_records_the_days_and_runs_the_experiment(self):
        from polybot import scalp
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            src = self.book(os.path.join(tmp, "src"), lambda k: 100 + 0.1 * min(k, 40))
            want = {d: src.load(d) for d in src.days()}

            def fetch(sym, start, end):
                return [b for d, by in want.items() for b in by.get(sym, []) if start <= b[0] < end]
            from zoneinfo import ZoneInfo
            now = dt.datetime(2026, 10, 20, 17, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
            self.assertTrue(A.lab_tick(now, fetch=fetch, wait=True))
            self.assertEqual(A.lab_bars.days(), sorted(want))
            v = A.lab_view()
            self.assertEqual(v["last"], "2026-10-20")
            self.assertEqual({r["variant"] for r in v["board"]}, set(scalp.VARIANTS))
            from polybot import intraday
            self.assertEqual({r["variant"] for r in v["board2"]}, set(intraday.VARIANTS))  # (experiment 2 beside it)
            self.assertEqual(v["last2"], "2026-10-20")
            self.assertFalse(A.lab_tick(now + 60, fetch=fetch, wait=True))  # (recorded: not again)


class AtlasReconnectTest(unittest.TestCase):
    """The owner (29.10.2026): his login on the IBKR app takes the gateway's; a button that connects again, and
    by itself every 3 minutes while the gateway doesn't answer."""

    def test_the_button_and_every_three_minutes_while_down(self):
        from unittest import mock
        from polybot import ibkr
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp)
            A.set_settings({"ibkr_on": True})
            sent = []
            with mock.patch.object(ibkr, "gateway_command", lambda host, cmd: sent.append(cmd) or "OK"):
                rec = A.reconnect(1000.0)
                self.assertEqual((rec["cmd"], rec["sent"], rec["by"]), ("RECONNECTACCOUNT", True, "owner"))
                self.assertIsInstance(A.ibkr, ibkr.IbGateway)  # (a new connection on our side)
                self.assertFalse(A._auto_reconnect(1100.0))  # (up as far as it knows: nothing by itself)
                A.status["ibkr_down_since"] = 1000.0
                self.assertFalse(A._auto_reconnect(1100.0))  # (not 3 minutes yet)
                self.assertTrue(A._auto_reconnect(1000.0 + A.RECONNECT_EVERY))
                self.assertEqual(sent, ["RECONNECTACCOUNT", "RESTART"])  # (the second try: a restart, no approval needed)
                self.assertFalse(A._auto_reconnect(1000.0 + A.RECONNECT_EVERY + 60))
                self.assertTrue(A._auto_reconnect(1000.0 + 2 * A.RECONNECT_EVERY))
                self.assertEqual(A.status["reconnect"]["by"], "auto")
            with mock.patch.object(ibkr, "gateway_command", mock.Mock(side_effect=ConnectionRefusedError("no"))):
                self.assertFalse(A.reconnect(5000.0)["sent"])  # (the gateway itself down: our side reconnects anyway)
            v = A.ibkr_view()
            self.assertEqual((v["down_since"], v["reconnect_every"]), (1000.0, 180))

    def test_a_deploy_closes_the_gateway_connection_first(self):
        from polybot.__main__ import shutdown
        closed, stopped = [], []
        gw = types.SimpleNamespace(close=lambda: closed.append(1))
        engine = types.SimpleNamespace(stop=lambda: stopped.append(1), atlas=types.SimpleNamespace(ibkr=gw))
        shutdown(engine)
        self.assertEqual((closed, stopped), ([1], [1]))
        shutdown(types.SimpleNamespace(stop=lambda: None, atlas=types.SimpleNamespace(ibkr=None)))  # (never connected)

    def test_the_live_labs_data_on_a_connection_of_its_own(self):
        from polybot import ibkr
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp)
            A.ibkr = ibkr.IbGateway("gw.internal", 4006)
            g = A.data_gateway()
            self.assertIsNot(g, A.ibkr)
            self.assertEqual((g.host, g.port, g.client_id), ("gw.internal", 4006, ibkr.CLIENT_ID + 1))
            self.assertIs(A.data_gateway(), g)  # (the same one, kept)

    def test_past_failures_can_be_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp)
            A.ibkr_alert("החיבור ל-IB Gateway", "TimeoutError", now=time.time() - 60)
            self.assertEqual(len(A.ibkr_view()["alerts"]), 1)
            A.clear_alerts()
            self.assertEqual(A.ibkr_view()["alerts"], [])
            A.ibkr_alert("קריאת מצב ההזמנות", "TimeoutError", now=time.time() + 1)  # (a new one shows again)
            self.assertEqual(len(A.ibkr_view()["alerts"]), 1)

    def test_back_resets_the_tries(self):
        from unittest import mock
        from polybot import ibkr
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp)
            A.set_settings({"ibkr_on": True})
            with mock.patch.object(ibkr, "gateway_command", lambda host, cmd: "OK"):
                A.status["ibkr_down_since"] = 10.0
                A.reconnect(500.0)
            A.ibkr_tick(600.0, AtlasToIbkrPaperTest.Gateway())
            self.assertEqual((A.status["reconnect_tries"], A.status["reconnect"]["back"]), (0, 600.0))
            self.assertNotIn("ibkr_down_since", A.status)


class AtlasSizingTest(unittest.TestCase):
    """The owner (29.10.2026): the best trade size for the account set in the system, and the best number of trades
    open at once; two research cards, and once there's an answer, every strategy updated."""

    def test_the_study_chooses_on_the_first_part_and_judges_on_the_locked_test(self):
        from polybot import equity, sizing
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp, n=420)
            out = sizing.study(eq.prepared, A.plans(), A.port(), equity.portfolio, 0.2)
            S, L = out["size"], out["slots"]
            self.assertEqual(len(S["rows"]), len(sizing.RISKS) * len(sizing.POSITIONS))
            self.assertEqual([x["max_open"] for x in L["rows"]], list(sizing.SLOTS))
            self.assertEqual((S["rows"][S["now"]]["risk_pct"], L["rows"][L["now"]]["max_open"]), (1.0, 10))  # (the settings now)
            r = S["rows"][0]["res"]
            self.assertLess(r["choose"]["days"], r["all"]["days"])
            self.assertEqual(r["choose"]["days"] + r["test"]["days"], r["all"]["days"])  # (the two parts: the whole)
            if S["best"] is not None:
                self.assertLessEqual(S["rows"][S["best"]]["res"]["choose"]["dd"], 0.2 * 2 / 3)  # (only the safe ones)
            self.assertIn(S["state"], ("found", "nothing", "promising"))

    def test_a_safe_peak_alone_loses_to_a_safe_plateau(self):
        from polybot import sizing
        row = lambda a, dd=0.05: {"res": {"choose": {"a_year": a, "dd": dd}}}
        rows = [row(0.1), row(0.1), row(0.5), row(0.0), row(0.0), row(0.3, 0.5), row(0.25), row(0.25), row(0.25)]
        best, smooth = sizing.pick(rows, lambda i: [j for j in (i - 1, i + 1) if 0 <= j < len(rows)], 0.2)
        self.assertIsNone(smooth[5])  # (fell too far: out)
        self.assertEqual(best, 7)  # (the middle of the plateau, not the lone peak at 2)

    def test_the_answer_updates_every_strategy_and_the_owner_can_keep_it_to_show(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp, n=420)
            out = {"size": {"state": "found", "answer": {"risk_pct": 1.5, "max_position": 20_000.0, "position_pct": 20}},
                   "slots": {"state": "found", "answer": {"max_open": 15}}}
            self.assertEqual(A.sizing_apply(out), {"risk_pct": [1.0, 1.5], "max_position": [10_000.0, 20_000.0], "max_open": [10, 15]})
            self.assertEqual({k: A.port()[k] for k in ("risk_pct", "max_position", "max_open")}, {"risk_pct": 1.5, "max_position": 20_000.0, "max_open": 15})
            self.assertTrue(all(P["risk_pct"] == 1.5 and P["max_open"] == 15 for P in A.plans().values()))
            self.assertIsNone(A.sizing_apply(out))  # (the same: nothing to change)
            A.set_settings({"sizing_auto": False})
            out["slots"]["answer"]["max_open"] = 8
            self.assertIsNone(A.sizing_apply(out))
            self.assertEqual(A.port()["max_open"], 15)
            out["slots"]["state"] = "nothing"
            A.set_settings({"sizing_auto": True})
            self.assertIsNone(A.sizing_apply(out))  # (an answer that didn't hold: not applied)

    def test_runs_once_and_again_when_the_account_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp, n=420)
            A.set_settings({"sizing_auto": False})
            self.assertTrue(A.sizing_tick(1000.0, wait=True, apart=False))
            v = A.view()["sizing"]
            self.assertTrue(v["current"])
            self.assertEqual(v["account"], 100_000.0)
            self.assertFalse(A.sizing_tick(1000.0 + 7200, wait=True, apart=False))  # (the same: not again)
            A.set_settings({"account": 50_000})
            self.assertFalse(A.view()["sizing"]["current"])
            self.assertTrue(A.sizing_tick(1000.0 + 7200, wait=True, apart=False))  # (a new size: at once)
            self.assertEqual(A.view()["sizing"]["account"], 50_000.0)


class AtlasStreaksTest(unittest.TestCase):
    def test_every_strategys_trades_over_the_history_once_a_data_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp, n=420)
            self.assertTrue(A.streaks_tick(wait=True))
            S = A._get("streaks")
            self.assertEqual(S["version"], eq.version)
            self.assertEqual(S["unit"], "days")
            self.assertEqual([r["lookback"] for r in S["rows"]], list(A.STREAK_LOOKBACKS))
            self.assertGreater(S["trades"], 0)
            self.assertIn(S["state"], ("found", "real_not_paying", "promising", "nothing", "collecting"))
            self.assertFalse(A.streaks_tick(wait=True))  # (the same data: not again)
            self.assertEqual(A.view()["streaks"]["version"], eq.version)


class AtlasMyTradesActionsTest(unittest.TestCase):
    """My trades' cards: an open trade of the simulation into the IBKR paper account, and adding to it."""

    def test_add_and_send_live(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            book = equity.new_book(100_000)
            t = {"id": "breakout:NVDA:2026-10-26", "strategy": "breakout", "symbol": "NVDA", "day": "2026-10-26", "entry": 100.0,
                 "stop": 95.0, "stop0": 95.0, "shares": 100, "target": 110.0, "risk": 5.0, "risk_usd": 500.0, "fees": 1.0,
                 "best": 100.0, "held": 1}
            book.update(last="2026-10-26", open=[t], cash=90_000.0)
            A._set("paper", book)
            A.quotes["NVDA"] = {"price": 102.0, "ts": 1}
            with self.assertRaises(ValueError):
                A.send_live(t["id"])  # the IBKR connection is off
            with self.assertRaises(ValueError):
                A.add_to(t["id"], 0.3)  # 25 / 50 / 100 only
            r = A.add_to(t["id"], 0.5)
            self.assertEqual((r["added"], r["shares"], r["live"]), (50, 150, None))
            o = A.paper_book()["open"][0]
            slip = 102.0 * (1 + equity.plan_for("breakout")["slippage"])
            self.assertAlmostEqual(o["entry"], (100.0 * 100 + slip * 50) / 150)
            self.assertAlmostEqual(o["risk_usd"], (o["entry"] - 95.0) * 150)
            self.assertLess(A.paper_book()["cash"], 90_000.0 - 50 * 102.0 + 1e-6)
            # Into the paper account: the simulation's shares, stop and target.
            A.set_settings({"ibkr_on": True})
            gw = AtlasToIbkrPaperTest.Gateway()
            A.ibkr = gw
            A.send_live(t["id"])
            self.assertEqual(gw.placed[0], ("NVDA", 150, None, 110.0, 95.0))
            with self.assertRaises(ValueError):
                A.send_live(t["id"])  # already there
            # Adding now adds there too, a bracket of its own.
            r = A.add_to(t["id"], 0.25)
            self.assertEqual(gw.placed[-1][:2], ("NVDA", 37))
            self.assertEqual(r["live"]["status"], "sent")
            self.assertEqual(len(A.ibkr_view()["rows"]), 2)
            # A real-money account: nothing is sent.
            A.ibkr = AtlasToIbkrPaperTest.Gateway(accounts=("U765432",))
            A.paper_book()
            with self.assertRaises(ValueError):
                A._gateway()


class TheBacktestCanBeTrustedTest(unittest.TestCase):
    """The Chrome audit's findings: a test the brain never sees, how much a result leans on a few, the
    account's value counting the exit's costs, no chasing a stock that opened far above its signal."""

    def _atlas(self, tmp):
        return AtlasTest._atlas(self, tmp, n=420)

    def test_the_history_is_in_three_parts_and_the_curve_matches_the_trades(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = self._atlas(tmp)
            r = equity.portfolio(eq.prepared, A.plans(), A.port())
            sp = r["split"]
            self.assertTrue(r["start"] < sp["val"] < sp["cut"])
            # (the backtest starts where a trade can open: no flat months of warm-up)
            self.assertEqual(r["start"], equity.first_tradable(eq.prepared))
            self.assertEqual(r["curve"][0][0], r["start"])
            n = sp["old"]["trades"] + sp["val_stats"]["trades"] + sp["new"]["trades"]
            self.assertEqual(n, r["stats"]["trades"])
            # The account at the end: the start plus every trade's result (the open ones' too), to the cent.
            s = r["stats"]
            self.assertAlmostEqual(s["final"], 100_000 + s["pnl"] + s["open_pnl"], delta=0.05 * (s["trades"] + s["open"] + 1))

    def test_the_brain_never_reads_the_locked_test(self):
        from unittest import mock
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = self._atlas(tmp)
            real = equity.backtest

            class Locked(dict):
                def __getitem__(self, k):
                    if k in ("new", "robust"):
                        raise AssertionError("the brain read the locked test")
                    return dict.__getitem__(self, k)

            def guarded(*a, **k):
                r = real(*a, **k)
                r["split"] = Locked(r["split"])
                return r
            with mock.patch.object(equity, "backtest", guarded):
                for _ in range(6):
                    A.learn_once("breakout")
            b = A.brain()["breakout"]
            self.assertIn("val", b)
            self.assertNotIn("new", b)
            # What an older brain learned (it chose on data now in the locked test) is dropped.
            A._set("brain_v", 1)
            self.assertEqual(A.brain(), {})

    def test_the_data_refreshes_right_after_the_close(self):
        from polybot import equity
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        eq = equity.Equities.__new__(equity.Equities)
        at = lambda h, m, d=26: dt.datetime(2026, 10, d, h, m, tzinfo=ny).timestamp()  # (Monday)
        eq.status = {"refreshed": at(13, 0)}
        self.assertFalse(eq.due(at(15, 0)))
        self.assertFalse(eq.due(at(16, 10)))  # (the bar is final from 16:20)
        self.assertTrue(eq.due(at(16, 26)))  # the day's bar is complete: at once, not 6 hours later
        eq.status = {"refreshed": at(16, 26)}
        self.assertFalse(eq.due(at(17, 0)))
        eq.status = {"refreshed": at(13, 0, 24)}  # (Saturday: no close)
        self.assertFalse(eq.due(at(16, 30, 24)))

    def test_how_much_a_result_leans_on_a_few(self):
        from polybot import equity
        t = lambda day, pnl: {"in": day, "pnl": pnl, "why": "target" if pnl > 0 else "stop"}
        trades = [t("2026-04-09", 5000), t("2026-04-09", 4000), t("2026-05-01", -300), t("2026-05-02", 200),
                  t("2026-05-03", -400), t("2026-05-04", 100)]
        r = equity.robust(trades)
        self.assertEqual(r["best_day"], "2026-04-09")
        self.assertEqual(r["best_day_trades"], 2)
        self.assertAlmostEqual(r["without_best_day"], -400)
        self.assertAlmostEqual(r["without_top5"], -400)
        self.assertGreater(r["top10_share"], 0.99)
        self.assertIsNone(equity.robust([]))

    def test_no_chasing_a_stock_that_opened_far_above_its_signal(self):
        from polybot import equity
        spy = StocksBacktestTest._bars(300)
        prepared = {"SPY": equity.prepare(spy)}
        days = [d for d, *_ in spy]
        # The signal's close at 100, a day's range about 2: tomorrow it opens at 132 (+32%, DELL's).
        bars = [(d, 100.0, 101.0, 99.0, 100.0, 1e6) for d in days[:251]] + [(days[251], 132.0, 135.0, 131.0, 134.0, 1e6)]
        prepared["NVDA"] = equity.prepare(bars)
        P = equity.plan_for("breakout")
        for ceiling, entered in ((100 + 1.0 * 2, False), (None, True)):
            book = equity.new_book(100_000)
            book["pending"] = [("breakout", "NVDA", 95.0, ceiling)]
            equity.step(book, prepared, days[251], {"breakout": P}, P, signals=False, symbols=("NVDA",))
            self.assertEqual(bool(book["open"]), entered)
            self.assertEqual(bool(book.get("skipped")), not entered)
        # (a book kept before the ceiling existed: its 3-part signals still enter)
        book = equity.new_book(100_000)
        book["pending"] = [("breakout", "NVDA", 95.0)]
        equity.step(book, prepared, days[251], {"breakout": P}, P, signals=False, symbols=("NVDA",))
        self.assertTrue(book["open"])

    def test_the_system_run_is_stamped_and_compared_with_the_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = self._atlas(tmp)
            A._set("brain", {"breakout": {"params": {"stop_atr": 3.0}, "kept": 1, "tried": 5, "since": 1.0}})
            from polybot import atlas as atlas_mod
            A._set("brain_v", atlas_mod.BRAIN_VERSION)
            r = A.backtest(["breakout"], "system")
            self.assertEqual(r["stamp"]["brain"]["breakout"]["kept"], 1)
            self.assertEqual(r["stamp"]["data_last"], eq.prepared["SPY"]["day"][-1])
            self.assertIn("default", r)
            self.assertIn("new", r["default"]["split"])
            # The owner's rules: no default to compare with; an unknown market filter is refused.
            self.assertNotIn("default", A.backtest(["breakout"], "custom", {"stop_atr": 2}))
            with self.assertRaises(ValueError):
                A.backtest(["breakout"], "custom", {"regime": "xyz"})
            # Each strategy's locked test, for the page's cards.
            v = A.view()
            self.assertIn("new", v["strategies"][0]["test"])

    def test_a_pending_signal_shows_its_estimated_entry_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = self._atlas(tmp)
            from polybot import equity
            book = equity.new_book(100_000)
            last = eq.prepared["NVDA"]["c"][-1]
            book.update(started=eq.prepared["SPY"]["day"][-1], last=eq.prepared["SPY"]["day"][-1],
                        pending=[("breakout", "NVDA", last - 10, last + 2)], curve=[(eq.prepared["SPY"]["day"][-1], 100_000)])
            A._set("paper", book)
            p = A.paper_view()["pending"][0]
            self.assertAlmostEqual(p["est_entry"], round(last, 2))
            self.assertAlmostEqual(p["est_target"], round(last + A.plans(["breakout"])["breakout"]["rr"] * 10, 2), delta=0.02)

    def test_only_market_news_counts(self):
        from polybot import atlas
        self.assertTrue(atlas.important({"title": "Fed holds rates steady as inflation cools"}))
        self.assertTrue(atlas.important({"title": "Nvidia earnings beat, guidance raised"}))
        self.assertFalse(atlas.important({"title": "Mortgage rates fall: should you refinance?"}))
        self.assertFalse(atlas.important({"title": "Trump sues a newspaper over a story"}))
        self.assertTrue(atlas.important({"title": "Statement on monetary policy", "source": "הפד"}))

    def test_supports_and_resistances_are_not_on_top_of_the_price(self):
        from polybot import equity
        rnd = __import__("random").Random(5)
        bars, d, p = [], dt.date(2024, 1, 1), 100.0
        while len(bars) < 300:
            if d.weekday() < 5:
                o = p
                p = p * (1 + rnd.gauss(0, 0.01))
                bars.append((d.isoformat(), o, max(o, p) * 1.005, min(o, p) * 0.995, p, 1e6))
            d += dt.timedelta(days=1)
        T = equity.technical(equity.prepare(bars))
        gap = 0.5 * T["atr"]
        for xs in (T["support"], T["resistance"]):
            for x in xs:
                self.assertGreaterEqual(abs(x - T["close"]), gap - 0.01)
            if len(xs) == 2:
                self.assertGreaterEqual(abs(xs[0] - xs[1]), gap - 0.01)


class AtlasMailsTest(unittest.TestCase):
    """Atlas's mails: Polymarket's mails, for each stocks trade that opens or closes."""

    def _setup(self, tmp):
        from polybot.atlas_mail import AtlasMail
        A, eq = AtlasTest._atlas(AtlasTest(), tmp)
        engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "p.sqlite")), api=FakeApi(),
                        env={"SMTP_USER": "bot@gmail.com", "SMTP_PASSWORD": "x"}, streaming=False, with_lab=False)
        engine.notifier.update(email="owner@example.com")
        sent = []

        def send(subject, body, images=()):
            sent.append((subject, body, images))
            engine.notifier.last_sent = time.time()
        engine.notifier.send = send
        M = AtlasMail(engine.notifier, A)
        M.maybe_daily = lambda *a, **k: None  # (the daily summary at 23:30 has its own test: a run after it would count it)
        if engine.notifier.atlas_mail is not None:
            engine.notifier.atlas_mail.maybe_daily = lambda *a, **k: None
        return A, M, sent, engine

    def _trade(self, i, pnl=None):
        sym = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V")[i]
        t = {"id": f"pullback:{sym}:2026-10-01", "strategy": "pullback", "symbol": sym, "day": "2026-10-01",
             "entry": 100.0, "stop": 96.0, "stop0": 96.0, "target": 112.0, "shares": 100, "risk_usd": 400.0}
        if pnl is not None:
            t.update({"in": "2026-10-01", "out": "2026-10-05", "exit": 100 + pnl / 100, "pnl": pnl, "r": pnl / 400, "why": "target" if pnl > 0 else "stop", "days": 3})
        return t

    def test_open_and_close_mails_after_quiet_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, M, sent, engine = self._setup(tmp)
            now = time.time()
            A.queue_mails("sim", [self._trade(1), self._trade(2)], [], now=now)
            M.tick(now + 5, True)
            self.assertEqual(sent, [])  # (waiting for the rest of the batch)
            M.tick(now + 40, True)
            self.assertEqual(len(sent), 1)
            self.assertIn("2 עסקאות נפתחו", sent[0][0])
            self.assertIn("MSFT", sent[0][1])
            self.assertTrue(sent[0][2])  # the charts
            self.assertTrue(A.mail_status()["pullback:MSFT:2026-10-01"]["open"]["ok"])
            A.queue_mails("sim", [], [self._trade(1, pnl=1200)], now=now + 50)
            M.tick(now + 90, True)
            self.assertIn("✅ MSFT", sent[-1][0])
            self.assertEqual(A.peek_mails(), [])
            engine.store.close()

    def test_every_n_off_and_quiet_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, M, sent, engine = self._setup(tmp)
            now = time.time()
            M.update({"open_every": 3})
            A.queue_mails("sim", [self._trade(1), self._trade(2)], [], now=now)
            M.tick(now + 60, True)
            self.assertEqual(sent, [])  # two of three
            M.update({"open_every": 1, "close": False})
            A.queue_mails("sim", [], [self._trade(3, pnl=-400)], now=now)
            M.tick(now + 60, True)
            self.assertEqual(len(sent), 1)  # the openings went; the close is off
            self.assertIn("כבוי", A.mail_status()["pullback:AMZN:2026-10-01"]["close"]["why"])
            with self.assertRaises(ValueError):
                M.update({"open_every": 0})
            # Quiet hours (all day): kept, then one mail when they end.
            M.update({"quiet": True, "quiet_from": "00:00", "quiet_to": "23:59", "close": True})
            A.queue_mails("sim", [self._trade(4)], [self._trade(5, pnl=300)], now=now)
            M.tick(now + 60, True)
            self.assertEqual(len(sent), 1)
            M.update({"quiet": False})
            M.tick(now + 120, True)
            self.assertIn("בשעות השקטות", sent[-1][0])
            engine.store.close()

    def test_daily_and_resend(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, M, sent, engine = self._setup(tmp)
            A.paper_step()
            subject, body = M.daily_mail(time.time())
            self.assertIn("אטלס", subject)
            self.assertIn("שווי התיק", body)
            book = A.paper_book()
            book["trades"].append(self._trade(9, pnl=500))
            A._set("paper", book)
            M.resend("pullback:V:2026-10-01", "close")
            self.assertIn("V", sent[-1][0])
            with self.assertRaises(ValueError):
                M.resend("nope", "open")
            engine.store.close()

    def test_the_simulation_queues_its_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, M, sent, engine = self._setup(tmp)
            spy = A.eq.prepared["SPY"]
            book = __import__("polybot.equity", fromlist=["new_book"]).new_book(100_000)
            book.update(started=spy["day"][250], last=spy["day"][250])
            A._set("paper", book)
            A.paper_step()
            q = A.peek_mails()
            book = A.paper_book()
            self.assertEqual(sum(1 for m in q if m["kind"] == "close"), len(book["trades"]))
            self.assertEqual(sum(1 for m in q if m["kind"] == "open"), len(book["trades"]) + len(book["open"]))
            engine.store.close()


class OrderPathSpeedTest(unittest.TestCase):
    """The expert panels' code changes (28.10.2026): nothing waits that needn't on the way to an order."""

    def test_real_moneys_markets_are_evaluated_before_the_wait_for_a_burst(self):
        import asyncio
        seen = []
        class T:
            def __init__(self, rules):
                self.live_rules = rules
        live, lab_only = T([object()]), T([])
        fake = types.SimpleNamespace(_stop=types.SimpleNamespace(is_set=lambda: len(seen) >= 2),
                                     health=types.SimpleNamespace(record_latency=lambda *a: None))

        async def run():
            fake._wake = asyncio.Event()
            fake._dirty = {lab_only: time.time(), live: time.time()}
            fake._evaluate = lambda t, now: seen.append((t, time.time()))
            fake._wake.set()
            woke = time.time()
            await Engine._evaluator(fake)
            return woke
        woke = asyncio.run(run())
        self.assertEqual([t for t, _ in seen], [live, lab_only])
        self.assertLess(seen[0][1] - woke, 0.01)  # at once
        self.assertGreaterEqual(seen[1][1] - woke, 0.015)  # the lab after the burst's wait

    def test_a_price_held_back_by_the_rate_limit_goes_out_when_the_limit_allows(self):
        import asyncio
        got = []

        class S(feeds.ExchangeStream):
            def parse(self, msg):
                return "X", msg["ts"], msg["p"]
        s = S({"btc": "X"}, lambda src, a, ts, p: got.append((time.time(), p)), min_interval=0.05)

        async def run():
            s._handle(json.dumps({"ts": 1, "p": 100.0}))
            s._handle(json.dumps({"ts": 2, "p": 101.0}))  # inside the 50 ms: held back
            await asyncio.sleep(0.1)
        asyncio.run(run())
        self.assertEqual([p for _, p in got], [100.0, 101.0])
        self.assertLess(got[1][0] - got[0][0], 0.09)  # not the next heartbeat's flush

    def test_the_notifiers_pass_reads_only_the_trades_not_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            try:
                plan = store.db.execute("EXPLAIN QUERY PLAN UPDATE bets SET notified = 3 WHERE notified != 3 AND"
                                        " (status != 'filled' OR mode NOT IN ('live'))").fetchall()
                self.assertIn("bets_unnotified", " ".join(str(tuple(r)) for r in plan))
            finally:
                store.close()


class LiveTakesWhatThereIsTest(unittest.TestCase):
    """Real money's orders since 28.10.2026: the strategies' own take what there is (fill-and-kill), and may
    pay 1-2 cents more once the room study found it pays; the owner's own trades stay all or nothing."""
    setUp, tearDown, trader = LiveTest.setUp, LiveTest.tearDown, LiveTest.trader

    def armed(self):
        t = self.trader()
        t.set_readiness(True)
        t.arm(True)
        return t

    def bet(self):
        return tuple(self.store.db.execute("SELECT status, shares, avg_price, seen FROM bets WHERE mode = 'live'").fetchone())

    def test_a_partial_fill_is_recorded_as_the_shares_that_traded(self):
        t = self.armed()
        t.client.response = types.SimpleNamespace(ok=True, order_id="0x1", status="matched",
                                                  making_amount=Decimal("3.50"), taking_amount=Decimal("5"))
        self.assertEqual(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market), "filled")
        self.assertEqual(t.client.orders[0]["order_type"], "FAK")
        status, shares, price, seen = self.bet()
        self.assertEqual((status, shares), ("filled", 5.0))
        self.assertAlmostEqual(price, 0.70)
        self.assertAlmostEqual(seen, 0.70)

    def test_nothing_left_is_a_refusal_in_plain_words(self):
        from polybot.live import reject_reason
        t = self.armed()
        t.client.response = types.SimpleNamespace(ok=False, code="fak_not_filled", message="no orders found to match with FAK order.")
        self.assertEqual(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market), "rejected")
        err = self.store.db.execute("SELECT error FROM bets WHERE mode = 'live'").fetchone()[0]
        self.assertIn("לא נשארה אף מניה", reject_reason(err))

    def test_a_delayed_answer_asks_what_filled(self):
        t = self.armed()
        t.MATCH_WAIT = 0
        t.client.response = types.SimpleNamespace(ok=True, order_id="0x2", status="delayed",
                                                  making_amount=Decimal("0"), taking_amount=Decimal("0"))
        t.client.get_order = lambda order_id: types.SimpleNamespace(size_matched="4", price="0.70")
        self.assertEqual(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market), "filled")
        self.assertEqual(self.bet()[:2], ("filled", 4.0))

    def test_the_room_pays_up_to_a_cent_more_only_when_on_and_only_within_the_edge(self):
        t = self.armed()
        t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market, room=(0.05, 0.0))
        self.assertAlmostEqual(t.client.orders[-1]["max_price"], 0.70)  # off: the price seen
        t.room = 1
        t.maybe_trade(0, "btc", 15, 900, 300, self.fill, self.market, room=(0.05, 0.0))
        self.assertAlmostEqual(t.client.orders[-1]["max_price"], 0.71)
        t.room = 2
        t.maybe_trade(0, "btc", 15, 1800, 300, self.fill, self.market, room=(0.18, 0.0))
        self.assertAlmostEqual(t.client.orders[-1]["max_price"], 0.70)  # a cent more would leave less than the edge
        t.maybe_trade(0, "btc", 15, 2700, 300, self.fill, self.market)  # a favourite / lottery rule: no room
        self.assertAlmostEqual(t.client.orders[-1]["max_price"], 0.70)

    def test_a_rules_own_room_from_the_research(self):
        t = self.armed()
        t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market, room=(0.05, 0.0, 3))
        self.assertGreater(t.client.orders[-1]["max_price"], 0.70)  # (real money's own room off; the rule's: up to 3c)
        self.assertLessEqual(t.client.orders[-1]["max_price"], 0.73 + 1e-9)

    def test_the_owners_own_trade_stays_all_or_nothing_without_room(self):
        t = self.armed()
        t.room = 2
        t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market, variant="manual", approved=True, room=(0.05, 0.0))
        self.assertEqual((t.client.orders[-1]["order_type"], round(t.client.orders[-1]["max_price"], 2)), ("FOK", 0.70))


    def test_a_cheap_side_may_pay_up_to_five_cents_more_within_the_edge(self):
        from polybot.paper import Fill
        t = self.armed()
        cheap = lambda fair: Fill(side="up", fair_prob=fair, best_ask=0.15, avg_price=0.15, shares=40, cost=6.0, fee=0.1,
                                  edge=fair - 0.15, limit_price=0.15)
        t.maybe_trade(0, "btc", 15, 0, 300, cheap(0.40), self.market, room=(0.12, 0.0))
        self.assertAlmostEqual(t.client.orders[-1]["max_price"], 0.15)  # off: the price seen
        t.cheap_room = 5
        t.maybe_trade(0, "btc", 15, 900, 300, cheap(0.40), self.market, room=(0.12, 0.0))
        self.assertAlmostEqual(t.client.orders[-1]["max_price"], 0.20)  # 5 cents more: the edge is still 12 points
        t.maybe_trade(0, "btc", 15, 1800, 300, cheap(0.30), self.market, room=(0.12, 0.0))
        self.assertLess(t.client.orders[-1]["max_price"], 0.19)  # only as far as the edge allows
        self.assertGreaterEqual(t.client.orders[-1]["max_price"], 0.15)
        t.maybe_trade(0, "btc", 15, 2700, 300, self.fill, self.market, room=(0.05, 0.0))
        self.assertAlmostEqual(t.client.orders[-1]["max_price"], 0.70)  # not a cheap side: the regular room (off)


class FillFirstTest(unittest.TestCase):
    """The owner (30.10.2026): get into 85% and more of the very same trades with real money."""

    def test_the_price_a_refused_order_needed_and_what_it_takes_for_85(self):
        from polybot import fills
        from polybot.store import Store, connect
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.sqlite")
            st = Store(path)
            refused = 0
            for i in range(100):
                ws = 1_800_000_000 + i * 300
                st._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', 5, ?, 'up')", (ws,))
                status = "rejected" if i % 5 < 2 else "filled"  # (40 refused among them, from the first)
                st._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee, avg_price, pnl)"
                      " VALUES ('live', ?, ?, 'btc', 5, ?, 'up', 10, 5, 0, 0.5, ?)", (status, ws + 100, ws, 5.0 if status == "filled" else None))
                if status == "rejected":  # refused: 30 of them needed 3 cents more, 10 needed 12
                    refused += 1
                    need = 0.53 if refused <= 30 else 0.62
                    st.add_fill_test((ws + 100, "btc", 5, ws, "live", "v", "up", 10, 0.50, 0, None, None, None, None, None, None, None, None, 1, need))
            db = connect(path)
            R = fills.reach(db, 0, "live")
            self.assertEqual((R["sent"], R["filled"], R["refused"]), (100, 60, 40))
            self.assertAlmostEqual(R["rate"], 0.6)
            self.assertEqual(R["r85"], 3)  # (60 + 30 = 90%)
            at3 = next(x for x in R["rooms"] if x["cents"] == 3)
            self.assertAlmostEqual(at3["pnl"], 30 * 10 * (1 - 0.53))
            self.assertAlmostEqual(next(x for x in R["rooms"] if x["cents"] == 15)["rate"], 1.0)
            db.close()
            st.close()

    def test_real_money_pays_what_it_takes_within_the_floor(self):
        t = LiveTakesWhatThereIsTest.armed(self)
        t.fill_room = ({}, 8)
        t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market, room=(0.15, 0.0, 0))
        self.assertGreater(t.client.orders[-1]["max_price"], 0.70)  # (the rule's 15 points held it at 70c before)
        self.assertLessEqual(t.client.orders[-1]["max_price"], 0.78 + 1e-9)
        t.fill_room = None
        t.maybe_trade(0, "btc", 15, 900, 300, self.fill, self.market, room=(0.15, 0.0, 0))
        self.assertAlmostEqual(t.client.orders[-1]["max_price"], 0.70)

    setUp, tearDown, trader = LiveTest.setUp, LiveTest.tearDown, LiveTest.trader

    def test_on_from_the_start_and_stopped_by_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            streaming=False, with_lab=False)
            self.assertEqual(engine.live.fill_room, ({}, engine.fill_cap()))
            for i in range(engine.FILL_STOP_N):
                engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee,"
                                " avg_price, seen, pnl) VALUES ('live', 'filled', ?, 'btc', 15, ?, 'up', 10, 5.6, 0.1, 0.56, 0.50, -5.7)",
                                (1001.0 + i, i * 900))
            engine._fill_first_tick(1000.0)
            C = engine._fill_first_tick(2000.0)
            self.assertTrue(C["stopped"])
            self.assertIsNone(engine.live.fill_room)
            engine.set_fill_first(True)
            self.assertIsNotNone(engine.live.fill_room)
            engine.store.close()


class CheapRoomDecisionTest(unittest.TestCase):
    def test_on_from_the_start_stopped_by_real_losses_and_switched_by_the_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            streaming=False, with_lab=False)
            self.assertEqual(engine.live.cheap_room, engine.CHEAP_ROOM)
            engine.fills = {"room_state": "promising", "room": {}}
            self.assertEqual(engine._room_tick(1000.0)["cents"], 0)  # the regular room: still off
            self.assertEqual(engine.cheap_room_status()["cents"], engine.CHEAP_ROOM)
            for i in range(engine.CHEAP_STOP_N):  # real trades on a cheap side that paid more, losing
                engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee,"
                                " avg_price, seen, pnl) VALUES ('live', 'filled', ?, 'btc', 15, ?, 'up', 10, 1.8, 0.1, 0.18, 0.15, -1.9)",
                                (1001.0 + i, i * 900))
            engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee,"
                            " avg_price, seen, pnl) VALUES ('live', 'filled', 1001, 'btc', 15, 99, 'up', 10, 7.1, 0.1, 0.71, 0.70, 2.0)")
            self.assertEqual(engine._room_live(0, False), (1, 2.0))  # (the regular room's own count: not the cheap ones)
            engine._room_tick(2000.0)
            C = engine.cheap_room_status()
            self.assertEqual((C["cents"], C["stopped"]["n"], engine.live.cheap_room), (0, engine.CHEAP_STOP_N, 0))
            engine.set_cheap_room(True)
            self.assertEqual(engine.live.cheap_room, engine.CHEAP_ROOM)  # on again: its count starts over
            engine.set_cheap_room(False)
            self.assertEqual((engine.live.cheap_room, engine._room_tick(3000.0)["cents"]), (0, 0))
            engine.store.close()

    def test_the_study_can_stop_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            streaming=False, with_lab=False)
            from polybot import fills
            engine.fills = {"cheap": {"rooms": {"5": {"settled": fills.NEED, "roi": -0.1, "pnl": -40.0}}}}
            self.assertEqual(engine._cheap_room_tick(1000.0)["stopped"]["why"], "study")
            self.assertEqual(engine.live.cheap_room, 0)
            engine.store.close()


class WouldHaveTest(unittest.TestCase):
    """The owner (29.10.2026): the real trades' table shows what each order would have made had it gone through."""

    def test_a_refused_order_once_its_round_is_decided(self):
        from polybot import stats
        from polybot.store import Store, connect
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.sqlite")
            store = Store(path)
            for i, (side, price, status, winner, pnl) in enumerate([("up", 0.11, "rejected", "up", None), ("down", 0.78, "rejected", "up", None),
                                                                    ("up", 0.60, "filled", "down", -5.0), ("up", 0.52, "rejected", None, None)]):
                ws = 1_800_000_000 + i * 300
                shares = 5 / price
                store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee, avg_price, pnl)"
                         " VALUES ('live', ?, ?, 'btc', 5, ?, ?, ?, ?, 0.1, ?, ?)", (status, ws + 10, ws, side, shares, 5.0, price, pnl))
                if winner:
                    store._x("INSERT INTO windows (asset, minutes, window_start, winner) VALUES ('btc', 5, ?, ?)", (ws, winner))
            db = connect(path)
            got = {round(b["avg_price"], 2): stats.would_have(b) for b in stats.recent_bets(db, "live")}
            self.assertAlmostEqual(got[0.11], round(5 / 0.11 - 5.1, 2))  # (the cheap side won: the big win missed)
            self.assertEqual(got[0.78], -5.1)
            self.assertEqual(got[0.6], -5.0)  # (filled: its own result)
            self.assertIsNone(got[0.52])  # (not decided yet)
            db.close()
            store.close()


class ResearchIntoLabTest(unittest.TestCase):
    """The owner (29.10.2026): whatever the research finds that helps becomes, by itself, a cloned strategy in the lab."""

    def _tracker(self, tmp):
        cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
        engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
        t = engine.trackers[0]
        t.window = 1_800_000_000
        return engine, t

    def _order(self, key, price=0.50, fair=0.70, ts=100.0):
        from polybot.paper import Fill
        f = Fill(side="up", fair_prob=fair, best_ask=price, avg_price=price, shares=10, cost=10 * price, fee=0.0,
                 edge=fair - price, limit_price=price)
        return {"due": ts, "ts": ts, "mode": "lab", "variant": key, "fill": f, "seconds_left": 200, "price_age": 0,
                "book_age": 0, "fair_raw": fair, "stake": 5.0}

    def _bets(self, engine):
        return engine.store._x("SELECT variant, status, avg_price FROM bets WHERE mode = 'lab' ORDER BY id")

    def test_room_intent_and_a_standing_bid(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            # "m3": up to 3 cents more, within the 12-point edge
            t.lab_orders = [self._order("twap|0.12|0-300|em3")]
            t._arrive(100.0, [(0.52, 20.0)], [])
            self.assertEqual(self._bets(engine)[-1][:2], ("twap|0.12|0-300|em3", "filled"))
            # "i15": refused at first; an offer within the edge that stays a second buys it
            t.lab_orders = [self._order("twap|0.12|0-300|ei15")]
            t._arrive(100.0, [(0.60, 20.0)], [])
            self.assertEqual(len(t.lab_pending), 1)
            fair = {"twap": 0.70, "point": 0.70}
            t._pending_run(101.0, 199, fair, [(0.55, 20.0)], [])
            t._pending_run(101.5, 198, fair, [(0.55, 20.0)], [])
            self.assertEqual(len(t.lab_pending), 1)  # (not steady a second yet)
            t._pending_run(102.1, 197, fair, [(0.55, 20.0)], [])
            self.assertEqual(self._bets(engine)[-1][1:], ("filled", 0.55))
            # "s1": a bid a cent under the offer seen, filled when sellers go through it
            t._send(200.0, "lab", "twap|0.12|0-300|es1", self._order("x")["fill"], 200, 0, 0, 0.7, 5.0)
            self.assertEqual(t.lab_pending[-1]["bid"], 0.49)
            t._pending_run(201.0, 199, fair, [(0.48, 20.0)], [])
            v, status, price = self._bets(engine)[-1]
            self.assertEqual((v, status, price), ("twap|0.12|0-300|es1", "filled", 0.49))
            # ... and cancelled when our model no longer sees the edge at it
            t._send(300.0, "lab", "twap|0.12|0-300|es1", self._order("x")["fill"], 200, 0, 0, 0.7, 5.0)
            t._pending_run(301.0, 199, {"twap": 0.55, "point": 0.55}, [(0.48, 20.0)], [])
            self.assertEqual(self._bets(engine)[-1][1], "rejected")
            engine.store.close()

    def test_a_finding_clones_the_running_strategies_into_the_lab(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            engine.set_lab_focus(False)  # (the whole grid: the lab's focus on the locked pair is off)
            m = engine.main_strategies()[0]
            engine.race = {"intent": {"state": "found", "best": "15"}, "shading": {"bands": [{"state": "found", "best": "3"}]}}
            B = engine._bridge_tick(1000.0)
            self.assertEqual(set(B), {"race_intent|i15", "race_shading|m3"})
            clone = lab.Variant.from_key(B["race_intent|i15"]["keys"][0][2])
            self.assertEqual((clone.entry, clone.min_edge), ("i15", lab.Variant.from_key(m["variant"]).min_edge))
            keys = {v.key for v in engine._lab_variants(m["asset"], m["minutes"])}
            self.assertIn(B["race_intent|i15"]["keys"][0][2], keys)
            hub = {x["key"]: x for x in engine.hub_view()["tracks"]}
            self.assertEqual(hub["race_intent"]["bridge"][0]["code"], "i15")
            self.assertEqual(engine._bridge_tick(2000.0), B)  # (the same findings: nothing new)
            engine.store.close()

    def test_real_money_takes_every_entry_and_gate(self):
        from polybot.engine import _lab_only
        for key in ("twap|0.12|0-300|em3", "twap|0.12|0-300|ei15", "twap|0.12|0-300|es1", "twap|0.12|0-300|gf",
                    "twap|0.12|0-300|gk", "twap|0.12|0-300|gpmom"):
            _lab_only(key, "live")

    def test_the_gates(self):
        from collections import deque
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            v = lambda g: lab.Variant.from_key("twap|0.12|0-300|g" + g)
            self.assertTrue(t._gate_ok(v("f"), "down", 1000.0))  # (a source, not a filter)
            self.assertFalse(t._gate_ok(v("k"), "up", 1000.0))
            t.hot = lambda: True
            self.assertTrue(t._gate_ok(v("k"), "up", 1000.0))
            t.vol.samples = deque([(930.0, 100.0), (1000.0, 100.2)])
            t.price = 100.2
            self.assertEqual((t._gate_ok(v("pmom"), "up", 1000.0), t._gate_ok(v("pmom"), "down", 1000.0)), (True, False))
            t.on_market_trade("up", 0.5, 30, "SELL", 990.0)  # (selling Up: pushes Down)
            self.assertEqual((t._gate_ok(v("pflow"), "down", 1000.0), t._gate_ok(v("pflow"), "up", 1000.0)), (True, False))
            self.assertFalse(t._gate_ok(v("pbrk15"), "up", 1000.0))  # (no hour of prices yet: no breakout)
            engine.store.close()

    def test_the_futures_feed_only_the_rules_that_read_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            t.price, t.price_ts = 100.0, 1000.0
            t.on_fast(999.0, 50.0, "binance")
            t.on_fast(1000.5, 50.5, "binance")  # spot +1%
            t.on_fast(999.0, 50.0, "binance_fut")
            t.on_fast(1000.5, 51.0, "binance_fut")  # futures +2%
            self.assertAlmostEqual(t.nowcast(1000.6)[0], 101.0)  # (our model: the futures left out)
            self.assertAlmostEqual(t.nowcast(1000.6, only="binance_fut")[0], 102.0)
            self.assertFalse(t.uses_futures())
            t.variants = list(t.variants) + [lab.Variant.from_key("twap|0.12|0-300|gf")]
            self.assertTrue(t.uses_futures())
            engine.store.close()

    def _stub_live(self, t, calls):
        stub = types.SimpleNamespace(approval=lambda: False, order_cfg=lambda cfg: cfg, blocked=lambda *a, **k: None,
                                     trades=lambda *a: True, note_skip=lambda *a, **k: None)
        t.live = stub
        t.dispatch = lambda fn, *a: calls.append((fn.__name__, a))
        t.market = {"up_token": "U", "down_token": "D", "tick_size": 0.01, "min_order_size": 1}

    def test_real_money_keeps_a_refused_order_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            calls = []
            self._stub_live(t, calls)
            rule = lab.Variant.from_key("twap|0.12|0-300|ei15")
            t.live_intents[rule.key] = {"until": 120.0, "steady": None, "window": t.window}
            asks = [(0.55, 50.0)]
            t._live_bet(100.0, 200, rule, 0.75, asks, [])
            t._live_bet(100.5, 199, rule, 0.75, asks, [])
            self.assertEqual(calls, [])  # (not steady a second yet)
            t._live_bet(101.1, 198, rule, 0.75, asks, [])
            self.assertEqual([c[0] for c in calls], ["_live_orders"])
            self.assertEqual(calls[0][1][-1], 120.0)  # (refused again: open until the same time)
            self.assertNotIn(rule.key, t.live_intents)
            t.live_intents[rule.key] = {"until": 120.0, "steady": None, "window": t.window}
            t._live_bet(130.0, 170, rule, 0.75, asks, [])  # (its time is over)
            self.assertNotIn(rule.key, t.live_intents)
            engine.store.close()

    def test_real_money_stands_a_bid_in_the_book(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            calls = []
            self._stub_live(t, calls)
            rule = lab.Variant.from_key("twap|0.12|0-300|es1")
            t._live_stand(rule, 100.0, 200, {"twap": 0.75}, [(0.55, 50.0)], [])
            want = t.live_maker_want[rule.key]
            self.assertEqual((want["side"], want["price"]), ("up", 0.54))
            self.assertEqual([c[0] for c in calls], ["_live_maker_place"])  # (a real resting bid)
            t.live_maker_have[rule.key].update(busy=False, order_id="0x1")
            t._live_stand(rule, 105.0, 195, {"twap": 0.60}, [(0.55, 50.0)], [])  # (the edge at it is gone)
            self.assertNotIn(rule.key, t.live_maker_want)
            self.assertEqual(calls[-1][0], "_live_maker_close")  # (cancelled)
            engine.store.close()

    def test_the_gates_findings_are_cloned_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            engine.streaks = {"state": "found", "best": 3600}
            engine.predict = {"state": "found", "best": {"feature": "flow", "moment": 60}}
            B = engine._bridge_tick(1000.0)
            self.assertEqual(set(B), {"streaks|gk", "predict|gpflow"})
            self.assertEqual(lab.Variant.from_key(B["predict|gpflow"]["keys"][0][2]).gate, "pflow")
            now = time.time()
            for i in range(25):  # the lab's trades closing in the last hour, winning
                engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee, pnl)"
                                " VALUES ('lab', 'filled', ?, 'btc', 5, ?, 'up', 1, 0.5, 0, 0.4)", (now - 600, int(now - 900)))
            self.assertTrue(engine._hot_tick(now))
            self.assertTrue(engine.trackers[0].hot())
            engine.store.close()


class RoomDecisionTest(unittest.TestCase):
    def test_on_when_the_study_finds_it_then_stopped_by_real_losses_and_started_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            streaming=False, with_lab=False)
            engine.fills = {"room_state": "promising", "room": {}}
            self.assertEqual(engine._room_tick(1000.0)["cents"], 0)
            self.assertEqual(engine.live.room, 0)
            engine.fills = {"room_state": "found", "room": {"paper2": {"z": 1.0, "roi": 0.02}}}
            self.assertEqual(engine._room_tick(1000.0)["cents"], 1)  # 1 cent first
            for i in range(engine.ROOM_STOP_N):  # real trades that paid a cent more, losing
                engine.store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee,"
                                " avg_price, seen, pnl) VALUES ('live', 'filled', ?, 'btc', 15, ?, 'up', 10, 7.1, 0.1, 0.71, 0.70, -7.2)",
                                (1001.0 + i, i * 900))
            R = engine._room_tick(2000.0)
            self.assertEqual((R["cents"], R["stopped"]["n"], engine.live.room), (0, engine.ROOM_STOP_N, 0))
            self.assertEqual(engine._room_tick(3000.0)["cents"], 0)  # stays stopped
            engine.restart_room()
            self.assertEqual(engine.live.room, 1)  # the owner started it again: its count starts over
            engine.store.close()


class LiveUnknownFillTest(unittest.TestCase):
    setUp, tearDown, trader = LiveTest.setUp, LiveTest.tearDown, LiveTest.trader

    def test_an_answer_without_amounts_that_cant_be_checked_is_kept_as_ordered(self):
        t = self.trader()
        t.set_readiness(True)
        t.arm(True)
        t.MATCH_WAIT = 0
        t.client.response = types.SimpleNamespace(ok=True, order_id="0x3", status="matched",
                                                  making_amount=Decimal("0"), taking_amount=Decimal("0"))

        def down(order_id):
            raise RuntimeError("timeout")
        t.client.get_order = down
        self.assertEqual(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market), "filled")  # never lost


class HotPathReadsTest(unittest.TestCase):
    """Real money's way to an order reads its settings from memory and its trades on a connection of its own."""

    def test_settings_from_memory_follow_every_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            try:
                self.assertIsNone(store.get_setting("x"))
                store.set_setting("x", 1)
                self.assertEqual(store.get_setting("x"), "1")
                store._x("DELETE FROM settings WHERE key = 'x'")  # a raw write: read again
                self.assertEqual(store.get_setting("x", "gone"), "gone")
                store._x("INSERT OR REPLACE INTO settings VALUES ('x', '2')")
                self.assertEqual(store.get_setting("x"), "2")
                store.set_setting("x", None)
                self.assertIsNone(store.get_setting("x"))
            finally:
                store.close()

    def test_the_reading_connection_sees_what_was_just_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            try:
                store._x("INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                         " VALUES ('live', 'filled', 1, 'btc', 15, 0, 'up', 10, 5, 0.1)")
                self.assertEqual(store.read("SELECT COUNT(*) FROM bets WHERE mode = 'live'")[0][0], 1)
            finally:
                store.close()

    def test_a_strategys_order_has_its_own_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={},
                            streaming=False, with_lab=False)
            try:
                lanes = []

                class T:
                    def _live_orders(self):
                        import threading
                        lanes.append(threading.current_thread().name)

                    def _live_maker_place(self):
                        import threading
                        lanes.append(threading.current_thread().name)
                engine._dispatch(T()._live_orders)
                engine._dispatch(T()._live_maker_place)
                engine.taker_pool.shutdown(wait=True)
                engine.live_pool.shutdown(wait=True)
                self.assertEqual(sorted(x.split("_")[0] for x in lanes), ["live", "taker"])
            finally:
                engine.store.close()


class GcPausesTest(unittest.TestCase):
    def test_the_collectors_pauses_are_measured(self):
        import gc
        from polybot.health import Health
        h = Health.__new__(Health)
        import collections
        h.gc_pauses, h._gc_started = collections.deque(maxlen=10), None
        h.watch_gc()
        try:
            gc.collect()
            v = h.gc_view()
            self.assertGreaterEqual(v["n"], 1)
            self.assertGreaterEqual(v["full"], 1)
            self.assertIsNotNone(v["max"])
        finally:
            gc.callbacks[:] = [c for c in gc.callbacks if getattr(c, "__qualname__", "").split(".")[0] != "Health"]


class AtlasRealAccountTest(unittest.TestCase):
    """Connected with real money, the page's strip and day are the real account's (the owner, 28.10.2026), and
    real money's own limits hold new entries back."""

    class Gateway(AtlasToIbkrPaperTest.Gateway):
        def __init__(self):
            super().__init__(accounts=("U1234567",))
            self.acc = {"account": "U1234567", "currency": "USD", "net_liq": 10_500.0, "cash": 9_000.0, "buying_power": 18_000.0,
                        "daily": -40.0, "realized": 12.0, "unrealized": -52.0,
                        "positions": [{"symbol": "NVDA", "shares": 1.0, "avg_cost": 103.0, "price": 101.0, "value": 101.0,
                                       "unrealized": -2.0, "stop": 98.0}]}

        def account(self):
            return dict(self.acc)

    def _live(self, tmp):
        from polybot import equity
        A, eq = AtlasTest._atlas(self, tmp)
        book = equity.new_book(100_000)
        book.update(last="2026-10-26", pending=[("breakout", "AAPL", 100.0, 105.0)], curve=[("2026-10-26", 100_000)])
        A._set("paper", book)
        A.set_settings({"ibkr_mode": "live", "confirm_live": True})
        A.set_settings({"ibkr_on": True, "confirm_live": True})
        return A

    def test_the_strip_is_the_real_accounts_while_connected(self):
        from zoneinfo import ZoneInfo
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A = self._live(tmp)
            gw = self.Gateway()
            self.assertTrue(A.live_book(evening)["never"])  # not read yet: "not connected", never the simulation's
            A.ibkr_tick(evening, gw)
            L = A.live_book(evening + 30)
            self.assertEqual((L["account"], L["value"], L["today"], L["cash"]), ("U1234567", 10_500.0, -40.0, 9_000.0))
            self.assertEqual(L["positions"][0]["symbol"], "NVDA")
            self.assertAlmostEqual(L["open_risk"], 5.0)  # (103 - 98) x 1 share
            self.assertEqual(L["total"], 0.0)  # since the connection
            gw.acc["net_liq"] = 10_700.0
            A.ibkr_tick(evening + 7200, gw)
            self.assertAlmostEqual(A.live_book(evening + 7230)["total"], 200.0)
            self.assertFalse(A.live_book(evening + 7200 + 3600)["connected"])  # an old reading: marked, not "now"
            A.set_settings({"ibkr_mode": "paper"})
            self.assertIsNone(A.live_book(evening + 7230))  # the paper account: the simulation's strip

    def _atlas_holds(self, A, now, pnl=None):
        """An order of Atlas' own in NVDA (open), or closed today at pnl."""
        st = A._get("ibkr", {"orders": {}})
        st["orders"]["mine"] = {"key": "mine", "strategy": "breakout", "symbol": "NVDA", "qty": 1, "filled": 1, "stop": 98.0,
                                "ids": {"parent": 1}, "status": "open" if pnl is None else "closed", "sent": now - 60, "mode": "live",
                                "entry": 103.0, "entry_day": "2026-10-26", "pnl": pnl, "closed": None if pnl is None else now - 30}
        A._set("ibkr", st)

    def test_the_owners_limits_hold_new_entries(self):
        # (on Atlas' own book only: the owner's own positions and losses in the account aren't its limit)
        from zoneinfo import ZoneInfo
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A = self._live(tmp)
            A.set_settings({"live_size": 1000, "live_max_open": 1})
            self._atlas_holds(A, evening)
            gw = self.Gateway()
            A.ibkr_tick(evening, gw)
            self.assertEqual(gw.placed, [])  # Atlas holds NVDA, its limit is 1
            o = A._get("ibkr")["orders"]["breakout:AAPL:2026-10-26"]
            self.assertEqual(o["status"], "skipped")
            self.assertIn("המגבלה 1", o["why"])
        with tempfile.TemporaryDirectory() as tmp:
            A = self._live(tmp)
            A.set_settings({"live_size": 1000, "live_max_open": 1})
            gw = self.Gateway()
            A.ibkr_tick(evening, gw)  # NVDA is the owner's own: not Atlas' to count
            self.assertEqual(gw.placed[0][0], "AAPL")
        with tempfile.TemporaryDirectory() as tmp:
            A = self._live(tmp)
            A.set_settings({"live_size": 1000, "live_daily_loss": 30})
            self._atlas_holds(A, evening, pnl=-50.0)
            gw = self.Gateway()
            A.ibkr_tick(evening, gw)  # Atlas lost $50 today, its limit is $30
            self.assertEqual(gw.placed, [])
            self.assertIn("הפסד היומי", A._get("ibkr")["orders"]["breakout:AAPL:2026-10-26"]["why"])
        with tempfile.TemporaryDirectory() as tmp:
            A = self._live(tmp)
            A.set_settings({"live_size": 1000, "live_daily_loss": 30})
            gw = self.Gateway()
            A.ibkr_tick(evening, gw)  # the account is down $40 today, but not Atlas: it goes
            self.assertEqual(gw.placed[0][0], "AAPL")
        with tempfile.TemporaryDirectory() as tmp:
            A = self._live(tmp)
            A.set_settings({"live_size": 1000})
            gw = self.Gateway()
            A.ibkr_tick(evening, gw)  # no limits: it goes
            self.assertEqual(gw.placed[0][0], "AAPL")

    def test_the_limits_settings_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, _ = AtlasTest._atlas(self, tmp)
            A.set_settings({"live_daily_loss": 250, "live_max_open": 4})
            self.assertEqual((A.settings()["live_daily_loss"], A.settings()["live_max_open"]), (250.0, 4))
            for bad in ({"live_daily_loss": -1}, {"live_max_open": 1000}):
                with self.assertRaises(ValueError):
                    A.set_settings(bad)


class IbkrAccountReadingTest(unittest.TestCase):
    def test_the_account_in_dollars_with_its_positions_and_stops(self):
        from polybot import ibkr
        V = lambda tag, value, cur: types.SimpleNamespace(tag=tag, value=value, currency=cur, account="U1")
        trade = lambda sym, action, kind, aux, trail=0.0: types.SimpleNamespace(
            contract=types.SimpleNamespace(symbol=sym), order=types.SimpleNamespace(action=action, orderType=kind, auxPrice=aux, trailStopPrice=trail))

        class FakeIb:
            asked = []

            def managedAccounts(self):
                return ["U1"]

            def accountValues(self, acct):  # (an account kept in shekels: 3.7 to the dollar)
                return [V("NetLiquidation", "37000", "ILS"), V("TotalCashValue", "3700", "ILS"), V("BuyingPower", "74000", "ILS"),
                        V("ExchangeRate", "3.7", "USD"), V("ExchangeRate", "1", "ILS")]

            def reqPnL(self, acct):
                self.asked.append(acct)

            def sleep(self, s):
                pass

            def pnl(self, acct):
                return [types.SimpleNamespace(dailyPnL=-370.0, realizedPnL=float("nan"), unrealizedPnL=-185.0)]

            def openTrades(self):
                return [trade("NVDA", "SELL", "TRAIL", 5.0, 98.5), trade("NVDA", "SELL", "LMT", 120.0), trade("AAPL", "BUY", "LMT", 0)]

            def portfolio(self, acct):
                return [types.SimpleNamespace(contract=types.SimpleNamespace(symbol="NVDA"), position=2, marketPrice=101.0,
                                              marketValue=202.0, averageCost=103.0, unrealizedPNL=-4.0)]
        gw = ibkr.IbGateway(host="x", port=1)
        fake = FakeIb()
        gw.call = lambda fn, timeout=40: fn(fake)
        a = gw.account()
        self.assertEqual((a["account"], a["currency"]), ("U1", "ILS"))
        self.assertAlmostEqual(a["net_liq"], 10_000.0)
        self.assertAlmostEqual(a["cash"], 1_000.0)
        self.assertAlmostEqual(a["daily"], -100.0)
        self.assertIsNone(a["realized"])  # (IBKR's "no figure")
        self.assertEqual(a["positions"][0]["stop"], 98.5)  # the trailing stop where it stands now
        gw.account()
        self.assertEqual(fake.asked, ["U1"])  # today's profit asked once, then streamed


class IbkrRealTimePricesTest(unittest.TestCase):
    """Atlas' prices in real time from IBKR, Yahoo's when IBKR has none (the owner, 28.10.2026)."""

    def _fake_ib(self, price=101.5, data_type=1, age=2.0):
        import datetime as _dt
        now = time.time()

        class FakeIb:
            def __init__(self):
                self.subscribed, self.cancelled = [], []

            def reqMarketDataType(self, t):
                self.type = t

            def qualifyContracts(self, c):
                pass

            def reqMktData(self, c, *a):
                self.subscribed.append(c.symbol)
                return types.SimpleNamespace(marketPrice=lambda: price, marketDataType=data_type, close=100.0,
                                             time=_dt.datetime.fromtimestamp(now - age, _dt.timezone.utc))

            def reqHistoricalData(self, c, *a, **kw):
                d = _dt.datetime.fromtimestamp(now - 600, _dt.timezone.utc)
                return [types.SimpleNamespace(date=d, open=100.0, high=102.0, low=99.0, close=101.0)]

            def cancelMktData(self, c):
                self.cancelled.append(c.symbol)

            def cancelHistoricalData(self, b):
                pass

            def sleep(self, s):
                pass
        return FakeIb(), now

    def _gateway(self, fake):
        from polybot import ibkr
        gw = ibkr.IbGateway(host="x", port=1)
        gw.call = lambda fn, timeout=40: fn(fake)
        return gw

    def test_a_real_time_price_and_todays_bars(self):
        fake, now = self._fake_ib()
        gw = self._gateway(fake)
        q = gw.quote("NVDA", now)
        self.assertEqual((q["price"], q["source"], fake.type, q["prev_close"]), (101.5, "ibkr", 1, 100.0))
        self.assertEqual(q["bars"][0][1:], (100.0, 102.0, 99.0, 101.0))
        gw.quote("NVDA", now + 5)
        self.assertEqual(fake.subscribed, ["NVDA"])  # subscribed once, then streamed

    def test_no_real_time_price_is_none(self):
        for kw in ({"price": float("nan")}, {"data_type": 3}, {"age": 600}):
            fake, now = self._fake_ib(**kw)
            self.assertIsNone(self._gateway(fake).quote("NVDA", now), kw)

    def test_the_least_used_stream_makes_room(self):
        fake, now = self._fake_ib()
        gw = self._gateway(fake)
        gw.QUOTE_LINES = 2
        gw.quote("A", now)
        gw.quote("B", now + 1)
        gw.quote("A", now + 2)
        gw.quote("C", now + 3)
        self.assertEqual(fake.cancelled, ["B"])
        self.assertEqual(sorted(gw._live), ["A", "C"])

    def test_atlas_takes_ibkrs_price_and_falls_back_to_yahoo(self):
        from unittest import mock
        from polybot import atlas
        with tempfile.TemporaryDirectory() as tmp:
            A, _ = AtlasTest._atlas(self, tmp)
            gw = types.SimpleNamespace(connected=lambda: True, quote=lambda sym, now=None: {"price": 10.0, "ts": 1, "bars": [], "source": "ibkr"})
            A.ibkr = gw
            yahoo = mock.patch.object(atlas, "fetch_intraday", return_value={"price": 9.9, "ts": 1, "bars": []})
            with yahoo:
                self.assertEqual(A.quote("NVDA")["source"], "ibkr")
                self.assertEqual(A.status["quotes_source"], "ibkr")
                gw.quote = lambda sym, now=None: None  # (no subscription for it)
                q = A.quote("NVDA")
                self.assertEqual((q["source"], q["price"]), ("yahoo", 9.9))
                self.assertIn("אין מחיר בזמן אמת", A.status["ibkr_data_error"])

                def down(sym, now=None):
                    raise TimeoutError("IB Gateway didn't answer")
                gw.quote = down
                self.assertEqual(A.quote("NVDA")["source"], "yahoo")
                A.ibkr = None  # (no gateway at all, not on the server: Yahoo, nothing tried)
                self.assertEqual(A.quote("NVDA")["source"], "yahoo")


class AtlasLivePositionHandsTest(unittest.TestCase):
    """The owner's hands on a real position (28.10.2026): close it, add 25/50/100%; and the dashboard by
    the mode, with the connection's state."""

    class Gateway(AtlasRealAccountTest.Gateway):
        def close_position(self, symbol, session):
            self.closed = getattr(self, "closed", []) + [(symbol, session)]
            self.state[777] = {"status": "Submitted", "filled": 0.0, "avg_price": 0.0}
            return {"order": 777, "shares": 1.0}

    def _open(self, tmp):
        from zoneinfo import ZoneInfo
        A = AtlasRealAccountTest._live(self, tmp)
        A.set_settings({"live_size": 1000})
        gw = self.Gateway()
        ny = ZoneInfo("America/New_York")
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ny).timestamp()
        A.ibkr_tick(evening, gw)  # AAPL sent for the open
        gw.state[100] = {"status": "Filled", "filled": 9, "avg_price": 104.0}
        morning = dt.datetime(2026, 10, 27, 10, 0, tzinfo=ny).timestamp()
        A.ibkr_tick(morning, gw)
        A.ibkr = gw
        return A, gw, morning

    def test_close_sends_the_sale_and_the_trade_closes_when_it_fills(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, gw, morning = self._open(tmp)
            A._gateway = lambda: gw
            r = A.close_live("AAPL", morning + 60)
            self.assertEqual((r["order"], r["at"], gw.closed), (777, "now", [("AAPL", True)]))
            o = A._get("ibkr")["orders"]["breakout:AAPL:2026-10-26"]
            self.assertEqual((o["ids"]["exit"], o["exit_why"]), (777, "manual"))
            gw.state[777] = {"status": "Filled", "filled": 9, "avg_price": 110.0}
            A.ibkr_tick(morning + 120, gw)
            o = A._get("ibkr")["orders"]["breakout:AAPL:2026-10-26"]
            self.assertEqual((o["status"], o["why"], o["exit"]), ("closed", "manual", 110.0))

    def test_add_sends_a_bracket_at_the_positions_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, gw, morning = self._open(tmp)
            A._gateway = lambda: gw
            gw.acc["positions"] = [{"symbol": "AAPL", "shares": 9.0, "avg_cost": 104.0, "price": 106.0, "value": 954.0,
                                    "unrealized": 18.0, "stop": 101.0}]
            r = A.add_live("AAPL", 0.5, morning + 60)
            self.assertEqual((r["added"], r["stop"], r["at"]), (4, 101.0, "now"))
            sym, qty, limit, target, stop = gw.placed[-1]
            self.assertEqual((sym, qty, stop), ("AAPL", 4, 101.0))
            with self.assertRaises(ValueError):
                A.add_live("AAPL", 0.3, morning + 61)  # only 25 / 50 / 100%
            gw.acc["positions"][0]["stop"] = None
            gw.acc["positions"][0]["price"] = 100.0
            A._get("ibkr")  # (the system's own stop is 100 there: the price is at it)
            with self.assertRaises(ValueError):
                A.add_live("AAPL", 0.25, morning + 62)

    def test_the_dashboard_follows_the_mode_with_the_connection_state(self):
        from zoneinfo import ZoneInfo
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)
            self.assertEqual(A.live_book(evening)["never"], True)  # real money on, not read yet: not the simulation's
            A.ibkr_tick(evening, AtlasRealAccountTest.Gateway())
            L = A.live_book(evening + 30)
            self.assertTrue(L["connected"])
            L = A.live_book(evening + 600)
            self.assertFalse(L["connected"])  # the last reading, marked as not connected
            self.assertEqual(L["value"], 10_500.0)


class AtlasWhyQuietTest(unittest.TestCase):
    def test_every_gate_is_reported(self):
        from polybot import equity
        from zoneinfo import ZoneInfo
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            last = eq.prepared["SPY"]["day"][-1]
            book = equity.new_book(100_000)
            book.update(last=last, pending=[("breakout", "NVDA", 90.0, 110.0, ["stop", 101.0])], curve=[(last, 100_000)])
            book["open"] = [{"symbol": f"S{i}", "day": last} for i in range(10)]
            A._set("paper", book)
            q = A.why_quiet(dt.datetime(2030, 1, 7, 12, 0, tzinfo=ZoneInfo("America/New_York")).timestamp())
            self.assertFalse(q["data_ok"])  # (the data's last day is long before)
            self.assertEqual((q["open"], q["max_open"], q["free"]), (10, 10, 0))
            self.assertEqual(q["pending"][0]["symbol"], "NVDA")
            self.assertEqual((q["pending"][0]["order"], q["pending"][0]["trigger"]), ("stop", 101.0))
            self.assertIn("market_ok", q)
            self.assertIsNotNone(A._quiet_safe())


class AtlasMailsByModeTest(unittest.TestCase):
    """Real money and the simulation, each its own mail settings and quiet hours (the owner, 28.10.2026)."""

    def test_quiet_hours_hold_only_their_own_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            T = AtlasMailsTest()
            A, M, sent, engine = T._setup(tmp)
            now = time.time()
            M.update({"quiet": True, "quiet_from": "00:00", "quiet_to": "23:59"})  # (the simulation's only)
            A.queue_mails("sim", [T._trade(1)], [], now=now)
            A.queue_mails("live", [T._trade(2)], [], now=now)
            M.tick(now + 60, True)
            self.assertEqual(len(sent), 1)
            self.assertIn("כסף אמיתי", sent[0][0])  # real money's went; the simulation's waits
            self.assertEqual([m["mode"] for m in A.peek_mails()], ["sim"])
            M.update({"quiet": False})
            M.tick(now + 120, True)
            self.assertIn("בשעות השקטות", sent[-1][0])
            self.assertNotIn("כסף אמיתי", sent[-1][0])
            engine.store.close()

    def test_real_moneys_settings_start_as_the_shared_ones_were(self):
        with tempfile.TemporaryDirectory() as tmp:
            T = AtlasMailsTest()
            A, M, sent, engine = T._setup(tmp)
            A._set("mail_settings", {"quiet": True, "quiet_from": "01:00", "quiet_to": "07:00", "daily_time": "23:00"})
            s = M.settings()
            self.assertEqual((s["quiet_live"], s["quiet_from_live"], s["quiet_to_live"], s["daily_time_live"]), (True, "01:00", "07:00", "23:00"))
            M.update({"quiet_live": False})
            self.assertEqual((M.settings()["quiet_live"], M.settings()["quiet"]), (False, True))
            engine.store.close()

    def test_same_from_and_to_is_no_quiet_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            T = AtlasMailsTest()
            A, M, sent, engine = T._setup(tmp)
            M.update({"quiet_live": True, "quiet_from_live": "05:00", "quiet_to_live": "05:00"})
            self.assertFalse(M.quiet_now(time.time(), sfx="_live"))
            engine.store.close()

    def test_the_daily_summary_is_kept_by_its_trading_day_and_survives_quiet_hours(self):
        from zoneinfo import ZoneInfo
        from polybot.atlas_mail import AtlasMail
        with tempfile.TemporaryDirectory() as tmp:
            T = AtlasMailsTest()
            A, M, sent, engine = T._setup(tmp)
            M = AtlasMail(engine.notifier, A)  # (the real maybe_daily)
            A.paper_step()
            il = ZoneInfo(engine.cfg.timezone)
            at = lambda d, h, m: dt.datetime(2026, 10, d, h, m, tzinfo=il).timestamp()
            M.update({"daily_time": "23:30", "quiet": True, "quiet_from": "23:00", "quiet_to": "08:00"})
            M.tick(at(26, 12, 0), True)  # Monday noon: the first time, from now on (no summary of a day before)
            self.assertEqual(sent, [])
            M.tick(at(27, 23, 31), True)  # Tuesday 23:31: in the quiet hours, held
            self.assertEqual(sent, [])
            M.tick(at(28, 8, 1), True)  # Wednesday 08:01: the quiet ended: Tuesday's summary goes now
            self.assertEqual(len(sent), 1)
            self.assertIn("סיכום היום", sent[0][0])
            M.tick(at(28, 9, 0), True)
            self.assertEqual(len(sent), 1)  # once
            M.update({"quiet": False})
            M.tick(at(31, 23, 40), True)  # Saturday night: Friday's (the 30th) goes; a weekend has none of its own
            M.tick(at(31, 23, 50), True)
            self.assertEqual(len(sent), 2)
            engine.store.close()

    def test_real_moneys_daily_summary_is_the_real_accounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            T = AtlasMailsTest()
            A, M, sent, engine = T._setup(tmp)
            nvda = [{"symbol": "NVDA", "shares": 1.0, "avg_cost": 103.0, "price": 101.0, "unrealized": -2.0}]
            A.live_book = lambda now=None: {"connected": True, "account": "U1", "value": 10_500.0, "today": -40.0, "cash": 9_000.0,
                                            "closed": [], "atlas": {"positions": nvda, "today": -2.0, "week": 120.0, "month": 500.0,
                                                                    "open_risk": 5.0, "value": 101.0}}
            subject, body = M.daily_mail(time.time(), "live")
            self.assertIn("כסף אמיתי", subject)
            self.assertIn("NVDA", body)
            self.assertIn("U1", body)
            engine.store.close()


class AtlasRealTimeLoopTest(unittest.TestCase):
    def test_every_few_seconds_the_account_and_the_prices(self):
        from zoneinfo import ZoneInfo
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)
            gw = AtlasRealAccountTest.Gateway()
            gw.connected = lambda: True
            gw.quote = lambda sym, now=None: {"price": 104.0, "ts": now, "bars": [], "source": "ibkr"}
            A.ibkr = gw
            session = dt.datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
            self.assertTrue(A.live_tick(session))
            self.assertEqual(A.quotes["NVDA"]["price"], 104.0)
            L = A.live_book(session + 1)
            p = L["positions"][0]
            self.assertEqual((p["price"], p["unrealized"]), (104.0, 1.0))  # (104 - 103) x 1, at the streamed price
            self.assertAlmostEqual(L["value"], 10_503.0)  # the account's value moved with it (101 -> 104)
            gw.connected = lambda: False
            self.assertFalse(A.live_tick(session + 5))  # no gateway: nothing (Yahoo stays on the minute)

    def test_a_paper_login_is_never_shown_as_the_real_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)
            gw = AtlasRealAccountTest.Gateway()
            gw.acc["account"] = "DU999"
            self.assertIsNone(A._read_account(gw, "live", time.time()))
            self.assertTrue(A.live_book()["never"])


class AtlasSystemTestTradeTest(unittest.TestCase):
    """The owner's check of the whole way: one share bought, sold seconds later, recorded like any trade."""

    def test_one_share_in_and_out(self):
        from zoneinfo import ZoneInfo
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)
            gw = AtlasRealAccountTest.Gateway()
            sent = []

            def market(symbol, action, qty):
                oid = 500 + len(sent)
                sent.append((symbol, action, qty))
                gw.state[oid] = {"status": "Filled", "filled": qty, "avg_price": 600.0 if action == "BUY" else 601.5}
                return oid
            gw.market = market
            A._gateway = lambda: gw
            session = dt.datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
            T = A.test_round_trip("meta", session, hold=0)
            self.assertEqual(T["step"], "done", T.get("error"))
            self.assertEqual(sent, [("META", "BUY", 1), ("META", "SELL", 1)])
            self.assertEqual((T["bought"], T["sold"], T["pnl"]), (600.0, 601.5, -0.5))  # +$1.50, $2 of commission
            o = next(o for o in A._get("ibkr")["orders"].values() if o["strategy"] == "test")
            self.assertEqual((o["status"], o["why"], o["entry"], o["exit"]), ("closed", "test", 600.0, 601.5))
            kinds = [(m["kind"], m["mode"]) for m in A.peek_mails()]
            self.assertIn(("open", "live"), kinds)
            self.assertIn(("close", "live"), kinds)  # (its mails, as any real trade's)
            with self.assertRaises(ValueError):  # outside the session: a market order wouldn't go
                A.test_round_trip("META", dt.datetime(2026, 10, 27, 20, 0, tzinfo=ZoneInfo("America/New_York")).timestamp(), hold=0)

    def test_the_positions_say_who_bought_them_and_todays_move(self):
        from zoneinfo import ZoneInfo
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)
            gw = AtlasRealAccountTest.Gateway()
            gw.acc["positions"].append({"symbol": "BBAI", "shares": 100.0, "avg_cost": 3.0, "price": 2.5, "value": 250.0,
                                        "unrealized": -50.0, "stop": None})
            evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
            A.ibkr_tick(evening, gw)
            st = A._get("ibkr")
            st["orders"]["x"] = {"key": "x", "strategy": "breakout", "symbol": "NVDA", "qty": 1, "filled": 1, "stop": 98.0,
                                 "ids": {}, "status": "open", "sent": evening, "mode": "live", "entry": 103.0}
            A._set("ibkr", st)
            A.quotes["NVDA"] = {"price": 101.0, "ts": evening + 1, "bars": [], "source": "ibkr", "prev_close": 100.0}
            L = A.live_book(evening + 2)
            by = {p["symbol"]: p for p in L["positions"]}
            self.assertEqual((by["NVDA"]["source"], by["BBAI"]["source"]), ("system", "manual"))
            self.assertAlmostEqual(by["NVDA"]["day_pct"], 0.01)
            self.assertIsNone(by["BBAI"]["day_pct"])  # (no price stream for it yet: no day's move shown)


class AtlasStopForAPositionTest(unittest.TestCase):
    def test_a_stop_on_a_position_without_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)
            gw = AtlasRealAccountTest.Gateway()
            gw.acc["positions"] = [{"symbol": "BBAI", "shares": 100.0, "avg_cost": 3.0, "price": 2.5, "value": 250.0,
                                    "unrealized": -50.0, "stop": None}]
            placed = []
            gw.place_stop = lambda sym, qty, stop, trail_pct=None: placed.append((sym, qty, stop, trail_pct)) or 900
            A._gateway = lambda: gw
            r = A.set_stop("BBAI", "trail", 3)
            self.assertEqual(placed[-1], ("BBAI", 100.0, 2.42, 3.0))  # 3% under $2.50, trailing from there
            self.assertEqual(r["stop"], 2.42)
            A.set_stop("BBAI", "fixed", 10)
            self.assertEqual(placed[-1], ("BBAI", 100.0, 2.25, None))
            for bad in (("trail", 7), ("fixed", 80), ("other", 3)):
                with self.assertRaises(ValueError):
                    A.set_stop("BBAI", *bad)
            gw.acc["positions"][0]["stop"] = 2.3
            with self.assertRaises(ValueError):  # it has one already
                A.set_stop("BBAI", "trail", 2)


class AtlasOwnBookTest(unittest.TestCase):
    def test_the_account_in_two_books(self):
        from zoneinfo import ZoneInfo
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)
            gw = AtlasRealAccountTest.Gateway()
            gw.acc["positions"] = [{"symbol": "NVDA", "shares": 3.0, "avg_cost": 102.0, "price": 110.0, "value": 330.0, "unrealized": 24.0, "stop": 98.0},
                                   {"symbol": "BBAI", "shares": 100.0, "avg_cost": 3.0, "price": 2.5, "value": 250.0, "unrealized": -50.0, "stop": None}]
            A.ibkr_tick(evening, gw)
            st = A._get("ibkr")
            st["orders"]["mine"] = {"key": "mine", "strategy": "breakout", "symbol": "NVDA", "qty": 1, "filled": 1, "stop": 98.0, "ids": {},
                                    "status": "open", "sent": evening, "mode": "live", "entry": 100.0, "entry_day": "2026-10-20"}
            st["orders"]["done"] = {"key": "done", "strategy": "pullback", "symbol": "MSFT", "qty": 1, "filled": 1, "stop": 1.0, "ids": {},
                                    "status": "closed", "sent": evening - 3600, "mode": "live", "entry": 400.0, "pnl": 12.0, "closed": evening - 60}
            A._set("ibkr", st)
            A.quotes["NVDA"] = {"price": 110.0, "ts": evening, "bars": [], "source": "ibkr", "prev_close": 108.0}
            L = A.live_book(evening + 1)
            a, m = L["atlas"], L["manual"]
            self.assertEqual([(p["symbol"], p["shares"]) for p in a["positions"]], [("NVDA", 1)])  # one of the three is Atlas'
            self.assertEqual(sorted((p["symbol"], p["shares"]) for p in m["positions"]), [("BBAI", 100.0), ("NVDA", 2.0)])
            self.assertAlmostEqual(a["open_pnl"], 10.0)  # its own entry, 100 -> 110
            self.assertAlmostEqual(a["today"], 12.0 + 2.0)  # MSFT's 12 closed today, NVDA's +2 since yesterday's close
            self.assertAlmostEqual(a["open_risk"], 2.0)  # (100 - 98) x 1: the owner's BBAI without a stop isn't Atlas' risk
            self.assertAlmostEqual(m["open_risk"], 2 * (102 - 98) + 250.0)


class AtlasAuditTest(unittest.TestCase):
    """The audit of every calculation (the owner, 28.10.2026)."""

    def test_a_clean_backtest_passes_and_doctored_trades_are_caught(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            plans, port = A.plans(), A.port()
            r = equity.portfolio(eq.prepared, plans, port)
            a = equity.audit_trades(eq.prepared, r["trades"], plans, port, r["curve"])
            self.assertEqual(a["problems"], [], a["problems"][:3])
            self.assertGreater(a["checks"], len(r["trades"]))
            t = dict(next(t for t in r["trades"] if t["why"] not in ("open",)))
            wrong = [dict(t, pnl=t["pnl"] + 500), dict(t, id="x", exit=t["exit"] * 3)]
            kinds = {p["kind"] for p in equity.audit_trades(eq.prepared, wrong, plans, port)["problems"]}
            self.assertIn("pnl", kinds)
            self.assertIn("exit", kinds)
            self.assertIn("twice", kinds)  # (the same stock twice at once)
            out = A.run_audit()
            self.assertTrue(out["ok"])
            self.assertEqual(A._get("audit")["ts"], out["ts"])


class AtlasLiveSignalsTest(unittest.TestCase):
    def test_a_full_simulation_doesnt_starve_real_money(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            A.set_settings({"regime": "off"})
            spy = eq.prepared["SPY"]
            day = next(d for d in reversed(spy["day"][:-1]) if equity.signals_for(eq.prepared, d, A.plans(), A.port()))
            every = equity.signals_for(eq.prepared, day, A.plans(), A.port())
            book = equity.new_book(100_000)
            book.update(last=day, pending=[], open=[{"symbol": f"S{i}"} for i in range(10)])  # (the simulation is full)
            got = A.live_signals(book)
            self.assertEqual([x[1] for x in got], [x[1] for x in every][:10])  # real money's own room: 10
            A.set_settings({"live_max_open": 1})
            self.assertEqual(len(A.live_signals(book)), 1)

    def test_the_most_a_real_trade_may_lose(self):
        from zoneinfo import ZoneInfo
        evening = dt.datetime(2026, 10, 26, 17, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)  # (AAPL at 105, its stop 100)
            A.set_settings({"live_size": 5000, "live_risk": 20})
            A.live_signals = lambda book=None: [("breakout", "AAPL", 100.0, 105.0)]
            gw = AtlasRealAccountTest.Gateway()
            A.ibkr_tick(evening, gw)
            self.assertEqual(gw.placed[0][1], 4)  # $20 at most to the stop, $5 a share: 4 shares (not $5,000's 47)


class MentorSignalsTest(unittest.TestCase):
    """The mentor's Discord signals (the owner, 28.10.2026): read, traded in the account in a book of their own."""

    class Gateway(AtlasRealAccountTest.Gateway):
        def __init__(self):
            super().__init__()
            self.prices, self.markets = {}, []

        def quote(self, symbol, now=None):
            p = self.prices.get(symbol)
            return p and {"price": p, "ts": now or time.time(), "bars": [], "source": "ibkr"}

        def market(self, symbol, action, qty):
            self.markets.append((symbol, action, qty))
            oid = 5000 + len(self.markets)
            self.state[oid] = {"status": "Submitted", "filled": 0.0, "avg_price": 0.0}
            return oid

    def _setup(self, tmp):
        from zoneinfo import ZoneInfo
        A = AtlasRealAccountTest._live(self, tmp)
        A.set_settings({"live_size": 1000, "live_risk": 0})
        gw = self.Gateway()
        A._gateway = lambda: gw
        A.ibkr = gw
        morning = dt.datetime(2026, 10, 27, 10, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
        A._read_account(gw, "live", morning)
        return A, gw, morning

    def _fill(self, A, gw, at):
        for o in A._get("ibkr")["orders"].values():
            if o.get("strategy") == "mentor" and o["status"] == "sent":
                gw.state[o["ids"]["parent"]] = {"status": "Filled", "filled": o["qty"], "avg_price": gw.prices[o["symbol"]]}
        A.ibkr_tick(at, gw)

    def test_reads_his_messages(self):
        from polybot.mentor import parse
        P = lambda t, c=None: parse(t, c)
        btc = P("ביטקוין שורט\n\nכניסה מיידית\n\nסיכון 0.25% מהתיק\n\nSTOP\n83200\n\nTP\n65600\n\n@everyone")
        self.assertEqual((btc["kind"], btc["symbol"], btc["side"], btc["asset"], btc["stop"], btc["tp"], btc["risk_pct"]),
                         ("enter", "BTC", "short", "future", 83200, 65600, 0.25))
        nq = P("נסדק שורט\nכניסה מיידית\nסיכון 0.15% מהתיק\nSTOP\n450 נקודות\nTP\n2000 נקודות")
        self.assertEqual((nq["symbol"], nq["stop_pts"], nq["tp_pts"]), ("NASDAQ", 450, 2000))
        self.assertEqual(P("זהב לונג, עסקה ספקולטיבית סיכון מזערי.\nכניסה מיידית\nסיכון 0.05% מהתיק\nSTOP\n4018\nTP\n4128")["stop"], 4018)
        cases = {"IONQ מפחית לכמות קטנה": ("reduce", "IONQ", 0.25, None), "ZETA הפחתתי לכמות קטנה": ("reduce", "ZETA", 0.25, None),
                 "VRT מעל 240 אכנס חצי כמות": ("enter", "VRT", 0.5, ("above", 240)), "SOFI כמות מלאה": ("enter", "SOFI", 1.0, None),
                 "SOFI כמות כפולה": ("enter", "SOFI", 2.0, None), "IONQ בפנים חצי כמות לפי תרחיש מהסשן": ("enter", "IONQ", 0.5, None),
                 "ZETA נכנסתי שליש כמות": ("enter", "ZETA", 1 / 3, None), "ENVX אכנס שלי שכמות\nמניה למתקדמים": ("enter", "ENVX", 1 / 3, None),
                 "XLAB מתחת 3.8 אסגור": ("exit", "XLAB", None, ("below", 3.8)), "ENVX 🔥": ("unclear", "ENVX", None, None)}
        for text, (kind, sym, size, trig) in cases.items():
            p = P(text)
            self.assertEqual((p["kind"], p["symbol"]), (kind, sym), text)
            self.assertAlmostEqual(p["size"] or 0, size or 0, msg=text)
            self.assertEqual(p["trigger"] and (p["trigger"]["dir"], p["trigger"]["price"]), trig, text)
        # (a line without a stock is about his last one)
        self.assertEqual((P("אם לא תחזיק 15.4 אפחית", "SOFI")["kind"], P("אם לא תחזיק 15.4 אפחית", "SOFI")["symbol"]), ("reduce", "SOFI"))
        # what he watches, holds or says happened is never a trade
        for t in ("ראיתם את התרחיש היחיד שנכתב פה אתמול ל be?\nריטסט מעולה ל202.5 ועכשיו כבר ב 213 🔥",
                  "המניות שכרגע על הפרק מבחינתי לפתיחת עסקה be zeta. רק למעקב בינתיים\nכרגע מחזיק רק sofi כמות קטנה ו envx כמות קטנה",
                  "שימו לב כרגע אין משהו מיוחד מבחינתי אני כמובן עוקב."):
            self.assertEqual(P(t)["kind"], "info", t)

    def test_in_add_and_down_to_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, gw, t0 = self._setup(tmp)
            gw.prices["IONQ"] = 50.0
            s = A.mentor.receive("IONQ בפנים חצי כמות", "Raz Gamliel", now=t0, run=False)
            self.assertEqual(s["status"], "go")
            s = A.mentor.execute(s["id"], t0)
            self.assertEqual(s["status"], "done", s.get("why"))
            self.assertEqual(gw.placed[-1], ("IONQ", 10, None, None, 47.0))  # $1,000 x half / $50; a 6% stop
            self._fill(A, gw, t0 + 30)
            s = A.mentor.receive("IONQ כמות מלאה", "Raz Gamliel", now=t0 + 60, run=False)
            A.mentor.execute(s["id"], t0 + 60)
            self.assertEqual(gw.placed[-1][:2], ("IONQ", 10))  # the other half
            self._fill(A, gw, t0 + 90)
            self.assertAlmostEqual(A.mentor.units("IONQ"), 1.0)
            s = A.mentor.receive("IONQ הפחתתי לכמות קטנה", "Raz Gamliel", now=t0 + 120, run=False)
            s = A.mentor.execute(s["id"], t0 + 120)
            self.assertEqual(s["result"]["sold"], 15)  # 20 shares down to a quarter
            self.assertEqual([m[1:] for m in gw.markets], [("SELL", 10), ("SELL", 5)])  # the latest first
            self.assertIn({"totalQuantity": 5.0}, [kw for _, kw in gw.modified])  # the stop left on 5 shares, never sells twice
            self.assertAlmostEqual(A.mentor.units("IONQ"), 0.25)
            # (the sales fill: the trades close at their price)
            for oid in (5001, 5002):
                gw.state[oid] = {"status": "Filled", "filled": 0, "avg_price": 53.0}
            A.ibkr_tick(t0 + 150, gw)
            closed = [o for o in A._get("ibkr")["orders"].values() if o.get("strategy") == "mentor" and o["status"] == "closed"]
            self.assertEqual(sorted(o["filled"] for o in closed), [5, 10])
            self.assertTrue(all(o["why"] == "mentor" and o["exit"] == 53.0 for o in closed))
            s = A.mentor.receive("XLAB מתחת 3.8 אסגור", "Raz Gamliel", now=t0 + 200, run=False)
            self.assertEqual(s["status"], "armed")

    def test_a_price_he_names_waits_for_the_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, gw, t0 = self._setup(tmp)
            s = A.mentor.receive("VRT מעל 240 אכנס חצי כמות", "Raz Gamliel", now=t0)
            self.assertEqual(s["status"], "armed")
            self.assertIn("VRT", A.mentor.watch())
            gw.prices["VRT"] = 239.0
            A.quotes["VRT"] = {"price": 239.0, "ts": t0 + 1, "bars": [], "source": "ibkr"}
            self.assertEqual(A.mentor.tick(t0 + 1), [])
            gw.prices["VRT"] = 241.0
            A.quotes["VRT"] = {"price": 241.0, "ts": t0 + 2, "bars": [], "source": "ibkr"}
            self.assertEqual(A.mentor.tick(t0 + 2), [s["id"]])
            self.assertEqual(gw.placed[-1][:2], ("VRT", 2))  # $500 / $241
            self.assertEqual(A.mentor._state()["signals"][s["id"]]["status"], "done")

    def test_what_is_not_clear_waits_for_the_owner_and_futures_wait_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, gw, t0 = self._setup(tmp)
            gw.prices["ENVX"] = 10.0
            s = A.mentor.receive("ENVX 🔥", "Raz Gamliel", now=t0)
            self.assertEqual(s["status"], "waiting")
            self.assertEqual(gw.placed, [])
            s = A.mentor.decide(s["id"], True, 1 / 3, t0 + 5)
            self.assertEqual((s["status"], gw.placed[-1][:2]), ("done", ("ENVX", 33)))
            n = len(gw.placed)
            s = A.mentor.receive("ביטקוין שורט\nכניסה מיידית\nסיכון 0.25% מהתיק\nSTOP\n83200\nTP\n65600", "Raz Gamliel", now=t0 + 10)
            self.assertEqual((s["status"], len(gw.placed)), ("unsupported", n))
            s = A.mentor.receive("מישהו אחר: SOFI כמות מלאה", "Someone", now=t0 + 20)
            self.assertEqual(s["status"], "ignored")
            A.set_settings({"mentor_auto": False})
            s = A.mentor.receive("SOFI כמות מלאה", "Raz Gamliel", now=t0 + 30)
            self.assertEqual(s["status"], "waiting")
            self.assertEqual(A.mentor.decide(s["id"], False)["status"], "declined")
            self.assertEqual(A.mentor.receive("SOFI כמות מלאה", "Raz Gamliel", now=t0 + 31)["duplicate"], True)
            A.set_settings({"mentor_on": False})
            self.assertEqual(A.mentor.receive("ZETA נכנסתי שליש כמות", "Raz Gamliel", now=t0 + 40)["status"], "off")
            with self.assertRaises(ValueError):  # (auto on real money again: only with the owner's yes)
                A.set_settings({"mentor_on": True})

    def test_his_detailed_signal_at_his_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, gw, t0 = self._setup(tmp)
            gw.prices["NVDA"] = 100.0
            s = A.mentor.receive("NVDA לונג\nכניסה מיידית\nסיכון 0.5% מהתיק\nSTOP\n95\nTP\n120", "Raz Gamliel", now=t0, run=False)
            s = A.mentor.execute(s["id"], t0)
            self.assertEqual(gw.placed[-1], ("NVDA", 10, None, 120.0, 95.0))  # 0.5% of $10,500 = $52.5 / $5 a share

    def test_the_account_in_three_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, gw, t0 = self._setup(tmp)
            gw.prices["IONQ"] = 50.0
            s = A.mentor.receive("IONQ בפנים חצי כמות", "Raz Gamliel", now=t0, run=False)
            A.mentor.execute(s["id"], t0)
            self._fill(A, gw, t0 + 30)
            gw.acc["positions"] = [{"symbol": "IONQ", "shares": 14.0, "avg_cost": 50.0, "price": 52.0, "value": 728.0, "unrealized": 28.0, "stop": None}]
            A._read_account(gw, "live", t0 + 40)
            A.quotes["IONQ"] = {"price": 52.0, "ts": t0 + 40, "bars": [], "source": "ibkr"}
            L = A.live_book(t0 + 41)
            self.assertEqual([(p["symbol"], p["shares"], p["stop"]) for p in L["mentor"]["positions"]], [("IONQ", 10, 47.0)])
            self.assertEqual([(p["symbol"], p["shares"]) for p in L["manual"]["positions"]], [("IONQ", 4.0)])
            self.assertEqual(L["atlas"]["positions"], [])
            self.assertAlmostEqual(L["mentor"]["open_pnl"], 20.0)
            self.assertEqual(L["positions"][0]["source"], "mixed")
            A.set_settings({"live_max_open": 1})
            self.assertIsNone(A._live_guard(A.status["ibkr_acct"], A._get("ibkr")["orders"], t0 + 42))  # not Atlas' room
            self.assertIn("mentor", A.view())


class AtlasStaleSignalsTest(unittest.TestCase):
    def test_only_the_last_closes_signals_go(self):
        from polybot.atlas import Atlas
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        at = lambda d, h, m=0: dt.datetime(2026, 10, d, h, m, tzinfo=ny).timestamp()
        tue = {"last": "2026-10-27"}
        self.assertTrue(Atlas._signals_fresh(tue, at(28, 8)))  # Wednesday morning: Tuesday's close
        self.assertFalse(Atlas._signals_fresh(tue, at(28, 16, 5)))  # after Wednesday's close, before its data: wait
        self.assertTrue(Atlas._signals_fresh({"last": "2026-10-28"}, at(28, 16, 40)))
        self.assertTrue(Atlas._signals_fresh({"last": "2026-10-30"}, at(31, 12)))  # the weekend: Friday's
        self.assertTrue(Atlas._signals_fresh({"last": "2026-10-30"}, dt.datetime(2026, 11, 2, 8, tzinfo=ny).timestamp()))
        self.assertFalse(Atlas._signals_fresh({"last": "2026-10-29"}, dt.datetime(2026, 11, 2, 8, tzinfo=ny).timestamp()))


class AtlasLabIntradayEtfTest(unittest.TestCase):
    """Experiment 2 (intraday.py, 28.10.2026): the research's intraday strategies on SPY / QQQ."""

    def book(self, tmp, last_day, prev=None):
        from polybot import scalp
        flat = lambda base: (lambda k: base * (1 + 0.001 * (((k // 3) % 4) - 1.5) / 1.5))
        b = scalp.Bars(tmp)
        days = ["2026-10-%02d" % d for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19)]
        for d in days:
            b.save(d, AtlasLabOpeningRangeTest.bars(d, {"SPY": (prev if prev and d == days[-1] else flat(600), lambda k: 5000)}))
        b.save("2026-10-20", AtlasLabOpeningRangeTest.bars("2026-10-20", {"SPY": (last_day, lambda k: 5000)}))
        return b

    def test_a_trend_day(self):
        from polybot import intraday
        up = lambda k: 600 * (1 + 0.00004 * k)  # +1.6% by the close, steadily
        with tempfile.TemporaryDirectory() as tmp:
            res, run = intraday.run_all(self.book(tmp, up), symbols=("SPY",))
            self.assertEqual(run[-1], "2026-10-20")
            day = lambda k: [t for t in res[k] if t["day"] == "2026-10-20"]
            c = day("N0")
            self.assertEqual((len(c), c[0]["in"], c[0]["why"]), (1, 600, "close"))  # in at 10:00, out at the close
            n1 = day("N1")
            self.assertEqual(len(n1), 1)
            self.assertEqual((n1[0]["side"], n1[0]["why"]), ("long", "close"))
            self.assertGreater(n1[0]["pnl"], 0)
            self.assertEqual(day("N2")[0]["side"], "long")
            n3 = day("N3")
            self.assertEqual((len(n3), n3[0]["in"], n3[0]["side"]), (1, 930, "long"))
            self.assertEqual(day("N4")[0]["side"], "long")  # (up 1.4% by 15:30: big enough)
            self.assertTrue(day("N5") and day("N5")[0]["side"] == "long")
            self.assertTrue(all(t["cost"] > 0 and t["shares"] * t["entry"] <= intraday.NOTIONAL_CAP + 1 for k in res for t in res[k]))
            rows = intraday.board(res, run)
            self.assertEqual([r["variant"] for r in rows], list(intraday.VARIANTS))
            self.assertIsNone(next(r for r in rows if r["variant"] == "N0")["vs_placebo"])

    def test_a_falling_day(self):
        from polybot import intraday
        down = lambda k: 600 * (1 - 0.00004 * k)
        with tempfile.TemporaryDirectory() as tmp:
            res, run = intraday.run_all(self.book(tmp, down), symbols=("SPY",))
            day = lambda k: [t for t in res[k] if t["day"] == "2026-10-20"]
            self.assertEqual(day("N1"), [])  # long only: nothing
            self.assertEqual(day("N2")[0]["side"], "short")
            self.assertGreater(day("N2")[0]["pnl"], 0)
            self.assertEqual(day("N3"), [])
            self.assertEqual(day("N4")[0]["side"], "short")
            self.assertEqual(day("N5"), [])
            c = day("N0")
            self.assertEqual(c[0]["why"], "stop")  # the control's stop (0.4%) on a day that falls 1.6%


class AtlasLabNarrowRangeTest(unittest.TestCase):
    def test_v8_takes_only_the_quiet_days_stocks(self):
        from polybot import scalp
        day = AtlasLabOpeningRangeTest.bars("2026-10-20", {"AAA": (lambda k: 100 + 0.01 * k, lambda k: 1000),
                                                           "BBB": (lambda k: 50 + 0.01 * k, lambda k: 1000)})
        base = {"AAA": (1000.0, 0.2), "BBB": (1000.0, 0.1)}
        P = scalp.VARIANTS["V8"]
        self.assertEqual(set(scalp.scan(day, base, P, {"AAA", "BBB"}, {"BBB"})), {"BBB"})
        self.assertEqual(scalp.scan(day, base, P, {"AAA", "BBB"}, None), {})


class LiveLabTest(unittest.TestCase):
    """The live lab (livelab.py, 28.10.2026): experiment 1 in real time on the scanner's stocks, and the picked
    variants' trades in the IBKR paper account."""

    class Gw:
        def __init__(self):
            from zoneinfo import ZoneInfo
            self.ny = ZoneInfo("America/New_York")
            self.now, self.followed_syms = 0, []
            day0 = dt.date(2026, 10, 7)
            self.hist = {}
            for sym, px in (("ZZZ", 100.0), ("SPY", 600.0)):
                rows = []
                d = day0
                while d < dt.date(2026, 10, 28):
                    if d.weekday() < 5:
                        for k in range(78):
                            m = 570 + 5 * k
                            ts = dt.datetime(d.year, d.month, d.day, m // 60, m % 60, tzinfo=self.ny).timestamp()
                            p = px * (1 + 0.001 * ((k % 4) - 1.5))
                            w = 0.5 if sym == "ZZZ" else 0.05
                            rows.append((ts, p, p + w, p - w, p, 1000.0 if sym == "ZZZ" else 50000.0))
                    d += dt.timedelta(days=1)
                self.hist[sym] = rows

        def connected(self):
            return True

        def scanner(self, code, rows=50, above_price=10.0, above_volume=None):
            return ["ZZZ"] if code == "HOT_BY_VOLUME" else []

        def today(self, sym, upto):
            out = []
            for k in range(390):
                m = 570 + k
                ts = dt.datetime(2026, 10, 28, m // 60, m % 60, tzinfo=self.ny).timestamp()
                if ts > upto:
                    break
                if sym == "SPY":
                    p0 = p1 = 600.0
                    v = 50000.0
                else:
                    f = lambda j: 100 + 0.2 * j if j < 5 else 101 + 0.15 * (j - 5) if j < 60 else 101 + 0.15 * 55
                    p0, p1, v = f(k), f(k + 1), 3000.0 if k < 5 else 800.0
                out.append((ts, p0, max(p0, p1) + 0.01, min(p0, p1) - 0.01, p1, v))
            return out

        def bars_many(self, symbols, duration, size):
            if size == "5 mins":
                return {s: self.hist[s] for s in symbols if s in self.hist}
            return {s: self.today(s, self.now) for s in symbols if s in self.hist}

        def follow(self, symbols):
            self.followed_syms = [s for s in symbols if s in self.hist]
            return self.followed_syms

        def followed(self):
            return {s: self.today(s, self.now) for s in self.followed_syms}

    class PaperGw:
        def __init__(self):
            self.state, self.next, self.placed, self.sold, self.modified, self.cancelled = {}, 1, [], [], [], []

        def place_bracket(self, symbol, qty, limit, target, stop, entry=None, trail=None):
            ids = {"parent": self.next, "target": None, "stop": self.next + 1}
            self.next += 2
            self.placed.append((symbol, qty, stop, entry))
            self.state[ids["parent"]] = {"status": "Filled", "filled": qty, "avg_price": entry[1] - 0.01}
            self.state[ids["stop"]] = {"status": "Submitted", "filled": 0, "avg_price": 0}
            return ids

        def orders(self):
            return self.state

        def modify(self, oid, **kw):
            self.modified.append((oid, kw))
            return True

        def cancel(self, oid):
            self.cancelled.append(oid)
            return True

        def market(self, symbol, action, qty):
            oid = self.next
            self.next += 1
            self.sold.append((symbol, action, qty))
            self.state[oid] = {"status": "Filled", "filled": qty, "avg_price": 108.0}
            return oid

    def test_a_day_in_real_time(self):
        from polybot import livelab
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        at = lambda h, m, s=2: dt.datetime(2026, 10, 28, h, m, s, tzinfo=ny).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            gw, pg = self.Gw(), self.PaperGw()
            A.ibkr = gw
            L = A.livelab
            L.real.gw = pg
            L.real.ready = lambda: True
            A.set_lane("live", "V1", "on", confirm_live=True)  # (the owner moved it to real money)
            gw.now = at(9, 1)
            self.assertEqual(L.tick(gw.now), "ready")
            self.assertIn("ZZZ", L.d["base"])
            self.assertIn("ZZZ", L.d["scanner"]["HOT_BY_VOLUME"])
            gw.now = at(9, 35, 5)
            self.assertEqual(L.tick(gw.now), "live")
            self.assertEqual([x["symbol"] for x in L.d["net"]], ["ZZZ"])  # 3x its normal opening volume, green
            self.assertEqual(sorted(gw.followed_syms), ["SPY", "ZZZ"])
            self.assertTrue(L.d["full"])
            for m in range(9 * 60 + 36, 16 * 60 + 2):
                gw.now = at(m // 60, m % 60)
                L.tick(gw.now)
            self.assertEqual(L.d["phase"], "done")
            day = A._get("livelab_day:2026-10-28")
            v1 = day["V1"]
            self.assertTrue(v1 and v1[0]["symbol"] == "ZZZ")
            self.assertIn("V0", day)  # (the placebo, drawn after the close)
            self.assertEqual(A._get("livelab_days")[0]["day"], "2026-10-28")
            B = A._get("livelab_board")
            self.assertEqual(B["days"], ["2026-10-28"])
            self.assertIn("V1", [r["variant"] for r in B["rows"]])
            # real money: the model's entry sent at once, filled, its stop raised, out by the end
            self.assertEqual(pg.placed[0][0], "ZZZ")
            self.assertEqual(pg.placed[0][3][0], "limit")
            orders = list(A._get("livelab_real")["orders"].values())
            self.assertEqual(orders[0]["status"], "closed")
            self.assertIsNotNone(orders[0]["slip"])
            self.assertTrue(pg.modified)  # (the model's stop rose: ours with it)
            v = A.livelab.view()
            self.assertEqual(v["real"]["by_variant"]["V1"]["closed"], 1)
            self.assertIn("V1", v["stages"])
            self.assertEqual(v["stages"]["V1"]["ready"], False)

    def test_intraday_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest._atlas(self, tmp)
            self.assertEqual((A.lanes()["sim"]["V1"], A.lanes()["live"]["V1"]), ("out", "out"))  # (the owner moves them)
            A.set_lane("sim", "V7", "on")
            with self.assertRaises(ValueError):
                A.set_lane("live", "V7", "on")  # (real money: the owner's yes, always)
            with self.assertRaises(ValueError):
                A.set_lane("sim", "V6", "on")  # (needs every stock's minutes: not live)
            self.assertTrue(A.settings()["lanes_since"]["sim"]["V7"])
            V = {x["key"]: x for x in A.lane_view()}
            self.assertEqual((V["V7"]["kind"], V["V7"]["sim_state"]), ("intraday", "on"))


class RefineWorkshopTest(unittest.TestCase):
    """The workshop (refine.py, 29.10.2026): a winning strategy cloned into the lab with versions of it, replayed
    on the recorded seconds, and where real money missed its trades."""
    BASE = "twap|0.12|10-120"

    def _db(self, tmp, n=60):
        from polybot.store import Store
        st = Store(os.path.join(tmp, "t.sqlite"))
        for i in range(n):
            w = 1_000_000 + i * 300
            winner = "up" if i % 10 < 7 else "down"
            st.upsert_window("btc", 5, w, winner=winner)
            for left in range(150, 0, -5):
                # the market's Up is 60c most of the round; 55c (a 12-point edge or more) only from 60s left
                ask = 0.55 if left <= 60 else 0.62
                st.add_tick(ts=w + 300 - left, asset="btc", minutes=5, window_start=w, price=1.0, open_price=1.0,
                            seconds_left=left, vol=0.1, fair_up=0.73, up_bid=ask - 0.01, up_ask=ask, down_bid=0.4,
                            down_ask=0.46, price_age=0.1, book_age=0.1)
        return st

    def test_versions_replay_and_judge(self):
        from polybot import refine
        from polybot.store import connect
        c = refine.candidates(self.BASE, 5)
        keys = [x["key"] for x in c]
        self.assertIn("twap|0.10|10-120", keys)  # (more: a lower edge)
        self.assertIn("twap3|0.12|10-120", keys)  # (more: several entries)
        self.assertIn("twap|0.12|10-120|c70", keys)  # (precise: our model surer)
        self.assertTrue(all(x["goal"] in ("more", "precise", "both") for x in c))
        self.assertNotIn(self.BASE, keys)
        with tempfile.TemporaryDirectory() as tmp:
            self._db(tmp)
            db = connect(os.path.join(tmp, "t.sqlite"))
            res, rounds = refine.replay(db, "btc", 5, [self.BASE] + keys)
            self.assertEqual(len(rounds), 60)
            base = res[self.BASE]
            self.assertEqual(len(base), 60)  # (one a round, the first second the edge is there)
            self.assertTrue(all(t["paid"] > 0.55 for t in base))  # (the fee on top)
            self.assertEqual(sum(1 for t in base if t["pnl"] > 0), 42)
            self.assertEqual(len(res["twap3|0.12|10-120"]), 180)  # (three a round, a third each)
            self.assertEqual(len(res["twap|0.12|10-120|c70"]), len(base))  # (our model gives 73%: all pass)
            self.assertEqual(len(res["twap|0.12|10-120|b40-80"]), len(base))
            J = refine.judge(res, rounds, self.BASE)
            self.assertEqual(J[self.BASE]["late"]["n"], 24)
            self.assertIn(J["twap|0.10|10-120"]["verdict"], ("same", "better", "worse", "few"))
            self.assertIsNotNone(J["twap|0.14|10-120"]["vs"])

    def test_where_real_money_missed(self):
        from polybot import refine
        from polybot.paper import Fill
        from polybot.store import connect
        with tempfile.TemporaryDirectory() as tmp:
            st = self._db(tmp, 6)
            f = Fill(side="up", fair_prob=0.73, best_ask=0.55, avg_price=0.55, shares=10, cost=5.5, fee=0.17, edge=0.15, limit_price=0.55)
            for i in range(6):
                w = 1_000_000 + i * 300
                st.add_bet(w + 250, "btc", 5, w, 50, f, mode="lab", variant=self.BASE)
                st._x("UPDATE bets SET pnl = 4.33 WHERE window_start = ? AND mode = 'lab'", (w,))
                if i < 3:
                    st.add_bet(w + 250, "btc", 5, w, 50, f, mode="live", variant=self.BASE)
            st.add_fill_test((1_000_000 + 4 * 300 + 250, "btc", 5, 1_000_000 + 4 * 300, "live", self.BASE, "up", 10, 0.55, 2, 5.6, 0.17, None, None))
            db = connect(os.path.join(tmp, "t.sqlite"))
            M = refine.misses(db, "btc", 5, self.BASE, 0)
            self.assertEqual((M["lab"], M["live"], M["missed"], M["refused"], M["room1"], M["not_tried"]), (6, 3, 3, 1, 1, 2))
            self.assertAlmostEqual(M["missed_pnl"], 3 * 4.33)
            res, rounds = refine.replay(db, "btc", 5, [self.BASE])
            F = refine.fidelity(db, "btc", 5, self.BASE, rounds, res)
            self.assertEqual((F["lab"], F["replay"], F["same_rounds"]), (6, 6, 1.0))

    def test_the_engine_clones_it_into_the_lab(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={})
            w = engine.refine_start("btc", 5, self.BASE)
            keys = {v.key for v in engine._lab_variants("btc", 5)}
            self.assertIn(self.BASE, keys)
            self.assertTrue(all(c["key"] in keys for c in w["variants"]))
            self.assertNotIn("twap3|0.12|10-120", {v.key for v in engine._lab_variants("btc", 15)})
            study = engine.refine_study(w["id"])
            self.assertEqual(study["rounds"], 0)
            V = engine.refine_view()
            self.assertEqual(V[0]["rows"][0]["goal"], "base")
            self.assertEqual(len(V[0]["rows"]), 1 + len(w["variants"]))
            engine.refine_stop(w["id"])
            self.assertNotIn("twap3|0.12|10-120", {v.key for v in engine._lab_variants("btc", 5)})


class ConsensusTest(unittest.TestCase):
    """The consensus rules (lab.CONS_RULES, 29.10.2026): enter the side enough of the lab's good rules agreed on."""

    def test_the_side_and_the_rules(self):
        from polybot import lab
        keys = {v.key for v in lab.variants_for(5)}
        self.assertTrue({"cons3g|0.85|5-270", "cons5g|0.85|5-270", "cons3g|0.70|5-270", "cons3a|0.85|5-270"} <= keys)
        v3, v5, va = (lab.Variant.from_key(k) for k in ("cons3g|0.85|5-270", "cons5g|0.85|5-270", "cons3a|0.85|5-270"))
        good = {"g1", "g2", "g3"}
        votes = {"g1": "up", "g2": "up", "g3": "up", "x": "up"}
        self.assertEqual(lab.consensus_side(v3, votes, good), "up")
        self.assertIsNone(lab.consensus_side(v5, votes, good))
        self.assertIsNone(lab.consensus_side(v3, {**votes, "g4": "down"}, good | {"g4"}))  # (one against: no consensus)
        self.assertEqual(lab.consensus_side(v3, {**votes, "y": "down"}, good), "up")  # (a rule that isn't good: not counted)
        self.assertIsNone(lab.consensus_side(va, {**votes, "y": "down"}, good))  # (the control counts every rule)
        self.assertIsNone(lab.consensus_side(v3, {"cons3a|0.85|5-270": "up", "g1": "up", "g2": "up"}, good))  # (never its own kind)
        self.assertIn("הסכמה", lab.describe("cons3g|0.85|5-270"))
        self.assertEqual(lab.neighbours("cons3g|0.85|5-270", 5), [])

    def test_a_tracker_enters_on_consensus(self):
        from polybot import lab
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite")), api=FakeApi(), env={})
            t = next(x for x in engine.trackers if x.minutes == 5)
            t.good = {"g1", "g2", "g3"}
            t.lab_votes = {"g1": "down", "g2": "down", "g3": "down"}
            sent = []
            t._send = lambda *a: sent.append(a)
            v = lab.Variant.from_key("cons3g|0.85|5-270")
            cfg = t._lab_cfg(v)
            t._consensus(v, 1000.0, 100, {"twap": 0.3}, {"twap": 0.3}, [(0.30, 100)], [(0.72, 100)], 0.1, 0.1, cfg)
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0][3].side, "down")
            t._consensus(v, 1001.0, 99, {"twap": 0.3}, {"twap": 0.3}, [(0.30, 100)], [(0.72, 100)], 0.1, 0.1, cfg)
            self.assertEqual(len(sent), 1)  # (once a round)
            t.lab_done.clear(), t.lab_entries.clear()
            t._consensus(v, 1002.0, 99, {"twap": 0.3}, {"twap": 0.3}, [(0.30, 100)], [(0.90, 100)], 0.1, 0.1, cfg)
            self.assertEqual(len(sent), 1)  # (dearer than its cap: nothing)


class LiveMirrorTest(unittest.TestCase):
    """Real money as the simulation (29.10.2026): a strategy in both trades with real money under the simulation's
    rules: its filter, "exactly like the lab", the mix with the market price; the owner's amount stays his."""

    def test_real_money_runs_the_simulations_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={"POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64}, streaming=False)
            engine.refresh_lab = lambda: None
            self.assertTrue(engine.live_mirror())  # (on from the start: the owner's word)
            key = "twap|0.12|10-120"
            sim = [{"asset": "btc", "minutes": 5, "variant": key, "ratio": 0.8, "ratio_src": "system"}]
            live = [{"asset": "btc", "minutes": 5, "variant": key, "ratio": 0.0}]
            engine.store.set_setting("main_strategies", json.dumps(sim))
            engine.store.set_setting("live_main", json.dumps(live))
            engine.set_lab_exact("sim", False)
            engine.set_lab_exact("live", True)  # (real money's own value, kept aside while it mirrors)
            self.assertEqual(engine.lab_exact(), {"sim": False, "live": False})
            self.assertFalse(engine.live.exact)
            rules = engine._rules_by_market(engine.live_main(), "live")[("btc", 5)]
            self.assertEqual(rules[0].filter_ratio, 0.8)  # (the simulation's filter)
            engine.set_blend_use("sim", False)
            self.assertFalse(engine.blend_use()["live"])
            engine.set_lab_exact("sim", True)
            self.assertTrue(engine.lab_exact()["live"] and engine.live.exact)
            self.assertEqual(engine._rules_by_market(engine.live_main(), "live")[("btc", 5)][0].filter_ratio, 0.0)
            engine.set_live_mirror(False)  # (back to real money's own)
            self.assertEqual(engine.lab_exact(), {"sim": True, "live": True})
            engine.set_lab_exact("sim", False)
            self.assertEqual(engine._rules_by_market(engine.live_main(), "live")[("btc", 5)][0].filter_ratio, 0.0)
            engine.store.close()


class AtlasMidSessionSwingTest(unittest.TestCase):
    """The owner (30.10.2026): swing trades that enter during the day: an order at the trigger, armed the evening
    before, filled mid-session the moment the price gets there; held days with the stop rising."""

    def _p(self, closes, highs=None, lows=None):
        from polybot import equity
        bars = []
        for k, c in enumerate(closes):
            d = (dt.date(2024, 1, 1) + dt.timedelta(days=k)).isoformat()
            h = highs[k] if highs else c * 1.005
            lo = lows[k] if lows else c * 0.995
            bars.append((d, c, h, lo, c, 1_000_000))
        return equity.prepare(bars)

    def test_the_setups_and_their_triggers(self):
        from polybot import equity
        closes = [100 + 0.2 * k for k in range(260)]  # a steady uptrend
        p = self._p(closes)
        i = len(closes) - 1
        P = equity.plan_for("breakout_id")
        hh = max(p["hh20"][i], p["h"][i])
        self.assertIsNotNone(equity.signal("breakout_id", p, i, P))  # (just under its 20 days' high)
        self.assertEqual(equity.entry_order(P, p, i), ["stop", round(hh * 1.001, 2)])
        Pb = equity.plan_for("burst_id")
        self.assertIsNotNone(equity.signal("burst_id", p, i, Pb))
        self.assertEqual(equity.entry_order(Pb, p, i), ["stop", round(closes[-1] * 1.04, 2)])
        Pp = equity.plan_for("pullback_id")
        self.assertIsNone(equity.signal("pullback_id", p, i, Pp))  # (no weakness: RSI(2) high)
        dip = closes[:-2] + [closes[-3] * 0.97, closes[-3] * 0.95]
        p2 = self._p(dip)
        self.assertIsNotNone(equity.signal("pullback_id", p2, len(dip) - 1, Pp))
        self.assertEqual(equity.entry_order(Pp, p2, len(dip) - 1)[0], "limit")
        # A trigger reached mid-session fills there; one not reached, no trade.
        self.assertEqual(equity.fill_price(["stop", 105.0], 104.0, 106.0, 103.5), 105.0)
        self.assertIsNone(equity.fill_price(["stop", 105.0], 104.0, 104.9, 103.5))

    def test_new_ones_start_in_the_lab_and_the_brain_keeps_their_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            A, eq = AtlasTest()._atlas(tmp)
            L = A.lanes()
            self.assertEqual({k: (L["sim"][k], L["live"][k]) for k in ("breakout_id", "pullback_id", "burst_id")},
                             {k: ("out", "out") for k in ("breakout_id", "pullback_id", "burst_id")})
            self.assertNotIn("burst_id", A.plans())  # (not in the simulation)
            import random as _r
            for _ in range(10):
                A.learn_once("breakout_id", rnd=_r.Random(_))
            self.assertNotIn("entry", (A.brain().get("breakout_id") or {}).get("params") or {})


class AtlasLanesTest(unittest.TestCase):
    """Each strategy's place (29.10.2026): lab -> simulation -> real money, as Polybot's, each lane its own."""

    def test_lanes_and_what_they_do(self):
        from polybot import equity
        with tempfile.TemporaryDirectory() as tmp:
            A = AtlasRealAccountTest._live(self, tmp)
            kinds = list(equity.STRATEGIES)
            L = A.lanes()
            # (the ones entering mid-session, new: in the lab only till the owner moves them)
            self.assertEqual({k: L["sim"][k] for k in kinds}, {k: "out" if k.endswith("_id") else "on" for k in kinds})
            self.assertEqual({k: L["live"][k] for k in kinds}, {k: L["sim"][k] for k in kinds})  # (as before: real money as the simulation)
            A.set_lane("sim", "breakout", "paused")
            self.assertFalse(A.settings()["strategies"]["breakout"])
            self.assertEqual(A.lanes()["live"]["breakout"], "on")  # (each lane its own)
            A.set_lane("live", "pullback", "out")
            with self.assertRaises(ValueError):
                A.set_lane("live", "pullback", "on")  # (real money: the owner's yes)
            A.set_lane("live", "pullback", "on", confirm_live=True)
            A.set_lane("live", "fib", "out")
            book = A.paper_book()
            book["pending"] = [("fib", "MSFT", 90.0, 101.0), ("breakout", "NVDA", 90.0, 101.0)]
            A._set("paper", book)
            got = {x[0] for x in A.live_signals(A.paper_book())}
            self.assertNotIn("fib", got)  # (out of real money: its signals don't go there)
            V = {x["key"]: x for x in A.lane_view()}
            self.assertEqual((V["breakout"]["sim_state"], V["fib"]["live_state"]), ("paused", "out"))
            self.assertTrue(all(set(x) >= {"sim", "live", "lab", "params"} for x in V.values()))
            self.assertEqual(V["V1"]["kind"], "intraday")
            A.set_settings({"strategies": {"gap": False}})  # (the old switch moves the lane too)
            self.assertEqual(A.lanes()["sim"]["gap"], "out")
            with self.assertRaises(ValueError):
                A.set_lane("sim", "nope", "on")

    def test_a_paused_strategy_keeps_its_open_trades_but_takes_no_new_ones(self):
        from polybot import equity
        P = {k: equity.plan_for(k) for k in equity.STRATEGIES}
        port = dict(equity.DEFAULTS)
        book = equity.new_book(100_000)
        calls = []
        orig = equity.signals_for
        try:
            equity.signals_for = lambda prepared, d, plans, *a, **kw: calls.append(set(plans)) or []
            equity.step(book, {"SPY": {"index": {}, "c": [], "day": []}}, "2026-10-28", P, port, active={"pullback"})
        finally:
            equity.signals_for = orig
        self.assertEqual(calls, [{"pullback"}])


class BothSidesLockedTest(unittest.TestCase):
    """The owner's trade that can't lose (30.10.2026): one side, then the other once the pair costs under $1; the
    lab runs three ways of it with real orders' fills (hedge.py), and a replay on the recorded seconds."""
    FEE = staticmethod(lambda p: 0.07 * p * (1 - p))

    def _run(self, key, path, fair=0.5):
        """path: [(t, seconds left, up ask, down ask)] -> the rule's position after them."""
        v = lab.Variant.from_key(key)
        pos = hedge.Pos()
        for t, left, ua, da in path:
            asks = {"up": [(ua, 1000)], "down": [(da, 1000)]}
            hedge.fills(pos, t, asks, self.FEE)
            hedge.step(v, pos, t, left, fair if not callable(fair) else fair(ua), asks, self.FEE, 0.01, 0.5)
        return pos

    def test_leg_then_lock_by_taking_the_offer(self):
        # Up at 46c where the model sees 60%; then Up runs and Down falls to 40c: 46 + 40 + fees < 98c.
        path = [(0, 250, 0.46, 0.55), (1, 249, 0.47, 0.54), (2, 248, 0.60, 0.41), (3, 247, 0.62, 0.40), (4, 246, 0.62, 0.40)]
        pos = self._run("bothAm98t|0.05|100-290", path, fair=lambda ua: 0.6 if ua < 0.5 else 0.6)
        self.assertAlmostEqual(pos.sh["up"], 10)
        self.assertAlmostEqual(pos.sh["down"], 10)
        self.assertIsNotNone(pos.locked)
        self.assertGreater(pos.locked, 0)  # (whoever wins)
        self.assertTrue(pos.done)

    def test_no_lock_if_the_other_side_never_turns_cheap(self):
        path = [(0, 250, 0.46, 0.55), (1, 249, 0.46, 0.55), (2, 248, 0.30, 0.71), (3, 247, 0.20, 0.81)]
        pos = self._run("bothAm98t|0.05|100-290", path, fair=0.6)
        self.assertEqual(pos.sh["down"], 0)
        self.assertIsNone(pos.locked)

    def test_the_second_leg_waits_in_the_book_with_no_fee(self):
        path = [(0, 250, 0.25, 0.76), (1, 249, 0.25, 0.76)]
        pos = self._run("bothAc95w|0.30|100-290", path)
        self.assertAlmostEqual(pos.sh["up"], 10)
        self.assertIn("down", pos.bids)  # (resting at 95c less what Up cost, under the offer)
        bid = pos.bids["down"]["price"]
        self.assertLessEqual(bid + pos.paid["up"] / 10, 0.95 + 1e-9)
        asks = {"up": [(0.33, 1000)], "down": [(bid - 0.02, 1000)]}  # sellers went through it
        hedge.fills(pos, 10, asks, self.FEE)
        self.assertAlmostEqual(pos.sh["down"], 10)
        self.assertAlmostEqual(pos.paid["down"], 10 * bid)  # (no fee for the maker)
        self.assertIsNotNone(pos.locked)

    def test_keep_buying_until_locked(self):
        path = [(0, 250, 0.44, 0.57), (1, 249, 0.44, 0.57), (3, 247, 0.38, 0.63), (4, 246, 0.38, 0.63), (6, 244, 0.60, 0.42),
                (7, 243, 0.62, 0.40), (9, 241, 0.62, 0.40), (11, 239, 0.62, 0.40), (13, 237, 0.62, 0.40)]
        pos = self._run("bothB98|0.45|100-290", path)
        self.assertGreater(pos.sh["up"], 2)  # (more Up under its average)
        self.assertIsNotNone(pos.locked)
        self.assertLessEqual(pos.total, hedge.BUDGET)

    def test_two_bids_add_up_to_the_target(self):
        v = lab.Variant.from_key("bothC94|0.00|100-290")
        pos = hedge.Pos()
        asks = {"up": [(0.52, 1000)], "down": [(0.50, 1000)]}
        hedge.step(v, pos, 0, 250, 0.5, asks, self.FEE)
        self.assertEqual({s: b["price"] for s, b in pos.bids.items()}, {"up": 0.47, "down": 0.47})
        hedge.fills(pos, 1, {"up": [(0.45, 1000)], "down": [(0.55, 1000)]}, self.FEE)  # Up's bid filled
        self.assertAlmostEqual(pos.sh["up"], 10)
        hedge.step(v, pos, 1, 249, 0.5, {"up": [(0.45, 1000)], "down": [(0.55, 1000)]}, self.FEE)
        self.assertAlmostEqual(pos.bids["down"]["price"], 0.47)  # (94c less the 47c Up cost)

    def test_the_insurance_caps_the_loss_in_the_last_minute(self):
        v = lab.Variant.from_key("bothAm98ti|0.05|100-290")
        pos = hedge.Pos(sh={"up": 10.0, "down": 0.0}, paid={"up": 4.7, "down": 0.0})
        asks = {"up": [(0.40, 1000)], "down": [(0.55, 1000)]}
        hedge.step(v, pos, 0, 50, 0.4, asks, self.FEE)  # the model says Up is losing
        self.assertTrue(pos.insured)
        hedge.fills(pos, 1, asks, self.FEE)
        self.assertAlmostEqual(pos.sh["down"], 10)
        self.assertLess(min(pos.sh.values()) - pos.total, 0)  # a small loss, capped
        self.assertGreater(min(pos.sh.values()) - pos.total, -0.5 - 1e-9)

    def test_a_price_under_a_cent_still_locks(self):
        # (the replay's error, 30.10.2026: an offer of 0.3c at a round's end is off the cent's grid; the order's
        # limit came out empty and the whole replay stopped)
        v = lab.Variant.from_key("bothAm98t|0.05|100-290")
        pos = hedge.Pos(sh={"up": 10.0, "down": 0.0}, paid={"up": 9.75, "down": 0.0})
        asks = {"up": [(0.99, 1000)], "down": [(0.003, 1000)]}
        hedge.step(v, pos, 0, 30, 0.99, asks, self.FEE)
        self.assertGreaterEqual(pos.takes[0]["limit"], 0.003)
        hedge.fills(pos, 1, asks, self.FEE)
        self.assertAlmostEqual(pos.sh["down"], 10)

    def test_the_lab_records_each_leg_and_the_board_counts_the_locked_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_fill_delay=0.5)
            store = Store(cfg.db_path)
            t = Tracker("btc", cfg, store, api=FakeApi())
            t.window, t.market, t.minutes = 1_800_000_000, {"tick_size": 0.01}, 5
            v = lab.Variant.from_key("bothAm98t|0.05|100-290")
            for now, left, ua, da in [(0, 250, 0.46, 0.55), (1, 249, 0.47, 0.54), (2, 248, 0.62, 0.40), (3, 247, 0.62, 0.40)]:
                t._both(v, now, left, {"twap": 0.6}, [(ua, 1000)], [(da, 1000)], 0.1, 0.1, cfg)
            rows = store._x("SELECT variant, side, shares, status FROM bets WHERE mode = 'lab' ORDER BY id")
            self.assertEqual([(r[1], round(r[2])) for r in rows], [("up", 10), ("down", 10)])
            for bid, side in store._x("SELECT id, side FROM bets"):
                b = store._x("SELECT shares, cost, fee FROM bets WHERE id = ?", (bid,))[0]
                store.settle_bet(bid, "down", (b[0] if side == "down" else 0) - b[1] - b[2])
            board = lab.leaderboard(store.db, cfg, current={("btc", 5, v.key)})
            row = board["board"][0]
            self.assertEqual((row["family"], row["bets"], row["locked"]), ("pair", 1, 1))
            self.assertGreater(row["pnl"], 0)
            # (its rounds since the lab runs on real money's conditions, apart)
            later = lab.leaderboard(store.db, cfg, current={("btc", 5, v.key)}, cond_since=-1)["board"][0]
            self.assertEqual((later["real"]["rounds"], later["real"]["locked"]), (1, 1))
            self.assertNotIn("real", lab.leaderboard(store.db, cfg, current={("btc", 5, v.key)}, cond_since=10 ** 10)["board"][0])
            store.close()

    def test_the_replay_on_the_recorded_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            now = 1_800_100_000
            for k in range(3):
                w = now - 3600 + k * 300
                store.upsert_window("btc", 5, w, winner="up" if k % 2 else "down", found=1)
                for i in range(40):
                    ua = 0.46 if i < 20 else 0.62
                    store.add_tick(ts=w + i, asset="btc", minutes=5, window_start=w, price=100, open_price=100,
                                   seconds_left=250 - i, vol=None, fair_up=0.6, up_bid=None, up_ask=ua, down_bid=None,
                                   down_ask=round(1.02 - ua, 2), price_age=0, book_age=0)
            out = hedge.replay(store.db, now, lab.both_for, cfg.fee_rate, cfg.fee_exponent)
            self.assertEqual(out["rounds"], 3)
            row = next(r for r in out["rows"] if r["key"] == "bothAm98t|0.05|100-290")
            self.assertEqual((row["all"]["n"], row["all"]["locked"]), (3, 3))
            self.assertGreater(row["all"]["pnl"], 0)
            self.assertEqual(out["state"], "collecting")  # (100 rounds before a verdict)
            store.close()

    def test_the_labs_focus_runs_the_lock_rules_and_the_owners_strategies_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",), lab_minutes=(15,))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            self.assertTrue(engine.lab_focus())  # (on from the start)
            keys = {v.key for v in engine._lab_variants("btc", 15)}
            mine = {v for a, m, v in engine._owner_keys() if (a, m) == ("btc", 15)}
            self.assertTrue({v.key for v in lab.both_for(15)} | mine <= keys)
            self.assertTrue(any(lab.is_maker(k) for k in keys))  # (the makers run too: their edge)
            self.assertTrue(any(lab.family(k) == "end" for k in keys))  # (the research cards' rules run too)
            plain = [k for k in keys if lab.family(k) == "main" and k not in mine]
            self.assertTrue(all(lab.Variant.from_key(k).min_chance for k in plain))  # (only the owner's 70-80c rule)
            engine.store.set_setting("lab_focus", "all")
            self.assertGreater(len(engine._lab_variants("btc", 15)), len(keys))  # (nothing erased: all back)
            hub = {x["key"]: x for x in engine.hub_view()["tracks"]}
            self.assertIn("both_lock", hub)
            engine.store.close()


class DiskFullTest(unittest.TestCase):
    """30.10.2026: the server's disk filled up and every write failed. Short of room, the oldest recorded
    seconds go, a day at a time (down to 3 days), a little at a time."""

    def test_short_of_room_the_oldest_seconds_go(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            now = 1_800_000_000.0
            for day in range(10):
                for i in range(5):
                    engine.store.add_tick(ts=now - day * 86400 - i, asset="btc", minutes=15, window_start=0, price=1.0,
                                          open_price=None, seconds_left=None, vol=None, fair_up=None, up_bid=None,
                                          up_ask=None, down_bid=None, down_ask=None, price_age=None, book_age=None)
            engine.store.BATCH = 3  # (a little at a time)
            full = {"total": 1000, "free": 10, "db": 900, "wal": 0, "reusable": 0}
            with mock.patch.object(engine.store, "disk", return_value=full):
                engine._disk_tick(now)
            self.assertEqual(engine.tick_days(), 3)
            left = engine.store._x("SELECT MIN(ts) FROM ticks")[0][0]
            self.assertGreaterEqual(left, now - 3 * 86400)
            self.assertFalse(next(c for c in engine.health.report(now)["checks"] if c["name"] == "disk")["ok"])
            roomy = dict(full, free=900)
            with mock.patch.object(engine.store, "disk", return_value=roomy):
                engine._disk_tick(now)
            self.assertTrue(next(c for c in engine.health.report(now)["checks"] if c["name"] == "disk")["ok"])
            self.assertEqual(engine.tick_days(), cfg.keep_ticks_days)  # (room again: the configured days)
            d = engine.store.disk()  # (the real one)
            self.assertGreater(d["total"], 0)
            engine.store.close()


class FullDiskAtStartTest(unittest.TestCase):
    def test_before_the_database_opens_the_files_that_can_go_go(self):
        import collections
        from unittest import mock
        from polybot import store as store_mod
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "polybot.sqlite")
            os.makedirs(os.path.join(tmp, "atlas_minutes"))
            for n in ("ballast.bin", "x.tmp", "atlas_minutes/2026-10-01.json.gz", "atlas_minutes/2026-10-02.json.gz"):
                open(os.path.join(tmp, n), "wb").write(b"0")
            U = collections.namedtuple("U", "total used free")
            frees = iter([0, 0, 0, 0, 10 ** 9, 10 ** 9])
            with mock.patch("shutil.disk_usage", side_effect=lambda p: U(10 ** 10, 0, next(frees, 10 ** 9))):
                gone = store_mod.free_room(db)
            self.assertEqual(gone, ["ballast.bin", "x.tmp", "2026-10-01.json.gz"])
            self.assertTrue(os.path.exists(os.path.join(tmp, "atlas_minutes/2026-10-02.json.gz")))  # (enough room by then)

    def test_real_money_against_the_simulation_round_by_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            v = "twap|0.05|15-600"
            f = lambda price: Fill(side="up", fair_prob=0.8, best_ask=price, avg_price=price, shares=10, cost=10 * price,
                                   fee=0.0, edge=0.1, limit_price=price)
            for ws, mode, price in ((900, "paper", 0.5), (900, "live", 0.55), (1800, "paper", 0.5)):
                engine.store.add_bet(ws + 1, "btc", 15, ws, 100, f(price), mode=mode, variant=v)
            D = engine.strategy_diff("live", "btc", 15, v, against="sim")
            self.assertEqual((D["both"]["here"]["rounds"], D["only_lab"]["rounds"], D["against"]), (1, 1, "sim"))
            self.assertAlmostEqual(D["both"]["here"]["price"] - D["both"]["lab"]["price"], 0.05)
            with self.assertRaises(ValueError):
                engine.strategy_diff("sim", "btc", 15, v, against="sim")
            engine.store.close()


class LockRulesInTheSimulationTest(unittest.TestCase):
    def test_a_lock_rule_goes_to_the_simulation_not_to_real_money_and_trades_there_its_own_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",), lab_minutes=(5,))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            key = "bothAm98t|0.05|100-290"
            engine.lab = {"board": [{"asset": "btc", "minutes": 5, "variant": key, "status": "promising"}]}
            with self.assertRaises(ValueError):
                engine.push([("btc", 5, key)], mode="live")
            engine.push([("btc", 5, key)], mode="sim", allow_unproven=True)
            t = next(x for x in engine.trackers if (x.asset, x.minutes) == ("btc", 5))
            self.assertIn(key, [r.key for r in t.main_rules])
            t.window, t.market = 1_800_000_000, {"tick_size": 0.01}
            t.can_open = None
            for now, ua, da in ((0, 0.46, 0.55), (1, 0.47, 0.54), (2, 0.62, 0.40), (3, 0.62, 0.40)):
                t._sim_rules_run(now, 250 - now, {}, {}, [(ua, 1000)], [(da, 1000)], 0.1, 0.1, {"twap": 0.6}, {"twap": 0.6})
            rows = engine.store._x("SELECT side, shares FROM bets WHERE mode = 'paper' AND variant = ? ORDER BY id", (key,))
            stake = t._rule_cfg(next(r for r in t.main_rules if r.key == key)).stake_usd
            self.assertEqual([r[0] for r in rows], ["up", "down"])
            self.assertAlmostEqual(rows[0][1], stake)  # (a $10 amount: 10 pairs)
            engine.store.close()

    def test_the_lock_rules_stages_count_rounds(self):
        rounds = [{"pnl": 0.5, "paid": 9.5}] * 60 + [{"pnl": -4.0, "paid": 4.0}] * 2
        self.assertGreater(lab.round_z(rounds), 0)
        self.assertIsNone(lab.round_z(rounds[:5]))


class StarsTest(unittest.TestCase):
    def test_a_star_is_kept_and_the_two_winners_come_named_and_starred(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",), lab_minutes=(5,))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            self.assertEqual(engine.starred(), [])  # (no workshop on the winner here: nothing seeded)
            engine.refine_start("btc", 5, "twap|0.12|10-120")
            engine._seed_winners()
            names = engine.strategy_names()
            self.assertEqual(names["btc|5|twap|0.12|10-120|r0.3"], "שיפור מנצחת 1")
            self.assertEqual(names["btc|5|twap|0.14|10-120"], "שיפור מנצחת 2")
            self.assertEqual(set(engine.starred()), {"btc|5|twap|0.12|10-120|r0.3", "btc|5|twap|0.14|10-120"})
            engine.set_star("btc", 5, "twap|0.14|10-120", False)
            engine.set_star("eth", 15, "twap|0.05|15-600", True)
            self.assertEqual(set(engine.starred()), {"btc|5|twap|0.12|10-120|r0.3", "eth|15|twap|0.05|15-600"})
            engine.store.close()
            again = Engine(cfg, api=FakeApi(), env={}, streaming=False)  # (seeded once only)
            self.assertNotIn("btc|5|twap|0.14|10-120", again.starred())
            self.assertIn("starred", again.state())
            again.store.close()


class LabIntoRealMoneyTest(unittest.TestCase):
    """The owner (30.10.2026): the crypto lab's strategies at stage 2 and up trade real money too, the moment they
    enter in the lab; one entry a round and side, or all of them."""

    def _engine(self, tmp):
        cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",), lab_minutes=(5,))
        engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
        engine.lab = {"board": [
            {"asset": "btc", "minutes": 5, "variant": "twap|0.05|10-120", "status": "promising"},
            {"asset": "btc", "minutes": 5, "variant": "twap|0.08|10-120", "status": "candidate"},
            {"asset": "btc", "minutes": 5, "variant": "twap|0.12|10-120", "status": "losing"},
            {"asset": "btc", "minutes": 5, "variant": "twapm|0.05|120-240", "status": "proven"},
            {"asset": "btc", "minutes": 5, "variant": "bothAm98t|0.05|100-290", "status": "proven"}]}
        return engine, next(t for t in engine.trackers if (t.asset, t.minutes) == ("btc", 5))

    def test_only_the_crypto_strategies_at_stage_two_and_up_and_only_with_the_owners_yes(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            self.assertEqual({v for _, _, v in engine.mirror_keys()}, {"twap|0.05|10-120", "twap|0.08|10-120"})
            with self.assertRaises(ValueError):
                engine.set_lab_mirror(True)  # (not without the popup's yes)
            self.assertIsNone(t.mirror)
            engine.set_lab_mirror(True, confirmed=True)
            self.assertEqual(t.mirror, {"keys": {"twap|0.05|10-120", "twap|0.08|10-120"}, "all": False})
            engine.set_lab_mirror(all_=True)  # (the second switch: no popup)
            self.assertTrue(t.mirror["all"])
            self.assertEqual(engine.lab_mirror(), {"on": True, "all": True})
            engine.set_lab_mirror(False)
            self.assertIsNone(t.mirror)
            engine.store.close()

    def test_one_entry_a_round_and_side_or_all(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            engine.set_lab_mirror(True, confirmed=True)
            t.window = 1_800_000_000
            calls = []

            def live_bet(now, left, rule, p, up, down, plain=None):
                calls.append((rule.key, "up" if up else "down"))
                t.live_entries[rule.key] = (t.live_entries.get(rule.key, (0, 0.0))[0] + 1, now)
            with mock.patch.object(t, "_live_bet", side_effect=live_bet):
                for key in ("twap|0.05|10-120", "twap|0.08|10-120"):
                    t._mirror(lab.Variant.from_key(key), 100.0, 60, 0.7, "up", [(0.5, 100)], [(0.5, 100)])
                self.assertEqual(calls, [("twap|0.05|10-120", "up")])  # (one entry: the first strategy's)
                t._mirror(lab.Variant.from_key("twap|0.08|10-120"), 101.0, 60, 0.3, "down", [(0.5, 100)], [(0.5, 100)])
                self.assertEqual(calls[-1], ("twap|0.08|10-120", "down"))  # (the other side is its own)
                engine.set_lab_mirror(all_=True)
                t._mirror(lab.Variant.from_key("twap|0.08|10-120"), 102.0, 60, 0.7, "up", [(0.5, 100)], [(0.5, 100)])
                self.assertEqual(calls[-1], ("twap|0.08|10-120", "up"))  # (all: every strategy's)
            engine.store.close()

    def test_the_lab_entering_sends_real_money_at_the_same_moment(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            engine.set_lab_mirror(True, confirmed=True)
            v = lab.Variant.from_key("twap|0.05|10-120")
            t.set_variants([v])
            t.window, t.market = 1_800_000_000, {"tick_size": 0.01}
            with mock.patch.object(t, "_mirror") as m:
                t._run_lab(1000.0, 60, {"twap": 0.8}, {"twap": 0.8}, [(0.5, 100)], [(0.5, 100)], 0.1, 0.1)
            self.assertEqual(m.call_count, 1)
            self.assertEqual(m.call_args[0][4], "up")
            engine.store.close()


class FillCapTest(unittest.TestCase):
    def test_the_most_paid_over_the_price_is_the_owners_choice_five_from_the_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine._fill_first_tick(time.time())
            self.assertEqual(engine.fill_cap(), 5)
            self.assertEqual(engine.live.fill_room, ({}, 5))
            engine.set_fill_cap(10)
            self.assertEqual(engine.live.fill_room, ({}, 10))
            with self.assertRaises(ValueError):
                engine.set_fill_cap(7)
            engine.store.close()


class RealConditionsTest(unittest.TestCase):
    """The owner (30.10.2026): the simulation makes money, real money doesn't. Each simulation order has a shadow that
    meets the book when a real one would; the delay is tuned until the shadows get in as often as real money does."""

    def test_the_simulation_meets_the_book_when_real_money_would_and_pays_what_it_would(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_fill_delay=0.5)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            t = engine.trackers[0]
            t.window, t.real_delay, t.market = 1_800_000_000, 2.0, {"tick_size": 0.01}
            # (real money pays up to 5c over the price seen to get in, within the edge: the owner's cap, 1.241)
            fill = Fill(side="up", fair_prob=0.8, best_ask=0.6, avg_price=0.6, shares=10, cost=6.0, fee=0.1, edge=0.1,
                        limit_price=0.6)
            t._send(100.0, "paper", "twap|0.05|15-600", fill, 200, 0, 0, 0.8, 6.0)
            (o,) = t.lab_orders
            self.assertEqual(o["due"], 102.0)  # (when a real order would meet the book)
            self.assertEqual(o["room_limit"], 0.65)  # (and what it would pay)
            t._arrive(101.0, [(0.61, 100)], [])
            self.assertEqual(len(t.lab_orders), 1)  # (not yet)
            t._arrive(102.1, [(0.61, 100)], [])
            row = engine.store._x("SELECT mode, status, avg_price FROM bets")[0]
            self.assertEqual(tuple(row), ("paper", "filled", 0.61))  # (in at 61c, as real money's room let it)
            engine.store.close()

    def test_the_delay_is_tuned_to_real_moneys_share_that_gets_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_fill_delay=0.5)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            now = time.time()
            add = lambda mode, status, n: [engine.store._x(
                "INSERT INTO bets (mode, status, ts, asset, minutes, window_start, side, shares, cost, fee)"
                " VALUES (?, ?, ?, 'btc', 15, 0, 'up', 10, 5, 0)", (mode, status, now - 60)) for _ in range(n)]
            add("live", "filled", 10); add("live", "rejected", 10)  # (real money: half got in)
            add("paper", "filled", 27); add("paper", "rejected", 3)  # (the simulation: 90%: it meets the book too early still)
            d0 = engine.real_delay()
            self.assertEqual(engine._real_tick(now), d0 + engine.REAL_STEP)
            self.assertTrue(all(t.real_delay == d0 + engine.REAL_STEP for t in engine.trackers))
            engine.store.close()


class BuiltForRealMoneyTest(unittest.TestCase):
    """The owner (30.10.2026): the winning strategies with real money's problems put in; each idea on real money's log."""

    def test_the_ideas_read_real_moneys_log(self):
        from polybot import livefit
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "t.sqlite"))
            now = 1_800_100_000.0
            v = "twap|0.05|10-120"
            for k in range(60):
                w = int(now - 86400 + k * 300)
                store.upsert_window("btc", 5, w, winner="up", found=1)
                for i in range(40):  # (the offer each second: calm, then back at 0.5 after the order)
                    store.add_tick(ts=w + 200 + i, asset="btc", minutes=5, window_start=w, price=1, open_price=1, seconds_left=100 - i,
                                   vol=None, fair_up=0.7, up_bid=None, up_ask=0.5, down_bid=None, down_ask=0.5, price_age=0, book_age=0)
                f = Fill(side="up", fair_prob=0.7, best_ask=0.5, avg_price=0.5, shares=10, cost=5.0, fee=0.1, edge=0.1, limit_price=0.5)
                bid = store.add_bet(w + 210, "btc", 5, w, 90, f, mode="live", variant=v, status="filled" if k % 2 else "rejected", seen=0.5)
            for bid, side, shares, cost, fee in store._x("SELECT id, side, shares, cost, fee FROM bets WHERE status = 'filled'"):
                store.settle_bet(bid, "up", shares - cost - fee)
            out = livefit.study(store.db, now)
            T = {t["key"]: t for t in out["tracks"]}
            self.assertEqual(set(T), {"adverse", "strategies", "calm", "rest", "retry", "together", "timing", "room"})
            self.assertEqual((T["adverse"]["filled"]["n"], T["adverse"]["refused"]["n"]), (30, 30))
            self.assertEqual(T["rest"]["got"]["n"], 30)  # (each refused order: the offer came back to its price)
            self.assertGreater(T["retry"]["got"]["pnl"], 0)
            self.assertEqual(T["strategies"]["rows"][0]["fill"], 0.5)
            self.assertEqual(out["final"]["orders"], 60)
            self.assertIn("how", out["final"])
            store.close()


class TheFinalTest(unittest.TestCase):
    """The owner (30.10.2026): the study's answer, ready, through one more final check, then "הסופית" (named, starred),
    its own mail to real money's tab, and real money only on the owner's yes."""

    def _engine(self, tmp):
        cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",), lab_minutes=(5,))
        engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
        sent = []
        engine.notifier.mail_final = lambda F, names: sent.append(names)
        return engine, sent

    def test_ready_then_checked_a_day_and_fifty_orders_later_then_named_starred_and_mailed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, sent = self._engine(tmp)
            F = {"ready": True, "orders": 120, "need": 100, "strategies": [["btc", 5, "twap|0.12|10-120"]], "how": "x",
                 "how_code": "r30", "together": True, "calm_only": False, "avoid": ["0-30"], "room": "2",
                 "per_day": 4.0, "late_per_day": 2.0, "now_per_day": 1.0, "days": 10}
            engine.livefit = {"final": F}
            st = engine._final_tick(1000.0)
            self.assertFalse(st.get("confirmed"))
            engine.livefit = {"final": dict(F, orders=180)}
            st = engine._final_tick(1000.0 + 86400)
            self.assertTrue(st["confirmed"])
            key = "twap|0.12|10-120|er30"
            self.assertEqual(st["keys"], [["btc", 5, key]])
            self.assertEqual(engine.strategy_names()[f"btc|5|{key}"], "הסופית")
            self.assertIn(f"btc|5|{key}", engine.starred())
            self.assertEqual(sent, [["הסופית"]])
            engine._final_tick(1000.0 + 90000)
            self.assertEqual(len(sent), 1)  # (once)
            with self.assertRaises(ValueError):
                engine.store.set_setting("final_state", json.dumps(dict(st, confirmed=None)))
                engine.apply_final()
            engine.store.set_setting("final_state", json.dumps(st))
            engine.apply_final()
            self.assertEqual([m["variant"] for m in engine.live_main()], [key])
            t = next(x for x in engine.trackers if (x.asset, x.minutes) == ("btc", 5))
            self.assertEqual(t.final_gates["keys"], {key})
            self.assertEqual(engine.fill_cap(), 5)  # (its room, 2c, isn't one of the caps: the cap stays)
            engine.store.close()

    def test_its_mail_takes_the_owner_to_real_moneys_tab(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={"POLYBOT_URL": "https://bot.example"}, streaming=False, with_lab=False)
            got = []
            engine.notifier.send = lambda subject, html_body, images=(): got.append((subject, html_body))
            engine.notifier.mail_final({"per_day": 4.1, "orders": 230, "days": 12.7, "now_per_day": 1.2, "how": "x",
                                        "together": True, "avoid": ["0-30"], "room": "2"}, ["הסופית"])
            subject, body = got[0]
            self.assertIn("הסופית", subject)
            self.assertIn("https://bot.example/#live", body)
            self.assertIn("לפתוח את הסופית במסחר החי", body)
            engine.store.close()

    def test_an_answer_that_changes_or_stops_making_money_starts_the_check_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, sent = self._engine(tmp)
            F = {"ready": True, "orders": 120, "strategies": [["btc", 5, "twap|0.12|10-120"]], "how_code": "",
                 "per_day": 4.0, "late_per_day": 2.0}
            engine.livefit = {"final": F}
            engine._final_tick(1000.0)
            engine.livefit = {"final": dict(F, late_per_day=-1.0)}
            self.assertEqual(engine._final_tick(2000.0), {})
            engine.livefit = {"final": dict(F, orders=500)}
            st = engine._final_tick(1000.0 + 86400)
            self.assertFalse(st.get("confirmed"))  # (its day starts again)
            engine.store.close()

    def test_the_final_gates_with_real_money(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = self._engine(tmp)
            t = next(x for x in engine.trackers if (x.asset, x.minutes) == ("btc", 5))
            v = lab.Variant.from_key("twap|0.12|10-120|er30")
            t.window = 1_800_000_000
            t.final_gates = {"keys": {v.key}, "calm": True, "together": True, "avoid": ["0-30"]}
            self.assertIn("0-30", t._final_gate(v, "up", 20, 100.0))
            self.assertIsNone(t._final_gate(v, "up", 90, 100.0))
            t.live_sent.add((t.window, "up"))
            self.assertIn("אחת לסבב", t._final_gate(v, "up", 90, 100.0))
            t.live_sent.clear()
            t._ask_hist["up"].extend([(97.0, 0.50), (100.0, 0.55)])
            self.assertIn("רגועים", t._final_gate(v, "up", 90, 100.0))
            other = lab.Variant.from_key("twap|0.05|10-120")
            self.assertIsNone(t._final_gate(other, "up", 20, 100.0))  # (only the final's own strategies)
            engine.store.close()

    def test_rest_after_a_refusal_in_the_lab(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_fill_delay=0.5)
            store = Store(cfg.db_path)
            t = Tracker("btc", cfg, store, api=FakeApi())
            t.window, t.market = 1_800_000_000, {"tick_size": 0.01}
            fill = Fill(side="up", fair_prob=0.8, best_ask=0.6, avg_price=0.6, shares=10, cost=6.0, fee=0.1, edge=0.1, limit_price=0.6)
            t._send(100.0, "lab", "twap|0.05|15-600|er30", fill, 200, 0, 0, 0.8, 6.0)
            t._arrive(100.6, [(0.7, 100)], [])  # (refused: the price went)
            self.assertEqual(t.lab_pending[0]["kind"], "s")
            self.assertEqual(t.lab_pending[0]["bid"], 0.6)
            t._pending_run(110.0, 150, {"twap": 0.8}, [(0.59, 100)], [])  # (a seller came under it)
            row = store._x("SELECT status, avg_price, fee FROM bets WHERE status = 'filled'")[0]
            self.assertEqual(tuple(row), ("filled", 0.6, 0.0))
            store.close()
