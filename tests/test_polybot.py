from decimal import Decimal
import dataclasses
import json
import os
import tempfile
import time
import types
import unittest

from polybot import feeds, lab
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
        self.assertEqual((order["token_id"], order["side"], order["max_price"], order["order_type"]), ("U", "BUY", 0.70, "FOK"))
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

    def test_stages_from_collecting_to_proven_on_new_data(self):
        strong = "twap|0.05|15-180"
        self.add("btc", 15, strong, [True, True, True, False] * 15)  # 75% wins at 50c, 60 bets
        self.add("eth", 15, strong, [True] * 10)  # perfect but only 10 bets
        self.add("sol", 5, strong, [True, False] * 30)  # coin flip at 50c: no edge
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
        self.add("btc", 15, strong, [True, True, True, False] * 5, t0=20_000.0)  # 40 new: still under the 50 needed
        self.assertEqual(self.status(self.board(candidates))[("btc", 15)], "candidate")
        self.add("btc", 15, strong, [True, True, True, False] * 5, t0=30_000.0)  # 60 new, still winning
        proven = self.board(candidates)
        self.assertEqual(self.status(proven)[("btc", 15)], "proven")
        self.assertEqual([(p["asset"], p["minutes"]) for p in proven["picks"]], [("btc", 15)])

    def test_candidate_that_loses_on_new_data_fails(self):
        v = "twap|0.05|15-180"
        self.add("btc", 15, v, [True, True, True, False] * 15)
        self.add("btc", 15, v, [False, False, True] * 20, t0=10_000.0)
        self.assertEqual(self.status(self.board({("btc", 15, v): 10_000.0}))[("btc", 15)], "failed")

    def test_picks_are_on_different_markets(self):
        keys = ("twap|0.05|15-180", "twap|0.08|15-180", "twap|0.03|15-180")
        candidates = {}
        for variant in keys:
            self.add("btc", 15, variant, [True, True, True, False] * 15, t0=10_000.0)
            candidates[("btc", 15, variant)] = 0.0
        self.add("eth", 5, "twap|0.05|10-120", [True, True, True, False] * 15, t0=10_000.0)
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
        # 12 rules + 6 maker + 8 several-entries rules + 4 both-sides + 8 last-minute
        self.assertEqual(len(lab.variants_for(15)), 38)
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
        self.post("/api/money", {"mode": "live", "values": {"copies": 3}})
        live.arm(True, override=True)
        self.assertEqual(self.signal(0.55)[1], 3)  # three orders of the same stake
        prices = [o["max_price"] for o in live.client.orders[-3:]]
        self.assertTrue(all(p <= prices[0] + 0.01 + 1e-9 for p in prices))  # at most a cent above the first
        self.assertEqual(self.engine.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live'")[0][0], 3)

    def test_real_money_tab_follows_its_own_settings_only(self):
        live = self.engine.live
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
            self.assertEqual(lab.family("lock75|0.50|3-60"), "lock")
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
            self.assertEqual(board["families"]["pair"]["per_round"]["rounds"], 2)
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
    """Real money on in the owner's approval mode: nothing buys by itself; the owner's click does."""
    _engine = AddToATradeTest._engine

    def test_nothing_buys_by_itself_and_the_owners_click_does_without_the_guards(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._engine(tmp)
            live, orders = engine.live, []

            def place(**kw):
                orders.append(kw)
                return types.SimpleNamespace(ok=True, order_id="x", status="matched", taking_amount=kw["amount"] / 0.85,
                                             making_amount=kw["amount"])
            with mock.patch.object(live, "configured", return_value=True):
                engine.set_mode(True, confirmed=True, approval=True)
            self.assertTrue(live.approval())
            self.assertEqual(engine.mode(), "paper")  # the simulation's view (and its mails)
            fill = types.SimpleNamespace(side="up", fair_prob=0.9, best_ask=0.85, avg_price=0.85, shares=10, cost=8.5,
                                         fee=0.1, edge=0.05, limit_price=0.85)
            with mock.patch.object(live, "trades", return_value=True), \
                    mock.patch.object(live, "_client", return_value=types.SimpleNamespace(place_market_order=place)):
                self.assertIsNone(live.maybe_trade(time.time(), "btc", 15, t.window, 300, fill, t.market))  # a rule alone: no
                live.store.set_setting("live_daily_loss_limit", "1")  # (a guard that would stop an automatic order)
                with self.assertRaises(ValueError):  # the price ran: over the strategy's 80c + 2c
                    engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "up", 0.80)
                done = engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "up", 0.84)
            self.assertEqual(done["status"], "filled")
            self.assertEqual(len(orders), 1)
            self.assertLessEqual(orders[0]["max_price"], 0.86)
            row = engine.store._x("SELECT mode, variant, side FROM bets")[0]
            self.assertEqual(tuple(row), ("live", "twap|0.05|15-600", "up"))
            engine.set_mode(False, confirmed=True)
            self.assertFalse(live.approval())
            with self.assertRaises(ValueError):  # the mode is off: no more clicks
                engine.approve_real("btc", 15, t.window, "twap|0.05|15-600", "up", 0.84)
            engine.store.close()


class CandidatesGoToTheSimulationTest(unittest.TestCase):
    def _row(self, variant, pess, family="main", asset="btc"):
        return {"asset": asset, "minutes": 15, "variant": variant, "rule": lab.describe(variant), "status": "candidate",
                "roi": 0.2, "roi_pessimistic": pess, "bets": 120, "family": family, "new_candidate": True}

    def test_a_new_candidate_goes_in_and_an_email_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), max_rules_per_market=3)
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
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
            main, pair = "twap|0.05|15-600", "pair85|0.20|120-885"
            for v, ts in ((main, now - 3600), (pair, now - 3600), (pair, now + 60)):
                engine.store._x("INSERT INTO bets (mode, variant, status, ts, asset, minutes, window_start, side, shares,"
                                " cost, fee, pnl, winner) VALUES ('lab', ?, 'filled', ?, 'btc', 15, ?, 'up', 10, 5, 0, 5, 'up')",
                                (v, ts, int(ts)))
            engine.set_count_since("pair", True)
            db = connect(cfg.db_path)
            fams = lab.leaderboard(db, cfg, since=engine.lab_since())["families"]
            self.assertEqual(fams["main"]["pnl"], 5)  # the crypto page: all its money
            self.assertEqual(fams["pair"]["pnl"], 5)  # both sides: one round of two, since starting over
            self.assertEqual(fams["pair"]["settled_bets"], 2)  # ...but still both rounds counted
            self.assertNotIn("lab", engine.count_since())
            db.close()
            engine.store.close()

    def test_an_older_lab_wide_start_carries_to_every_lab_page_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False, with_lab=False)
            engine.store.set_setting("count_since", json.dumps({"lab": 123.0}))
            self.assertEqual(engine.lab_since(), {"main": 123.0, "pair": 123.0, "lock": 123.0})
            self.assertEqual(engine.count_since()["stocks"], 123.0)
            engine.set_count_since("lock", False)
            self.assertEqual(engine.lab_since()["lock"], None)  # and each is its own from then on
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


class GuestVenueTest(unittest.TestCase):
    def test_limitless_rounds_books_trades_and_results_are_recorded_read_only(self):
        from polybot import guest, marketcheck
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            now = 1_800_000_000.0
            end_ms = int((now + 1800) * 1000)
            calls = []

            def fetch(path, params=None):
                calls.append(path)
                if path == "/markets/search":
                    return {"markets": [
                        {"slug": "btc-1h", "title": "BTC Up or Down - 1 hour", "expirationTimestamp": str(end_ms),
                         "expired": False, "tokens": {"yes": "Y", "no": "N"}},
                        {"slug": "btc-week", "title": "BTC Up or Down - 1 week", "expirationTimestamp": str(end_ms),
                         "tokens": {"yes": "Y2", "no": "N2"}},  # (a week: another game, left out)
                        {"slug": "x", "title": "Trump out?", "expirationTimestamp": str(end_ms)}]}
                if path == "/markets/btc-1h/orderbook":
                    return {"bids": [{"price": "0.55", "size": "20000000"}], "asks": [{"price": "0.57", "size": "5000000"}]}
                if path == "/markets/btc-1h/events":
                    return {"events": [{"createdAt": "2027-01-15T08:20:00.000Z", "matchedSize": "10000000", "price": 0.57,
                                        "side": 0, "tokenId": "Y", "txHash": "0xa"}]}
                if path == "/markets/btc-1h":
                    return {"winningOutcomeIndex": 0}
                raise AssertionError(path)
            w = guest.LimitlessWatch(store, fetch=fetch)
            w.step(now)
            self.assertEqual(list(w.rounds), ["btc-1h"])
            tick = store._x("SELECT asset, minutes, up_bid, up_ask, down_bid, down_ask, bid_size FROM lx_ticks")[0]
            self.assertEqual(tuple(tick), ("btc", 60, 0.55, 0.57, 0.43, 0.45, 20.0))
            trade = store._x("SELECT outcome, price, size, taker FROM lx_trades")[0]
            self.assertEqual(tuple(trade), ("up", 0.57, 10.0, "BUY"))
            w.step(now + 1)  # the same trade again: kept once
            self.assertEqual(store._x("SELECT COUNT(*) FROM lx_trades")[0][0], 1)
            w._last["resolve"] = 0
            w.step(now + 1900)  # the round ended: its result
            self.assertEqual(store._x("SELECT winner FROM lx_windows")[0][0], "up")
            self.assertTrue(all(c.startswith("/markets") for c in calls))  # read only: no order paths
            db = connect(cfg.db_path)
            r = marketcheck.study(db, cfg, "lx_ticks", "lx_windows", marketcheck.GUEST_TIMES, fee=False)
            self.assertEqual(r["rounds"], 1)
            db.close()
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
            self.assertEqual([tuple(r) for r in rows], [("paper", "filled", 10.0), ("paper", "rejected", 10.0)])
            engine.store.close()

    def test_a_book_that_moved_away_refuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, t = self._tracker(tmp)
            t._arrive(100.6, [(0.6, 3), (0.62, 100)], [(0.39, 100)])  # only 3 shares left at 60c
            self.assertEqual(tuple(engine.store._x("SELECT status, pnl FROM bets")[0]), ("rejected", None))
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

    def test_the_lab_runs_only_the_picked_times_and_the_owners_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.refresh_lab = lambda: None
            t = next(x for x in engine.trackers if x.key == "btc-15m")
            every = len(t.variants)
            self.assertIsNone(engine.state()["lab_times"]["picked"])  # the system's: every time
            engine.set_lab_times(["30", "60"])
            self.assertEqual(engine.lab_times(), {"30", "60"})
            mine = {(m["variant"]) for m in engine.strategies_for("sim") if (m["asset"], m["minutes"]) == ("btc", 15)}
            others = {v.key for v in t.variants} - mine
            self.assertTrue(others)
            self.assertTrue(all(lab.time_group(k) in ("30", "60") for k in others))
            self.assertTrue(mine <= {v.key for v in t.variants})  # the owner's always measured
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
