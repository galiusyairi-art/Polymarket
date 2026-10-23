import json
import os
import tempfile
import time
import unittest

from polybot import feeds, lab
from polybot.config import Config
from polybot.model import VolatilityEstimator, fair_up_probability, fair_up_probability_twap, taker_fee_per_share
from polybot.paper import find_bet, settle_pnl, simulate_buy
from polybot.engine import Engine, Tracker, settle
from polybot.live import LiveTrader
from polybot.stats import readiness, replay
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
        self.assertAlmostEqual(taker_fee_per_share(0.5, 0.25, 2) / 0.5, 0.015625)
        self.assertAlmostEqual(taker_fee_per_share(0.5, 0.07, 1) / 0.5, 0.0175)
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
        self.assertEqual(len(keys), len(set(keys)))  # one bet per variant per window
        # Variants only bet inside their own timing: none of the "last 60s" ones at 400s left.
        early = self.store.db.execute("SELECT variant FROM bets WHERE seconds_left > 300").fetchall()
        self.assertTrue(all(not v.endswith("-60") for (v,) in early))
        tracker.step(start + 901)
        self.api.winner = "up"
        settle(self.store, self.cfg, start + 2000, api=self.api)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM bets WHERE pnl IS NULL").fetchone()[0], 0)


class ReplayTest(unittest.TestCase):
    def test_one_bet_per_window_at_first_qualifying_tick(self):
        cfg = Config(fee_rate=0.0)
        rows = [
            ("btc", 0, 300, 0.80, 0.70, 0.31, "up"),
            ("btc", 0, 200, 0.90, 0.70, 0.31, "up"),
            ("btc", 900, 300, 0.20, 0.25, 0.70, "up"),
        ]
        bets, wins, pnl = replay(rows, cfg, 0.05)
        self.assertEqual((bets, wins), (2, 1))
        self.assertAlmostEqual(pnl, (1 - 0.70) + (0 - 0.70))


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
    def __init__(self):
        self.orders = []

    def create_order(self, args):
        return args

    def post_order(self, order, kind):
        self.orders.append((order, kind))
        return {"success": True, "orderID": "0xabc"}

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
        try:
            import py_clob_client  # noqa: F401
        except ImportError:
            self.skipTest("py-clob-client not installed")
        self.assertEqual(t.maybe_trade(0, "btc", 15, 0, 300, self.fill, self.market), "filled")
        order, kind = t.client.orders[0]
        self.assertEqual((order.token_id, order.price), ("U", 0.70))
        self.assertLessEqual(order.size * 0.70, t.stake())

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
        t.set_readiness(False)
        self.assertFalse(t.armed())

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
        self.add("btc", 15, strong, [True, True, True, False] * 5, t0=20_000.0)  # 40 new, still winning
        proven = self.board(candidates)
        self.assertEqual(self.status(proven)[("btc", 15)], "proven")
        self.assertEqual([(p["asset"], p["minutes"]) for p in proven["picks"]], [("btc", 15)])

    def test_candidate_that_loses_on_new_data_fails(self):
        v = "twap|0.05|15-180"
        self.add("btc", 15, v, [True, True, True, False] * 15)
        self.add("btc", 15, v, [False, False, True] * 12, t0=10_000.0)
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
        self.assertEqual(len(lab.variants_for(15)), 12)


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
        self.assertEqual(len(self.sent), 1)
        opened = self.sent[0]
        self.assertTrue(opened["Subject"].startswith("📈 ") or opened["Subject"].startswith("📉 "))
        self.assertNotIn("כסף אמיתי", opened["Subject"])  # simulation: no label
        self.assertEqual(opened["To"], "owner@example.com")
        body = self.html(opened)
        self.assertIn("עלייה או ירידה", body)
        self.assertIn("#1f8a4c", body)  # our side's button is green
        images = [p for p in opened.walk() if p.get_content_type() == "image/png"]
        self.assertEqual(len(images), 1)
        self.assertIn(images[0]["Content-ID"][1:-1], body)

        self.engine.notifier.tick(self.start + 602)
        self.assertEqual(len(self.sent), 1)  # nothing new, nothing re-sent

        self.t.step(self.start + 901)
        self.api.winner = "up"
        settle(self.engine.store, self.engine.cfg, self.start + 2000, api=self.api)
        self.engine.notifier.tick(self.start + 2001)
        self.assertEqual(len(self.sent), 2)
        closed = self.sent[1]
        self.assertTrue(closed["Subject"].startswith("✅ \u2066+$"))
        self.assertIn("#1f8a4c", self.html(closed))

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
        self.assertIsNotNone(find_bet(0.9, [(0.30, 100)], [], dataclasses.replace(cfg, min_ratio=0.0)))

    def test_table_and_automatic_choice(self):
        from polybot import riskreward
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(db_path=os.path.join(tmp, "t.sqlite"))
            store = Store(cfg.db_path)
            db = connect(cfg.db_path)
            # Cheap entries (20c, ratio 4) win often; dear ones (80c, ratio 0.25) lose.
            for i in range(300):
                cheap = i % 2 == 0
                price = 0.2 if cheap else 0.8
                won = (i % 4 == 0) if cheap else (i % 10 == 1)
                shares = 10 / price
                db.execute("INSERT INTO bets (ts, mode, asset, minutes, window_start, side, seconds_left, fair_prob,"
                           " best_ask, avg_price, shares, cost, fee, edge, status, pnl, variant) VALUES"
                           " (?, 'lab', 'btc', 15, ?, 'up', 60, 0.5, ?, ?, ?, 10, 0, 0.1, 'filled', ?, 'v')",
                           (i, i * 900, price, price, shares, shares - 10 if won else -10))
            db.commit()
            rows = riskreward.table(db, cfg)
            db.close()
            store.close()
        by = {r["ratio"]: r for r in rows}
        self.assertEqual(by[0.0]["bets"], 300)
        self.assertEqual(by[3.0]["bets"], 150)  # only the 20c entries pass 1:3
        self.assertGreater(by[3.0]["roi"], by[0.0]["roi"])
        ratio, reason = riskreward.choose(rows, 100)
        self.assertGreaterEqual(ratio, 0.5)
        self.assertEqual(reason, "best")
        self.assertEqual(riskreward.choose(rows, 1000), (0.0, "waiting"))

    def test_setting_applies_to_the_main_strategy_not_the_lab(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(assets=("btc",), db_path=os.path.join(tmp, "t.sqlite"), lab_assets=("btc",))
            engine = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            engine.refresh_lab = lambda: None
            engine.set_risk_reward("manual", 3)
            t = engine.main_trackers[0]
            self.assertEqual(t.main_cfg.min_ratio, 3)
            self.assertTrue(all(c.min_ratio == 0 for c in t.lab_cfgs.values()))
            engine.set_risk_reward("auto")  # no evidence yet: no filter
            self.assertEqual(t.main_cfg.min_ratio, 0)
            with self.assertRaises(ValueError):
                engine.set_risk_reward("manual", -1)
            engine.set_risk_reward("manual", 2)
            engine.store.close()
            again = Engine(cfg, api=FakeApi(), env={}, streaming=False)
            self.assertEqual(again.main_trackers[0].main_cfg.min_ratio, 2)  # survives a restart
            self.assertEqual(again.state()["risk_reward"]["effective"], 2)
            again.store.close()


class StockLabTest(unittest.TestCase):
    CLOSE = 1793044800  # 2026-10-26 16:00 New York

    def market(self, slug="aapl-close-above-on-october-26-2026", title="$340", threshold="340", closed=False):
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
