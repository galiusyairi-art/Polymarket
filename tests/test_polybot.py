import os
import tempfile
import unittest

from polybot import feeds
from polybot.config import Config
from polybot.model import VolatilityEstimator, fair_up_probability, taker_fee_per_share
from polybot.paper import find_bet, settle_pnl, simulate_buy
from polybot.report import replay
from polybot.run import Tracker, settle
from polybot.store import Store


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
        self.assertLess(taker_fee_per_share(0.95, 0.25, 2), taker_fee_per_share(0.5, 0.25, 2))

    def test_volatility_recovers_known_step_size(self):
        vol = VolatilityEstimator(1800)
        price = 100.0
        for t in range(600):
            price *= 1.001 if t % 2 else 1 / 1.001
            vol.add(float(t), price)
        self.assertAlmostEqual(vol.per_sqrt_second(), 0.001, places=5)
        self.assertEqual(vol.history_seconds(), 599)

    def test_volatility_forgets_old_samples(self):
        vol = VolatilityEstimator(60)
        for t in range(200):
            vol.add(float(t), 100.0)
        self.assertLessEqual(vol.history_seconds(), 60)


class PaperTest(unittest.TestCase):
    cfg = Config(min_edge=0.05, stake_usd=10.0, fee_rate=0.0)

    def test_buys_only_levels_with_enough_edge(self):
        asks = [(0.80, 5), (0.70, 5), (0.86, 100)]  # unsorted on purpose
        shares, cost, fee = simulate_buy(asks, 100.0, 0.90, self.cfg)
        self.assertEqual(shares, 10)  # 0.86 leaves only 4 points of edge
        self.assertAlmostEqual(cost, 7.5)

    def test_stops_at_budget(self):
        shares, cost, fee = simulate_buy([(0.50, 1000)], 10.0, 0.90, self.cfg)
        self.assertAlmostEqual(cost, 10.0)
        self.assertAlmostEqual(shares, 20.0)

    def test_budget_includes_fees(self):
        cfg = Config(min_edge=0.05, stake_usd=10.0)
        shares, cost, fee = simulate_buy([(0.50, 1000)], 10.0, 0.90, cfg)
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
        return {"slug": slug, "up_token": "U", "down_token": "D",
                "price_to_beat": self.price_to_beat, "resolution_source": "test"}

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
            "SELECT our_end_price, winner FROM windows WHERE window_start = ?", (start,)).fetchone()
        self.assertEqual((end_price, w), (101.0, "up"))

    def test_never_bets_on_a_window_joined_midway(self):
        start = feeds.window_start(1_800_000_000, 15)
        self.run_seconds(start + 100, 300)
        self.api.price *= 1.05
        self.tracker.step(start + 600)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM bets").fetchone()[0], 0)

    def test_uses_polymarket_opening_price_when_available(self):
        start = feeds.window_start(1_800_000_000, 15)
        self.api.price_to_beat = 123.0
        self.tracker.step(start + 1)
        self.assertEqual(self.tracker.open_price, 123.0)
        src = self.store.db.execute("SELECT open_source FROM windows").fetchone()[0]
        self.assertEqual(src, "polymarket")


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


if __name__ == "__main__":
    unittest.main()
