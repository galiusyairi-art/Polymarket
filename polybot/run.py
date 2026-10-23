"""Stage 1 recorder: every second, record price, fair probability and Polymarket's order book,
and log the paper bets the strategy would have made. No real money is involved.

    python -m polybot.run            # run until stopped (Ctrl+C)
    python -m polybot.run --check    # test the connections once and exit
"""
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from . import feeds
from .config import Config
from .model import VolatilityEstimator, fair_up_probability
from .paper import find_bet, settle_pnl
from .store import Store

log = logging.getLogger("polybot")


class Tracker:
    """Follows one asset through its consecutive windows."""

    def __init__(self, asset, cfg, store, api=feeds):
        self.asset, self.cfg, self.store, self.api = asset, cfg, store, api
        self.vol = VolatilityEstimator(cfg.vol_lookback_seconds)
        self.window = None
        self.market = None
        self.open_price = None
        self.bets_this_window = 0
        self.last_lookup = 0.0

    def step(self, now):
        cfg = self.cfg
        price = self.api.spot_price(self.asset)
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

        self.store.add_tick(
            ts=now, asset=self.asset, window_start=start, price=price, open_price=self.open_price,
            seconds_left=seconds_left, vol=vol, fair_up=fair_up,
            up_bid=max((p for p, _ in up_bids), default=None),
            up_ask=min((p for p, _ in up_asks), default=None),
            down_bid=max((p for p, _ in down_bids), default=None),
            down_ask=min((p for p, _ in down_asks), default=None),
        )

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
        self.store.upsert_window(self.asset, start, found=1)
        if self.market["price_to_beat"] and self.open_price is not None:
            # Polymarket's own opening price is what the market resolves against.
            self.open_price = self.market["price_to_beat"]
            self.store.upsert_window(self.asset, start, open_price=self.open_price, open_source="polymarket")


def settle(store, cfg, now, api=feeds):
    """Fill in winners for finished windows and settle their paper bets."""
    for asset, start, slug in store.unresolved_windows(now - cfg.window_minutes * 60 - 60):
        winner = api.market_winner(slug)
        if not winner:
            continue
        store.upsert_window(asset, start, winner=winner)
        for bet_id, side, shares, cost, fee in store.unsettled_bets(asset, start):
            pnl = settle_pnl(side, shares, cost, fee, winner)
            store.settle_bet(bet_id, winner, pnl)
            log.info("SETTLED %s %s bet: winner %s, P&L $%.2f", asset.upper(), side.upper(), winner.upper(), pnl)


def check(cfg):
    """One pass over every data source, printed plainly."""
    ok = True
    now = time.time()
    for asset in cfg.assets:
        try:
            print(f"[ok] {asset.upper()} price: {feeds.spot_price(asset):,.2f}")
        except Exception as e:
            ok = False
            print(f"[FAIL] {asset.upper()} price (Coinbase): {e}")
        slug = feeds.market_slug(cfg, asset, feeds.window_start(now, cfg.window_minutes))
        try:
            market = feeds.find_market(slug)
            if not market:
                ok = False
                print(f"[FAIL] no Polymarket market named {slug}")
                continue
            print(f"[ok] Polymarket market {slug}, resolves on: {market['resolution_source']}")
            bids, asks = feeds.order_book(market["up_token"])
            print(f"[ok]   Up: best bid {max(p for p, _ in bids):.2f}, best ask {min(p for p, _ in asks):.2f}")
        except Exception as e:
            ok = False
            print(f"[FAIL] Polymarket {slug}: {e}")
    print("All connections work." if ok else "Some connections failed (see above).")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="test connections once and exit")
    parser.add_argument("--assets", default=",".join(Config.assets), help="e.g. btc,eth,sol")
    parser.add_argument("--minutes", type=int, default=Config.window_minutes, help="window length: 5 or 15")
    parser.add_argument("--min-edge", type=float, default=Config.min_edge)
    parser.add_argument("--stake", type=float, default=Config.stake_usd)
    parser.add_argument("--db", default=Config.db_path)
    args = parser.parse_args()

    cfg = Config(assets=tuple(args.assets.split(",")), window_minutes=args.minutes,
                 min_edge=args.min_edge, stake_usd=args.stake, db_path=args.db)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.check:
        raise SystemExit(0 if check(cfg) else 1)

    store = Store(cfg.db_path)
    trackers = [Tracker(a, cfg, store) for a in cfg.assets]
    log.info("Recording %s, %d-minute windows, into %s. No real money. Ctrl+C to stop.",
             ", ".join(a.upper() for a in cfg.assets), cfg.window_minutes, cfg.db_path)
    log.info("Paper bets start once %d minutes of price history exist.", cfg.vol_min_seconds // 60)

    last_settle = 0.0
    last_error = {}
    with ThreadPoolExecutor(len(trackers)) as pool:
        while True:
            now = time.time()
            for tracker, future in [(t, pool.submit(t.step, now)) for t in trackers]:
                try:
                    future.result()
                except Exception as e:
                    # Network hiccups are normal; log each asset's error at most once a minute.
                    if now - last_error.get(tracker.asset, 0) > 60:
                        log.warning("%s: %s", tracker.asset.upper(), e)
                        last_error[tracker.asset] = now
            if now - last_settle > 30:
                try:
                    settle(store, cfg, now)
                except Exception as e:
                    log.warning("settle: %s", e)
                last_settle = now
            time.sleep(max(0.0, cfg.poll_seconds - (time.time() - now)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run `python -m polybot.report` to see the results.")
