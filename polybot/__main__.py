"""Start the bot and its dashboard:

    python3 -m polybot            # then open http://localhost:8000
    python3 -m polybot --check    # test the connections once and exit
"""
import argparse
import logging
import os
import threading
import time
import webbrowser

from . import feeds
from .config import Config
from .engine import Engine
from .web import serve


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
            print(f"[ok] Polymarket market {slug}, resolves on: {market['resolution_source']},"
                  f" opening price shown: {market['price_to_beat']}")
            bids, asks = feeds.order_book(market["up_token"])
            print(f"[ok]   Up: best bid {max(p for p, _ in bids):.2f}, best ask {min(p for p, _ in asks):.2f}")
        except Exception as e:
            ok = False
            print(f"[FAIL] Polymarket {slug}: {e}")
    ok = check_streams(cfg) and ok
    print("All connections work." if ok else "Some connections failed (see above).")
    return ok


def check_streams(cfg, seconds=8):
    """Listen to both live streams briefly and report what arrived."""
    import asyncio

    prices, books_seen = {}, set()
    now = time.time()
    tokens = []
    for asset in cfg.assets:
        market = feeds.find_market(feeds.market_slug(cfg, asset, feeds.window_start(now, cfg.window_minutes)))
        if market:
            tokens += [market["up_token"], market["down_token"]]

    async def listen():
        price_stream = feeds.ChainlinkStream({a: f"{a}/usd" for a in cfg.assets},
                                             lambda a, ts, p: prices.__setitem__(a, (ts, p, time.time())))
        book_stream = feeds.BookStream(books_seen.add)
        book_stream.set_tokens(tokens)
        tasks = [asyncio.create_task(price_stream.run()), asyncio.create_task(book_stream.run())]
        await asyncio.sleep(seconds)
        for t in tasks:
            t.cancel()
        return price_stream, book_stream

    price_stream, book_stream = asyncio.run(listen())
    ok = True
    for asset in cfg.assets:
        if asset in prices:
            ts, p, got = prices[asset]
            print(f"[ok] {asset.upper()} Chainlink live: {p:,.4f} (arrived {got - ts:.2f}s after its timestamp)")
        else:
            ok = False
            print(f"[FAIL] {asset.upper()} Chainlink live: nothing received ({price_stream.last_error})")
    if tokens and set(tokens) <= books_seen:
        print(f"[ok] Order book stream: live books for all {len(tokens)} outcomes")
    else:
        ok = False
        print(f"[FAIL] Order book stream: {len(books_seen & set(tokens))}/{len(tokens)} books ({book_stream.last_error})")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="test connections once and exit")
    parser.add_argument("--assets", default=",".join(Config.assets), help="e.g. btc,eth,sol")
    parser.add_argument("--minutes", type=int, default=Config.window_minutes, help="window length: 5 or 15")
    parser.add_argument("--min-edge", type=float, default=Config.min_edge)
    parser.add_argument("--db", default=os.environ.get("POLYBOT_DB", Config.db_path))
    parser.add_argument("--host", default=os.environ.get("POLYBOT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--rest", action="store_true", help="poll once a second instead of live streams")
    args = parser.parse_args()

    cfg = Config(assets=tuple(args.assets.split(",")), window_minutes=args.minutes,
                 min_edge=args.min_edge, db_path=args.db)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.check:
        raise SystemExit(0 if check(cfg) else 1)

    password = os.environ.get("POLYBOT_PASSWORD") or None
    if args.host not in ("127.0.0.1", "localhost") and not password:
        raise SystemExit("Refusing to open the dashboard to the network without POLYBOT_PASSWORD set.")

    engine = Engine(cfg, streaming=not args.rest)
    engine.start()
    logging.getLogger("polybot").info(
        "Recording %s, %d-minute windows. Simulation only until the readiness checks pass.",
        ", ".join(a.upper() for a in cfg.assets), cfg.window_minutes)
    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        threading.Timer(1.5, webbrowser.open, (f"http://localhost:{args.port}",)).start()
    serve(engine, args.host, args.port, password)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
