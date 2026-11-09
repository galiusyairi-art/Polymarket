"""Start the bot and its dashboard:

    python3 -m polybot            # then open http://localhost:8000
    python3 -m polybot --check    # test the connections once and exit
"""
import argparse
import logging
import os
import signal
import threading
import time
import webbrowser

from . import feeds, startup
from .config import Config
from .engine import Engine
from .store import Store
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
    parser.add_argument("--no-lab", action="store_true", help="run only the main strategy")
    parser.add_argument("--import", dest="import_zip", metavar="ZIP",
                        help="carry on from another machine: load an export ZIP (settings, trades) first")
    parser.add_argument("--no-exchanges", action="store_true",
                        help="don't connect to Binance and Coinbase directly (use Polymarket's relay only)")
    load_env_file()
    args = parser.parse_args()

    cfg = Config(assets=tuple(args.assets.split(",")), window_minutes=args.minutes,
                 min_edge=args.min_edge, db_path=args.db, exchange_feeds=not args.no_exchanges)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.check:
        raise SystemExit(0 if check(cfg) else 1)

    if args.import_zip:
        from . import export
        Store(cfg.db_path).close()  # the tables, on a new machine
        counts = export.read(cfg, args.import_zip)
        logging.getLogger("polybot").warning("Imported %s: %s", args.import_zip, counts)

    password = os.environ.get("POLYBOT_PASSWORD") or None
    if args.host not in ("127.0.0.1", "localhost") and not password:
        raise SystemExit("Refusing to open the dashboard to the network without POLYBOT_PASSWORD set.")

    # (the owner, 09.11.2026: "an update used to be quick, it never took the whole site down": the site answers from the
    # moment the process starts, a short "updating" page, while the engine is built, which takes longer as the data
    # grows; before, nothing answered on the port until then, and the browser showed Fly's bare 503)
    warm = warming_server(args.host, args.port)
    try:
        engine = Engine(cfg, streaming=not args.rest, with_lab=not args.no_lab)
        startup.stage("start")
        engine.start()
        startup.stage("web")
    finally:
        if warm is not None:
            warm.shutdown()
            warm.server_close()
    logging.getLogger("polybot").info(
        "Recording %s, %d-minute windows. Simulation only until the readiness checks pass.",
        ", ".join(a.upper() for a in cfg.assets), cfg.window_minutes)
    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        threading.Timer(1.5, webbrowser.open, (f"http://localhost:{args.port}",)).start()
    # A deploy stops this machine (Fly: SIGINT, then SIGKILL 5 s later). The IB Gateway connection is closed
    # first, properly: otherwise the gateway keeps the old one, under the same client id, and refuses the new
    # server's for minutes (the owner's alert mail of 29.10.2026, after three deploys in a row).
    signal.signal(signal.SIGTERM, lambda *_: (shutdown(engine), os._exit(0)))
    if not args.rest:  # (the guard: a bot that stops answering restarts by itself; 06.11.2026)
        from .watchdog import Watchdog
        Watchdog(engine, args.port, stop=lambda: shutdown(engine)).start()
    try:
        serve(engine, args.host, args.port, password)
    finally:
        shutdown(engine)


WARMING_PAGE = """<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>מתעדכן…</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0d1117;color:#e6edf3;font:16px system-ui,Arial,sans-serif}
.box{text-align:center;padding:24px}.ring{width:46px;height:46px;margin:0 auto 16px;border:4px solid #30363d;border-top-color:#58a6ff;
border-radius:50%;animation:s 1s linear infinite}@keyframes s{to{transform:rotate(360deg)}}small{color:#8b949e}
.bar{width:min(360px,80vw);height:10px;margin:14px auto 8px;background:#30363d;border-radius:99px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:#58a6ff;transition:width .9s ease}
.atlas{display:block;width:min(200px,50vw);height:auto;margin:0 auto 6px}</style></head>
<body><div class="box"><img class="atlas" src="/atlas-loader.svg" alt="" width="200" height="200"><b>הבוט מתעדכן לגרסה חדשה</b><div class="bar"><i id="f"></i></div>
<div id="t">עולה…</div><small id="e">הדף יעבור לדשבורד לבד</small></div>
<script>
// (each second: the start's stage and its bar, from /api/boot; once the bot is up, the dashboard. 09.11.2026)
let shown = 0;
async function look() {
  try {
    const r = await fetch("/api/boot", { cache: "no-store" });
    const b = await r.json();
    if (b.done) { location.reload(); return; }
    shown = Math.max(shown, b.pct || 0);
    document.getElementById("f").style.width = shown + "%";
    document.getElementById("t").textContent = `${Math.round(shown)}% · שלב ${b.index} מתוך ${b.of}: ${b.name}`;
    document.getElementById("e").textContent = `עולה כבר ${b.elapsed} שניות · השלב הזה ${b.in_stage} שניות`;
  } catch (e) { document.getElementById("t").textContent = "השרת עולה מחדש…"; }
  setTimeout(look, 1000);
}
look();
</script></body></html>"""


def warming_server(host, port):
    """Until the engine is built: every page gets the short "updating" one (it refreshes itself), the page's data asks
    get 503 {"warming": true} (the open dashboard keeps its last data and its loading bar), and /healthz answers. None
    when the port can't be had."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json as _json

    class Warm(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                body, kind, code = b"warming", "text/plain", 200
            elif path == "/atlas-loader.svg":  # (its Atlas, animated: 09.11.2026)
                with open(os.path.join(os.path.dirname(__file__), "static", "atlas-loader.svg"), "rb") as f:
                    body, kind, code = f.read(), "image/svg+xml", 200
            elif path == "/api/boot":
                body, kind, code = _json.dumps(startup.view()).encode(), "application/json", 200
            elif path.startswith("/api/"):
                body, kind, code = _json.dumps({"warming": True, "error": "השרת עולה", "boot": startup.view()}).encode(), \
                    "application/json", 503
            else:
                body, kind, code = WARMING_PAGE.encode(), "text/html; charset=utf-8", 200
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = _json.dumps({"warming": True, "error": "השרת עולה: נסה שוב בעוד רגע"}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        srv = ThreadingHTTPServer((host, port), Warm)
    except OSError as e:
        logging.getLogger("polybot").warning("Warming page: %s", e)
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="warming").start()
    return srv


def shutdown(engine, wait=3.0):
    """Stop the engine's loops and close the IB Gateway connection, waiting at most `wait` seconds."""
    engine.stop()
    atlas = getattr(engine, "atlas", None)
    gws = [g for g in (getattr(atlas, "ibkr", None), getattr(atlas, "_data_gw", None)) if g is not None]
    threads = [threading.Thread(target=g.close, daemon=True, name="ibkr-close") for g in gws]
    for t in threads:
        t.start()
    for t in threads:
        t.join(wait)


def load_env_file():
    """On the owner's own computer, secrets live in polybot.env next to start.bat (KEY=value per
    line), never in the code or the database. Values already set in the environment win."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(os.getcwd(), "polybot.env"), os.path.join(here, "polybot.env")):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
