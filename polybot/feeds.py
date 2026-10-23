"""Talking to the outside world: Coinbase for spot prices, Polymarket for markets and order books.

Read-only. Nothing in this file can place an order or touch money.
"""
import asyncio
import json
import logging
import time
import urllib.request

from .config import COINBASE_PRODUCTS

log = logging.getLogger("polybot")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
COINBASE = "https://api.exchange.coinbase.com"


def get_json(url, timeout=3.0):
    req = urllib.request.Request(url, headers={"User-Agent": "polybot-recorder/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def spot_price(asset):
    data = get_json(f"{COINBASE}/products/{COINBASE_PRODUCTS[asset]}/ticker")
    return float(data["price"])


def window_start(ts, minutes):
    size = minutes * 60
    return int(ts // size) * size


def market_slug(cfg, asset, start):
    return cfg.slug_template.format(asset=asset, minutes=cfg.window_minutes, start=start)


def _maybe_json(value):
    # Gamma returns some list fields as JSON-encoded strings.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _price_to_beat(event, market):
    """Polymarket's own opening price (from Chainlink) if the API exposes it."""
    for source in (event, market):
        meta = _maybe_json(source.get("eventMetadata")) or {}
        if isinstance(meta, dict) and meta.get("priceToBeat") not in (None, ""):
            try:
                return float(meta["priceToBeat"])
            except (TypeError, ValueError):
                pass
    return None


def find_market(slug):
    """Look up a window's market. Returns a dict, or None if Polymarket has no such market."""
    events = get_json(f"{GAMMA}/events?slug={slug}")
    event = events[0] if events else {}
    markets = event.get("markets") or get_json(f"{GAMMA}/markets?slug={slug}")
    if not markets:
        return None
    market = markets[0]
    outcomes = [o.lower() for o in _maybe_json(market.get("outcomes"))]
    tokens = _maybe_json(market.get("clobTokenIds"))
    by_outcome = dict(zip(outcomes, tokens))
    if "up" not in by_outcome or "down" not in by_outcome:
        return None
    fees = market.get("feeSchedule") or {}
    return {
        "slug": slug,
        "fee_rate": float(fees["rate"]) if market.get("feesEnabled") and "rate" in fees else
        (0.0 if market.get("feesEnabled") is False else None),
        "fee_exponent": float(fees.get("exponent", 1)),
        "min_order_size": float(market.get("orderMinSize") or 0),
        "tick_size": float(market.get("orderPriceMinTickSize") or 0.01),
        "up_token": by_outcome["up"],
        "down_token": by_outcome["down"],
        "price_to_beat": _price_to_beat(event, market),
        "resolution_source": market.get("resolutionSource") or event.get("resolutionSource"),
    }


def market_winner(slug):
    """"up" / "down" once Polymarket has resolved the market, else None.

    Gamma leaves closed markets out of /markets unless asked with closed=true, and a
    market stays open, trading near 0/1, for a minute or more after its window ends,
    so only a closed market with a 1/0 price counts as decided.
    """
    markets = get_json(f"{GAMMA}/markets?slug={slug}&closed=true")
    if not markets or not markets[0].get("closed"):
        return None
    market = markets[0]
    outcomes = [o.lower() for o in _maybe_json(market.get("outcomes"))]
    prices = [float(p) for p in _maybe_json(market.get("outcomePrices"))]
    for outcome, price in zip(outcomes, prices):
        if price >= 0.99:
            return outcome
    return None


def order_book(token_id):
    """Returns (bids, asks) as lists of (price, size)."""
    book = get_json(f"{CLOB}/book?token_id={token_id}")

    def levels(key):
        return [(float(lv["price"]), float(lv["size"])) for lv in book.get(key) or []]

    return levels("bids"), levels("asks")


# Live streams. Both run inside one asyncio loop (see engine.Engine) and call back
# synchronously on every update.

RTDS_URL = "wss://ws-live-data.polymarket.com"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


async def _keepalive(ws, every):
    # Polymarket closes idle sockets; both endpoints answer a text PING with PONG.
    while True:
        await asyncio.sleep(every)
        await ws.send("PING")


class ChainlinkStream:
    """Chainlink prices as Polymarket relays them: the same feed its crypto markets resolve on.

    One update per asset per second, stamped with Chainlink's own timestamp, arriving
    about 1.2s after that timestamp. Optionally also Binance prices from the same relay
    (about 0.3s behind), used to bring the Chainlink price up to the present.

    Each topic has its own connection and its own watchdog: the relay can go quiet on a
    topic while the socket stays open, so a topic silent for `stale_after` seconds is
    closed and reconnected, and the event is counted and kept for the dashboard.
    """

    def __init__(self, symbols, on_price, fast_symbols=None, on_fast=None, stale_after=10.0):
        self.by_symbol = {symbol: asset for asset, symbol in symbols.items()}
        self.fast_by_symbol = {symbol: asset for asset, symbol in (fast_symbols or {}).items()}
        self.on_price = on_price  # (asset, source_ts_seconds, price)
        self.on_fast = on_fast
        self.stale_after = stale_after
        self.connected = False
        self.last_message = None
        self.last_price = None  # when the latest Chainlink price for a followed asset arrived
        self.last_fast = None
        self.last_error = None
        self.reconnects = 0

    async def run(self):
        topics = [("crypto_prices_chainlink", "*")]
        if self.on_fast:
            topics.append(("crypto_prices", "update"))
        await asyncio.gather(*(self._run_topic(topic, kind) for topic, kind in topics))

    async def _run_topic(self, topic, kind):
        import websockets

        backoff = 1
        while True:
            try:
                async with websockets.connect(RTDS_URL, ping_interval=None, open_timeout=10) as ws:
                    await ws.send(json.dumps({"action": "subscribe", "subscriptions": [
                        {"topic": topic, "type": kind, "filters": ""}]}))
                    if topic == "crypto_prices_chainlink":
                        self.connected = True
                    self._mark(topic)  # the watchdog's clock starts at connect
                    pinger = asyncio.create_task(_keepalive(ws, 5))
                    watchdog = asyncio.create_task(self._watch(ws, topic))
                    try:
                        async for raw in ws:
                            self.last_message = time.time()
                            self._handle(raw)
                            backoff = 1
                    finally:
                        pinger.cancel()
                        watchdog.cancel()
            except Exception as e:
                self.last_error = f"{topic}: {type(e).__name__}: {e}"
            if topic == "crypto_prices_chainlink":
                self.connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def _mark(self, topic):
        if topic == "crypto_prices_chainlink":
            self.last_price = time.time()
        else:
            self.last_fast = time.time()

    async def _watch(self, ws, topic):
        while True:
            await asyncio.sleep(2)
            last = self.last_price if topic == "crypto_prices_chainlink" else self.last_fast
            if last and time.time() - last > self.stale_after:
                self.reconnects += 1
                self.last_error = (f"{time.strftime('%H:%M:%S')} {topic}: no prices for "
                                   f"{time.time() - last:.0f}s, reconnecting")
                log.warning(self.last_error)
                await ws.close()
                return

    def _handle(self, raw):
        if not raw or raw == "PONG":
            return
        try:
            msg = json.loads(raw)
            payload = msg.get("payload") or {}
        except (ValueError, AttributeError):
            return
        if payload.get("value") is None or not payload.get("timestamp"):
            return
        ts, value = payload["timestamp"] / 1000.0, float(payload["value"])
        if msg.get("topic") == "crypto_prices_chainlink":
            asset = self.by_symbol.get(payload.get("symbol"))
            if asset:
                self.last_price = time.time()
                self.on_price(asset, ts, value)
        elif msg.get("topic") == "crypto_prices" and self.on_fast:
            asset = self.fast_by_symbol.get(payload.get("symbol"))
            if asset:
                self.last_fast = time.time()
                self.on_fast(asset, ts, value)


class BookStream:
    """Full order books for a set of Polymarket tokens, kept current from the market channel.

    When the set of tokens changes, a new connection is opened for the new set and the
    old one is closed only after the new one has delivered its snapshots, so the books
    never go dark (make-before-break).
    """

    def __init__(self, on_update):
        self.on_update = on_update  # (token)
        self.books = {}  # token -> {"bids": {price: size}, "asks": {price: size}, "ts": float}
        self.tokens = frozenset()
        self.connected = False
        self.last_message = None
        # Polymarket stamps each update with its own send time. The newest stamp we
        # have applied says how current our books are; a backlog shows up here.
        self.last_server_ts = None
        self.reconnects = 0
        self.lag = None  # seconds between Polymarket sending the latest update and us applying it
        self.last_error = None
        self._wanted = asyncio.Event()

    def set_tokens(self, tokens):
        tokens = frozenset(tokens)
        if tokens != self.tokens:
            self.tokens = tokens
            self._wanted.set()

    def get(self, token):
        """(bids, asks, ts): ts is the Polymarket time our view of this book is current as of.

        A quiet book is as current as the newest update applied on the connection, so its
        age measures the real delay: network, plus any backlog in applying updates.
        """
        book = self.books.get(token)
        if not book:
            return None
        ts = book["ts"]
        if self.connected and self.last_server_ts:
            ts = max(ts, self.last_server_ts)
        return list(book["bids"].items()), list(book["asks"].items()), ts

    async def run(self):
        current = None  # (task, tokens)
        while True:
            await self._wanted.wait()
            self._wanted.clear()
            tokens = self.tokens
            if not tokens:
                continue
            ready = asyncio.Event()
            task = asyncio.create_task(self._connection(tokens, ready))
            try:
                await asyncio.wait_for(ready.wait(), 15)
            except asyncio.TimeoutError:
                pass  # keep the old connection too; the new one keeps retrying
            if current:
                current[0].cancel()
            current = (task, tokens)
            for stale in set(self.books) - set(tokens):
                del self.books[stale]

    async def _connection(self, tokens, ready):
        import websockets

        backoff = 1
        while True:
            try:
                async with websockets.connect(CLOB_WS_URL, ping_interval=None, open_timeout=10,
                                              max_size=None) as ws:
                    await ws.send(json.dumps({"assets_ids": sorted(tokens), "type": "market"}))
                    pinger = asyncio.create_task(_keepalive(ws, 5))
                    watchdog = asyncio.create_task(self._watch(ws))
                    self.last_message = time.time()
                    seen = set()
                    try:
                        async for raw in ws:
                            self.last_message = time.time()
                            self.connected = True
                            seen |= self._handle(raw)
                            if not ready.is_set() and tokens <= seen:
                                ready.set()
                            backoff = 1
                    finally:
                        pinger.cancel()
                        watchdog.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
            self.connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _watch(self, ws, stale_after=15.0):
        """With dozens of books subscribed the market channel is never quiet for long;
        silence means a stalled stream, so close it and let the loop reconnect."""
        while True:
            await asyncio.sleep(3)
            if self.last_message and time.time() - self.last_message > stale_after:
                self.reconnects += 1
                self.last_error = f"{time.strftime('%H:%M:%S')} order books silent, reconnecting"
                log.warning(self.last_error)
                await ws.close()
                return

    def _handle(self, raw):
        """Apply one message; returns the tokens that received a full snapshot."""
        if not raw or raw == "PONG":
            return set()
        try:
            msg = json.loads(raw)
        except ValueError:
            return set()
        now = time.time()
        snapshots, touched = set(), set()
        for event in msg if isinstance(msg, list) else [msg]:
            try:
                sent = float(event.get("timestamp")) / 1000.0
            except (TypeError, ValueError):
                sent = now
            if self.last_server_ts is None or sent > self.last_server_ts:
                self.last_server_ts = sent
            self.lag = now - sent
            if "bids" in event and "asks" in event and "asset_id" in event:
                token = event["asset_id"]
                self.books[token] = {
                    "bids": {float(l["price"]): float(l["size"]) for l in event["bids"] if float(l["size"]) > 0},
                    "asks": {float(l["price"]): float(l["size"]) for l in event["asks"] if float(l["size"]) > 0},
                    "ts": sent,
                }
                snapshots.add(token)
                touched.add(token)
            for change in event.get("price_changes") or []:
                book = self.books.get(change.get("asset_id"))
                if book is None:
                    continue
                side = book["bids"] if change["side"] == "BUY" else book["asks"]
                price, size = float(change["price"]), float(change["size"])
                if size > 0:
                    side[price] = size
                else:
                    side.pop(price, None)
                book["ts"] = sent
                touched.add(change["asset_id"])
        for token in touched:
            self.on_update(token)
        return snapshots


class ExchangeStream:
    """Prices straight from an exchange's own websocket: several updates a second with a delay
    of tens of milliseconds, against about one a second and 0.3-1.2s through Polymarket's relay.

    Subclasses say where to connect, what to send, and how to read a message. Each asset is
    passed on at most every `min_interval` seconds (the newest price wins), which keeps the
    engine's load flat however busy the market gets. A silent connection is reconnected.
    """

    name = "exchange"
    url = ""

    def __init__(self, symbols, on_tick, min_interval=0.05, stale_after=15.0):
        self.by_symbol = {symbol: asset for asset, symbol in symbols.items()}
        self.symbols = symbols
        self.on_tick = on_tick  # (source, asset, ts, price)
        self.min_interval = min_interval
        self.stale_after = stale_after
        self.connected = False
        self.last_message = None
        self.last_tick = {}  # asset -> when it was last passed on
        self.pending = {}  # asset -> (ts, price) held back by the rate limit
        self.last_error = None
        self.reconnects = 0

    def subscribe_messages(self):
        return []

    def parse(self, msg):
        """(symbol, ts, price) or None."""
        raise NotImplementedError

    async def run(self):
        import websockets

        backoff = 1
        while True:
            try:
                async with websockets.connect(self.url, open_timeout=10, max_size=None) as ws:
                    for message in self.subscribe_messages():
                        await ws.send(json.dumps(message))
                    self.connected, self.last_message = True, time.time()
                    watchdog = asyncio.create_task(self._watch(ws))
                    try:
                        async for raw in ws:
                            self.last_message = time.time()
                            self._handle(raw)
                            backoff = 1
                    finally:
                        watchdog.cancel()
            except Exception as e:
                self.last_error = f"{self.name}: {type(e).__name__}: {e}"
            self.connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _watch(self, ws):
        while True:
            await asyncio.sleep(3)
            if self.last_message and time.time() - self.last_message > self.stale_after:
                self.reconnects += 1
                self.last_error = f"{time.strftime('%H:%M:%S')} {self.name}: silent, reconnecting"
                log.warning(self.last_error)
                await ws.close()
                return

    def _handle(self, raw):
        try:
            parsed = self.parse(json.loads(raw))
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        if not parsed:
            return
        symbol, ts, price = parsed
        asset = self.by_symbol.get(symbol)
        if asset is None or price <= 0:
            return
        now = time.time()
        if now - self.last_tick.get(asset, 0) < self.min_interval:
            self.pending[asset] = (ts, price)  # the newest wins; flush() sends it shortly
            return
        self.pending.pop(asset, None)
        self.last_tick[asset] = now
        self.on_tick(self.name, asset, ts, price)

    def flush(self):
        """Pass on prices held back by the rate limit (called a few times a second)."""
        now = time.time()
        for asset, (ts, price) in list(self.pending.items()):
            if now - self.last_tick.get(asset, 0) >= self.min_interval:
                del self.pending[asset]
                self.last_tick[asset] = now
                self.on_tick(self.name, asset, ts, price)


class BinanceStream(ExchangeStream):
    """Binance spot best bid/ask (bookTicker) as a mid price, for every change at the top."""

    name = "binance"

    def __init__(self, symbols, on_tick, **kw):
        super().__init__(symbols, on_tick, **kw)
        streams = "/".join(f"{s.lower()}@bookTicker" for s in symbols.values())
        self.url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    def parse(self, msg):
        d = msg.get("data", msg)
        # Spot bookTicker carries no event time; it is sent the instant the top changes, so
        # arrival time is accurate to the network delay (tens of milliseconds).
        return d["s"], time.time(), (float(d["b"]) + float(d["a"])) / 2


class CoinbaseStream(ExchangeStream):
    """Coinbase ticker (best bid/ask after every trade) as a mid price, in USD like Chainlink."""

    name = "coinbase"
    url = "wss://ws-feed.exchange.coinbase.com"

    def subscribe_messages(self):
        # One subscription per product: a product Coinbase doesn't list fails on its own
        # instead of taking the whole subscription down with it.
        return [{"type": "subscribe", "product_ids": [p], "channels": ["ticker"]} for p in self.symbols.values()]

    def parse(self, msg):
        if msg.get("type") != "ticker":
            if msg.get("type") == "error":
                self.last_error = f"coinbase: {msg.get('reason') or msg.get('message')}"
            return None
        bid, ask = msg.get("best_bid"), msg.get("best_ask")
        price = (float(bid) + float(ask)) / 2 if bid and ask else float(msg["price"])
        ts = dt_parse(msg["time"]) if msg.get("time") else time.time()
        return msg["product_id"], ts, price


def dt_parse(iso):
    """Coinbase's '2026-10-23T21:57:43.123456Z' as unix seconds."""
    import datetime as _dt

    return _dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


BINANCE_SYMBOLS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT", "doge": "DOGEUSDT",
                   "bnb": "BNBUSDT", "hype": "HYPEUSDT"}
COINBASE_SYMBOLS = {"btc": "BTC-USD", "eth": "ETH-USD", "sol": "SOL-USD", "xrp": "XRP-USD", "doge": "DOGE-USD",
                    "bnb": "BNB-USD", "hype": "HYPE-USD"}
