"""Experimental lab: Polymarket's daily stock markets, on paper.

Polymarket lists daily markets on US stocks and ETFs ("Apple closes above $345 on
October 26?", "SPY Up or Down on October 26?") that resolve on Pyth's closing price at
the 4pm New York close. Polymarket relays the same live prices (topic equity_prices,
about one update per second per symbol), so the same question as the crypto markets
applies: given where the price is now, how much time is left and how much it moves,
what is the fair chance it closes above the line, and is the order book cheaper?

A study of 5,851 resolved markets (Aug-Oct 2026) found the books slow near the close:
sides priced 70-95c five minutes before the close won about 98% of the time. That used
mid prices, not the prices one could actually buy at, so this lab measures it for real:
every variant below bets on paper against the live order books, and is settled by
Polymarket's own results. Nothing here touches the crypto strategy or real money.
"""
import dataclasses
import datetime as dt
import json
import logging
import re
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from . import feeds, lab, stats
from .model import VolatilityEstimator, fair_up_probability
from .paper import find_bet, settle_pnl

log = logging.getLogger("polybot")
NY = ZoneInfo("America/New_York")
SLUG = re.compile(r"^([a-z0-9]+)-(up-or-down|close-above|closes-above|above)-on-([a-z]+)-(\d+)-(\d{4})$")
NAMES = {"spy": "S&P 500 (SPY)", "qqq": "נאסד\"ק (QQQ)", "aapl": "אפל", "msft": "מיקרוסופט", "nvda": "אנבידיה",
         "tsla": "טסלה", "amzn": "אמזון", "googl": "גוגל", "meta": "מטא", "nflx": "נטפליקס", "pltr": "פלנטיר",
         "coin": "קוינבייס", "hood": "רובינהוד", "abnb": "Airbnb", "rklb": "Rocket Lab", "open": "Opendoor",
         "ewy": "קוריאה (EWY)", "mu": "מיקרון", "spcx": "SPCX"}
WATCH_SECONDS = 3 * 3600  # follow a market's book from three hours before its close
PYTH_HISTORY = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
PYTH_BENCHMARKS = "https://benchmarks.pyth.network/v1/updates/price"
HERMES = "https://hermes.pyth.network"


@dataclass(frozen=True)
class StockVariant:
    kind: str  # "model": our fair price against the book; "fav": buy the favourite, no model
    level: float  # model: minimum edge; fav: lowest price that counts as the favourite
    max_seconds_left: int

    @property
    def key(self):
        return f"{self.kind}|{self.level:.2f}|{self.max_seconds_left}"

    @classmethod
    def from_key(cls, key):
        kind, level, seconds = key.split("|")
        return cls(kind, float(level), int(seconds))


VARIANTS = tuple(
    [StockVariant("model", e, s) for e in (0.03, 0.05, 0.08) for s in (300, 900, 1800, 3600, 10800)]
    + [StockVariant("fav", 0.70, 300), StockVariant("fav", 0.70, 900)]
)


def describe(key):
    v = StockVariant.from_key(key)
    when = (f"ב-{v.max_seconds_left // 60} הדקות האחרונות" if v.max_seconds_left < 3600
            else f"ב-{v.max_seconds_left // 3600} השעות האחרונות" if v.max_seconds_left > 3600 else "בשעה האחרונה")
    if v.kind == "fav":
        return f"קונה את הצד המוביל כשהוא עולה {v.level * 100:.0f}-95¢, {when} לפני הסגירה (בלי מודל)"
    return f"פער של {v.level * 100:.0f} נקודות ומעלה מול המחיר ההוגן שלנו, {when} לפני הסגירה"


def _maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return []
    return value or []


def _strike(market):
    """The market's line: "$345" in its title, which must agree with "-above-345-on-" in its slug.
    (groupItemThreshold is only the item's position in the list: 0, 1, 2... never the price.)"""
    title = str(market.get("groupItemTitle") or "").replace("$", "").replace(",", "").strip()
    try:
        from_title = float(title)
    except ValueError:
        from_title = None
    m = re.search(r"-above-(\d+(?:pt\d+)?)-on-", market.get("slug") or "")
    from_slug = float(m.group(1).replace("pt", ".")) if m else None
    if from_title is not None and from_slug is not None and abs(from_title - from_slug) > 1e-6:
        return None  # they disagree: don't guess
    return from_title if from_title is not None else from_slug


def parse_market(event_slug, market):
    """A followed market from Gamma's event/market, or None if it isn't one we handle."""
    m = SLUG.match(event_slug)
    if not m or market.get("closed") or not market.get("enableOrderBook", True):
        return None
    symbol, kind = m.group(1), m.group(2)
    tokens = _maybe_json(market.get("clobTokenIds"))
    try:
        close = dt.datetime.fromisoformat(market["endDate"].replace("Z", "+00:00")).timestamp()
    except (KeyError, ValueError, AttributeError):
        return None
    if len(tokens) != 2:
        return None
    strike = None if kind == "up-or-down" else _strike(market)
    if kind != "up-or-down" and strike is None:
        return None
    fee = market.get("feeSchedule") or {}
    return {
        "slug": market.get("slug") or event_slug, "event": event_slug, "symbol": symbol, "kind": kind,
        "strike": strike, "close": close, "yes_token": str(tokens[0]), "no_token": str(tokens[1]),
        "title": market.get("groupItemTitle") or "", "question": market.get("question") or "",
        "fee_rate": float(fee.get("rate") or 0.0), "fee_exponent": float(fee.get("exponent") or 1.0),
        "tick_size": float(market.get("orderPriceMinTickSize") or 0.01),
        "min_order_size": float(market.get("orderMinSize") or 5),
    }


def discover(now, get=feeds.get_json, horizon=4 * 86400):
    """Every stock market closing within `horizon` seconds (four days: over a weekend, Monday's)."""
    fmt = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    found = {}
    for offset in range(0, 1000, 100):
        events = get(f"{feeds.GAMMA}/events?active=true&closed=false&tag_slug=stocks&limit=100&offset={offset}"
                     f"&end_date_min={fmt(now - 600)}&end_date_max={fmt(now + horizon)}", timeout=15)
        for event in events:
            for market in event.get("markets") or []:
                info = parse_market(event.get("slug", ""), market)
                if info:
                    found[info["slug"]] = info
        if len(events) < 100:
            break
    return found


def pyth_close(symbol, close_ts, get=feeds.get_json):
    """Pyth's close for a session: the close of its final 1-minute candle (TradingView history)."""
    data = get(f"{PYTH_HISTORY}?symbol={urllib.parse.quote(f'Equity.US.{symbol.upper()}/USD')}&resolution=1"
               f"&from={int(close_ts) - 600}&to={int(close_ts)}", timeout=10)
    if not data or data.get("s") != "ok":
        return None
    candles = [(t, c) for t, c in zip(data.get("t", []), data.get("c", [])) if t < close_ts]
    return float(candles[-1][1]) if candles else None


def pyth_feed_id(symbol, get=feeds.get_json):
    """The 64-hex id of a US stock's Pyth price feed."""
    for feed in get(f"{HERMES}/v2/price_feeds?query={symbol.upper()}&asset_type=equity", timeout=10) or []:
        attrs = feed.get("attributes") or {}
        if (str(attrs.get("base", "")).upper() == symbol.upper() and str(attrs.get("quote_currency", "")).upper() == "USD"
                and str(attrs.get("country", "US")).upper() == "US"):
            return feed.get("id")
    return None


def pyth_price_at(feed_id, ts, get=feeds.get_json):
    """Pyth's price for a feed at a past second (its Benchmarks history)."""
    data = get(f"{PYTH_BENCHMARKS}/{int(ts)}?ids={feed_id}&parsed=true", timeout=10)
    parsed = (data or {}).get("parsed") or []
    if not parsed:
        return None
    price = parsed[0]["price"]
    return int(price["price"]) * 10 ** int(price["expo"])


def previous_sessions(close_ts, days=5):
    """4pm New York on the weekdays before close_ts, newest first (holidays are found by trying)."""
    day = dt.datetime.fromtimestamp(close_ts, NY).date()
    out = []
    while len(out) < days:
        day -= dt.timedelta(days=1)
        if day.weekday() < 5:
            out.append(dt.datetime(day.year, day.month, day.day, 16, tzinfo=NY).timestamp())
    return out


def ny_close(ts):
    """4pm New York on the day of ts (the regular close; Polymarket's endDate covers early closes)."""
    day = dt.datetime.fromtimestamp(ts, NY).date()
    return dt.datetime(day.year, day.month, day.day, 16, tzinfo=NY).timestamp()


class EquityStream:
    """Live stock prices from Polymarket's relay, the ones its stock markets show."""

    TOPIC = "equity_prices"

    def __init__(self, on_price, stale_after=60.0):
        self.on_price = on_price  # (symbol, source_ts_seconds, price)
        self.stale_after = stale_after
        self.connected = False
        self.last_price = None
        self.last_error = None
        self.reconnects = 0

    async def run(self):
        import asyncio
        import websockets

        backoff = 1
        while True:
            try:
                async with websockets.connect(feeds.RTDS_URL, ping_interval=None, open_timeout=10) as ws:
                    await ws.send(json.dumps({"action": "subscribe", "subscriptions": [
                        {"topic": self.TOPIC, "type": "*", "filters": ""}]}))
                    self.connected = True
                    connected_at = time.time()
                    pinger = asyncio.create_task(feeds._keepalive(ws, 5))
                    try:
                        while True:
                            raw = await asyncio.wait_for(ws.recv(), self.stale_after)
                            self.handle(raw)
                            backoff = 1
                    except asyncio.TimeoutError:
                        # Quiet outside trading hours is normal; reconnect quietly and keep listening.
                        if time.time() - connected_at > self.stale_after:
                            self.reconnects += 1
                    finally:
                        pinger.cancel()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
            self.connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def handle(self, raw):
        if not raw or raw == "PONG":
            return
        try:
            msg = json.loads(raw)
            payload = msg.get("payload") or {}
        except (ValueError, AttributeError):
            return
        if msg.get("topic") != self.TOPIC or payload.get("value") is None or not payload.get("timestamp"):
            return
        self.last_price = time.time()
        self.on_price(str(payload.get("symbol", "")).lower(), payload["timestamp"] / 1000.0, float(payload["value"]))


class StockLab:
    def __init__(self, cfg, store, get=feeds.get_json):
        self.cfg, self.store, self.get = cfg, store, get
        self.count_since = 0  # the owner started counting afresh then (the results, not the trades)
        self.markets = {}  # slug -> parse_market dict
        self.vols = {}  # symbol -> VolatilityEstimator
        self.last = {}  # symbol -> (source ts, price, arrived)
        self.discovered_at = None
        self.last_error = None
        self.stream = None
        self.closes = self._load_closes()
        self._close_tried_at = 0.0
        self._feed_ids = {}
        self.missing_lines = []
        self.done = {(m, v) for m, v in store._x("SELECT market, variant FROM bets WHERE mode = 'stocks'")}
        # The stock market's main strategy: a lab variant the owner pushed into the simulation
        # (and, when they choose, real money). Bets once per market, like the crypto one.
        self.live = None
        self.dispatch = lambda fn, *args: fn(*args)
        self.main_rule = None
        self.enabled = True  # the owner can switch the stock strategy's simulation off
        self.can_open = None  # the engine's open-trade and daily-loss limits
        self.sim_stake = cfg.stake_usd
        self.min_ratio = self.max_ratio = 0.0  # the simulation's risk/reward range, set by the engine
        self._load_main()
        self.main_done = {m for (m,) in store._x("SELECT market FROM bets WHERE mode = 'paper' AND minutes = 0")}
        self.live_done = {m for (m,) in store._x("SELECT market FROM bets WHERE mode = 'live' AND minutes = 0")}

    # The main stock strategy
    def main(self):
        try:
            value = json.loads(self.store.get_setting("stocks_main") or "null")
        except ValueError:
            return None
        return value if value and value.get("variant") else None

    def _load_main(self):
        main = self.main()
        self.main_rule = StockVariant.from_key(main["variant"]) if main else None

    def set_main(self, variant):
        """Make a lab variant the stock market's main strategy (None: no stock strategy)."""
        if variant is not None and variant not in {v.key for v in VARIANTS}:
            raise ValueError("אסטרטגיה לא מוכרת")
        self.store.set_setting("stocks_main", json.dumps({"variant": variant, "since": time.time()} if variant else None))
        self._load_main()

    def readiness(self, db, now=None):
        """Is the stock strategy ready for real money? Its own simulated bets since it was pushed."""
        main = self.main()
        if not main:
            return None
        summary = stats.bet_summary(db, "paper", since=main["since"], strategy="stocks")
        rows = summary["rows"]
        n = len(rows)
        _, lower = lab.money_weighted(rows, 1.65)
        cfg = self.cfg
        tail = rows[-cfg.ready_recent_bets:]
        recent = lab.money_weighted(tail, 1.65)[0] if len(tail) >= min(cfg.ready_recent_bets, 10) else None
        if n < cfg.dash_min_bets:
            lower = None  # a handful of bets can't tell skill from luck either way
        days = len({dt.datetime.fromtimestamp(t, NY).date() for (t,) in db.execute(
            "SELECT ts FROM bets WHERE mode = 'paper' AND minutes = 0 AND pnl IS NOT NULL AND ts >= ?",
            (main["since"],)).fetchall()})
        checks = [
            {"id": "sdays", "value": days, "target": 5, "passed": days >= 5},
            {"id": "bets", "value": n, "target": cfg.ready_min_bets, "passed": n >= cfg.ready_min_bets},
            {"id": "roi", "value": summary["roi"], "target": self.cfg.ready_min_roi,
             "passed": summary["roi"] is not None and summary["roi"] >= self.cfg.ready_min_roi},
            {"id": "luck", "value": lower, "target": 0.0, "passed": lower is not None and lower > 0},
            {"id": "recent", "value": recent, "target": 0.0, "count": cfg.ready_recent_bets,
             "passed": recent is not None and recent > 0},
        ]
        return {"ready": all(c["passed"] for c in checks), "checks": checks, "since": main["since"]}

    def recommendation(self, board):
        """Which lab variant, if any, is worth pushing into the simulation as the stock strategy."""
        current = (self.main() or {}).get("variant")
        best = next((b for b in board if b["status"] == "promising"), None)
        if best and best["variant"] != current:
            return {"verdict": "push", "variant": best["variant"], "text": best["text"],
                    "why": f"{best['bets']} הימורים, {best['roi'] * 100:+.1f}%, וגם בתרחיש הפסימי {best['roi_pessimistic'] * 100:+.1f}%"}
        if best:
            return {"verdict": "current", "variant": best["variant"], "text": best["text"],
                    "why": "האסטרטגיה הכי טובה כבר רצה בסימולציה"}
        leader = next((b for b in board if b["bets"]), None)
        return {"verdict": "wait", "variant": None, "text": None,
                "why": (f"המובילה ({leader['text']}) עוד לא הוכיחה את עצמה: {leader['bets']} מתוך {self.cfg.lab_min_bets} הימורים"
                        if leader else "עוד אין הימורים שהוכרעו בניסוי")}

    # Prices
    def on_price(self, symbol, ts, price):
        prev = self.last.get(symbol)
        if prev and ts <= prev[0]:
            return
        if prev:
            # The close is Pyth's last price before 4pm: the latest sample before the boundary.
            close = ny_close(prev[0])
            if prev[0] < close <= ts and dt.datetime.fromtimestamp(close, NY).weekday() < 5:
                self._record_close(symbol, close, prev[1])
        self.last[symbol] = (ts, price, time.time())
        vol = self.vols.get(symbol)
        if vol is None:
            vol = self.vols[symbol] = VolatilityEstimator(self.cfg.vol_lookback_seconds)
        vol.add(ts, price)

    def _load_closes(self):
        try:
            return {s: [tuple(x) for x in v] for s, v in json.loads(self.store.get_setting("stock_closes") or "{}").items()}
        except ValueError:
            return {}

    def _record_close(self, symbol, close_ts, price):
        series = [x for x in self.closes.get(symbol, []) if x[0] != close_ts] + [(close_ts, price)]
        self.closes[symbol] = sorted(series)[-10:]
        self.store.set_setting("stock_closes", json.dumps(self.closes))

    def prior_close(self, symbol, close_ts):
        """The most recent recorded close before this market's close (an Up or Down market's line)."""
        earlier = [(t, p) for t, p in self.closes.get(symbol, []) if t < close_ts - 3600]
        if not earlier or close_ts - earlier[-1][0] > 4 * 86400:  # a long gap: a close is missing
            return None
        return earlier[-1][1]

    def line(self, market):
        return market["strike"] if market["kind"] != "up-or-down" else self.prior_close(market["symbol"], market["close"])

    def volatility(self, symbol):
        vol = self.vols.get(symbol)
        if vol and vol.history_seconds() >= self.cfg.vol_min_seconds:
            return vol.per_sqrt_second()
        return None

    # Markets
    def refresh_markets(self, now=None):
        now = now or time.time()
        try:
            self.markets = discover(now, self.get)
            self.discovered_at = now
            self.last_error = None
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            raise
        self.fill_missing_closes(now)

    def fill_missing_closes(self, now):
        """An Up or Down market's line is the previous session's close. The stream records it at
        4pm while the system runs; when it wasn't running then, Pyth's history is asked, one
        stock at a time and gently (it limits requests). Without a line, the market is skipped."""
        missing = sorted({m["symbol"]: m for m in self.markets.values()
                          if m["kind"] == "up-or-down" and self.prior_close(m["symbol"], m["close"]) is None}.values(),
                         key=lambda m: m["symbol"])
        self.missing_lines = [m["symbol"] for m in missing]
        if not missing or now - self._close_tried_at < 1800:
            return
        self._close_tried_at = now
        failures = []
        for m in missing:
            try:
                session = next((t for t in previous_sessions(m["close"]) if t < now), None)
                price = self._fetch_close(m["symbol"], session) if session else None
            except urllib.error.HTTPError as e:
                failures.append(f"{m['symbol']}: HTTP {e.code}")
                if e.code == 429:
                    break  # asked too often: try again on the next round
                continue
            except Exception as e:
                failures.append(f"{m['symbol']}: {type(e).__name__}")
                continue
            if price:
                self._record_close(m["symbol"], session, price)
            else:
                failures.append(f"{m['symbol']}: no data")
            time.sleep(1.0)
        self.missing_lines = [m["symbol"] for m in missing if self.prior_close(m["symbol"], m["close"]) is None]
        if failures:
            # Not an error: those markets wait for today's close, which the stream records.
            log.info("Previous close not available yet for %d stocks (%s); recorded at today's close instead",
                     len(failures), ", ".join(failures[:4]))

    def _fetch_close(self, symbol, session):
        try:
            price = pyth_close(symbol, session, self.get)
            if price:
                return price
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise
        feed = self._feed_ids.get(symbol) or pyth_feed_id(symbol, self.get)
        if not feed:
            return None
        self._feed_ids[symbol] = feed
        return pyth_price_at(feed, session - 1, self.get)

    def watching(self, now):
        return [m for m in self.markets.values() if 0 < m["close"] - now <= WATCH_SECONDS]

    def tokens(self, now):
        return {t for m in self.watching(now) for t in (m["yes_token"], m["no_token"])}

    def evaluate(self, now, book):
        """Every variant against every market near its close. book(token) -> (bids, asks, ts) or None."""
        for m in self.watching(now):
            seconds_left = m["close"] - now
            if seconds_left < 10:
                continue
            last = self.last.get(m["symbol"])
            if not last or now - last[2] > 5 or now - last[0] > 10:
                continue
            line, vol = self.line(m), self.volatility(m["symbol"])
            yes, no = book(m["yes_token"]), book(m["no_token"])
            if not (yes and no) or now - min(yes[2], no[2]) > 10:
                continue
            fair = fair_up_probability(last[1], line, seconds_left, vol) if (line and vol) else None
            cfg = dataclasses.replace(self.cfg, fee_rate=m["fee_rate"], fee_exponent=m["fee_exponent"], min_ratio=0.0)
            for v in VARIANTS:
                if (m["slug"], v.key) in self.done or seconds_left > v.max_seconds_left:
                    continue
                fill = self._decide(v, fair, yes[1], no[1], cfg)
                if fill is None:
                    continue
                self.done.add((m["slug"], v.key))
                raw = None if fair is None else (fair if fill.side == "up" else 1 - fair)
                self.store.add_bet(now, m["symbol"], 0, int(m["close"]), seconds_left, fill, mode="stocks",
                                   variant=v.key, price_age=now - last[0], book_age=now - min(yes[2], no[2]),
                                   fair_raw=raw, market=m["slug"], stake=cfg.stake_usd)
                log.info("STOCK LAB %s %s %s: paid %.1fc, fair %s, %ds left", v.key, m["slug"], fill.side,
                         fill.avg_price * 100, "-" if fair is None else f"{fair * 100:.1f}%", seconds_left)
            rule = self.main_rule
            if rule and self.enabled and m["slug"] not in self.main_done and seconds_left <= rule.max_seconds_left:
                fill = self._decide(rule, fair, yes[1], no[1],
                                    dataclasses.replace(cfg, stake_usd=self.sim_stake, min_ratio=self.min_ratio,
                                                        max_ratio=self.max_ratio))
                if fill and self.can_open and not self.can_open("stocks", fill.total_paid):
                    fill = None  # the owner's limits
                if fill:
                    self.main_done.add(m["slug"])
                    raw = None if fair is None else (fair if fill.side == "up" else 1 - fair)
                    self.store.add_bet(now, m["symbol"], 0, int(m["close"]), seconds_left, fill, mode="paper",
                                       variant="stocks:" + rule.key, price_age=now - last[0],
                                       book_age=now - min(yes[2], no[2]), fair_raw=raw, market=m["slug"],
                                       stake=self.sim_stake)
                    log.info("STOCK STRATEGY %s %s: paid %.1fc", m["slug"], fill.side, fill.avg_price * 100)
            # Real money: the same rule under its own settings, whether or not the simulation bet.
            if rule and self.live and m["slug"] not in self.live_done and seconds_left <= rule.max_seconds_left:
                fill = self._decide(rule, fair, yes[1], no[1], self.live.order_cfg(cfg, "stocks"))
                if fill and self.live.blocked("stocks", fill.side, int(m["close"])) is None:
                    self.live_done.add(m["slug"])
                    market = {"up_token": m["yes_token"], "down_token": m["no_token"],
                              "tick_size": m.get("tick_size", 0.01), "min_order_size": m.get("min_order_size", 5)}
                    self.dispatch(self.live.maybe_trade, now, m["symbol"], 0, int(m["close"]), seconds_left,
                                  fill, market, cfg, "stocks", m["slug"], "stocks:" + rule.key)

    @staticmethod
    def _decide(v, fair, yes_asks, no_asks, cfg):
        if v.kind == "model":
            if fair is None:
                return None
            return find_bet(fair, yes_asks, no_asks, dataclasses.replace(cfg, min_edge=v.level))
        # The favourite: whichever side's best ask is between level and 95c, bought up to 95c.
        for side, asks in (("up", yes_asks), ("down", no_asks)):
            best = min((p for p, _ in asks), default=None)
            if best is not None and v.level <= best <= 0.95:
                cap = dataclasses.replace(cfg, min_edge=0.05 - 1e-9)  # "fair" 1.0: buys up to 95c incl. fee
                fill = find_bet(1.0 if side == "up" else 0.0, yes_asks if side == "up" else [],
                                no_asks if side == "down" else [], cap)
                if fill:
                    fill.fair_prob = best  # no model: record the price it was judged at
                    fill.edge = 0.0
                return fill
        return None

    # Results
    def settle(self, now=None):
        now = now or time.time()
        rows = self.store._x("SELECT id, market, side, shares, cost, fee FROM bets WHERE minutes = 0"
                             " AND market IS NOT NULL AND mode IN ('stocks', 'paper', 'live')"
                             " AND status = 'filled' AND pnl IS NULL AND window_start < ?", (now - 60,))
        by_market = {}
        for row in rows:
            by_market.setdefault(row[1], []).append(row)
        for slug, bets in by_market.items():
            markets = self.get(f"{feeds.GAMMA}/markets?slug={slug}&closed=true")
            if not markets or not markets[0].get("closed"):
                continue
            prices = [float(p) for p in _maybe_json(markets[0].get("outcomePrices"))]
            if len(prices) != 2:
                continue
            if prices[0] >= 0.99:
                winner = "up"
            elif prices[1] >= 0.99:
                winner = "down"
            elif abs(prices[0] - 0.5) < 0.01 and abs(prices[1] - 0.5) < 0.01:
                winner = "half"  # a 50-50 resolution: each share pays 50c
            else:
                continue
            for bet_id, _, side, shares, cost, fee in bets:
                pnl = shares * 0.5 - cost - fee if winner == "half" else settle_pnl(side, shares, cost, fee, winner)
                self.store.settle_bet(bet_id, winner, pnl)

    CHART_SECONDS = 1800  # a stock card's chart: the last half hour before the close

    def card(self, mode, slug, symbol, close, now, book):
        """A position card for a stock market, in the same shape as the crypto ones: the line,
        the live price and its recent path, both sides' prices, and whether it is still open."""
        m = self.markets.get(slug) or {}
        kind = m.get("kind") or ("up-or-down" if "-up-or-down-" in slug else "close-above")
        name = NAMES.get(symbol, symbol.upper())
        start = close - self.CHART_SECONDS
        live = now < close
        last = self.last.get(symbol)
        vol = self.vols.get(symbol)
        path = [(ts, p) for ts, p in (list(vol.samples) if vol else []) if start - 5 <= ts <= min(now, close)]
        step = max(1, len(path) // 120)
        line = self.line(m) if m else None
        price = last[1] if last else None
        if not live:
            recorded = [p for t, p in self.closes.get(symbol, []) if abs(t - close) < 60]
            price = recorded[0] if recorded else (path[-1][1] if path else price)
        quotes = {}
        for side, token in (("up", m.get("yes_token")), ("down", m.get("no_token"))):
            b = book(token) if token else None
            quotes[f"{side}_bid"] = max((p for p, _ in b[0]), default=None) if b else None
            quotes[f"{side}_ask"] = min((p for p, _ in b[1]), default=None) if b else None
        sigma = self.volatility(symbol)
        fair = (fair_up_probability(price, line, close - now, sigma)
                if live and price and line and sigma else None)
        updown = kind == "up-or-down"
        return dict(quotes, mode=mode, asset=symbol, minutes=0, window_start=close, chart_start=start,
                    span=self.CHART_SECONDS, end=close, live=live, kind=kind, market=slug,
                    title=f"{name}: עולה או יורדת היום?" if updown else f"{name} נסגרת מעל {m.get('title') or ''}?",
                    sub="מוכרע במחיר הסגירה, 16:00 בניו יורק",
                    labels={"up": "עלייה", "down": "ירידה"} if updown else {"up": "כן", "down": "לא"},
                    line_label="סגירה של אתמול" if updown else "הקו", open_price=line, price=price,
                    seconds_left=max(0.0, close - now), fair_up=fair,
                    series=[(round(ts - start, 1), p) for ts, p in path[::step]], bets=[])

    def board(self, db):
        rows = db.execute("SELECT variant, ts, pnl, cost + fee AS paid FROM bets WHERE mode = 'stocks'"
                          " AND status = 'filled' AND pnl IS NOT NULL AND ts >= ? ORDER BY ts", (self.count_since,)).fetchall()
        by = {}
        for r in rows:
            by.setdefault(r["variant"], []).append((r["pnl"], r["paid"]))
        out = []
        for v in VARIANTS:
            bets = by.get(v.key, [])
            roi, low = lab.money_weighted(bets, self.cfg.lab_z)
            pnl = sum(x for x, _ in bets)
            if len(bets) < self.cfg.lab_min_bets:
                status = "collecting"
            elif low is not None and low > 0:
                status = "promising"
            elif roi is not None and roi < 0:
                status = "losing"
            else:
                status = "mixed"
            out.append({"variant": v.key, "text": describe(v.key), "bets": len(bets),
                        "wins": sum(1 for x, _ in bets if x > 0), "pnl": pnl, "roi": roi, "roi_pessimistic": low,
                        "status": status})
        order = {"promising": 0, "mixed": 1, "collecting": 2, "losing": 3}
        out.sort(key=lambda b: (order[b["status"]], -(b["roi_pessimistic"] if b["roi_pessimistic"] is not None
                                                      else (b["roi"] or -9))))
        return out

    def state(self, db, now):
        open_rows = db.execute(
            "SELECT ts, asset, window_start, market, side, variant, fair_prob, avg_price, shares, cost + fee AS paid"
            " FROM bets WHERE mode = 'stocks' AND status = 'filled' AND pnl IS NULL ORDER BY ts DESC LIMIT 60"
        ).fetchall()
        open_bets = []
        for r in open_rows:
            m = self.markets.get(r["market"]) or {}
            last = self.last.get(r["asset"])
            line = self.line(m) if m else None
            open_bets.append({"ts": r["ts"], "symbol": r["asset"], "close": r["window_start"], "market": r["market"],
                              "title": m.get("title") or m.get("question") or r["market"], "kind": m.get("kind"),
                              "line": line, "price": last[1] if last else None, "side": r["side"],
                              "variant": r["variant"], "avg_price": r["avg_price"], "paid": r["paid"],
                              "if_win": r["shares"] - r["paid"]})
        recent = [dict(r) for r in db.execute(
            "SELECT ts, asset AS symbol, market, side, variant, avg_price, cost + fee AS paid, pnl, winner FROM bets"
            " WHERE mode = 'stocks' AND pnl IS NOT NULL ORDER BY ts DESC LIMIT 20").fetchall()]
        totals = db.execute("SELECT COUNT(*), COALESCE(SUM(pnl), 0), COALESCE(SUM(cost + fee), 0),"
                            " SUM(pnl > 0) FROM bets WHERE mode = 'stocks' AND pnl IS NOT NULL AND ts >= ?",
                            (self.count_since,)).fetchone()
        closes = [m["close"] for m in self.markets.values() if m["close"] > now]
        symbols = {}
        for symbol, (ts, price, arrived) in list(self.last.items()):
            symbols[symbol] = {"price": price, "age": now - arrived, "name": NAMES.get(symbol, symbol.upper()),
                               "vol_ready": self.volatility(symbol) is not None}
        return {
            "markets": len(self.markets),
            "watching": len(self.watching(now)),
            "next_close": min(closes) if closes else None,
            "watch_hours": WATCH_SECONDS / 3600,
            "symbols": symbols,
            "stream": {"connected": bool(self.stream and self.stream.connected),
                       "error": self.stream.last_error if self.stream else None},
            "error": self.last_error,
            "missing_lines": self.missing_lines,
            "board": self.board(db),
            "open": open_bets,
            "recent": recent,
            "settled": totals[0], "pnl": totals[1], "paid": totals[2], "wins": totals[3] or 0,
            "min_bets": self.cfg.lab_min_bets,
        }
