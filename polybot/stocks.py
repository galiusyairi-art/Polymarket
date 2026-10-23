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
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from . import feeds, lab
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
    for field in ("groupItemThreshold", "groupItemTitle"):
        raw = market.get(field)
        if raw in (None, ""):
            continue
        try:
            return float(str(raw).replace("$", "").replace(",", "").strip())
        except ValueError:
            continue
    return None


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
    """Pyth's official close for a session: the close of its final 1-minute candle."""
    data = get(f"{PYTH_HISTORY}?symbol=Equity.US.{symbol.upper()}/USD&resolution=1"
               f"&from={int(close_ts) - 600}&to={int(close_ts)}", timeout=10)
    if not data or data.get("s") != "ok":
        return None
    candles = [(t, c) for t, c in zip(data.get("t", []), data.get("c", [])) if t < close_ts]
    return float(candles[-1][1]) if candles else None


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
        self.markets = {}  # slug -> parse_market dict
        self.vols = {}  # symbol -> VolatilityEstimator
        self.last = {}  # symbol -> (source ts, price, arrived)
        self.discovered_at = None
        self.last_error = None
        self.stream = None
        self.closes = self._load_closes()
        self._close_tried = {}
        self.done = {(m, v) for m, v in store._x("SELECT market, variant FROM bets WHERE mode = 'stocks'")}

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
        4pm; if the system wasn't running then, it is fetched from Pyth's own history."""
        for m in self.markets.values():
            if m["kind"] != "up-or-down" or self.prior_close(m["symbol"], m["close"]) is not None:
                continue
            key = (m["symbol"], m["close"])
            if now - self._close_tried.get(key, 0) < 600:
                continue
            self._close_tried[key] = now
            for session in previous_sessions(m["close"]):
                if session > now:
                    continue
                try:
                    price = pyth_close(m["symbol"], session, self.get)
                except Exception as e:
                    log.warning("Pyth close for %s: %s", m["symbol"], e)
                    break
                if price:
                    self._record_close(m["symbol"], session, price)
                    break

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
                                   fair_raw=raw, market=m["slug"])
                log.info("STOCK LAB %s %s %s: paid %.1fc, fair %s, %ds left", v.key, m["slug"], fill.side,
                         fill.avg_price * 100, "-" if fair is None else f"{fair * 100:.1f}%", seconds_left)

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
        rows = self.store._x("SELECT id, market, side, shares, cost, fee FROM bets WHERE mode = 'stocks'"
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

    def board(self, db):
        rows = db.execute("SELECT variant, ts, pnl, cost + fee AS paid FROM bets WHERE mode = 'stocks'"
                          " AND status = 'filled' AND pnl IS NOT NULL ORDER BY ts").fetchall()
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
                            " SUM(pnl > 0) FROM bets WHERE mode = 'stocks' AND pnl IS NOT NULL").fetchone()
        closes = [m["close"] for m in self.markets.values() if m["close"] > now]
        symbols = {}
        for symbol, (ts, price, arrived) in self.last.items():
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
            "board": self.board(db),
            "open": open_bets,
            "recent": recent,
            "settled": totals[0], "pnl": totals[1], "paid": totals[2], "wins": totals[3] or 0,
            "min_bets": self.cfg.lab_min_bets,
        }
