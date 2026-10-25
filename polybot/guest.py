"""A guest venue, read only: Limitless (limitless.exchange), to see whether it has the edge
Polymarket doesn't. No account, no wallet, no orders: only its public data.

What it records, in tables of its own (lx_*), so that the market check and the maker check
(marketcheck.py) run on it exactly as on Polymarket:
- its short crypto rounds ("BTC Up or Down - 1 hour", and shorter ones as they come): the round,
  and once decided, which side won (lx_windows);
- their order books every few seconds (lx_ticks): the best price to buy and to sell each side;
- every trade in them (lx_trades): price, size, and who took (bought or sold).

Its API (api.limitless.exchange, the paths CCXT's Limitless integration uses): /markets/search
lists markets, /markets/{slug}/orderbook gives the book (quoted in the YES token, sizes in
millionths), /markets/{slug}/events the trades, /markets/{slug} the result (winningOutcomeIndex:
0 YES, 1 NO). YES is read as Up, NO as Down.
"""

import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime

log = logging.getLogger("polybot")

API = "https://api.limitless.exchange"
TITLE = re.compile(r"^\W*(\w+)\s+Up or Down\s*-\s*(\d+)\s*(min|minute|minutes|hour|hours)\b", re.I)
MAX_MINUTES = 60  # rounds of an hour or less (the daily and weekly ones are another game)
SCALE = 1e6  # sizes come in millionths


def parse_round(market):
    """(asset, minutes, window_start, end) of an "X Up or Down - N min/hour" market, or None."""
    m = TITLE.match(str(market.get("title") or ""))
    if not m:
        return None
    n, unit = int(m.group(2)), m.group(3).lower()
    minutes = n * 60 if unit.startswith("hour") else n
    try:
        end = int(market.get("expirationTimestamp")) / 1000
    except (TypeError, ValueError):
        return None
    if not 0 < minutes <= MAX_MINUTES:
        return None
    return m.group(1).lower(), minutes, int(round(end - minutes * 60)), end


def _iso(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class LimitlessWatch:
    BOOK_SECONDS = 3  # each round's book, this often
    TRADES_SECONDS = 15  # each round's trades
    LIST_SECONDS = 60  # the list of rounds
    RESOLVE_SECONDS = 60  # results of the rounds that ended

    def __init__(self, store, fetch=None):
        self.store = store
        self.fetch = fetch or self._fetch
        self.rounds = {}  # slug -> {"asset", "minutes", "window_start", "end", "yes", "no"}
        self.seen_trades = set()
        self.status = {"connected": False, "error": None, "last_ok": None, "requests": 0, "failures": 0,
                       "rounds": 0, "books": 0, "trades": 0, "resolved": 0, "markets": []}
        self._last = {"list": 0.0, "resolve": 0.0}
        self._trades_at = {}
        self._books_at = {}

    @staticmethod
    def _fetch(path, params=None):
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, headers={"User-Agent": "polybot (read-only research)", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def _get(self, path, params=None):
        self.status["requests"] += 1
        try:
            out = self.fetch(path, params)
        except Exception as e:
            self.status["failures"] += 1
            self.status["error"] = f"{time.strftime('%H:%M:%S')} {path}: {type(e).__name__}: {e}"
            self.status["connected"] = False
            raise
        self.status["connected"], self.status["last_ok"] = True, time.time()
        return out

    # ---------- the rounds ----------
    def refresh_rounds(self, now):
        found = self._get("/markets/search", {"query": "Up or Down", "limit": 50})
        rows = found.get("markets") if isinstance(found, dict) else found
        for market in rows or []:
            for m in (market.get("markets") or [market]):  # (a group row carries its markets inside)
                slug, parsed, tokens = m.get("slug"), parse_round(m), m.get("tokens") or {}
                if not slug or not parsed or m.get("expired") or parsed[3] <= now:
                    continue
                asset, minutes, start, end = parsed
                if slug not in self.rounds:
                    self.rounds[slug] = {"asset": asset, "minutes": minutes, "window_start": start, "end": end,
                                         "yes": str(tokens.get("yes") or ""), "no": str(tokens.get("no") or "")}
                    self.store._x("INSERT OR IGNORE INTO lx_windows (slug, asset, minutes, window_start, title)"
                                  " VALUES (?, ?, ?, ?, ?)", (slug, asset, minutes, start, m.get("title")))
        self.status["rounds"] = sum(1 for r in self.rounds.values() if r["end"] > now)

    # ---------- the books ----------
    def poll_book(self, slug, r, now):
        book = self._get(f"/markets/{slug}/orderbook")
        bids = [(float(x["price"]), float(x["size"]) / SCALE) for x in book.get("bids") or []]
        asks = [(float(x["price"]), float(x["size"]) / SCALE) for x in book.get("asks") or []]
        bid, ask = max(bids, default=(None, None)), min(asks, default=(None, None))
        row = (now, r["asset"], r["minutes"], r["window_start"], r["end"] - now, bid[0], ask[0],
               None if ask[0] is None else round(1 - ask[0], 6), None if bid[0] is None else round(1 - bid[0], 6),
               bid[1], ask[1], _num(book.get("lastTradePrice")), slug)
        self.store._x("INSERT INTO lx_ticks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
        self.status["books"] += 1
        r["book"] = {"up_bid": bid[0], "up_ask": ask[0], "bid_size": bid[1], "ask_size": ask[1], "ts": now}

    # ---------- the trades ----------
    def poll_trades(self, slug, r, now):
        found = self._get(f"/markets/{slug}/events", {"limit": 100})
        new = []
        for e in found.get("events") or []:
            key = (e.get("txHash"), e.get("tokenId"), e.get("matchedSize"), e.get("price"), e.get("createdAt"))
            ts = _iso(e.get("createdAt"))
            if key in self.seen_trades or ts is None or e.get("price") is None or e.get("matchedSize") is None:
                continue
            self.seen_trades.add(key)
            outcome = "up" if str(e.get("tokenId")) == r["yes"] else "down" if str(e.get("tokenId")) == r["no"] else None
            taker = {0: "BUY", 1: "SELL", "0": "BUY", "1": "SELL"}.get(e.get("side"))
            if outcome and taker:
                new.append((ts, r["asset"], r["minutes"], r["window_start"], outcome, float(e["price"]),
                            float(e["matchedSize"]) / SCALE, taker, r["end"] - ts))
        if new:
            self.store.add_trades(new, table="lx_trades")
            self.status["trades"] += len(new)
        if len(self.seen_trades) > 200_000:
            self.seen_trades.clear()

    # ---------- the results ----------
    def resolve(self, now):
        pending = self.store._x("SELECT slug FROM lx_windows WHERE winner IS NULL AND window_start + minutes * 60 < ?"
                                " AND window_start + minutes * 60 > ? ORDER BY window_start LIMIT 30", (now - 30, now - 2 * 86400))
        for (slug,) in pending:
            try:
                m = self._get(f"/markets/{slug}")
            except Exception:
                continue
            won = m.get("winningOutcomeIndex")
            if won in (0, 1, "0", "1"):
                self.store._x("UPDATE lx_windows SET winner = ? WHERE slug = ?", ("up" if int(won) == 0 else "down", slug))
                self.status["resolved"] += 1
        for slug in [s for s, r in self.rounds.items() if r["end"] < now - 600]:
            self.rounds.pop(slug, None)

    # ---------- the loop ----------
    def step(self, now):
        if now - self._last["list"] >= self.LIST_SECONDS:
            self._last["list"] = now
            self.refresh_rounds(now)
        for slug, r in list(self.rounds.items()):
            if r["window_start"] <= now < r["end"]:  # running now
                if now - self._books_at.get(slug, 0) >= self.BOOK_SECONDS:
                    self._books_at[slug] = now
                    self.poll_book(slug, r, now)
                if now - self._trades_at.get(slug, 0) >= self.TRADES_SECONDS:
                    self._trades_at[slug] = now
                    self.poll_trades(slug, r, now)
        if now - self._last["resolve"] >= self.RESOLVE_SECONDS:
            self._last["resolve"] = now
            self.resolve(now)
        self.status["markets"] = sorted(
            ({"slug": s, "asset": r["asset"], "minutes": r["minutes"], "end": r["end"], **(r.get("book") or {})}
             for s, r in self.rounds.items() if r["window_start"] <= now < r["end"]),
            key=lambda x: (x["end"], x["asset"]))

    def run(self, stop):
        """Its own thread: a fault here never touches the trading."""
        while not stop.is_set():
            try:
                self.step(time.time())
            except Exception as e:
                if str(e) != getattr(self, "_logged", None):  # (the same failure again: logged once)
                    self._logged = str(e)
                    log.warning("Limitless (read only): %s", e)
                stop.wait(10)  # (backs off after a failure)
            stop.wait(1)

    def start(self, stop):
        threading.Thread(target=self.run, args=(stop,), daemon=True, name="limitless").start()


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
