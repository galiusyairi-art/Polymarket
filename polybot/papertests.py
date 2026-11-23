"""The research zone's paper tests (the owner, 17.11.2026: "build both, A and B"; then "both: tabs, a progress bar, a
status, and a button that makes the strategy").

The research round "many small profitable trades" left two ideas worth a forward test, and both are recorded here
for four weeks on live public data, with no order ever sent:

A, the maker ("עושה שוק"): a $1,000 two-sided quote, half the reward band from the middle, on every slow
  Polymarket market that pays a liquidity reward, chosen by fixed rules (MM_* below, the research's §4). Every
  minute the books are read; a real trade print through our price fills us (queue ahead of us counted), and each
  fill is marked 1, 10 and 60 minutes later and at the market's resolution. The rewards are Polymarket's scoring
  rule on the books we saw, and the maker rebate on fills. The research's own two-hour run: about zero, with
  fills that took days of rewards each (research_notes/הרבה עסקאות קטנות ומרוויחות/polymarket_anchored_market_making.md).

B, the ladder ("ביטקוין מעל X"): Polymarket's daily "Bitcoin/Ethereum above $X" strikes against the probability
  Deribit's option smile gives the same event. When the price one could really buy at (the offer, or the bid's
  other side for No) is LADDER_GAP or more away from Deribit's, a paper position is taken on Deribit's side, once per
  strike, held to settlement, the taker fee paid. The research's backtest on hourly prices: +6.8c a contract; a
  live check the same day found such gaps rare. This decides which.

A runs four ways of quoting side by side on the same books and prints (MM_VARIANTS): halfway out in the band (the
research's), at its far edge, stepping aside for ten minutes after any trade in the market, and both. C (1.356)
scans every set on Polymarket for a sure profit (arb_candidate, walk_set).

Each has its own go rules, judged after DAYS days from its own start: "investigating" until then, then "found" or
"not found". A research found becomes, by the owner's click, a winner ("המנצחת"): the same rule, frozen, in the
simulation with its own record from then; and with real money when he switches it (the trader's real orders). Its own database file, so
nothing here writes to the trading data; its own threads, started by the engine in production only.
"""

import asyncio
import contextlib
import gc
import datetime as dt
import json
import logging
import math
import os
import random
import re
import sqlite3
import threading
import time
import urllib.request

log = logging.getLogger("polybot.papertests")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DERIBIT = "https://www.deribit.com/api/v2/public"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

DAYS = 28  # the test's length: four weeks
WEEK = 7 * 86400

# --- A: the maker ---
MM_EVERY = 60  # seconds between book reads (the scoring samples about once a minute)
MM_MIN_RATE = 20.0  # a market's reward pool, $ a day, at least
MM_MIN_DAYS = 7.0  # its end date at least this far out when it joins
MM_PULL = 86400  # no quote in a market's last day before its end date
MM_MOVE = 0.01  # no quote in a minute after the middle moved this much since the last read
MM_MAX = 150  # markets followed at most (the largest pools first)
MM_CAPITAL = 1000.0  # the quote's money in each market: balanced Yes and No bids
MM_EXPOSURE = 250.0  # one side's net holding in a market, $, beyond which only the side that reduces it is quoted
MM_UNIVERSE_EVERY = 86400
MM_GAP = 180  # a read later than this after the one before: the minutes between aren't counted
MARKS = (("m1", 60), ("m10", 600), ("m60", 3600))
# the ways of quoting, side by side: (key, how far out in the band, seconds stepped aside after a trade, name)
MM_VARIANTS = (("mid", 0.5, 0, "באמצע הטווח"), ("far", 0.9, 0, "בקצה הרחוק"),
               ("step", 0.5, 600, "זז הצידה 10 דק' אחרי עסקה"), ("farstep", 0.9, 600, "קצה רחוק וזז הצידה"))
LIVE_MM_MARKETS = 10  # a live maker winner quotes the largest pools only
# the research's go rules (§5)
GO_FILLS = 300
GO_RESOLVED = 50
GO_MONTH = 1.5  # % a month of the money quoted, at least
GO_WEEKS = 3  # positive weeks of the four
GO_WORST_WEEK = -2.0  # %
GO_FILL_DAYS = 3.0  # no single fill losing more than this many days of all the rewards
GO_CONTESTED = 0.70  # of the rewards, from minutes others quoted in too

# --- B: the ladder ---
LADDER_EVERY = 120
LADDER_GAP = 0.05
LADDER_PRICED = (0.03, 0.97)  # strikes priced between (the backtest's)
LADDER_COINS = (("BTC", "bitcoin"), ("ETH", "ethereum"))
LADDER_AHEAD = 4  # days of ladders read
LADDER_FEE = 0.07  # crypto's taker fee rate, when the market doesn't say
LADDER_GO_CENTS = 3.0
LADDER_GO_T = 2.0
LADDER_NOGO_CENTS = 1.0
LADDER_NOGO_PER_DAY = 1.0
LADDER_STAKE = 100.0  # $ a signal, for the dollars shown

# --- C: arbitrage ---
# (seconds between scans of every event. Ten minutes until 23.11.2026: the heart of the system showed the scan, a minute of
# work on the one processor, beside every long wait of the trading loop, and at 12:32 UTC the books ran up to 9 seconds late
# for the 51 seconds of one scan while a real order was refused. Once an hour until the scan has a process of its own)
ARB_EVERY = 3600
ARB_PAGES = 300  # Gamma's pages of events (it gives 100 a page), at most
ARB_MIN_EDGE = 0.005  # a set's profit after the fees, $ a set, at least
ARB_SAME = 1800  # the same set seen again within this: the same opportunity
ARB_GO_COUNT = 20  # opportunities one could really buy, in four weeks
ARB_GO_MONTH = 50.0  # $ a month of sure profit, at least
ARB_GO_APR = 15.0  # % a year on the money each needed, at least (weighted)

LENGTH = {}  # a research with its own number of days (the others: DAYS)

UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Accept": "application/json", "Accept-Encoding": "gzip"}


try:  # (a page of events is 3 MB of JSON: read by the fast reader, a third of the time with every thread waiting)
    from orjson import loads as _loads
except ImportError:  # pragma: no cover
    _loads = json.loads


@contextlib.contextmanager
def _collector_resting():
    """The garbage collector off for a moment (and back as it was): for work that makes very many objects and drops
    them at once, where a collection in the middle only walks what is about to go."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def http(url, body=None, timeout=30, gz=True):
    import gzip
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
        return _loads(gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b)


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS mm_markets (cid TEXT PRIMARY KEY, q TEXT, cat TEXT, yes TEXT, no TEXT, rate REAL, v REAL,
    min_size REAL, fee REAL, rebate REAL, end_ts REAL, added REAL, mid REAL, outcome REAL);
CREATE TABLE IF NOT EXISTS mm_rewards (cid TEXT, day TEXT, minutes REAL, reward REAL, contested REAL, PRIMARY KEY (cid, day));
CREATE TABLE IF NOT EXISTS mm_fills (id INTEGER PRIMARY KEY, ts REAL, cid TEXT, side TEXT, px REAL, sh REAL, mid0 REAL,
    m1 REAL, m10 REAL, m60 REAL, rebate REAL);
CREATE TABLE IF NOT EXISTS ld_signals (id INTEGER PRIMARY KEY, ts REAL, cur TEXT, cid TEXT UNIQUE, expiry REAL, strike REAL,
    side TEXT, price REAL, size REAL, model REAL, gap REAL, hours REAL, fee REAL, outcome REAL);
CREATE TABLE IF NOT EXISTS ld_days (day TEXT PRIMARY KEY, scans INTEGER, strikes INTEGER, max_gap REAL);
CREATE TABLE IF NOT EXISTS mm_rew (cid TEXT, var TEXT, day TEXT, minutes REAL, reward REAL, contested REAL, PRIMARY KEY (cid, var, day));
CREATE TABLE IF NOT EXISTS ar_opps (id INTEGER PRIMARY KEY, first REAL, last REAL, key TEXT, slug TEXT, title TEXT, kind TEXT,
    legs INTEGER, edge REAL, sets REAL, cost REAL, profit REAL, end_ts REAL, days REAL, apr REAL, ok INTEGER, least REAL, seen INTEGER);
CREATE INDEX IF NOT EXISTS ar_key ON ar_opps (key, last);
CREATE TABLE IF NOT EXISTS live_trades (id INTEGER PRIMARY KEY, ts REAL, test TEXT, key TEXT, detail TEXT, cost REAL, status TEXT,
    error TEXT);
"""


def qmin(q1, q2, mid):
    """Polymarket's two-sided score: one side alone counts a third, between 10c and 90c; outside, both are needed."""
    if 0.10 <= mid <= 0.90:
        return max(min(q1, q2), max(q1, q2) / 3.0)
    return min(q1, q2)


def book_state(bids, asks, min_size, v):
    """A Yes book's middle (by orders of the reward's least size, as the scoring takes it), its best prices, and the
    others' score inside the band: None when a side is empty."""
    if not bids or not asks:
        return None
    bb = max(p for p, _ in bids)
    ba = min(p for p, _ in asks)
    qb = [p for p, s in bids if s >= min_size]
    qa = [p for p, s in asks if s >= min_size]
    mid = ((max(qb) if qb else bb) + (min(qa) if qa else ba)) / 2
    q1 = sum(((v - (mid - p)) / v) ** 2 * s for p, s in bids if 0 <= mid - p < v)
    q2 = sum(((v - (p - mid)) / v) ** 2 * s for p, s in asks if 0 <= p - mid < v)
    return {"mid": mid, "bb": bb, "ba": ba, "others": qmin(q1, q2, mid)}


def quote(mid, v, tick, others, net, bids, asks, frac=0.5):
    """Our quote for the next minute: half the band from the middle, rounded away from it to the tick; balanced
    shares for MM_CAPITAL; past MM_EXPOSURE one way, the side that adds to it is left out. Returns None (no quote)
    or {bid, ask, nb, na, share, qa_bid, qa_ask} (nb/na 0 for a side not quoted)."""
    half = max(tick, round(v * frac / tick) * tick)
    bid = round(math.floor((mid - half) / tick + 1e-9) * tick, 6)
    ask = round(math.ceil((mid + half) / tick - 1e-9) * tick, 6)
    if bid <= 0.001 or ask >= 0.999:
        return None
    n = MM_CAPITAL / (bid + 1 - ask)
    nb = na = n
    if net > 0 and net * mid >= MM_EXPOSURE:
        nb = 0.0  # (long Yes already: only the ask, which sells it back)
    elif net < 0 and -net * (1 - mid) >= MM_EXPOSURE:
        na = 0.0
    sb, sa = mid - bid, ask - mid
    m1 = ((v - sb) / v) ** 2 * nb if 0 <= sb < v else 0.0
    m2 = ((v - sa) / v) ** 2 * na if 0 <= sa < v else 0.0
    mine = qmin(m1, m2, mid)
    share = mine / (mine + others) if mine + others > 0 else 0.0
    return {"bid": bid, "ask": ask, "nb": nb, "na": na, "share": share,
            "qa_bid": sum(s for p, s in bids if abs(p - bid) < 1e-9), "qa_ask": sum(s for p, s in asks if abs(p - ask) < 1e-9)}


def fill(q, px, size, side):
    """A trade print (Yes price, size, the taker's side) against our quote q (changed in place): the shares it fills
    us, and on which side ("buy": our bid, Yes bought; "sell": our ask, No bought). A print through our price fills
    up to its size; one at our price only past the queue ahead of us."""
    if side == "SELL" and q["nb"] > 0 and px <= q["bid"] + 1e-9:
        at = px >= q["bid"] - 1e-9
        n = max(0.0, min(q["nb"], size - q["qa_bid"])) if at else min(q["nb"], size)
        if at:
            q["qa_bid"] = max(0.0, q["qa_bid"] - size)
        q["nb"] -= n
        return n, "buy"
    if side == "BUY" and q["na"] > 0 and px >= q["ask"] - 1e-9:
        at = px <= q["ask"] + 1e-9
        n = max(0.0, min(q["na"], size - q["qa_ask"])) if at else min(q["na"], size)
        if at:
            q["qa_ask"] = max(0.0, q["qa_ask"] - size)
        q["na"] -= n
        return n, "sell"
    return 0.0, None


def fill_pnl(side, px, sh, mark):
    """A fill's value against a later price of Yes (the middle, or the result 0/1)."""
    return (1 if side == "buy" else -1) * (mark - px) * sh


# --- B's prices ---

def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_call(F, K, T, s):
    if T <= 0 or s <= 0:
        return max(F - K, 0.0)
    d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return F * _N(d1) - K * _N(d1 - s * math.sqrt(T))


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] > xs[i - 1] else 0
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


MON = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)


def deribit_expiry(code):
    m = re.match(r"(\d+)([A-Z]{3})(\d\d)$", code)
    return dt.datetime(2000 + int(m.group(3)), MON[m.group(2)], int(m.group(1)), 8, 0, tzinfo=dt.timezone.utc).timestamp()


def smiles(summary):
    """Deribit's option book summary into a smile per expiry: the out-of-the-money options' mark volatility by
    log-moneyness (the research's 06_btc_eth_anchor_gap.py)."""
    by = {}
    for x in summary:
        try:
            _, e, K, cp = x["instrument_name"].split("-")
            K, F, iv = float(K), float(x["underlying_price"]), x.get("mark_iv")
        except (KeyError, ValueError, TypeError):
            continue
        if not iv or F <= 0:
            continue
        if (cp == "C" and K >= F) or (cp == "P" and K < F):
            by.setdefault(e, {"F": F, "pts": []})["pts"].append((math.log(K / F), iv / 100))
    out = []
    for e, v in by.items():
        p = sorted(v["pts"])
        if len(p) < 5:
            continue
        try:
            T = deribit_expiry(e)
        except (AttributeError, KeyError):
            continue
        out.append({"T": T, "F": v["F"], "k": [a for a, _ in p], "iv": [b for _, b in p]})
    return sorted(out, key=lambda s: s["T"])


def _total_var(sms, now, T, K):
    year = 365 * 86400
    tau = (T - now) / year
    after = [s for s in sms if s["T"] >= T]
    before = [s for s in sms if now < s["T"] < T]
    if not after:
        return None
    s2 = after[0]
    t2 = (s2["T"] - now) / year
    w2 = _interp(math.log(K / s2["F"]), s2["k"], s2["iv"]) ** 2 * t2
    if not before:
        return w2 / t2 * tau  # (flat volatility before Deribit's first expiry)
    s1 = before[-1]
    t1 = (s1["T"] - now) / year
    w1 = _interp(math.log(K / s1["F"]), s1["k"], s1["iv"]) ** 2 * t1
    return w1 + (w2 - w1) * (tau - t1) / (t2 - t1)


def digital(sms, now, T, K, spot):
    """Deribit's chance that the price is above K at T: a tight call spread on the smile (skew counted)."""
    tau = (T - now) / (365 * 86400)
    if tau <= 0:
        return None
    h = 0.005 * K
    c = []
    for KK in (K - h, K + h):
        w = _total_var(sms, now, T, KK)
        if not w or w <= 0:
            return None
        c.append(black_call(spot, KK, tau, math.sqrt(w / tau)))
    return min(1.0, max(0.0, (c[0] - c[1]) / (2 * h)))


def ladder_signal(model, bid, ask, gap=LADDER_GAP):
    """("yes", price) when the offer is gap or more below Deribit's chance; ("no", price of No) when the bid is gap
    or more above it; else None."""
    if model is None or bid is None or ask is None:
        return None
    if model - ask >= gap - 1e-9:
        return "yes", ask
    if bid - model >= gap - 1e-9:
        return "no", round(1 - bid, 6)
    return None


def ladder_pnl(side, price, fee, outcome):
    """A contract's result to settlement, the taker fee paid (cents of a dollar as a fraction)."""
    win = outcome if side == "yes" else 1 - outcome
    return win - price - fee


def t_stat(xs):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m / (sd / math.sqrt(n)) if sd > 0 else None


def strike_of(market):
    try:
        return float(re.sub(r"[^\d.]", "", (market.get("groupItemTitle") or "").replace(",", "")))
    except ValueError:
        return None


def ladder_slug(name, day):
    return f"{name}-above-on-{day.strftime('%B').lower()}-{day.day}-{day.year}"


# --- C: arbitrage inside Polymarket (1.356) ---

def walk_set(legs, payout, min_edge):
    """Buy one share of every leg at a time, level by level, while a set still pays min_edge or more over its cost and
    fees. legs: [[(price, size), ...] asks low to high, fee rate] per leg. Returns (sets, cost with fees, the dearest
    price taken per leg, the first set's edge)."""
    books = [[list(lv) for lv in sorted(levels)] for levels, _ in legs]
    rates = [r for _, r in legs]
    if any(not b for b in books):
        return 0.0, 0.0, [], None
    idx = [0] * len(books)
    sets = cost = 0.0
    first = None
    top = [0.0] * len(books)
    while all(i < len(b) for i, b in zip(idx, books)):
        per = sum(b[i][0] + r * b[i][0] * (1 - b[i][0]) for b, i, r in zip(books, idx, rates))
        edge = payout - per
        if first is None:
            first = edge
        if edge < min_edge - 1e-12:
            break
        chunk = min(b[i][1] for b, i in zip(books, idx))
        if chunk <= 0:
            break
        sets += chunk
        cost += chunk * per
        for k, (b, i) in enumerate(zip(books, idx)):
            top[k] = b[i][0]
            b[i][1] -= chunk
            if b[i][1] <= 1e-9:
                idx[k] += 1
    return sets, cost, top, first


def least_sets(prices, min_sizes):
    """The fewest sets Polymarket takes: each leg's order at least its minimum shares and a dollar."""
    return max(max(ms or 0, 1.0 / p if p > 0 else 0) for p, ms in zip(prices, min_sizes)) if prices else 0.0


MM_NAMES = {v[0]: v[3] for v in MM_VARIANTS}
KEYS = tuple(f"mm:{v[0]}" for v in MM_VARIANTS) + ("ladder", "arb")
TITLES = {"mm:mid": "עושה שוק · באמצע הטווח", "mm:far": "עושה שוק · בקצה הרחוק", "mm:step": "עושה שוק · זז הצידה אחרי עסקה",
          "mm:farstep": "עושה שוק · קצה רחוק וזז הצידה", "ladder": "ביטקוין מעל X מול Deribit", "arb": "ארביטראז' בפולימרקט"}
WINNER_NAMES = {"arb": "המנצחת · ארביטראז'", "ladder": "המנצחת · ביטקוין מול Deribit"}
STAKES = {"arb": 100.0, "ladder": 20.0, "mm": 100.0}  # the winner's money: a set, a signal, a market's quote


class PaperTests:
    def __init__(self, path, get=http):
        self.path = path
        self.get = get
        self.on_job = None  # the heart of the system: each scan with when it ran and how long (health.job)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.executescript(SCHEMA)
        for sql in ("ALTER TABLE ld_days ADD COLUMN max_exec REAL",  # (1.355)
                    "ALTER TABLE mm_fills ADD COLUMN var TEXT DEFAULT 'mid'",  # (1.356: four ways of quoting, side by side)
                    "ALTER TABLE ld_signals ADD COLUMN token TEXT"):
            try:
                self.db.execute(sql)
            except sqlite3.OperationalError:
                pass
        if self.meta("weather_gone") is None:  # (1.379: the owner took the weather research out; what it left goes with it)
            self.db.execute("DROP TABLE IF EXISTS wx_buckets")
            self.db.execute("DELETE FROM meta WHERE key = 'started_weather'")
            W = self.winners()
            if W.pop("weather", None) is not None:
                self._save_winners(W)
            self.set_meta("weather_gone", 1)
        if self.meta("basis_gone") is None:  # (1.381: and the crypto basis research, the same way)
            self.db.execute("DROP TABLE IF EXISTS bs_days")
            self.db.execute("DROP TABLE IF EXISTS bs_pos")
            self.db.execute("DELETE FROM meta WHERE key = 'started_basis'")
            W = self.winners()
            if W.pop("basis", None) is not None:
                self._save_winners(W)
            self.set_meta("basis_gone", 1)
        if self.meta("rew_moved") is None:  # (the first test's minutes, as the "mid" way's)
            self.db.execute("INSERT OR IGNORE INTO mm_rew SELECT cid, 'mid', day, minutes, reward, contested FROM mm_rewards")
            self.set_meta("rew_moved", 1)
        self.db.commit()
        self.lock = threading.RLock()
        self.mm = {}  # cid -> {"m": the market, "mid", "t": last read, "others", "last_print", "v": {var: {"quote", "net"}}}
        self.pending = []  # fills still waiting for a mark
        self.status = {"mm": {}, "ladder": {}, "arb": {}}
        self.ws_assets = []
        self.ws_changed = threading.Event()
        self.trader = None  # real money (the engine's LiveTrader), for a winner switched to it
        self.live_orders = {}  # (cid, "yes"/"no") -> {"id", "price", "shares", "matched"}: a live maker winner's resting orders
        self._load()

    # --- storage ---
    def meta(self, key, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))
        self.db.commit()

    START_KEYS = {"mm:mid": "started", "ladder": "started"}

    def started(self, key="mm:mid", now=None):
        """When a research began (each its own four weeks): the first two on the day they were built, the others when
        they joined (17.11.2026)."""
        k = self.START_KEYS.get(key, "started_" + key)
        s = self.meta(k)
        if s is None:
            s = now or time.time()
            self.set_meta(k, s)
        return s

    def _load(self):
        with self.lock:
            now = time.time()
            for r in self.db.execute("SELECT cid, q, cat, yes, no, rate, v, min_size, fee, rebate, end_ts, mid FROM mm_markets "
                                     "WHERE outcome IS NULL").fetchall():
                self._track(dict(zip(("cid", "q", "cat", "yes", "no", "rate", "v", "min_size", "fee", "rebate", "end_ts"), r[:11])), r[11])
            for cid, side, sh, var in self.db.execute("SELECT cid, side, sh, COALESCE(var, 'mid') FROM mm_fills"):
                if cid in self.mm and var in self.mm[cid]["v"]:
                    self.mm[cid]["v"][var]["net"] += sh if side == "buy" else -sh
            for r in self.db.execute("SELECT id, cid, ts, side, px, sh, m1, m10, m60 FROM mm_fills WHERE m60 IS NULL AND ts > ?",
                                     (now - 3600 - 300,)).fetchall():
                self.pending.append({"id": r[0], "cid": r[1], "t": r[2], "side": r[3], "px": r[4], "sh": r[5],
                                     "m1": r[6], "m10": r[7], "m60": r[8]})
            self.ws_assets = self._assets()

    def _track(self, m, mid=None):
        self.mm[m["cid"]] = {"m": m, "mid": mid, "t": None, "others": 0.0, "last_print": None,
                             "v": {v[0]: {"quote": None, "net": 0.0} for v in MM_VARIANTS}}

    def _assets(self):
        return sorted({a for s in self.mm.values() for a in (s["m"]["yes"], s["m"]["no"])})

    # --- the winners: a research found, made a strategy (the owner, 17.11.2026) ---
    def winners(self):
        return self.meta("winners", {}) or {}

    def _save_winners(self, W):
        self.set_meta("winners", W)

    def winner_live(self, key):
        w = self.winners().get(key)
        return bool(w and w.get("live"))

    def create_winner(self, key, now=None):
        """The button "create the strategy": the research's rule exactly, frozen, into the simulation from now (its own
        record from zero). Only once the research found it."""
        now = now or time.time()
        if key not in KEYS:
            return {"error": "אין מחקר כזה"}
        if self.research_status(key, now)["status"] != "found":
            return {"error": "המחקר עוד לא מצא: האסטרטגיה נוצרת רק ממה שנמצא מרוויח"}
        W = self.winners()
        if key not in W:
            W[key] = {"created": now, "live": False, "stake": STAKES["mm" if key.startswith("mm:") else key], "name": self.winner_name(key)}
            self._save_winners(W)
        return {"ok": True, "winner": W[key]}

    @staticmethod
    def winner_name(key):
        return WINNER_NAMES.get(key) or ("המנצחת · " + TITLES.get(key, key))

    def winner_set(self, key, live=None, stake=None, remove=False, now=None):
        now = now or time.time()
        W = self.winners()
        if key not in W:
            return {"error": "האסטרטגיה לא קיימת"}
        if remove:
            if W[key].get("live"):
                return {"error": "קודם להעביר אותה מכסף אמיתי לסימולציה"}
            W.pop(key)
        if stake is not None:
            W[key]["stake"] = max(1.0, min(5000.0, float(stake)))
        if live is not None:
            if live and not (self.trader and self.trader.configured()):
                return {"error": "הארנק של פולימרקט לא מחובר: אין לאן לשלוח הוראות"}
            W[key]["live"] = bool(live)
            W[key]["live_since" if live else "live_until"] = now
        self._save_winners(W)
        if live is False and key.startswith("mm:"):
            self._live_cancel_all()
        return {"ok": True, "winners": W}

    # --- A: the universe ---
    def mm_universe(self, now=None):
        """The fixed rules' markets: a reward pool of MM_MIN_RATE a day or more, ending MM_MIN_DAYS or more ahead,
        taking orders; the largest pools first, up to MM_MAX followed (those already followed stay to their end)."""
        now = now or time.time()
        rewards, cur = [], ""
        for _ in range(100):
            d = self.get(f"{CLOB}/rewards/markets/current" + (f"?next_cursor={cur}" if cur else ""))
            rewards += d.get("data") or []
            cur = d.get("next_cursor")
            if not cur or cur == "LTE=" or not d.get("data"):
                break
        rate = {x["condition_id"]: x for x in rewards}
        with self.lock:
            for cid, s in self.mm.items():  # (pools change: the followed markets' terms today, 0 when gone)
                r = rate.get(cid)
                s["m"].update(rate=float(r["total_daily_rate"]) if r else 0.0,
                              **({"v": r["rewards_max_spread"] / 100, "min_size": float(r["rewards_min_size"])} if r else {}))
        want = sorted((x for x in rewards if float(x.get("total_daily_rate") or 0) >= MM_MIN_RATE and x["condition_id"] not in self.mm),
                      key=lambda x: -float(x["total_daily_rate"]))
        room = MM_MAX - len(self.mm)
        added = []
        for i in range(0, len(want), 40):
            if len(added) >= room:
                break
            chunk = want[i:i + 40]
            got = self.get(f"{GAMMA}/markets?limit=100&" + "&".join("condition_ids=" + x["condition_id"] for x in chunk)) or []
            by = {m.get("conditionId"): m for m in got}
            for x in chunk:
                g = by.get(x["condition_id"])
                if not g or g.get("closed") or not g.get("acceptingOrders") or not g.get("endDate"):
                    continue
                try:
                    end = dt.datetime.fromisoformat(g["endDate"].replace("Z", "+00:00")).timestamp()
                    yes, no = json.loads(g["clobTokenIds"])
                except (ValueError, TypeError, KeyError):
                    continue
                if end < now + MM_MIN_DAYS * 86400 or x["rewards_max_spread"] <= 0:
                    continue
                fs = g.get("feeSchedule") or {}
                m = {"cid": x["condition_id"], "q": g.get("question") or "", "cat": (g.get("feeType") or "none").replace("_fees", "").replace("_v2", "").replace("_v3", ""),
                     "yes": yes, "no": no, "rate": float(x["total_daily_rate"]), "v": x["rewards_max_spread"] / 100,
                     "min_size": float(x["rewards_min_size"]), "fee": float(fs.get("rate") or 0), "rebate": float(fs.get("rebateRate") or 0),
                     "end_ts": end}
                added.append(m)
                if len(added) >= room:
                    break
        with self.lock:
            for m in added:
                self.db.execute("INSERT OR IGNORE INTO mm_markets (cid,q,cat,yes,no,rate,v,min_size,fee,rebate,end_ts,added) "
                                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", tuple(m[k] for k in ("cid", "q", "cat", "yes", "no", "rate", "v",
                                                                                       "min_size", "fee", "rebate", "end_ts")) + (now,))
                self._track(m)
            for cid, s in self.mm.items():
                self.db.execute("UPDATE mm_markets SET rate=?, v=?, min_size=? WHERE cid=?", (s["m"]["rate"], s["m"]["v"], s["m"]["min_size"], cid))
            self.db.commit()
            self.ws_assets = self._assets()
        self.ws_changed.set()
        self.set_meta("mm_universe", now)
        return len(added)

    # --- A: a minute ---
    def mm_read(self, books, now=None):
        """One minute's books (Yes token -> {bids, asks, tick_size} as Polymarket gives them): the minute before is
        paid its reward share (each way of quoting its own), the fills' marks that came due are taken, and the next
        minute's quotes are set."""
        now = now or time.time()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with self.lock:
            by_yes = {s["m"]["yes"]: s for s in self.mm.values()}
            for tok, b in books.items():
                s = by_yes.get(tok)
                if not s:
                    continue
                m = s["m"]
                try:
                    bids = [(float(x["price"]), float(x["size"])) for x in b.get("bids") or []]
                    asks = [(float(x["price"]), float(x["size"])) for x in b.get("asks") or []]
                    tick = float(b.get("tick_size") or 0.01)
                except (TypeError, ValueError):
                    continue
                if s["t"] is not None and now - s["t"] <= MM_GAP:
                    mins = (now - s["t"]) / 60
                    for var, V in s["v"].items():
                        if V["quote"] is None:
                            continue
                        rew = V["quote"]["share"] * m["rate"] * mins / 1440
                        self.db.execute("INSERT INTO mm_rew VALUES (?,?,?,?,?,?) ON CONFLICT(cid, var, day) DO UPDATE SET "
                                        "minutes=minutes+excluded.minutes, reward=reward+excluded.reward, contested=contested+excluded.contested",
                                        (m["cid"], var, day, mins, rew, rew if s["others"] > 0 else 0.0))
                st = book_state(bids, asks, m["min_size"], m["v"]) if m["v"] > 0 else None
                prev = s["mid"]
                s["t"] = now
                if st is None:
                    for V in s["v"].values():
                        V["quote"] = None
                    continue
                mid = st["mid"]
                s["mid"] = mid
                s["others"] = st["others"]
                self.db.execute("UPDATE mm_markets SET mid=? WHERE cid=?", (mid, m["cid"]))
                for f in self.pending:
                    if f["cid"] == m["cid"]:
                        for k, h in MARKS:
                            if f[k] is None and now >= f["t"] + h:
                                f[k] = fill_pnl(f["side"], f["px"], f["sh"], mid)
                                self.db.execute(f"UPDATE mm_fills SET {k}=? WHERE id=?", (f[k], f["id"]))
                pulled = (m["rate"] <= 0 or now >= m["end_ts"] - MM_PULL or (prev is not None and abs(mid - prev) >= MM_MOVE - 1e-9))
                for var, frac, cool, _ in MM_VARIANTS:
                    V = s["v"][var]
                    away = cool and s["last_print"] is not None and now - s["last_print"] < cool
                    V["quote"] = None if pulled or away else quote(mid, m["v"], tick, st["others"], V["net"], bids, asks, frac)
            self.pending = [f for f in self.pending if f["m60"] is None and now < f["t"] + 3600 + 300]
            self.db.commit()
        live = [k for k in self.winners() if k.startswith("mm:") and self.winner_live(k)]
        if live:
            try:
                self._live_maker(live[0][3:], now)
            except Exception as e:
                self._note("live", False, e)

    def mm_print(self, asset, price, size, side, ts=None):
        """One trade print from the market's socket: a fill of each way's standing quote it crossed. Returns the fills
        [(var, side, price, shares)]."""
        ts = ts or time.time()
        out = []
        with self.lock:
            s = next((s for s in self.mm.values() if asset in (s["m"]["yes"], s["m"]["no"])), None)
            if not s:
                return out
            m = s["m"]
            if asset == m["no"]:  # (a No print is the same trade on Yes's book, mirrored)
                price, side = round(1 - price, 6), ("SELL" if side == "BUY" else "BUY")
            fresh = s["t"] and ts - s["t"] <= MM_GAP
            for var, V in s["v"].items():
                if not fresh or not V["quote"]:
                    continue
                n, ours = fill(V["quote"], price, size, side)
                if n <= 0:
                    continue
                px = V["quote"]["bid"] if ours == "buy" else V["quote"]["ask"]
                reb = m["rebate"] * m["fee"] * px * (1 - px) * n
                cur = self.db.execute("INSERT INTO mm_fills (ts,cid,side,px,sh,mid0,rebate,var) VALUES (?,?,?,?,?,?,?,?)",
                                      (ts, m["cid"], ours, px, n, s["mid"], reb, var))
                V["net"] += n if ours == "buy" else -n
                self.pending.append({"id": cur.lastrowid, "cid": m["cid"], "t": ts, "side": ours, "px": px, "sh": n,
                                     "m1": None, "m10": None, "m60": None})
                out.append((var, ours, px, n))
            if out:
                self.db.commit()
            s["last_print"] = ts
            # (the ways that step aside after a trade do it now, not at the next minute)
            for var, frac, cool, _ in MM_VARIANTS:
                if cool:
                    s["v"][var]["quote"] = None
        return out

    def mm_resolve(self, now=None):
        """The followed markets Polymarket has settled: their result, and they're no longer followed."""
        now = now or time.time()
        with self.lock:
            due = list(self.mm)  # (all of them, hourly: a market can be settled before its end date)
        for i in range(0, len(due), 40):
            got = self.get(f"{GAMMA}/markets?limit=100&closed=true&" + "&".join("condition_ids=" + c for c in due[i:i + 40])) or []
            for g in got:
                out = resolved_outcome(g)
                if out is None:
                    continue
                with self.lock:
                    self.db.execute("UPDATE mm_markets SET outcome=? WHERE cid=?", (out, g.get("conditionId")))
                    self.mm.pop(g.get("conditionId"), None)
        with self.lock:
            self.db.commit()
            self.ws_assets = self._assets()
        self.ws_changed.set()

    # --- A with real money: the winning way's quotes as real resting orders ---
    def _live_maker(self, var, now):
        """The winning way of quoting, for real: on the LIVE_MM_MARKETS largest pools, its quote of the minute as two
        post-only buys (Yes at our bid, No at one minus our ask), sized for the winner's money; moved when the quote moves,
        cancelled when it's pulled. What sellers took is recorded."""
        t = self.trader
        W = self.winners().get("mm:" + var)
        if not t or not W or not t.configured() or t.stopped():
            return
        stake = float(W.get("stake") or STAKES["mm"])
        with self.lock:
            pick = sorted(self.mm.values(), key=lambda s: -s["m"]["rate"])[:LIVE_MM_MARKETS]
            want = {}
            for s in pick:
                q = s["v"][var]["quote"]
                if not q:
                    continue
                n = stake / (q["bid"] + 1 - q["ask"])
                net = self._live_net(s["m"]["cid"])
                mid = s["mid"] or 0.5
                if q["nb"] > 0 and not (net * mid >= MM_EXPOSURE * stake / MM_CAPITAL):
                    want[(s["m"]["cid"], "yes")] = (s["m"]["yes"], q["bid"], n)
                if q["na"] > 0 and not (-net * (1 - mid) >= MM_EXPOSURE * stake / MM_CAPITAL):
                    want[(s["m"]["cid"], "no")] = (s["m"]["no"], round(1 - q["ask"], 6), n)
        for key, o in list(self.live_orders.items()):
            self._live_check(key, o, now)
            w = want.get(key)
            if not w or abs(w[1] - o["price"]) > 1e-9:
                t.cancel_maker(o["id"])
                self._live_check(key, o, now)  # (what filled before it went)
                self.live_orders.pop(key, None)
        for key, (token, price, n) in want.items():
            if key in self.live_orders:
                continue
            oid, err = t.place_maker(token, price, n)
            if oid:
                self.live_orders[key] = {"id": oid, "price": price, "shares": n, "matched": 0.0, "token": token}
            else:
                self._live_trade(now, "mm:" + var, key[0], {"side": key[1], "price": price, "shares": n}, 0.0, "rejected", err)

    def _live_check(self, key, o, now):
        try:
            matched, _ = self.trader.maker_matched(o["id"])
        except Exception:
            return
        if matched > o["matched"] + 1e-9:
            n = matched - o["matched"]
            o["matched"] = matched
            yes_px = o["price"] if key[1] == "yes" else round(1 - o["price"], 6)
            self._live_trade(now, "mm:" + self._live_var(), key[0], {"side": "buy" if key[1] == "yes" else "sell", "price": yes_px,
                                                                    "shares": n, "token": o["token"]}, n * o["price"], "filled")

    def _live_var(self):
        return next((k[3:] for k in self.winners() if k.startswith("mm:") and self.winner_live(k)), "mid")

    def _live_net(self, cid):
        net = 0.0
        for d, st in self.db.execute("SELECT detail, status FROM live_trades WHERE key=? AND status='filled'", (cid,)):
            x = json.loads(d)
            net += x.get("shares", 0) if x.get("side") == "buy" else -x.get("shares", 0)
        return net

    def _live_cancel_all(self):
        for key, o in list(self.live_orders.items()):
            if self.trader:
                self.trader.cancel_maker(o["id"])
            self.live_orders.pop(key, None)

    def _live_trade(self, now, test, key, detail, cost, status, error=None):
        with self.lock:
            self.db.execute("INSERT INTO live_trades (ts, test, key, detail, cost, status, error) VALUES (?,?,?,?,?,?,?)",
                            (now, test, key, json.dumps(detail), cost, status, (error or "")[:300] or None))
            self.db.commit()

    # --- B: a scan ---
    def ladder_scan(self, now=None):
        now = now or time.time()
        found = 0
        strikes = 0
        top = 0.0
        best = None  # (the largest gap at the offer, or at the bid for No: what a signal needs LADDER_GAP of)
        for cur, name in LADDER_COINS:
            sms = smiles(self.get(f"{DERIBIT}/get_book_summary_by_currency?currency={cur}&kind=option")["result"])
            spot = self.get(f"{DERIBIT}/get_index_price?index_name={cur.lower()}_usd")["result"]["index_price"]
            if not sms:
                continue
            et = dt.datetime.fromtimestamp(now, dt.timezone.utc) - dt.timedelta(hours=5)
            toks = {}
            for k in range(LADDER_AHEAD):
                try:
                    ev = (self.get(f"{GAMMA}/events?slug=" + ladder_slug(name, (et + dt.timedelta(days=k)).date())) or [None])[0]
                except Exception:
                    ev = None
                if not ev or not ev.get("endDate"):
                    continue
                T = dt.datetime.fromisoformat(ev["endDate"].replace("Z", "+00:00")).timestamp()
                if T <= now + 600:
                    continue
                for m in ev.get("markets") or []:
                    if m.get("closed"):
                        continue
                    try:
                        yes, no = json.loads(m["clobTokenIds"])
                        toks[yes] = (m, T, no)
                    except (ValueError, TypeError, KeyError):
                        continue
            if not toks:
                continue
            books = {}
            ids = list(toks)
            for i in range(0, len(ids), 100):
                for b in self.get(f"{CLOB}/books", [{"token_id": t} for t in ids[i:i + 100]]) or []:
                    books[b.get("asset_id")] = b
            for tok, (m, T, no) in toks.items():
                b = books.get(tok)
                K = strike_of(m)
                if not b or not K or not b.get("bids") or not b.get("asks"):
                    continue
                bid = max(float(x["price"]) for x in b["bids"])
                ask = min(float(x["price"]) for x in b["asks"])
                if not (LADDER_PRICED[0] <= (bid + ask) / 2 <= LADDER_PRICED[1]):
                    continue
                p = digital(sms, now, T, K, spot)
                if p is None:
                    continue
                strikes += 1
                top = max(top, abs((bid + ask) / 2 - p))
                best = max(best if best is not None else -1.0, p - ask, bid - p)
                sig = ladder_signal(p, bid, ask)
                if not sig:
                    continue
                side, price = sig
                size = (sum(float(x["size"]) for x in b["asks"] if abs(float(x["price"]) - ask) < 1e-9) if side == "yes"
                        else sum(float(x["size"]) for x in b["bids"] if abs(float(x["price"]) - bid) < 1e-9))
                rate = float((m.get("feeSchedule") or {}).get("rate") or LADDER_FEE)
                token = tok if side == "yes" else no
                with self.lock:
                    c = self.db.execute("INSERT OR IGNORE INTO ld_signals (ts,cur,cid,expiry,strike,side,price,size,model,gap,hours,fee,token) "
                                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                        (now, cur, m.get("conditionId"), T, K, side, price, size, p,
                                         (p - ask) if side == "yes" else (bid - p), (T - now) / 3600, rate * price * (1 - price), token))
                    self.db.commit()
                    found += c.rowcount
                if c.rowcount and self.winner_live("ladder"):
                    self._live_buy("ladder", m.get("conditionId"), token, price, min(self.winners()["ladder"]["stake"], size * price), now,
                                   {"side": side, "strike": K, "cur": cur, "expiry": T})
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with self.lock:
            self.db.execute("INSERT INTO ld_days (day, scans, strikes, max_gap, max_exec) VALUES (?,1,?,?,?) ON CONFLICT(day) DO UPDATE SET "
                            "scans=scans+1, strikes=strikes+excluded.strikes, max_gap=MAX(max_gap, excluded.max_gap), "
                            "max_exec=MAX(COALESCE(max_exec, -1), COALESCE(excluded.max_exec, -1))", (day, strikes, top, best))
            self.db.commit()
        return found

    def _live_buy(self, test, key, token, price, dollars, now, detail, kind="FAK"):
        """One real buy for a live winner: what it got is recorded either way."""
        t = self.trader
        if not t or not t.configured():
            self._live_trade(now, test, key, detail, 0.0, "rejected", "הארנק לא מחובר")
            return 0.0, 0.0
        shares, paid, oid, err = t.buy_token(token, dollars, price, kind)
        self._live_trade(now, test, key, dict(detail, token=token, price=price, shares=shares, order=oid), paid,
                         "filled" if shares > 0 else "rejected", err)
        return shares, paid

    def ladder_settle(self, now=None):
        now = now or time.time()
        with self.lock:
            due = [r[0] for r in self.db.execute("SELECT cid FROM ld_signals WHERE outcome IS NULL AND expiry < ?", (now,))]
        for i in range(0, len(due), 40):
            got = self.get(f"{GAMMA}/markets?limit=100&closed=true&" + "&".join("condition_ids=" + c for c in due[i:i + 40])) or []
            with self.lock:
                for g in got:
                    out = resolved_outcome(g)
                    if out is not None:
                        self.db.execute("UPDATE ld_signals SET outcome=? WHERE cid=?", (out, g.get("conditionId")))
                self.db.commit()

    # --- C: a scan of every set on Polymarket ---
    def arb_scan(self, now=None):
        """Every open negative-risk event (one outcome of several wins), all its outcomes still trading: buying every
        Yes for under a dollar, or every No for under n-1 dollars, after the taker fees, is a sure profit at its
        resolution. Gamma's best prices find the candidates; the books then say what could really be bought (each
        leg's minimum and a dollar counted). An opportunity is recorded once, and kept up while it lasts."""
        now = now or time.time()
        cands, cur, pages = [], None, 0
        for _ in range(ARB_PAGES):
            # (a page is megabytes of JSON, tens of thousands of objects for a moment: the garbage collector, which stops
            # every thread to walk what is new, rests until the page is read down to its few candidates and dropped.
            # 22.11.2026, the heart of the system: the trading loop waited 0.26-0.39 s every ten minutes, each time with
            # this scan and collector stops of 55-89 ms)
            with _collector_resting():
                d = self.get(f"{GAMMA}/events/keyset?active=true&closed=false&limit=500" + (f"&after_cursor={cur}" if cur else ""), gz=True)
                pages += 1
                events = (d.get("events") or []) if d else []
                for e in events:
                    c = arb_candidate(e, now)
                    if c:
                        cands.append(c)
                cur = d.get("next_cursor") if d else None
                more = bool(cur and events)
                d = events = e = None
            if not more:
                break
        found = 0
        for c in cands:
            toks = [leg["token"] for leg in c["legs"]]
            books = {}
            for i in range(0, len(toks), 100):
                for b in self.get(f"{CLOB}/books", [{"token_id": t} for t in toks[i:i + 100]]) or []:
                    books[b.get("asset_id")] = b
            legs, mins = [], []
            for leg in c["legs"]:
                b = books.get(leg["token"]) or {}
                asks = [(float(x["price"]), float(x["size"])) for x in b.get("asks") or []]
                legs.append((asks, leg["fee"]))
                mins.append(float(b.get("min_order_size") or 5))
            payout = 1.0 if c["kind"] == "yes" else len(legs) - 1.0
            sets, cost, top, first = walk_set(legs, payout, ARB_MIN_EDGE)
            if sets <= 0:
                continue
            least = least_sets(top, mins)
            profit = sets * payout - cost
            days = max(1.0, (c["end"] - now) / 86400)
            apr = profit / cost * 365 / days * 100 if cost > 0 else 0.0
            ok = sets >= least
            found += self._arb_record(now, c, sets, cost, profit, first, days, apr, ok, least)
            if ok and self.winner_live("arb"):
                self._arb_live(now, c, legs, payout, top, least)
        self.status["arb"].update(pages=pages, candidates=len(cands))
        return found

    def _arb_record(self, now, c, sets, cost, profit, edge, days, apr, ok, least):
        key = f"{c['slug']}:{c['kind']}"
        with self.lock:
            r = self.db.execute("SELECT id, profit FROM ar_opps WHERE key=? AND last > ?", (key, now - ARB_SAME)).fetchone()
            if r:
                self.db.execute("UPDATE ar_opps SET last=?, seen=seen+1, profit=MAX(profit, ?), sets=MAX(sets, ?), cost=MAX(cost, ?), "
                                "ok=MAX(ok, ?) WHERE id=?", (now, profit, sets, cost, int(ok), r[0]))
                self.db.commit()
                return 0
            self.db.execute("INSERT INTO ar_opps (first, last, key, slug, title, kind, legs, edge, sets, cost, profit, end_ts, days, apr, ok, least, seen) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                            (now, now, key, c["slug"], c["title"], c["kind"], len(c["legs"]), edge, sets, cost, profit, c["end"], days, apr,
                             int(ok), least))
            self.db.commit()
            return 1

    def _arb_live(self, now, c, legs, payout, top, least):
        """A live arbitrage winner: every leg bought, all or nothing each (fill-or-kill at the dearest level the set
        needed), the winner's money a set at most; legs in order of the thinnest first. A leg refused after others filled
        leaves a partial set: recorded as such, for the owner to see."""
        key = f"{c['slug']}:{c['kind']}"
        if self.db.execute("SELECT 1 FROM live_trades WHERE test='arb' AND key=? AND ts > ?", (key, now - ARB_SAME)).fetchone():
            return
        stake = float(self.winners()["arb"].get("stake") or STAKES["arb"])
        per = sum(p for p in top) or 1
        sets = max(least, min(stake / per, min(sum(s for _, s in lv) for lv, _ in legs)))
        order = sorted(range(len(legs)), key=lambda i: sum(s for _, s in legs[i][0]))
        got = []
        for i in order:
            sh, paid = self._live_buy("arb", key, c["legs"][i]["token"], top[i], round(sets * top[i] + 0.005, 2), now,
                                      {"leg": c["legs"][i]["name"], "kind": c["kind"], "sets": sets}, kind="FOK")
            got.append(sh)
            if sh < sets * 0.99:
                break
        if len(got) < len(legs) or min(got) < sets * 0.99:
            self._live_trade(now, "arb", key, {"partial": True, "legs": len(legs), "filled": len(got)}, 0.0, "partial",
                             "חלק מהרגליים לא התמלאו: נשאר סט חלקי")

    # --- the threads ---
    def start(self, stop):
        for key in KEYS:
            self.started(key)
        for name, fn in (("paper-mm", self._mm_loop), ("paper-ladder", self._ladder_loop), ("paper-arb", self._arb_loop)):
            threading.Thread(target=fn, args=(stop,), daemon=True, name=name).start()
        threading.Thread(target=lambda: asyncio.run(self._ws(stop)), daemon=True, name="paper-mm-ws").start()

    def _note(self, which, ok, err=None):
        st = self.status.setdefault(which, {})
        if ok:
            st.update(last=time.time(), error=None)
        else:
            st.update(error=str(err)[:200], error_at=time.time())
            log.warning("paper test %s: %s", which, err)

    def _ran(self, name, began):
        """A scan that ended, for the heart of the system: the trading loop's waits are set beside what ran then."""
        if self.on_job:
            try:
                self.on_job(name, began, time.time() - began)
            except Exception:
                pass

    def _mm_loop(self, stop):
        last_res = 0
        while not stop.is_set():
            t0 = time.time()
            try:
                if t0 - (self.meta("mm_universe") or 0) >= MM_UNIVERSE_EVERY:
                    self.mm_universe(t0)
                with self.lock:
                    toks = [s["m"]["yes"] for s in self.mm.values()]
                books = {}
                for i in range(0, len(toks), 100):
                    for b in self.get(f"{CLOB}/books", [{"token_id": t} for t in toks[i:i + 100]], timeout=45) or []:
                        books[b.get("asset_id")] = b
                self.mm_read(books, time.time())
                if t0 - last_res >= 3600:
                    last_res = t0
                    self.mm_resolve(t0)
                self._note("mm", True)
            except Exception as e:
                self._note("mm", False, e)
            self._ran("research-mm", t0)
            stop.wait(max(5, MM_EVERY - (time.time() - t0)))

    def _ladder_loop(self, stop):
        last_settle = 0
        while not stop.is_set():
            t0 = time.time()
            try:
                self.ladder_scan(t0)
                if t0 - last_settle >= 900:
                    last_settle = t0
                    self.ladder_settle(t0)
                self._note("ladder", True)
            except Exception as e:
                self._note("ladder", False, e)
            self._ran("research-ladder", t0)
            stop.wait(max(5, LADDER_EVERY - (time.time() - t0)))

    def _arb_loop(self, stop):
        stop.wait(30)
        while not stop.is_set():
            t0 = time.time()
            try:
                self.arb_scan(t0)
                self._note("arb", True)
            except Exception as e:
                self._note("arb", False, e)
            self._ran("research-arb", t0)
            stop.wait(max(30, ARB_EVERY - (time.time() - t0)))

    async def _ws(self, stop):
        import websockets  # (requirements.txt: the trading feeds use it too)
        while not stop.is_set():
            assets = list(self.ws_assets)
            if not assets:
                await asyncio.sleep(10)
                continue
            self.ws_changed.clear()
            try:
                async with websockets.connect(WS_URL, ping_interval=None, open_timeout=30, max_size=2 ** 24) as ws:
                    await ws.send(json.dumps({"assets_ids": assets, "type": "market"}))
                    ping = time.time()
                    while not stop.is_set() and not self.ws_changed.is_set():
                        if time.time() - ping > 10:
                            await ws.send("PING")
                            ping = time.time()
                        try:
                            msg = await asyncio.wait_for(ws.recv(), 5)
                        except asyncio.TimeoutError:
                            continue
                        if msg == "PONG":
                            continue
                        try:
                            data = json.loads(msg)
                        except ValueError:
                            continue
                        for ev in data if isinstance(data, list) else [data]:
                            if isinstance(ev, dict) and ev.get("event_type") == "last_trade_price":
                                try:
                                    ts = int(ev["timestamp"]) / 1000 if ev.get("timestamp") else time.time()
                                    self.mm_print(ev["asset_id"], float(ev["price"]), float(ev["size"]), ev.get("side"), ts)
                                except (KeyError, TypeError, ValueError):
                                    continue
                                self._live_step_aside(ev.get("asset_id"))
                        self.status["mm"]["ws"] = time.time()
            except Exception as e:
                self.status["mm"]["ws_error"] = str(e)[:200]
                await asyncio.sleep(5)

    def _live_step_aside(self, asset):
        """A live maker winner whose way steps aside after a trade: its orders in that market cancelled at once."""
        if not self.live_orders:
            return
        var = self._live_var()
        if not dict((v[0], v[2]) for v in MM_VARIANTS).get(var):
            return
        cid = next((s["m"]["cid"] for s in self.mm.values() if asset in (s["m"]["yes"], s["m"]["no"])), None)
        for key in [k for k in self.live_orders if k[0] == cid]:
            o = self.live_orders.pop(key, None)
            if o and self.trader:
                threading.Thread(target=self.trader.cancel_maker, args=(o["id"],), daemon=True).start()

    # --- the card ---
    def research_status(self, key, now=None):
        """A research's place: its day of DAYS, and "investigating" / "not_found" / "found"."""
        now = now or time.time()
        since = self.started(key)
        v = self._view_of(key, since, now)
        status = {"go": "found", "nogo": "not_found", "unclear": "not_found"}.get(v.get("verdict"), "investigating")
        n = LENGTH.get(key, DAYS)
        return {"key": key, "title": TITLES[key], "since": since, "day": (now - since) / 86400, "days": n,
                "progress": min(1.0, max(0.0, (now - since) / (n * 86400))), "status": status}

    def _view_of(self, key, since, now):
        if key.startswith("mm:"):
            return self.mm_view(key[3:], since, now)
        return self.ladder_view(since, now) if key == "ladder" else self.arb_view(since, now)

    VIEW_KEEP = 20  # seconds the card's numbers are kept

    def view(self, now=None):
        """The card, never waited for once it was worked out a first time (the owner, 18.11.2026: "it takes ages to load"):
        it was worked out on every look, behind the researches' own lock, so a look during a scan waited for the scan. The
        one kept goes out at once; older than VIEW_KEEP, a fresh one is worked out behind it. A change of the owner's (a
        winner made, moved, removed) drops it: his click shows at once. now given (the tests): worked out as asked."""
        if now is not None:
            return self._view(now)
        hit = self.__dict__.get("_view_hit")
        if hit is None:
            out = self._view(time.time())
            self._view_hit = (time.time(), out)
            return out
        if time.time() - hit[0] >= self.VIEW_KEEP and not self.__dict__.get("_view_busy"):
            self._view_busy = True

            def go():
                try:
                    self._view_hit = (time.time(), self._view(time.time()))
                except Exception:
                    log.exception("paper tests' card")
                finally:
                    self._view_busy = False
            threading.Thread(target=go, daemon=True, name="paper-view").start()
        return hit[1]

    def _view(self, now):
        with self.lock:
            out = {"days": DAYS, "research": [], "mm": {}, "winners": {}}
            for key in KEYS:
                since = self.started(key)
                v = self._view_of(key, since, now)
                status = {"go": "found", "nogo": "not_found", "unclear": "not_found"}.get(v.get("verdict"), "investigating")
                n = LENGTH.get(key, DAYS)
                out["research"].append({"key": key, "title": TITLES[key], "since": since, "day": (now - since) / 86400, "days": n,
                                        "progress": min(1.0, max(0.0, (now - since) / (n * 86400))), "status": status})
                if key.startswith("mm:"):
                    out["mm"][key[3:]] = dict(v, name=MM_NAMES[key[3:]])
                else:
                    out[key] = v
            st = self.status["mm"]
            out["mm_info"] = {"markets": len(self.mm), "pools": round(sum(s["m"]["rate"] for s in self.mm.values()), 0),
                              "quoting": {var: sum(1 for s in self.mm.values() if s["v"][var]["quote"]) for var in MM_NAMES},
                              "last": st.get("last"), "ws": st.get("ws"),
                              "error": st.get("error") if (st.get("error_at") or 0) > (st.get("last") or 0) else None,
                              "variants": [{"var": v[0], "name": v[3], "frac": v[1], "cool": v[2]} for v in MM_VARIANTS],
                              "rules": {"capital": MM_CAPITAL, "min_rate": MM_MIN_RATE, "min_days": MM_MIN_DAYS, "max": MM_MAX,
                                        "exposure": MM_EXPOSURE}}
            for key, w in self.winners().items():
                out["winners"][key] = self.winner_view(key, w, now)
            return out

    def winner_view(self, key, w, now):
        """A winner in the simulation: the research's numbers from its creation on; with real money, its own trades."""
        sim = self._view_of(key, w["created"], now)
        stake = float(w.get("stake") or 0)
        if key == "ladder":  # (the research's $100 a signal: the winner's own money instead)
            sim = dict(sim, dollars=round(sim["dollars"] * stake / LADDER_STAKE, 2), rules=dict(sim["rules"], stake=stake))
        elif key == "arb":  # (each set as deep as the winner's money a set allows)
            rows = self.db.execute("SELECT cost, profit FROM ar_opps WHERE first >= ? AND ok=1", (w["created"],)).fetchall()
            got = sum(p * min(1.0, stake / c) for c, p in rows if c > 0)
            sim = dict(sim, profit=round(got, 2), month=round(got / max(1.0, (now - w["created"]) / 86400) * 30, 2))
        else:  # (the research's $1,000 in each market: the winner's own)
            f = stake / MM_CAPITAL
            sim = dict(sim, **{k: round((sim.get(k) or 0) * f, 2) for k in ("rewards", "marks", "rebates", "net")})
        rows = self.db.execute("SELECT ts, key, detail, cost, status, error FROM live_trades WHERE test=? ORDER BY ts DESC LIMIT 30",
                               (key,)).fetchall()
        live = [{"ts": r[0], "key": r[1], "detail": json.loads(r[2]), "cost": r[3], "status": r[4], "error": r[5]} for r in rows]
        spent = self.db.execute("SELECT COALESCE(SUM(cost), 0), COUNT(*) FROM live_trades WHERE test=? AND status='filled'", (key,)).fetchone()
        return dict(w, key=key, sim=sim, live_trades=live, live_spent=round(spent[0], 2), live_fills=spent[1],
                    connected=bool(self.trader and self.trader.configured()))

    def mm_view(self, var, since, now):
        fills = self.db.execute("SELECT f.ts, f.cid, f.side, f.px, f.sh, f.mid0, f.m1, f.m10, f.m60, f.rebate, m.mid, m.outcome, m.q, m.cat "
                                "FROM mm_fills f JOIN mm_markets m ON m.cid=f.cid WHERE COALESCE(f.var, 'mid')=? AND f.ts >= ? ORDER BY f.ts",
                                (var, since)).fetchall()
        rew = [r for r in self.db.execute("SELECT cid, day, minutes, reward, contested FROM mm_rew WHERE var=?", (var,)).fetchall()
               if _day_ts(r[1]) + 86400 > since]
        mins = sum(r[2] for r in rew)
        mdays = mins / 1440
        rewards = sum(r[3] for r in rew)
        contested = sum(r[4] for r in rew)
        rebates = sum(f[9] or 0 for f in fills)
        per_fill = []
        by_mkt = {}
        for f in fills:
            mark = f[11] if f[11] is not None else f[10]
            v = fill_pnl(f[2], f[3], f[4], mark) if mark is not None else 0.0
            per_fill.append(v)
            by_mkt.setdefault(f[1], [0.0, 0.0])[0] += v + (f[9] or 0)
        for r in rew:
            x = by_mkt.setdefault(r[0], [0.0, 0.0])
            x[0] += r[3]
            x[1] += r[2] / 1440
        marks = sum(per_fill)
        net = rewards + marks + rebates
        per_mday = net / mdays if mdays else None
        month = per_mday * 30 / MM_CAPITAL * 100 if per_mday is not None else None
        low = None
        mk = [v for v in by_mkt.values() if v[1] > 0]
        if len(mk) >= 5:
            rng = random.Random(7)
            res = []
            for _ in range(400):
                pick = [mk[rng.randrange(len(mk))] for _ in mk]
                d = sum(p[1] for p in pick)
                res.append(sum(p[0] for p in pick) / d * 30 / MM_CAPITAL * 100 if d else 0)
            res.sort()
            low = res[len(res) // 10]
        weeks = []
        for w in range(4):
            a, b = since + w * WEEK, since + (w + 1) * WEEK
            if a > now:
                break
            wm = sum(r[2] for r in rew if a <= _day_ts(r[1]) + 43200 < b)
            wn = (sum(r[3] for r in rew if a <= _day_ts(r[1]) + 43200 < b)
                  + sum(per_fill[i] + (f[9] or 0) for i, f in enumerate(fills) if a <= f[0] < b))
            cap = wm / (7 * 1440) * MM_CAPITAL
            weeks.append({"n": w + 1, "net": round(wn, 2), "pct": round(wn / cap * 100, 2) if cap else None, "done": now >= b})
        m10 = [f[7] / f[4] for f in fills if f[7] is not None and f[4]]
        edge = [(1 if f[2] == "buy" else -1) * (f[5] - f[3]) for f in fills if f[5] is not None]
        days_reward = rewards / max(1.0, (now - since) / 86400)  # (all the markets' rewards in a day)
        worst = min(per_fill) if per_fill else None
        resolved = len({f[1] for f in fills if f[11] is not None})
        avg = lambda xs: sum(xs) / len(xs) if xs else None
        done = now - since >= DAYS * 86400
        checks = [
            {"k": "fills", "ok": len(fills) >= GO_FILLS and resolved >= GO_RESOLVED, "v": f"{len(fills)} / {resolved}"},
            {"k": "month", "ok": month is not None and month >= GO_MONTH and low is not None and low > 0,
             "v": (f"{month:+.2f}% · {low:+.2f}%" if low is not None else f"{month:+.2f}%") if month is not None else "–"},
            {"k": "weeks", "ok": sum(1 for w in weeks if w["done"] and (w["pct"] or 0) > 0) >= GO_WEEKS
             and all((w["pct"] or 0) >= GO_WORST_WEEK for w in weeks if w["done"]),
             "v": f"{sum(1 for w in weeks if w['done'] and (w['pct'] or 0) > 0)}/{sum(1 for w in weeks if w['done'])}"},
            {"k": "adverse", "ok": bool(m10) and avg(m10) >= 0, "v": m10 and f"{avg(m10) * 100:+.2f}¢ · {avg(edge) * 100:+.2f}¢"},
            {"k": "worst", "ok": worst is None or worst >= -GO_FILL_DAYS * days_reward,
             "v": worst is not None and f"{'-' if worst < 0 else ''}${abs(worst):,.0f} · ${days_reward:,.0f}"},
            {"k": "contested", "ok": rewards > 0 and contested / rewards >= GO_CONTESTED,
             "v": f"{contested / rewards * 100:.0f}%" if rewards else "–"},
        ]
        verdict = ("go" if all(c["ok"] for c in checks) else "nogo") if done else None
        recent = [{"ts": f[0], "q": f[12], "side": f[2], "px": f[3], "sh": round(f[4], 1), "pnl": round(per_fill[i], 2),
                   "m10": None if f[7] is None else round(f[7], 2), "resolved": f[11] is not None} for i, f in enumerate(fills)][-8:][::-1]
        return {"fills": len(fills), "resolved": resolved, "market_days": round(mdays, 1), "rewards": round(rewards, 2), "marks": round(marks, 2),
                "rebates": round(rebates, 2), "net": round(net, 2), "per_mday": None if per_mday is None else round(per_mday, 2),
                "month": None if month is None else round(month, 2), "month_low": None if low is None else round(low, 2),
                "weeks": weeks, "checks": checks, "verdict": verdict, "recent": recent}

    def ladder_view(self, since, now):
        rows = self.db.execute("SELECT ts, cur, strike, side, price, size, model, gap, hours, fee, outcome, expiry FROM ld_signals "
                               "WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
        days = self.db.execute("SELECT day, scans, strikes, max_gap, max_exec FROM ld_days ORDER BY day").fetchall()
        settled = [r for r in rows if r[10] is not None]
        pnl = [ladder_pnl(r[3], r[4], r[9], r[10]) for r in settled]
        mean = sum(pnl) / len(pnl) * 100 if pnl else None
        t = t_stat(pnl)
        span = max(1e-9, (now - since) / 86400)
        per_day = len(rows) / max(1.0, span)
        weeks = []
        for w in range(4):
            a, b = since + w * WEEK, since + (w + 1) * WEEK
            if a > now:
                break
            xs = [ladder_pnl(r[3], r[4], r[9], r[10]) for r in settled if a <= r[0] < b]
            weeks.append({"n": w + 1, "n_sig": len(xs), "cents": round(sum(xs) / len(xs) * 100, 2) if xs else None, "done": now >= b})
        split = {}
        for name, test in (("near", lambda h: h < 12), ("far", lambda h: h >= 12)):
            xs = [ladder_pnl(r[3], r[4], r[9], r[10]) for r in settled if test(r[8])]
            split[name] = {"n": len(xs), "cents": round(sum(xs) / len(xs) * 100, 2) if xs else None}
        done = now - since >= DAYS * 86400
        wk_pos = sum(1 for w in weeks if w["done"] and (w["cents"] or 0) > 0)
        checks = [
            {"k": "mean", "ok": mean is not None and mean >= LADDER_GO_CENTS, "v": mean is not None and f"{mean:+.2f}¢"},
            {"k": "t", "ok": t is not None and t >= LADDER_GO_T, "v": t is not None and f"{t:.2f}"},
            {"k": "weeks", "ok": wk_pos >= GO_WEEKS, "v": f"{wk_pos}/{sum(1 for w in weeks if w['done'])}"},
            {"k": "often", "ok": per_day >= LADDER_NOGO_PER_DAY, "v": f"{per_day:.1f}"},
        ]
        verdict = None
        if done:
            if mean is None or mean < LADDER_NOGO_CENTS or per_day < LADDER_NOGO_PER_DAY:
                verdict = "nogo"
            elif all(c["ok"] for c in checks):
                verdict = "go"
            else:
                verdict = "unclear"
        recent = [{"ts": r[0], "cur": r[1], "strike": r[2], "side": r[3], "price": r[4], "size": r[5] and round(r[5], 0),
                   "model": round(r[6], 3), "gap": round(r[7] * 100, 1), "hours": round(r[8], 1),
                   "pnl": None if r[10] is None else round(ladder_pnl(r[3], r[4], r[9], r[10]) * 100, 1)} for r in rows][-10:][::-1]
        st = self.status["ladder"]
        return {"signals": len(rows), "settled": len(settled), "open": len(rows) - len(settled), "mean_cents": None if mean is None else round(mean, 2),
                "t": None if t is None else round(t, 2), "per_day": round(per_day, 2),
                "dollars": round(sum(x * LADDER_STAKE / r[4] for x, r in zip(pnl, settled)), 2) if pnl else 0.0,
                "weeks": weeks, "split": split, "checks": checks, "verdict": verdict, "recent": recent,
                "today": days and {"day": days[-1][0], "scans": days[-1][1], "max_gap": round(days[-1][3] * 100, 1),
                                   "max_exec": None if days[-1][4] is None or days[-1][4] < -0.5 else round(days[-1][4] * 100, 1)},
                "last": st.get("last"), "error": st.get("error") if (st.get("error_at") or 0) > (st.get("last") or 0) else None,
                "rules": {"gap": LADDER_GAP, "stake": LADDER_STAKE, "go_cents": LADDER_GO_CENTS, "go_t": LADDER_GO_T,
                          "nogo_cents": LADDER_NOGO_CENTS, "nogo_per_day": LADDER_NOGO_PER_DAY}}

    def arb_view(self, since, now):
        rows = self.db.execute("SELECT first, last, slug, title, kind, legs, edge, sets, cost, profit, end_ts, days, apr, ok, least, seen "
                               "FROM ar_opps WHERE first >= ? ORDER BY first", (since,)).fetchall()
        ok = [r for r in rows if r[13]]
        span = max(1.0, (now - since) / 86400)
        profit = sum(r[9] for r in ok)
        cost = sum(r[8] for r in ok)
        month = profit / span * 30
        apr = sum(r[12] * r[8] for r in ok) / cost if cost else None  # (weighted by the money each needed)
        weeks = []
        for w in range(4):
            a, b = since + w * WEEK, since + (w + 1) * WEEK
            if a > now:
                break
            xs = [r for r in ok if a <= r[0] < b]
            weeks.append({"n": w + 1, "n_opp": len(xs), "dollars": round(sum(r[9] for r in xs), 2), "done": now >= b})
        done = now - since >= DAYS * 86400
        checks = [
            {"k": "count", "ok": len(ok) >= ARB_GO_COUNT, "v": f"{len(ok)}"},
            {"k": "month", "ok": month >= ARB_GO_MONTH, "v": f"${month:,.0f}"},
            {"k": "apr", "ok": apr is not None and apr >= ARB_GO_APR, "v": "–" if apr is None else f"{apr:.0f}%"},
            {"k": "weeks", "ok": sum(1 for w in weeks if w["done"] and w["n_opp"] > 0) >= GO_WEEKS,
             "v": f"{sum(1 for w in weeks if w['done'] and w['n_opp'] > 0)}/{sum(1 for w in weeks if w['done'])}"},
        ]
        verdict = ("go" if all(c["ok"] for c in checks) else "nogo") if done else None
        recent = [{"ts": r[0], "last": r[1], "slug": r[2], "title": r[3], "kind": r[4], "legs": r[5], "edge": round((r[6] or 0) * 100, 2),
                   "sets": round(r[7], 1), "cost": round(r[8], 2), "profit": round(r[9], 2), "days": round(r[11], 0), "apr": round(r[12], 1),
                   "ok": bool(r[13]), "least": round(r[14] or 0, 1), "seen": r[15]} for r in rows][-12:][::-1]
        st = self.status["arb"]
        return {"opps": len(rows), "executable": len(ok), "profit": round(profit, 2), "cost": round(cost, 2), "month": round(month, 2),
                "apr": None if apr is None else round(apr, 1), "weeks": weeks, "checks": checks, "verdict": verdict, "recent": recent,
                "last": st.get("last"), "pages": st.get("pages"), "candidates": st.get("candidates"),
                "error": st.get("error") if (st.get("error_at") or 0) > (st.get("last") or 0) else None,
                "rules": {"min_edge": ARB_MIN_EDGE, "every": ARB_EVERY, "go_count": ARB_GO_COUNT, "go_month": ARB_GO_MONTH, "go_apr": ARB_GO_APR}}


def arb_candidate(e, now):
    """A negative-risk event whose every outcome still trades, priced (by Gamma's best prices) as a sure profit: the
    set to buy, every Yes or every No, its legs' tokens and fees. None otherwise."""
    if not e.get("negRisk") or e.get("negRiskAugmented"):  # (augmented: outcomes can still be added; a set isn't sure)
        return None
    ms = e.get("markets") or []
    if len(ms) < 2 or any(m.get("closed") or not m.get("acceptingOrders") for m in ms):
        return None
    try:
        end = dt.datetime.fromisoformat((e.get("endDate") or "").replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    if end < now:
        return None
    legs = []
    for m in ms:
        try:
            yes, no = json.loads(m["clobTokenIds"])
            a, b = float(m.get("bestAsk")), float(m.get("bestBid"))
        except (TypeError, ValueError, KeyError):
            return None
        legs.append({"yes": yes, "no": no, "ask": a, "bid": b, "fee": float((m.get("feeSchedule") or {}).get("rate") or 0),
                     "name": m.get("groupItemTitle") or m.get("question") or ""})
    fee = lambda r, p: r * p * (1 - p)
    out = {"slug": e.get("slug"), "title": e.get("title"), "end": end}
    if all(x["ask"] > 0 for x in legs) and sum(x["ask"] + fee(x["fee"], x["ask"]) for x in legs) < 1 - ARB_MIN_EDGE:
        return dict(out, kind="yes", legs=[{"token": x["yes"], "fee": x["fee"], "name": x["name"]} for x in legs])
    if all(0 < x["bid"] < 1 for x in legs) and sum((1 - x["bid"]) + fee(x["fee"], 1 - x["bid"]) for x in legs) < len(legs) - 1 - ARB_MIN_EDGE:
        return dict(out, kind="no", legs=[{"token": x["no"], "fee": x["fee"], "name": x["name"]} for x in legs])
    return None


def _day_ts(day):
    return dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()


def resolved_outcome(g):
    """A closed Gamma market's Yes result (1 or 0), or None while it isn't settled."""
    if not g or not g.get("closed"):
        return None
    try:
        p = [float(x) for x in json.loads(g.get("outcomePrices") or "[]")]
    except (ValueError, TypeError):
        return None
    if len(p) >= 1 and p[0] in (0.0, 1.0):
        return p[0]
    return None


def open_at(data_dir):
    os.makedirs(data_dir or ".", exist_ok=True)
    return PaperTests(os.path.join(data_dir or ".", "paper_tests.sqlite"))
