"""Two paper tests that never trade (the owner, 17.11.2026: "build both, A and B").

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

Each has the research's own go rules, judged after DAYS days (the view's "verdict"). Its own database file, so
nothing here writes to the trading data; its own threads, started by the engine in production only.
"""

import asyncio
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

UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Accept": "application/json"}


def http(url, body=None, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


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


def quote(mid, v, tick, others, net, bids, asks):
    """Our quote for the next minute: half the band from the middle, rounded away from it to the tick; balanced
    shares for MM_CAPITAL; past MM_EXPOSURE one way, the side that adds to it is left out. Returns None (no quote)
    or {bid, ask, nb, na, share, qa_bid, qa_ask} (nb/na 0 for a side not quoted)."""
    half = max(tick, round(v / 2 / tick) * tick)
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


class PaperTests:
    def __init__(self, path, get=http):
        self.path = path
        self.get = get
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.executescript(SCHEMA)
        try:  # (1.355: the day's largest gap at a price one could really buy, beside the middle's)
            self.db.execute("ALTER TABLE ld_days ADD COLUMN max_exec REAL")
        except sqlite3.OperationalError:
            pass
        self.lock = threading.RLock()
        self.mm = {}  # cid -> the live state: market row, quote, last middle, net shares, last read
        self.pending = []  # fills still waiting for a mark (dicts with id, cid, t, side, px, sh)
        self.status = {"mm": {}, "ladder": {}}
        self.ws_assets = []
        self.ws_changed = threading.Event()
        self._load()

    # --- storage ---
    def meta(self, key, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))
        self.db.commit()

    def started(self):
        s = self.meta("started")
        if s is None:
            s = time.time()
            self.set_meta("started", s)
        return s

    def _load(self):
        with self.lock:
            now = time.time()
            for r in self.db.execute("SELECT cid, q, cat, yes, no, rate, v, min_size, fee, rebate, end_ts, mid FROM mm_markets "
                                     "WHERE outcome IS NULL").fetchall():
                self._track(dict(zip(("cid", "q", "cat", "yes", "no", "rate", "v", "min_size", "fee", "rebate", "end_ts"), r[:11])), r[11])
            for cid, side, sh in self.db.execute("SELECT cid, side, sh FROM mm_fills"):
                if cid in self.mm:
                    self.mm[cid]["net"] += sh if side == "buy" else -sh
            # (fills whose marks are still due and not yet too late to take: after a restart the minutes between are lost)
            for r in self.db.execute("SELECT id, cid, ts, side, px, sh, m1, m10, m60 FROM mm_fills WHERE m60 IS NULL AND ts > ?",
                                     (now - 3600 - 300,)).fetchall():
                self.pending.append({"id": r[0], "cid": r[1], "t": r[2], "side": r[3], "px": r[4], "sh": r[5],
                                     "m1": r[6], "m10": r[7], "m60": r[8]})
            self.ws_assets = self._assets()

    def _track(self, m, mid=None):
        self.mm[m["cid"]] = {"m": m, "quote": None, "mid": mid, "net": 0.0, "t": None, "share": 0.0, "others": 0.0}

    def _assets(self):
        return sorted({a for s in self.mm.values() for a in (s["m"]["yes"], s["m"]["no"])})

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
        paid its reward share, the fills' marks that came due are taken, and the next minute's quote is set."""
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
                # the minute just gone: its reward
                if s["t"] is not None and s["quote"] is not None and now - s["t"] <= MM_GAP:
                    mins = (now - s["t"]) / 60
                    rew = s["quote"]["share"] * m["rate"] * mins / 1440
                    self.db.execute("INSERT INTO mm_rewards VALUES (?,?,?,?,?) ON CONFLICT(cid, day) DO UPDATE SET "
                                    "minutes=minutes+excluded.minutes, reward=reward+excluded.reward, contested=contested+excluded.contested",
                                    (m["cid"], day, mins, rew, rew if s["others"] > 0 else 0.0))
                st = book_state(bids, asks, m["min_size"], m["v"]) if m["v"] > 0 else None
                prev = s["mid"]
                s["t"] = now
                if st is None:
                    s["quote"] = None
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
                s["quote"] = None if pulled else quote(mid, m["v"], tick, st["others"], s["net"], bids, asks)
            self.pending = [f for f in self.pending if f["m60"] is None and now < f["t"] + 3600 + 300]
            self.db.commit()

    def mm_print(self, asset, price, size, side, ts=None):
        """One trade print from the market's socket: a fill of our standing quote, if it crossed it."""
        ts = ts or time.time()
        with self.lock:
            s = next((s for s in self.mm.values() if asset in (s["m"]["yes"], s["m"]["no"])), None)
            if not s or not s["quote"] or not s["t"] or ts - s["t"] > MM_GAP:
                return None
            m = s["m"]
            if asset == m["no"]:  # (a No print is the same trade on Yes's book, mirrored)
                price, side = round(1 - price, 6), ("SELL" if side == "BUY" else "BUY")
            n, ours = fill(s["quote"], price, size, side)
            if n <= 0:
                return None
            px = s["quote"]["bid"] if ours == "buy" else s["quote"]["ask"]
            reb = m["rebate"] * m["fee"] * px * (1 - px) * n
            cur = self.db.execute("INSERT INTO mm_fills (ts,cid,side,px,sh,mid0,rebate) VALUES (?,?,?,?,?,?,?)",
                                  (ts, m["cid"], ours, px, n, s["mid"], reb))
            self.db.commit()
            s["net"] += n if ours == "buy" else -n
            self.pending.append({"id": cur.lastrowid, "cid": m["cid"], "t": ts, "side": ours, "px": px, "sh": n,
                                 "m1": None, "m10": None, "m60": None})
            return ours, px, n

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
                        toks[json.loads(m["clobTokenIds"])[0]] = (m, T)
                    except (ValueError, TypeError, KeyError):
                        continue
            if not toks:
                continue
            books = {}
            ids = list(toks)
            for i in range(0, len(ids), 100):
                for b in self.get(f"{CLOB}/books", [{"token_id": t} for t in ids[i:i + 100]]) or []:
                    books[b.get("asset_id")] = b
            for tok, (m, T) in toks.items():
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
                with self.lock:
                    c = self.db.execute("INSERT OR IGNORE INTO ld_signals (ts,cur,cid,expiry,strike,side,price,size,model,gap,hours,fee) "
                                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                        (now, cur, m.get("conditionId"), T, K, side, price, size, p,
                                         (p - ask) if side == "yes" else (bid - p), (T - now) / 3600, rate * price * (1 - price)))
                    found += c.rowcount
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with self.lock:
            self.db.execute("INSERT INTO ld_days (day, scans, strikes, max_gap, max_exec) VALUES (?,1,?,?,?) ON CONFLICT(day) DO UPDATE SET "
                            "scans=scans+1, strikes=strikes+excluded.strikes, max_gap=MAX(max_gap, excluded.max_gap), "
                            "max_exec=MAX(COALESCE(max_exec, -1), COALESCE(excluded.max_exec, -1))", (day, strikes, top, best))
            self.db.commit()
        return found

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

    # --- the threads ---
    def start(self, stop):
        self.started()
        for name, fn in (("paper-mm", self._mm_loop), ("paper-ladder", self._ladder_loop)):
            threading.Thread(target=fn, args=(stop,), daemon=True, name=name).start()
        threading.Thread(target=lambda: asyncio.run(self._ws(stop)), daemon=True, name="paper-mm-ws").start()

    def _note(self, which, ok, err=None):
        st = self.status[which]
        if ok:
            st.update(last=time.time(), error=None)
        else:
            st.update(error=str(err)[:200], error_at=time.time())
            log.warning("paper test %s: %s", which, err)

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
            stop.wait(max(5, LADDER_EVERY - (time.time() - t0)))

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
                        self.status["mm"]["ws"] = time.time()
            except Exception as e:
                self.status["mm"]["ws_error"] = str(e)[:200]
                await asyncio.sleep(5)

    # --- the card ---
    def view(self, now=None):
        now = now or time.time()
        with self.lock:
            since = self.meta("started")
            return {"since": since, "days": DAYS, "day": (now - since) / 86400 if since else 0,
                    "mm": self.mm_view(since, now), "ladder": self.ladder_view(since, now)}

    def mm_view(self, since, now):
        fills = self.db.execute("SELECT f.ts, f.cid, f.side, f.px, f.sh, f.mid0, f.m1, f.m10, f.m60, f.rebate, m.mid, m.outcome, m.q, m.cat "
                                "FROM mm_fills f JOIN mm_markets m ON m.cid=f.cid ORDER BY f.ts").fetchall()
        rew = self.db.execute("SELECT cid, day, minutes, reward, contested FROM mm_rewards").fetchall()
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
        if since:
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
        days_reward = rewards / max(1.0, (now - since) / 86400) if since else 0  # (all the markets' rewards in a day)
        worst = min(per_fill) if per_fill else None
        resolved = len({f[1] for f in fills if f[11] is not None})
        avg = lambda xs: sum(xs) / len(xs) if xs else None
        done = since and now - since >= DAYS * 86400
        checks = [
            {"k": "fills", "ok": len(fills) >= GO_FILLS and resolved >= GO_RESOLVED, "v": f"{len(fills)} / {resolved}"},
            {"k": "month", "ok": month is not None and month >= GO_MONTH and low is not None and low > 0,
             "v": month is not None and f"{month:+.2f}% · {low:+.2f}%" if low is not None else (f"{month:+.2f}%" if month is not None else "–")},
            {"k": "weeks", "ok": sum(1 for w in weeks if w["done"] and (w["pct"] or 0) > 0) >= GO_WEEKS
             and all((w["pct"] or 0) >= GO_WORST_WEEK for w in weeks if w["done"]),
             "v": f"{sum(1 for w in weeks if w['done'] and (w['pct'] or 0) > 0)}/{sum(1 for w in weeks if w['done'])}"},
            {"k": "adverse", "ok": bool(m10) and avg(m10) >= 0, "v": m10 and f"{avg(m10) * 100:+.2f}¢ · {avg(edge) * 100:+.2f}¢"},
            {"k": "worst", "ok": worst is None or worst >= -GO_FILL_DAYS * days_reward,
             "v": worst is not None and f"{'-' if worst < 0 else ''}${abs(worst):,.0f} · ${days_reward:,.0f}"},
            {"k": "contested", "ok": rewards > 0 and contested / rewards >= GO_CONTESTED,
             "v": f"{contested / rewards * 100:.0f}%" if rewards else "–"},
        ]
        verdict = None
        if done:
            verdict = "go" if all(c["ok"] for c in checks) else "nogo"
        recent = [{"ts": f[0], "q": f[12], "side": f[2], "px": f[3], "sh": round(f[4], 1), "pnl": round(per_fill[i], 2),
                   "m10": None if f[7] is None else round(f[7], 2), "resolved": f[11] is not None} for i, f in enumerate(fills)][-8:][::-1]
        st = self.status["mm"]
        return {"markets": len(self.mm), "quoting": sum(1 for s in self.mm.values() if s["quote"]), "pools": round(sum(s["m"]["rate"] for s in self.mm.values()), 0),
                "fills": len(fills), "resolved": resolved, "market_days": round(mdays, 1), "rewards": round(rewards, 2), "marks": round(marks, 2),
                "rebates": round(rebates, 2), "net": round(net, 2), "per_mday": per_mday and round(per_mday, 2), "month": month and round(month, 2),
                "month_low": low and round(low, 2), "weeks": weeks, "checks": checks, "verdict": verdict, "recent": recent,
                "last": st.get("last"), "error": st.get("error") if (st.get("error_at") or 0) > (st.get("last") or 0) else None,
                "ws": st.get("ws"), "rules": {"capital": MM_CAPITAL, "min_rate": MM_MIN_RATE, "min_days": MM_MIN_DAYS, "max": MM_MAX,
                                              "exposure": MM_EXPOSURE}}

    def ladder_view(self, since, now):
        rows = self.db.execute("SELECT ts, cur, strike, side, price, size, model, gap, hours, fee, outcome, expiry FROM ld_signals ORDER BY ts").fetchall()
        days = self.db.execute("SELECT day, scans, strikes, max_gap, max_exec FROM ld_days ORDER BY day").fetchall()
        settled = [r for r in rows if r[10] is not None]
        pnl = [ladder_pnl(r[3], r[4], r[9], r[10]) for r in settled]
        mean = sum(pnl) / len(pnl) * 100 if pnl else None
        t = t_stat(pnl)
        span = max(1e-9, (now - since) / 86400) if since else 1
        per_day = len(rows) / span
        weeks = []
        if since:
            for w in range(4):
                a, b = since + w * WEEK, since + (w + 1) * WEEK
                if a > now:
                    break
                xs = [ladder_pnl(r[3], r[4], r[9], r[10]) for r in settled if a <= r[0] < b]
                weeks.append({"n": w + 1, "n_sig": len(xs), "cents": round(sum(xs) / len(xs) * 100, 2) if xs else None,
                              "done": now >= b})
        split = {}
        for name, test in (("near", lambda h: h < 12), ("far", lambda h: h >= 12)):
            xs = [ladder_pnl(r[3], r[4], r[9], r[10]) for r in settled if test(r[8])]
            split[name] = {"n": len(xs), "cents": round(sum(xs) / len(xs) * 100, 2) if xs else None}
        done = since and now - since >= DAYS * 86400
        wk_pos = sum(1 for w in weeks if w["done"] and (w["cents"] or 0) > 0)
        checks = [
            {"k": "mean", "ok": mean is not None and mean >= LADDER_GO_CENTS, "v": mean is not None and f"{mean:+.2f}¢"},
            {"k": "t", "ok": t is not None and t >= LADDER_GO_T, "v": t is not None and f"{t:.2f}"},
            {"k": "weeks", "ok": wk_pos >= GO_WEEKS, "v": f"{wk_pos}/{sum(1 for w in weeks if w['done'])}"},
            {"k": "often", "ok": per_day >= LADDER_NOGO_PER_DAY, "v": f"{per_day:.1f}"},
        ]
        verdict = None
        if done:
            if (mean is not None and mean < LADDER_NOGO_CENTS) or per_day < LADDER_NOGO_PER_DAY or mean is None:
                verdict = "nogo"
            elif all(c["ok"] for c in checks):
                verdict = "go"
            else:
                verdict = "unclear"
        recent = [{"ts": r[0], "cur": r[1], "strike": r[2], "side": r[3], "price": r[4], "size": r[5] and round(r[5], 0),
                   "model": round(r[6], 3), "gap": round(r[7] * 100, 1), "hours": round(r[8], 1),
                   "pnl": None if r[10] is None else round(ladder_pnl(r[3], r[4], r[9], r[10]) * 100, 1)} for r in rows][-10:][::-1]
        st = self.status["ladder"]
        return {"signals": len(rows), "settled": len(settled), "open": len(rows) - len(settled), "mean_cents": mean and round(mean, 2),
                "t": t and round(t, 2), "per_day": round(per_day, 2),
                "dollars": round(sum(x * LADDER_STAKE / r[4] for x, r in zip(pnl, settled)), 2) if pnl else 0.0,
                "weeks": weeks, "split": split, "checks": checks, "verdict": verdict, "recent": recent,
                "today": days and {"day": days[-1][0], "scans": days[-1][1], "max_gap": round(days[-1][3] * 100, 1),
                                   "max_exec": None if days[-1][4] is None or days[-1][4] < -0.5 else round(days[-1][4] * 100, 1)},
                "last": st.get("last"), "error": st.get("error") if (st.get("error_at") or 0) > (st.get("last") or 0) else None,
                "rules": {"gap": LADDER_GAP, "stake": LADDER_STAKE, "go_cents": LADDER_GO_CENTS, "go_t": LADDER_GO_T,
                          "nogo_cents": LADDER_NOGO_CENTS, "nogo_per_day": LADDER_NOGO_PER_DAY}}


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
