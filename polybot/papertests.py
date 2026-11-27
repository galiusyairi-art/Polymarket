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

The real trial ("ניסוי אמת"; the owner, 23.11.2026, of the maker that steps aside, six days in and +30% a month on
paper: "how can we build and test it, if it really is as written there"). The paper's profit is its reward estimate
(our score over the book's, times the pool) and nothing had ever set that beside what Polymarket pays. A maker
research still investigating can be made a trial by his click: the same rule with a small quote (PILOT_STAKE) in the
few markets where the paper earned the most reward (PILOT_MARKETS), real orders only when he switches it on, as for
a winner. Each minute its orders rested is scored the paper's way on the real book (our own orders taken out of the
others': live_share), and beside it stand Polymarket's own numbers: the share it says we hold (every SHARE_EVERY)
and what it paid for each day (every PAID_EVERY). A loss past PILOT_STOP of the money quoted ends it by itself.
"""

import asyncio
import contextlib
import gc
import datetime as dt
import collections
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
               ("step", 0.5, 600, "זז הצידה 10 דק' אחרי עסקה"), ("farstep", 0.9, 600, "קצה רחוק וזז הצידה"),
               ("own", 0.5, 600, "זז הצידה רק אחרי מילוי שלנו"), ("oneside", 0.5, 600, "רק הצד שמולא זז הצידה"),
               ("calm", 0.5, 600, "זז אחרי מילוי שלנו וחוזר כשהמחיר נרגע"),
               ("safe", 0.9, 0, "בטוחה: רחוק מהמחיר, נועלת כל מילוי"))
# The three after them (the owner, 26.11.2026, of the real "step" after its first days: "there is something here, and I
# think it can be made more precise and improved ... you have my approval to do what you see fit to start"). Three
# independent reviews found that "step" steps aside after any trade in the market, anyone's, not only after a fill of
# ours: in the busy markets it is out of the book much of the day, which is reward lost (its counted reward ran at about
# half of what the book alone would pay). Whether that also spares it fills is the question, so each of these is the
# same quote, halfway out in the band, stepping aside only after a fill of its own:
#   own: both sides out for the cool's seconds;
#   oneside: only the side that was filled out for them, the other side quoting on (one side alone still earns a third
#     of the score between 10c and 90c);
#   calm: both sides out until the middle has stood still for MM_CALM reads in a row, the cool at most.
# The lab only: a real trial of one of them needs its live side built first (winner_set refuses it).
MM_AFTER = {"own": "both", "oneside": "side", "calm": "calm"}
MM_CALM = 2

# The safe way (the owner, 27.11.2026: "we are filling up with fills, nothing protects us from it ... I don't mind losing
# more [of the reward]: I am not after losing less but after making this strategy as safe as possible"). A fill held to
# its market's end can lose all that was paid for it: what makes it safe is not holding fills to the end. So:
#   far from the middle (SAFE_FRAC of the band): fewer fills, a smaller reward;
#   every fill locked: once Yes is bought, only No is bid, at a price that makes the pair cost under $1 by SAFE_MARGIN
#     (Yes + No pay $1 at the end whatever happens), and nothing new in that market till the pair is whole; the mirror
#     for No;
#   calm markets only: SAFE_DAYS_LEFT days or more to the end, no sports, at most SAFE_MAX_PRINTS trades in a day;
#   at most SAFE_MAX_OPEN markets unlocked at a time, and SAFE_DAY_FILLS fills a day; past them, only the locks.
# In the lab beside the others; real money by the owner's button once the lab shows it safer (safe_proof).
SAFE_FRAC = 0.9
SAFE_MARGIN = 0.01
SAFE_DAYS_LEFT = 7
SAFE_MAX_PRINTS = 40  # (a floor: the calmer SAFE_CALM_SHARE of the markets followed are calm too, however busy)
SAFE_CALM_SHARE = 0.6
SAFE_MAX_OPEN = 3
SAFE_DAY_FILLS = 6
SAFE_PROOF_DAYS = 5

# "Improve the one that runs" (the owner, 27.11.2026: "a card in the control centre with a button: push and improve the
# existing strategy; a click improves it by what we found ... call it 'improving the existing one'"). The click is his,
# on real money: the live maker keeps its money, markets and switch, and quotes on, improved:
#   the way of stepping aside: the best of "step" and the three that step aside after their own fill only, by their net
#     in the lab over the same days (reward + what the fills are worth + rebates), once IMPROVE_MIN_DAYS of them; till
#     then "step" as it is;
#   the markets: by their net over the last PILOT_RANK_DAYS in the lab ("step"'s fills and reward there), not by the
#     reward alone, and one whose net there is below zero is let go;
#   a market whose real fills lost IMPROVE_BAD_FILLS times in the last IMPROVE_BAD_HOURS hours rests that long;
#   no quote in a market's last IMPROVE_PULL_DAYS days (MM_PULL is one): the end brings the news.
# A second click brings it back as it was.
IMPROVE_WAYS = ("step", "own", "oneside", "calm")
IMPROVE_MIN_DAYS = 2.0
IMPROVE_PULL_DAYS = 3
IMPROVE_BAD_FILLS = 2
IMPROVE_BAD_HOURS = 24
IMPROVE_WIN_SHARE = 0.55
IMPROVE_JUDGE_HOURS = 48
LIVE_MM_MARKETS = 10  # a live maker winner quotes the largest pools only
LIVE_REJ_REST = 600  # seconds an order Polymarket refused isn't sent again (no money for it, say: not a refusal a minute)
# A refusal that is the moment's, not the order's (the owner, 27.11.2026: Polymarket's outage of 15:07-15:21 UTC refused
# every order with "trading is disabled", "order manager not ready", "internal server error", and each market then waited
# its ten minutes, out of the book and earning nothing, "that mustn't happen"; so does a post-only order the price moved
# across): tried again after a minute.
LIVE_REJ_BRIEF = 60
LIVE_REJ_PASSING = ("trading is disabled", "not ready", "internal server error", "please retry", "crosses", "cloudflare", "timed out",
                    "timeout", "502", "503", "504", "service unavailable", "too many requests", "rate limit")
# A market whose reward Polymarket counts far under ours (the owner, 27.11.2026: Canada's unemployment, $10.10 counted by us
# against $1.87 by Polymarket that day): ranked by what it really pays, and let go under MM_OVERRATED of ours, once
# MM_OVERRATED_MIN dollars of ours are in to judge by.
MM_OVERRATED = 0.4
# Never in the book for nothing (the owner, 27.11.2026: "I never want rewards to stop while the fills go on"): an order
# that rests and earns no reward (under the market's least size after a partial fill, out of the reward's band, one side
# alone where the price is under 10c or over 90c) can still be filled. A market whose orders earned nothing for
# MM_ZERO_MINUTES minutes in a row is taken out of the book for MM_ZERO_PAUSE seconds.
MM_ZERO_MINUTES = 2
MM_ZERO_PAUSE = 1800
MM_OVERRATED_MIN = 2.0
PILOT_STAKE = 100.0  # a real trial's quote in each market, $
PILOT_MARKETS = 5  # the markets it quotes in: those where the paper's way earned the most reward in PILOT_RANK_DAYS
PILOT_RANK_DAYS = 3
PILOT_STOP = 0.15  # of the money quoted (the quote times the markets): the real loss (fills, less rewards paid) that ends it
PILOT_DAYS = 7  # the days it is judged on
PILOT_RATIO = 0.7  # paid of predicted, at least, for the paper's rewards to be called real
SHARE_EVERY = 300  # seconds between reads of the share Polymarket says our orders hold
PAID_EVERY = 600  # and of what it paid (an hour it was; the owner, 25.11.2026, wants each rise of it told as it comes)
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
CREATE TABLE IF NOT EXISTS live_events (test TEXT, ts REAL, kind TEXT, amount REAL);
CREATE TABLE IF NOT EXISTS live_hour (test TEXT, hour REAL, minutes REAL, predicted REAL, PRIMARY KEY (test, hour));
CREATE TABLE IF NOT EXISTS mm_prints (cid TEXT, day TEXT, n INTEGER, vol REAL, PRIMARY KEY (cid, day));
CREATE TABLE IF NOT EXISTS live_trades (id INTEGER PRIMARY KEY, ts REAL, test TEXT, key TEXT, detail TEXT, cost REAL, status TEXT,
    error TEXT);
CREATE TABLE IF NOT EXISTS live_rew (test TEXT, cid TEXT, day TEXT, minutes REAL, predicted REAL, mine REAL, theirs REAL, n INTEGER,
    paid REAL, PRIMARY KEY (test, cid, day));
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
    return {"mid": mid, "bb": bb, "ba": ba, "others": qmin(q1, q2, mid), "q1": q1, "q2": q2}


def live_share(st, v, min_size, orders):
    """Our real resting orders' share of a market's reward, by the same scoring, on a book that holds them: st is that
    book's state (book_state), orders [(Yes price, shares left, "bid" or "ask")] (a No buy is Yes's ask at one minus
    its price). Ours are scored, taken out of each side's total, and what is left is the others'. An order under the
    reward's least size scores nothing."""
    mid, m1, m2 = st["mid"], 0.0, 0.0
    for px, sh, side in orders:
        d = mid - px if side == "bid" else px - mid
        if sh + 1e-9 < min_size or not (0 <= d < v):
            continue
        sc = ((v - d) / v) ** 2 * sh
        m1, m2 = (m1 + sc, m2) if side == "bid" else (m1, m2 + sc)
    mine = qmin(m1, m2, mid)
    others = qmin(max(0.0, st["q1"] - m1), max(0.0, st["q2"] - m2), mid)
    return mine / (mine + others) if mine + others > 0 else 0.0


def _day(ts):
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


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
    return {"bid": bid, "ask": ask, "nb": nb, "na": na, "share": share, "mid": mid, "m1": m1, "m2": m2, "others": others,
            "qa_bid": sum(s for p, s in bids if abs(p - bid) < 1e-9), "qa_ask": sum(s for p, s in asks if abs(p - ask) < 1e-9)}


def lock_quote(mid, v, tick, others, net, avg, asks_or_bids, margin=SAFE_MARGIN):
    """The quote that locks a holding: net > 0 (Yes bought at avg): an ask (No bought) at avg + margin or above, for the
    shares held; net < 0 (No bought, Yes sold at avg): a bid at avg - margin or below. Scored as one side alone. None
    when the price it needs is out of the book's range."""
    if net > 0:
        ask = round(math.ceil((avg + margin) / tick - 1e-9) * tick, 6)
        ask = max(ask, round(math.ceil((mid + tick) / tick - 1e-9) * tick, 6))
        if ask >= 0.999:
            return None
        d, n = ask - mid, net
        m2 = ((v - d) / v) ** 2 * n if 0 <= d < v else 0.0
        mine = qmin(0.0, m2, mid)
        return {"bid": 0.0, "ask": ask, "nb": 0.0, "na": n, "share": mine / (mine + others) if mine + others > 0 else 0.0, "mid": mid,
                "m1": 0.0, "m2": m2, "others": others, "qa_bid": 0.0, "qa_ask": sum(sz for p, sz in asks_or_bids if abs(p - ask) < 1e-9),
                "lock": True}
    bid = round(math.floor((avg - margin) / tick + 1e-9) * tick, 6)
    bid = min(bid, round(math.floor((mid - tick) / tick + 1e-9) * tick, 6))
    if bid <= 0.001:
        return None
    d, n = mid - bid, -net
    m1 = ((v - d) / v) ** 2 * n if 0 <= d < v else 0.0
    mine = qmin(m1, 0.0, mid)
    return {"bid": bid, "ask": 1.0, "nb": n, "na": 0.0, "share": mine / (mine + others) if mine + others > 0 else 0.0, "mid": mid,
            "m1": m1, "m2": 0.0, "others": others, "qa_bid": sum(sz for p, sz in asks_or_bids if abs(p - bid) < 1e-9), "qa_ask": 0.0,
            "lock": True}


def one_side_out(q, side):
    """The quote q with one side taken out ("buy": our bid, "sell": our ask), its share scored again; None if nothing
    of it is left to score."""
    if q is None:
        return None
    q = dict(q, **({"nb": 0.0, "m1": 0.0} if side == "buy" else {"na": 0.0, "m2": 0.0}))
    if q["nb"] <= 0 and q["na"] <= 0:
        return None
    mine = qmin(q["m1"], q["m2"], q["mid"])
    q["share"] = mine / (mine + q["others"]) if mine + q["others"] > 0 else 0.0
    return q


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


def exit_pnl(side, px, sh, bb, ba, fee=0.0):
    """What getting out of a fill at once would make: the shares sold at the price one can really sell at, the taker's
    fee paid. A Yes fill (side "buy", bought at px) is sold at the best bid; a No fill (side "sell": No bought at
    1 - px) is sold at No's best bid, one minus Yes's best ask. (fill_pnl marks a fill at the middle, which nobody
    can sell at: the difference is half the spread, each time.)"""
    out = bb if side == "buy" else ba
    return ((out - px) if side == "buy" else (px - out)) * sh - (fee or 0.0) * out * (1 - out) * sh


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
TITLES = {"mm:safe": "עושה שוק · הגרסה הבטוחה", "mm:own": "עושה שוק · זז רק אחרי מילוי שלנו", "mm:oneside": "עושה שוק · רק הצד שמולא זז", "mm:calm": "עושה שוק · חוזר כשהמחיר נרגע",
          "mm:mid": "עושה שוק · באמצע הטווח", "mm:far": "עושה שוק · בקצה הרחוק", "mm:step": "עושה שוק · זז הצידה אחרי עסקה",
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
                    "ALTER TABLE ld_signals ADD COLUMN token TEXT",
                    # (1.427: what selling a fill would really have fetched, at each mark: see exit_pnl)
                    "ALTER TABLE mm_fills ADD COLUMN x1 REAL", "ALTER TABLE mm_fills ADD COLUMN x10 REAL",
                    "ALTER TABLE mm_fills ADD COLUMN x60 REAL"):
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
        self.live_pick = []  # the markets real money quotes in (kept from minute to minute: no hopping between markets)
        self.live_rej = {}  # (cid, side) -> when Polymarket last refused that order
        self.live_lock = threading.Lock()  # (an order's fills are counted once: the minute's check and a step aside can meet)
        self._live_shares = {}  # cid -> our share by our own count, this minute (set beside Polymarket's word on it)
        self._load()
        # (the orders resting when the machine went down are still in Polymarket's books: they are ours again)
        kept = self.meta("live_state") or {}
        self.live_orders = {(c, side): o for c, side, o in kept.get("orders") or []}
        self.live_pick = list(kept.get("pick") or [])
        self._live_saved = json.dumps(kept, sort_keys=True)

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
            self._mem_load(now)
            self.ws_assets = self._assets()

    # The short memory (the owner, 27.11.2026, of each new version's restart: "keep the memory in the database"): when each
    # market last traded (the step aside's ten minutes), each way's last fill of its own there (its own stepping aside), the
    # markets out of the book for earning nothing and when they come back, the orders Polymarket refused and when they are
    # tried again, and each market's trades in the last day (the safe way's calm markets). Until then a restart forgot them
    # all: a market traded a minute before was quoted again at once.
    MEM_KEY = "mm_memory"

    def _mem_save(self, now):
        mem = {"ts": now, "m": {}, "pause": {c: t for c, t in (self.__dict__.get("_live_pause") or {}).items() if t > now},
               "rej": [[k[0], k[1], t] for k, t in self.live_rej.items() if now - t < LIVE_REJ_REST]}
        for cid, st in self.mm.items():
            x = {}
            if st.get("last_print") and now - st["last_print"] < 3600:
                x["lp"] = st["last_print"]
            v = {var: {k: V[k] for k in ("fill_t", "fill_side", "steady") if V.get(k) is not None}
                 for var, V in st["v"].items() if V.get("fill_t") and now - V["fill_t"] < 3600}
            if v:
                x["v"] = v
            if x:
                mem["m"][cid] = x
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (self.MEM_KEY, json.dumps(mem)))

    def _mem_load(self, now):
        mem = self.meta(self.MEM_KEY) or {}
        for cid, x in (mem.get("m") or {}).items():
            st = self.mm.get(cid)
            if not st:
                continue
            st["last_print"] = x.get("lp")
            for var, d in (x.get("v") or {}).items():
                if var in st["v"]:
                    st["v"][var].update(d)
        self._live_pause = {c: t for c, t in (mem.get("pause") or {}).items() if t > now}
        self.live_rej.update({(a, b): t for a, b, t in (mem.get("rej") or [])})
        # (the last day's trades of each market, from its count by the day: yesterday's spread over yesterday, today's over
        #  today so far; the oldest leave as the day goes on)
        day0 = _day_ts(_day(now))
        for cid, day, n in self.db.execute("SELECT cid, day, n FROM mm_prints WHERE day >= ?", (_day(now - 86400),)).fetchall():
            st = self.mm.get(cid)
            if not st or not n:
                continue
            a, b = (day0 - 86400, day0) if day < _day(now) else (day0, now)
            k = min(int(n), 2000)
            pt = st.setdefault("prints", collections.deque(maxlen=2000))
            pt.extend(a + (b - a) * (i + 0.5) / k for i in range(k))
            st["prints"] = collections.deque(sorted(pt), maxlen=2000)

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

    def create_winner(self, key, now=None, pilot=False):
        """The button "create the strategy": the research's rule exactly, frozen, into the simulation from now (its own
        record from zero). Only once the research found it. pilot: a real trial of a maker research still
        investigating (see the module's last paragraph): the same rule, small, off until the owner switches it on."""
        now = now or time.time()
        if key not in KEYS:
            return {"error": "אין מחקר כזה"}
        status = self.research_status(key, now)["status"]
        W = self.winners()
        if pilot:
            if not key.startswith("mm:"):
                return {"error": "ניסוי אמת יש רק לעושה השוק"}
            if status == "not_found":
                return {"error": "המחקר הסתיים בלי למצוא: אין מה לנסות בכסף אמיתי"}
            if key not in W:
                W[key] = {"created": now, "live": False, "pilot": True, "stake": PILOT_STAKE, "markets": PILOT_MARKETS,
                          "name": "ניסוי אמת · " + TITLES[key]}
                self._save_winners(W)
            return {"ok": True, "winner": W[key]}
        if status != "found":
            return {"error": "המחקר עוד לא מצא: האסטרטגיה נוצרת רק ממה שנמצא מרוויח"}
        if key not in W:
            W[key] = {"created": now, "live": False, "stake": STAKES["mm" if key.startswith("mm:") else key], "name": self.winner_name(key)}
            self._save_winners(W)
        return {"ok": True, "winner": W[key]}

    @staticmethod
    def winner_name(key):
        return WINNER_NAMES.get(key) or ("המנצחת · " + TITLES.get(key, key))

    def winner_set(self, key, live=None, stake=None, remove=False, now=None, markets=None):
        now = now or time.time()
        W = self.winners()
        if key not in W:
            return {"error": "האסטרטגיה לא קיימת"}
        if remove:
            if W[key].get("live"):
                return {"error": "קודם להעביר אותה מכסף אמיתי לסימולציה"}
            W.pop(key)
            self._save_winners(W)
            return {"ok": True, "winners": W}
        if stake is not None:
            W[key]["stake"] = max(1.0, min(5000.0, float(stake)))
        if markets is not None and key.startswith("mm:"):
            W[key]["markets"] = max(1, min(20, int(markets)))
        if live is not None:
            if live and not (self.trader and self.trader.configured()):
                return {"error": "הארנק של פולימרקט לא מחובר: אין לאן לשלוח הוראות"}
            if live and (key[3:] in MM_AFTER or key[3:] == "safe"):
                return {"error": "הגרסה הזו רצה רק במעבדה: לכסף אמיתי צריך קודם לבנות לה את הצד החי"}
            if live and key.startswith("mm:"):
                other = next((k for k in W if k != key and k.startswith("mm:") and W[k].get("live")), None)
                if other:
                    return {"error": f"\"{W[other].get('name') or other}\" כבר בכסף אמיתי: עושה שוק אחד בכל פעם"}
                # (the quotes rest in the books and hold their money: without it Polymarket refuses them, and the
                #  other real strategies' orders with them)
                need = float(W[key]["stake"]) * int(W[key].get("markets") or LIVE_MM_MARKETS)
                read = getattr(self.trader, "balance", None)
                have = read() if read else None
                if have is not None and have < need:
                    return {"error": f"צריך ${need:,.0f} פנויים בארנק (${W[key]['stake']:,.0f} לכל שוק, {int(W[key].get('markets') or LIVE_MM_MARKETS)} שווקים), "
                                     f"ויש ${have:,.2f}. אפשר להקטין את הסכום לשוק או את מספר השווקים"}
                W[key].pop("ended", None)
            W[key]["live"] = bool(live)
            W[key]["live_since" if live else "live_until"] = now
            if live and not W[key].get("first_live"):
                W[key]["first_live"] = now
        self._save_winners(W)
        if live is False and key.startswith("mm:"):
            self._live_cancel_all(now)
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
            safe_open = sum(1 for x in self.mm.values() if abs((x["v"].get("safe") or {}).get("net") or 0.0) > 1e-6)
            self._safe_calm_cut(now)
            safe_day = self.db.execute("SELECT COUNT(*) FROM mm_fills WHERE var='safe' AND ts>=?", (_day_ts(day),)).fetchone()[0]
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
                s["st"] = st
                s["tick"] = tick
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
                                self.db.execute(f"UPDATE mm_fills SET {k}=?, x{k[1:]}=? WHERE id=?",
                                                (f[k], exit_pnl(f["side"], f["px"], f["sh"], st["bb"], st["ba"], m["fee"]), f["id"]))
                pulled = (m["rate"] <= 0 or now >= m["end_ts"] - MM_PULL or (prev is not None and abs(mid - prev) >= MM_MOVE - 1e-9))
                for var, frac, cool, _ in MM_VARIANTS:
                    V = s["v"][var]
                    if var == "safe":
                        V["quote"] = self._safe_quote(s, V, mid, tick, st["others"], bids, asks, now, safe_open, safe_day, pulled)
                        continue
                    how = MM_AFTER.get(var)
                    if how:  # (after a fill of its own only)
                        ft = V.get("fill_t")
                        recent = bool(ft is not None and now - ft < cool)
                        if how == "calm" and ft is not None and prev is not None:
                            V["steady"] = (V.get("steady") or 0) + 1 if abs(mid - prev) < 1e-9 else 0
                        away = recent and (how == "both" or (how == "calm" and (V.get("steady") or 0) < MM_CALM))
                    else:
                        away = cool and s["last_print"] is not None and now - s["last_print"] < cool
                    V["quote"] = None if pulled or away else quote(mid, m["v"], tick, st["others"], V["net"], bids, asks, frac)
                    if how == "side" and recent and V["quote"]:
                        V["quote"] = one_side_out(V["quote"], V.get("fill_side"))
            self.pending = [f for f in self.pending if f["m60"] is None and now < f["t"] + 3600 + 300]
            self._mem_save(now)
            self.db.commit()
        live = [k for k in self.winners() if k.startswith("mm:") and self.winner_live(k)]
        try:
            if live:
                self._live_maker(live[0][3:], now)
            elif self.live_orders and self.trader:  # (orders of a maker no longer live: nothing of ours stays in the books)
                self._live_cancel_all(now)
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
                if var == "safe":  # (the price the holding was bought at, for its lock)
                    d = n if ours == "buy" else -n
                    before = V["net"] - d
                    if abs(before) < 1e-6 or before * d > 0:
                        V["avg"] = (abs(before) * (V.get("avg") or px) + n * px) / (abs(before) + n)
                    elif abs(d) > abs(before) + 1e-6:
                        V["avg"] = px
                    if abs(V["net"]) < 1e-6:
                        V["avg"] = None
                    V["quote"] = None  # (the next read quotes its lock, or nothing)
                how = MM_AFTER.get(var)
                if how:  # (a fill of its own: out now, not at the next minute)
                    V.update(fill_t=ts, fill_side=ours, steady=0)
                    V["quote"] = one_side_out(V["quote"], ours) if how == "side" else None
            self.db.execute("INSERT INTO mm_prints VALUES (?,?,1,?) ON CONFLICT(cid, day) DO UPDATE SET n=n+1, vol=vol+excluded.vol",
                            (m["cid"], _day(ts), size * price))
            self.db.commit()
            s["last_print"] = ts
            pt = s.setdefault("prints", collections.deque(maxlen=2000))  # (a day's trades, for the safe way's calm markets)
            pt.append(ts)
            # (the ways that step aside after anyone's trade do it now, not at the next minute)
            for var, frac, cool, _ in MM_VARIANTS:
                if cool and var not in MM_AFTER:
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
        """The way of quoting, for real: in the markets picked (_live_pick), its quote of the minute as two post-only
        buys (Yes at our bid, No at one minus our ask), sized for the winner's money; moved when the quote moves,
        cancelled when it's pulled. What sellers took is recorded. The minute that passed is scored first (_live_score),
        and Polymarket's own numbers are read beside it (_live_reads)."""
        t = self.trader
        key = "mm:" + var
        W = self.winners().get(key)
        if not t or not W or not t.configured():
            return
        if self._live_held(W):  # (real money's own stop, made since this maker was switched on: nothing of ours stays in the books)
            self._live_cancel_all(now)
            return
        stake = float(W.get("stake") or STAKES["mm"])
        self._live_score(key, now)
        if W.get("pilot") and self._pilot_over(key, W, stake, now):
            return
        qv = self._live_way(var, W)
        with self.lock:
            want = {}
            picked = self._live_pick(var, W, stake, now)
            if W.get("safe"):
                want = self._live_safe_want(picked, stake, now, key)
                picked = []
            pause = self.__dict__.get("_live_pause") or {}
            for s in picked:
                q = s["v"][qv]["quote"]
                if not q or pause.get(s["m"]["cid"], 0) > now:
                    continue
                n = stake / (q["bid"] + 1 - q["ask"])
                if n + 1e-9 < s["m"]["min_size"]:
                    continue  # (under the reward's least size: an order that would earn nothing)
                net = self._live_net(s["m"]["cid"])
                mid = s["mid"] or 0.5
                if q["nb"] > 0 and not (net * mid >= MM_EXPOSURE * stake / MM_CAPITAL):
                    want[(s["m"]["cid"], "yes")] = (s["m"]["yes"], q["bid"], n)
                if q["na"] > 0 and not (-net * (1 - mid) >= MM_EXPOSURE * stake / MM_CAPITAL):
                    want[(s["m"]["cid"], "no")] = (s["m"]["no"], round(1 - q["ask"], 6), n)
                self._live_pair_cap(want, s, net)
        for k, o in list(self.live_orders.items()):
            self._live_check(k, o, now)
            if o.get("gone"):
                # (Polymarket took it out itself, a market that closed or a cancel from elsewhere: it rests no more, and
                #  until 25.11.2026 it stayed here as resting for good, earning nothing; the market gets a fresh one)
                self.live_orders.pop(k, None)
                continue
            w = want.get(k)
            if o.get("out") or o.get("refresh") or not w or abs(w[1] - o["price"]) > 1e-9 or o["matched"] >= o["shares"] - 0.01:
                self._live_drop(k, o, now)  # (moved, pulled, or taken whole: the next is a fresh one)
        for k, (token, price, n) in want.items():
            if k in self.live_orders or now - self.live_rej.get(k, 0) < LIVE_REJ_REST:
                continue
            oid, err = t.place_maker(token, price, n)
            if oid:
                self.live_orders[k] = {"id": oid, "price": price, "shares": n, "matched": 0.0, "token": token, "since": now, "test": key}
            else:
                passing = any(w in str(err or "").lower() for w in LIVE_REJ_PASSING)
                self.live_rej[k] = now - (LIVE_REJ_REST - LIVE_REJ_BRIEF if passing else 0)
                self._live_trade(now, key, k[0], {"side": k[1], "price": price, "shares": n}, 0.0, "rejected", err)
        self._live_save()
        self._live_reads(key, now)

    def _at_stake(self, stake=PILOT_STAKE):
        """The lab's fills and rewards at a real quote's money (the lab quotes $MM_CAPITAL a market): a fill took what a
        print brought past the queue ahead, so a smaller quote takes the same shares up to its own size (not a tenth of
        them: a tenth would understate the fills against the reward); the reward is a tenth of the lab's (at least what
        the smaller quote gets: alone in a band it gets all of it, so this side errs low). Returns (reward factor,
        fill shares(cid, shares))."""
        v = dict(self.db.execute("SELECT cid, v FROM mm_markets").fetchall())

        def shares(cid, sh):
            half = max(0.01, round((v.get(cid) or 0.04) * 0.5 / 0.01) * 0.01)
            return min(sh, stake / max(0.5, 1 - 2 * half))
        return stake / MM_CAPITAL, shares

    def _safe_ok(self, s, now):
        """A market calm enough for the safe way: far from its end, not sport, not busy."""
        m = s["m"]
        if now >= m["end_ts"] - SAFE_DAYS_LEFT * 86400 or "sport" in str(m.get("cat") or "").lower():
            return False
        return self._prints_day(s, now) <= max(SAFE_MAX_PRINTS, self.__dict__.get("_safe_cut") or 0)

    @staticmethod
    def _prints_day(s, now):
        return sum(1 for t in s.get("prints") or () if t >= now - 86400)

    def _safe_calm_cut(self, now):
        """The trades a day under which a market is among the calmer SAFE_CALM_SHARE of those followed (the owner, 27.11.2026:
        at 40 a day nearly every market was too busy, and the safe way was in the book 5% of the time)."""
        n = sorted(self._prints_day(s, now) for s in self.mm.values())
        self._safe_cut = n[min(len(n) - 1, int(len(n) * SAFE_CALM_SHARE))] if n else 0

    def _safe_avg(self, cid):
        """The safe way's holding in a market and the price it was bought at, from its fills (after a restart)."""
        net, avg = 0.0, None
        for side, px, sh in self.db.execute("SELECT side, px, sh FROM mm_fills WHERE var='safe' AND cid=? ORDER BY ts", (cid,)).fetchall():
            d = sh if side == "buy" else -sh
            if abs(net) < 1e-6 or net * d > 0:
                avg = (abs(net) * (avg or px) + sh * px) / (abs(net) + sh)
            elif abs(d) > abs(net) + 1e-6:
                avg = px
            net += d
            if abs(net) < 1e-6:
                avg = None
        return avg

    def _safe_quote(self, s, V, mid, tick, others, bids, asks, now, n_open, n_day, pulled):
        """The safe way's quote for the next minute: its holding locked first; else, in a calm market and under its caps,
        the far quote."""
        net = V["net"]
        if abs(net) > 1e-6:
            if V.get("avg") is None:
                V["avg"] = self._safe_avg(s["m"]["cid"])
            if V.get("avg") is None:
                return None
            return lock_quote(mid, s["m"]["v"], tick, others, net, V["avg"], asks if net > 0 else bids)
        if pulled or n_open >= SAFE_MAX_OPEN or n_day >= SAFE_DAY_FILLS or not self._safe_ok(s, now):
            return None
        return quote(mid, s["m"]["v"], tick, others, 0.0, bids, asks, SAFE_FRAC)

    SAFE_RULE = 2  # (2: the calmer share of the markets, 27.11.2026; its research starts again from it)

    def safe_compare(self, now=None):
        """The safe way against "step" (the way real money runs) in the lab, over the same whole days from the first
        midnight after the safe way began, at a real quote's money (_at_stake): the net, the worst day, and what is held
        unlocked now (bought and not locked: what could still be lost whole at its market's end). The proof, after
        SAFE_PROOF_DAYS days: the safe way makes money, its worst day is no worse than step's, and it holds less unlocked."""
        now = now or time.time()
        if (self.meta("safe_rule") or 1) < self.SAFE_RULE:
            self.set_meta("safe_rule", self.SAFE_RULE)
            self.set_meta("started_mm:safe", now)
        frm = self.started("mm:safe")
        t0 = _day_ts(_day(frm)) + (0 if frm == _day_ts(_day(frm)) else 86400)
        days = max(0.0, (now - t0) / 86400)
        f, shares = self._at_stake()
        mids = {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT cid, mid, outcome FROM mm_markets").fetchall()}
        out = {}
        for w in ("step", "safe"):
            by_day = {}
            for day, r_ in self.db.execute("SELECT day, SUM(reward) FROM mm_rew WHERE var=? AND day>=? GROUP BY day", (w, _day(t0))).fetchall():
                by_day[day] = by_day.get(day, 0.0) + (r_ or 0.0) * f
            n = locked = 0
            held = {}
            for ts, cid, side, px, sh, rb in self.db.execute("SELECT ts, cid, side, px, sh, rebate FROM mm_fills WHERE var=? AND ts>=? ORDER BY ts",
                                                            (w, t0)).fetchall():
                mid, outcome = mids.get(cid, (None, None))
                mark = outcome if outcome is not None else mid
                k = shares(cid, sh)
                x = (fill_pnl(side, px, k, mark) if mark is not None else 0.0) + (rb or 0.0) * (k / sh if sh else 0.0)
                by_day[_day(ts)] = by_day.get(_day(ts), 0.0) + x
                d = k if side == "buy" else -k
                h = held.get(cid, 0.0)
                locked += abs(h) > 1e-6 and h * d < 0
                held[cid] = h + d
                n += 1
            exposed = 0.0
            for cid, h in held.items():
                mid, outcome = mids.get(cid, (None, None))
                if outcome is None and abs(h) > 1e-6 and mid is not None:
                    exposed += h * mid if h > 0 else -h * (1 - mid)
            today = _day(now)
            whole = [v for d, v in by_day.items() if d < today]
            net = sum(by_day.values())
            out[w] = {"net": round(net, 2), "per_day": round(net / days, 2) if days >= 0.25 else None, "fills": n, "locks": int(locked),
                      "worst": round(min(whole), 2) if whole else None, "exposed": round(exposed, 2),
                      "days": {d: round(v, 2) for d, v in sorted(by_day.items())}}
        S, T = out["safe"], out["step"]
        checks = [{"k": "money", "ok": S["net"] > 0, "say": "מרוויחה (נטו מעל 0)"},
                  {"k": "worst", "ok": S["worst"] is not None and (T["worst"] is None or S["worst"] >= T["worst"]), "say": "היום הכי גרוע שלה לא גרוע מזה של הקיימת"},
                  {"k": "exposed", "ok": S["exposed"] <= T["exposed"], "say": "מחזיקה פחות מניות לא נעולות מהקיימת"}]
        ready = days >= SAFE_PROOF_DAYS
        return {"from": t0, "days": round(days, 2), "need": SAFE_PROOF_DAYS, "ready": ready, "ways": out, "checks": checks,
                "proven": ready and all(c["ok"] for c in checks)}

    def safe_view(self, now=None):
        """The safe way's research card in the maker's tab, and whether real money runs it."""
        now = now or time.time()
        key = self._improve_key()
        W = self.winners().get(key) or {}
        return {"key": key, "on": bool(W.get("safe")), "safe": W.get("safe"), "live": bool(W.get("live")), "compare": self.safe_compare(now),
                "rules": {"frac": SAFE_FRAC, "margin": SAFE_MARGIN, "days_left": SAFE_DAYS_LEFT, "prints": SAFE_MAX_PRINTS,
                          "open": SAFE_MAX_OPEN, "day_fills": SAFE_DAY_FILLS}}

    def safe_set(self, on, now=None):
        """The owner's click: real money's maker to the safe way (only once the lab proved it), or back."""
        now = now or time.time()
        key = self._improve_key()
        if not key:
            return {"error": "אין עושה שוק קיים: קודם פותחים ניסוי אמת"}
        Ws = self.winners()
        if on:
            C = self.safe_compare(now)
            if not C["proven"]:
                return {"error": f"עוד אין הוכחה: {C['days']} ימים מתוך {C['need']}" if not C["ready"] else "המעבדה לא הראתה שהיא בטוחה יותר"}
            Ws[key]["safe"] = {"since": now}
        else:
            Ws[key].pop("safe", None)
        self._save_winners(Ws)
        self.__dict__.pop("_view_hit", None)
        log.warning("MAKER SAFE %s", "ON" if on else "OFF")
        return dict(self.safe_view(now), ok=True)

    def _live_avg(self, cid):
        """Real money's price for what it holds in a market (Yes's price; a No holding's as one minus it)."""
        net, avg = 0.0, None
        for d, in self.db.execute("SELECT detail FROM live_trades WHERE key=? AND status='filled' ORDER BY ts", (cid,)).fetchall():
            x = json.loads(d)
            sh, px = float(x.get("shares") or 0), float(x.get("price") or 0)
            dd = sh if x.get("side") == "buy" else -sh
            if abs(net) < 1e-6 or net * dd > 0:
                avg = (abs(net) * (avg or px) + sh * px) / (abs(net) + sh)
            elif abs(dd) > abs(net) + 1e-6:
                avg = px
            net += dd
            if abs(net) < 1e-6:
                avg = None
        return avg

    def _live_pair_cap(self, want, s, net):
        """A market we hold one side of: the other side's bid no higher than makes the pair cost under $1 by SAFE_MARGIN
        (the owner, 27.11.2026, of the Iran ceasefire: Yes bought at 78c and No at 25c, $1.03 for a pair that pays $1)."""
        if abs(net) < 1:
            return
        cid, tick = s["m"]["cid"], s.get("tick") or 0.01
        avg = self._live_avg(cid)
        if avg is None:
            return
        k = (cid, "no") if net > 0 else (cid, "yes")
        if k not in want:
            return
        token, price, n = want[k]
        cap = round(math.floor(((1 - avg if net > 0 else avg) - SAFE_MARGIN) / tick + 1e-9) * tick, 6)
        if cap <= 0.001:
            want.pop(k)
        elif price > cap:
            want[k] = (token, cap, n)

    def _live_paid_ratio(self, key):
        """Each market's reward as Polymarket counted it against ours (paid / predicted), where enough of ours is in."""
        out = {}
        for cid, pred, paid in self.db.execute("SELECT cid, SUM(predicted), SUM(COALESCE(paid, 0)) FROM live_rew WHERE test=? "
                                               "AND paid IS NOT NULL GROUP BY cid", (key,)).fetchall():
            if (pred or 0) >= MM_OVERRATED_MIN:
                out[cid] = (paid or 0) / pred
        return out

    def _live_safe_want(self, picked, stake, now, key):
        """Real money in the safe way: each market's holding locked (a No bid under one minus what Yes cost, or the
        mirror, for the shares held); new quotes, far out, only in calm markets and under the caps."""
        want = {}
        n_open = sum(1 for c in set(self.live_pick) | {k[0] for k in self.live_orders} if abs(self._live_net(c)) >= 1)
        n_day = self.db.execute("SELECT COUNT(*) FROM live_trades WHERE test=? AND status='filled' AND ts>=?",
                                (key, _day_ts(_day(now)))).fetchone()[0]
        pause = self.__dict__.get("_live_pause") or {}
        for s in picked:
            m, cid, mid, st = s["m"], s["m"]["cid"], s.get("mid"), s.get("st")
            if mid is None or not st or (pause.get(cid, 0) > now and abs(self._live_net(cid)) < 1):
                continue
            tick = s.get("tick") or 0.01
            net = self._live_net(cid)
            if abs(net) >= 1:
                avg = self._live_avg(cid)
                q = avg is not None and lock_quote(mid, m["v"], tick, st["others"], net, avg, [])
                if not q:
                    continue
                if net > 0 and net >= 5:
                    want[(cid, "no")] = (m["no"], round(1 - q["ask"], 6), net)
                elif net < 0 and -net >= 5:
                    want[(cid, "yes")] = (m["yes"], q["bid"], -net)
                continue
            if n_open >= SAFE_MAX_OPEN or n_day >= SAFE_DAY_FILLS or now >= m["end_ts"] - MM_PULL or not self._safe_ok(s, now):
                continue
            q = quote(mid, m["v"], tick, st["others"], 0.0, [], [], SAFE_FRAC)
            if not q:
                continue
            n = stake / (q["bid"] + 1 - q["ask"])
            if n + 1e-9 < m["min_size"]:
                continue
            want[(cid, "yes")] = (m["yes"], q["bid"], n)
            want[(cid, "no")] = (m["no"], round(1 - q["ask"], 6), n)
        return want

    def improve_compare(self, now=None):
        """The four ways of stepping aside, in the lab over the same whole days (from the first midnight after the
        newest of them began: the reward is counted by the day, and a day begun before it would count for the old way
        only): the minutes in the book a market, the reward, what the fills are worth now (the result if the market
        ended, else the middle), the rebates, the net; at a real quote's money ($PILOT_STAKE a market, _at_stake). And
        market by market against "step": a way is better only if its net is higher and it is ahead in at least
        IMPROVE_WIN_SHARE of the markets (two days are few: one lucky market should not decide it)."""
        now = now or time.time()
        frm = max(self.started("mm:" + w) for w in IMPROVE_WAYS)
        t0 = _day_ts(_day(frm)) + (0 if frm == _day_ts(_day(frm)) else 86400)
        days = max(0.0, (now - t0) / 86400)
        f, shares = self._at_stake()
        mids = {r[0]: (r[1], r[2]) for r in self.db.execute("SELECT cid, mid, outcome FROM mm_markets").fetchall()}
        mkts = max(1, len(self.mm))
        out, per = [], {}
        for w in IMPROVE_WAYS:
            by = per.setdefault(w, {})
            mins = rew = 0.0
            for cid, m_, r_ in self.db.execute("SELECT cid, minutes, reward FROM mm_rew WHERE var=? AND day>=?", (w, _day(t0))).fetchall():
                mins += m_ or 0.0
                rew += (r_ or 0.0) * f
                by[cid] = by.get(cid, 0.0) + (r_ or 0.0) * f
            marks = reb = 0.0
            n = 0
            for cid, side, px, sh, rb in self.db.execute("SELECT cid, side, px, sh, rebate FROM mm_fills WHERE var=? AND ts>=?", (w, t0)).fetchall():
                mid, outcome = mids.get(cid, (None, None))
                mark = outcome if outcome is not None else mid
                k = shares(cid, sh)
                x = (fill_pnl(side, px, k, mark) if mark is not None else 0.0) + (rb or 0.0) * (k / sh if sh else 0.0)
                marks += x - (rb or 0.0) * (k / sh if sh else 0.0)
                reb += (rb or 0.0) * (k / sh if sh else 0.0)
                by[cid] = by.get(cid, 0.0) + x
                n += 1
            net = rew + marks + reb
            out.append({"way": w, "name": MM_NAMES[w], "fills": n, "reward": round(rew, 2), "marks": round(marks, 2),
                        "rebates": round(reb, 2), "net": round(net, 2),
                        "per_day": round(net / days, 2) if days >= 0.25 else None,
                        "in_book": round(min(1.0, mins / (mkts * (now - t0) / 60)), 3) if now - t0 >= 3600 else None})
        base = per.get("step", {})
        for x in out:
            if x["way"] == "step":
                continue
            mine = per[x["way"]]
            cids = [c for c in set(base) | set(mine) if abs(mine.get(c, 0.0) - base.get(c, 0.0)) > 1e-9]
            ahead = sum(1 for c in cids if mine.get(c, 0.0) > base.get(c, 0.0))
            x.update(ahead=ahead, markets=len(cids))
        ready = days >= IMPROVE_MIN_DAYS
        step = next(x for x in out if x["way"] == "step")
        good = [x for x in out if x["way"] != "step" and x["net"] > step["net"] and x.get("markets")
                and x["ahead"] / x["markets"] >= IMPROVE_WIN_SHARE]
        best = (max(good, key=lambda x: x["net"])["way"] if good else "step") if ready else None
        return {"from": t0, "days": round(days, 2), "ways": out, "best": best, "ready": ready, "min_days": IMPROVE_MIN_DAYS,
                "win_share": IMPROVE_WIN_SHARE}

    def _improve_key(self):
        """The live maker the button improves: the one switched to real money, else the one made for it."""
        W = self.winners()
        mm = [k for k in W if k.startswith("mm:")]
        return next((k for k in mm if W[k].get("live")), None) or next((k for k in mm if W[k].get("pilot")), None) or (mm[0] if mm else None)

    def improve_effect(self, key, W, now):
        """Did the click improve it, in real money (the owner, 27.11.2026, having clicked: "check whether it really
        improved")? A day of it before the click (the whole days since it began) against the time since, each a day:
        the reward counted, what its fills are worth now, the net, and the share of the time in the book. Judged once
        IMPROVE_JUDGE_HOURS of it are in."""
        imp = (W or {}).get("improve")
        if not imp:
            return None
        since, n = imp["since"], int(W.get("markets") or LIVE_MM_MARKETS)
        first = W.get("first_live") or since
        d0 = _day_ts(_day(first)) + (0 if first == _day_ts(_day(first)) else 86400)
        d1 = _day_ts(_day(since))
        mids = {c: (s.get("mid")) for c, s in self.mm.items()}

        def fills(a, b):
            out = 0.0
            for cid, detail in self.db.execute("SELECT key, detail FROM live_trades WHERE test=? AND status='filled' AND ts>=? AND ts<?",
                                               (key, a, b)).fetchall():
                x, mid = json.loads(detail), mids.get(cid)
                if mid is not None:
                    yes = x.get("side") in ("buy", "yes")
                    out += fill_pnl("buy" if yes else "sell", float(x.get("price") or 0), float(x.get("shares") or 0), mid)
            return out
        before = None
        if d1 > d0:
            days = (d1 - d0) / 86400
            rew, mins = self.db.execute("SELECT COALESCE(SUM(predicted), 0), COALESCE(SUM(minutes), 0) FROM live_rew WHERE test=? AND day>=? AND day<?",
                                        (key, _day(d0), _day(d1))).fetchone()
            f = fills(d0, d1)
            before = {"days": round(days, 2), "reward": round(rew / days, 2), "fills": round(f / days, 2), "net": round((rew + f) / days, 2),
                      "in_book": round(min(1.0, mins / (n * 1440 * days)), 3)}
        a = max(since, (self.db.execute("SELECT MIN(hour) FROM live_hour WHERE test=? AND hour>=?", (key, since // 3600 * 3600)).fetchone()[0] or now))
        rew, mins = self.db.execute("SELECT COALESCE(SUM(predicted), 0), COALESCE(SUM(minutes), 0) FROM live_hour WHERE test=? AND hour>=?",
                                    (key, a // 3600 * 3600)).fetchone()
        hours = (now - since) / 3600
        span = max(1.0, now - a) / 86400
        f = fills(since, now + 1)
        after = {"hours": round(hours, 1), "reward": round(rew / span, 2), "fills": round(f / max(hours / 24, 1 / 24), 2),
                 "net": round(rew / span + f / max(hours / 24, 1 / 24), 2), "in_book": round(min(1.0, mins / (n * span * 1440)), 3)
                 if now - a >= 1800 else None}
        verdict = None
        if before and hours >= IMPROVE_JUDGE_HOURS:
            verdict = "better" if after["net"] > before["net"] else "worse"
        return {"before": before, "after": after, "judge_hours": IMPROVE_JUDGE_HOURS, "progress": round(min(1.0, hours / IMPROVE_JUDGE_HOURS), 3),
                "verdict": verdict}

    def improve_view(self, now=None):
        """The card "improving the one that runs": what was found, the lab's comparison, what the button changes, and
        whether it is on."""
        now = now or time.time()
        C = self.improve_compare(now)
        key = self._improve_key()
        W = self.winners().get(key) or {}
        imp = W.get("improve")
        bad = sorted(self._live_bad(key, now)) if key else []
        lead = max((x for x in C["ways"] if x["way"] != "step"), key=lambda x: x["net"], default=None)
        step = next((x for x in C["ways"] if x["way"] == "step"), None)
        return {"key": key, "name": W.get("name"), "live": bool(W.get("live")), "on": bool(imp), "improve": imp, "compare": C,
                "effect": self.improve_effect(key, W, now) if key else None,
                "lead": lead and step and {"way": lead["way"], "name": lead["name"], "by": round(lead["net"] - step["net"], 2)},
                "newer": bool(imp and C["best"] and C["best"] != imp.get("way")), "now": now,
                "bad": [{"cid": c, "q": (self.mm.get(c) or {}).get("m", {}).get("q") or c} for c in bad],
                "rules": {"pull_days": IMPROVE_PULL_DAYS, "bad_fills": IMPROVE_BAD_FILLS, "bad_hours": IMPROVE_BAD_HOURS,
                          "rank_days": PILOT_RANK_DAYS, "pull_now": MM_PULL / 86400}}

    def improve_set(self, on, now=None):
        """The owner's click: the live maker improved (the best way the lab knows now, and the rules above), or back as
        it was."""
        now = now or time.time()
        key = self._improve_key()
        if not key:
            return {"error": "אין עושה שוק קיים לשפר: קודם פותחים ניסוי אמת"}
        Ws = self.winners()
        if on:
            C = self.improve_compare(now)
            if Ws[key].get("improve") and not (C["best"] and C["best"] != Ws[key]["improve"].get("way")):
                return {"error": "כבר משופרת: אין כרגע מה לשפר"}
            Ws[key]["improve"] = {"since": now, "way": C["best"] or (key[3:] if key[3:] in IMPROVE_WAYS else "step"),
                                  "chosen_on_days": C["days"], "ready": C["ready"]}
        else:
            Ws[key].pop("improve", None)
            Ws[key]["improve_until"] = now
        self._save_winners(Ws)
        self.__dict__.pop("_view_hit", None)
        log.warning("MAKER IMPROVE %s: %s", "ON" if on else "OFF", Ws[key].get("improve"))
        return dict(self.improve_view(now), ok=True)

    def _live_way(self, var, W):
        """The way whose quote real money follows: the improved one's, if the owner improved it."""
        way = ((W or {}).get("improve") or {}).get("way")
        return way if way in {v[0] for v in MM_VARIANTS} else var

    def _live_bad(self, key, now):
        """The markets whose real fills lost IMPROVE_BAD_FILLS times in the last IMPROVE_BAD_HOURS hours (their worth by
        the middle now): they rest."""
        bad = {}
        for cid, detail in self.db.execute("SELECT key, detail FROM live_trades WHERE test=? AND status='filled' AND ts>=?",
                                           (key, now - IMPROVE_BAD_HOURS * 3600)).fetchall():
            x = json.loads(detail)
            s = self.mm.get(cid) or {}
            mid = s.get("mid")
            if mid is None:
                continue
            yes = x.get("side") in ("buy", "yes")
            if fill_pnl("buy" if yes else "sell", float(x.get("price") or 0), float(x.get("shares") or 0), mid) < 0:
                bad[cid] = bad.get(cid, 0) + 1
        return {c for c, n in bad.items() if n >= IMPROVE_BAD_FILLS}

    def _live_net_score(self, now):
        """Each market's net in the lab over the last PILOT_RANK_DAYS ("step"'s reward, its fills' worth, its rebates)."""
        since = now - PILOT_RANK_DAYS * 86400
        f, shares = self._at_stake()
        score = {c: (r or 0.0) * f for c, r in self.db.execute(
            "SELECT cid, SUM(reward) FROM mm_rew WHERE var='step' AND day>=? GROUP BY cid", (_day(since),)).fetchall()}
        for cid, side, px, sh, rb in self.db.execute("SELECT cid, side, px, sh, rebate FROM mm_fills WHERE var='step' AND ts>=?", (since,)).fetchall():
            mid = (self.mm.get(cid) or {}).get("mid")
            k = shares(cid, sh)
            score[cid] = (score.get(cid) or 0.0) + (fill_pnl(side, px, k, mid) if mid is not None else 0.0) + (rb or 0.0) * (k / sh if sh else 0.0)
        return score

    def _live_held(self, W):
        """Real money's own stop that holds a live maker too (the day's loss limit, or Polymarket refusing orders from
        where the machine is): the stop's record, or None. The owner, 25.11.2026: real money had stopped that morning
        on the day's loss limit, a record that stays till real money is switched on again; that afternoon he switched
        the trial on, by its own switch and its own confirmation, and it sent nothing and said nothing. A stop made
        BEFORE the maker's switch no longer holds it: the switch is the later word. One made after it does, and so
        does a refusal by where the machine is, whenever it was read."""
        t = self.trader
        stop = t.stopped() if t else None
        if not stop:
            return None
        if not isinstance(stop, dict):
            return {"reason": "stop"}
        ts = stop.get("ts")
        if stop.get("reason") == "region" or ts is None or float(ts) >= float((W or {}).get("live_since") or 0):
            return stop
        return None

    def live_off(self, why, now=None):
        """Every live maker off and its orders out of the books (the owner's stop button for real money stops this
        too: until 25.11.2026 it took the orders out and the maker put them back a minute later)."""
        now = now or time.time()
        Ws, off = self.winners(), []
        for key, W in Ws.items():
            if key.startswith("mm:") and W.get("live"):
                W.update(live=False, live_until=now, ended={"ts": now, "why": why})
                off.append(key)
        if off:
            self._save_winners(Ws)
            self._live_cancel_all(now)
            self.__dict__.pop("_view_hit", None)
            log.warning("LIVE MAKERS OFF (%s): %s", why, off)
        return off

    def _live_pick(self, var, W, stake, now):
        """The markets real money quotes in (the lock held): those of the last minute stay while they pay and the quote
        meets their least size (no hopping from market to market, which would leave shares behind in each); room is
        filled, for a trial, by the reward the paper's same way earned there in PILOT_RANK_DAYS, and for a winner by the
        largest pools."""
        n = int(W.get("markets") or LIVE_MM_MARKETS)
        held = {k[0] for k in list(self.live_orders)}
        ratio = self._live_paid_ratio("mm:" + var)
        over = {c for c, r in ratio.items() if r < MM_OVERRATED}
        worth = lambda c, v: v * min(1.0, max(0.2, ratio.get(c, 1.0))) if v > 0 else v
        qv = self._live_way(var, W)
        frac = {v[0]: v[1] for v in MM_VARIANTS}[qv]
        imp = W.get("improve")
        pull = IMPROVE_PULL_DAYS * 86400 if imp else MM_PULL

        def fits(s):
            m, q = s["m"], s["v"][qv]["quote"]
            shares = stake / ((q["bid"] + 1 - q["ask"]) if q else max(0.5, 1 - 2 * frac * m["v"]))
            return m["rate"] > 0 and shares + 1e-9 >= m["min_size"] and now < m["end_ts"] - pull

        if W.get("safe"):  # (the safe way: every market holding something stays till it's locked; then calm ones)
            hold = [c for c in self.mm if abs(self._live_net(c)) >= 1]
            net = {c: worth(c, v) for c, v in self._live_net_score(now).items()}
            keep = hold + [c for c in self.live_pick if c in self.mm and c not in hold and c not in over and fits(self.mm[c]) and self._safe_ok(self.mm[c], now)]
            more = sorted((s for c, s in self.mm.items() if c not in keep and c not in over and fits(s) and self._safe_ok(s, now)),
                          key=lambda s: -(net.get(s["m"]["cid"]) or 0.0))
            keep += [s["m"]["cid"] for s in more[:max(0, n - len(keep))]]
            self.live_pick = keep
            return [self.mm[c] for c in keep]
        if imp:  # (improved: by the net in the lab, the losing ones let go, the ones whose real fills lost resting)
            net = {c: worth(c, v) for c, v in self._live_net_score(now).items()}
            out = self._live_bad("mm:" + var, now) | over
            keep = [c for c in self.live_pick if c in self.mm and c not in out and (net.get(c) or 0.0) >= 0 and fits(self.mm[c])][:n]
            more = sorted((s for c, s in self.mm.items() if c not in keep and c not in out and fits(s) and (net.get(c) or 0.0) > 0),
                          key=lambda s: -net.get(s["m"]["cid"], 0.0))
            keep += [s["m"]["cid"] for s in more[:n - len(keep)]]
            self.live_pick = keep
            return [self.mm[c] for c in keep]
        keep = [c for c in self.live_pick if c in self.mm and c not in over and (c in held or fits(self.mm[c]))][:n]
        if len(keep) < n:
            if W.get("pilot"):
                rank = dict(self.db.execute("SELECT cid, SUM(reward) FROM mm_rew WHERE var=? AND day>=? GROUP BY cid",
                                            (var, _day(now - PILOT_RANK_DAYS * 86400))).fetchall())
                score = lambda s: worth(s["m"]["cid"], rank.get(s["m"]["cid"]) or 0.0)
            else:
                score = lambda s: s["m"]["rate"]
            more = sorted((s for c, s in self.mm.items() if c not in keep and c not in over and fits(s) and score(s) > 0), key=lambda s: -score(s))
            keep += [s["m"]["cid"] for s in more[:n - len(keep)]]
        self.live_pick = keep
        return [self.mm[c] for c in keep]

    def _live_score(self, key, now):
        """The reward the minute that passed should pay our real orders, the paper's way on the real book: only orders
        that rested through all of it (there at the read before, still there now), their shares left, our own taken
        out of the others' score."""
        last, self._live_at = self.__dict__.get("_live_at"), now
        self._live_shares = {}
        if not last or now - last > MM_GAP:
            return
        mins, by = (now - last) / 60, {}
        for (cid, side), o in list(self.live_orders.items()):
            if o.get("since", now) > last:
                continue
            by.setdefault(cid, []).append((o["price"] if side == "yes" else round(1 - o["price"], 6),
                                           o["shares"] - o["matched"], "bid" if side == "yes" else "ask"))
        with self.lock:
            for cid, orders in by.items():
                s = self.mm.get(cid)
                if not s or s["t"] != now or not s.get("st"):
                    continue
                share = live_share(s["st"], s["m"]["v"], s["m"]["min_size"], orders)
                self._live_shares[cid] = share
                zero = self.__dict__.setdefault("_live_zero", {})
                zero[cid] = zero.get(cid, 0) + 1 if share <= 0 else 0
                if zero[cid] >= MM_ZERO_MINUTES:  # (in the book and earning nothing)
                    part = [o for k, o in list(self.live_orders.items()) if k[0] == cid and o.get("matched", 0) > 0]
                    if part:  # (a part of it filled, what is left under the least size: a fresh, whole one in its place)
                        for o in part:
                            o["refresh"] = True
                    else:  # (out of the book for a while)
                        self.__dict__.setdefault("_live_pause", {})[cid] = now + MM_ZERO_PAUSE
                    zero[cid] = 0
                self.db.execute("INSERT INTO live_rew (test, cid, day, minutes, predicted) VALUES (?,?,?,?,?) ON CONFLICT(test, cid, day) "
                                "DO UPDATE SET minutes=minutes+excluded.minutes, predicted=predicted+excluded.predicted",
                                (key, cid, _day(now), mins, share * s["m"]["rate"] * mins / 1440))
                self.db.execute("INSERT INTO live_hour VALUES (?,?,?,?) ON CONFLICT(test, hour) "
                                "DO UPDATE SET minutes=minutes+excluded.minutes, predicted=predicted+excluded.predicted",
                                (key, now // 3600 * 3600, mins, share * s["m"]["rate"] * mins / 1440))
            self.db.commit()

    def _live_reads(self, key, now):
        """Polymarket's own word, set beside ours: the share of each pool it says our orders hold now, and what it paid
        for each of the last days (a day's payment comes after the day ends). Never in the trading's way: a read that
        fails is noted and tried again at its next time."""
        t = self.trader
        if self._live_shares and hasattr(t, "reward_shares") and now - self.__dict__.get("_share_at", 0) >= SHARE_EVERY:
            self._share_at = now
            try:
                theirs = {str(k).lower(): float(v) for k, v in (t.reward_shares() or {}).items()}  # (percent of the pool)
                with self.lock:
                    for cid, mine in self._live_shares.items():
                        self.db.execute("UPDATE live_rew SET mine=COALESCE(mine, 0)+?, theirs=COALESCE(theirs, 0)+?, n=COALESCE(n, 0)+1 "
                                        "WHERE test=? AND cid=? AND day=?", (mine, theirs.get(cid.lower(), 0.0) / 100, key, cid, _day(now)))
                    self.db.commit()
            except Exception as e:
                self._note("live", False, f"reward shares: {e}")
        if hasattr(t, "reward_paid") and now - self.__dict__.get("_paid_at", 0) >= PAID_EVERY:
            try:
                for back in range(3):
                    day = _day(now - back * 86400)
                    cids = [r[0] for r in self.db.execute("SELECT cid FROM live_rew WHERE test=? AND day=?", (key, day)).fetchall()]
                    if not cids:
                        continue
                    self._paid_at = now  # (from the first day there is something to ask about)
                    paid = {str(k).lower(): float(v) for k, v in (t.reward_paid(day) or {}).items()}
                    with self.lock:
                        was = self.db.execute("SELECT COALESCE(SUM(paid), 0) FROM live_rew WHERE test=? AND day=?", (key, day)).fetchone()[0]
                        for cid in cids:
                            self.db.execute("UPDATE live_rew SET paid=? WHERE test=? AND cid=? AND day=?", (paid.get(cid.lower(), 0.0), key, cid, day))
                        got = self.db.execute("SELECT COALESCE(SUM(paid), 0) FROM live_rew WHERE test=? AND day=?", (key, day)).fetchone()[0]
                        if day == _day(now) and got - was >= 0.005:  # (the day's chart steps up the moment it is read)
                            self.db.execute("INSERT INTO live_events VALUES (?,?,?,?)", (key, now, "reward", round(got - was, 4)))
                        self.db.commit()
            except Exception as e:
                self._note("live", False, f"rewards paid: {e}")

    def live_real(self, key, now):
        """Real money's own record for a maker: by day, the minutes its orders rested, the reward they should have
        earned (our count) and what Polymarket paid; the shares each says we held; the real fills, valued at their
        market's result or its middle now. "predicted"/"paid"/"ratio" are over days that ended and whose payment was
        read, so the two are the same days."""
        today, days = _day(now), {}
        mine = theirs = n = 0
        for cid, day, mins, pred, a, b, c, paid in self.db.execute(
                "SELECT cid, day, minutes, predicted, mine, theirs, n, paid FROM live_rew WHERE test=?", (key,)).fetchall():
            d = days.setdefault(day, {"day": day, "minutes": 0.0, "predicted": 0.0, "paid": None, "done": day < today})
            d["minutes"] += mins or 0
            d["predicted"] += pred or 0
            if paid is not None:
                d["paid"] = (d["paid"] or 0.0) + paid
            mine, theirs, n = mine + (a or 0), theirs + (b or 0), n + (c or 0)
        done = [d for d in days.values() if d["done"] and d["paid"] is not None]
        marks = {r[0]: (r[2] if r[2] is not None else r[1]) for r in self.db.execute("SELECT cid, mid, outcome FROM mm_markets").fetchall()}
        pnl, fills = 0.0, 0
        for cid, detail in self.db.execute("SELECT key, detail FROM live_trades WHERE test=? AND status='filled'", (key,)).fetchall():
            x = json.loads(detail)
            fills += 1
            if marks.get(cid) is not None:
                pnl += fill_pnl(x.get("side"), float(x.get("price") or 0), float(x.get("shares") or 0), marks[cid])
        pred, paid = sum(d["predicted"] for d in done), sum(d["paid"] for d in done)
        paid_all = sum(d["paid"] or 0.0 for d in days.values())
        return {"days": [dict(d, minutes=round(d["minutes"]), predicted=round(d["predicted"], 2),
                              paid=None if d["paid"] is None else round(d["paid"], 2))
                         for d in sorted(days.values(), key=lambda d: d["day"], reverse=True)[:10]],
                "days_done": len(done), "predicted": round(pred, 2), "paid": round(paid, 2), "ratio": round(paid / pred, 3) if pred > 0 else None,
                "predicted_all": round(sum(d["predicted"] for d in days.values()), 2), "paid_all": round(paid_all, 2),
                "minutes": round(sum(d["minutes"] for d in days.values())),
                "share_mine": round(mine / n, 4) if n else None, "share_theirs": round(theirs / n, 4) if n else None, "share_n": n,
                "fills": fills, "fills_pnl": round(pnl, 2), "net": round(paid_all + pnl, 2)}

    LEDGER_ROWS = 300

    def live_ledger(self, key, now):
        """The real maker's own book, for its tab (the owner, 25.11.2026: "a new tab, Market maker (live): the summary
        cards, the strategy with its switch, a graph of the profits, under it what is on the way now, and the table of
        trades"). Every real order of it with its market's name and what it made: a fill in a market that ended, its
        result ("settled"); in one still running, its value at the market's middle now ("open": on the way, not money
        yet). The graph's points: the rewards Polymarket paid, at the end of their day, and the settled fills, summed.
        net = rewards paid + settled + open, the same sum the trial's stop counts (live_real)."""
        marks = {r[0]: r[1:] for r in self.db.execute("SELECT cid, q, mid, outcome, end_ts FROM mm_markets").fetchall()}
        rows = self.db.execute("SELECT ts, key, detail, cost, status, error FROM live_trades WHERE test=? ORDER BY ts", (key,)).fetchall()
        trades, events, pos = [], [], {}
        done = on_way = spent = 0.0
        fills = settled = wins = 0
        # What really came into the wallet (the owner, 25.11.2026: "one more, final card: the total money that came into
        # the wallet: the profits that came in at 01:00, if they did, and the losses if there were any; the real and final
        # figure, so that at the end of each week we know how much really came in, everything included"): the rewards
        # of days that have ended (Polymarket pays a day's reward at midnight UTC) and the fills whose market ended.
        # Not in it: today's reward, still being counted, and shares in markets still running.
        week_from, week = now - 7 * 86400, 0.0
        # (and today's, from midnight UTC, 01:00 in Israel, when Polymarket pays the day before: the owner, 26.11.2026,
        #  "the leftmost card should start over each day at one, the moment Poly transfers the money, so I see the total and
        #  the day's")
        day0, done_day, fills_day = _day_ts(_day(now)), 0.0, 0
        # (the owner, 26.11.2026, the four cards: "the day's net, rewards and fills included; the month's return on the
        #  money, out of how much and how much on each dollar bought; the day's loss: the number of fills and their money,
        #  in the minus and red, since they are what I lose; and the total that came into the wallet". A fill of today
        #  counts with its result if its market ended, else with its value at the market's middle now.)
        pnl_day, month_from = 0.0, now - 30 * 86400
        m_fills = m_spent = 0.0
        for ts, cid, detail, cost, status, error in rows:
            x = json.loads(detail)
            q, mid, outcome, end_ts = marks.get(cid) or (None, None, None, None)
            yes = x.get("side") in ("buy", "yes")
            t = {"ts": ts, "cid": cid, "q": q or cid, "yes": yes, "price": x.get("price"), "shares": x.get("shares"), "cost": round(cost or 0.0, 2),
                 "status": status, "error": error, "pnl": None, "state": None, "mid": x.get("mid"), "end_ts": end_ts,
                 "won": None if outcome is None else bool(outcome >= 0.5) == yes}
            if status == "filled":
                fills += 1
                fills_day += ts >= day0
                spent += cost or 0.0
                m_spent += (cost or 0.0) if ts >= month_from else 0.0
                sh = float(x.get("shares") or 0)
                mark = outcome if outcome is not None else mid
                pnl = fill_pnl("buy" if yes else "sell", float(x.get("price") or 0), sh, mark) if mark is not None else None
                t.update(pnl=None if pnl is None else round(pnl, 2), state="settled" if outcome is not None else "open")
                pnl_day += (pnl or 0.0) if ts >= day0 else 0.0
                m_fills += (pnl or 0.0) if ts >= month_from else 0.0
                if outcome is not None:
                    done += pnl
                    settled += 1
                    wins += pnl > 0
                    events.append((ts, pnl))
                    week += pnl if ts >= week_from else 0.0
                    done_day += pnl if ts >= day0 else 0.0
                else:
                    P = pos.setdefault(cid, {"cid": cid, "q": q or cid, "net": 0.0, "cost": 0.0, "pnl": 0.0, "mid": mid, "fills": 0,
                                             "end_ts": end_ts})
                    P["net"] += sh if yes else -sh
                    P["cost"] += cost or 0.0
                    P["pnl"] += pnl or 0.0
                    P["fills"] += 1
                    on_way += pnl or 0.0
            trades.append(t)
        rewards = paid_in = m_rew = 0.0
        paid_days, today = 0, _day(now)
        rew_day = {}
        day_minutes = self.db.execute("SELECT COALESCE(SUM(minutes), 0) FROM live_rew WHERE test=? AND day=?", (key, today)).fetchone()[0]
        for day, paid in self.db.execute("SELECT day, SUM(paid) FROM live_rew WHERE test=? AND paid IS NOT NULL GROUP BY day", (key,)).fetchall():
            if paid:
                rewards += paid
                rew_day[day] = paid
                m_rew += paid if _day_ts(day) + 86400 > month_from else 0.0
                events.append((min(now, _day_ts(day) + 86400), paid))
                if day < today:  # (a day that ended: its reward was paid at its midnight)
                    paid_in += paid
                    paid_days += 1
                    week += paid if _day_ts(day) + 86400 >= week_from else 0.0
        events.sort()
        W = self.winners().get(key) or {}
        start = min([e[0] for e in events] + [W.get("first_live") or now])
        began = min([r[0] for r in rows] + [start])
        m_days = max(1, min(30, -(-(now - max(began, month_from)) // 86400)))
        series, cum = [[start, 0.0]], 0.0
        for ts, v in events:
            cum += v
            series.append([ts, round(cum, 2)])
        if events:
            series.append([now, round(cum, 2)])
        mine = [o for o in list(self.live_orders.values()) if (o.get("test") or key) == key]
        # The two charts (the owner, 27.11.2026: "one chart within the day that is updated on every reward or fill, and
        # one by the day that is updated every day at one in the morning"). Today's: each reward the moment it was read
        # and each fill the moment it was made, at what it is worth now ("temp" while its market runs: what it costs
        # can still change); the reward counted before the moments were kept stands at the day's start. By the day:
        # the reward, the fills of that day at what they are worth now, the part of them still running, the net.
        fills_today = [{"ts": t["ts"], "kind": "fill", "amount": round(t["pnl"] or 0.0, 2), "temp": t["state"] == "open", "q": t["q"],
                        "yes": t["yes"], "price": t["price"], "shares": t["shares"]}
                       for t in trades if t["status"] == "filled" and t["ts"] >= day0]
        rew_ev = [{"ts": r[0], "kind": "reward", "amount": round(r[1], 4)} for r in self.db.execute(
            "SELECT ts, amount FROM live_events WHERE test=? AND kind='reward' AND ts>=? ORDER BY ts", (key, day0)).fetchall()]
        before = max(0.0, (rewards - paid_in) - sum(e["amount"] for e in rew_ev))
        intraday = {"from": day0, "now": now, "base": round(before, 2),
                    "events": sorted(rew_ev + fills_today, key=lambda e: e["ts"])}
        by_day = {}
        for t in trades:
            if t["status"] == "filled":
                D = by_day.setdefault(_day(t["ts"]), {"fills": 0.0, "temp": 0.0, "n": 0})
                D["fills"] += t["pnl"] or 0.0
                D["temp"] += (t["pnl"] or 0.0) if t["state"] == "open" else 0.0
                D["n"] += 1
        first = min([_day(r[0]) for r in rows] + list(rew_day) + [today])
        daily, d = [], _day_ts(first)
        while d <= _day_ts(today):
            k = _day(d)
            D = by_day.get(k) or {"fills": 0.0, "temp": 0.0, "n": 0}
            daily.append({"day": k, "from": d, "reward": round(rew_day.get(k, 0.0), 2), "fills": round(D["fills"], 2),
                          "temp": round(D["temp"], 2), "n": D["n"], "net": round(rew_day.get(k, 0.0) + D["fills"], 2), "today": k == today})
            d += 86400
        return {"trades": trades[::-1][:self.LEDGER_ROWS], "orders": len(rows), "series": series, "intraday": intraday, "daily": daily,
                "wallet": {"total": round(paid_in + done, 2), "rewards": round(paid_in, 2), "fills": round(done, 2), "days": paid_days,
                           "week": round(week, 2), "today": round(rewards - paid_in, 2),
                           "day": round(rewards - paid_in + done_day, 2), "day_fills": fills_day, "day_from": day0,
                           "day_fills_pnl": round(pnl_day, 2), "day_net": round(rewards - paid_in + pnl_day, 2),
                           "day_minutes": round(day_minutes, 1)},
                "month": {"net": round(m_rew + m_fills, 2), "spent": round(m_spent, 2), "days": int(m_days)},
                "rewards": round(rewards, 2), "settled_pnl": round(done, 2), "open_pnl": round(on_way, 2),
                "net": round(rewards + done + on_way, 2), "spent": round(spent, 2), "fills": fills, "settled": settled, "wins": wins,
                "positions": sorted(({"cid": P["cid"], "q": P["q"], "net": round(P["net"], 1), "cost": round(P["cost"], 2),
                                      "pnl": round(P["pnl"], 2), "mid": P["mid"], "fills": P["fills"], "end_ts": P["end_ts"]}
                                     for P in pos.values()),
                                    key=lambda P: P["pnl"]),
                "resting": len(mine), "stuck": sum(1 for o in mine if o.get("out")),
                "resting_money": round(sum(o["price"] * max(0.0, o["shares"] - o["matched"]) for o in mine), 2)}

    def live_markets(self, key, now, book=None):
        """The real maker's markets one by one, for its tab (the owner, 25.11.2026, the trial running: "I don't really
        understand what goes on there, what the strategy does and how it works: I want to see exactly why we got in,
        how long till it closes, what we got, what we lost, the status and everything"). Each market it quotes in, or
        holds shares or an order in: where its book stands (the middle, the best prices, the reward's band), what the
        maker does there now and why ("state"), each side's order or why there is none, the reward (the pool, our
        share, what we counted and what was paid), what was bought, and when the market ends."""
        var = key[3:]
        W = self.winners().get(key) or {}
        live = bool(W.get("live"))
        stake = float(W.get("stake") or STAKES["mm"])
        cool = {v[0]: v[2] for v in MM_VARIANTS}.get(var) or 0
        cap = MM_EXPOSURE * stake / MM_CAPITAL
        book = book or self.live_ledger(key, now)
        pos = {P["cid"]: P for P in book["positions"]}
        made = {}
        for t in book["trades"]:
            if t["status"] == "filled":
                x = made.setdefault(t["cid"], {"fills": 0, "settled_pnl": 0.0})
                x["fills"] += 1
                if t["state"] == "settled":
                    x["settled_pnl"] += t["pnl"] or 0.0
        rew, today = {}, _day(now)
        for cid, day, mins, pred, mine, theirs, n, paid in self.db.execute(
                "SELECT cid, day, minutes, predicted, mine, theirs, n, paid FROM live_rew WHERE test=?", (key,)).fetchall():
            r = rew.setdefault(cid, {"minutes": 0.0, "predicted": 0.0, "paid": 0.0, "today": 0.0, "mine": 0.0, "theirs": 0.0, "n": 0})
            r["minutes"] += mins or 0
            r["predicted"] += pred or 0
            r["paid"] += paid or 0
            r["today"] += (pred or 0) if day == today else 0.0
            r["mine"], r["theirs"], r["n"] = r["mine"] + (mine or 0), r["theirs"] + (theirs or 0), r["n"] + (n or 0)
        why = {}
        for cid, detail, error in self.db.execute("SELECT key, detail, error FROM live_trades WHERE test=? AND status='rejected' "
                                                  "ORDER BY ts DESC LIMIT 80", (key,)).fetchall():
            why.setdefault((cid, json.loads(detail).get("side")), error)
        orders = dict(self.live_orders)
        cids = [c for c in (list(self.live_pick) if live else [])]
        cids += [c for c in list(dict.fromkeys([k[0] for k in orders] + list(pos) + list(made))) if c not in cids]
        out = []
        for cid in cids:
            row = self.db.execute("SELECT q, cat, rate, v, min_size, end_ts, mid, outcome FROM mm_markets WHERE cid=?", (cid,)).fetchone()
            if not row:
                continue
            q, cat, rate, v, min_size, end_ts, mid, outcome = row
            s = self.mm.get(cid) or {}
            st = s.get("st") or {}
            mid = s.get("mid") if s.get("mid") is not None else mid
            qv = self._live_way(var, W)
            quote_ = ((s.get("v") or {}).get(qv) or {}).get("quote")
            last = ((s.get("v") or {}).get(qv) or {}).get("fill_t") if qv in MM_AFTER else s.get("last_print")
            away = bool(cool and last is not None and now - last < cool)
            mine = [k for k in orders if k[0] == cid]
            paused = (self.__dict__.get("_live_pause") or {}).get(cid, 0)
            state = ("ended" if outcome is not None else "off" if not live else "held" if self._live_held(W)
                     else "no_reward" if paused > now
                     else "last_day" if end_ts and now >= end_ts - (IMPROVE_PULL_DAYS * 86400 if W.get("improve") else MM_PULL)
                     else "aside" if away and not mine
                     else "quoting" if mine else ("moved" if st else "no_book") if quote_ is None else "waiting")
            net = (pos.get(cid) or {}).get("net") or 0.0
            sides = {}
            for side in ("yes", "no"):
                o = orders.get((cid, side))
                if o:
                    x = {"state": "out" if o.get("out") else "resting", "price": o["price"], "shares": round(o["shares"], 1),
                         "matched": round(o["matched"], 1), "since": o.get("since")}
                else:
                    r = self.live_rej.get((cid, side))
                    full = mid is not None and ((net * mid >= cap) if side == "yes" else (-net * (1 - mid) >= cap))
                    small = bool(quote_) and stake / (quote_["bid"] + 1 - quote_["ask"]) + 1e-9 < (min_size or 0)
                    x = ({"state": "refused", "retry_in": round(LIVE_REJ_REST - (now - r)), "retry_at": r + LIVE_REJ_REST,
                          "error": why.get((cid, side))}
                         if r and now - r < LIVE_REJ_REST else {"state": "full" if full else "small" if small else "none"})
                    if quote_:  # (where the rule would put it)
                        x["would"] = quote_["bid"] if side == "yes" else round(1 - quote_["ask"], 6)
                sides[side] = x
            r = rew.get(cid) or {}
            share = self._live_shares.get(cid) if state == "quoting" else None
            P = pos.get(cid)
            out.append({"cid": cid, "q": q, "cat": cat, "state": state, "end_ts": end_ts, "pull_ts": (end_ts or 0) - MM_PULL,
                        "outcome": outcome, "mid": mid, "bb": st.get("bb"), "ba": st.get("ba"), "band": v, "min_size": min_size,
                        "back_in": round(cool - (now - last)) if away else None, "back_at": last + cool if away else None, "cool": cool,
                        "yes": sides["yes"], "no": sides["no"], "cap": round(cap, 2),
                        "pool": rate, "share": share, "per_day": None if share is None else round(share * rate, 3),
                        "share_avg": round(r["mine"] / r["n"], 4) if r.get("n") else None,
                        "share_theirs": round(r["theirs"] / r["n"], 4) if r.get("n") else None,
                        "minutes": round(r.get("minutes") or 0), "predicted": round(r.get("predicted") or 0, 3),
                        "today": round(r.get("today") or 0, 3), "paid": round(r.get("paid") or 0, 2),
                        "fills": (made.get(cid) or {}).get("fills", 0), "settled_pnl": round((made.get(cid) or {}).get("settled_pnl", 0.0), 2),
                        "pos": P and {"net": P["net"], "cost": P["cost"], "pnl": P["pnl"],
                                      "value": round(abs(P["net"]) * (mid if P["net"] >= 0 else 1 - mid), 2) if mid is not None else None}})
        return out

    def _pilot_over(self, key, W, stake, now):
        """A trial that lost its limit ends by itself: the orders out of the books, back to the simulation, and why."""
        R = self.live_real(key, now)
        limit = PILOT_STOP * stake * int(W.get("markets") or PILOT_MARKETS)
        if R["net"] > -limit:
            return False
        Ws = self.winners()
        Ws[key].update(live=False, live_until=now, ended={"ts": now, "why": f"ההפסד בכסף אמיתי הגיע ל-${-R['net']:,.2f}, מעבר לקו העצירה של הניסוי "
                                                                           f"(${limit:,.2f}): ההוראות בוטלו והניסוי חזר לסימולציה"})
        self._save_winners(Ws)
        self._live_cancel_all(now)
        self.__dict__.pop("_view_hit", None)
        log.warning("THE REAL TRIAL %s ended itself: net $%.2f, limit $%.2f", key, R["net"], limit)
        return True

    def _live_check(self, key, o, now):
        """What sellers took of a resting order since the last look, recorded. True when Polymarket answered."""
        try:
            matched, status = self.trader.maker_matched(o["id"])
        except Exception:
            return False
        if str(status or "").startswith("cancel") and not o.get("out"):
            o["gone"] = True
        with self.live_lock:
            n = matched - o["matched"]
            if n > 1e-9:
                o["matched"] = matched
        if n > 1e-9:
            yes_px = o["price"] if key[1] == "yes" else round(1 - o["price"], 6)
            # (under the name of the maker that placed it: by now it may be back in the simulation, its orders on their way out)
            self._live_trade(now, o.get("test") or "mm:" + self._live_var(), key[0],
                             {"side": "buy" if key[1] == "yes" else "sell", "price": yes_px, "shares": n, "token": o.get("token"),
                              "mid": (self.mm.get(key[0]) or {}).get("mid")},
                             n * o["price"], "filled")
        return True

    def _live_drop(self, key, o, now, tries=1, save=False):
        """An order of ours out of the book, and what sellers took of it before it went, recorded (a fill is what makes
        the trade print that sends it out: unrecorded, the shares held would be wrong from then on)."""
        self.live_orders.pop(key, None)
        gone = True
        if o["matched"] < o["shares"] - 0.01:
            gone = self.trader.cancel_maker(o["id"]) is not False
        for i in range(tries):
            if self._live_check(key, o, now) or i == tries - 1:
                break
            time.sleep(1)
        if not gone and o["matched"] < o["shares"] - 0.01:
            # (25.11.2026: Polymarket didn't confirm the cancel, so it may still rest in the book. Until then it was
            #  forgotten here; now it stays the market's order: nothing new is placed on that side of that market, the
            #  cancel is tried again at the next minute, and what sellers take of it meanwhile is recorded)
            o["out"] = o.get("out") or now
            self.live_orders.setdefault(key, o)
        if save:
            self._live_save()

    def _live_save(self):
        """The resting orders and the markets picked, kept for the machine's next start (only when they changed)."""
        state = {"orders": [[k[0], k[1], o] for k, o in list(self.live_orders.items())], "pick": list(self.live_pick)}
        text = json.dumps(state, sort_keys=True)
        if text != self.__dict__.get("_live_saved"):
            with self.lock:
                self.set_meta("live_state", state)
            self._live_saved = text

    def _live_var(self):
        return next((k[3:] for k in self.winners() if k.startswith("mm:") and self.winner_live(k)), "mid")

    def _live_net(self, cid):
        net = 0.0
        for d, st in self.db.execute("SELECT detail, status FROM live_trades WHERE key=? AND status='filled'", (cid,)):
            x = json.loads(d)
            net += x.get("shares", 0) if x.get("side") == "buy" else -x.get("shares", 0)
        return net

    def _live_cancel_all(self, now=None):
        now = now or time.time()
        for key, o in list(self.live_orders.items()):
            if self.trader:
                self._live_drop(key, o, now)  # (one whose cancel isn't confirmed stays, and is tried again next minute)
            else:
                self.live_orders.pop(key, None)
        self._live_save()

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
                                self._live_step_aside(ev.get("asset_id"), ev.get("price"))
                        self.status["mm"]["ws"] = time.time()
            except Exception as e:
                self.status["mm"]["ws_error"] = str(e)[:200]
                await asyncio.sleep(5)

    def _live_step_aside(self, asset, price=None):
        """A live maker winner whose way steps aside after a trade: its orders in that market cancelled at once.
        Improved to a way that steps aside after its own fill only: only a print at or through our order's price (a
        sale to us, as the lab's fill takes it), and for "oneside" that order alone."""
        if not self.live_orders:
            return
        var = self._live_var()
        if (self.winners().get("mm:" + var) or {}).get("safe"):
            return
        way = self._live_way(var, self.winners().get("mm:" + var))
        if not dict((v[0], v[2]) for v in MM_VARIANTS).get(way):
            return
        cid = next((s["m"]["cid"] for s in self.mm.values() if asset in (s["m"]["yes"], s["m"]["no"])), None)
        keys = [k for k in list(self.live_orders) if k[0] == cid]
        how = MM_AFTER.get(way)
        if how:
            try:
                px = float(price)
            except (TypeError, ValueError):
                return
            hit = [k for k in keys if self.live_orders[k].get("token") == asset and px <= self.live_orders[k]["price"] + 1e-9]
            keys = hit if how == "side" else (keys if hit else [])
        for key in keys:
            o = self.live_orders.pop(key, None)
            if o and self.trader:  # (cancelled, and what was taken of it recorded: the print may be our own fill)
                threading.Thread(target=self._live_drop, args=(key, o, time.time(), 3, True), daemon=True).start()

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
        out = dict(w, key=key, sim=sim, live_trades=live, live_spent=round(spent[0], 2), live_fills=spent[1],
                   connected=bool(self.trader and self.trader.configured()))
        if key.startswith("mm:"):
            n = int(w.get("markets") or LIVE_MM_MARKETS)
            held = self._live_held(w) if w.get("live") else None
            book = self.live_ledger(key, now)
            mkts = self.live_markets(key, now, book)
            cool = {v[0]: v[2] for v in MM_VARIANTS}.get(key[3:]) or 0
            if key == self._improve_key():
                out["imp"] = self.improve_view(now)
                out["safe_card"] = self.safe_view(now)
            out.update(markets=n, need=round(stake * n, 2), real=self.live_real(key, now), book=book, mkts=mkts,
                       per_day=round(sum(m["per_day"] or 0.0 for m in mkts), 2), cool=cool, now=now,
                       held=held and {"reason": held.get("reason"), "ts": held.get("ts")},
                       resting=len(self.live_orders) if w.get("live") else 0,
                       quoting=[{"q": self.mm[c]["m"]["q"], "rate": self.mm[c]["m"]["rate"], "net": round(self._live_net(c), 1),
                                 "resting": sum(1 for k in list(self.live_orders) if k[0] == c)} for c in list(self.live_pick) if c in self.mm]
                       if w.get("live") else [])
            # (the share of today its orders were in the book, the owner's markets each: the reviews of 26.11.2026 found
            #  "step" out of the book much of the day, stepping aside after anyone's trade)
            since = max(book["wallet"]["day_from"], w.get("live_since") or 0)
            if w.get("live") and now - since >= 600:
                out["uptime"] = round(min(1.0, book["wallet"]["day_minutes"] / (n * (now - since) / 60)), 3)
            if w.get("pilot"):
                out.update(stop=round(PILOT_STOP * stake * n, 2), days=PILOT_DAYS, pass_ratio=PILOT_RATIO,
                           day=(now - w["first_live"]) / 86400 if w.get("first_live") else None)
        return out

    def mm_view(self, var, since, now):
        fills = self.db.execute("SELECT f.ts, f.cid, f.side, f.px, f.sh, f.mid0, f.m1, f.m10, f.m60, f.rebate, m.mid, m.outcome, m.q, m.cat, "
                                "f.x1, f.x10, f.x60, m.fee "
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
        # What if every fill had been sold instead of held to the market's end (the owner, 25.11.2026, the real trial
        # running: "check it: maybe that's an excellent way to make the money without the risk: the moment we get a few
        # cents, get out and go in somewhere else"). The same fills, sold 1, 10 or 60 minutes after: at the middle
        # (what the research always marked, and nobody can sell there) and at the price one really sells at (kept from
        # 1.427 on; for the fills before it, the middle's mark less half the market's spread now and the fee, an
        # estimate, and said so). The rewards are the same either way; "month" is the whole of it, a month, on the money.
        half = {c: (x["st"]["ba"] - x["st"]["bb"]) / 2 for c, x in list(self.mm.items()) if x.get("st")}
        ways = []
        for i, mins in ((6, 1), (7, 10), (8, 60)):
            at_mid = touch = 0.0
            n = exact = 0
            worst_x = None
            for k, f in enumerate(fills):
                if f[i] is None:  # (not that old yet, or its market ended before: as it stands)
                    at_mid, touch = at_mid + per_fill[k], touch + per_fill[k]
                    continue
                n += 1
                x = f[i + 8]
                if x is None:
                    mid_then = f[3] + (1 if f[2] == "buy" else -1) * f[i] / f[4] if f[4] else f[5] or 0.5
                    x = f[i] - half.get(f[1], 0.01) * f[4] - (f[17] or 0.0) * mid_then * (1 - mid_then) * f[4]
                else:
                    exact += 1
                at_mid, touch = at_mid + f[i], touch + x
                worst_x = x if worst_x is None else min(worst_x, x)
            pm = lambda v: round((rewards + rebates + v) / mdays * 30 / MM_CAPITAL * 100, 2) if mdays else None
            ways.append({"min": mins, "n": n, "exact": exact, "mid": round(at_mid, 2), "touch": round(touch, 2),
                         "worst": None if worst_x is None else round(worst_x, 2), "month_mid": pm(at_mid), "month": pm(touch)})
        exits = {"hold": {"pnl": round(marks, 2), "worst": None if not per_fill else round(min(per_fill), 2),
                          "month": None if month is None else round(month, 2)}, "ways": ways, "fills": len(fills)}
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
                "weeks": weeks, "checks": checks, "verdict": verdict, "recent": recent, "exits": exits}

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
