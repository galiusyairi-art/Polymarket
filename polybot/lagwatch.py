"""Who moves first: a live measurement of how Polymarket's prices follow the price itself.

Ideas 2 and 3 of the owner's panel. Every time our model's chance for a round jumps (the Chainlink
price moved: JUMP points or more within a second), an event starts, and for a few seconds this
watches:
- the same round (kind "self"): how long Polymarket's price (the middle of the book) takes to move
  half as far the same way, and whether it had moved first;
- the other coins' rounds of the same length (kind "cross", led by the coin that jumped): whether
  their own model chance and their market price follow, and how fast.
And for each, the money question: buying the side the jump favours, as a real order would (DELAY
after the jump, at the offer, the taker fee paid), what the share is worth HOLD seconds later: the
middle of the book then (a mark-to-market), and what the buyers then pay (the bid: what selling it
would get, after the fee). A positive average, on many events, is a speed edge; the recording keeps
the last events, so a restart doesn't lose them.
Once real orders have been timed (fills.speed), the same buy is also priced at our own measured delay
(fast_delay: the signal to the order reaching the book) beside DELAY's: whether a window opens at the
speed we really have (the owner, 28.10.2026: "when there are measurements").

Read only: it never trades. Samples are taken at most every SAMPLE seconds per round.
"""

import collections
import math
import threading

from .model import taker_fee_per_share

JUMP = 0.05  # the model's chance moving this much within LOOK seconds starts an event
LOOK = 1.0
FOLLOW = 0.02  # a follower's chance or price moving this much counts as following
DELAY = 0.5  # a real order's way to the book after the jump (what the lab's fills use)
HOLD = 10.0  # the mark-to-market this long after the jump
WATCH = 12.0  # seconds an event is followed
COOLDOWN = 10.0  # a round's next event no sooner
SAMPLE = 0.1
KEEP = 5000  # events kept


def _sample_at(samples, ts):
    """The last sample at or before ts (samples are oldest first)."""
    out = None
    for s in samples:
        if s[0] > ts:
            break
        out = s
    return out


class LagWatch:
    def __init__(self, events=None):
        self.lock = threading.Lock()
        self.samples = collections.defaultdict(lambda: collections.deque(maxlen=int(20 / SAMPLE)))
        self.live = []  # events still being followed
        self.done = collections.deque(events or [], maxlen=KEEP)
        self.last_event = {}
        self.fees = {}
        self.fast_delay = None  # our real orders' measured delay (set by the engine), or None

    def observe(self, key, asset, minutes, ts, fair, up_bid, up_ask, down_bid, down_ask, fee_rate, fee_exponent):
        """One round's state now: its model chance of Up and its book's best prices."""
        if fair is None or None in (up_bid, up_ask, down_bid, down_ask):
            return
        with self.lock:
            q = self.samples[key]
            if q and ts - q[-1][0] < SAMPLE:
                return
            mid = (up_bid + up_ask) / 2
            q.append((ts, fair, mid, up_bid, up_ask, down_bid, down_ask))
            self.fees[key] = (fee_rate, fee_exponent, asset, minutes)
            self._follow(key, ts)
            before = _sample_at(q, ts - LOOK)
            if before is None or ts - self.last_event.get(key, 0) < COOLDOWN:
                return
            move = fair - before[1]
            if abs(move) < JUMP:
                return
            self.last_event[key] = ts
            way = 1 if move > 0 else -1
            for other, oq in list(self.samples.items()):
                o = self.fees.get(other)
                if not oq or o is None or o[3] != minutes or ts - oq[-1][0] > 5:
                    continue
                start = _sample_at(oq, ts - LOOK) or oq[0]
                now_o = oq[-1]
                self.live.append({
                    "kind": "self" if other == key else "cross", "leader": asset, "follower": o[2], "minutes": minutes,
                    "key": other, "t0": ts, "way": way, "jump": move,
                    "fair0": now_o[1] if other != key else before[1],
                    "mid0": start[2] if other == key else now_o[2],  # (a follower: from where it is at the jump)
                    # (the market had already moved half as far, the same way, before the jump)
                    "market_first": (now_o[2] - start[2]) * way >= abs(move) / 2 if other == key else None,
                    "lag_market": None, "lag_fair": 0.0 if other == key else None,
                    "entry": None, "mark_mid": None, "mark_bid": None,
                    "fast_delay": self.fast_delay, "entry_fast": None, "mark_fast": None,
                })

    def _follow(self, key, ts):
        keep = []
        for e in self.live:
            if e["key"] != key:
                keep.append(e)
                continue
            s = self.samples[key][-1]
            t, fair, mid = s[0], s[1], s[2]
            dt = t - e["t0"]
            need = abs(e["jump"]) / 2 if e["kind"] == "self" else FOLLOW
            if e["lag_market"] is None and (mid - e["mid0"]) * e["way"] >= need:
                e["lag_market"] = dt
            if e["lag_fair"] is None and (fair - e["fair0"]) * e["way"] >= FOLLOW:
                e["lag_fair"] = dt
            rate, exp = self.fees[key][:2]
            if e["entry"] is None and dt >= DELAY:
                ask = s[4] if e["way"] > 0 else s[6]
                e["entry"] = ask + taker_fee_per_share(ask, rate, exp) if 0 < ask < 1 else None
                if e["entry"] is None:
                    continue  # (nothing to buy: the event is dropped)
            fd = e.get("fast_delay")
            if fd is not None and e.get("entry_fast") is None and dt >= fd:
                ask = s[4] if e["way"] > 0 else s[6]
                e["entry_fast"] = ask + taker_fee_per_share(ask, rate, exp) if 0 < ask < 1 else False
            if e.get("entry_fast") and e.get("mark_fast") is None and dt >= HOLD:
                e["mark_fast"] = (mid if e["way"] > 0 else 1 - mid) - e["entry_fast"]
            if e["entry"] is not None and e["mark_mid"] is None and dt >= HOLD:
                side_mid = mid if e["way"] > 0 else 1 - mid
                bid = s[3] if e["way"] > 0 else s[5]
                e["mark_mid"] = side_mid - e["entry"]
                e["mark_bid"] = bid - taker_fee_per_share(bid, rate, exp) - e["entry"] if bid else None
            if dt >= WATCH:
                if e["entry"] is not None and e["mark_mid"] is not None:
                    self.done.append({k: e.get(k) for k in ("kind", "leader", "follower", "minutes", "t0", "way", "jump",
                                                            "market_first", "lag_market", "lag_fair", "mark_mid", "mark_bid",
                                                            "fast_delay", "mark_fast")})
                continue
            keep.append(e)
        self.live = keep

    def events(self):
        with self.lock:
            return list(self.done)

    def summary(self):
        """By kind and pair of coins (and the round's length): how many events, how often and how fast
        the follower moved, and the mark-to-market of buying right after the jump (mean, its spread)."""
        groups = collections.defaultdict(list)
        for e in self.events():
            groups[(e["kind"], e["leader"], e["follower"], e["minutes"])].append(e)
        out = []
        for (kind, leader, follower, minutes), es in groups.items():
            n = len(es)
            mids = [e["mark_mid"] for e in es]
            bids = [e["mark_bid"] for e in es if e["mark_bid"] is not None]
            mean = sum(mids) / n
            se = math.sqrt(sum((x - mean) ** 2 for x in mids) / (n - 1) / n) if n > 1 else None
            lags = sorted(e["lag_market"] for e in es if e["lag_market"] is not None)
            flags = sorted(e["lag_fair"] for e in es if e["lag_fair"] is not None)
            fast = [e["mark_fast"] for e in es if e.get("mark_fast") is not None]
            nf = len(fast)
            f_mean = sum(fast) / nf if nf else None
            f_se = math.sqrt(sum((x - f_mean) ** 2 for x in fast) / (nf - 1) / nf) if nf > 1 else None
            out.append({
                "fast_n": nf, "mark_fast": f_mean, "mark_fast_low": f_mean - 1.96 * f_se if f_se is not None else None,
                "fast_edge": bool(nf >= 50 and f_se is not None and f_mean - 1.96 * f_se > 0),
                "kind": kind, "leader": leader, "follower": follower, "minutes": minutes, "events": n,
                "followed": len(lags) / n, "lag_median": lags[len(lags) // 2] if lags else None,
                "fair_followed": len(flags) / n if kind == "cross" else None,
                "fair_lag_median": flags[len(flags) // 2] if flags and kind == "cross" else None,
                "market_first": (sum(1 for e in es if e["market_first"]) / n) if kind == "self" else None,
                "mark_mid": mean, "mark_mid_low": mean - 1.96 * se if se is not None else None,
                "mark_bid": sum(bids) / len(bids) if bids else None,
                "edge_found": bool(n >= 50 and se is not None and mean - 1.96 * se > 0
                                   and bids and sum(bids) / len(bids) > 0),
            })
        out.sort(key=lambda g: (g["kind"] != "self", -(g["mark_mid_low"] if g["mark_mid_low"] is not None else -9)))
        return {"groups": out, "events": sum(g["events"] for g in out), "live": len(self.live),
                "settings": {"jump": JUMP, "look": LOOK, "delay": DELAY, "hold": HOLD, "fast_delay": self.fast_delay}}
