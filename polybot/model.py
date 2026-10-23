"""The math: how likely is "Up" given where the price is now and how much time is left."""
import math
from collections import deque


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fair_up_probability(price, open_price, seconds_left, vol_per_sqrt_second):
    """Chance the price ends at or above open_price, assuming a driftless random walk.

    vol_per_sqrt_second is the standard deviation of log returns over one second.
    """
    if seconds_left <= 0 or vol_per_sqrt_second <= 0:
        return 1.0 if price >= open_price else 0.0
    spread = vol_per_sqrt_second * math.sqrt(seconds_left)
    return norm_cdf(math.log(price / open_price) / spread)


def fair_up_probability_twap(price, open_price, seconds_left, vol_per_sqrt_second, twap_seconds,
                             realized_avg=None):
    """Chance that the average price over the window's last twap_seconds ends at or above open_price.

    Polymarket's short crypto markets resolve on a Chainlink 60-second TWAP, so the
    last minute is partly decided already: realized_avg is the average so far
    within that last minute (None before it starts).
    """
    L = twap_seconds
    if seconds_left <= 0:
        final = realized_avg if realized_avg is not None else price
        return 1.0 if final >= open_price else 0.0
    if vol_per_sqrt_second <= 0:
        return 1.0 if price >= open_price else 0.0
    if seconds_left >= L:
        # Average of a random walk over [T-L, T], seen from now: variance sigma^2 * (tau - 2L/3).
        spread = vol_per_sqrt_second * math.sqrt(seconds_left - 2 * L / 3)
        return norm_cdf(math.log(price / open_price) / spread)
    done = (L - seconds_left) / L
    realized = price if realized_avg is None else realized_avg
    mean = done * realized + (1 - done) * price
    # The rest of the average is a random walk's mean over seconds_left: variance tau/3.
    spread = (1 - done) * price * vol_per_sqrt_second * math.sqrt(seconds_left / 3)
    return norm_cdf((mean - open_price) / spread)


def taker_fee_per_share(price, fee_rate, fee_exponent):
    return price * fee_rate * (price * (1.0 - price)) ** fee_exponent


class VolatilityEstimator:
    """Rolling realized volatility from irregularly spaced price samples.

    Every operation the hot path uses is O(1) or touches only the samples it needs:
    the engine re-evaluates on every order-book change, hundreds of times a second.
    """

    def __init__(self, lookback_seconds):
        self.lookback = lookback_seconds
        self.samples = deque()  # (ts, price)
        self._terms = deque()  # per sample: (squared log return, dt) from the previous sample
        self._sum_sq = 0.0
        self._sum_dt = 0.0

    def add(self, ts, price):
        if self.samples and ts <= self.samples[-1][0]:
            return
        if self.samples:
            prev_ts, prev_price = self.samples[-1]
            # Gaps (missed updates, restarts) are capped so one stale jump
            # doesn't count as a long quiet period.
            term = (math.log(price / prev_price) ** 2, min(ts - prev_ts, 10.0))
        else:
            term = (0.0, 0.0)
        self.samples.append((ts, price))
        self._terms.append(term)
        self._sum_sq += term[0]
        self._sum_dt += term[1]
        while self.samples[0][0] < ts - self.lookback:
            self.samples.popleft()
            self._terms.popleft()
            # The new first sample's term pointed at the one just dropped.
            sq, dt = self._terms[0]
            self._sum_sq -= sq
            self._sum_dt -= dt
            self._terms[0] = (0.0, 0.0)

    def history_seconds(self):
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1][0] - self.samples[0][0]

    def price_at(self, t):
        """The last price at or before t, or None."""
        for ts, price in reversed(self.samples):
            if ts <= t:
                return price
        return None

    def twap(self, t0, t1):
        """Time-weighted average price over [t0, t1], or None without history reaching back to t0."""
        if not self.samples or self.samples[0][0] > t0 + 5 or t1 <= t0:
            return None
        # Walk back from the newest sample to the one in force at t0.
        inside, start_price = [], None
        for ts, price in reversed(self.samples):
            if ts <= t0:
                start_price = price
                break
            if ts < t1:
                inside.append((ts, price))
        inside.reverse()
        if start_price is None:
            if not inside:
                return None
            start_price = inside[0][1]  # first sample landed just after t0
        acc, cur_t, cur_p = 0.0, t0, start_price
        for ts, price in inside:
            acc += cur_p * (ts - cur_t)
            cur_t, cur_p = ts, price
        acc += cur_p * (t1 - cur_t)
        return acc / (t1 - t0)

    def per_sqrt_second(self):
        """Standard deviation of log returns over one second, or None without data."""
        if self._sum_dt <= 0:
            return None
        return math.sqrt(max(self._sum_sq, 0.0) / self._sum_dt)
