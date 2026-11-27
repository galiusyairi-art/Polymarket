"""The US stock market's days: which are trading days, and when each closes.

Until 27.11.2026 the system took every weekday for a trading day. On Thanksgiving (26.11.2026) that cost the owner his
real orders: the entries sent on the evening of the 25th for the next open were judged expired at "the close" of the
26th, a day with no session, and cancelled at IBKR; and on the 27th the day's signals waited for the 26th's data, which
can never come, so nothing was sent. The NYSE's own calendar, by its published rules (a holiday on a Saturday is kept
the Friday before, on a Sunday the Monday after; New Year's Day on a Saturday is not kept), with its early closes at
13:00. Extend it each year."""
import datetime as dt

HOLIDAYS = {dt.date(*d) for d in (
    # 2025
    (2025, 1, 1), (2025, 1, 9), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18), (2025, 5, 26), (2025, 6, 19), (2025, 7, 4),
    (2025, 9, 1), (2025, 11, 27), (2025, 12, 25),
    # 2026
    (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7),
    (2026, 11, 26), (2026, 12, 25),
    # 2027
    (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26), (2027, 5, 31), (2027, 6, 18), (2027, 7, 5), (2027, 9, 6),
    (2027, 11, 25), (2027, 12, 24),
    # 2028 (New Year's Day 2028 is a Saturday: not kept)
    (2028, 1, 17), (2028, 2, 21), (2028, 4, 14), (2028, 5, 29), (2028, 6, 19), (2028, 7, 4), (2028, 9, 4), (2028, 11, 23),
    (2028, 12, 25),
)}
# The days the market closes at 13:00 New York.
EARLY = {dt.date(*d) for d in (
    (2025, 7, 3), (2025, 11, 28), (2025, 12, 24),
    (2026, 11, 27), (2026, 12, 24),
    (2027, 11, 26),
    (2028, 7, 3), (2028, 11, 24),
)}
CLOSE = (16, 0)
EARLY_CLOSE = (13, 0)


def is_trading_day(d):
    """A New York date the market trades on."""
    return d.weekday() < 5 and d not in HOLIDAYS


def close_hm(d):
    """When the market closes on that date, (hour, minute) New York."""
    return EARLY_CLOSE if d in EARLY else CLOSE


def prev_trading_day(d):
    """The last trading day before d."""
    d -= dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def next_trading_day(d):
    """The first trading day after d."""
    d += dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def in_session(t, start=(9, 30), end=None):
    """Whether the New York moment t is within a trading day's hours, from start to its close (or end)."""
    return is_trading_day(t.date()) and start <= (t.hour, t.minute) < (end or close_hm(t.date()))
