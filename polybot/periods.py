"""Calendar periods (today, this week, this month) in the owner's time zone, not the server's."""
import datetime as dt
from zoneinfo import ZoneInfo


def period_starts(now, tz):
    """Unix start of the current day, week (Sunday, as in Israel) and month in time zone tz."""
    local = dt.datetime.fromtimestamp(now, ZoneInfo(tz))
    day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    week = day - dt.timedelta(days=(day.weekday() + 1) % 7)
    month = day.replace(day=1)
    return {"day": day.timestamp(), "week": week.timestamp(), "month": month.timestamp()}
