"""The weekly cleaning (the owner, 09.11.2026: "without hurting any strategy, of Atlas or Polybot: throw out and shrink all
the information that isn't important, every week on Saturday; what is important kept in an Excel table, where the system
can still reach it"; the database had grown so large that a start took hours).

Every Saturday the lab's trades older than LAB_DAYS (the bulk of the database: hundreds of experimental rules, every
round, every coin, each with its book) and the studies' shadow trades and order timings older than SHADOW_DAYS go into
compressed CSV files beside the database (archive/, Excel opens each once unzipped; the page downloads them as plain CSV),
and only then out of the database, a little at a time. Never touched: the simulation's and real money's trades, the
settings, what the strategies learned, Atlas; nor the lab's trades of any rule that runs in the simulation or with real
money, or is starred (their record is their proof). The recorded seconds keep their own LIVE window (keep_ticks_days, 14).

A file is written whole (to a temporary name, then renamed) before a single row of it is deleted: when the disk can't
take the file, nothing is deleted."""
import csv
import datetime as dt
import gzip
import io
import json
import logging
import os
import time
from zoneinfo import ZoneInfo

log = logging.getLogger("polybot")

LAB_DAYS = 30
SHADOW_DAYS = 30
SHADOW_MODES = ("lab_real", "paper_real", "guarded")
DAY = 5  # Saturday (Monday is 0)
KEEP_RUNS = 60


def folder(db_path):
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "archive")


def _write_month(store, path, table, span, args, start, end):
    """A month's rows into path (CSV, gzip), a day at a time (read under the store's lock, written outside it), whole or
    not at all. The number of rows."""
    tmp = path + ".part"
    n = 0
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
            w = None
            day = start
            while day < end:
                nxt = min(day + 86400, end)
                with store.lock:
                    cur = store.db.execute(f"SELECT * FROM {table} WHERE {span} AND ts >= ? AND ts < ? ORDER BY ts",
                                           list(args) + [day, nxt])
                    rows = cur.fetchall()
                    cols = [c[0] for c in cur.description]
                if w is None:
                    w = csv.writer(f)
                    w.writerow(cols)
                w.writerows(rows)
                n += len(rows)
                day = nxt
        if n:
            os.replace(tmp, path)
        else:
            os.remove(tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return n


def _months(lo, hi):
    """[(start, end, "YYYY-MM")] of the months between two timestamps (UTC)."""
    out = []
    d = dt.datetime.fromtimestamp(lo, dt.timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while d.timestamp() < hi:
        nxt = (d + dt.timedelta(days=32)).replace(day=1)
        out.append((d.timestamp(), min(nxt.timestamp(), hi), d.strftime("%Y-%m")))
        d = nxt
    return out


def kept_variants(engine):
    """The lab's rules whose old trades stay: in the simulation, with real money, or starred."""
    keep = set()
    for mode in ("sim", "live"):
        for x in engine.strategies_for(mode):
            keep.add(x["variant"])
    for k in engine.starred():
        parts = str(k).split("|", 2)
        if len(parts) == 3:
            keep.add(parts[2])
    return keep


def run(engine, now=None):
    """One cleaning. {"ts", "files": [{"name", "rows"}], "rows", "kept", "freed"}."""
    now = now or time.time()
    store = engine.store
    out_dir = folder(store.path)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d", time.gmtime(now))
    report = {"ts": now, "files": [], "rows": 0, "kept": 0}
    before = store.disk()
    keep = sorted(kept_variants(engine))
    report["kept"] = len(keep)
    jobs = []
    lab_cut = now - LAB_DAYS * 86400
    marks = ",".join("?" * len(keep)) or "''"
    jobs.append(("lab-trades", "bets", f"mode = 'lab' AND variant NOT IN ({marks})", list(keep), lab_cut))
    shadow_cut = now - SHADOW_DAYS * 86400
    jobs.append(("shadow-trades", "bets", f"mode IN ({','.join('?' * len(SHADOW_MODES))})", list(SHADOW_MODES), shadow_cut))
    jobs.append(("order-timing", "order_timing", "1 = 1", [], shadow_cut))
    for name, table, where, args, cut in jobs:
        lo = store._x(f"SELECT MIN(ts) FROM {table} WHERE {where} AND ts < ?", args + [cut])[0][0]
        if lo is None:
            continue
        for start, end, month in _months(lo, cut):
            span = f"{where} AND ts >= ? AND ts < ?"
            span_args = args + [start, end]
            path = os.path.join(out_dir, f"{name}-{month}-{stamp}.csv.gz")
            n = _write_month(store, path, table, where, args, start, end)
            if not n:
                continue
            deleted = store._delete(table, span, tuple(span_args))  # (only once its file is whole on the disk)
            report["files"].append({"name": os.path.basename(path), "rows": n, "deleted": deleted})
            report["rows"] += deleted
            log.warning("Weekly cleaning: %s %s: %d rows archived, %d out of the database", name, month, n, deleted)
    store.checkpoint()
    after = store.disk()
    report["freed"] = max(0, (after["reusable"] + after["free"]) - (before["reusable"] + before["free"]))
    report["db"] = after["db"]
    runs = json.loads(store.get_setting("archive_runs") or "[]")
    store.set_setting("archive_runs", json.dumps((runs + [report])[-KEEP_RUNS:]))
    return report


def due(store, now, tz):
    """Saturday (the owner's time) and not run yet this week."""
    t = dt.datetime.fromtimestamp(now, ZoneInfo(tz))
    week = t.strftime("%G-%V")
    return t.weekday() == DAY and store.get_setting("archive_week") != week, week


def files(db_path):
    """The archive's files, newest first: [{"name", "size", "ts"}]."""
    d = folder(db_path)
    if not os.path.isdir(d):
        return []
    out = []
    for n in os.listdir(d):
        if n.endswith(".csv.gz"):
            p = os.path.join(d, n)
            out.append({"name": n, "size": os.path.getsize(p), "ts": os.path.getmtime(p)})
    return sorted(out, key=lambda x: -x["ts"])


def read_csv(db_path, name):
    """One archive file's CSV text (for the page's download, and for any code that needs the old rows); None when no
    such file. Only a plain name inside the archive's folder."""
    if not name or "/" in name or "\\" in name or not name.endswith(".csv.gz"):
        return None
    p = os.path.join(folder(db_path), name)
    if not os.path.isfile(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
        return f.read()


def rows(db_path, name):
    """One archive file's rows as dicts (code reaching the old data)."""
    text = read_csv(db_path, name)
    return list(csv.DictReader(io.StringIO(text))) if text else []
