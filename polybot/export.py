"""Everything the system recorded, as one download the owner can hand over for research.

A ZIP of CSV files, one per table with every column, numbers written in full precision (a
float is written exactly as Python stores it, so nothing is rounded away), plus the settings,
a manifest and a README that explains each column. Nothing secret is in it: the wallet key
and the passwords live only in the server's environment, never in the database.
"""
import csv
import dataclasses
import io
import json
import time
import zipfile

from .store import connect

# Small and always included: every trade (simulation, real money, lab, stock lab), every
# window's result, and Claude's research summaries.
TABLES = {
    "bets": "SELECT * FROM bets ORDER BY id",
    "windows": "SELECT * FROM windows ORDER BY window_start, asset, minutes",
    "research": "SELECT * FROM research ORDER BY id",
}
# Large: the price and order-book recording. Only in the full export, and only its last day
# (the server keeps two weeks: hundreds of megabytes, too much to download and pass around).
TICKS = "SELECT * FROM ticks WHERE ts >= {since} ORDER BY ts"
TICKS_HOURS = 24

README = """Polybot export · ייצוא של כל הנתונים

Files (UTF-8 CSV, one row per record, empty cell = no value; times are Unix seconds, UTC):

bets.csv - every trade the system made or simulated.
  mode: paper = the simulation's main strategies, live = real money, lab = crypto lab
        experiments, stocks = stock lab experiments. (paper with minutes = 0 is the stock strategy.)
  variant: the rule that made it ("twap|0.05|15-600" = model reading | minimum edge |
        seconds before the close it may enter; "stocks:..." for the stock strategy).
  status: filled, or rejected (real money only; error says why).
  ts: when it was entered. asset/minutes/window_start: the market (window_start = the window's
        opening time; for stocks it is the close). market: Polymarket's slug when known.
  side: up / down (for stocks up = yes). seconds_left: seconds to the close at entry.
  fair_prob: the system's chance for that side (calibrated); fair_raw: the model's own, before calibration.
  best_ask: the cheapest price on offer; avg_price: the average price paid per share.
  shares, cost (paid for shares), fee (Polymarket's fee), edge = fair_prob - (cost + fee) / shares.
  stake: what the system meant to spend (cost + fee is what the order book let it buy).
  book: JSON [[price, shares], ...], every level of the order book the rule would have bought at that
        moment (empty for older trades); used to replay the trade at other stakes.
  winner: the side that won (empty until resolved); pnl: profit in dollars (empty while open).
  price_age / book_age: how old the price and the order book were when it decided, in seconds.
  notified: email bookkeeping.

windows.csv - every market window: winner, the opening and closing prices (twap = 60-second
  average, the reading Polymarket resolves on; point = a single price), and where they came from.

ticks.csv (full export only, the last 24 hours) - the recording: price, the model's chance
  for Up, both sides' best bid and ask, every second or so, per market.

research.csv - Claude's daily research summaries (JSON in "data").
settings.json - every setting as stored (strategies, versions, limits, risk/reward, mail).
manifest.json - when it was exported, the version, row counts, and the configuration.
"""


def write(cfg, out, version="", full=False, extra=None):
    """Write the export ZIP to `out` (a path or a binary file). Returns the row counts."""
    db = connect(cfg.db_path)
    counts = {}
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            since = time.time() - TICKS_HOURS * 3600
            tables = dict(TABLES, **({"ticks": TICKS.format(since=since)} if full else {}))
            for name, query in tables.items():
                with z.open(f"{name}.csv", "w", force_zip64=True) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                    writer = csv.writer(text)
                    cursor = db.execute(query)
                    writer.writerow([c[0] for c in cursor.description])
                    n = 0
                    for row in cursor:
                        writer.writerow(["" if v is None else v for v in row])
                        n += 1
                    text.flush()
                    text.detach()
                    counts[name] = n
            settings = {}
            for key, value in db.execute("SELECT key, value FROM settings ORDER BY key"):
                try:
                    settings[key] = json.loads(value)
                except (TypeError, ValueError):
                    settings[key] = value
            z.writestr("settings.json", json.dumps(settings, ensure_ascii=False, indent=1))
            config = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "db_path"}
            z.writestr("manifest.json", json.dumps(
                dict({"exported_at": time.time(), "exported_at_text": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                      "version": version, "full": full, "rows": counts, "config": config}, **(extra or {})),
                ensure_ascii=False, indent=1, default=str))
            z.writestr("README.txt", README)
    finally:
        db.close()
    return counts


# Settings that describe this machine's moment, not the owner's choices: never carried over.
# (Real money always arrives switched off: the owner switches it on, on the machine that trades.)
NOT_IMPORTED = {"live_armed", "live_stopped", "live_daily_keep", "notify_lab_pending", "notify_rr_pending"}


def read(cfg, path, replace=True):
    """Load an export ZIP (from write above) into this machine's database, so it carries on
    where the other one was: the settings (strategies, names, money, versions...), every trade,
    every window, the research. replace: what this machine recorded so far goes first (its row
    numbers would collide with the imported ones). Returns the row counts."""
    db = connect(cfg.db_path)
    counts = {}
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "bets.csv" not in names or "settings.json" not in names:
                raise ValueError("זה לא קובץ ייצוא של המערכת")
            if replace:
                for table in ("bets", "windows", "research", "ticks"):
                    db.execute(f"DELETE FROM {table}")
            # (ticks: the last day of the price recording, in the full export; the market-price
            # mix is learned from it, so with it nothing has to be learned again)
            for table in ("bets", "windows", "research", "ticks"):
                if f"{table}.csv" not in names:
                    continue
                have = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                with z.open(f"{table}.csv") as raw:
                    rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                    header = next(rows, [])
                    cols = [c for c in header if c in have]
                    idx = [header.index(c) for c in cols]
                    sql = f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
                    n = 0
                    for row in rows:
                        db.execute(sql, [None if row[i] == "" else row[i] for i in idx])
                        n += 1
                    counts[table] = n
            if "settings.json" in names:
                settings = json.loads(z.read("settings.json").decode("utf-8"))
                for key, value in settings.items():
                    if key in NOT_IMPORTED:
                        continue
                    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, text))
                counts["settings"] = len(settings)
        db.commit()
    finally:
        db.close()
    return counts
