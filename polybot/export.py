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
import os
import subprocess
import sys
import threading
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
  notified, mail_open, mail_close: email bookkeeping (each trade mail: when, sent or not, and why not).

windows.csv - every market window: winner, the opening and closing prices (twap = 60-second
  average, the reading Polymarket resolves on; point = a single price), and where they came from.

ticks.csv (full export only, the last 24 hours) - the recording: price, the model's chance
  for Up, both sides' best bid and ask, every second or so, per market.

research.csv - Claude's daily research summaries (JSON in "data").
settings.json - every setting as stored (strategies, versions, limits, risk/reward, mail).
manifest.json - when it was exported, the version, row counts, and the configuration.
"""


# The light file (the owner, 18.11.2026: a file he can hand over in the chat; the whole one is hundreds of megabytes):
# every trade of the simulation and of real money in the last LIGHT_DAYS days, and every lab trade of the same rules in the
# same markets (and of the versions made from them: a rule's key and whatever follows it), without the heavy columns.
LIGHT_DAYS = 45
LIGHT_DROP = ("book", "notified", "order_id", "mail_open", "mail_close")


def light_tables(db, now=None):
    """{file: (query, args)} of the light export."""
    since = (now or time.time()) - LIGHT_DAYS * 86400
    cols = ", ".join(c[1] for c in db.execute("PRAGMA table_info(bets)") if c[1] not in LIGHT_DROP)
    rules = db.execute("SELECT DISTINCT variant, asset, minutes FROM bets WHERE mode IN ('paper', 'live') AND ts >= ? "
                       "AND variant IS NOT NULL", (since,)).fetchall()
    out = {"bets": (f"SELECT {cols} FROM bets WHERE mode IN ('paper', 'live') AND ts >= ? ORDER BY id", (since,))}
    # (by the rule's own index: left to itself SQLite went by the market's, and read every one of the market's hundreds of
    # thousands of lab trades for each rule: minutes on the server, 18.11.2026)
    by = " INDEXED BY bets_variant" if db.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'bets_variant'").fetchone() else ""
    seen = set()
    for i, (variant, asset, minutes) in enumerate(sorted(rules, key=lambda r: (str(r[0]), str(r[1]), r[2] or 0))):
        if (variant, asset, minutes) in seen or not variant:
            continue
        seen.add((variant, asset, minutes))
        # (the lab's own copy of the rule, and the versions made from it: the same key with more after it)
        out[f"lab_bets_{i}"] = (f"SELECT {cols} FROM bets{by} WHERE variant >= ? AND variant < ? AND mode NOT IN ('paper', 'live') "
                                "AND asset = ? AND minutes = ? AND ts >= ? ORDER BY id",
                                (variant, variant + "\uffff", asset, minutes, since))
    out["windows"] = ("SELECT * FROM windows WHERE window_start >= ? ORDER BY window_start, asset, minutes", (since,))
    if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'fill_tests'").fetchone():
        out["fill_tests"] = ("SELECT * FROM fill_tests WHERE ts >= ? ORDER BY ts", (since,))
    return out


# The prices file (the owner, 23.11.2026: the whole file with the recorded prices came to 32 megabytes, too large to hand
# over in the chat): the simulation's and real money's trades as in the light file, and the recorded seconds of the last
# TICKS_HOURS hours of the markets they traded in that time only, with those hours' windows. What a study of getting out
# of a trade early needs, in a few megabytes.
def prices_tables(db, now=None):
    """{file: (query, args)} of the prices export."""
    now = now or time.time()
    day = now - TICKS_HOURS * 3600
    cols = ", ".join(c[1] for c in db.execute("PRAGMA table_info(bets)") if c[1] not in LIGHT_DROP)
    out = {"bets": (f"SELECT {cols} FROM bets WHERE mode IN ('paper', 'live') AND ts >= ? ORDER BY id", (now - LIGHT_DAYS * 86400,))}
    markets = db.execute("SELECT DISTINCT asset, minutes FROM bets WHERE mode IN ('paper', 'live') AND ts >= ? AND minutes > 0", (day,)).fetchall()
    for i, (asset, minutes) in enumerate(sorted(markets)):
        # (by the market's own index: its rounds of the last day, not two weeks of every market's seconds)
        out[f"ticks_{i}"] = ("SELECT * FROM ticks WHERE asset = ? AND minutes = ? AND window_start >= ? AND ts >= ? ORDER BY ts",
                             (asset, minutes, int(day) - 3600, day))
    out["windows"] = ("SELECT * FROM windows WHERE window_start >= ? ORDER BY window_start, asset, minutes", (int(day) - 3600,))
    return out


JOINED = ("lab_bets", "ticks")  # files written from several queries, one after another ("lab_bets_3" goes into lab_bets.csv)


def write(cfg, out, version="", full=False, extra=None, light=False, progress=None, db_path=None, config=None, prices=False):
    """Write the export ZIP to `out` (a path or a binary file). Returns the row counts. light: the small file
    (light_tables); prices: the trades and the recorded prices of their markets (prices_tables). progress(rows): called
    as the rows are written. db_path and config: in place of cfg (the build in a process of its own, in_process)."""
    db = connect(db_path or cfg.db_path)
    counts = {}
    done = 0
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            since = time.time() - TICKS_HOURS * 3600
            if prices:
                tables = prices_tables(db)
            elif light:
                tables = light_tables(db)
            else:
                tables = {k: (q, ()) for k, q in dict(TABLES, **({"ticks": TICKS.format(since=since)} if full else {})).items()}
            lab_open = joined = None  # (the light file's lab trades, the prices file's seconds: one file, one after another)
            for name, (query, args) in tables.items():
                group = next((g for g in JOINED if name.startswith(g + "_")), None)
                if lab_open is not None and group != joined:
                    lab_text.flush()
                    lab_text.detach()
                    lab_open.close()
                    lab_open = None
                if group:
                    cursor = db.execute(query, args)
                    if lab_open is None:
                        joined = group
                        lab_open = z.open(f"{group}.csv", "w", force_zip64=True)
                        lab_text = io.TextIOWrapper(lab_open, encoding="utf-8", newline="")
                        lab_writer = csv.writer(lab_text)
                        lab_writer.writerow([c[0] for c in cursor.description])
                        counts[group] = 0
                    for row in cursor:
                        lab_writer.writerow(["" if v is None else v for v in row])
                        counts[group] += 1
                        done += 1
                        if progress and done % 5000 == 0:
                            progress(done)
                    continue
                with z.open(f"{name}.csv", "w", force_zip64=True) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                    writer = csv.writer(text)
                    cursor = db.execute(query, args)
                    writer.writerow([c[0] for c in cursor.description])
                    n = 0
                    for row in cursor:
                        writer.writerow(["" if v is None else v for v in row])
                        n += 1
                        done += 1
                        if progress and done % 5000 == 0:
                            progress(done)
                    text.flush()
                    text.detach()
                    counts[name] = n
            if lab_open is not None:
                lab_text.flush()
                lab_text.detach()
                lab_open.close()
            settings = {}
            for key, value in db.execute("SELECT key, value FROM settings ORDER BY key"):
                try:
                    settings[key] = json.loads(value)
                except (TypeError, ValueError):
                    settings[key] = value
            z.writestr("settings.json", json.dumps(settings, ensure_ascii=False, indent=1))
            config = {k: v for k, v in (config if config is not None else dataclasses.asdict(cfg)).items() if k != "db_path"}
            z.writestr("manifest.json", json.dumps(
                dict({"exported_at": time.time(), "exported_at_text": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                      "version": version, "full": full, "light": light, "prices": prices, "rows": counts, "config": config},
                     **(extra or {})),
                ensure_ascii=False, indent=1, default=str))
            z.writestr("README.txt", README + (PRICES_README if prices else LIGHT_README if light else ""))
    finally:
        db.close()
    return counts


LIGHT_README = f"""
THE LIGHT FILE: the last {LIGHT_DAYS} days only, without the heavy columns ({", ".join(LIGHT_DROP)}).
bets.csv - the simulation's and real money's trades (mode paper / live).
lab_bets.csv - every lab trade of the same rules in the same markets, and of the versions made from them
  (a rule's key with more after it), on the lab's own fills.
fill_tests.csv - orders the book refused when they arrived, and whether a cent or two more would have filled them.
"""


PRICES_README = """
THE PRICES FILE: small enough to hand over in a chat.
bets.csv - the simulation's and real money's trades (mode paper / live), the last 45 days, without the heavy columns.
ticks.csv - the recording (price, the model's chance for Up, both sides' best bid and ask, every second or so) of the
  last 24 hours, only of the markets the simulation or real money traded in those hours.
windows.csv - the windows of those hours.
"""


def in_process(cfg, path, kind, progress, version="", extra=None):
    """The file built in a process of its own, at a low priority (a million rows written inside the bot's own process
    held its one interpreter for minutes: every order and every page waited their turn). Its progress comes back a line
    at a time. Returns the row counts."""
    spec = path + ".json"
    with open(spec, "w") as f:
        json.dump({"db": cfg.db_path, "out": path, "kind": kind, "version": version, "extra": extra or {},
                   "config": dataclasses.asdict(cfg)}, f, default=str)
    try:
        proc = subprocess.Popen([sys.executable, "-m", "polybot.export", spec], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        last = ""
        for line in proc.stdout:
            line = line.strip()
            if line.isdigit():
                progress(int(line))
            elif line:
                last = line
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError((err.strip().splitlines() or ["the export's process failed"])[-1][:300])
        return json.loads(last)
    finally:
        try:
            os.remove(spec)
        except OSError:
            pass


def main(argv):
    with open(argv[1]) as f:
        spec = json.load(f)
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    counts = write(None, spec["out"], version=spec["version"], full=spec["kind"] == "full", light=spec["kind"] == "light",
                   prices=spec["kind"] == "prices", extra=spec["extra"], progress=lambda n: print(n, flush=True), db_path=spec["db"], config=spec["config"])
    print(json.dumps(counts), flush=True)


class Job:
    """The download, made behind the page (the owner, 18.11.2026: "a click there does nothing": the whole file was built
    before the first byte was sent, a minute and more with nothing to see, and the connection gave up). One at a time: the
    click starts it, the page shows the rows written, and the file is fetched when it is ready. Kept KEEP seconds."""
    KEEP = 2 * 3600
    KINDS = ("light", "trades", "full", "prices")

    def __init__(self, folder, build):
        self.folder, self.build = folder, build  # build(path, kind, progress) -> row counts
        self.lock = threading.Lock()
        self.job = None

    def _clean(self, keep=None):
        try:
            for name in os.listdir(self.folder):
                path = os.path.join(self.folder, name)
                if path != keep and name.endswith(".zip"):
                    os.remove(path)
        except OSError:
            pass

    def start(self, kind, total=None, now=None):
        if kind not in self.KINDS:
            raise ValueError("bad request")
        with self.lock:
            if self.job and self.job["state"] == "building":
                return self.status(now)
            os.makedirs(self.folder, exist_ok=True)
            self._clean()
            name = time.strftime("polybot-%Y%m%d-%H%M", time.gmtime()) + {"light": "-research", "full": "-full", "prices": "-prices"}.get(kind, "") + ".zip"
            job = self.job = {"kind": kind, "state": "building", "rows": 0, "total": total, "started": now or time.time(),
                              "name": name, "path": os.path.join(self.folder, name), "size": None, "error": None, "done": None}

        def go():
            try:
                counts = self.build(job["path"], kind, lambda n: job.__setitem__("rows", n))
                job.update(state="ready", rows=sum(counts.values()), size=os.path.getsize(job["path"]), done=time.time(), counts=counts)
            except Exception as e:
                job.update(state="error", error=f"{type(e).__name__}: {e}", done=time.time())
                self._clean()
        threading.Thread(target=go, daemon=True, name="export").start()
        return self.status(now)

    def status(self, now=None):
        j = self.job
        if not j:
            return {"state": "idle"}
        if j["state"] == "ready" and (now or time.time()) - j["done"] > self.KEEP:
            self._clean()
            self.job = None
            return {"state": "idle"}
        return {k: j.get(k) for k in ("kind", "state", "rows", "total", "started", "name", "size", "error", "done")}

    def file(self):
        j = self.job
        return (j["path"], j["name"]) if j and j["state"] == "ready" and os.path.exists(j["path"]) else None


# Settings that describe this machine's moment, not the owner's choices: never carried over.
# (Real money always arrives switched off: the owner switches it on, on the machine that trades.)
NOT_IMPORTED = {"live_armed", "live_stopped", "live_daily_keep", "notify_lab_pending", "notify_rr_pending",
                "notify_cand_pending"}


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


if __name__ == "__main__":
    main(sys.argv)
