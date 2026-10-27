"""SQLite storage. A market is an (asset, minutes) pair, e.g. BTC in 15-minute windows.

ticks:    one row per market per second (lab-only markets: every few seconds)
windows:  one row per market window, with both readings of its open/close and the real winner
bets:     paper ("paper" = the main strategy, "lab" = experiments) and real ("live") bets
research: Claude's written research summaries
"""
import json
import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    ts REAL, asset TEXT, minutes INTEGER, window_start INTEGER, price REAL, open_price REAL,
    seconds_left REAL, vol REAL, fair_up REAL,
    up_bid REAL, up_ask REAL, down_bid REAL, down_ask REAL,
    price_age REAL, book_age REAL
);
CREATE INDEX IF NOT EXISTS ticks_window ON ticks (asset, minutes, window_start);
CREATE INDEX IF NOT EXISTS ticks_ts ON ticks (ts);
CREATE TABLE IF NOT EXISTS windows (
    asset TEXT, minutes INTEGER, window_start INTEGER, slug TEXT, found INTEGER,
    open_price REAL, open_source TEXT, winner TEXT,
    open_twap REAL, close_twap REAL, open_point REAL, close_point REAL,
    PRIMARY KEY (asset, minutes, window_start)
);
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY, mode TEXT, variant TEXT, status TEXT, order_id TEXT, error TEXT,
    ts REAL, asset TEXT, minutes INTEGER, window_start INTEGER, side TEXT,
    seconds_left REAL, fair_prob REAL, best_ask REAL, avg_price REAL,
    shares REAL, cost REAL, fee REAL, edge REAL, winner TEXT, pnl REAL,
    price_age REAL, book_age REAL, notified INTEGER DEFAULT 0, fair_raw REAL, market TEXT,
    stake REAL,  -- what the system meant to spend; cost + fee is what the order book let it buy
    book TEXT  -- JSON [[price, shares], ...]: every level the rule would have bought, to replay other stakes
);
CREATE INDEX IF NOT EXISTS bets_window ON bets (asset, minutes, window_start);
CREATE INDEX IF NOT EXISTS bets_mode ON bets (mode, ts);
-- (one rule's trades among the lab's millions: a rule's results, the real-money rules' summary)
CREATE INDEX IF NOT EXISTS bets_variant ON bets (variant, mode);
-- (the trades not settled yet, the few among millions: the page counts the open ones every few seconds)
CREATE INDEX IF NOT EXISTS bets_unsettled ON bets (status, mode) WHERE pnl IS NULL;
CREATE TABLE IF NOT EXISTS trades (
    ts REAL, asset TEXT, minutes INTEGER, window_start INTEGER, outcome TEXT,  -- the side traded: up / down
    price REAL, size REAL, taker TEXT, seconds_left REAL  -- taker: BUY (they bought it) or SELL
);
CREATE INDEX IF NOT EXISTS trades_window ON trades (asset, minutes, window_start);
CREATE INDEX IF NOT EXISTS trades_ts ON trades (ts);
-- A guest venue, read only (guest.py): Limitless's short crypto rounds, their books and trades,
-- in the shape of windows / ticks / trades so the market and maker checks run on them alike.
CREATE TABLE IF NOT EXISTS lx_windows (
    slug TEXT PRIMARY KEY, asset TEXT, minutes INTEGER, window_start INTEGER, title TEXT, winner TEXT
);
CREATE TABLE IF NOT EXISTS lx_ticks (
    ts REAL, asset TEXT, minutes INTEGER, window_start INTEGER, seconds_left REAL,
    up_bid REAL, up_ask REAL, down_bid REAL, down_ask REAL, bid_size REAL, ask_size REAL, last_price REAL, slug TEXT
);
CREATE INDEX IF NOT EXISTS lx_ticks_window ON lx_ticks (asset, minutes, window_start);
CREATE TABLE IF NOT EXISTS lx_trades (
    ts REAL, asset TEXT, minutes INTEGER, window_start INTEGER, outcome TEXT,
    price REAL, size REAL, taker TEXT, seconds_left REAL
);
CREATE INDEX IF NOT EXISTS lx_trades_window ON lx_trades (asset, minutes, window_start);
-- An order of the lab or the simulation that the book refused when it arrived (fill-or-kill): what it
-- wanted, what was left at its price by then, and whether 1 or 2 cents more would have filled it and at
-- what cost (fills.py: why orders don't go through, and whether a little room pays).
CREATE TABLE IF NOT EXISTS fill_tests (
    ts REAL, asset TEXT, minutes INTEGER, window_start INTEGER, mode TEXT, variant TEXT, side TEXT,
    want REAL, limit_price REAL, left_at_limit REAL,
    cost1 REAL, fee1 REAL, cost2 REAL, fee2 REAL  -- (NULL: even with that room, not all of it)
);
CREATE INDEX IF NOT EXISTS fill_tests_ts ON fill_tests (ts);
-- Speed (fills.py, speed()): each real order's timing (the signal, the order sent, Polymarket's
-- answer), and the network's time to Polymarket's order book (a connection's round trip).
CREATE TABLE IF NOT EXISTS order_timing (ts REAL, asset TEXT, minutes INTEGER, status TEXT, decide REAL, answer REAL);
CREATE TABLE IF NOT EXISTS net_ping (ts REAL, host TEXT, ms REAL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS research (id INTEGER PRIMARY KEY, ts REAL, model TEXT, data TEXT, error TEXT, cost REAL);
"""


def connect(path):
    """A separate read connection, for threads that only query (dashboard, analysis)."""
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def _book_json(depth):
    """The levels a rule would have bought, compact: the cheapest 40, prices to 0.1¢."""
    if not depth:
        return None
    return json.dumps([[round(p, 4), round(q, 2)] for p, q in depth[:40]], separators=(",", ":"))


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        # Columns added after a database was first created.
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(bets)")}
        if "notified" not in columns:
            self.db.execute("ALTER TABLE bets ADD COLUMN notified INTEGER DEFAULT 0")
        if "fair_raw" not in columns:
            self.db.execute("ALTER TABLE bets ADD COLUMN fair_raw REAL")
        if "market" not in columns:
            self.db.execute("ALTER TABLE bets ADD COLUMN market TEXT")
        if "stake" not in columns:
            self.db.execute("ALTER TABLE bets ADD COLUMN stake REAL")
        if "book" not in columns:
            self.db.execute("ALTER TABLE bets ADD COLUMN book TEXT")
        for name in ("mail_open", "mail_close"):  # each trade mail's outcome: JSON {ts, ok, why}
            if name not in columns:
                self.db.execute(f"ALTER TABLE bets ADD COLUMN {name} TEXT")
        if "fair_raw" not in {row[1] for row in self.db.execute("PRAGMA table_info(ticks)")}:
            self.db.execute("ALTER TABLE ticks ADD COLUMN fair_raw REAL")
        if "cost" not in {row[1] for row in self.db.execute("PRAGMA table_info(research)")}:
            self.db.execute("ALTER TABLE research ADD COLUMN cost REAL")
        self.lock = threading.Lock()

    def _x(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def close(self):
        self.db.close()

    def add_tick(self, **row):
        self._x(
            "INSERT INTO ticks (ts, asset, minutes, window_start, price, open_price, seconds_left, vol, fair_up,"
            " up_bid, up_ask, down_bid, down_ask, price_age, book_age, fair_raw) VALUES (:ts, :asset, :minutes,"
            " :window_start, :price, :open_price, :seconds_left, :vol, :fair_up, :up_bid, :up_ask, :down_bid,"
            " :down_ask, :price_age, :book_age, :fair_raw)",
            dict({"fair_raw": None}, **row),
        )

    def prune_ticks(self, before_ts):
        self._x("DELETE FROM ticks WHERE ts < ?", (before_ts,))
        self._x("DELETE FROM trades WHERE ts < ?", (before_ts,))
        self._x("DELETE FROM lx_ticks WHERE ts < ?", (before_ts,))
        self._x("DELETE FROM lx_trades WHERE ts < ?", (before_ts,))
        self._x("DELETE FROM fill_tests WHERE ts < ?", (before_ts - 30 * 86400,))  # (small: kept longer)
        self._x("DELETE FROM net_ping WHERE ts < ?", (before_ts,))

    def add_order_timing(self, row):
        self._x("INSERT INTO order_timing VALUES (?, ?, ?, ?, ?, ?)", row)

    def add_ping(self, ts, host, ms):
        self._x("INSERT INTO net_ping VALUES (?, ?, ?)", (ts, host, ms))

    def add_fill_test(self, row):
        self._x("INSERT INTO fill_tests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)

    def add_trades(self, rows, table="trades"):
        """Trades that happened in the markets (anyone's): (ts, asset, minutes, window_start, outcome,
        price, size, taker, seconds_left). table: lx_trades for the guest venue."""
        if rows:
            with self.lock:
                self.db.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def upsert_window(self, asset, minutes, window_start, **fields):
        key = (asset, minutes, window_start)
        self._x("INSERT OR IGNORE INTO windows (asset, minutes, window_start) VALUES (?, ?, ?)", key)
        for name, value in fields.items():
            self._x(f"UPDATE windows SET {name} = ? WHERE asset = ? AND minutes = ? AND window_start = ?",
                    (value, *key))

    def add_bet(self, ts, asset, minutes, window_start, seconds_left, fill, mode="paper", variant="main",
                status="filled", order_id=None, error=None, price_age=None, book_age=None, fair_raw=None,
                market=None, stake=None):
        self._x(
            "INSERT INTO bets (mode, variant, status, order_id, error, ts, asset, minutes, window_start, side,"
            " seconds_left, fair_prob, best_ask, avg_price, shares, cost, fee, edge, price_age, book_age, fair_raw,"
            " market, stake, book) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mode, variant, status, order_id, error, ts, asset, minutes, window_start, fill.side, seconds_left,
             fill.fair_prob, fill.best_ask, fill.avg_price, fill.shares, fill.cost, fill.fee, fill.edge,
             price_age, book_age, fair_raw, market, stake, _book_json(getattr(fill, "depth", None))),
        )

    def unresolved_windows(self, before_ts, after_ts):
        return self._x(
            "SELECT asset, minutes, window_start, slug FROM windows"
            " WHERE found = 1 AND winner IS NULL AND window_start < ? AND window_start > ?",
            (before_ts, after_ts),
        )

    def unsettled_bets(self, asset, minutes, window_start):
        return self._x(
            "SELECT id, side, shares, cost, fee FROM bets WHERE asset = ? AND minutes = ? AND window_start = ?"
            " AND status = 'filled' AND winner IS NULL",
            (asset, minutes, window_start),
        )

    def settle_bet(self, bet_id, winner, pnl):
        self._x("UPDATE bets SET winner = ?, pnl = ? WHERE id = ?", (winner, pnl, bet_id))

    def get_setting(self, key, default=None):
        rows = self._x("SELECT value FROM settings WHERE key = ?", (key,))
        return rows[0][0] if rows else default

    def set_setting(self, key, value):
        if value is None:  # cleared: gone, never the text "None"
            self._x("DELETE FROM settings WHERE key = ?", (key,))
            return
        self._x("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, str(value)))

    def add_research(self, ts, model, data=None, error=None, cost=None):
        self._x("INSERT INTO research (ts, model, data, error, cost) VALUES (?, ?, ?, ?, ?)",
                (ts, model, json.dumps(data, ensure_ascii=False) if data is not None else None, error, cost))
