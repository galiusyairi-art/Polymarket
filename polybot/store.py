"""SQLite storage: one row per asset per second, one per window, one per bet (paper or live)."""
import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    ts REAL, asset TEXT, window_start INTEGER, price REAL, open_price REAL,
    seconds_left REAL, vol REAL, fair_up REAL,
    up_bid REAL, up_ask REAL, down_bid REAL, down_ask REAL
);
CREATE INDEX IF NOT EXISTS ticks_window ON ticks (asset, window_start);
CREATE TABLE IF NOT EXISTS windows (
    asset TEXT, window_start INTEGER, slug TEXT, found INTEGER,
    open_price REAL, open_source TEXT, our_end_price REAL, winner TEXT,
    PRIMARY KEY (asset, window_start)
);
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY, mode TEXT, status TEXT, order_id TEXT, error TEXT,
    ts REAL, asset TEXT, window_start INTEGER, side TEXT,
    seconds_left REAL, fair_prob REAL, best_ask REAL, avg_price REAL,
    shares REAL, cost REAL, fee REAL, edge REAL, winner TEXT, pnl REAL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path):
    """A separate read connection, for threads that only query (dashboard, analysis)."""
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    return db


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _x(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def add_tick(self, **row):
        self._x(
            "INSERT INTO ticks VALUES (:ts, :asset, :window_start, :price, :open_price, :seconds_left,"
            " :vol, :fair_up, :up_bid, :up_ask, :down_bid, :down_ask)",
            row,
        )

    def upsert_window(self, asset, window_start, **fields):
        self._x("INSERT OR IGNORE INTO windows (asset, window_start) VALUES (?, ?)", (asset, window_start))
        for key, value in fields.items():
            self._x(f"UPDATE windows SET {key} = ? WHERE asset = ? AND window_start = ?",
                    (value, asset, window_start))

    def add_bet(self, ts, asset, window_start, seconds_left, fill, mode="paper", status="filled",
                order_id=None, error=None):
        self._x(
            "INSERT INTO bets (mode, status, order_id, error, ts, asset, window_start, side, seconds_left,"
            " fair_prob, best_ask, avg_price, shares, cost, fee, edge)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mode, status, order_id, error, ts, asset, window_start, fill.side, seconds_left,
             fill.fair_prob, fill.best_ask, fill.avg_price, fill.shares, fill.cost, fill.fee, fill.edge),
        )

    def unresolved_windows(self, before_ts):
        return self._x(
            "SELECT asset, window_start, slug FROM windows"
            " WHERE found = 1 AND winner IS NULL AND window_start < ?",
            (before_ts,),
        )

    def unsettled_bets(self, asset, window_start):
        return self._x(
            "SELECT id, side, shares, cost, fee FROM bets"
            " WHERE asset = ? AND window_start = ? AND status = 'filled' AND winner IS NULL",
            (asset, window_start),
        )

    def settle_bet(self, bet_id, winner, pnl):
        self._x("UPDATE bets SET winner = ?, pnl = ? WHERE id = ?", (winner, pnl, bet_id))

    def close(self):
        self.db.close()

    def get_setting(self, key, default=None):
        rows = self._x("SELECT value FROM settings WHERE key = ?", (key,))
        return rows[0][0] if rows else default

    def set_setting(self, key, value):
        self._x("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, str(value)))
