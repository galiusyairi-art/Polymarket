"""SQLite storage: one row per asset per second, one per window, one per paper bet."""
import os
import sqlite3

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
    id INTEGER PRIMARY KEY, ts REAL, asset TEXT, window_start INTEGER, side TEXT,
    seconds_left REAL, fair_prob REAL, best_ask REAL, avg_price REAL,
    shares REAL, cost REAL, fee REAL, edge REAL, winner TEXT, pnl REAL
);
"""


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # The recorder writes from several threads; sqlite serialises the writes.
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def add_tick(self, **row):
        self.db.execute(
            "INSERT INTO ticks VALUES (:ts, :asset, :window_start, :price, :open_price, :seconds_left,"
            " :vol, :fair_up, :up_bid, :up_ask, :down_bid, :down_ask)",
            row,
        )

    def upsert_window(self, asset, window_start, **fields):
        self.db.execute(
            "INSERT OR IGNORE INTO windows (asset, window_start) VALUES (?, ?)", (asset, window_start)
        )
        for key, value in fields.items():
            self.db.execute(
                f"UPDATE windows SET {key} = ? WHERE asset = ? AND window_start = ?",
                (value, asset, window_start),
            )

    def add_bet(self, ts, asset, window_start, seconds_left, fill):
        self.db.execute(
            "INSERT INTO bets (ts, asset, window_start, side, seconds_left, fair_prob, best_ask,"
            " avg_price, shares, cost, fee, edge) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, asset, window_start, fill.side, seconds_left, fill.fair_prob, fill.best_ask,
             fill.avg_price, fill.shares, fill.cost, fill.fee, fill.edge),
        )

    def unresolved_windows(self, before_ts):
        return self.db.execute(
            "SELECT asset, window_start, slug FROM windows"
            " WHERE found = 1 AND winner IS NULL AND window_start < ?",
            (before_ts,),
        ).fetchall()

    def unsettled_bets(self, asset, window_start):
        return self.db.execute(
            "SELECT id, side, shares, cost, fee FROM bets"
            " WHERE asset = ? AND window_start = ? AND winner IS NULL",
            (asset, window_start),
        ).fetchall()

    def settle_bet(self, bet_id, winner, pnl):
        self.db.execute("UPDATE bets SET winner = ?, pnl = ? WHERE id = ?", (winner, pnl, bet_id))
