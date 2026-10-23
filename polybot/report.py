"""Stage 1 results: did the strategy make money on paper, and is our probability better than Polymarket's?

    python -m polybot.report
"""
import argparse
import os
import sqlite3

from .config import Config
from .model import taker_fee_per_share


def brier(pairs):
    pairs = list(pairs)
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else None


def replay(rows, cfg, min_edge):
    """Re-run the entry rule at a different threshold using top-of-book prices.

    Optimistic: assumes the best ask had enough size for a 1-share bet.
    rows: (asset, window_start, seconds_left, fair_up, up_ask, down_ask, winner), time-ordered.
    Returns (bets, wins, pnl_per_share_total).
    """
    taken = set()
    bets = wins = 0
    pnl = 0.0
    for asset, start, left, fair_up, up_ask, down_ask, winner in rows:
        if (asset, start) in taken or not (cfg.min_seconds_left <= left <= cfg.max_seconds_left):
            continue
        for side, prob, ask in (("up", fair_up, up_ask), ("down", 1 - fair_up, down_ask)):
            if ask is None:
                continue
            fee = taker_fee_per_share(ask, cfg.fee_rate, cfg.fee_exponent)
            if prob - ask - fee >= min_edge:
                taken.add((asset, start))
                bets += 1
                won = side == winner
                wins += won
                pnl += (1.0 if won else 0.0) - ask - fee
                break
    return bets, wins, pnl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=Config.db_path)
    args = parser.parse_args()
    cfg = Config()
    if not os.path.exists(args.db):
        print("No data yet. Start the recorder with: python -m polybot.run")
        return
    db = sqlite3.connect(args.db)
    q = lambda sql, *a: db.execute(sql, a).fetchall()

    n, first, last = q("SELECT COUNT(*), MIN(ts), MAX(ts) FROM ticks")[0]
    if not n:
        print("No data yet. Start the recorder with: python -m polybot.run")
        return
    print(f"Recorded {n:,} samples over {(last - first) / 3600:.1f} hours.\n")

    # 1. Does our price feed agree with how Polymarket resolved?
    rows = q("SELECT open_price, our_end_price, winner FROM windows"
             " WHERE winner IS NOT NULL AND open_price IS NOT NULL AND our_end_price IS NOT NULL")
    if rows:
        agree = sum(("up" if end >= op else "down") == w for op, end, w in rows)
        print(f"Price feed check: our price agreed with Polymarket's result in {agree}/{len(rows)} windows"
              f" ({agree / len(rows):.0%}). Below ~97% means the feed is too different to trust.\n")

    # 2. Paper bets actually logged by the recorder.
    bets = q("SELECT asset, side, seconds_left, edge, cost + fee, pnl FROM bets WHERE pnl IS NOT NULL")
    pending = q("SELECT COUNT(*) FROM bets WHERE pnl IS NULL")[0][0]
    print(f"PAPER BETS (min edge {cfg.min_edge:.0%}, ${cfg.stake_usd:.0f} each): {len(bets)} settled, {pending} waiting")
    if bets:
        staked = sum(b[4] for b in bets)
        pnl = sum(b[5] for b in bets)
        wins = sum(1 for b in bets if b[5] > 0)
        print(f"  won {wins}/{len(bets)} ({wins / len(bets):.0%}), staked ${staked:,.2f},"
              f" profit ${pnl:+,.2f} ({pnl / staked:+.1%} per dollar)")
        for asset in sorted({b[0] for b in bets}):
            sub = [b for b in bets if b[0] == asset]
            print(f"  {asset.upper():4} {len(sub):4} bets, profit ${sum(b[5] for b in sub):+,.2f}")
    print()

    # 3. Who predicts better: our formula or Polymarket's price?
    ticks = q("SELECT t.asset, t.window_start, t.seconds_left, t.fair_up, t.up_ask, t.down_ask, w.winner,"
              " t.up_bid FROM ticks t JOIN windows w USING (asset, window_start)"
              " WHERE w.winner IS NOT NULL AND t.fair_up IS NOT NULL ORDER BY t.ts")
    if not ticks:
        print("Not enough finished windows yet to compare predictions. Let it run longer.")
        return
    ours = [(t[3], 1.0 if t[6] == "up" else 0.0) for t in ticks]
    market = [((t[7] + t[4]) / 2, 1.0 if t[6] == "up" else 0.0) for t in ticks if t[4] is not None and t[7] is not None]
    print("PREDICTION ACCURACY (Brier score, lower is better; 0.25 = coin flip)")
    print(f"  our formula:        {brier(ours):.4f}")
    if market:
        print(f"  Polymarket price:   {brier(market):.4f}")
    print("  If Polymarket's score is lower, the market already knows more than our formula.\n")

    # 4. What if the threshold were different?
    windows = len({(t[0], t[1]) for t in ticks})
    print(f"WHAT-IF: other thresholds, {windows} finished windows, 1 share per bet at best ask (optimistic)")
    print("  min edge   bets   win rate   profit per share")
    replay_rows = [t[:7] for t in ticks]
    for edge in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
        b, w, p = replay(replay_rows, cfg, edge)
        rate = f"{w / b:.0%}" if b else "-"
        avg = f"{p / b * 100:+.1f}c" if b else "-"
        print(f"  {edge:>7.0%}  {b:5}   {rate:>8}   {avg:>10}")


if __name__ == "__main__":
    main()
