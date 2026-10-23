"""Settings for the recorder, the paper trader and the live trader.

Every strategy number here is a guess to be tested, not a known-good value. The
simulation exists to find out which values (if any) would have made money.
"""
from dataclasses import dataclass

# Price feed used to compute the "fair" probability. Polymarket's short crypto
# markets resolve on a Chainlink price, which tracks the big exchanges closely
# but is not identical; the dashboard measures how often the two disagree.
COINBASE_PRODUCTS = {"btc": "BTC-USD", "eth": "ETH-USD", "sol": "SOL-USD", "xrp": "XRP-USD"}


@dataclass
class Config:
    assets: tuple = ("btc", "eth", "sol")
    window_minutes: int = 15
    # Polymarket names each window's market after its start time (unix seconds).
    slug_template: str = "{asset}-updown-{minutes}m-{start}"
    poll_seconds: float = 1.0

    # Volatility: estimated from our own 1-second price samples.
    vol_lookback_seconds: int = 1800
    vol_min_seconds: int = 300  # no bets until we have this much price history

    # Entry rule (shared by simulation and live).
    min_edge: float = 0.05  # bet only if fair prob - price - fee >= 5 points
    stake_usd: float = 10.0  # simulated dollars per paper bet
    min_seconds_left: int = 15  # too close to the end: our price may differ from Chainlink's
    max_seconds_left: int = 600  # too early: the model knows little yet
    max_bets_per_window: int = 1

    # Polymarket taker fee, per share: price * fee_rate * (price * (1 - price)) ** fee_exponent.
    # Each market publishes its own feeSchedule and the engine uses that; these
    # defaults (the crypto schedule as of Oct 2026: about 1.75% of the stake at
    # 50 cents) apply only if a market doesn't say.
    fee_rate: float = 0.07
    fee_exponent: float = 1.0

    # Markets resolve on Chainlink's 60-second TWAP: the window's opening value is
    # the average of the minute before it opens, the closing value the average of
    # its last minute.
    twap_seconds: int = 60

    # When is the simulation good enough to risk real money? All must hold.
    ready_min_hours: float = 72
    ready_min_windows: int = 100
    ready_min_feed_agreement: float = 0.97
    ready_min_bets: int = 100
    ready_min_roi: float = 0.03
    ready_recent_bets: int = 50

    # Real-money limits. Stake and daily loss can be lowered in the Live tab,
    # never raised above the max.
    live_stake_usd: float = 5.0
    live_max_stake_usd: float = 50.0
    live_daily_loss_limit_usd: float = 20.0

    db_path: str = "data/polybot.sqlite"
