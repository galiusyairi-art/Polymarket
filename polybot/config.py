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
    # Risk/reward: a win must pay at least this many times what a loss costs (0 = no filter).
    # Set from the settings page; applies to the main strategy only, never to the lab.
    min_ratio: float = 0.0
    # Like a real fill-or-kill order: a simulated bet happens only if the whole stake can be bought
    # at prices the rule accepts; otherwise none. (Off, it buys what it can up to the stake.)
    full_fill_only: bool = True
    # A market can run several strategies at once; past this many, adding one drops the weakest.
    max_rules_per_market: int = 3
    max_ratio: float = 0.0  # 0: no cap. Above it a share is too cheap: a long shot the owner chose to skip

    # Polymarket taker fee, per share: fee_rate * (price * (1 - price)) ** fee_exponent.
    # Each market publishes its own feeSchedule and the engine uses that; these
    # defaults (the crypto schedule as of Oct 2026: 1.75c a share at 50 cents,
    # 3.5% of the stake) apply only if a market doesn't say.
    fee_rate: float = 0.07
    fee_exponent: float = 1.0

    # Markets resolve on Chainlink's 60-second TWAP: the window's opening value is
    # the average of the minute before it opens, the closing value the average of
    # its last minute.
    twap_seconds: int = 60
    # Which reading of the rules the model trades on: "twap" (60s average at open
    # vs at close) or "point" (Chainlink value at the open second vs the close
    # second). The dashboard measures both against real outcomes.
    resolution: str = "twap"

    # Never bet on stale data. A price (Binance-nowcast Chainlink) is normally
    # 0.3-1.3s old; a book is as old as the newest Polymarket update applied, normally
    # well under a second. Anything older means a backlog or a stalled feed.
    max_price_age: float = 2.5
    # Prices straight from Binance and Coinbase websockets, to bring Chainlink up to the present.
    exchange_feeds: bool = True
    max_book_age: float = 3.0

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

    # The lab: every variant in lab.py, on these markets, on paper.
    lab_assets: tuple = ("btc", "eth", "sol", "xrp", "doge", "bnb", "hype")
    lab_minutes: tuple = (5, 15)
    lab_min_bets: int = 30
    lab_z: float = 2.5  # stricter than the main strategy's 1.65: many variants are tested
    lab_record_seconds: float = 5.0  # tick history for lab-only markets is coarser, to save disk
    # The lab's fills as a real order gets them: the order reaches Polymarket this long after the
    # signal, and fills (all of it, at the price seen or better) only if the book still has it then.
    lab_fill_delay: float = 0.5
    equities: bool = True  # the stocks area (equity.py): daily data, the market's picture, the backtest
    research_hour_utc: int = 6  # daily Claude summary
    keep_ticks_days: int = 14

    # Automatic calibration (calibration.py): needs this many independent settled
    # outcomes, shrinks toward "no change" with a prior worth calib_prior_samples, and is
    # applied only if it improves the average log loss by calib_min_gain.
    calib_min_samples: int = 150
    calib_prior_samples: int = 300
    calib_min_gain: float = 0.002

    # Validation on new data: a strategy that looks good becomes a candidate, and is
    # proven only by bets placed after that moment.
    lab_oos_min_bets: int = 50
    lab_oos_z: float = 2.0
    # The crypto lab's verdicts (lab.leaderboard): price_z at the checkpoints, on real fills only.
    lab_gate_z: float = 3.0  # (some 800 rules run at once: luck alone gives a few over 2.5)
    lab_gate_oos_z: float = 2.0

    # Search near winners: neighbours of promising strategies are added automatically.
    explore_every_seconds: int = 1800
    explore_per_market: int = 12
    explore_total: int = 60
    explore_ratio_total: int = 90  # risk/reward versions ($3/$5/$7 per $10) of the leading and the owner's rules
    hosting_usd_month: float = 31.0  # Fly.io performance-1x; editable in settings
    blend_min_windows: int = 150  # resolved windows before the model/market mix is fitted
    blend_max_windows: int = 4000  # fitted on the most recent windows only
    blend_min_gain: float = 0.002  # log loss the mix must beat both alone by, on unseen windows
    mail_quiet_seconds: float = 30.0  # trade mails wait this long after the latest trade of their kind
    mail_max_wait_seconds: float = 300.0  # ...but never longer than this in all
    stocks_lab: bool = True  # the experimental lab on Polymarket's daily stock markets (paper only)  # automatic risk/reward choice: bets a ratio needs before it counts

    # Daily summary email, local time.
    daily_email_time: str = "23:55"

    # Where "today", "this week" and "this month" begin, and when the daily loss limit resets.
    timezone: str = "Asia/Jerusalem"

    # Dashboard strategy buckets, judged on a period's settled lab bets.
    dash_min_bets: int = 10  # fewer than this: not enough data to say
    dash_hot_roi: float = 0.10  # at least +10% per dollar: "making a lot"
    dash_losing_roi: float = -0.05  # -5% or worse: "losing"

    db_path: str = "data/polybot.sqlite"
