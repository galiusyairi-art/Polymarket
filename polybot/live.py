"""Real-money trading on Polymarket. Locked unless every one of these holds:

1. the simulation passed the readiness checks,
2. a Polymarket wallet key is configured (environment variables, never the browser),
3. the person switched Live on in the dashboard,
4. today's losses plus the money in open bets are under the daily limit.

It mirrors the simulation's signals: same entry rule, smaller stake.
"""
import logging
import os
import time

from .model import taker_fee_per_share
from .periods import period_starts

log = logging.getLogger("polybot")

CLOB_HOST = "https://clob.polymarket.com"
POLYGON = 137


def start_of_today(tz):
    return period_starts(time.time(), tz)["day"]


class LiveTrader:
    def __init__(self, cfg, store, env=None):
        env = os.environ if env is None else env
        self.cfg, self.store = cfg, store
        self.private_key = env.get("POLYMARKET_PRIVATE_KEY", "").strip()
        # The address Polymarket shows as your deposit/profile wallet. Needed for
        # accounts made with email or a browser wallet (they trade through a proxy).
        self.funder = env.get("POLYMARKET_FUNDER", "").strip() or None
        self.signature_type = int(env.get("POLYMARKET_SIGNATURE_TYPE", "1" if self.funder else "0"))
        self.client = None
        self.readiness_passed = False
        self.last_error = None
        self._balance = None  # (fetched_at, usdc)

    # Settings shown and changed in the Live tab.
    def configured(self):
        return bool(self.private_key)

    def armed(self):
        return self.store.get_setting("live_armed") == "1"

    def stake(self):
        value = float(self.store.get_setting("live_stake", self.cfg.live_stake_usd))
        return min(value, self.cfg.live_max_stake_usd)

    def daily_loss_limit(self):
        return float(self.store.get_setting("live_daily_loss_limit", self.cfg.live_daily_loss_limit_usd))

    def update_settings(self, stake=None, daily_loss_limit=None):
        if stake is not None:
            if not 1 <= stake <= self.cfg.live_max_stake_usd:
                raise ValueError(f"stake must be between $1 and ${self.cfg.live_max_stake_usd:.0f}")
            self.store.set_setting("live_stake", stake)
        if daily_loss_limit is not None:
            if not 1 <= daily_loss_limit <= 10_000:
                raise ValueError("daily loss limit must be between $1 and $10,000")
            self.store.set_setting("live_daily_loss_limit", daily_loss_limit)

    def arm(self, on):
        if on and not self.readiness_passed:
            raise ValueError("the simulation has not passed the readiness checks")
        if on and not self.configured():
            raise ValueError("no Polymarket wallet is configured")
        self.store.set_setting("live_armed", "1" if on else "0")
        log.warning("LIVE TRADING %s", "ON" if on else "OFF")

    def emergency_stop(self):
        self.store.set_setting("live_armed", "0")
        log.warning("LIVE TRADING EMERGENCY STOP")
        if self.client:
            try:
                self.client.cancel_all()
            except Exception as e:
                self.last_error = f"cancel_all failed: {e}"

    def set_readiness(self, passed):
        self.readiness_passed = passed
        if not passed and self.armed():
            self.store.set_setting("live_armed", "0")
            log.warning("LIVE TRADING switched off: the simulation no longer passes the readiness checks")

    def today_exposure(self):
        """Realized loss today plus everything still at stake. Positive = money at risk or lost."""
        rows = self.store._x(
            "SELECT COALESCE(SUM(CASE WHEN pnl IS NULL THEN cost + fee ELSE -pnl END), 0) FROM bets"
            " WHERE mode = 'live' AND status = 'filled' AND ts >= ?",
            (start_of_today(self.cfg.timezone),),
        )
        return rows[0][0]

    def balance(self):
        """USDC available in the Polymarket account, or None when no wallet is connected.
        Cached for a minute; failures are kept in last_error, never raised."""
        if not self.configured():
            return None
        now = time.time()
        if self._balance and now - self._balance[0] < 60:
            return self._balance[1]
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            resp = self._client().get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=self.signature_type))
            value = int(resp["balance"]) / 1e6  # USDC has 6 decimals
            self._balance = (now, value)
            return value
        except Exception as e:
            self.last_error = f"balance: {e}"
            return None

    def status(self):
        return {
            "configured": self.configured(),
            "armed": self.armed(),
            "readiness_passed": self.readiness_passed,
            "stake": self.stake(),
            "max_stake": self.cfg.live_max_stake_usd,
            "daily_loss_limit": self.daily_loss_limit(),
            "today_exposure": self.today_exposure(),
            "last_error": self.last_error,
            "balance": self._balance[1] if self._balance else None,
        }

    # Trading.
    def _client(self):
        if self.client is None:
            # Polymarket's official client; only needed for real trading.
            from py_clob_client.client import ClobClient

            client = ClobClient(CLOB_HOST, key=self.private_key, chain_id=POLYGON,
                                signature_type=self.signature_type, funder=self.funder)
            client.set_api_creds(client.create_or_derive_api_creds())
            self.client = client
        return self.client

    def maybe_trade(self, now, asset, minutes, window_start, seconds_left, fill, market, market_cfg=None):
        """Place a real order mirroring a paper signal, if everything allows it."""
        if not (self.readiness_passed and self.armed() and self.configured()):
            return None
        stake = self.stake()
        if self.today_exposure() + stake > self.daily_loss_limit():
            log.info("LIVE skipped %s: daily loss limit reached", asset.upper())
            return None

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        fees = market_cfg or self.cfg
        # The limit is a level we saw in the book, so it is already on the market's
        # price grid (0.01, or 0.001 near 0 and 1); snap anyway against float noise.
        tick = market.get("tick_size", 0.01)
        price = round(round(fill.limit_price / tick) * tick, 6)
        per_share = price + taker_fee_per_share(price, fees.fee_rate, fees.fee_exponent)
        size = int(stake / per_share * 100) / 100
        if size < market.get("min_order_size", 0):
            log.info("LIVE skipped %s: $%.2f buys %.2f shares, under Polymarket's minimum of %g",
                     asset.upper(), stake, size, market["min_order_size"])
            return None
        token = market["up_token"] if fill.side == "up" else market["down_token"]
        order_id = error = None
        status = "rejected"
        try:
            client = self._client()
            order = client.create_order(OrderArgs(token_id=token, price=price, size=size, side=BUY))
            # Fill-or-kill: all of it at this price or better, or nothing. No orders left hanging.
            resp = client.post_order(order, OrderType.FOK)
            order_id = resp.get("orderID") or resp.get("orderId")
            if resp.get("success") and not resp.get("errorMsg"):
                status = "filled"
            else:
                error = resp.get("errorMsg") or str(resp)
        except Exception as e:
            error = str(e)
        if error:
            self.last_error = f"{time.strftime('%H:%M:%S')} {error}"

        cost = size * price
        fee = size * (per_share - price)
        live_fill = type(fill)(side=fill.side, fair_prob=fill.fair_prob, best_ask=fill.best_ask, avg_price=price,
                               shares=size, cost=cost, fee=fee, edge=fill.fair_prob - per_share, limit_price=price)
        self.store.add_bet(now, asset, minutes, window_start, seconds_left, live_fill, mode="live", status=status,
                           order_id=order_id, error=error)
        log.warning("LIVE ORDER %s %s %.2f shares @ %.2f: %s %s", asset.upper(), fill.side.upper(), size, price,
                    status, error or "")
        return status
