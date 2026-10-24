"""Real-money trading on Polymarket. Locked unless every one of these holds:

1. the simulation passed the readiness checks,
2. a Polymarket wallet key is configured (environment variables, never the browser),
3. the person switched Live on in the dashboard,
4. today's losses plus the money in open bets are under the daily limit.

It mirrors the simulation's signals: same entry rule, smaller stake.
"""
import logging
import os
import threading
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
        # How Polymarket holds the account: 0 a plain wallet, 1 an email/Google login (a proxy
        # wallet), 2 a browser-wallet login (a Safe). The balance and orders are read for the
        # address that type implies, so a wrong guess shows $0; "check connection" finds it.
        self._env_signature_type = int(env.get("POLYMARKET_SIGNATURE_TYPE", "1" if self.funder else "0"))
        self.client = None
        self.readiness_passed = False  # the crypto strategy's checks (kept for the dashboard)
        self.ready = {"crypto": None, "stocks": None}  # None: not evaluated yet in this run
        self.last_error = None
        self._balance = None  # (fetched_at, usdc)
        # One order at a time: two strategies signalling at once must not both pass the
        # daily-limit check before either is recorded.
        self._order_lock = threading.Lock()

    @property
    def signature_type(self):
        detected = self.store.get_setting("live_signature_type")
        return int(detected) if detected not in (None, "") else self._env_signature_type

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

    def arm(self, on, override=False):
        """Switch real money on or off. A strategy marked for real money whose checks haven't
        passed trades only with override: the owner saw which checks fail and confirmed anyway.
        The limits still apply."""
        if on and not self.configured():
            raise ValueError("לא מחובר ארנק פולימרקט")
        waiting = [n for n, marked in self.strategies().items() if marked and not self.ready.get(n)]
        if on and waiting and not override:
            raise ValueError("הסימולציה עוד לא עברה את הבדיקות; צריך אישור מפורש")
        if on:
            for name, marked in self.strategies().items():
                if marked:  # confirmed without its checks: kept; with them: follows the checks
                    self.store.set_setting(f"live_{name}_override", "1" if name in waiting else "0")
            self.store.set_setting("live_override", "0")
        self.store.set_setting("live_armed", "1" if on else "0")
        log.warning("LIVE TRADING %s%s", "ON" if on else "OFF",
                    f" (owner override for: {', '.join(waiting)})" if on and waiting else "")

    STRATEGIES = ("crypto", "stocks")

    def strategies(self):
        """Which main strategies trade real money while Live is on (the rest stay simulated)."""
        on = {"crypto": True, "stocks": False}
        for name in self.STRATEGIES:
            value = self.store.get_setting(f"live_{name}")
            if value is not None:
                on[name] = value == "1"
        return on

    def set_strategy(self, name, on, confirmed=False):
        """Mark a strategy for real money. Before its checks pass, only when the owner confirmed."""
        if name not in self.STRATEGIES:
            raise ValueError("אסטרטגיה לא מוכרת")
        if on and not self.ready.get(name) and not confirmed:
            raise ValueError("האסטרטגיה עוד לא עברה את הבדיקות; צריך אישור מפורש")
        self.store.set_setting(f"live_{name}", "1" if on else "0")
        self.store.set_setting(f"live_{name}_override", "1" if (on and not self.ready.get(name)) else "0")
        log.warning("Live trading for the %s strategy: %s", name, "on" if on else "off")

    def set_strategy_readiness(self, name, passed):
        self.ready[name] = bool(passed)
        if name == "crypto":
            self.readiness_passed = bool(passed)

    def overridden(self, name="crypto"):
        if self.store.get_setting(f"live_{name}_override") == "1":
            return True
        return name == "crypto" and self.store.get_setting("live_override") == "1"  # before 1.14

    def trades(self, strategy):
        """Does this strategy trade real money right now? Live is on, the strategy is marked for
        it, and its own checks passed in this run, or the owner confirmed it without them."""
        return (self.armed() and self.configured() and self.strategies().get(strategy, False)
                and (bool(self.ready.get(strategy)) or self.overridden(strategy)))

    def emergency_stop(self):
        self.store.set_setting("live_armed", "0")
        log.warning("LIVE TRADING EMERGENCY STOP")
        if self.client:
            try:
                self.client.cancel_all()
            except Exception as e:
                self.last_error = f"cancel_all failed: {e}"

    def set_readiness(self, passed):
        """The crypto strategy's checks. A strategy that trades live because its checks passed
        stops by itself when they stop passing (trades() says no); one the owner confirmed
        without them keeps going. The limits apply either way."""
        self.set_strategy_readiness("crypto", passed)

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
            value = self._read_balance(self._client(), self.signature_type)
            self._balance = (now, value)
            return value
        except Exception as e:
            self.last_error = f"balance: {e}"
            return None

    @staticmethod
    def _read_balance(client, signature_type):
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=signature_type)
        try:
            # Polymarket caches balances; ask it to look again before reading.
            client.update_balance_allowance(params)
        except Exception:
            pass
        return int(client.get_balance_allowance(params)["balance"]) / 1e6  # USDC has 6 decimals

    def check_connection(self):
        """Connect afresh and read the balance: the settings page's "check connection". Tries
        the account type in use first, then the others; the first that holds money is kept."""
        if not self.configured():
            raise ValueError("POLYMARKET_PRIVATE_KEY לא מוגדר ב-Render")
        self.client, self._balance, self.last_error = None, None, None
        current = self.signature_type
        found, errors = {}, []
        for kind in [current] + [k for k in (1, 2, 0) if k != current]:
            try:
                found[kind] = self._read_balance(self._make_client(kind), kind)
            except Exception as e:
                errors.append(f"type {kind}: {e}")
                continue
            if found[kind] > 0:
                break
        if not found:
            raise ValueError("לא הצלחתי להתחבר לפולימרקט: " + "; ".join(errors)[:300])
        kind = next((k for k, v in found.items() if v > 0), current)
        if kind != current:
            self.store.set_setting("live_signature_type", str(kind))
            log.warning("Polymarket account type detected: %s (was %s)", kind, current)
        self.client = None
        self._balance = (time.time(), found.get(kind, 0.0))
        return {"balance": found.get(kind, 0.0), "signature_type": kind, "changed": kind != current,
                "tried": found, "signer": self._signer_address()}

    def _signer_address(self):
        try:
            from eth_account import Account
            return Account.from_key(self.private_key).address
        except Exception:
            return None

    def status(self):
        return {
            "configured": self.configured(),
            "armed": self.armed(),
            "override": self.armed() and any(self.overridden(n) for n in self.STRATEGIES),
            "strategies": self.strategies(),
            "trading": {n: self.trades(n) for n in self.STRATEGIES},
            "ready": dict(self.ready),
            "funder": (self.funder[:6] + "…" + self.funder[-4:]) if self.funder else None,
            "signature_type": self.signature_type,
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
            self.client = self._make_client(self.signature_type)
        return self.client

    def _make_client(self, signature_type):
        # Polymarket's official client; only needed for real trading.
        from py_clob_client.client import ClobClient

        client = ClobClient(CLOB_HOST, key=self.private_key, chain_id=POLYGON,
                            signature_type=signature_type, funder=self.funder)
        client.set_api_creds(client.create_or_derive_api_creds())
        return client

    def maybe_trade(self, now, asset, minutes, window_start, seconds_left, fill, market, market_cfg=None,
                    strategy="crypto", market_slug=None):
        """Place a real order mirroring a paper signal, if everything allows it: Live is on (the
        owner switched it on, after seeing the checks) and this strategy is set to trade live."""
        if not self.trades(strategy):
            return None
        with self._order_lock:
            return self._trade(now, asset, minutes, window_start, seconds_left, fill, market, market_cfg, market_slug)

    def _trade(self, now, asset, minutes, window_start, seconds_left, fill, market, market_cfg, market_slug):
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
                           order_id=order_id, error=error, market=market_slug)
        log.warning("LIVE ORDER %s %s %.2f shares @ %.2f: %s %s", asset.upper(), fill.side.upper(), size, price,
                    status, error or "")
        return status
