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
from .riskreward import passes_range
from .periods import period_starts

log = logging.getLogger("polybot")

# How Polymarket holds an account, in the words the settings page shows. The official SDK works it
# out from the key and the wallet address, so nothing here has to guess any more.
WALLET_KINDS = {"EOA": "ארנק רגיל", "POLY_PROXY": "התחברות עם מייל / גוגל",
                "GNOSIS_SAFE": "התחברות עם ארנק בדפדפן", "DEPOSIT_WALLET": "ארנק הפקדה של פולימרקט"}


def key_wallets(signer):
    """The Polymarket wallet addresses a key controls (email login, browser wallet, deposit wallet),
    worked out offline the way the SDK does: shows which address POLYMARKET_FUNDER should be."""
    if not signer:
        return []
    try:
        from polymarket import PRODUCTION
        from polymarket._internal import wallet as W
        from polymarket._internal.environment import get_environment_config

        config = get_environment_config(PRODUCTION).wallet_derivation
        derive = (W.derive_proxy_wallet_address, W.derive_safe_wallet_address, W.derive_beacon_deposit_wallet_address)
        return [signer] + [f(signer, config) for f in derive]
    except Exception:
        return [signer]


def explain_connection_error(error, signer, wallet):
    """A failed connection in words the owner can act on."""
    text = str(error)
    if "does not match the expected Deposit Wallet" in text or "does not exist" in text:
        mine = key_wallets(signer)
        return (f"המפתח וכתובת הארנק לא שייכים לאותו חשבון בפולימרקט. POLYMARKET_FUNDER: {wallet or 'לא הוגדר'}. "
                + (f"הארנקים שהמפתח הזה פותח: {', '.join(mine)}. " if mine else "")
                + "בדוק שהמפתח יוצא מאותו חשבון שבו רואים את הכסף, ושהכתובת היא בדיוק זו שמופיעה בפולימרקט ליד הפרופיל.")
    if "private_key" in text:
        return "המפתח (POLYMARKET_PRIVATE_KEY) לא תקין: בדוק שהעתקת אותו במלואו, כולל 0x בהתחלה."
    if "wallet address" in text.lower():
        return "כתובת הארנק (POLYMARKET_FUNDER) לא תקינה: היא צריכה להתחיל ב-0x ולהכיל 40 תווים אחרי זה."
    return "לא הצלחתי להתחבר לפולימרקט: " + text[:300]


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
        self.client = None
        self.exact = False  # "exactly like the lab": the engine sets it from the owner's switch
        self.readiness_passed = False  # the crypto strategy's checks (kept for the dashboard)
        self.ready = {"crypto": None, "stocks": None}  # None: not evaluated yet in this run
        self.last_error = None
        self._balance = None  # (fetched_at, usdc)
        # One order at a time: two strategies signalling at once must not both pass the
        # daily-limit check before either is recorded.
        self._order_lock = threading.Lock()
        self.min_ratio = 0.0  # real money's own risk/reward filter on crypto (shown on the page), set by the engine
        self.ratios = {"crypto": (0.0, 0.0), "stocks": (0.0, 0.0)}  # strategy -> (min, max; 0 = none), set by the engine

    @property
    def wallet_type(self):
        """How Polymarket holds the account, as the SDK found it on the last connection."""
        return self.store.get_setting("live_wallet_type") or None

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

    def arm(self, on, override=False, keep_daily=False):
        """Switch real money on or off. A strategy marked for real money whose checks haven't
        passed trades only with override: the owner saw which checks fail and confirmed anyway.
        The limits still apply. keep_daily: the day's loss limit was reached and the owner goes on
        anyway, for the rest of today (the limit then shows as passed)."""
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
        if on:
            self.store.set_setting("live_stopped", "")
            if keep_daily:
                self.store.set_setting("live_daily_keep", str(int(start_of_today(self.cfg.timezone))))
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

    def max_open(self, raw=False):
        """Real money's cap on open trades per strategy (None: no cap)."""
        import json
        try:
            stored = json.loads(self.store.get_setting("live_max_open") or "{}")
        except ValueError:
            stored = {}
        return {n: (int(stored[n]) if stored.get(n) else None) for n in self.STRATEGIES}

    def set_max_open(self, caps):
        import json
        current = self.max_open()
        for name, n in (caps or {}).items():
            if name in current:
                current[name] = int(n) if n and int(n) > 0 else None
        self.store.set_setting("live_max_open", json.dumps(current))

    def same_direction(self, raw=False):
        value = self.store.get_setting("live_same_direction")
        return int(value) if value not in (None, "") else 1

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
            value = self._read_balance(self._client())
            self._balance = (now, value)
            return value
        except Exception as e:
            self.last_error = f"balance: {e}"
            return None

    @staticmethod
    def _read_balance(client):
        """The cash (pUSD, 6 decimals) Polymarket's order book sees in the account."""
        return client.get_balance_allowance(asset_type="COLLATERAL").balance / 1e6

    @staticmethod
    def _allowance_set(client):
        """False when the exchange may not spend the account's cash yet: orders would fail."""
        allowances = client.get_balance_allowance(asset_type="COLLATERAL").allowances
        return not allowances or any(v > 0 for v in allowances.values())

    def check_connection(self):
        """Connect afresh and read the balance: the settings page's "check connection"."""
        if not self.configured():
            raise ValueError("POLYMARKET_PRIVATE_KEY לא מוגדר ב-Render")
        self.client, self._balance, self.last_error = None, None, None
        try:
            client = self._make_client()
        except Exception as e:
            raise ValueError(explain_connection_error(e, self._signer_address(), self.funder)) from e
        balance = self._read_balance(client)
        try:
            approved = self._allowance_set(client)
        except Exception:
            approved = None
        kind = str(getattr(client, "wallet_type", "") or "")
        if kind:
            self.store.set_setting("live_wallet_type", kind)
        self.client = client
        self._balance = (time.time(), balance)
        return {"balance": balance, "wallet_type": kind, "wallet_kind": WALLET_KINDS.get(kind, kind),
                "approved": approved, "signer": self._signer_address(), "wallet": str(getattr(client, "wallet", ""))}

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
            "wallet_type": self.wallet_type,
            "readiness_passed": self.readiness_passed,
            "stake": self.stake(),
            "max_stake": self.cfg.live_max_stake_usd,
            "max_open": self.max_open(),
            "same_direction": self.same_direction(),
            "daily_loss_limit": self.daily_loss_limit(),
            "today_exposure": self.today_exposure(),
            "stopped": self.stopped(),
            "daily_kept": self.daily_kept(),
            "last_error": self.last_error,
            "balance": self._balance[1] if self._balance else None,
        }

    # Trading.
    def _client(self):
        if self.client is None:
            self.client = self._make_client()
        return self.client

    def _make_client(self):
        # Polymarket's official SDK (CLOB V2, pUSD cash); only needed for real trading. The
        # wallet is always named: left out, the SDK would create a new deposit wallet instead.
        from polymarket import SecureClient

        return SecureClient.create(private_key=self.private_key, wallet=self.funder or self._signer_address())

    def maybe_trade(self, now, asset, minutes, window_start, seconds_left, fill, market, market_cfg=None,
                    strategy="crypto", market_slug=None, variant="main", stake=None, copy=False):
        """Place a real order mirroring a paper signal, if everything allows it: Live is on (the
        owner switched it on, after seeing the checks) and this strategy is set to trade live."""
        if not self.trades(strategy):
            return None
        with self._order_lock:
            return self._trade(now, asset, minutes, window_start, seconds_left, fill, market, market_cfg, market_slug,
                               strategy, variant, stake, copy)

    def check_region(self):
        """How Polymarket sees the machine that sends the orders: its country, and whether
        orders from there are refused (Polymarket's own check)."""
        import json
        import urllib.request
        req = urllib.request.Request("https://polymarket.com/api/geoblock", headers={"User-Agent": "polybot"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        return {"blocked": bool(data.get("blocked")), "country": data.get("country"), "region": data.get("region")}

    def daily_kept(self):
        """The owner went on past today's loss limit (for today only)."""
        return self.store.get_setting("live_daily_keep") == str(int(start_of_today(self.cfg.timezone)))

    def stopped(self):
        import json
        try:
            return json.loads(self.store.get_setting("live_stopped") or "null")
        except ValueError:
            return None

    def blocked(self, strategy, side, window_start, copy=False):
        """Why real money can't take this trade now (None: it can). Only real money's own
        settings count here: the simulation's limits are the simulation's business."""
        if not self.trades(strategy):
            return "this strategy doesn't trade real money now"
        if self.exact:
            return self._daily_check()  # exactly like the lab: no caps; the day's loss limit still holds
        same_cap = self.same_direction()
        if strategy == "crypto" and same_cap and not copy:  # (extra entries of one trade are the owner's choice)
            same = self.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live' AND status = 'filled' AND minutes > 0"
                                 " AND window_start = ? AND side = ?", (window_start, side))[0][0]
            if same >= same_cap:
                return f"{same} {side} bets already in this window (the same-direction cap)"
        cap = self.max_open().get(strategy)
        if cap is not None:
            which = " AND minutes > 0" if strategy == "crypto" else " AND minutes = 0"
            if self.store._x("SELECT COUNT(*) FROM bets WHERE mode = 'live' AND status = 'filled' AND pnl IS NULL"
                             + which)[0][0] >= cap:
                return f"{cap} {strategy} trades already open (the cap)"
        return self._daily_check()

    def _daily_check(self):
        """Today's loss limit (losses so far plus everything still at stake). Reaching it switches
        real money off; the owner can switch it back on, raising the limit or going past it."""
        if self.daily_kept() or self.today_exposure() + self.stake() <= self.daily_loss_limit():
            return None
        if self.armed():
            import json
            self.store.set_setting("live_armed", "0")
            self.store.set_setting("live_stopped", json.dumps({"ts": time.time(), "reason": "daily_loss",
                                                              "loss": self.today_exposure(), "limit": self.daily_loss_limit()}))
            log.warning("LIVE TRADING OFF: today's loss limit reached ($%.2f of $%.2f)",
                        self.today_exposure(), self.daily_loss_limit())
        return "daily loss limit reached"

    def order_cfg(self, cfg, strategy="crypto"):
        """The bet sizing real money uses: its own stake and its own risk/reward filter."""
        import dataclasses
        lo, hi = self.ratios.get(strategy, (0.0, 0.0))
        return dataclasses.replace(cfg, stake_usd=self.stake(), min_ratio=lo, max_ratio=hi)

    def _trade(self, now, asset, minutes, window_start, seconds_left, fill, market, market_cfg, market_slug,
               strategy="crypto", variant="main", stake=None, copy=False):
        reason = self.blocked(strategy, fill.side, window_start, copy)
        if reason:
            log.info("LIVE skipped %s: %s", asset.upper(), reason)
            return None
        stake = self.stake() if stake is None else min(stake, self.stake())

        fees = market_cfg or self.cfg
        # The limit is a level we saw in the book, so it is already on the market's
        # price grid (0.01, or 0.001 near 0 and 1); snap anyway against float noise.
        tick = market.get("tick_size", 0.01)
        price = round(round(fill.limit_price / tick) * tick, 6)
        per_share = price + taker_fee_per_share(price, fees.fee_rate, fees.fee_exponent)
        lo, hi = self.ratios.get(strategy, (0.0, 0.0))
        if not passes_range(per_share, lo, hi):
            log.info("LIVE skipped %s: pays %.2f to win %.2f, outside the risk/reward set for real money (%g-%g)",
                     asset.upper(), per_share, 1 - per_share, lo, hi)
            return None
        size = int(stake / per_share * 100) / 100
        if size < market.get("min_order_size", 0):
            log.info("LIVE skipped %s: $%.2f buys %.2f shares, under Polymarket's minimum of %g",
                     asset.upper(), stake, size, market["min_order_size"])
            return None
        token = market["up_token"] if fill.side == "up" else market["down_token"]
        order_id = error = None
        status = "rejected"
        spend = int(size * price * 100) / 100  # dollars; the SDK drops sub-cent digits anyway
        try:
            # Fill-or-kill, never above the price we saw: all of it, or nothing. No orders left hanging.
            resp = self._client().place_market_order(token_id=token, side="BUY", amount=spend, max_price=price,
                                                     order_type="FOK")
            if getattr(resp, "ok", False):
                order_id = str(resp.order_id)
                if resp.status == "live":  # resting on the book: not what fill-or-kill should ever give
                    error = "order rested on the book instead of filling"
                else:
                    status = "filled"
                    # What actually traded: the cash given and the shares taken (kept only when they
                    # read as plain dollars and shares near the order, never a unit mix-up).
                    got, paid = float(resp.taking_amount), float(resp.making_amount)
                    if got > 0 and paid > 0 and 0.5 * size <= got <= 1.5 * size and 0 < paid / got <= price + 1e-9:
                        size, price = got, paid / got
            else:
                error = f"{getattr(resp, 'code', '')}: {getattr(resp, 'message', resp)}"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        if error:
            self.last_error = f"{time.strftime('%H:%M:%S')} {error}"
            if "restricted in your region" in error.lower() or "geoblock" in error.lower():
                # Polymarket refuses orders from where this machine is: no point trying again.
                import json
                self.store.set_setting("live_armed", "0")
                self.store.set_setting("live_stopped", json.dumps({"ts": time.time(), "reason": "region", "error": error}))
                log.warning("LIVE TRADING OFF: Polymarket doesn't take orders from this machine's region")

        cost = size * price
        fee = size * taker_fee_per_share(price, fees.fee_rate, fees.fee_exponent)
        live_fill = type(fill)(side=fill.side, fair_prob=fill.fair_prob, best_ask=fill.best_ask, avg_price=price,
                               shares=size, cost=cost, fee=fee, edge=fill.fair_prob - per_share, limit_price=price)
        self.store.add_bet(now, asset, minutes, window_start, seconds_left, live_fill, mode="live", status=status,
                           order_id=order_id, error=error, market=market_slug, stake=stake, variant=variant)
        log.warning("LIVE ORDER %s %s %.2f shares @ %.2f: %s %s", asset.upper(), fill.side.upper(), size, price,
                    status, error or "")
        return status
