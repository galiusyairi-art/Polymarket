"""Real-money trading on Polymarket. Locked unless every one of these holds:

1. the simulation passed the readiness checks,
2. a Polymarket wallet key is configured (environment variables, never the browser),
3. the person switched Live on in the dashboard,
4. today's losses plus the money in open bets are under the daily limit.

It mirrors the simulation's signals: same entry rule, smaller stake.
"""
import collections
import logging
import math
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


def reject_reason(error):
    """Polymarket's answer to a real order it refused, in plain words (for the page and the mail)."""
    e = str(error or "").lower()
    if "fok_not_filled" in e or "fully filled" in e:
        return ("המחיר זז לפני שההזמנה הגיעה: לא היו מספיק מניות במחיר שראינו, אז פולימרקט ביטלה את כולה"
                " (הכל או כלום). לא נגבה כסף")
    if "fak_not_filled" in e or "no orders found to match" in e or "nothing filled" in e:
        return ("המחיר זז לפני שההזמנה הגיעה: לא נשארה אף מניה במחיר שראינו (או במרווח שמותר)."
                " לא נגבה כסף")
    if "not_enough_balance" in e or "balance / allowance" in e:
        return "אין מספיק מזומן בארנק (או שזכיות עוד לא מומשו ב-Redeem)"
    if "market_not_ready" in e:
        return "השוק עוד לא מוכן לקבל הזמנות"
    if "min" in e and "size" in e:
        return "מתחת לגודל המינימלי שפולימרקט מקבלת"
    if "restricted" in e or "geoblock" in e:
        return "פולימרקט לא מקבלת הזמנות מהאזור של השרת"
    if "timeout" in e or "timed out" in e or "connection" in e:
        return "התקשורת עם פולימרקט נכשלה"
    if "rested on the book" in e:
        return "ההזמנה נשארה בספר ולא בוצעה"
    return str(error or "")[:200]


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
        # Signals real money didn't act on, and why (newest last): the owner sees them in the live tab.
        self.skips = collections.deque(maxlen=100)
        self._balance = None  # (fetched_at, usdc)
        # The wallet as Polymarket counts it: the cash, and the positions held (open, and won but
        # not yet redeemed) with their value. Read only (see refresh_wallet).
        self.wallet = None  # the last reading of the wallet
        # What the page and the mails show: the reading taken when trades were last published (or one
        # taken with nothing waiting), so the wallet, the P&L and the close mail always move together.
        self.wallet_shown = None
        self.wallet_slugs = None  # {market slug: [(redeemable, value)]}: a round's positions, once resolved
        # One order at a time: two strategies signalling at once must not both pass the
        # daily-limit check before either is recorded.
        self._order_lock = threading.Lock()
        self._warm = {}  # token -> when its market's details were last loaded into the SDK (warm())
        # Cents an automatic order may pay over the price its strategy saw (0, 1 or 2): the engine sets it
        # from the room study (fills.py) and real money's own results with it (Engine._room_tick).
        self.room = 0
        # ... and on a cheap side (under CHEAP_BELOW, whose win pays 4x and more), up to this many (the owner's
        # decision of 29.10.2026: the big wins the book refused): the same bounds, the edge and the filter.
        self.cheap_room = 0
        # ... and, getting into most of the very same trades (the owner, 30.10.2026: 85%): (cents by the price band
        # the order wants, the cents elsewhere), never past fill_floor of edge after the fee (Engine._fill_first_tick)
        self.fill_room = None
        self.fill_floor = 0.03
        self._warm_conn = 0.0
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

    def approval(self):
        """Real money on in the owner's approval mode: nothing is bought by itself; each signal is
        offered on its card, and only the owner's click sends the order."""
        return self.armed() and self.store.get_setting("live_approval") == "1"

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

    def arm(self, on, override=False, keep_daily=False, approval=False):
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
        self.store.set_setting("live_approval", "1" if on and approval else "0")
        log.warning("LIVE TRADING %s%s%s", "ON" if on else "OFF", " (the owner approves each trade)" if on and approval else "",
                    f" (owner override for: {', '.join(waiting)})" if on and waiting else "")

    STRATEGIES = ("crypto", "stocks")

    def _fill_cents(self, seen):
        """The cents real money may pay over the price seen to get in (fill_room), at this price."""
        if not self.fill_room:
            return 0
        bands, default = self.fill_room
        for lo, c in bands.items():
            if float(lo) - 1e-9 <= seen < float(lo) + 0.2 - 1e-9:
                return int(c)
        return int(default or 0)
    CHEAP_BELOW = 0.20  # a side under 20 cents: the cheap room may apply (fills.CHEAP)

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
        rows = self.store.read(
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

    def refresh_wallet(self):
        """Every half minute: the cash and the positions, read from Polymarket. A won position
        turns into cash only once it is redeemed (Polymarket's "Redeem"); until then the cash alone
        looks short, so the positions and what they are worth are shown next to it. Reads only;
        failures are kept, never raised."""
        if not self.configured():
            return
        try:
            client = self._client()
            positions = list(client.list_positions().iter_items())
            won = [p for p in positions if p.redeemable and float(p.current_value) > 0]
            cash = self._read_balance(client)
            self._balance = (time.time(), cash)
            held = sum(float(p.current_value) for p in positions)
            slugs = {}
            for p in positions:
                slug = getattr(p, "slug", None)
                if slug:
                    slugs.setdefault(str(slug), []).append((bool(p.redeemable), float(p.current_value)))
            self.wallet_slugs = slugs if slugs or not positions else None  # (None: Polymarket didn't say which market)
            self.wallet = {"ts": time.time(), "cash": cash, "positions": held, "total": cash + held,
                           "to_redeem": sum(float(p.current_value) for p in won), "won": len(won),
                           "open": sum(1 for p in positions if not p.redeemable and float(p.current_value) > 0),
                           "error": None}
        except Exception as e:
            self.wallet = dict(self.wallet or {}, error=f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {e}")
            log.warning("Wallet refresh: %s", e)

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
            "approval": self.approval(),
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
            "skips": list(self.skips)[-15:][::-1],
            "balance": self._balance[1] if self._balance else None,
            "wallet": self.wallet_shown,
        }

    # Speed (the owner, 28.10.2026: most real orders found the shares gone). Polymarket's SDK looks up a
    # market's details (two requests) the first time it orders on one of its tokens, and keeps them ten
    # minutes; every round has new tokens. warm() loads them as soon as a round is known, so an order pays
    # only its own trip; and it keeps the connection to the order book open (it closes after 30 idle
    # seconds, and a new one costs a few more trips).
    WARM_AGAIN = 480  # seconds: reloaded before the SDK's ten minutes run out
    CONN_EVERY = 20

    def warm(self, tokens, now=None):
        if not (self.armed() and self.configured()):
            return 0
        now = now or time.time()
        try:
            ctx = self._client()._ctx
        except Exception as e:
            log.warning("Warm-up: %s", e)
            return 0
        loaded = 0
        for tok in tokens:
            last = self._warm.get(tok)
            if last is not None and now - last < self.WARM_AGAIN:
                continue
            try:
                meta = ctx.order_metadata
                (meta.fetch_current_market if last is not None else meta.resolve_market)(ctx, token_id=tok)
                self._warm[tok] = now
                loaded += 1
            except Exception as e:
                log.warning("Warm-up of a market: %s", e)
        if now - self._warm_conn >= self.CONN_EVERY:
            self._warm_conn = now
            try:
                ctx.secure_clob.get_json("/time")
            except Exception:
                pass  # (any answer keeps the connection open)
        if len(self._warm) > 400:
            self._warm = {k: v for k, v in self._warm.items() if now - v < 3600}
        return loaded

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

    def room_price(self, seen, fair_prob, room, tick, fees, strategy="crypto"):
        """The price real money's order goes at for a strategy's signal seen at `seen` (room: (min_edge, ratio, own
        cents)): the room (the owner's decision of 28.10.2026, on the study's word), up to its cents over the price
        seen, never past the strategy's own edge or the risk/reward filter. The order still buys the cheapest shares
        first: an order that would have gone through pays what it would have. The lab and the simulation price their
        orders the same way (the owner, 30.10.2026)."""
        lo, hi = self.ratios.get(strategy, (0.0, 0.0))
        own = room[2] if room and len(room) > 2 else 0  # (the rule's own room, entry "m" from the research)
        fc = self._fill_cents(seen) if room else 0
        cents = max(self.room, self.cheap_room if seen < self.CHEAP_BELOW - 1e-9 else 0, own, fc)
        if not (room and cents):
            return seen
        min_edge, ratio = room[0], room[1]
        if fc and fc >= cents:  # (getting in first: the edge kept is the floor, not the rule's whole)
            min_edge = min(min_edge, self.fill_floor)
        p = round(round((seen + 0.01 * cents) / tick) * tick, 6)
        while p > seen + 1e-9:
            per = p + taker_fee_per_share(p, fees.fee_rate, fees.fee_exponent)
            if (fair_prob - per >= min_edge - 1e-9 and passes_range(per, lo, hi)
                    and (not ratio or 1 - per >= ratio * per - 1e-12)):
                break
            p = round(p - tick, 6)
        return max(p, seen)

    def maybe_trade(self, now, asset, minutes, window_start, seconds_left, fill, market, market_cfg=None,
                    strategy="crypto", market_slug=None, variant="main", stake=None, copy=False, round_stake=None,
                    approved=False, room=None):
        """Place a real order mirroring a paper signal, if everything allows it: Live is on (the
        owner switched it on, after seeing the checks) and this strategy is set to trade live. In the
        approval mode only what the owner approved (approved=True) goes."""
        if not self.trades(strategy) or (self.approval() and not approved):
            return None
        with self._order_lock:
            return self._trade(now, asset, minutes, window_start, seconds_left, fill, market, market_cfg, market_slug,
                               strategy, variant, stake, copy, round_stake, approved, room)

    def sell(self, token, shares, min_price):
        """Sell shares held back to the book (the owner closing a trade early): fill-or-kill, all of
        them at min_price or better, or nothing. (status, dollars received, shares sold, error)."""
        if not self.configured():
            return "rejected", 0.0, 0.0, "הארנק לא מחובר"
        with self._order_lock:
            try:
                resp = self._client().place_market_order(token_id=token, side="SELL", shares=round(shares, 2),
                                                         min_price=min_price, order_type="FOK")
            except Exception as e:
                return "rejected", 0.0, 0.0, f"{type(e).__name__}: {e}"
            if not getattr(resp, "ok", False) or getattr(resp, "status", None) == "live":
                return "rejected", 0.0, 0.0, f"{getattr(resp, 'code', '')}: {getattr(resp, 'message', resp)}"
            # (a sell gives shares and takes dollars; kept only when they read as a sale near the order)
            gave, got = float(resp.making_amount or 0), float(resp.taking_amount or 0)
            if not 0.5 * shares <= gave <= 1.5 * shares or got <= 0:
                gave, got = shares, shares * min_price
            log.warning("LIVE CLOSE %.2f shares, got $%.2f", gave, got)
            return "filled", got, gave, None

    # ---------- maker orders: a real resting buy at our own price ----------
    # The maker strategies (a resting order on the model's price, or on the market's leader) post a
    # post-only limit order: it never buys at the offer (Polymarket refuses it if it would), it waits
    # in the book for a seller, and pays no taker fee. The market's tracker (engine.py) decides where
    # it rests, moves it, checks it and cancels it; these are the calls to Polymarket.
    def place_maker(self, token, price, shares):
        """(order_id, None) once resting on the book, or (None, why not)."""
        with self._order_lock:
            try:
                resp = self._client().place_limit_order(token_id=token, price=f"{price:.4f}".rstrip("0").rstrip("."),
                                                        size=f"{shares:.2f}", side="BUY", post_only=True)
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"[:300]
            if not getattr(resp, "ok", False):
                return None, f"{getattr(resp, 'code', '')}: {getattr(resp, 'message', resp)}"[:300]
            log.warning("LIVE MAKER order %s: %.2f shares at %.3f, resting", resp.order_id, shares, price)
            return str(resp.order_id), None

    def maker_matched(self, order_id):
        """How many of a resting order's shares sellers took so far (raises when Polymarket can't say)."""
        order = self._client().get_order(order_id=order_id)
        return float(order.size_matched or 0), str(order.status or "").lower()

    def cancel_maker(self, order_id):
        with self._order_lock:
            try:
                self._client().cancel_order(order_id=order_id)
                return True
            except Exception as e:
                self.last_error = f"{time.strftime('%H:%M:%S')} cancel {order_id}: {type(e).__name__}: {e}"[:300]
                return False

    def record_maker_fill(self, now, asset, minutes, window_start, seconds_left, side, price, shares, fair, order_id,
                          variant, stake):
        """A maker order's shares that sellers took: a real trade (no fee for the maker)."""
        from .paper import Fill
        fill = Fill(side=side, fair_prob=fair, best_ask=price, avg_price=price, shares=shares, cost=shares * price,
                    fee=0.0, edge=fair - price, limit_price=price)
        self.store.add_bet(now, asset, minutes, window_start, seconds_left, fill, mode="live", status="filled",
                           order_id=order_id, variant=variant, stake=stake)
        log.warning("LIVE MAKER FILL %s %s %.2f shares @ %.3f (order %s)", asset.upper(), side.upper(), shares, price, order_id)

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

    def blocked(self, strategy, side, window_start, copy=False, stake=None):
        """Why real money can't take this trade now (None: it can). Only real money's own
        settings count here: the simulation's limits are the simulation's business."""
        if not self.trades(strategy):
            return "המסחר בכסף אמיתי כבוי"
        if self.exact:
            return self._daily_check(stake)  # exactly like the lab: no caps; the day's loss limit still holds
        same_cap = self.same_direction()
        if strategy == "crypto" and same_cap and not copy:  # (extra entries of one trade are the owner's choice)
            same = self.store.read("SELECT COUNT(*) FROM bets WHERE mode = 'live' AND status = 'filled' AND minutes > 0"
                                 " AND window_start = ? AND side = ?", (window_start, side))[0][0]
            if same >= same_cap:
                return f"כבר {same} {'על עלייה' if side == 'up' else 'על ירידה'} בסבב הזה (הגבלת אותו כיוון)"
        cap = self.max_open().get(strategy)
        if cap is not None:
            which = " AND minutes > 0" if strategy == "crypto" else " AND minutes = 0"
            if self.store.read("SELECT COUNT(*) FROM bets WHERE mode = 'live' AND status = 'filled' AND pnl IS NULL"
                             + which)[0][0] >= cap:
                return f"כבר {cap} טריידים פתוחים (התקרה לטריידים פתוחים)"
        return self._daily_check(stake)

    def _daily_check(self, stake=None):
        """Today's loss limit (losses so far plus everything still at stake). Reaching it switches
        real money off; the owner can switch it back on, raising the limit or going past it."""
        if self.daily_kept() or self.today_exposure() + (stake or self.stake()) <= self.daily_loss_limit():
            return None
        if self.armed():
            import json
            self.store.set_setting("live_armed", "0")
            self.store.set_setting("live_stopped", json.dumps({"ts": time.time(), "reason": "daily_loss",
                                                              "loss": self.today_exposure(), "limit": self.daily_loss_limit()}))
            log.warning("LIVE TRADING OFF: today's loss limit reached ($%.2f of $%.2f)",
                        self.today_exposure(), self.daily_loss_limit())
        return "הגעת להפסד המקסימלי היום"

    def note_skip(self, asset, minutes, side, reason):
        """A signal real money had (the rule found its edge) and didn't trade, with the reason."""
        now = time.time()
        for last in list(self.skips)[-10:]:
            if (last["asset"], last["minutes"], last["side"], last["reason"]) == (asset, minutes, side, reason) \
                    and now - last["ts"] < 60:
                last.update(ts=now, n=last["n"] + 1)  # the same reason at each new price: one line, counted
                return
        self.skips.append({"ts": now, "asset": asset, "minutes": minutes, "side": side, "reason": reason, "n": 1})
        log.info("LIVE skipped %s %sm %s: %s", asset.upper(), minutes, side, reason)

    def order_cfg(self, cfg, strategy="crypto"):
        """The bet sizing real money uses: its own stake and its own risk/reward filter."""
        import dataclasses
        lo, hi = self.ratios.get(strategy, (0.0, 0.0))
        return dataclasses.replace(cfg, stake_usd=self.stake(), min_ratio=lo, max_ratio=hi)

    MATCH_WAIT = 1.5  # seconds before asking what a delayed order filled
    MIN_BUY_USD = 1.0  # Polymarket refuses a marketable buy whose shares cost under a dollar (the fee apart)

    def _matched(self, order_id, price):
        """(shares, dollars) an order filled, asked from Polymarket: for an answer that came without them.
        None when Polymarket couldn't say."""
        try:
            time.sleep(self.MATCH_WAIT)
            order = self._client().get_order(order_id=order_id)
            got = float(order.size_matched or 0)
            return got, got * float(getattr(order, "price", None) or price)
        except Exception as e:
            log.warning("Order %s: what filled unknown: %s", order_id, e)
            return None

    def _trade(self, now, asset, minutes, window_start, seconds_left, fill, market, market_cfg, market_slug,
               strategy="crypto", variant="main", stake=None, copy=False, round_stake=None, approved=False, room=None):
        # round_stake: the strategy's own amount a round (its research's, or the owner's for it),
        # never over real money's hard maximum. approved: the owner's own click (the approval mode):
        # their decision, so none of the automatic guards (the day's limit, the caps, the filter).
        round_stake = min(round_stake or self.stake(), self.cfg.live_max_stake_usd)
        reason = None if approved else self.blocked(strategy, fill.side, window_start, copy, round_stake)
        if reason:
            self.note_skip(asset, minutes, fill.side, reason)
            return None
        stake = round_stake if stake is None else min(stake, round_stake)

        fees = market_cfg or self.cfg
        # The limit is a level we saw in the book, so it is already on the market's
        # price grid (0.01, or 0.001 near 0 and 1); snap anyway against float noise.
        tick = market.get("tick_size", 0.01)
        price = seen = round(round(fill.limit_price / tick) * tick, 6)
        lo, hi = self.ratios.get(strategy, (0.0, 0.0))
        automatic = variant != "manual" and not approved
        if automatic and room:
            price = self.room_price(seen, fill.fair_prob, room, tick, fees, strategy)
        per_share = price + taker_fee_per_share(price, fees.fee_rate, fees.fee_exponent)
        if variant != "manual" and not approved and not passes_range(per_share, lo, hi):  # (the owner's own call)
            self.note_skip(asset, minutes, fill.side, f"{per_share * 100:.0f}¢ למניה: מחוץ לטווח סיכוי / סיכון של הכסף האמיתי")
            return None
        size = int(stake / per_share * 100) / 100
        least = market.get("min_order_size", 0)
        if size < least:
            # Under Polymarket's minimum (a rule that splits its amount into a few entries, or a small
            # amount): the entry grows to the minimum, as long as the round stays within the amount
            # set for a trade. Skipping instead would leave only the cheap shares, never the dear
            # ones: not the strategy the lab measured.
            spent = self.store.read("SELECT COALESCE(SUM(cost + fee), 0) FROM bets WHERE mode = 'live' AND status = 'filled'"
                                  " AND asset = ? AND minutes = ? AND window_start = ? AND variant = ?",
                                  (asset, minutes, window_start, variant))[0][0]
            bigger = math.ceil(least * 1.01 * 100) / 100  # a hair over, so rounding never lands under it
            if not copy and spent + bigger * per_share <= round_stake + 1e-9:
                size, stake = bigger, bigger * per_share
            else:
                self.note_skip(asset, minutes, fill.side,
                               f"${stake:.2f} קונים רק {size:g} מניות ב-{per_share * 100:.0f}¢, ופולימרקט מקבל רק מ-{least:g};"
                               f" {least:g} מניות עולות ${least * per_share:.2f}, יותר מהסכום לטרייד (${round_stake:g})")
                return None
        if size * price < self.MIN_BUY_USD - 1e-9:
            # Polymarket's dollar is the shares' price alone, without the fee (a "$1.00" buy is $0.93 of shares at
            # some prices: refused, 30.10.2026). The entry grows to a dollar's shares while the round stays within
            # its amount, like the minimum in shares above; never past it.
            spent = self.store.read("SELECT COALESCE(SUM(cost + fee), 0) FROM bets WHERE mode = 'live' AND status = 'filled'"
                                  " AND asset = ? AND minutes = ? AND window_start = ? AND variant = ?",
                                  (asset, minutes, window_start, variant))[0][0]
            bigger = math.ceil(self.MIN_BUY_USD * 1.01 / price * 100) / 100
            if not copy and spent + bigger * per_share <= round_stake + 1e-9:
                size, stake = bigger, bigger * per_share
            else:
                self.note_skip(asset, minutes, fill.side,
                               f"המניות עולות ${size * price:.2f} בלי העמלה, ופולימרקט מקבל קנייה רק מ-$1;"
                               f" דולר של מניות עולה ${bigger * per_share:.2f} עם העמלה, יותר מהסכום לטרייד (${round_stake:g})")
                return None
        token = market["up_token"] if fill.side == "up" else market["down_token"]
        order_id = error = None
        actual_fee = None  # what Polymarket charged, when its answer says
        status = "rejected"
        spend = int(size * price * 100) / 100  # dollars; the SDK drops sub-cent digits anyway
        sent_at = time.time()
        # The strategies' own orders take what there is ("fill-and-kill", the owner's decision of 28.10.2026):
        # the shares left at the price, the rest dropped, nothing left hanging. A trade the owner placed or
        # approved stays all or nothing (fill-or-kill): what they decided, or nothing.
        kind = "FAK" if automatic else "FOK"
        try:
            resp = self._client().place_market_order(token_id=token, side="BUY", amount=spend, max_price=price,
                                                     order_type=kind)
            if getattr(resp, "ok", False):
                order_id = str(resp.order_id)
                if resp.status == "live":  # resting on the book: not what a fill-or-kill / fill-and-kill gives
                    error = "order rested on the book instead of filling"
                else:
                    got, paid = float(resp.taking_amount or 0), float(resp.making_amount or 0)
                    if resp.status == "delayed" or not (got > 0 and paid > 0):
                        # Matched later (Polymarket's delay) or an answer without the amounts: ask what filled.
                        # Unknown even then: taken as ordered (a real position is never left unrecorded; the
                        # wallet shows what it really is).
                        asked = self._matched(order_id, price)
                        got, paid = asked if asked is not None else (size, size * price)
                    # What actually traded: the cash given and the shares taken (kept only when they read as
                    # plain dollars and shares within the order, never a unit mix-up). Fill-and-kill may take
                    # fewer shares than asked; nothing at all is a refusal.
                    if got > 0 and paid > 0 and got <= 1.5 * size:
                        status = "filled"
                        if paid / got <= price + 1e-9:  # the cash for the shares; the fee comes on top
                            size, price = got, paid / got
                        elif paid / got <= per_share + 0.01:  # the cash with the fee in it
                            size, actual_fee = got, paid - got * price
                        elif kind == "FAK":
                            size = got  # (the shares are what counts; the price stays the limit, the dearest)
                    elif got <= 0:
                        error = "nothing filled"
                    else:
                        status = "filled"  # (amounts that don't read as shares and dollars: as ordered)
            else:
                error = f"{getattr(resp, 'code', '')}: {getattr(resp, 'message', resp)}"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        try:  # (speed: from the signal to the order sent, and the order's way there and back)
            self.store.add_order_timing((now, asset, minutes, status, sent_at - now, time.time() - sent_at))
        except Exception:
            pass
        if error:
            self.last_error = f"{time.strftime('%H:%M:%S')} {error}"
            if "restricted in your region" in error.lower() or "geoblock" in error.lower():
                # Polymarket refuses orders from where this machine is: no point trying again.
                import json
                self.store.set_setting("live_armed", "0")
                self.store.set_setting("live_stopped", json.dumps({"ts": time.time(), "reason": "region", "error": error}))
                log.warning("LIVE TRADING OFF: Polymarket doesn't take orders from this machine's region")

        cost = size * price
        fee = actual_fee if actual_fee is not None else size * taker_fee_per_share(price, fees.fee_rate, fees.fee_exponent)
        live_fill = type(fill)(side=fill.side, fair_prob=fill.fair_prob, best_ask=fill.best_ask, avg_price=price,
                               shares=size, cost=cost, fee=fee, edge=fill.fair_prob - per_share, limit_price=price)
        self.store.add_bet(now, asset, minutes, window_start, seconds_left, live_fill, mode="live", status=status,
                           order_id=order_id, error=error, market=market_slug, stake=stake, variant=variant, seen=seen)
        log.warning("LIVE ORDER %s %s %.2f shares @ %.2f: %s %s", asset.upper(), fill.side.upper(), size, price,
                    status, error or "")
        return status
