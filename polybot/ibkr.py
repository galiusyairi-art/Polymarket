"""Interactive Brokers, through IB Gateway (its own small server: ibgw/ in this repository), for
Atlas: the simulation's trades sent as real orders to the owner's IBKR account: the paper one, or the
real one once the owner picks real money (each trade then only a small amount, live_size dollars).
It sends nothing when the account the gateway is logged into isn't the kind picked (account_problem).

Each signal of the simulation becomes one bracket: the entry the next day by the strategy's entry
order (a buy-stop over the signal day's high, a limit under its close, or at the open: limit-on-open
at the "no chasing" ceiling), good for that day only, and once it is in, a
stop and a take-profit resting at IBKR (good till cancelled), so they hold even with Atlas down. Every
minute Atlas looks at them: the fills, the stop moved with the simulation's (to the entry, then
trailing), and a sale at the next open when the simulation left on time.

The library (ib_async) is asyncio's: every call runs on one thread of its own (IbGateway.call).
"""

import asyncio
import logging
import os
import queue
import threading

log = logging.getLogger("polybot")

PAPER_PREFIX = "DU"  # IBKR's paper accounts (a real one's number has no "DU": "U1234567")
PORT = 4006  # the gateway's paper API, relayed for Fly's private network (ibgw/start.sh)
CLIENT_ID = 7
COMMAND_PORT = 7463  # IBC's command server, relayed for Fly's private network (ibgw/start.sh)


def account_problem(accounts, mode):
    """Why nothing may be sent to these accounts in this mode (None: they are the kind picked)."""
    if not accounts:
        return "IB Gateway לא החזיר חשבון: לא נשלח כלום"
    paper = all(a.startswith(PAPER_PREFIX) for a in accounts)
    real = not any(a.startswith(PAPER_PREFIX) for a in accounts)
    if mode == "live" and not real:
        return f"נבחר כסף אמיתי, אבל IB Gateway מחובר לחשבון Paper ({', '.join(accounts)}): לא נשלח כלום"
    if mode != "live" and not paper:
        return f"IB Gateway מחובר לחשבון אמיתי ({', '.join(accounts)}), ונבחר Paper: לא נשלח כלום. לכסף אמיתי בוחרים \"כסף אמיתי\" בכרטיס של אינטראקטיב"
    return None


def default_host():
    """The gateway's app on Fly's private network: this app's name with "-ibgw" (see ibgw/)."""
    if os.environ.get("IBKR_HOST"):
        return os.environ["IBKR_HOST"]
    app = os.environ.get("FLY_APP_NAME")
    return f"{app}-ibgw.internal" if app else "127.0.0.1"


def gateway_command(host, command, port=None, timeout=10):
    """A command to IBC, the program that logs the gateway in (its command server): "RECONNECTACCOUNT" (log in
    to IBKR again, as Ctrl-Alt-R: after another session, the owner's app, took the login) or "RESTART" (the
    gateway restarts with the session's own credentials, no second approval). What it answered."""
    import socket
    with socket.create_connection((host, int(port or os.environ.get("IBKR_COMMAND_PORT") or COMMAND_PORT)), timeout=timeout) as s:
        s.sendall(f"{command}\nEXIT\n".encode())
        got = b""
        try:
            while len(got) < 4000:
                part = s.recv(1000)
                if not part:
                    break
                got += part
        except socket.timeout:
            pass
    return got.decode(errors="replace").strip()


class IbGateway:
    """The calls to IB Gateway. Each returns plain values (no library objects leave this class)."""

    def __init__(self, host=None, port=None, client_id=CLIENT_ID):
        self.host = host or default_host()
        self.port = int(port or os.environ.get("IBKR_PORT") or PORT)
        self.client_id = client_id
        self._q = queue.Queue()
        self._thread = None
        self.ib = None
        self._pnl = set()  # accounts whose daily profit IBKR streams to us (reqPnL)
        self._live = {}  # symbol -> {"ticker", "bars", "used"}: the prices IBKR streams to us (quote())

    # ---------- the library's own thread ----------
    def _run(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        from ib_async import IB
        self.ib = IB()
        while True:
            try:
                fn, box = self._q.get(timeout=0.5)
            except queue.Empty:
                try:
                    if self.ib.isConnected():
                        self.ib.sleep(0.2)  # (lets the library take in what IBKR sent: fills, order states)
                except Exception:
                    pass
                continue
            try:
                box["result"] = fn(self.ib)
            except Exception as e:
                box["error"] = e
            box["done"].set()

    def call(self, fn, timeout=40):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="ibkr")
            self._thread.start()
        box = {"done": threading.Event()}
        self._q.put((fn, box))
        if not box["done"].wait(timeout):
            raise TimeoutError("IB Gateway didn't answer")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    # ---------- the calls ----------
    def connect(self):
        """The accounts this login manages (connecting first when needed)."""
        def go(ib):
            if not ib.isConnected():
                ib.connect(self.host, self.port, clientId=self.client_id, timeout=20)
            return list(ib.managedAccounts())
        return self.call(go)

    def connected(self):
        return bool(self.ib and self.ib.isConnected())

    def close(self):
        """Let this connection go (a stuck one too: its thread is left to end by itself)."""
        def go(ib):
            ib.disconnect()
        try:
            if self._thread is not None:
                self.call(go, timeout=5)
        except Exception:
            pass

    def place_bracket(self, symbol, qty, limit, target, stop, entry=None, trail=None):
        """The next day's entry with the stop and the take-profit (None: no target) as its children.
        entry: None at the open (limit-on-open at `limit`, the no-chasing ceiling, or market-on-open);
        ("stop", price): a buy-stop for the day (limit at the ceiling); ("limit", price): a limit for
        the day; ("market", None): at the market now (the owner's own, during the session).
        trail: the stop as IBKR's own trailing stop (the owner's rule: it rises with the price, tick by
        tick at the broker, even with Atlas down), this many dollars under the best price, starting at
        `stop`. {"parent", "target", "stop"}: order ids."""
        def go(ib):
            from ib_async import LimitOrder, MarketOrder, Order, Stock, StopLimitOrder, StopOrder
            c = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(c)
            if entry and entry[0] == "stop":
                parent = StopLimitOrder("BUY", qty, round(limit or entry[1] * 1.02, 2), round(entry[1], 2), tif="DAY")
            elif entry and entry[0] == "limit":
                parent = LimitOrder("BUY", qty, round(entry[1], 2), tif="DAY")
            elif entry and entry[0] == "market":
                parent = MarketOrder("BUY", qty, tif="DAY")
            else:
                parent = (LimitOrder("BUY", qty, round(limit, 2), tif="OPG") if limit else MarketOrder("BUY", qty, tif="OPG"))
            parent.orderId, parent.transmit = ib.client.getReqId(), False
            kids = []
            if target:
                kids.append(LimitOrder("SELL", qty, round(target, 2), tif="GTC", orderId=ib.client.getReqId(),
                                       parentId=parent.orderId, transmit=False))
            if trail:
                kids.append(Order(action="SELL", totalQuantity=qty, orderType="TRAIL", auxPrice=round(trail, 2),
                                  trailStopPrice=round(stop, 2), tif="GTC", orderId=ib.client.getReqId(),
                                  parentId=parent.orderId, transmit=True))
            else:
                kids.append(StopOrder("SELL", qty, round(stop, 2), tif="GTC", orderId=ib.client.getReqId(),
                                      parentId=parent.orderId, transmit=True))
            for o in [parent, *kids]:
                ib.placeOrder(c, o)
            ib.sleep(1)
            return {"parent": parent.orderId, "target": kids[0].orderId if target else None, "stop": kids[-1].orderId}
        return self.call(go)

    def orders(self):
        """{order id: {"status", "filled", "avg_price"}} of every order IBKR knows of this session."""
        def go(ib):
            return {t.order.orderId: {"status": t.orderStatus.status, "filled": float(t.orderStatus.filled or 0),
                                      "avg_price": float(t.orderStatus.avgFillPrice or 0),
                                      # (IBKR's own last word on it: why it was refused, when it was)
                                      "why": next((e.message for e in reversed(t.log or []) if getattr(e, "message", "")), "")}
                    for t in ib.trades()}
        return self.call(go)

    def account(self):
        """The account as IBKR counts it, in dollars: its value (net liquidation), cash, today's profit /
        loss (IBKR's own, realized and not), and each position with its price now, value, profit and the
        stop resting on it (the price of its sell stop: a trailing one's where it stands now)."""
        def go(ib):
            acct = (ib.managedAccounts() or [""])[0]
            vals = [v for v in ib.accountValues(acct) if not v.account or v.account == acct]
            base = next((v.currency for v in vals if v.tag == "NetLiquidation"), "USD")

            def num(tag, currency=None):
                for v in vals:
                    if v.tag == tag and v.currency == (currency or base):
                        try:
                            return float(v.value)
                        except (TypeError, ValueError):
                            return None
                return None
            # (an account kept in another currency: its figures in dollars, at IBKR's own rate)
            rate = 1.0 if base == "USD" else (num("ExchangeRate", "USD") or 1.0)
            usd = lambda x: None if x is None else x / rate if base != "USD" else x
            if acct and acct not in self._pnl:  # (IBKR's figure for today: asked once, then it streams)
                ib.reqPnL(acct)
                self._pnl.add(acct)
                ib.sleep(1)
            pnl = next((p for p in ib.pnl(acct)), None)
            fig = lambda name: (None if pnl is None or getattr(pnl, name, None) is None or getattr(pnl, name) != getattr(pnl, name)
                                else usd(float(getattr(pnl, name))))
            stops = {}
            for t in ib.openTrades():
                o = t.order
                if o.action == "SELL" and o.orderType in ("STP", "TRAIL", "STP LMT", "TRAIL LIMIT"):
                    price = (o.trailStopPrice if o.orderType.startswith("TRAIL") else o.auxPrice)
                    if price and price < 1e300:
                        stops[t.contract.symbol] = max(stops.get(t.contract.symbol, 0.0), float(price))
            positions = [{"symbol": p.contract.symbol, "shares": float(p.position), "avg_cost": float(p.averageCost),
                          "price": float(p.marketPrice), "value": float(p.marketValue), "unrealized": float(p.unrealizedPNL),
                          "stop": stops.get(p.contract.symbol)} for p in ib.portfolio(acct) if p.position]
            return {"account": acct, "currency": base, "net_liq": usd(num("NetLiquidation")),
                    "cash": usd(num("TotalCashValue")), "buying_power": usd(num("BuyingPower")),
                    "daily": fig("dailyPnL"), "realized": fig("realizedPnL"), "unrealized": fig("unrealizedPnL"),
                    "positions": positions}
        return self.call(go)

    QUOTE_LINES = 40  # symbols streamed at once (IBKR allows 100 lines; the least used is dropped)
    QUOTE_IDLE = 900  # seconds unasked before a symbol's stream is dropped
    QUOTE_STALE = 180  # seconds: a price IBKR hasn't moved in this long, with the market open, isn't "now"

    def quote(self, symbol, now=None):
        """The price now and today's 5-minute bars, in real time, as Atlas' Yahoo reading has them:
        {"price", "ts", "bars": [(ts, o, h, l, c)], "source": "ibkr"}; None when IBKR has no real-time
        price for it (no subscription, or not yet arrived). Streamed: the first ask subscribes it (its
        price and its bars kept up to date by IBKR from then on), the next ones only read."""
        import math
        import time as _t
        now = now or _t.time()

        def go(ib):
            from ib_async import Stock
            s = self._live.get(symbol)
            if s is None:
                if len(self._live) >= self.QUOTE_LINES:
                    self._drop(ib, min(self._live, key=lambda k: self._live[k]["used"]))
                ib.reqMarketDataType(1)  # (real time only: a delayed price is Yahoo's job)
                c = Stock(symbol, "SMART", "USD")
                ib.qualifyContracts(c)
                s = self._live[symbol] = {"contract": c, "ticker": ib.reqMktData(c, "", False, False), "bars": None, "used": now}
                try:
                    s["bars"] = ib.reqHistoricalData(c, "", "1 D", "5 mins", "TRADES", True, 2, keepUpToDate=True, timeout=15)
                except Exception as e:
                    log.warning("IBKR bars %s: %s", symbol, e)
                ib.sleep(1.5)
            s["used"] = now
            for k in [k for k, v in self._live.items() if now - v["used"] > self.QUOTE_IDLE]:
                self._drop(ib, k)
            t = s["ticker"]
            price = t.marketPrice()
            if price is None or math.isnan(price) or price <= 0 or t.marketDataType not in (1, 2):
                return None
            at = t.time.timestamp() if t.time else None
            if at and now - at > self.QUOTE_STALE:
                return None  # (no tick in minutes: not a price of now)
            bars = [(b.date.timestamp(), float(b.open), float(b.high), float(b.low), float(b.close))
                    for b in (s["bars"] or []) if hasattr(b.date, "timestamp")]
            prev = getattr(t, "close", None)  # (yesterday's close)
            prev = prev if prev and not math.isnan(prev) and prev > 0 else None
            return {"price": float(price), "ts": at or now, "bars": bars, "source": "ibkr", "prev_close": prev}
        return self.call(go, timeout=25)

    # ---------- the live lab (livelab.py): the market's scanner, many bars at once, minutes followed ----------
    def scanner(self, code, rows=50, above_price=10.0, above_volume=None):
        """IBKR's own market scanner (its servers scan every US stock): the symbols of `code` (HOT_BY_VOLUME,
        TOP_PERC_GAIN, MOST_ACTIVE...), best first."""
        def go(ib):
            from ib_async import ScannerSubscription
            sub = ScannerSubscription(instrument="STK", locationCode="STK.US.MAJOR", scanCode=code,
                                      numberOfRows=rows, abovePrice=above_price)
            if above_volume:
                sub.aboveVolume = int(above_volume)
            out = []
            for r in ib.reqScannerData(sub):
                sym = r.contractDetails.contract.symbol
                if sym and sym not in out and r.contractDetails.contract.secType == "STK":
                    out.append(sym)
            return out
        return self.call(go, timeout=60)

    def bars_many(self, symbols, duration, size, limit=20):
        """Many symbols' bars at once (the regular session's, consolidated): {symbol: [(ts, o, h, l, c, v)]};
        a symbol IBKR has nothing for is left out. limit: requests at a time."""
        def go(ib):
            from ib_async import Stock

            async def all_():
                gate = asyncio.Semaphore(limit)
                out = {}

                async def one(sym):
                    async with gate:
                        try:
                            c = Stock(sym, "SMART", "USD")
                            await ib.qualifyContractsAsync(c)
                            bars = await ib.reqHistoricalDataAsync(c, "", duration, size, "TRADES", True, 2, timeout=30)
                            got = [(b.date.timestamp(), float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume or 0))
                                   for b in bars or [] if hasattr(b.date, "timestamp")]
                            if got:
                                out[sym] = got
                        except Exception as e:
                            log.warning("IBKR bars %s: %s", sym, e)
                await asyncio.gather(*(one(s) for s in symbols))
                return out
            return ib.run(all_())
        return self.call(go, timeout=240)

    def follow(self, symbols):
        """Today's 1-minute bars of these symbols kept up to date by IBKR (the others stop): what followed()
        reads."""
        def go(ib):
            from ib_async import Stock
            self._follow = getattr(self, "_follow", {})
            for sym in [s for s in self._follow if s not in symbols]:
                try:
                    ib.cancelHistoricalData(self._follow.pop(sym))
                except Exception:
                    pass
            for sym in symbols:
                if sym in self._follow:
                    continue
                try:
                    c = Stock(sym, "SMART", "USD")
                    ib.qualifyContracts(c)
                    self._follow[sym] = ib.reqHistoricalData(c, "", "1 D", "1 min", "TRADES", True, 2, keepUpToDate=True, timeout=30)
                except Exception as e:
                    log.warning("IBKR follow %s: %s", sym, e)
            return sorted(self._follow)
        return self.call(go, timeout=180)

    def followed(self):
        """{symbol: [(ts, o, h, l, c, v)]}: the followed symbols' bars so far (the last one still forming)."""
        def go(ib):
            return {sym: [(b.date.timestamp(), float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume or 0))
                          for b in bars if hasattr(b.date, "timestamp")] for sym, bars in getattr(self, "_follow", {}).items()}
        return self.call(go, timeout=20)

    def _drop(self, ib, symbol):
        s = self._live.pop(symbol, None)
        if not s:
            return
        try:
            ib.cancelMktData(s["contract"])
            if s["bars"] is not None:
                ib.cancelHistoricalData(s["bars"])
        except Exception:
            pass

    def positions(self):
        """{symbol: (shares, average cost)}."""
        def go(ib):
            return {p.contract.symbol: (float(p.position), float(p.avgCost)) for p in ib.positions() if p.position}
        return self.call(go)

    def modify(self, order_id, **fields):
        """Change a resting order (its lmtPrice / auxPrice); False when it no longer rests."""
        def go(ib):
            for t in ib.openTrades():
                if t.order.orderId == order_id:
                    for k, v in fields.items():
                        setattr(t.order, k, round(v, 2) if isinstance(v, float) else v)
                    ib.placeOrder(t.contract, t.order)
                    return True
            return False
        return self.call(go)

    def cancel(self, order_id):
        def go(ib):
            for t in ib.openTrades():
                if t.order.orderId == order_id:
                    ib.cancelOrder(t.order)
                    return True
            return False
        return self.call(go)

    def close_position(self, symbol, session):
        """The owner closes a position: the stops and targets resting on it cancelled, then all its shares
        sold at the market (during the session; at the next open otherwise). {"order", "shares"}."""
        def go(ib):
            from ib_async import MarketOrder, Stock
            for t in ib.openTrades():
                if t.contract.symbol == symbol and t.order.action == "SELL":
                    ib.cancelOrder(t.order)
            held = sum(float(p.position) for p in ib.positions() if p.contract.symbol == symbol)
            if held <= 0:
                raise ValueError(f"אין פוזיציה ב-{symbol} בחשבון")
            c = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(c)
            o = MarketOrder("SELL", held, tif="DAY" if session else "OPG")
            ib.placeOrder(c, o)
            ib.sleep(1)
            return {"order": o.orderId, "shares": held}
        return self.call(go)

    def place_stop(self, symbol, qty, stop, trail_pct=None):
        """A stop for a position (good till cancelled, resting at IBKR): at `stop`, or trailing trail_pct
        percent under the best price from here (starting at `stop`). Its order id."""
        def go(ib):
            from ib_async import Order, Stock, StopOrder
            c = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(c)
            if trail_pct:
                o = Order(action="SELL", totalQuantity=qty, orderType="TRAIL", trailingPercent=round(trail_pct, 2),
                          trailStopPrice=round(stop, 2), tif="GTC")
            else:
                o = StopOrder("SELL", qty, round(stop, 2), tif="GTC")
            ib.placeOrder(c, o)
            ib.sleep(1)
            return o.orderId
        return self.call(go)

    def market(self, symbol, action, qty):
        """A plain market order for the day (BUY / SELL qty shares): its order id."""
        def go(ib):
            from ib_async import MarketOrder, Stock
            c = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(c)
            o = MarketOrder(action, qty, tif="DAY")
            ib.placeOrder(c, o)
            ib.sleep(0.5)
            return o.orderId
        return self.call(go)

    def sell_at_open(self, symbol, qty):
        def go(ib):
            from ib_async import MarketOrder, Stock
            c = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(c)
            o = MarketOrder("SELL", qty, tif="OPG")
            ib.placeOrder(c, o)
            ib.sleep(1)
            return o.orderId
        return self.call(go)
