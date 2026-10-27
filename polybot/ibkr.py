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


class IbGateway:
    """The calls to IB Gateway. Each returns plain values (no library objects leave this class)."""

    def __init__(self, host=None, port=None, client_id=CLIENT_ID):
        self.host = host or default_host()
        self.port = int(port or os.environ.get("IBKR_PORT") or PORT)
        self.client_id = client_id
        self._q = queue.Queue()
        self._thread = None
        self.ib = None

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
                                      "avg_price": float(t.orderStatus.avgFillPrice or 0)} for t in ib.trades()}
        return self.call(go)

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
