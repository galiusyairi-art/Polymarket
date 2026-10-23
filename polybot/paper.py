"""Paper trading: decide whether a bet is worth it and simulate filling it against the real order book."""
from dataclasses import dataclass

from .model import taker_fee_per_share


@dataclass
class Fill:
    side: str  # "up" or "down"
    fair_prob: float
    best_ask: float
    avg_price: float
    shares: float
    cost: float  # dollars paid for shares
    fee: float  # dollars paid in fees
    edge: float  # fair_prob - (cost + fee) / shares
    limit_price: float = 0.0  # highest price level the fill reached

    @property
    def total_paid(self):
        return self.cost + self.fee


def simulate_buy(asks, budget, fair_prob, cfg):
    """Walk the asks from cheapest up, buying while each level still has enough edge.

    asks: list of (price, size). Returns (shares, cost, fee, highest price taken).
    """
    shares = cost = fee = top = 0.0
    for price, size in sorted(asks):
        fee_per_share = taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
        if fair_prob - price - fee_per_share < cfg.min_edge:
            break
        remaining = budget - cost - fee
        if remaining <= 0:
            break
        take = min(size, remaining / (price + fee_per_share))
        shares += take
        cost += take * price
        fee += take * fee_per_share
        top = price
    return shares, cost, fee, top


def find_bet(fair_up, up_asks, down_asks, cfg, stake=None):
    """Return the better of buying Up or Down if either clears min_edge, else None."""
    stake = cfg.stake_usd if stake is None else stake
    best = None
    for side, prob, asks in (("up", fair_up, up_asks), ("down", 1.0 - fair_up, down_asks)):
        if not asks:
            continue
        shares, cost, fee, top = simulate_buy(asks, stake, prob, cfg)
        if shares <= 0:
            continue
        fill = Fill(
            side=side,
            fair_prob=prob,
            best_ask=min(p for p, _ in asks),
            avg_price=cost / shares,
            shares=shares,
            cost=cost,
            fee=fee,
            edge=prob - (cost + fee) / shares,
            limit_price=top,
        )
        if best is None or fill.edge * fill.shares > best.edge * best.shares:
            best = fill
    return best


def settle_pnl(side, shares, cost, fee, winner):
    """Profit in dollars once the market resolves. Each winning share pays $1."""
    payout = shares if side == winner else 0.0
    return payout - cost - fee
