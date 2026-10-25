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
    # Every level of the book the rule would have bought, with no budget limit: [(price, shares)].
    # Kept with the bet, so it can be replayed later at any other stake (see stakes.py).
    depth: list = None

    @property
    def total_paid(self):
        return self.cost + self.fee


def simulate_buy(asks, budget, fair_prob, cfg, taken=None):
    """Walk the asks from cheapest up, buying while each level still has enough edge.

    asks: list of (price, size). Returns (shares, cost, fee, highest price taken).
    taken: a list to append each level bought, as (price, shares).
    """
    shares = cost = fee = top = 0.0
    for price, size in sorted(asks):
        fee_per_share = taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
        if fair_prob - price - fee_per_share < cfg.min_edge:
            break
        # Risk/reward: what a win pays (1 - cost) against what a loss costs (cost), per share.
        if cfg.min_ratio and 1.0 - (price + fee_per_share) < cfg.min_ratio * (price + fee_per_share) - 1e-12:
            break
        # And, if the owner set one, an upper bound: a share that pays more than that is a long
        # shot. Asks rise from here, so a dearer level may still be in range.
        if getattr(cfg, "max_ratio", 0) and 1.0 - (price + fee_per_share) > cfg.max_ratio * (price + fee_per_share) + 1e-12:
            continue
        remaining = budget - cost - fee
        if remaining <= 0:
            break
        take = min(size, remaining / (price + fee_per_share))
        if taken is not None:
            taken.append((price, take))
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
        if getattr(cfg, "full_fill_only", False) and cost + fee < stake * 0.999:
            continue  # the book can't take the whole stake at good prices: a real order would be killed
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
    if best is not None:
        best.depth = []
        simulate_buy({"up": up_asks, "down": down_asks}[best.side], float("inf"), best.fair_prob, cfg, best.depth)
    return best


def find_lottery(fair_up, up_asks, down_asks, cap, min_fair, cfg, stake=None):
    """A lottery ticket: the whole stake of a side offered at no more than `cap` a share (fee
    included), on a side our model gives at least `min_fair`. The likelier of the sides that
    qualify (the model's view), or None."""
    stake = cfg.stake_usd if stake is None else stake
    best = None
    for side, prob, asks in (("up", fair_up, up_asks), ("down", 1.0 - fair_up, down_asks)):
        if not asks or prob < min_fair:
            continue
        shares = cost = fee = top = 0.0
        depth = []
        for price, size in sorted(asks):
            fee_per_share = taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
            if price + fee_per_share > cap + 1e-12:
                break
            depth.append((price, size))
            take = min(size, (stake - cost - fee) / (price + fee_per_share))
            if take <= 0:
                break
            shares, cost, fee, top = shares + take, cost + take * price, fee + take * fee_per_share, price
        if shares <= 0 or (getattr(cfg, "full_fill_only", False) and cost + fee < stake * 0.999):
            continue
        fill = Fill(side=side, fair_prob=prob, best_ask=min(p for p, _ in asks), avg_price=cost / shares, shares=shares,
                    cost=cost, fee=fee, edge=prob - (cost + fee) / shares, limit_price=top, depth=depth)
        if best is None or fill.fair_prob > best.fair_prob:
            best = fill
    return best


def find_lock(fair_up, up_asks, down_asks, least, cap, edge, cfg, stake=None):
    """The last minute (see lab.py): a side the model gives at least `least`, bought only up to `cap`
    a share (fee included) and while each share keeps `edge` of edge; the whole stake or nothing.
    The likelier side if both qualify."""
    stake = cfg.stake_usd if stake is None else stake
    best = None
    for side, prob, asks in (("up", fair_up, up_asks), ("down", 1.0 - fair_up, down_asks)):
        if not asks or prob < least:
            continue
        shares = cost = fee = top = 0.0
        depth = []
        for price, size in sorted(asks):
            per_share = price + taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
            if per_share > cap + 1e-12 or prob - per_share < edge:
                break
            depth.append((price, size))
            take = min(size, (stake - cost - fee) / per_share)
            if take <= 0:
                break
            shares, cost, fee, top = shares + take, cost + take * price, fee + take * (per_share - price), price
        if shares <= 0 or cost + fee < stake * 0.999:
            continue  # the whole stake, or nothing (as a real fill-or-kill order)
        fill = Fill(side=side, fair_prob=prob, best_ask=min(p for p, _ in asks), avg_price=cost / shares, shares=shares,
                    cost=cost, fee=fee, edge=prob - (cost + fee) / shares, limit_price=top, depth=depth)
        if best is None or fill.fair_prob > best.fair_prob:
            best = fill
    return best


def find_lead(fair_up, up_asks, down_asks, least, cap, cfg, stake=None):
    """The leader at the very end (see lab.py): the side whose best price is at least `least` (the
    market leads it), bought up to `cap` a share (fee included); the whole stake or nothing. No edge
    asked of the model (its chance is only recorded)."""
    stake = cfg.stake_usd if stake is None else stake
    for side, prob, asks in (("up", fair_up, up_asks), ("down", 1.0 - fair_up, down_asks)):
        if not asks or min(p for p, _ in asks) < least:
            continue
        shares = cost = fee = top = 0.0
        depth = []
        for price, size in sorted(asks):
            per_share = price + taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
            if per_share > cap + 1e-12:
                break
            depth.append((price, size))
            take = min(size, (stake - cost - fee) / per_share)
            if take <= 0:
                break
            shares, cost, fee, top = shares + take, cost + take * price, fee + take * (per_share - price), price
        if shares <= 0 or cost + fee < stake * 0.999:
            return None  # the whole stake, or nothing (as a real fill-or-kill order)
        return Fill(side=side, fair_prob=prob, best_ask=min(p for p, _ in asks), avg_price=cost / shares, shares=shares,
                    cost=cost, fee=fee, edge=prob - (cost + fee) / shares, limit_price=top, depth=depth)
    return None


def buy_shares(asks, shares, max_per_share, cfg):
    """Exactly `shares` from the cheapest asks up, if they cost at most `max_per_share` a share on
    average, fee included: (cost, fee, highest price taken, levels taken), or None."""
    got = cost = fee = top = 0.0
    depth = []
    for price, size in sorted(asks or ()):
        if got >= shares - 1e-9:
            break
        take = min(size, shares - got)
        depth.append((price, size))
        got, cost, top = got + take, cost + take * price, price
        fee += take * taker_fee_per_share(price, cfg.fee_rate, cfg.fee_exponent)
    if got < shares - 1e-9 or (cost + fee) / shares > max_per_share + 1e-12:
        return None
    return cost, fee, top, depth


def settle_pnl(side, shares, cost, fee, winner):
    """Profit in dollars once the market resolves. Each winning share pays $1."""
    payout = shares if side == winner else 0.0
    return payout - cost - fee
