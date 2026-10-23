"""Talking to the outside world: Coinbase for spot prices, Polymarket for markets and order books.

Read-only. Nothing in this file can place an order or touch money.
"""
import json
import urllib.request

from .config import COINBASE_PRODUCTS

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
COINBASE = "https://api.exchange.coinbase.com"


def get_json(url, timeout=3.0):
    req = urllib.request.Request(url, headers={"User-Agent": "polybot-recorder/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def spot_price(asset):
    data = get_json(f"{COINBASE}/products/{COINBASE_PRODUCTS[asset]}/ticker")
    return float(data["price"])


def window_start(ts, minutes):
    size = minutes * 60
    return int(ts // size) * size


def market_slug(cfg, asset, start):
    return cfg.slug_template.format(asset=asset, minutes=cfg.window_minutes, start=start)


def _maybe_json(value):
    # Gamma returns some list fields as JSON-encoded strings.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _price_to_beat(event, market):
    """Polymarket's own opening price (from Chainlink) if the API exposes it."""
    for source in (event, market):
        meta = _maybe_json(source.get("eventMetadata")) or {}
        if isinstance(meta, dict) and meta.get("priceToBeat") not in (None, ""):
            try:
                return float(meta["priceToBeat"])
            except (TypeError, ValueError):
                pass
    return None


def find_market(slug):
    """Look up a window's market. Returns a dict, or None if Polymarket has no such market."""
    events = get_json(f"{GAMMA}/events?slug={slug}")
    event = events[0] if events else {}
    markets = event.get("markets") or get_json(f"{GAMMA}/markets?slug={slug}")
    if not markets:
        return None
    market = markets[0]
    outcomes = [o.lower() for o in _maybe_json(market.get("outcomes"))]
    tokens = _maybe_json(market.get("clobTokenIds"))
    by_outcome = dict(zip(outcomes, tokens))
    if "up" not in by_outcome or "down" not in by_outcome:
        return None
    return {
        "slug": slug,
        "up_token": by_outcome["up"],
        "down_token": by_outcome["down"],
        "price_to_beat": _price_to_beat(event, market),
        "resolution_source": market.get("resolutionSource") or event.get("resolutionSource"),
    }


def market_winner(slug):
    """"up" / "down" once Polymarket has resolved the market, else None."""
    markets = get_json(f"{GAMMA}/markets?slug={slug}")
    if not markets or not markets[0].get("closed"):
        return None
    market = markets[0]
    outcomes = [o.lower() for o in _maybe_json(market.get("outcomes"))]
    prices = [float(p) for p in _maybe_json(market.get("outcomePrices"))]
    for outcome, price in zip(outcomes, prices):
        if price >= 0.99:
            return outcome
    return None


def order_book(token_id):
    """Returns (bids, asks) as lists of (price, size)."""
    book = get_json(f"{CLOB}/book?token_id={token_id}")

    def levels(key):
        return [(float(lv["price"]), float(lv["size"])) for lv in book.get(key) or []]

    return levels("bids"), levels("asks")
