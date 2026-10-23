"""Claude's research summary of the lab: a short, exact, list-shaped brief in Hebrew.

Claude reads only the numbers the lab measured and is told to quote them exactly and
never invent any. The answer is constrained to a fixed structure (headline, findings,
top picks, warnings, next experiments) so the dashboard can show it as plain lists.

Needs ANTHROPIC_API_KEY on the server. Without it the lab still works, minus this summary.
"""
import json
import logging
import os
import time

from pydantic import BaseModel

log = logging.getLogger("polybot")

MODEL = "claude-opus-5-5"

SYSTEM = """You are the research desk of a small quantitative trading system that trades Polymarket's \
short "Up or Down" crypto markets (5- and 15-minute windows that resolve on Chainlink prices). The \
system compares a fair probability, computed from the live price, volatility and time left, with the \
Polymarket order book, and bets when the book is cheaper than fair value by a minimum edge after fees. \
A lab runs many variants of that rule on paper, on every market, with real prices and real order books.

You receive the lab's measured results as JSON. Write the daily research brief in Hebrew.

Rules:
- Use only numbers that appear in the data, quoted exactly as given (you may round to one decimal). \
Never estimate, extrapolate or invent a number.
- Every list item is one short line, starts with the fact, and carries its number. No filler.
- top_picks: only variants whose status is "proven", at most 3, on different markets, ranked by \
roi_pessimistic. If none are proven, return an empty list and say so in the headline.
- Judge evidence the way a risk manager would: few bets or a falling recent return is a warning, \
not a finding. Mention when the price feed check (twap vs point agreement) is too low to trust results.
- next_experiments: concrete changes to try in the lab (market, edge, timing, reading of the rules), \
each with the reason from the data.
"""


class Pick(BaseModel):
    rank: int
    market: str
    rule: str
    evidence: list[str]
    risk: str


class Research(BaseModel):
    headline: str
    key_findings: list[str]
    top_picks: list[Pick]
    warnings: list[str]
    next_experiments: list[str]


def configured():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def lab_brief(lab, readiness, feed, hours):
    """The facts Claude gets: compact, numbers only."""
    pick = lambda b: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()
                      if k not in ("first_ts", "last_ts", "paid")}
    return {
        "hours_of_data": round(hours, 1),
        "settled_lab_bets": lab["settled_bets"],
        "ranking_rule": "roi = mean return per dollar staked; roi_pessimistic = mean - 2.5 standard errors; "
                        "proven = at least 30 bets, roi_pessimistic > 0 and the recent half profitable",
        "leaderboard_top": [pick(b) for b in lab["board"][:20]],
        "markets": lab["markets"],
        "price_feed_agreement": feed,
        "main_strategy_readiness": readiness,
    }


# Dollars per million tokens (input, output), for the operating-cost line on the dashboard.
PRICES = {"claude-opus-5-5": (4.0, 20.0)}


def cost_of(model, usage):
    """What one call cost, from the usage the API returned (an estimate if the model is unknown)."""
    inp, out = PRICES.get(model, PRICES[MODEL])
    tokens_in = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * 1.25
    tokens_in += (getattr(usage, "cache_read_input_tokens", 0) or 0) * 0.05
    return (tokens_in * inp + (getattr(usage, "output_tokens", 0) or 0) * out) / 1e6


def spent_since(db, since):
    """Dollars spent on Claude since a timestamp."""
    return db.execute("SELECT COALESCE(SUM(cost), 0) FROM research WHERE ts >= ?", (since,)).fetchone()[0]


class NoSummary(RuntimeError):
    """The call went through (and is recorded with its cost) but returned no summary."""


def run(store, lab, readiness, feed, hours):
    """Ask Claude for today's brief and store it. Returns the stored dict or raises."""
    import anthropic

    client = anthropic.Anthropic()
    brief = lab_brief(lab, readiness, feed, hours)
    started = time.time()
    try:
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(brief, ensure_ascii=False)}],
            output_format=Research,
            output_config={"effort": "high"},
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        cost = cost_of(response.model, response.usage)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            store.add_research(started, response.model, error=f"no summary (stop reason: {response.stop_reason})",
                               cost=cost)
            raise NoSummary(f"no summary returned (stop reason: {response.stop_reason})")
        data = response.parsed_output.model_dump()
        store.add_research(started, response.model, data=data, cost=cost)
        log.info("Research summary written by %s", response.model)
        return data
    except NoSummary:
        raise
    except Exception as e:
        store.add_research(started, MODEL, error=f"{type(e).__name__}: {e}")
        raise


def latest(db):
    row = db.execute("SELECT ts, model, data, error FROM research ORDER BY ts DESC LIMIT 1").fetchone()
    if not row:
        return None
    return {"ts": row["ts"], "model": row["model"], "error": row["error"],
            "data": json.loads(row["data"]) if row["data"] else None}
