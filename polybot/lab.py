"""The lab: many variants of the strategy, run on paper side by side on every market, and an honest ranking.

Testing hundreds of variants guarantees that some look great by luck, so a variant goes
through stages, each judged on data it hasn't been judged on before:

1. collecting: fewer than lab_min_bets settled bets.
2. candidate: judged only at fixed checkpoints (100, 200, 400, ... trades; never after every trade,
   which lets luck through sooner or later), on trades with real fills only, by a test that knows
   what each trade had to win (price_z: wins against the price paid, fee included, so a streak of
   wins at 97c proves little), at lab_gate_z, and its recent half above break-even too.
   From that moment on, its bets are counted separately: new data, unseen when it was picked.
3. proven: on that new data alone, at the same checkpoints, at lab_gate_oos_z; or failed (below
   break-even at a checkpoint). Both are final: the first verdict stands.
(The five expert panels of 27.10.2026: the old gate, mean minus z standard errors re-checked after
every bet, passed rules with no edge at all, most of all at high prices.)

The search near winners (explore) adds neighbours of promising variants: a slightly
different edge or timing. They start at stage 1 like any other, on their own new data.
"""
import dataclasses
import functools
import math
from dataclasses import dataclass

# (3 points is gone: about what the fee and a moved price take, so it only found luck)
EDGES = (0.05, 0.08, 0.12)
ENTRY_WINDOWS = {
    15: ((15, 60), (15, 180), (15, 600), (5, 30)),
    5: ((10, 45), (10, 120), (10, 240), (5, 30), (10, 60)),
}
MAKER_FROM = 120  # a resting order needs time to be filled: only windows opening this early
# Measured against real outcomes: the 60s TWAP reading matched Polymarket in 61 of 61
# resolved windows, the point reading in 58 of 61, so only TWAP variants are run.
# Windows still record both readings, and the dashboard keeps checking them.
# The fat-tailed pricing of the same reading ("twapt") is gone from the grid: a near twin of "twap",
# it doubled the versions tested, and with them the ones that look good by chance.
RESOLUTIONS = ("twap",)


@dataclass(frozen=True)
class Variant:
    min_edge: float
    min_seconds_left: int
    max_seconds_left: int
    resolution: str
    # Risk/reward, part of the rule itself: buy only where a win pays at least this much per
    # dollar at risk (0.7 = $7 for every $10). 0 = no filter. The lab tests the same rule with
    # and without it, so a filter that helps is proven like anything else and then runs as is.
    min_ratio: float = 0.0
    # A filter put on the rule where it runs (the simulation or real money), by the system's
    # research on this rule's own trades or by the owner. Not part of the rule's name: the
    # strategy stays the same strategy, with the same record.
    filter_ratio: float = 0.0
    # The amount a round where it runs (the simulation or real money), set by the system's research
    # on this rule's own trades or by the owner; 0 = the mode's amount. Not part of the name either.
    stake: float = 0.0
    # How the lab fills a maker's resting order: "" when sellers break through it (the pessimistic
    # way: only the fills that come with the price going against us), "q" by its place in the queue
    # (the shares resting at its price when it was placed have to trade first). Part of the name: the
    # two are measured side by side.
    fill: str = ""
    # The filters a strategy carries (the lab tests them as part of it, then they go wherever it runs):
    # buy only at an average price a share in [price_lo, price_hi) (0, 0: any price), and only when our
    # model gives the side at least min_chance (0: any chance). Part of the name.
    price_lo: float = 0.0
    price_hi: float = 0.0
    min_chance: float = 0.0
    # How it enters, from a finding of the research center (the owner, 29.10.2026: what the research learns
    # becomes a cloned strategy by itself, see ENTRIES): "" as always (take what's offered at the price
    # seen); "iW": a refused order stays open W seconds and buys at a steady offer still within the edge;
    # "mK": up to K cents over the price seen, within the edge; "sU": a resting bid U cents under the offer
    # seen, at the signal, instead of taking it (no fee). Part of the name.
    entry: str = ""
    # When it may enter, from a finding of the research center (see GATES): "" any time; "f" our model fed by
    # Binance's futures instead of spot (the futures move first); "k" only while the lab's recent trades are
    # winning (the streaks); "p<feature>" only when the signal the prediction study found points the same way.
    gate: str = ""

    @property
    def ratio(self):
        """The risk/reward filter in force: the rule's own, or the one put on it, whichever is stricter."""
        return max(self.min_ratio, self.filter_ratio)

    @property
    def key(self):
        base = f"{self.resolution}|{self.min_edge:.2f}|{self.min_seconds_left}-{self.max_seconds_left}"
        return (base + (f"|r{self.min_ratio:.1f}" if self.min_ratio else "")
                + (f"|b{round(self.price_lo * 100)}-{round(self.price_hi * 100)}" if self.price_hi else "")
                + (f"|c{round(self.min_chance * 100)}" if self.min_chance else "") + ("|q" if self.fill == "q" else "")
                + (f"|e{self.entry}" if self.entry else "") + (f"|g{self.gate}" if self.gate else ""))

    @classmethod
    def from_key(cls, key):
        resolution, edge, window, *rest = key.split("|")
        lo, hi = window.split("-")
        ratio = next((float(x[1:]) for x in rest if x.startswith("r")), 0.0)
        band = next((x[1:].split("-") for x in rest if x.startswith("b")), None)
        chance = next((int(x[1:]) / 100 for x in rest if x.startswith("c")), 0.0)
        entry = next((x[1:] for x in rest if x.startswith("e")), "")
        gate = next((x[1:] for x in rest if x.startswith("g")), "")
        return cls(float(edge), int(lo), int(hi), resolution, ratio, fill="q" if "q" in rest else "",
                   price_lo=int(band[0]) / 100 if band else 0.0, price_hi=int(band[1]) / 100 if band else 0.0,
                   min_chance=chance, entry=entry, gate=gate)

    def with_ratio(self, ratio):
        return dataclasses.replace(self, min_ratio=ratio, filter_ratio=0.0, stake=0.0)

    def passes(self, avg_price, chance):
        """The strategy's own filters on a buy: its price group and its least chance."""
        if self.price_hi and not self.price_lo - 1e-9 <= avg_price < self.price_hi - 1e-9:
            return False
        return not self.min_chance or (chance is not None and chance >= self.min_chance - 1e-9)


# The risk/reward filters the lab tries on promising rules and on the owner's own: a win paying
# at least $3, $5 or $7 for every $10 at risk (buying up to 77¢, 67¢, 59¢ a share, fee included).
RATIOS = (0.3, 0.5, 0.7)


def takes_ratio(v):
    """Rules a risk/reward filter applies to: the usual taker rules (not maker orders, whose price
    is their own, nor lottery tickets, which are capped at a few cents anyway)."""
    return v.resolution in ("twap", "twapt", "twap3", "point")


# Maker variants ("twapm"): instead of buying an existing offer (and paying the taker fee),
# post a buy order at our own price and wait for a seller. No fee, a better price, but filled
# only when the market comes to us, which is often when it moves against us: the lab measures it.
MAKER_EDGES = (0.03, 0.05, 0.08)

# Both sides ("pair<T>"): buy the side that turned cheap (at most min_edge a share, fee included),
# then the other side the moment the two together cost at most T cents a share (fees included),
# the same number of shares of each. One of the two always pays $1 a share, so once both are in, the
# round makes (1 - pair cost) a share whoever wins. The risk is the second side never getting that
# cheap: then the first is an ordinary bet on its side. Known on Polymarket as "pair cost" or
# "dump and hedge". The first side only while at least min_seconds_left remain, so there is time
# for the second; the second until PAIR_LAST seconds before the close.
# On a first day of recordings the cheap first legs (15-25c) with a low target (80-85c) did best
# and the dearer ones lost; the grid spans both, and the lab's stages decide.
# Kept few: a both-sides rule enters nearly every round, so each version adds hundreds of trades a
# day (40 versions a market were 16,000 trades an hour, more than the server could keep up with).
PAIR_CAPS = (0.15, 0.25)
PAIR_TARGETS = (80, 90)
PAIR_FIRST = {15: ((450, 885),), 5: ((150, 285),)}
PAIR_LAST = 5

# The last minute ("lock<P>"): Polymarket settles on the average of the last 60 seconds, so in that
# minute part of the result is already locked in, and the model (fair_up_probability_twap) counts
# the part that has been. On recordings the market seemed to price the last minute by the price of
# the moment: when the model gave a side 50% or more and the market sold it under 50c, it won 84
# times in 100 (bought at 31c on average). A rule buys the side the model gives at least P%, only
# up to min_edge a share (fee included) and with at least LOCK_EDGE of edge, between
# max_seconds_left and min_seconds_left before the close. Small amounts (LOCK_STAKE): the cheap
# shares in the last seconds are few, and a real order must find them still there.
LOCK_LEAST = (60, 75)
LOCK_CAPS = (0.50, 0.85)
LOCK_WINDOWS = ((3, 60), (3, 25))
LOCK_EDGE = 0.02
LOCK_STAKE = 5.0

# The leader at the very end ("lead<F>", in the last minute's tab): with 30 to 15 seconds left, buy
# the side the market itself leads (its price at least F cents), up to min_edge a share (fee
# included), LOCK_STAKE at a time. No model: it asks only whether the market underprices the leader in
# the last seconds. On recordings, the trades whose price was still the same then were mostly the
# trailing side (about 23c), which won only 16 times in 100: the leader's side of those looked cheap.
LEAD_LEAST = (55, 70)
LEAD_CAPS = (0.80, 0.90, 0.95)
LEAD_WINDOWS = ((15, 30),)

# The market's leader as a maker ("fav<F>"): a resting buy on the side the market leads (priced at
# least F cents), a tick under its offer and at most min_edge, with no model. The other face of the
# maker check's finding: the people taking the trailing side (30-40c with 3 to 5 minutes left, or
# under 10c in the last two minutes) lose to whoever sells it to them, and selling the trailing side
# at 35c is the same as buying the leader at 65c. Filled only when sellers come down through it.
FAV_RULES = ((60, 0.65, 180, 300), (60, 0.70, 180, 300), (90, 0.97, 30, 120))

# The owner's strategy of 26.10.2026 ("filtered"): the usual rule (our model's chance over the price),
# in the last two minutes only, on every coin, carrying its own filters: a price group and a least
# chance. The most a trade can lose is its amount (the same for every strategy). The owner's rule
# only (70-80c, 85%); the lab's stages decide.
FILTERED_BANDS = ((0.70, 0.80),)  # (one rule: its neighbours only multiplied the tests, the panels)
FILTERED_CHANCES = (0.85,)
FILTERED_EDGE = 0.02
FILTERED_WINDOWS = ((5, 120),)

# The end-of-round favourite ("end", the owner's idea of 27.10.2026, in the tab the both-sides rules
# had): in the round's last seconds, buy the side the market prices at least price_lo, up to
# price_hi (END_CAP at most) a share fee included, the whole amount or nothing. Polymarket settles on
# the last 60 seconds' average, so by then part of the result is locked in; whether the market's
# 90-99c prices it right is the question. "endm": the same, only when our model gives the side at
# least what it costs (fee included). Twelve rules, set before any data (the panels): four entry
# windows x two price groups, and four with the model. One loss at 97c costs 35 wins, so they are
# judged pooled over every coin (end_pooled), on wins against break-even, at checkpoints.
END_WINDOWS = ((45, 60), (30, 45), (15, 30), (5, 15))
END_BANDS = ((0.90, 0.95), (0.95, 1.00))
END_MODEL_WINDOWS = ((30, 60), (5, 30))
END_CAP = 0.995  # the most a share may cost, fee included (a win still pays half a cent)
END_STAKE = 5.0

# Consensus ("cons<K><g|a>", the owner's idea of 29.10.2026): the lab runs dozens of rules on every round;
# when at least K of the good ones ("g": promising, candidate or proven on the lab's board; "a": any rule,
# the control) went into the same side of a round, and none into the other, enter that side too, the whole
# amount at no more than min_edge a share (fee included). Whether many rules agreeing is worth more than
# any one of them: judged as every rule, and against the good rules' own pooled result (Engine.consensus_view).
CONS_RULES = ((3, "g", 0.85), (5, "g", 0.85), (3, "g", 0.70), (3, "a", 0.85))


def is_cons(v):
    return v.resolution.startswith("cons")


def consensus_side(v, votes, good):
    """The side K agreeing rules bought in this round (votes: rule key -> side), or None: K or more of them
    on one side and none on the other; only the good rules' votes for a "g" rule."""
    need, only_good = int(v.resolution[4:-1]), v.resolution.endswith("g")
    n = {"up": 0, "down": 0}
    for k, side in votes.items():
        if k.startswith("cons") or (only_good and k not in good):
            continue
        n[side] = n.get(side, 0) + 1
    for side, other in (("up", "down"), ("down", "up")):
        if n[side] >= need and n[other] == 0:
            return side
    return None


# Several entries ("twap3"): the same rule, but up to three entries a window (a third of the stake
# each, the edge checked again each time, ENTRY_GAP seconds apart) instead of one: whether
# going in more than once beats a single bet of the whole amount.
MULTI_ENTRIES = 3
MULTI_EDGES = (0.05, 0.08)
ENTRY_GAP = 20


def variants_for(minutes):
    entries = ENTRY_WINDOWS.get(minutes) or ((15, minutes * 60 * 2 // 3),)
    taker = [Variant(e, lo, hi, r) for r in RESOLUTIONS for e in EDGES for lo, hi in entries]
    maker = [Variant(e, lo, hi, "twapm") for e in MAKER_EDGES for lo, hi in entries if hi >= MAKER_FROM]
    multi = [Variant(e, lo, hi, "twap3") for e in MULTI_EDGES for lo, hi in entries]
    # (the both-sides rules no longer run: their legs filled at once, not as real orders would, so
    # their results said nothing. Their rules stay readable for the trades they made.)
    favs = [Variant(cap, lo, hi, f"fav{f}") for f, cap, lo, hi in FAV_RULES if hi <= minutes * 60 - 30]
    filtered = [Variant(FILTERED_EDGE, lo, hi, "twap", price_lo=b_lo, price_hi=b_hi, min_chance=c)
                for lo, hi in FILTERED_WINDOWS for b_lo, b_hi in FILTERED_BANDS for c in FILTERED_CHANCES]
    # (the last minute's kinds, locks and leads, no longer run: they didn't work. Their rules stay
    # readable for the trades they made.)
    end = [Variant(0.0, lo, hi, "end", price_lo=b_lo, price_hi=b_hi) for lo, hi in END_WINDOWS for b_lo, b_hi in END_BANDS]
    end += [Variant(0.0, lo, hi, "endm", price_lo=b_lo, price_hi=b_hi) for lo, hi in END_MODEL_WINDOWS for b_lo, b_hi in END_BANDS]
    cons = [Variant(cap, 5, minutes * 60 - 30, f"cons{k}{g}") for k, g, cap in CONS_RULES]
    return taker + maker + multi + favs + filtered + end + cons


# The lab's entry times, by when a strategy starts buying (the window's earliest second before the
# close). The owner may run the lab on some of them only; by default, all.
TIME_GROUPS = (
    ("30", "30 השניות האחרונות", 30),
    ("60", "הדקה האחרונה", 60),
    ("120", "2 הדקות האחרונות", 120),
    ("240", "4 הדקות האחרונות", 240),
    ("600", "10 הדקות האחרונות", 600),
    ("more", "מוקדם יותר", None),
)


def time_group(key):
    """Which entry time a variant belongs to (see TIME_GROUPS)."""
    try:
        hi = Variant.from_key(key).max_seconds_left
    except (ValueError, IndexError):
        return "more"
    return next(g for g, _, top in TIME_GROUPS if top is None or hi <= top)


@functools.lru_cache(maxsize=None)
def family(key):
    """Which part of the lab a variant belongs to: the both-sides rules and the makers (a resting order
    that waits to be filled: on the model's price, or on the market's leader) have their own tabs."""
    return ("pair" if key.startswith("pair") else "end" if key.startswith("end")
            else "maker" if key.startswith(("twapm", "fav", "zone")) else "main")


def is_maker(key):
    return family(key) == "maker"


FAMILIES = ("main", "end", "maker")  # (the both-sides rules' tab went to the end-of-round favourite)


def reading(v):
    """The model reading a variant decides on (the special kinds all use the 60-second average)."""
    return ("twap" if v.resolution in ("twapm", "twap3") or is_lotto(v) or is_cons(v) or is_lock(v) or is_fav(v) or is_zone(v) or is_end(v)
            else v.resolution)


def is_lotto(v):
    return v.resolution.startswith("lotto")


def is_lock(v):
    """The last minute's kinds (both small, a single entry, on the 60-second average): the locked
    average's ("lock") and the market's leader ("lead")."""
    return v.resolution.startswith(("lock", "lead"))


# A maker's zone ("zone<lo><hi>", cents, three digits each): built from what the maker check found (a
# price range and a time before the close where taking a side lost to whoever was on the other side of
# it): a resting buy on whichever side is priced in [lo, hi), a tick under its offer, at that time.
# min_edge holds hi (the most it pays).
def zone_key(lo, hi, t_lo, t_hi):
    return f"zone{round(lo * 100):03d}{round(hi * 100):03d}|{hi:.2f}|{max(3, t_lo)}-{t_hi}"


def is_zone(v):
    return v.resolution.startswith("zone")


def zone_band(v):
    return int(v.resolution[4:7]) / 100, int(v.resolution[7:10]) / 100


def is_end(v):
    """The end-of-round favourite ("end", or "endm" when our model must agree too)."""
    return v.resolution.startswith("end")


def is_fav(v):
    return v.resolution.startswith("fav")


def fav_least(v):
    """The least price that makes a side the leader."""
    return int(v.resolution[3:]) / 100


def is_lead(v):
    return v.resolution.startswith("lead")


def lead_least(v):
    """The least price that makes a side the leader."""
    return int(v.resolution[4:]) / 100


def lock_least(v):
    """The least chance the model must give the side bought."""
    return int(v.resolution[4:]) / 100


def is_pair(v):
    return v.resolution.startswith("pair")


def pair_target(v):
    """The most the two sides may cost together, per share (fees included)."""
    return int(v.resolution[4:]) / 100


def lotto_min_fair(v):
    return int(v.resolution[5:] or 0) / 100


def entries(v):
    """How many times a variant may enter in one window."""
    return MULTI_ENTRIES if v.resolution == "twap3" else 1


def describe(key):
    """A variant's rule in plain Hebrew."""
    v = Variant.from_key(key)
    when = f"{v.max_seconds_left // 60} הדקות האחרונות" if v.max_seconds_left >= 120 else f"{v.max_seconds_left} השניות האחרונות"
    q = " · מילוי לפי התור (הוגן יותר)" if v.fill == "q" else ""
    if is_cons(v):
        k, good = int(v.resolution[4:-1]), v.resolution.endswith("g")
        return (f"הסכמה: כשלפחות {k} {'אסטרטגיות טובות' if good else 'אסטרטגיות (כל אחת, ביקורת)'} במעבדה נכנסו לאותו צד בסבב,"
                f" ואף אחת לא לצד השני, נכנסים גם, עד {v.min_edge * 100:.0f}¢ למניה (עמלה כלולה)")
    if is_end(v):
        return (f"הפייבוריט בסוף הסבב: ב-{v.max_seconds_left} עד {v.min_seconds_left} השניות שלפני הסגירה, קונים את הצד שהשוק"
                f" מתמחר {v.price_lo * 100:.0f}¢ עד {min(v.price_hi, END_CAP) * 100:.1f}¢ (עמלה כלולה), ${END_STAKE:g}, הכל או כלום"
                + (". רק כשגם המודל שלנו נותן לצד לפחות את מה שהוא עולה" if v.resolution == "endm" else ". בלי מודל"))
    if is_zone(v):
        lo, hi = zone_band(v)
        return (f"עושה שוק מהממצאים: ב-{v.max_seconds_left} עד {v.min_seconds_left} השניות שלפני הסגירה, הזמנת קנייה ממתינה על הצד"
                f" שהמחיר שלו בין {lo * 100:.0f}¢ ל-{hi * 100:.0f}¢, טיק מתחת להצעה, בלי עמלה. נבנתה ממה שבדיקת עושי השוק מצאה" + q)
    if is_fav(v):
        return (f"עושה שוק על המוביל: ב-{v.max_seconds_left} עד {v.min_seconds_left} השניות שלפני הסגירה, הזמנת קנייה ממתינה על הצד"
                f" שהשוק מוביל (מחיר {fav_least(v) * 100:.0f}¢ ומעלה), טיק מתחת להצעה ולא יותר מ-{v.min_edge * 100:.0f}¢, בלי עמלה."
                " בלי מודל: כמו למכור את הצד החלש למי שקונה אותו ביוקר" + q)
    if is_lead(v):
        return (f"המוביל בסוף: ב-{v.max_seconds_left} עד {v.min_seconds_left} השניות שלפני הסגירה, קונים את הצד שהשוק מוביל"
                f" (מחיר {lead_least(v) * 100:.0f}¢ ומעלה) עד {v.min_edge * 100:.0f}¢ למניה (עמלה כלולה), ${LOCK_STAKE:g} בכל פעם."
                " בלי מודל: האם השוק מעריך בחסר את המוביל בשניות האחרונות")
    if is_lock(v):
        return (f"הדקה האחרונה: כשהממוצע שכבר נעול נותן לצד לפחות {lock_least(v) * 100:.0f}%, קונים אותו עד"
                f" {v.min_edge * 100:.0f}¢ למניה (עמלה כלולה), ב-{v.max_seconds_left} עד {v.min_seconds_left} השניות"
                f" שלפני הסגירה, ${LOCK_STAKE:g} בכל פעם")
    if is_pair(v):
        lo = f"{v.min_seconds_left // 60} דקות" if v.min_seconds_left >= 120 else f"{v.min_seconds_left} שניות"
        return (f"שני הצדדים: קונים צד כשהוא עד {v.min_edge * 100:.0f}¢, ואת הצד השני כשהשניים יחד עולים עד"
                f" {pair_target(v) * 100:.0f}¢ למניה (עמלות כלולות). הצד הראשון רק כשנשארו לפחות {lo}")
    if is_lotto(v):
        least = lotto_min_fair(v)
        return (f"כרטיס לוטו: קונים רק עד {v.min_edge * 100:.0f}¢ למניה, ב-{when}"
                + (f", כשהמודל נותן לצד הזה לפחות {least * 100:.0f}%" if least else ""))
    if v.resolution == "twap3":
        return (f"פער של {v.min_edge * 100:.0f} נקודות ומעלה, ב-{when}, עד {MULTI_ENTRIES} כניסות (שליש מהסכום בכל אחת)" + ratio_text(v)
                + entry_text(v) + gate_text(v))
    if v.resolution == "twapm":
        return f"פקודה ממתינה במחיר שלנו (בלי עמלה), פער של {v.min_edge * 100:.0f} נקודות, ב-{when}" + q
    reading = {"twap": "ממוצע 60 שניות", "twapt": "ממוצע 60 שניות, עם קפיצות פתאומיות (זנבות עבים)"}.get(
        v.resolution, "מחיר בשנייה המדויקת")
    return f"פער של {v.min_edge * 100:.0f} נקודות ומעלה, ב-{when}, לפי {reading}" + ratio_text(v) + entry_text(v) + gate_text(v)


# The entries a finding can give a strategy (see Variant.entry): (kind, number) from "i15", "m3", "s1".
ENTRY_KINDS = ("i", "m", "s")
TAKERS = ("twap", "twapt", "twap3")  # the rules that take the offer on our model's edge: the ones an entry fits


def entry_of(v):
    e = getattr(v, "entry", "") or ""
    if len(e) < 2 or e[0] not in ENTRY_KINDS:
        return None, 0
    try:
        return e[0], int(e[1:])
    except ValueError:
        return None, 0


def entry_text(v):
    kind, n = entry_of(v)
    return {"i": f" · מהמחקר: הזמנה שנדחתה מחכה עד {n} שניות ונכנסת כשהמחיר עדיין שווה",
            "m": f" · מהמחקר: עד {n}¢ מעל המחיר שנראה, כל עוד נשאר היתרון",
            "s": f" · מהמחקר: הצעת קנייה שעומדת בספר {n}¢ מתחת להצעה, במקום לקנות (בלי עמלה)"}.get(kind, "")


GATES = ("f", "k", "pmom", "pflow", "pbrk15", "pbrk60")
PREDICT_WORDS = {"mom": "תנועת המטבע", "flow": "זרימת הקניות", "brk15": "פריצה של 15 דקות", "brk60": "פריצה של שעה"}


def gate_text(v):
    g = getattr(v, "gate", "") or ""
    if g == "f":
        return " · מהמחקר: המודל קורא את החוזים העתידיים של בינאנס"
    if g == "k":
        return " · מהמחקר: רק כשהטריידים האחרונים של המעבדה מרוויחים"
    if g.startswith("p"):
        return f" · מהמחקר: רק כש{PREDICT_WORDS.get(g[1:], g[1:])} מצביעה לאותו צד"
    return ""


def ratio_text(v):
    out = ""
    if v.min_ratio:
        out += f", רק כשניצחון מרוויח לפחות ${v.min_ratio * 10:g} על כל $10 (קונים עד {100 / (1 + v.min_ratio):.0f}¢)"
    if v.price_hi:
        out += f", רק במחיר {v.price_lo * 100:.0f}¢–{v.price_hi * 100:.0f}¢ למניה"
    if v.min_chance:
        out += f", רק כשהסיכוי שלנו לצד {v.min_chance * 100:.0f}% ומעלה"
    return out


def money_weighted(rows, z):
    """(return per dollar, pessimistic return per dollar) for bets given as (pnl, paid).

    Return = total profit / total staked, so it always agrees with the dollar figures.
    Pessimistic = that minus z standard errors, with each bet weighted by its stake and
    the effective number of bets reflecting uneven stakes. None with under two bets.
    """
    paid = sum(p for _, p in rows)
    if paid <= 0:
        return None, None
    mean = sum(x for x, _ in rows) / paid
    n_eff = paid ** 2 / sum(p * p for _, p in rows)
    if n_eff < 2:
        return mean, None
    var = sum(p * (x / p - mean) ** 2 for x, p in rows) / paid
    return mean, mean - z * math.sqrt(var * n_eff / (n_eff - 1) / n_eff)


def neighbours(key, minutes):
    """Variants a small step away from `key`: edge one or two points either side, and an entry
    window a third shorter or half again longer."""
    v = Variant.from_key(key)
    if is_cons(v):
        return []  # (a consensus rule has no neighbours: its rules are the others)
    length = minutes * 60
    out = set()
    for de in (-0.02, -0.01, 0.01, 0.02):
        edge = round(v.min_edge + de, 2)
        if 0.01 <= edge <= 0.20:
            out.add(dataclasses.replace(v, min_edge=edge, fill="").key)
    for factor in (2 / 3, 1.5):
        hi = int(round(v.max_seconds_left * factor / 5) * 5)
        if v.min_seconds_left + 20 <= hi <= length - 30 and hi != v.max_seconds_left:
            out.add(dataclasses.replace(v, max_seconds_left=hi, fill="").key)
    out.discard(key)
    return sorted(out)


def ratio_versions(key):
    """The same rule with each risk/reward filter the lab tries (none for rules it doesn't fit)."""
    v = Variant.from_key(key)
    if not takes_ratio(v) or v.min_ratio or v.price_hi:
        return []
    return [v.with_ratio(r).key for r in RATIOS]


# The checkpoints a rule is judged at (trades): 100, 200, 400, ... Between them nothing changes.
FIRST_CHECKPOINT = 100


def checkpoints(n):
    """The checkpoints reached with n trades, in order."""
    out, c = [], FIRST_CHECKPOINT
    while c <= n:
        out.append(c)
        c *= 2
    return out


def next_checkpoint(n):
    c = FIRST_CHECKPOINT
    while c <= n:
        c *= 2
    return c


def price_z(trades):
    """How far a rule's wins are above what its prices needed, in standard errors: trades as
    (won, break_even) with break_even the cost of a share fee included (what a win pays back).
    Each trade's spread comes from its price (b(1-b)), not from the sample, so a streak without a
    loss isn't taken for certainty. None without trades."""
    trades = [(w, b) for w, b in trades if 0 < b < 1]
    var = sum(b * (1 - b) for _, b in trades)
    if not trades or var <= 0:
        return None
    return sum(w - b for w, b in trades) / math.sqrt(var)


def next_stage(cfg, status, n, oos_n, first_ts, last_ts):
    """What it takes to reach the next stage: trades so far, the next checkpoint, and how many
    trades an hour it places (to turn that into time)."""
    hours = (last_ts - first_ts) / 3600 if last_ts > first_ts else 0
    rate = n / hours if hours >= 0.25 else None
    if status == "collecting":
        return {"to": "promising", "have": n, "need": cfg.lab_min_bets, "rate": rate}
    if status in ("promising", "losing"):
        return {"to": "candidate", "have": n, "need": next_checkpoint(n), "rate": rate}
    if status == "candidate":
        return {"to": "proven", "have": oos_n, "need": next_checkpoint(oos_n), "rate": rate}
    return None


def leaderboard(db, cfg, current=None, candidates=None, dynamic=None, since=None, real_since=None):
    """Every lab variant's paper results, best first, plus the top picks and per-market totals.

    current: the (asset, minutes, variant) keys the lab runs now; retired variants are left out.
    candidates: {(asset, minutes, variant): when it became a candidate}.
    dynamic: keys added by the search near winners (marked as such).
    since: the owner started over (one time for all, or one a family: {"main": ts, "pair": ts,
           "maker": ts}): the headline money figures (profit / loss and return) count from then on.
           The ranking, the trade counts and the stages always use every trade.
    real_since: when the lab's fills became a real order's (half a second late, against the book
           then): the test on new data counts only trades from then on, so a strategy is proven
           only on fills real money would have had.
    """
    candidates = candidates or {}
    dynamic = dynamic or set()
    # (only the variants that run: retired ones' trades stay in the database, unread)
    variants = sorted({v for _, _, v in current}) if current is not None else None
    # (plain dicts, not sqlite rows: a row's field read by name costs several times a dict's, and the
    # lab has millions of trades)
    cur = db.cursor()
    cur.row_factory = None
    cur.execute(
        "SELECT asset, minutes, variant, ts, pnl, cost + fee AS paid, window_start, side, shares FROM bets"
        " WHERE mode = 'lab' AND pnl IS NOT NULL AND cost + fee > 0 AND shares > 0 AND ts >= ?"
        + (f" AND variant IN ({','.join('?' * len(variants))})" if variants is not None else "") + " ORDER BY ts",
        (0, *(variants or ())))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur]
    cut = since if isinstance(since, dict) else {f: since for f in FAMILIES}
    groups = {}
    fresh, fresh_all = {f: [] for f in FAMILIES}, []  # the running rules' trades since the owner started over
    for r in rows:
        key = (r["asset"], r["minutes"], r["variant"])
        if current is None or key in current:
            groups.setdefault(key, []).append(r)
            f = family(r["variant"])
            if r["ts"] >= (cut.get(f) or 0):
                fresh_all.append(r)
                if f in fresh:
                    fresh[f].append(r)

    board = []
    for (asset, minutes, variant), bets in groups.items():
        locked = None
        if family(variant) == "pair":
            bets, locked = _rounds(bets)
        rows_ = [(b["pnl"], b["paid"]) for b in bets]
        n = len(rows_)
        mean, lower = money_weighted(rows_, cfg.lab_z)
        recent, _ = money_weighted(rows_[n // 2:], cfg.lab_z)
        paid = sum(b["paid"] for b in bets)
        pnl = sum(b["pnl"] for b in bets)
        key = (asset, minutes, variant)
        since = candidates.get(key)
        start = None if since is None else max(since, real_since or 0)
        oos_bets = [b for b in bets if start is not None and b["ts"] >= start]
        oos = [(b["pnl"], b["paid"]) for b in oos_bets]
        oos_roi, oos_lower = money_weighted(oos, cfg.lab_oos_z) if oos else (None, None)
        # The verdicts: price_z at the checkpoints only, on real fills only (see the header).
        won = lambda bs: [(1.0 if b["pnl"] > 0 else 0.0, b["paid"] / b["shares"]) for b in bs
                          if "shares" in b.keys() and b["shares"]]  # (a both-sides round has no one price)
        real = [b for b in bets if b["ts"] >= (real_since or 0)]
        z = z_oos = None
        becomes_candidate = False
        if since is not None:
            status = "candidate"
            for c in checkpoints(len(oos_bets)):
                z_oos = price_z(won(oos_bets[:c]))
                if z_oos is not None and z_oos >= cfg.lab_gate_oos_z:
                    status = "proven"
                    break
                if z_oos is not None and z_oos <= (-1.0 if c == FIRST_CHECKPOINT else 0.0):
                    status = "failed"
                    break
        elif n < cfg.lab_min_bets:
            status = "collecting"
        else:
            reached = checkpoints(len(real))
            if reached:
                c = reached[-1]
                z = price_z(won(real[:c]))
                z_recent = price_z(won(real[c // 2:c]))
            if reached and z is not None and z >= cfg.lab_gate_z and (z_recent or 0) > 0:
                status, becomes_candidate = "candidate", True
            elif mean > 0:
                status = "promising"
            else:
                status = "losing"
        board.append({
            "asset": asset, "minutes": minutes, "variant": variant, "rule": describe(variant),
            "bets": n, "wins": sum(1 for x, _ in rows_ if x > 0), "paid": paid, "pnl": pnl,
            # With only a handful of bets a pessimistic figure means nothing; don't show one.
            "roi": mean, "roi_pessimistic": lower if n >= cfg.dash_min_bets else None,
            # The same distance above: what a lucky run of the same strategy could look like.
            "roi_optimistic": 2 * mean - lower if lower is not None and n >= cfg.dash_min_bets else None,
            "roi_recent": recent if n >= cfg.dash_min_bets else None, "status": status,
            "first_ts": bets[0]["ts"], "last_ts": bets[-1]["ts"],
            "candidate_since": since, "new_candidate": becomes_candidate,
            "oos_bets": len(oos), "oos_roi": oos_roi,
            "oos_pessimistic": oos_lower if len(oos) >= cfg.dash_min_bets else None,
            "z": z, "z_oos": z_oos, "real_bets": len(real),
            "auto": key in dynamic, "family": family(variant), "locked": locked,
            "next": next_stage(cfg, status, len(real), len(oos), bets[0]["ts"], bets[-1]["ts"]),
        })
    order = {"proven": 0, "candidate": 1, "promising": 2, "collecting": 3, "failed": 4, "losing": 5}
    board.sort(key=lambda b: (order[b["status"]], -(b["roi_pessimistic"] if b["roi_pessimistic"] is not None
                                                     else -1e9), -b["bets"]))

    # Top picks: proven variants on three different markets (variants of one market
    # are highly correlated, so three of them would be one bet, not three).
    picks, seen = [], set()
    for b in board:
        if b["status"] == "proven" and (b["asset"], b["minutes"]) not in seen:
            picks.append(b)
            seen.add((b["asset"], b["minutes"]))
        if len(picks) == 3:
            break
    leading = [b for b in board if b["status"] in ("candidate", "promising")][:5]

    markets = {}
    for r in db.execute(
        "SELECT asset, minutes, COUNT(*) AS windows, SUM(winner IS NOT NULL) AS resolved FROM windows"
        " GROUP BY asset, minutes"
    ).fetchall():
        markets[(r["asset"], r["minutes"])] = {"asset": r["asset"], "minutes": r["minutes"],
                                               "windows": r["windows"], "resolved": r["resolved"] or 0,
                                               "bets": 0, "best": None}
    for b in board:
        m = markets.setdefault((b["asset"], b["minutes"]), {"asset": b["asset"], "minutes": b["minutes"],
                                                            "windows": 0, "resolved": 0, "bets": 0, "best": None})
        m["bets"] += b["bets"]
        if m["best"] is None:
            m["best"] = {k: b[k] for k in ("variant", "rule", "status", "roi", "roi_pessimistic", "bets")}
    return {
        "board": board,
        "picks": picks,
        "candidates": leading,
        "stage_counts": {s: sum(1 for b in board if b["status"] == s) for s in order},
        "markets": sorted(markets.values(), key=lambda m: (m["asset"], m["minutes"])),
        "settled_bets": sum(len(bets) for bets in groups.values()),
        "active": len(groups),
        "pnl": sum(r["pnl"] for r in fresh_all),
        "paid": sum(r["paid"] for r in fresh_all),
        "per_round": per_round(fresh_all),
        "families": {f: _family_summary(f, board, fresh[f], order) for f in FAMILIES},
    }


def board_from(db_path, cfg, current, candidates, dynamic, since, real_since, mix_starts=None):
    """The leaderboard with the end-of-round test, read from the database file: what the engine runs
    in a process of its own. A pass over millions of trades takes tens of seconds of the processor,
    and in the server's own process it would hold up every page and click meanwhile. With it, the
    dashboard's count of the lab's strategies a period ({period: since}): another pass over them all,
    which the page used to make itself every half minute (seconds of waiting, each time)."""
    from . import stats
    from .store import connect
    db = connect(db_path)
    try:
        board = leaderboard(db, cfg, current, candidates, dynamic, since=since, real_since=real_since)
        board["end_pooled"] = end_pooled(db, real_since)
        board["mix"] = {p: [s, stats.strategy_mix(db, cfg, s, current)] for p, s in (mix_starts or {}).items()}
        return board
    finally:
        db.close()


def win_bounds(wins, n, z=1.645):
    """Exact (Clopper-Pearson) one-sided 95% bounds on a win rate: (lower, upper). At 97c and up the
    losses are few, so the normal approximation won't do."""
    if n <= 0:
        return None, None

    def tail_ge(k, p):  # P(X >= k), summed from the side with the fewest terms
        if k <= 0:
            return 1.0
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        lg = math.lgamma(n + 1)
        term = lambda i: math.exp(lg - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log(1 - p))
        if n - k < k:
            return min(1.0, sum(term(i) for i in range(k, n + 1)))
        return max(0.0, 1.0 - sum(term(i) for i in range(0, k)))

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if wins == 0 else solve(lambda p: tail_ge(wins, p) < 0.05)
    upper = 1.0 if wins == n else solve(lambda p: 1.0 - tail_ge(wins + 1, p) >= 0.05)
    return lower, upper


END_CHECKPOINTS = (200, 400, 800, 1600, 3200, 6400)


def end_pooled(db, real_since=None):
    """The end-of-round favourite, each rule over every coin together (one rule, one test): its
    trades on real fills, its win rate against what its prices needed to win (break-even: the
    average cost of a share fee included), the exact bounds, and the verdict at the checkpoints
    (proven: the lower bound over break-even; stopped: the upper bound under it); plus how many of
    its orders found the book still there (filled / tried)."""
    out = []
    rows = db.execute("SELECT variant, ts, status, pnl, cost + fee AS paid, shares FROM bets WHERE mode = 'lab'"
                      " AND variant LIKE 'end%' AND ts >= ? ORDER BY ts", (real_since or 0,)).fetchall()
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)
    for variant, rs in sorted(by.items()):
        tried = len(rs)
        done = [r for r in rs if r["status"] == "filled" and r["pnl"] is not None and r["shares"]]
        n, wins = len(done), sum(1 for r in done if r["pnl"] > 0)
        need = sum(r["paid"] / r["shares"] for r in done) / n if n else None
        lower, upper = win_bounds(wins, n)
        verdict = "collecting"
        for c in END_CHECKPOINTS:
            if n < c:
                break
            part = done[:c]
            w, b = sum(1 for r in part if r["pnl"] > 0), sum(r["paid"] / r["shares"] for r in part) / c
            lo_c, up_c = win_bounds(w, c)
            if lo_c > b:
                verdict = "proven"
                break
            if up_c < b:
                verdict = "stopped"
                break
            verdict = "undecided"
        out.append({"variant": variant, "rule": describe(variant), "tried": tried,
                    "filled": sum(1 for r in rs if r["status"] == "filled"), "trades": n, "wins": wins,
                    "win_rate": wins / n if n else None, "break_even": need, "lower": lower, "upper": upper,
                    "pnl": sum(r["pnl"] for r in done), "paid": sum(r["paid"] for r in done), "verdict": verdict,
                    "next": next((c for c in END_CHECKPOINTS if c > n), None)})
    return out


def _rounds(bets):
    """A both-sides rule's legs, one row per round (its two legs together), and how many rounds
    got both legs (locked: they make money whoever wins)."""
    by = {}
    for b in bets:
        r = by.setdefault(b["window_start"], {"ts": b["ts"], "pnl": 0.0, "paid": 0.0, "window_start": b["window_start"],
                                              "side": "pair", "sides": set()})
        r["pnl"] += b["pnl"]
        r["paid"] += b["paid"]
        r["sides"].add(b["side"])
    rounds = sorted(by.values(), key=lambda r: r["ts"])
    return rounds, sum(1 for r in rounds if len(r["sides"]) == 2)


def _family_summary(fam, board, money, order):
    """A lab page's figures: its strategies, trades and stages on every trade; its money (profit /
    loss, return) on its running rules' trades since the owner started over (`money`)."""
    mine = [b for b in board if b["family"] == fam]
    return {"active": len(mine), "settled_bets": sum(b["bets"] for b in mine),
            "locked": sum(b["locked"] or 0 for b in mine) if fam == "pair" else None,
            "pnl": sum(r["pnl"] for r in money), "paid": sum(r["paid"] for r in money),
            "stage_counts": {s: sum(1 for b in mine if b["status"] == s) for s in order},
            "per_round": per_round(money)}


def per_round(rows, stake=10.0):
    """The lab's result counted once per round and side: many variants buying the same round would
    each count the same opportunity (which, for real, only the first buyer gets). Each round-side's
    return per dollar (averaged over the variants that bought it) times one stake. A both-sides
    rule's two legs are one position (its "side" is the pair)."""
    pairs, rest = {}, []
    for r in rows:
        if "variant" not in r.keys() or family(r["variant"]) != "pair":
            rest.append(r)
            continue
        k = (r["asset"], r["minutes"], r["variant"], r["window_start"])
        m = pairs.setdefault(k, {"asset": r["asset"], "minutes": r["minutes"], "window_start": r["window_start"],
                                 "side": "pair", "pnl": 0.0, "paid": 0.0})
        m["pnl"] += r["pnl"]
        m["paid"] += r["paid"]
    # (profit, money in) per entry: a stake's worth at the entry's return, or a pair as it was bought
    # (a $10 pair is 10 pairs of shares, so it is already one stake when both legs are in; a round
    # left with only its cheap first leg put less in, and counts as what it was).
    rounds = {}
    for r in rest:
        if r["paid"] > 0:
            rounds.setdefault((r["asset"], r["minutes"], r["window_start"], r["side"]), []).append(
                (stake * r["pnl"] / r["paid"], stake))
    for r in pairs.values():
        if r["paid"] > 0:
            rounds.setdefault((r["asset"], r["minutes"], r["window_start"], "pair"), []).append((r["pnl"], r["paid"]))
    avg = [(sum(x for x, _ in v) / len(v), sum(p for _, p in v) / len(v)) for v in rounds.values()]
    pnl, paid = sum(x for x, _ in avg), sum(p for _, p in avg)
    return {"rounds": len(rounds), "pnl": pnl, "roi": pnl / paid if paid else None,
            "wins": sum(1 for x, _ in avg if x > 0)}
