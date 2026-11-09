"""The server's start, stage by stage (the owner, 09.11.2026: "I refreshed and it's stuck on that picture: put the progress
bar here too, that fills as it goes, and keeps moving while the system isn't stuck"). Each stage of building the bot says
its name when it begins; the "updating" page and the dashboards show the stage, how long it has run, and a bar that fills
by the stages done and creeps on within one. Each stage's time is written to the log, so a start that hangs says where."""
import logging
import math
import time

log = logging.getLogger("polybot")

# The stages, in order (Engine.__init__, then main()): their number is the bar's whole.
NAMES = {
    "db": "פותח את מסד הנתונים",
    "results": "מסדר את התוצאות של הסבבים",
    "markets": "בונה את השווקים והאסטרטגיות",
    "lab": "טוען את המעבדה",
    "settings": "טוען את ההגדרות והמחקרים",
    "atlas": "טוען את אטלס (מניות)",
    "money": "מכין את הכסף והכללים",
    "start": "מפעיל את המסחר",
    "web": "פותח את האתר",
}
ORDER = list(NAMES)
T0 = time.time()
_now = {"key": None, "since": T0, "done": []}


def stage(key):
    """A stage begins (the one before is done: its time to the log)."""
    t = time.time()
    if _now["key"]:
        log.warning("Start: %s took %.1f s", _now["key"], t - _now["since"])
        _now["done"].append(_now["key"])
    _now.update(key=key, since=t)


def view(now=None):
    """{"stage", "name", "index", "of", "elapsed", "in_stage", "pct", "done"} for the pages."""
    now = now or time.time()
    key = _now["key"]
    i = ORDER.index(key) if key in ORDER else 0
    within = now - _now["since"]
    # (the bar: the stages done, and within the one running a creep toward its end that never reaches it, as a wait feels)
    pct = 100 * (i + 0.9 * (1 - math.exp(-within / 25))) / len(ORDER)
    return {"stage": key, "name": NAMES.get(key, "מתחיל"), "index": i + 1, "of": len(ORDER), "elapsed": round(now - T0),
            "in_stage": round(within), "pct": round(min(99.0, pct), 1), "done": False}
