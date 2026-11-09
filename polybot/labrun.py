"""The experiments lab's run over every recorded day (scalp.py: experiment 1; intraday.py: experiment 2), made to hold
years of minute bars (the history's backfill, backfill.py; the owner, 09.11.2026).

Each day's results are kept beside the day files (lab_cache/), under a key of the code's version and the days before
it that the day's rules read: a day is computed once, again only when either changes. The whole run (the new days, then
the board from all of them) goes in a process of its own at a low priority when there are many days, so the bots'
own work is never held behind it; only the board, the last day's trades and scan come back."""
import gzip
import hashlib
import json
import os

from . import intraday, scalp

INLINE_DAYS = 60  # fewer days than this: run here (a test's, the first month's)


def version():
    src = "".join(open(m.__file__, encoding="utf-8").read() for m in (scalp, intraday))
    return hashlib.sha1(src.encode()).hexdigest()[:16]


def _key(ver, d, hist, hist2):
    return hashlib.sha1(f"{ver}|{d}|{','.join(hist)}|{','.join(hist2)}".encode()).hexdigest()[:20]


def _read(path):
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def compute(folder, cache, ver, daily_atr=None, nr7=None, symbols=None, budget=None):
    """The days whose kept results are missing or stale: both experiments run on them, kept. The number computed."""
    bars = scalp.Bars(folder)
    days = bars.days()
    os.makedirs(cache, exist_ok=True)
    loaded, n = {}, 0
    load = lambda x: loaded.get(x) or loaded.setdefault(x, bars.load(x))
    for i, d in enumerate(days):
        if i < 5:
            continue
        hist = days[max(0, i - scalp.BASE_DAYS):i]
        hist2 = days[max(0, i - intraday.NOISE_DAYS):i]
        k = _key(ver, d, hist, hist2)
        path = os.path.join(cache, f"{d}.json.gz")
        old = _read(path) if os.path.exists(path) else None
        if old and old.get("key") == k:
            continue
        day = load(d)
        base = scalp.baselines([load(x) for x in hist])
        atr = (daily_atr or {}).get(d)
        e1, n_base = {}, None
        for v in ("V1",) + tuple(x for x in scalp.VARIANTS if x != "V1"):
            P = scalp.VARIANTS[v]
            if P.get("watch_only"):
                continue
            rows = scalp.run_day(day, base, P, atr, symbols if P["universe_only"] else None, seed=d,
                                 placebo_n=n_base, nr7=(nr7 or {}).get(d))
            if v == "V1":
                n_base = len(rows)
            e1[v] = [dict(r, day=d) for r in rows]
        et = lambda x: intraday._etfs(load(x), intraday.ETFS)
        e2 = {v: [dict(r, day=d) for sym in intraday.ETFS for r in intraday.run_day(et(d), et(days[i - 1]), [et(x) for x in hist2], P, sym)]
              for v, P in intraday.VARIANTS.items()}
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump({"key": k, "e1": e1, "e2": e2}, f, separators=(",", ":"), default=list)
        os.replace(tmp, path)
        n += 1
        keep = days[max(0, i - max(scalp.BASE_DAYS, intraday.NOISE_DAYS))]
        for x in list(loaded):
            if x < keep:
                loaded.pop(x)
        if budget and n >= budget:
            break
    return n


def assemble(folder, cache, ver):
    """Every kept day's results, as scalp.run_all and intraday.run_all give them: (res, run, res2, run2). A day not
    computed yet is left out."""
    days = scalp.Bars(folder).days()
    res, res2 = {k: [] for k in scalp.VARIANTS}, {k: [] for k in intraday.VARIANTS}
    run = []
    for i, d in enumerate(days):
        if i < 5:
            continue
        x = _read(os.path.join(cache, f"{d}.json.gz"))
        if not x or x.get("key") != _key(ver, d, days[max(0, i - scalp.BASE_DAYS):i], days[max(0, i - intraday.NOISE_DAYS):i]):
            continue
        for k, rows in x["e1"].items():
            res.setdefault(k, []).extend(rows)
        for k, rows in x["e2"].items():
            res2.setdefault(k, []).extend(rows)
        run.append(d)
    return res, run, res2, list(run)


def full(folder, cache, ver, daily_atr=None, nr7=None, symbols=None):
    """The whole run, as atlas.lab_run keeps it: the new days computed, then the boards and the last day's trades and
    scan. Small enough to come back from a process of its own."""
    try:
        os.nice(15)
    except (AttributeError, OSError):
        pass
    compute(folder, cache, ver, daily_atr, nr7, symbols)
    res, run, res2, run2 = assemble(folder, cache, ver)
    bars = scalp.Bars(folder)
    days = bars.days()
    scan, last = [], run[-1] if run else None
    if last:
        i = days.index(last)
        hist = [bars.load(x) for x in days[max(0, i - scalp.BASE_DAYS):i]]
        net = scalp.scan(bars.load(last), scalp.baselines(hist), scalp.VARIANTS["V1"])
        scan = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in net.values()]
    recent = sorted(([dict(t, variant=k) for k, ts in res.items() for t in ts if t["day"] == last] if last else []),
                    key=lambda t: (t["in"], t["variant"]))
    last2 = run2[-1] if run2 else None
    recent2 = sorted(([dict(t, variant=k) for k, ts in res2.items() for t in ts if t["day"] == last2] if last2 else []),
                     key=lambda t: (t["in"], t["variant"]))
    return {"board": scalp.board(res, run), "days": run, "recorded": days, "scan": scan, "last": last, "recent": recent[:200],
            "board2": intraday.board(res2, run2), "days2": run2, "last2": last2, "recent2": recent2[:100]}


def run(folder, cache, daily_atr=None, nr7=None, symbols=None):
    """full(), here for a few days, else in a process of its own (spawned: a clean interpreter, nothing of the bot's
    threads in it)."""
    ver = version()
    if len(scalp.Bars(folder).days()) < INLINE_DAYS:
        return full(folder, cache, ver, daily_atr, nr7, symbols)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(full, folder, cache, ver, daily_atr, nr7, symbols).result()
