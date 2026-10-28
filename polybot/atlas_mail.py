"""Atlas's mails: the same as Polymarket's (notify.py), for the stocks bot's trades.

- A mail when trades open (each with its card: the price, the stop, the target, what a win pays and what
  the stop costs, and the last two months' chart) and when they close (the result, large, in green or
  red, confetti on a win), each kind on its own, the simulation's and real money's apart.
- Every N trades of a kind, or every one; trades of one refresh go out together.
- Quiet hours: nothing is sent, and one mail with everything when they end.
- A daily summary after New York's close: the account, today, this week and month, the day's trades, the
  open ones, the market and what the brain learned.
The address and the mail server are Polymarket's (one owner, one inbox); the choices are Atlas's own.
"""

import datetime as dt
import html
import io
import re
import time
from zoneinfo import ZoneInfo

from . import equity
from .notify import BAD, GOOD, LINE, MUTED, ACCENT, _iso, _ltr, _money, confetti_gif

# Each mode its own (the owner, 28.10.2026: a tab for real money and one for the simulation): the trade mails,
# the daily summary and the quiet hours; the simulation's keys plain, real money's with "_live".
DEFAULTS = {"open": True, "close": True, "open_every": 1, "close_every": 1,
            "open_live": True, "close_live": True, "open_every_live": 1, "close_every_live": 1,
            "daily": True, "daily_time": "23:30", "daily_live": True, "daily_time_live": "23:30",
            "quiet": False, "quiet_from": "02:00", "quiet_to": "10:00",
            "quiet_live": False, "quiet_from_live": "02:00", "quiet_to_live": "10:00"}
SHARED_ONCE = ("daily", "daily_time", "quiet", "quiet_from", "quiet_to")  # (real money's start as the simulation's were)
MODES = (("sim", ""), ("live", "_live"))
QUIET_SECONDS = 30  # trades of one kind wait this long for more before their mail goes
WHY = {"target": "🎯 יעד", "stop": "🛑 סטופ", "trail": "📈 סטופ נגרר", "breakeven": "↔️ נקודת הכניסה", "time": "⏱ זמן",
       "open": "עדיין פתוח", "manual": "✋ ידני", "test": "🧪 בדיקה", "mentor": "🎓 המנטור יצא"}


def _real(mode):
    return " · כסף אמיתי" if mode == "live" else ""


def _pct(x, signed=False):
    return "–" if x is None else f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def shell(title, body, url, preheader=""):
    hidden = (f'<div style="display:none;max-height:0;overflow:hidden;opacity:0">{html.escape(preheader)}'
              f'{"&#847;&zwnj;&nbsp;" * 40}</div>' if preheader else "")
    link = (f'<p style="margin:24px 0 0"><a href="{html.escape(url)}" style="background:{ACCENT};color:#fff;'
            f'padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600">פתח את אטלס</a></p>' if url else "")
    return f"""<!doctype html><html lang="he" dir="rtl"><body style="margin:0;background:#f6f7f9;
font-family:Arial,Helvetica,sans-serif;color:#16181d">{hidden}<div dir="rtl" style="max-width:600px;margin:0 auto;padding:20px">
<div style="font-size:13px;color:{MUTED};margin-bottom:6px">🌐 אטלס · בוט שוק ההון</div>
<h1 style="font-size:20px;margin:0 0 16px">{title}</h1>{body}{link}
<p style="font-size:12px;color:{MUTED};margin-top:28px">את ההתראות של אטלס אפשר לשנות או לכבות באטלס ← הגדרות ← התראות במייל.</p>
</div></body></html>"""


def chart_png(p, trade, days=60):
    """The stock's last `days` closes as a PNG (mail apps show no SVG), with the entry (blue), the stop (red)
    and the target (green). No text, so no font is needed."""
    from PIL import Image, ImageDraw
    if not p or len(p["c"]) < 5:
        return None
    k = max(0, len(p["c"]) - days)
    closes, hi_, lo_ = p["c"][k:], p["h"][k:], p["l"][k:]
    lines = [(trade.get("entry"), ACCENT), (trade.get("stop") or trade.get("stop0"), BAD), (trade.get("target"), GOOD)]
    ys = hi_ + lo_ + [v for v, _ in lines if v]
    lo, hi = min(ys), max(ys)
    W, H, pad, s = 560, 150, 10, 2
    img = Image.new("RGB", (W * s, H * s), "white")
    d = ImageDraw.Draw(img)
    x = lambda i: (i + 0.5) / len(closes) * W * s
    y = lambda v: (pad + (hi - v) / ((hi - lo) or 1) * (H - 2 * pad)) * s
    for v, color in lines:
        if v:
            for x0 in range(0, W * s, 14 * s // 2):
                d.line([(x0, y(v)), (x0 + 4 * s, y(v))], fill=color, width=s)
    for i in range(len(closes)):
        d.line([(x(i), y(hi_[i])), (x(i), y(lo_[i]))], fill="#c9ced6", width=s)
    d.line([(x(i), y(c)) for i, c in enumerate(closes)], fill="#16181d", width=2 * s, joint="curve")
    img = img.resize((W, H), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def strategy_name(kind):
    return {"test": "בדיקת מערכת", "manual": "ידני", "mentor": "🎓 המנטור"}.get(kind) or (equity.STRATEGIES.get(kind) or {}).get("name", kind)


def card_html(t, account, cid=None):
    entry, stop, target, shares = t["entry"], t.get("stop") or t.get("stop0"), t.get("target"), t["shares"]
    win = (target - entry) * shares if target else None
    loss = (entry - stop) * shares
    row = lambda k, v, color=None: (f'<tr><td style="padding:7px 0;border-top:1px solid {LINE};color:{MUTED}">{k}</td>'
                                    f'<td style="padding:7px 0;border-top:1px solid {LINE};font-weight:700{";color:" + color if color else ""}">{v}</td></tr>')
    img = (f'<img src="cid:{cid[1:-1]}" width="560" style="width:100%;height:auto;display:block;margin:10px 0 4px" alt="גרף">'
           f'<div style="font-size:11px;color:{MUTED}">חודשיים אחרונים · כחול: כניסה · אדום: סטופ · ירוק: יעד</div>' if cid else "")
    return f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:16px;margin-bottom:14px">
<div style="display:flex;justify-content:space-between"><b style="font-size:22px">📈 {html.escape(t["symbol"])}</b></div>
<div style="margin:6px 0 8px"><span style="display:inline-block;background:#eef2ff;color:#3730a3;border-radius:999px;padding:3px 12px;font-size:13px;font-weight:700">🧭 {html.escape(strategy_name(t["strategy"]))}</span>
<span style="color:{MUTED};font-size:13px"> · נכנס ב-{html.escape(t.get("day") or t.get("in") or "")}</span></div>
<table width="100%" cellpadding="0" cellspacing="0" style="font-size:14px">
{row("אסטרטגיה", html.escape(strategy_name(t["strategy"])))}
{row("מחיר כניסה", _ltr(_money(entry)))}
{row("מניות · גודל הפוזיציה", f"{shares} · {_ltr(_money(entry * shares))}")}
{row("סטופ", f"{_ltr(_money(stop))} ({_ltr(_pct(stop / entry - 1, True))})", BAD)}
{row("יעד", f"{_ltr(_money(target))} ({_ltr(_pct(target / entry - 1, True))})" if target else "סטופ נגרר (בלי יעד קבוע)", GOOD)}
{row("אם הסטופ נפגע", f"{_ltr(_money(-loss, signed=True))} ({_ltr(_pct(loss / account))} מהתיק)", BAD)}
{row("אם היעד נפגע", _ltr(_money(win, signed=True)) if win is not None else "–", GOOD)}
{row("יחס סיכון / סיכוי", f"1:{(win / loss):.1f}" if win is not None and loss > 0 else "–")}
</table>{img}</div>"""


class AtlasMail:
    def __init__(self, notifier, atlas):
        self.n, self.atlas = notifier, atlas
        self.last = None  # the last trade mail: when, which kind, which mode, how many, sent
        self.daily_sent = None

    # ---------- settings (Atlas's own; the address and the server are Polymarket's) ----------
    def settings(self):
        stored = self.atlas._get("mail_settings", {}) or {}
        s = {**DEFAULTS, **stored}
        for k in SHARED_ONCE:  # (settings saved before each mode had its own: real money's start as they were)
            if k + "_live" not in stored and k in stored:
                s[k + "_live"] = stored[k]
        return s

    def update(self, changes):
        s = self.settings()
        for k, v in (changes or {}).items():
            if k not in DEFAULTS:
                continue
            if isinstance(DEFAULTS[k], bool):
                s[k] = bool(v)
            elif k.endswith("_every") or k.endswith("_every_live"):
                if not 1 <= int(v) <= 100:
                    raise ValueError("כל כמה טריידים: בין 1 ל-100")
                s[k] = int(v)
            else:
                if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(v)):
                    raise ValueError("שעה לא תקינה (למשל 23:30)")
                s[k] = str(v)
        self.atlas._set("mail_settings", s)
        return s

    def quiet_now(self, now, s=None, sfx=""):
        s = s or self.settings()
        if not s["quiet" + sfx]:
            return False
        local = dt.datetime.fromtimestamp(now, ZoneInfo(self.n.cfg.timezone))
        t = local.hour * 60 + local.minute
        a, b = (int(x[:2]) * 60 + int(x[3:]) for x in (s["quiet_from" + sfx], s["quiet_to" + sfx]))
        if a == b:
            return False  # (from a time to the same time: no quiet hours, never all day by mistake)
        return a <= t < b if a < b else (t >= a or t < b)

    def status(self):
        q, now = self.atlas.peek_mails(), time.time()
        s = self.settings()
        return dict(s, email=self.n.settings()["email"], configured=self.n.configured(),
                    waiting={"open": sum(1 for m in q if m["kind"] == "open"), "close": sum(1 for m in q if m["kind"] == "close")},
                    by_mode={mode: {"open": sum(1 for m in q if m["kind"] == "open" and m["mode"] == mode),
                                    "close": sum(1 for m in q if m["kind"] == "close" and m["mode"] == mode),
                                    "quiet_now": self.quiet_now(now, s, sfx),
                                    "daily_day": self.atlas._get("mail_daily_day" + sfx)} for mode, sfx in MODES},
                    last=self.last, last_error=self.n.last_error, daily_sent=self.daily_sent)

    # ---------- the loop (the notifier's, every few seconds) ----------
    def tick(self, now, can):
        s = self.settings()
        # A failure with IBKR is said at once, quiet hours or not (it may be costing real money).
        alerts = self.atlas._get("alert_mails", [])
        if alerts:
            self.atlas._set("alert_mails", [])
            if can:
                self.n._attempt(lambda: self.mail_alerts(alerts), "atlas alert")
        for mode, sfx in MODES:
            self._tick_mode(now, can, s, mode, sfx)

    def _tick_mode(self, now, can, s, mode, sfx):
        """One mode's mails, by its own settings and its own quiet hours: in them nothing of this mode goes
        (its trades wait, kept); when they end, one mail sums up what happened, then on as usual."""
        key = "mail_quiet_since" + sfx
        if self.quiet_now(now, s, sfx):
            if self.atlas._get(key) is None:
                self.atlas._set(key, now)
            return
        since = self.atlas._get(key)
        if since is not None:
            self.atlas._set(key, None)
            opened = self.atlas.take_mails("open", mode) if s["open" + sfx] else []
            closed = self.atlas.take_mails("close", mode) if s["close" + sfx] else []
            if (opened or closed) and can:
                ok, why = self.n._attempt(lambda: self.mail_night([m["trade"] for m in opened], [m["trade"] for m in closed], mode, since))
                self.atlas.record_mail([m["trade"] for m in opened], "open", ok, why)
                self.atlas.record_mail([m["trade"] for m in closed], "close", ok, why)
            elif opened or closed:  # (no mail server or address: kept as not sent, with why)
                self.atlas.record_mail([m["trade"] for m in opened], "open", False, "המייל לא מוגדר (כתובת או שרת מייל)")
                self.atlas.record_mail([m["trade"] for m in closed], "close", False, "המייל לא מוגדר (כתובת או שרת מייל)")
        q = self.atlas.peek_mails()
        for kind, mail in (("open", self.mail_opened), ("close", self.mail_closed)):
            mine = [m for m in q if m["kind"] == kind and m["mode"] == mode]
            if not mine:
                continue
            enabled = s[kind + sfx] and can
            # Every N trades (or every one), once nothing new came for QUIET_SECONDS.
            if enabled and (len(mine) < s[f"{kind}_every{sfx}"] or now - max(m["ts"] for m in mine) < QUIET_SECONDS):
                continue
            rows = [m["trade"] for m in self.atlas.take_mails(kind, mode)]
            if enabled:
                ok, why = self.n._attempt(lambda: mail(rows, mode), f"atlas {kind}/{mode}")
                self.last = {"ts": now, "kind": kind, "mode": mode, "trades": len(rows), "sent": ok}
                self.atlas.record_mail(rows, kind, ok, why)
            else:
                self.atlas.record_mail(rows, kind, False, "המייל לא מוגדר (כתובת או שרת מייל)" if not can else
                                       f"מייל {'הפתיחה' if kind == 'open' else 'הסגירה'} כבוי בהגדרות של אטלס")
        if s["daily" + sfx] and can:
            self.maybe_daily(now, s, mode)

    # ---------- the mails ----------
    def mail_alerts(self, alerts):
        tz = ZoneInfo("America/New_York")
        rows = "".join(
            f'<tr><td style="padding:8px 0;border-top:1px solid {LINE};white-space:nowrap">{dt.datetime.fromtimestamp(a["ts"], tz):%H:%M} NY</td>'
            f'<td style="padding:8px 0;border-top:1px solid {LINE}"><b>{html.escape(a["where"])}</b>{" · " + html.escape(a["symbol"]) if a.get("symbol") else ""}'
            f'<br><span style="color:{MUTED};font-size:12px" dir="ltr">{html.escape(a["what"])}</span></td></tr>' for a in alerts)
        label = "כסף אמיתי" if any(a.get("mode") == "live" for a in alerts) else "Paper"
        body = f"""<div style="background:#fff;border:1px solid {BAD};border-radius:14px;padding:18px">
<div style="font-size:18px;font-weight:800;color:{BAD}">⚠️ תקלה במסחר באינטראקטיב · {label}</div>
<div style="color:{MUTED};font-size:13px;margin:6px 0 10px">איפה זה נכשל, ומה אינטראקטיב או IB Gateway החזירו. אותו פירוט באטלס, ב-Hub למעלה.</div>
<table width="100%" cellpadding="0" cellspacing="0" style="font-size:14px">{rows}</table></div>"""
        first = alerts[0]
        subject = f"⚠️ אטלס: {first['where']}{' · ' + first['symbol'] if first.get('symbol') else ''}" + (f" (+{len(alerts) - 1})" if len(alerts) > 1 else "")
        self.n._safe_send(subject, shell("תקלה באינטראקטיב", body, self.n.url, first["what"][:120]))

    def _prepared(self):
        with self.atlas.eq.lock:
            return self.atlas.eq.prepared

    def mail_opened(self, trades, mode):
        account = self.atlas.settings()["account"]
        prepared = self._prepared()
        parts, images = [], []
        for t in trades:
            png = chart_png(prepared.get(t["symbol"]), t)
            cid = None
            if png:
                from email.utils import make_msgid
                cid = make_msgid(domain="atlas")
                images.append((cid, png))
            parts.append(card_html(t, account, cid))
        win = sum((t["target"] - t["entry"]) * t["shares"] for t in trades if t.get("target"))
        loss = sum((t["entry"] - (t.get("stop") or t["stop0"])) * t["shares"] for t in trades)
        odds = f"{_iso(_money(win, signed=True))} / {_iso(_money(-loss, signed=True))}"
        label = "סימולציה" if mode == "sim" else "כסף אמיתי"
        if len(trades) == 1:
            t = trades[0]
            subject = f"📈 {t['symbol']} · {strategy_name(t['strategy'])} || {odds}{_real(mode)}"
        else:
            subject = f"📊 {len(trades)} עסקאות נפתחו · {', '.join(t['symbol'] for t in trades[:5])} || {odds}{_real(mode)}"
        preheader = " · ".join(f"{t['symbol']} ב-{_money(t['entry'])}, {_money((t['entry'] - (t.get('stop') or t['stop0'])) * t['shares'])} בסיכון"
                               for t in trades) + f" · {label}"
        title = "נפתח טרייד" if len(trades) == 1 else f"נפתחו {len(trades)} טריידים"
        self.n._safe_send(subject, shell(f"{title} · {label}", "".join(parts), self.n.url, preheader), images)

    def mail_closed(self, trades, mode):
        total = sum(t["pnl"] for t in trades)
        color = GOOD if total >= 0 else BAD
        label = "סימולציה" if mode == "sim" else "כסף אמיתי"
        td = f'style="padding:8px 0;border-top:1px solid {LINE}"'
        rows = "".join(
            f'<tr><td {td}><b>{html.escape(t["symbol"])}</b> · {html.escape(strategy_name(t["strategy"]))}<br>'
            f'<span style="color:{MUTED};font-size:12px">{_ltr(_money(t["entry"]))} ← {_ltr(_money(t["exit"]))} · {t.get("days", "–")} ימים</span></td>'
            f'<td {td}>{WHY.get(t.get("why"), t.get("why") or "")}</td>'
            f'<td style="padding:8px 0;border-top:1px solid {LINE};font-weight:700;color:{GOOD if t["pnl"] >= 0 else BAD}">'
            f'{_ltr(_money(t["pnl"], signed=True))}<br><span style="font-size:12px;font-weight:400">{_ltr(f"{t["r"]:+.2f}R") if t.get("r") is not None else ""}</span></td></tr>'
            for t in trades)
        body = f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;text-align:center">
<div style="color:{MUTED};font-size:14px">{"רווח" if total >= 0 else "הפסד"} · {len(trades)} {"טרייד" if len(trades) == 1 else "טריידים"} · {label}</div>
<div style="font-size:44px;font-weight:800;color:{color};margin:6px 0">{_ltr(_money(total, signed=True))}</div></div>
<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px;font-size:14px">{rows}</table>"""
        mark = "✅" if total >= 0 else "❌"
        subject = (f"{mark} {trades[0]['symbol']} {_iso(_money(total, signed=True))}" if len(trades) == 1
                   else f"{mark} {len(trades)} עסקאות נסגרו · {_iso(_money(total, signed=True))}") + _real(mode)
        preheader = " · ".join(f"{t['symbol']} {_money(t['pnl'], signed=True)}" for t in trades) + f" · {label}"
        images = []
        if total > 0:
            gif = confetti_gif()
            if gif:
                from email.utils import make_msgid
                cid = make_msgid(domain="atlas")
                images.append((cid, gif, "gif"))
                body = (f'<img src="cid:{cid[1:-1]}" width="560" style="width:100%;height:auto;display:block;'
                        f'border-radius:14px;margin-bottom:12px" alt="🎉">') + body
        self.n._safe_send(subject, shell("טרייד נסגר" if len(trades) == 1 else f"נסגרו {len(trades)} טריידים", body,
                                         self.n.url, preheader), images)

    def mail_night(self, opened, closed, mode, since):
        label = "סימולציה" if mode == "sim" else "כסף אמיתי"
        total = sum(t["pnl"] for t in closed)
        td = f'style="padding:7px 4px;border-top:1px solid {LINE}"'
        closed_rows = "".join(f'<tr><td {td}><b>{html.escape(t["symbol"])}</b> · {html.escape(strategy_name(t["strategy"]))}</td><td {td}>{WHY.get(t.get("why"), "")}</td>'
                              f'<td {td}><b style="color:{GOOD if t["pnl"] > 0 else BAD}">{_ltr(_money(t["pnl"], signed=True))}</b></td></tr>' for t in closed)
        opened_rows = "".join(f'<tr><td {td}><b>{html.escape(t["symbol"])}</b> · {html.escape(strategy_name(t["strategy"]))}</td>'
                              f'<td {td}>{_ltr(_money(t["entry"]))}</td><td {td}>סטופ {_ltr(_money(t.get("stop") or t.get("stop0")))}</td></tr>' for t in opened)
        body = f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;text-align:center">
<div style="color:{MUTED};font-size:14px">בשעות השקטות · {label}</div>
<div style="font-size:44px;font-weight:800;color:{GOOD if total >= 0 else BAD};margin:6px 0">{_ltr(_money(total, signed=True))}</div>
<div style="font-size:15px">{len(opened)} טריידים נפתחו · {len(closed)} נסגרו</div></div>
{f'<h2 style="font-size:16px;margin:20px 0 8px">נסגרו</h2><table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">{closed_rows}</table>' if closed else ""}
{f'<h2 style="font-size:16px;margin:20px 0 8px">נפתחו</h2><table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">{opened_rows}</table>' if opened else ""}"""
        subject = (f"🌙 אטלס בשעות השקטות: {len(opened)} נפתחו · {len(closed)} נסגרו"
                   + (f" · {_iso(_money(total, signed=True))}" if closed else "") + _real(mode))
        self.n._safe_send(subject, shell("מה קרה בשעות השקטות", body, self.n.url,
                                         f"{len(opened)} נפתחו, {len(closed)} נסגרו, {_money(total, signed=True)}"))

    def maybe_daily(self, now, s=None, mode="sim"):
        """The day's summary of one mode, at its time. Kept by the trading day it sums up (Monday to Friday),
        not by the date it goes out: a summary whose time falls in the quiet hours goes when they end, even
        the next morning, and a weekend sends none."""
        s = s or self.settings()
        sfx = "_live" if mode == "live" else ""
        local = dt.datetime.fromtimestamp(now, ZoneInfo(self.n.cfg.timezone))
        hh, mm = (int(x) for x in s["daily_time" + sfx].split(":"))
        due = local.date() if (local.hour, local.minute) >= (hh, mm) else local.date() - dt.timedelta(days=1)
        while due.weekday() >= 5:
            due -= dt.timedelta(days=1)
        key = "mail_daily_day" + sfx
        done = self.atlas._get(key)
        if done is not None and done >= due.isoformat():
            return
        if done is None:  # (the first time: from today on, never a summary of a day before it was on)
            self.atlas._set(key, due.isoformat())
            return
        if mode == "live" and not self.atlas.live_book(now):
            return  # (no real money: no real money's summary)
        self.atlas._set(key, due.isoformat())
        self.n._safe_send(*self.daily_mail(now, mode))
        self.daily_sent = now

    def daily_mail(self, now, mode="sim"):
        if mode == "live":
            return self.daily_mail_live(now)
        A = self.atlas
        v = A.paper_view() or {}
        account = A.settings()["account"]
        prepared = self._prepared()
        pic = equity.picture(prepared) if prepared.get("SPY") else None
        day = (v.get("last") or "")
        closed = [t for t in v.get("trades", []) if t.get("out") == day]
        opened = [t for t in v.get("open", []) if t.get("day") == day] + [t for t in closed if t.get("in") == day]
        today = v.get("today") or 0.0
        color = lambda x: GOOD if (x or 0) >= 0 else BAD
        td = f'style="padding:7px 4px;border-top:1px solid {LINE}"'
        trades = "".join(
            f'<tr><td {td}><b>{html.escape(t["symbol"])}</b> · {html.escape(strategy_name(t["strategy"]))}</td><td {td}>{"נסגר: " + WHY.get(t.get("why"), "") if t in closed else "נפתח"}</td>'
            f'<td style="padding:7px 4px;border-top:1px solid {LINE};font-weight:700;color:{color(t.get("pnl"))}">{_ltr(_money(t.get("pnl"), signed=True)) if t.get("pnl") is not None else ""}</td></tr>'
            for t in closed + [t for t in opened if t not in closed])
        opens = "".join(
            f'<tr><td {td}><b>{html.escape(t["symbol"])}</b></td><td {td}>{_ltr(_money(t["entry"]))} → עכשיו {_ltr(_money(t["price"]))}</td>'
            f'<td style="padding:7px 4px;border-top:1px solid {LINE};font-weight:700;color:{color(t["pnl"])}">{_ltr(_money(t["pnl"], signed=True))}</td></tr>'
            for t in v.get("open", []))
        learned = sorted((h for b in A.brain().values() for h in (b.get("history") or []) if h["kept"] and h["ts"] > now - 86400),
                         key=lambda h: -h["ts"])
        verdict = {"risk_on": "🟢 השוק בעד", "mixed": "🟡 מעורב", "risk_off": "🔴 השוק נגד"}.get((pic or {}).get("verdict"), "–")
        body = f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;text-align:center">
<div style="color:{MUTED};font-size:14px">רווח / הפסד היום · סימולציה</div>
<div style="font-size:44px;font-weight:800;color:{color(today)};margin:6px 0">{_ltr(_money(today, signed=True))}</div>
<div style="font-size:15px">שווי התיק {_ltr(_money(v.get("value")))} · השבוע <b style="color:{color(v.get("week"))}">{_ltr(_money(v.get("week"), signed=True))}</b>
 · החודש <b style="color:{color(v.get("month"))}">{_ltr(_money(v.get("month"), signed=True))}</b></div>
<div style="font-size:13px;color:{MUTED};margin-top:6px">מההתחלה {_ltr(_money(v.get("total"), signed=True))} · בסיכון עכשיו {_ltr(_money(v.get("open_risk")))} ({_ltr(_pct((v.get("open_risk") or 0) / account))} מהתיק)</div></div>
<p style="font-size:14px;margin:14px 0 0">השוק: <b>{verdict}</b>{f" · S&P {_ltr(_pct(pic.get('spy_5d'), True))} בחמישה ימים" if pic else ""}</p>
<h2 style="font-size:16px;margin:22px 0 8px">הטריידים של היום</h2>
{f'<table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">{trades}</table>' if trades else f'<p style="color:{MUTED};font-size:14px">אין טריידים היום.</p>'}
<h2 style="font-size:16px;margin:22px 0 8px">פתוחים עכשיו ({len(v.get("open", []))})</h2>
{f'<table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">{opens}</table>' if opens else f'<p style="color:{MUTED};font-size:14px">אין.</p>'}
<h2 style="font-size:16px;margin:22px 0 8px">מה המוח למד היום</h2>
<ul style="margin:0;padding-inline-start:20px;font-size:14px;line-height:1.7">{"".join(f"<li>{html.escape(strategy_name(h['strategy']))}: {html.escape(', '.join(f'{k} {x}' for k, x in h['change'].items()))}</li>" for h in learned[:8]) or "<li>לא נשמר שינוי היום</li>"}</ul>"""
        date = dt.datetime.fromtimestamp(now, ZoneInfo(self.n.cfg.timezone)).strftime("%d.%m")
        return (f"📊 אטלס · סיכום היום: {_iso(_money(today, signed=True))}",
                shell(f"סיכום יומי · {date}", body, self.n.url, f"שווי התיק {_money(v.get('value'))} · החודש {_money(v.get('month'), signed=True)}"))

    def daily_mail_live(self, now):
        """The real account's day (IBKR's figures): its value, today, the week and the month, the trades
        that went in and out today, and the positions held now."""
        L = self.atlas.live_book(now) or {}
        color = lambda x: GOOD if (x or 0) >= 0 else BAD
        td = f'style="padding:7px 4px;border-top:1px solid {LINE}"'
        if L.get("never"):
            body = f'<p style="font-size:15px">אין עדיין קריאה של החשבון האמיתי: IB Gateway לא ענה{" (" + html.escape(L["error"]) + ")" if L.get("error") else ""}.</p>'
            return ("💵 אטלס · סיכום היום: אין חיבור לאינטראקטיב", shell("סיכום יומי · כסף אמיתי", body, self.n.url, "אין חיבור לאינטראקטיב"))
        B = L.get("atlas") or {"positions": [], "today": None, "week": None, "month": None, "open_risk": 0.0, "value": 0.0}
        today = B.get("today")  # (Atlas' own book: the owner's own positions in the account aren't in its summary)
        closed = "".join(f'<tr><td {td}><b>{html.escape(t["symbol"])}</b> · {html.escape(strategy_name(t["strategy"]))}</td><td {td}>{WHY.get(t.get("why"), "") or ("ידני" if t.get("why") == "manual" else "")}</td>'
                         f'<td style="padding:7px 4px;border-top:1px solid {LINE};font-weight:700;color:{color(t.get("pnl"))}">{_ltr(_money(t.get("pnl"), signed=True)) if t.get("pnl") is not None else ""}</td></tr>'
                         for t in L.get("closed", []))
        opens = "".join(f'<tr><td {td}><b>{html.escape(p["symbol"])}</b> · {p["shares"]:g} מניות</td><td {td}>{_ltr(_money(p["avg_cost"]))} → עכשיו {_ltr(_money(p["price"]))}</td>'
                        f'<td style="padding:7px 4px;border-top:1px solid {LINE};font-weight:700;color:{color(p["unrealized"])}">{_ltr(_money(p["unrealized"], signed=True))}</td></tr>'
                        for p in B.get("positions", []))
        stale = "" if L.get("connected") else f'<p style="color:{BAD};font-size:13px">⚠️ אין חיבור לאינטראקטיב כרגע: המספרים מהקריאה האחרונה.</p>'
        body = f"""{stale}<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;text-align:center">
<div style="color:{MUTED};font-size:14px">רווח / הפסד היום · הטריידים של אטלס בכסף אמיתי ({html.escape(str(L.get("account") or ""))})</div>
<div style="font-size:44px;font-weight:800;color:{color(today)};margin:6px 0">{_ltr(_money(today, signed=True)) if today is not None else "–"}</div>
<div style="font-size:15px">אטלס בתיק {_ltr(_money(B.get("value")))} · השבוע <b style="color:{color(B.get("week"))}">{_ltr(_money(B.get("week"), signed=True)) if B.get("week") is not None else "–"}</b>
 · החודש <b style="color:{color(B.get("month"))}">{_ltr(_money(B.get("month"), signed=True)) if B.get("month") is not None else "–"}</b></div>
<div style="font-size:13px;color:{MUTED};margin-top:6px">בסיכון עכשיו (של אטלס) {_ltr(_money(B.get("open_risk")))} · החשבון כולו {_ltr(_money(L.get("value")))}, מזומן {_ltr(_money(L.get("cash")))}</div></div>
<h2 style="font-size:16px;margin:22px 0 8px">נסגרו היום</h2>
{f'<table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">{closed}</table>' if closed else f'<p style="color:{MUTED};font-size:14px">אין.</p>'}
<h2 style="font-size:16px;margin:22px 0 8px">הפוזיציות של אטלס ({len(B.get("positions", []))})</h2>
{f'<table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">{opens}</table>' if opens else f'<p style="color:{MUTED};font-size:14px">אין.</p>'}"""
        date = dt.datetime.fromtimestamp(now, ZoneInfo(self.n.cfg.timezone)).strftime("%d.%m")
        return (f"💵 אטלס · סיכום היום בכסף אמיתי: {_iso(_money(today, signed=True)) if today is not None else '–'}",
                shell(f"סיכום יומי · כסף אמיתי · {date}", body, self.n.url, f"שווי החשבון {_money(L.get('value'))}"))

    def send_test(self):
        self.n.send("בדיקה: ההתראות של אטלס עובדות",
                    shell("ההתראות עובדות ✓", "<p>מעכשיו יגיעו לכאן מיילים על כל טרייד של אטלס שנפתח ונסגר, וסיכום יומי.</p>", self.n.url))

    def send_daily_now(self, mode="sim"):
        if not (self.n.configured() and self.n.settings()["email"]):
            raise ValueError("אין כתובת מייל או ששרת המייל לא מוגדר")
        if mode == "live" and not self.atlas.live_book(time.time()):
            raise ValueError("אין מסחר בכסף אמיתי פעיל באטלס: אין סיכום של החשבון האמיתי")
        self.n.send(*self.daily_mail(time.time(), mode))
        self.daily_sent = time.time()

    def resend(self, trade_id, kind):
        if kind not in ("open", "close"):
            raise ValueError("bad request")
        if not (self.n.configured() and self.n.settings()["email"]):
            raise ValueError("המייל לא מוגדר (כתובת או שרת מייל)")
        book = self.atlas.paper_book() or {}
        t = next((x for x in (book.get("trades") or []) + (book.get("open") or []) if x.get("id") == trade_id), None)
        if t is None:
            raise ValueError("אין עסקה כזאת")
        if kind == "close" and t in (book.get("open") or []):
            raise ValueError("העסקה עוד פתוחה: מייל הסגירה יוצא כשהיא נסגרת")
        t = {**t, "day": t.get("day") or t.get("in")}
        ok, why = self.n._attempt(lambda: (self.mail_opened if kind == "open" else self.mail_closed)([t], "sim"), f"atlas resend {kind}")
        self.atlas.record_mail([t], kind, ok, why)
        if not ok:
            raise ValueError("לא נשלח: " + str(why))
        return {"ok": True}
