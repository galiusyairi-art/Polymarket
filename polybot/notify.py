"""Email notifications for the main strategy's trades: one when trades open (each with its
market card and price chart), one when they close (the result, large, in green or red), and
a daily summary (trades today, P&L today and this month, Polymarket balance).

Mail goes out over SMTP. Gmail with an app password works (smtp.gmail.com, up to ~500 mails a
day), as does any provider's SMTP. Credentials are environment variables on the server; the
address to send to and which mails to send are set on the dashboard.

Only the main strategy is mailed, never the lab (it opens hundreds of paper bets an hour):
simulated trades while real trading is off, real ones once it is on. Trades opened or closed
while a mail type is switched off are marked as handled, so switching it on never floods.
"""
import datetime as dt
import html
import io
import logging
import os
import re
import smtplib
import threading
import time
from email.message import EmailMessage
from email.utils import make_msgid
from zoneinfo import ZoneInfo

from .periods import period_starts

log = logging.getLogger("polybot")

OPENED, CLOSED = 1, 2  # bits in bets.notified
NAMES = {"btc": "ביטקוין", "eth": "את'ריום", "sol": "סולנה", "xrp": "ריפל", "doge": "דוג'קוין",
         "bnb": "בינאנס", "hype": "הייפ"}
GOOD, BAD, ACCENT, MUTED, LINE, TEXT = "#1f8a4c", "#c92a2a", "#3b5bdb", "#646b78", "#e3e6eb", "#16181d"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _money(x, digits=2, signed=False):
    if x is None:
        return "–"
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}${abs(x):,.{digits}f}"


def _cents(x):
    return "–" if x is None else f"{x * 100:.0f}¢"


def _ltr(s):
    # Keep numbers and signs in order inside right-to-left text.
    return f'<span dir="ltr" style="unicode-bidi:isolate">{html.escape(s)}</span>'


def chart_png(card):
    """The window's price path as a PNG (email clients don't show SVG): the price line, the
    dashed price to beat, and a dot where each bet entered. No text, so no font is needed."""
    from PIL import Image, ImageDraw

    pts, base, length = card.get("series") or [], card.get("open_price"), card["minutes"] * 60
    if len(pts) < 2 or base is None:
        return None
    W, H, pad, scale = 560, 140, 10, 2
    img = Image.new("RGB", (W * scale, H * scale), "white")
    d = ImageDraw.Draw(img)
    ys = [p for _, p in pts] + [base]
    lo, hi = min(ys), max(ys)
    if hi - lo < base * 0.0002:
        mid, half = (hi + lo) / 2, base * 0.0001
        lo, hi = mid - half, mid + half
    x = lambda t: max(0.0, min(length, t)) / length * W * scale
    y = lambda v: (pad + (1 - (v - lo) / (hi - lo)) * (H - 2 * pad)) * scale
    by = y(base)
    for x0 in range(0, W * scale, 14 * scale // 2):
        d.line([(x0, by), (x0 + 4 * scale, by)], fill=MUTED, width=scale)
    d.line([(x(t), y(p)) for t, p in pts], fill=ACCENT, width=2 * scale, joint="curve")

    def at(t):
        best = pts[0][1]
        for ts, p in pts:
            if ts <= t:
                best = p
        return best

    for b in card["bets"]:
        t = b["ts"] - card["window_start"]
        cx, cy, r = x(t), y(at(t)), 5 * scale
        d.ellipse([cx - r - 2 * scale, cy - r - 2 * scale, cx + r + 2 * scale, cy + r + 2 * scale], fill="white")
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GOOD if b["side"] == "up" else BAD)
    img = img.resize((W, H), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def _hhmm(ts, tz):
    return dt.datetime.fromtimestamp(ts, ZoneInfo(tz)).strftime("%H:%M")


def _shell(title, body, url):
    link = (f'<p style="margin:24px 0 0"><a href="{html.escape(url)}" style="background:{ACCENT};color:#fff;'
            f'padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600">פתח את הדשבורד</a></p>'
            if url else "")
    return f"""<!doctype html><html lang="he" dir="rtl"><body style="margin:0;background:#f6f7f9;
font-family:Arial,Helvetica,sans-serif;color:{TEXT}"><div dir="rtl" style="max-width:600px;margin:0 auto;padding:20px">
<div style="font-size:13px;color:{MUTED};margin-bottom:6px">בוט פולימרקט</div>
<h1 style="font-size:20px;margin:0 0 16px">{title}</h1>{body}{link}
<p style="font-size:12px;color:{MUTED};margin-top:28px">את ההתראות אפשר לשנות או לכבות בדשבורד, באזור "התראות במייל".</p>
</div></body></html>"""


def card_html(card, bet, tz, chart_cid=None):
    name = NAMES.get(card["asset"], card["asset"].upper())
    digits = 4 if (card.get("open_price") or 100) < 10 else 2
    diff = None if card.get("price") is None or card.get("open_price") is None else card["price"] - card["open_price"]

    def button(side, label, ask):
        on = bet["side"] == side
        color = GOOD if side == "up" else BAD
        style = (f"background:{color};color:#fff;border:1px solid {color}" if on
                 else f"background:#fff;color:{MUTED};border:1px solid {LINE}")
        return (f'<td width="50%" style="padding:4px"><div style="{style};border-radius:10px;padding:12px;'
                f'text-align:center;font-weight:700;font-size:16px">{label} {_ltr(_cents(ask))}</div></td>')

    side = "עלייה" if bet["side"] == "up" else "ירידה"
    chart = (f'<img src="cid:{chart_cid[1:-1]}" width="560" style="width:100%;height:auto;display:block;'
             f'margin:8px 0" alt="מסלול המחיר בחלון">' if chart_cid else "")
    move = ""
    if diff is not None:
        color = GOOD if diff >= 0 else BAD
        move = f' <span style="color:{color};font-weight:600">({"▲" if diff >= 0 else "▼"} {_ltr(_money(abs(diff), digits))})</span>'
    return f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:16px;margin-bottom:14px">
<div style="font-weight:700;font-size:17px">{name}: עלייה או ירידה</div>
<div style="color:{MUTED};font-size:13px;margin-bottom:10px">{card['minutes']} דקות · {_ltr(_hhmm(card['window_start'], tz) + '–' + _hhmm(card['end'], tz))}
 · נכנסנו {_ltr(f"{int(bet['seconds_left'] // 60)}:{int(bet['seconds_left'] % 60):02d}")} לפני הסגירה</div>
<table width="100%" cellpadding="0" cellspacing="0"><tr>
<td><div style="color:{MUTED};font-size:12px">מחיר לניצחון</div><div style="font-weight:700">{_ltr(_money(card.get('open_price'), digits))}</div></td>
<td><div style="color:{MUTED};font-size:12px">מחיר נוכחי</div><div style="font-weight:700">{_ltr(_money(card.get('price'), digits))}{move}</div></td>
</tr></table>{chart}
<table width="100%" cellpadding="0" cellspacing="0"><tr>{button("up", "עלייה", card.get("up_ask"))}{button("down", "ירידה", card.get("down_ask"))}</tr></table>
<div style="font-size:14px;margin-top:10px;line-height:1.7">נכנסנו ל<b>{side}</b> ב-{_ltr(_cents(bet['avg_price']))} ·
 {_ltr(_money(bet['paid']))} · {_ltr(f"{bet['shares']:.1f}")} מניות · הסיכוי שלנו {_ltr(f"{bet['fair_prob'] * 100:.0f}%")}<br>
אם ננצח: <b style="color:{GOOD}">{_ltr(_money(bet['shares'] - bet['paid'], signed=True))}</b> · אם נפסיד: <b style="color:{BAD}">{_ltr(_money(-bet['paid'], signed=True))}</b></div>
</div>"""


class Notifier:
    def __init__(self, cfg, store, engine, env=None):
        env = os.environ if env is None else env
        self.cfg, self.store, self.engine = cfg, store, engine
        self.user = env.get("SMTP_USER", "").strip()
        self.host = env.get("SMTP_HOST", "").strip() or ("smtp.gmail.com" if self.user.endswith("@gmail.com") else "")
        self.port = int(env.get("SMTP_PORT", "587"))
        # Gmail shows app passwords in groups of four; the spaces aren't part of it.
        self.password = env.get("SMTP_PASSWORD", "").replace(" ", "")
        self.sender = env.get("SMTP_FROM", "").strip() or self.user
        self.url = env.get("POLYBOT_URL") or env.get("RENDER_EXTERNAL_URL") or ""
        self.last_error = None
        self.last_sent = None
        self.lock = threading.Lock()
        if self.store.get_setting("notify_initialized") is None:
            # First start: everything already in the database counts as handled.
            self.store._x("UPDATE bets SET notified = 3")
            self.store.set_setting("notify_initialized", "1")

    # Settings
    def configured(self):
        return bool(self.host and self.user and self.password)

    def settings(self):
        get = self.store.get_setting
        return {"email": get("notify_email", ""), "open": get("notify_open", "1") == "1",
                "close": get("notify_close", "1") == "1", "daily": get("notify_daily", "1") == "1"}

    def update(self, email=None, open=None, close=None, daily=None):
        if email is not None:
            email = email.strip()
            if email and not EMAIL_RE.match(email):
                raise ValueError("כתובת המייל לא תקינה")
            self.store.set_setting("notify_email", email)
        for key, value in (("notify_open", open), ("notify_close", close), ("notify_daily", daily)):
            if value is not None:
                self.store.set_setting(key, "1" if value else "0")

    def status(self):
        return dict(self.settings(), configured=self.configured(), sender=self.sender,
                    last_error=self.last_error, last_sent=self.last_sent,
                    daily_time=self.cfg.daily_email_time)

    # Sending
    def send(self, subject, html_body, images=()):
        to = self.settings()["email"]
        if not (self.configured() and to):
            raise ValueError("אין כתובת מייל או ששרת המייל לא מוגדר")
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, self.sender, to
        msg.set_content("המייל הזה מוצג כ-HTML. פתח אותו בתוכנת מייל רגילה.")
        msg.add_alternative(html_body, subtype="html")
        for cid, png in images:
            msg.get_payload()[1].add_related(png, "image", "png", cid=cid)
        with self.lock:
            try:
                with smtplib.SMTP(self.host, self.port, timeout=20) as smtp:
                    smtp.starttls()
                    smtp.login(self.user, self.password)
                    smtp.send_message(msg)
                self.last_sent, self.last_error = time.time(), None
            except (smtplib.SMTPException, OSError) as e:
                self.last_error = f"{time.strftime('%H:%M')} {type(e).__name__}: {e}"
                log.warning("Email failed: %s", e)
                raise

    def send_test(self):
        self.send("בדיקה: התראות בוט פולימרקט עובדות",
                  _shell("ההתראות עובדות ✓", "<p>מעכשיו יגיעו לכאן מיילים על טריידים שנפתחים ונסגרים, וסיכום יומי.</p>",
                         self.url))

    # The loop
    def tick(self, now=None):
        now = now or time.time()
        mode = "live" if self.engine.live.armed() else "paper"
        s, can = self.settings(), self.configured() and bool(self.settings()["email"])
        # Bets of the other mode, and the lab, are never mailed: mark them handled.
        self.store._x("UPDATE bets SET notified = 3 WHERE notified != 3 AND (mode != ? OR status != 'filled')", (mode,))
        opened = self._take(f"notified & {OPENED} = 0", OPENED, mode)
        if opened and s["open"] and can:
            self._mail_opened(opened, mode, now)
        closed = self._take(f"notified & {CLOSED} = 0 AND pnl IS NOT NULL", CLOSED, mode)
        if closed and s["close"] and can:
            self._mail_closed(closed, mode)
        if s["daily"] and can:
            self._maybe_daily(now)

    def _take(self, condition, bit, mode):
        rows = self.store._x(
            f"SELECT id, asset, minutes, window_start, side, seconds_left, fair_prob, avg_price, shares,"
            f" cost + fee, ts, winner, pnl FROM bets WHERE mode = ? AND {condition} ORDER BY ts", (mode,))
        if rows:
            self.store._x(f"UPDATE bets SET notified = notified | {bit} WHERE id IN ({','.join('?' * len(rows))})",
                          [r[0] for r in rows])
        keys = ("id", "asset", "minutes", "window_start", "side", "seconds_left", "fair_prob", "avg_price",
                "shares", "paid", "ts", "winner", "pnl")
        return [dict(zip(keys, r)) for r in rows]

    def _mail_opened(self, bets, mode, now):
        from .store import connect

        db = connect(self.cfg.db_path)
        try:
            cards = {(c["asset"], c["minutes"], c["window_start"]): c
                     for c in self.engine._positions(db, now).get(mode, [])}
        finally:
            db.close()
        parts, images = [], []
        for bet in bets:
            card = cards.get((bet["asset"], bet["minutes"], bet["window_start"]))
            if card is None:
                continue
            card = dict(card, bets=[b for b in card["bets"] if b["ts"] == bet["ts"]] or card["bets"])
            png = chart_png(card)
            cid = make_msgid(domain="polybot") if png else None
            if png:
                images.append((cid, png))
            parts.append(card_html(card, bet, self.cfg.timezone, cid))
        if not parts:
            return
        label = "סימולציה" if mode == "paper" else "כסף אמיתי"
        if len(bets) == 1:
            b = bets[0]
            subject = (f"נפתח טרייד: {NAMES.get(b['asset'], b['asset'])} "
                       f"{'עלייה' if b['side'] == 'up' else 'ירידה'} ב-{_cents(b['avg_price'])} ({label})")
        else:
            subject = f"נפתחו {len(bets)} טריידים ({label})"
        title = "נפתח טרייד" if len(bets) == 1 else f"נפתחו {len(bets)} טריידים"
        self._safe_send(subject, _shell(f"{title} · {label}", "".join(parts), self.url), images)

    def _mail_closed(self, bets, mode):
        total = sum(b["pnl"] for b in bets)
        color = GOOD if total >= 0 else BAD
        rows = "".join(
            f'<tr><td style="padding:8px 0;border-top:1px solid {LINE}">{NAMES.get(b["asset"], b["asset"])} · '
            f'{"עלייה" if b["side"] == "up" else "ירידה"} ב-{_ltr(_cents(b["avg_price"]))}</td>'
            f'<td style="padding:8px 0;border-top:1px solid {LINE}">{"✅ ניצחון" if b["pnl"] > 0 else "❌ הפסד"}</td>'
            f'<td style="padding:8px 0;border-top:1px solid {LINE};font-weight:700;color:{GOOD if b["pnl"] >= 0 else BAD}">'
            f'{_ltr(_money(b["pnl"], signed=True))}</td></tr>' for b in bets)
        label = "סימולציה" if mode == "paper" else "כסף אמיתי"
        body = f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;text-align:center">
<div style="color:{MUTED};font-size:14px">{"רווח" if total >= 0 else "הפסד"} · {len(bets)} {"טרייד" if len(bets) == 1 else "טריידים"} · {label}</div>
<div style="font-size:44px;font-weight:800;color:{color};margin:6px 0">{_ltr(_money(total, signed=True))}</div></div>
<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px;font-size:14px">{rows}</table>"""
        self._safe_send(f"נסגר: {_money(total, signed=True)} ({label})", _shell("טרייד נסגר", body, self.url))

    def _maybe_daily(self, now):
        tz = ZoneInfo(self.cfg.timezone)
        local = dt.datetime.fromtimestamp(now, tz)
        hh, mm = (int(x) for x in self.cfg.daily_email_time.split(":"))
        today = local.strftime("%Y-%m-%d")
        if (local.hour, local.minute) < (hh, mm) or self.store.get_setting("notify_daily_day") == today:
            return
        self.store.set_setting("notify_daily_day", today)
        self._safe_send(*self.daily_mail(now))

    def daily_mail(self, now):
        from . import stats
        from .store import connect

        mode = "live" if self.engine.live.armed() else "paper"
        starts = period_starts(now, self.cfg.timezone)
        db = connect(self.cfg.db_path)
        try:
            day = stats.bet_summary(db, mode, since=starts["day"])
            month = stats.bet_summary(db, mode, since=starts["month"])
        finally:
            db.close()
        balance = self.engine.live.balance()
        day_color = GOOD if day["pnl"] >= 0 else BAD
        month_color = GOOD if month["pnl"] >= 0 else BAD
        label = "סימולציה" if mode == "paper" else "כסף אמיתי"
        date = dt.datetime.fromtimestamp(now, ZoneInfo(self.cfg.timezone)).strftime("%d.%m")
        readiness = self.engine.readiness
        passed = sum(c["passed"] for c in readiness["checks"]) if readiness else 0
        body = f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;text-align:center">
<div style="color:{MUTED};font-size:14px">רווח / הפסד היום · {label}</div>
<div style="font-size:44px;font-weight:800;color:{day_color};margin:6px 0">{_ltr(_money(day['pnl'], signed=True))}</div>
<div style="font-size:15px">{day['settled'] + day['open']} טריידים נפתחו היום · {day['wins']} ניצחונות מתוך {day['settled']} שנסגרו</div>
<div style="font-size:14px;color:{MUTED};margin-top:10px">החודש: <b style="color:{month_color}">{_ltr(_money(month['pnl'], signed=True))}</b>
 ({_ltr(f"{(month['roi'] or 0) * 100:+.1f}%")}) על {month['settled']} טריידים</div></div>
<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px;font-size:14px">
<tr><td style="padding:8px 0;border-top:1px solid {LINE}">יתרה בחשבון פולימרקט</td><td style="padding:8px 0;border-top:1px solid {LINE};font-weight:700">
{_ltr(_money(balance)) if balance is not None else "החשבון עוד לא מחובר"}</td></tr>
<tr><td style="padding:8px 0;border-top:1px solid {LINE}">מוכנות לכסף אמיתי</td><td style="padding:8px 0;border-top:1px solid {LINE};font-weight:700">{passed} מתוך 6 בדיקות</td></tr>
</table>"""
        return f"סיכום יומי {date}: {_money(day['pnl'], signed=True)} ({label})", _shell(f"סיכום יומי · {date}", body, self.url)

    def _safe_send(self, subject, body, images=()):
        try:
            self.send(subject, body, images)
        except Exception:
            pass  # recorded in last_error and shown on the dashboard
