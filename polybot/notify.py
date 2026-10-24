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


def _iso(s):
    """Plain-text version of _ltr for subject lines: a left-to-right isolate, so "+$4.21"
    doesn't turn into "$4.21+" in a right-to-left inbox."""
    return "\u2066" + s + "\u2069"


def _arrow(side):
    return "📈" if side == "up" else "📉"


def _real(mode):
    # Simulation is the everyday case; only real money gets a word in the subject.
    return " · כסף אמיתי" if mode == "live" else ""


def _ltr(s):
    # Keep numbers and signs in order inside right-to-left text.
    return f'<span dir="ltr" style="unicode-bidi:isolate">{html.escape(s)}</span>'


def chart_png(card):
    """The window's price path as a PNG (email clients don't show SVG): the price line, the
    dashed price to beat, and a dot where each bet entered. No text, so no font is needed."""
    from PIL import Image, ImageDraw

    pts, base = card.get("series") or [], card.get("open_price")
    length = card.get("span") or card["minutes"] * 60  # stock cards: the chart's own span
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
        t = b["ts"] - card.get("chart_start", card["window_start"])  # stock cards: the chart's own start
        cx, cy, r = x(t), y(at(t)), 5 * scale
        d.ellipse([cx - r - 2 * scale, cy - r - 2 * scale, cx + r + 2 * scale, cy + r + 2 * scale], fill="white")
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GOOD if b["side"] == "up" else BAD)
    img = img.resize((W, H), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def _hhmm(ts, tz):
    return dt.datetime.fromtimestamp(ts, ZoneInfo(tz)).strftime("%H:%M")


def _shell(title, body, url, preheader=""):
    # The preheader is what the inbox shows under the subject, before the mail is opened.
    hidden = (f'<div style="display:none;max-height:0;overflow:hidden;opacity:0">{html.escape(preheader)}'
              f'{"&#847;&zwnj;&nbsp;" * 40}</div>' if preheader else "")
    link = (f'<p style="margin:24px 0 0"><a href="{html.escape(url)}" style="background:{ACCENT};color:#fff;'
            f'padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600">פתח את הדשבורד</a></p>'
            if url else "")
    return f"""<!doctype html><html lang="he" dir="rtl"><body style="margin:0;background:#f6f7f9;
font-family:Arial,Helvetica,sans-serif;color:{TEXT}">{hidden}<div dir="rtl" style="max-width:600px;margin:0 auto;padding:20px">
<div style="font-size:13px;color:{MUTED};margin-bottom:6px">בוט פולימרקט</div>
<h1 style="font-size:20px;margin:0 0 16px">{title}</h1>{body}{link}
<p style="font-size:12px;color:{MUTED};margin-top:28px">את ההתראות אפשר לשנות או לכבות בדשבורד, באזור "התראות במייל".</p>
</div></body></html>"""


_CONFETTI = None


def confetti_gif():
    """Confetti fired from two cannons, bottom left and bottom right, for about two seconds,
    played once. Drawn once and kept; None if Pillow can't draw it."""
    global _CONFETTI
    if _CONFETTI is not None:
        return _CONFETTI or None
    try:
        import random
        from PIL import Image, ImageDraw

        rnd = random.Random(7)
        W, H, frames, g = 560, 200, 34, 0.55
        colors = ["#ff5d5d", "#ffc53d", "#3ddc84", "#4dabf7", "#b197fc", "#ff8fab", "#ffd43b"]
        pieces = []
        for side in (0, 1):
            for _ in range(70):
                angle = rnd.uniform(0.75, 1.25)  # radians above the horizontal, aimed inward
                speed = rnd.uniform(11, 19)
                vx = speed * (1 if side == 0 else -1) * abs(__import__("math").cos(angle))
                vy = -speed * __import__("math").sin(angle)
                pieces.append({"x": 0 if side == 0 else W, "y": H, "vx": vx, "vy": vy, "c": rnd.choice(colors),
                               "w": rnd.randint(4, 8), "h": rnd.randint(7, 12), "delay": rnd.randint(0, 6),
                               "spin": rnd.uniform(0, 6.28)})
        images = []
        for f in range(frames):
            img = Image.new("RGB", (W, H), "#ffffff")
            d = ImageDraw.Draw(img)
            for c in ([(0, H), (46, H - 30), (40, H - 46), (-6, H - 16)], [(W, H), (W - 46, H - 30), (W - 40, H - 46), (W + 6, H - 16)]):
                d.polygon(c, fill="#343a40")  # the two cannons
            for p in pieces:
                t = f - p["delay"]
                if t < 0:
                    continue
                drag = (1 - 0.965 ** t) / 0.035  # air slows them: distance flattens out
                x = p["x"] + p["vx"] * drag * 1.5
                y = p["y"] + p["vy"] * drag * 1.5 + g * t * t * 0.9
                if -20 < x < W + 20 and -20 < y < H + 20:
                    flip = abs(__import__("math").cos(p["spin"] + t * 0.6))
                    w = max(1, int(p["w"] * flip))
                    d.rectangle([x - w / 2, y - p["h"] / 2, x + w / 2, y + p["h"] / 2], fill=p["c"])
            images.append(img.convert("P", palette=Image.ADAPTIVE, colors=32))
        buf = io.BytesIO()
        # No loop count: mail apps play it once and stop on the last frame.
        images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=70, disposal=2)
        _CONFETTI = buf.getvalue()
    except Exception as e:
        log.warning("Confetti image failed: %s", e)
        _CONFETTI = b""
    return _CONFETTI or None


def trade_name(asset, minutes, market=None):
    """How a trade is named in mails: "ביטקוין 15 דק׳", or a stock market ("AAPL מעל $340")."""
    if minutes:
        return f"{NAMES.get(asset, asset)} {minutes} דק׳"
    market = market or ""
    if "-up-or-down-" in market:
        return f"{asset.upper()} עולה/יורדת"
    strike = re.search(r"-above-(\d+(?:pt\d+)?)-on-", market)
    return f"{asset.upper()} מעל ${strike.group(1).replace('pt', '.')}" if strike else asset.upper()


def side_name(side, minutes, market=None):
    if not minutes and "-up-or-down-" not in (market or ""):
        return "כן" if side == "up" else "לא"
    return "עלייה" if side == "up" else "ירידה"


def card_html(card, bet, tz, chart_cid=None):
    name = NAMES.get(card["asset"], card["asset"].upper())
    labels = card.get("labels") or {"up": "עלייה", "down": "ירידה"}
    digits = 4 if (card.get("open_price") or 100) < 10 else 2
    diff = None if card.get("price") is None or card.get("open_price") is None else card["price"] - card["open_price"]

    def button(side, label, ask):
        on = bet["side"] == side
        color = GOOD if side == "up" else BAD
        style = (f"background:{color};color:#fff;border:1px solid {color}" if on
                 else f"background:#fff;color:{MUTED};border:1px solid {LINE}")
        return (f'<td width="50%" style="padding:4px"><div style="{style};border-radius:10px;padding:12px;'
                f'text-align:center;font-weight:700;font-size:16px">{label} {_ltr(_cents(ask))}</div></td>')

    side = labels[bet["side"]]
    chart = (f'<img src="cid:{chart_cid[1:-1]}" width="560" style="width:100%;height:auto;display:block;'
             f'margin:8px 0" alt="מסלול המחיר בחלון">' if chart_cid else "")
    move = ""
    if diff is not None:
        color = GOOD if diff >= 0 else BAD
        move = f' <span style="color:{color};font-weight:600">({"▲" if diff >= 0 else "▼"} {_ltr(_money(abs(diff), digits))})</span>'
    return f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:16px;margin-bottom:14px">
<div style="font-weight:700;font-size:17px">{html.escape(card.get("title") or name)}</div>
<div style="color:{MUTED};font-size:13px;margin-bottom:10px">{html.escape(card["sub"]) if card.get("sub") else f"{card['minutes']} דקות"} · {_ltr(_hhmm(card.get('chart_start', card['window_start']), tz) + '–' + _hhmm(card['end'], tz))}
 · נכנסנו {_ltr(f"{int(bet['seconds_left'] // 60)}:{int(bet['seconds_left'] % 60):02d}")} לפני הסגירה</div>
<table width="100%" cellpadding="0" cellspacing="0"><tr>
<td><div style="color:{MUTED};font-size:12px">{html.escape(card.get("line_label") or "מחיר לניצחון")}</div><div style="font-weight:700">{_ltr(_money(card.get('open_price'), digits))}</div></td>
<td><div style="color:{MUTED};font-size:12px">מחיר נוכחי</div><div style="font-weight:700">{_ltr(_money(card.get('price'), digits))}{move}</div></td>
</tr></table>{chart}
<table width="100%" cellpadding="0" cellspacing="0"><tr>{button("up", labels["up"], card.get("up_ask"))}{button("down", labels["down"], card.get("down_ask"))}</tr></table>
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
        self._problem_since = None  # when the current run of alerts began
        self._problem_mailed = False
        self._batch = {}  # trade-mail kind -> the batch waiting for things to go quiet
        if self.store.get_setting("notify_initialized") is None:
            # First start: everything already in the database counts as handled.
            self.store._x("UPDATE bets SET notified = 3")
            self.store.set_setting("notify_initialized", "1")

    # Settings
    def configured(self):
        return bool(self.host and self.user and self.password)

    def settings(self):
        get = self.store.get_setting
        # Per-trade mails are opt-in: with three markets they add up to dozens a day.
        return {"email": get("notify_email", ""), "open": get("notify_open", "0") == "1",
                "close": get("notify_close", "0") == "1", "daily": get("notify_daily", "1") == "1",
                "problems": get("notify_problems", "1") == "1",
                # Quiet hours: no mails between these times (local), one summary when they end.
                "quiet": get("notify_quiet", "0") == "1", "quiet_from": get("notify_quiet_from", "02:00"),
                "quiet_to": get("notify_quiet_to", "10:00"),
                "quiet_problems": get("notify_quiet_problems", "1") == "1",
                # A trade mail every N trades of its kind (1: every trade, after 30 quiet seconds).
                "open_every": int(get("notify_open_every", "1")), "close_every": int(get("notify_close_every", "1"))}

    def update(self, email=None, open=None, close=None, daily=None, problems=None, quiet=None, quiet_from=None,
               quiet_to=None, quiet_problems=None, open_every=None, close_every=None):
        if email is not None:
            email = email.strip()
            if email and not EMAIL_RE.match(email):
                raise ValueError("כתובת המייל לא תקינה")
            self.store.set_setting("notify_email", email)
        for key, value in (("notify_open", open), ("notify_close", close), ("notify_daily", daily),
                           ("notify_problems", problems), ("notify_quiet", quiet),
                           ("notify_quiet_problems", quiet_problems)):
            if value is not None:
                self.store.set_setting(key, "1" if value else "0")
        for key, value in (("notify_quiet_from", quiet_from), ("notify_quiet_to", quiet_to)):
            if value is not None:
                if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(value)):
                    raise ValueError("שעה לא תקינה (למשל 02:00)")
                self.store.set_setting(key, value)
        for key, value in (("notify_open_every", open_every), ("notify_close_every", close_every)):
            if value is not None:
                if not 1 <= int(value) <= 100:
                    raise ValueError("כל כמה טריידים: בין 1 ל-100")
                self.store.set_setting(key, str(int(value)))

    def in_quiet_hours(self, now, s=None):
        s = s or self.settings()
        if not s["quiet"]:
            return False
        local = dt.datetime.fromtimestamp(now, ZoneInfo(self.cfg.timezone))
        t = local.hour * 60 + local.minute
        a, b = (int(x[:2]) * 60 + int(x[3:]) for x in (s["quiet_from"], s["quiet_to"]))
        return a <= t < b if a <= b else (t >= a or t < b)  # a range may cross midnight

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
        for image in images:
            cid, data, kind = image if len(image) == 3 else (*image, "png")
            msg.get_payload()[1].add_related(data, "image", kind, cid=cid)
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
        # Trade mails wait until things go quiet: 30 seconds with nothing new of that kind
        # (each new trade restarts the wait, up to 5 minutes), then one mail sums them all up.
        # Openings and results are never mixed: each kind is its own mail.
        open_cond, close_cond = f"notified & {OPENED} = 0", f"notified & {CLOSED} = 0 AND pnl IS NOT NULL"
        quiet = self.in_quiet_hours(now, s)
        if quiet:
            # Nothing is sent; what happens is kept for one summary when the quiet hours end.
            if (s["open"] or s["close"]) and can and self.store.get_setting("notify_quiet_since") is None:
                self.store.set_setting("notify_quiet_since", str(now))
        else:
            since = self.store.get_setting("notify_quiet_since")
            if since is not None:
                self.store.set_setting("notify_quiet_since", None)
                opened = self._take(open_cond, OPENED, mode) if s["open"] else []
                closed = self._take(close_cond, CLOSED, mode) if s["close"] else []
                self._batch.clear()
                if (opened or closed) and can:
                    self._mail_night(opened, closed, mode, float(since))
            for kind, condition, bit, enabled, every, mail in (
                    ("open", open_cond, OPENED, s["open"] and can, s["open_every"],
                     lambda rows: self._mail_opened(rows, mode, now)),
                    ("close", close_cond, CLOSED, s["close"] and can, s["close_every"],
                     lambda rows: self._mail_closed(rows, mode))):
                if enabled and not self._quiet(kind, condition, mode, now, every):
                    continue
                rows = self._take(condition, bit, mode)
                self._batch.pop(kind, None)
                if rows and enabled:
                    mail(rows)
            if s["daily"] and can:
                self._maybe_daily(now)
        self._watch_problems(now, s["problems"] and can and (not quiet or s["quiet_problems"]))

    def _quiet(self, kind, condition, mode, now, every=1):
        """True once a batch of `kind` has had nothing new for mail_quiet_seconds (or has
        waited mail_max_wait_seconds in all, so a steady stream still gets mailed). With
        every > 1, only once that many trades are waiting (then the same 30 quiet seconds)."""
        newest, count = self.store._x(f"SELECT MAX(id), COUNT(*) FROM bets WHERE mode = ? AND {condition}", (mode,))[0]
        if not count:
            self._batch.pop(kind, None)
            return False
        if count < every:
            return False
        batch = self._batch.get(kind)
        if batch is None or batch["newest"] != newest:
            self._batch[kind] = batch = {"newest": newest, "last_new": now, "first": batch["first"] if batch else now}
        return (now - batch["last_new"] >= self.cfg.mail_quiet_seconds
                or (every <= 1 and now - batch["first"] >= self.cfg.mail_max_wait_seconds))

    def _watch_problems(self, now, mail):
        """Mail when a problem lasts two minutes (no flapping on a brief reconnect), and again
        when it clears. A dead server can't mail anyone: that case is the job of an outside
        monitor watching /health."""
        alerts = self.engine._alerts(now)
        if alerts:
            self._problem_since = self._problem_since or now
            if mail and not self._problem_mailed and now - self._problem_since >= 120:
                self._problem_mailed = True
                items = "".join(f"<li style='margin:6px 0'>{html.escape(a)}</li>" for a in alerts)
                self._safe_send("⚠️ תקלה בבוט פולימרקט",
                                _shell("⚠️ יש תקלה", f"<p>הבעיה נמשכת כבר {int(now - self._problem_since)} שניות:</p>"
                                       f"<ul>{items}</ul><p>כל עוד הבעיה נמשכת, המערכת לא מהמרת. "
                                       "יגיע מייל נוסף כשהיא תיפתר.</p>", self.url))
        else:
            if self._problem_mailed and mail:
                self._safe_send("🟢 הבוט חזר לפעול",
                                _shell("🟢 הבעיה נפתרה", f"<p>המערכת חזרה לפעול כרגיל אחרי "
                                       f"{int(now - self._problem_since)} שניות.</p>", self.url))
            self._problem_since, self._problem_mailed = None, False

    def _take(self, condition, bit, mode):
        rows = self.store._x(
            f"SELECT id, asset, minutes, window_start, side, seconds_left, fair_prob, avg_price, shares,"
            f" cost + fee, ts, winner, pnl, market FROM bets WHERE mode = ? AND {condition} ORDER BY ts", (mode,))
        if rows:
            self.store._x(f"UPDATE bets SET notified = notified | {bit} WHERE id IN ({','.join('?' * len(rows))})",
                          [r[0] for r in rows])
        keys = ("id", "asset", "minutes", "window_start", "side", "seconds_left", "fair_prob", "avg_price",
                "shares", "paid", "ts", "winner", "pnl", "market")
        return [dict(zip(keys, r)) for r in rows]

    def _mail_opened(self, bets, mode, now):
        from .store import connect

        db = connect(self.cfg.db_path)
        try:
            cards = {(c["asset"], c["minutes"], c["window_start"], c.get("market")): c
                     for c in self.engine._positions(db, now).get(mode, [])}
        finally:
            db.close()
        parts, images = [], []
        for bet in bets:
            card = cards.get((bet["asset"], bet["minutes"], bet["window_start"], bet["market"] if not bet["minutes"] else None))
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
        win = sum(b["shares"] - b["paid"] for b in bets)
        loss = sum(b["paid"] for b in bets)
        chance = sum(b["fair_prob"] or 0 for b in bets) / len(bets)
        odds = f"{_iso(f'{chance * 100:.0f}%')} || {_iso(_money(win, signed=True))} / {_iso(_money(-loss, signed=True))}"
        if len(bets) == 1:
            # 📈 bet on Up, 📉 on Down; then the estimated chance || what a win pays / what a loss costs.
            b = bets[0]
            subject = f"{_arrow(b['side'])} {trade_name(b['asset'], b['minutes'], b['market'])} {odds}{_real(mode)}"
        else:
            sides = {b["side"] for b in bets}
            arrow = _arrow(sides.pop()) if len(sides) == 1 else "📊"
            subject = f"{arrow} {len(bets)} עסקאות נפתחו · {odds}{_real(mode)}"
            parts.insert(0, self._opened_summary(bets, chance, win, loss))
        preheader = " · ".join(f"{side_name(b['side'], b['minutes'], b['market'])} {trade_name(b['asset'], b['minutes'], b['market'])}, "
                               f"{_money(b['paid'])} בסיכון" for b in bets) + f" · {label}"
        title = "נפתח טרייד" if len(bets) == 1 else f"נפתחו {len(bets)} טריידים"
        self._safe_send(subject, _shell(f"{title} · {label}", "".join(parts), self.url, preheader), images)

    def _opened_summary(self, bets, chance, win, loss):
        """Several openings at once: the average chance, the chance all of them win, and each one."""
        all_win = 1.0
        for b in bets:
            all_win *= b["fair_prob"] or 0
        expected = sum((b["fair_prob"] or 0) * b["shares"] - b["paid"] for b in bets)
        td = f'style="padding:7px 4px;border-top:1px solid {LINE}"'
        rows = "".join(
            f'<tr><td {td}>{_arrow(b["side"])} {html.escape(trade_name(b["asset"], b["minutes"], b["market"]))}'
            f' · {side_name(b["side"], b["minutes"], b["market"])}</td><td {td}>{_ltr(f"{(b["fair_prob"] or 0) * 100:.0f}%")}</td>'
            f'<td {td} style="padding:7px 4px;border-top:1px solid {LINE};color:{GOOD};font-weight:700">{_ltr(_money(b["shares"] - b["paid"], signed=True))}</td>'
            f'<td style="padding:7px 4px;border-top:1px solid {LINE};color:{BAD};font-weight:700">{_ltr(_money(-b["paid"], signed=True))}</td></tr>'
            for b in bets)
        head = "".join(f'<th style="text-align:right;padding:6px 4px;color:{MUTED};font-weight:600">{h}</th>'
                       for h in ("טרייד", "סיכוי", "אם ננצח", "אם נפסיד"))
        return f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;margin-bottom:14px;text-align:center">
<div style="color:{MUTED};font-size:14px">סיכוי ממוצע לנצח</div>
<div style="font-size:44px;font-weight:800;margin:4px 0">{_ltr(f"{chance * 100:.0f}%")}</div>
<div style="font-size:14px">הסיכוי שכל ה-{len(bets)} ינצחו: <b>{_ltr(f"{all_win * 100:.0f}%")}</b> · רווח צפוי: <b style="color:{GOOD if expected >= 0 else BAD}">{_ltr(_money(expected, signed=True))}</b></div>
<div style="font-size:14px;margin-top:4px">אם כולן ינצחו: <b style="color:{GOOD}">{_ltr(_money(win, signed=True))}</b> · אם כולן יפסידו: <b style="color:{BAD}">{_ltr(_money(-loss, signed=True))}</b></div>
<table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px;margin-top:12px;text-align:right"><tr>{head}</tr>{rows}</table></div>"""

    def _mail_closed(self, bets, mode):
        total = sum(b["pnl"] for b in bets)
        color = GOOD if total >= 0 else BAD
        rows = "".join(
            f'<tr><td style="padding:8px 0;border-top:1px solid {LINE}">{trade_name(b["asset"], b["minutes"], b["market"])} · '
            f'{side_name(b["side"], b["minutes"], b["market"])} ב-{_ltr(_cents(b["avg_price"]))}</td>'
            f'<td style="padding:8px 0;border-top:1px solid {LINE}">{"✅ ניצחון" if b["pnl"] > 0 else "❌ הפסד"}</td>'
            f'<td style="padding:8px 0;border-top:1px solid {LINE};font-weight:700;color:{GOOD if b["pnl"] >= 0 else BAD}">'
            f'{_ltr(_money(b["pnl"], signed=True))}</td></tr>' for b in bets)
        label = "סימולציה" if mode == "paper" else "כסף אמיתי"
        body = f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;text-align:center">
<div style="color:{MUTED};font-size:14px">{"רווח" if total >= 0 else "הפסד"} · {len(bets)} {"טרייד" if len(bets) == 1 else "טריידים"} · {label}</div>
<div style="font-size:44px;font-weight:800;color:{color};margin:6px 0">{_ltr(_money(total, signed=True))}</div></div>
<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px;font-size:14px">{rows}</table>"""
        # ✅ made money, ❌ lost money, and how much: the whole subject.
        mark = "✅" if total >= 0 else "❌"
        subject = (f"{mark} {_iso(_money(total, signed=True))}" if len(bets) == 1
                   else f"{mark} {len(bets)} עסקאות נסגרו · {_iso(_money(total, signed=True))}") + _real(mode)
        preheader = " · ".join(f"{trade_name(b['asset'], b['minutes'], b['market'])} {side_name(b['side'], b['minutes'], b['market'])} "
                               f"{_money(b['pnl'], signed=True)}" for b in bets) + f" · {label}"
        images = []
        if total > 0:
            # A win: confetti fired from two cannons, once (an animated image: mail apps run no code).
            gif = confetti_gif()
            if gif:
                cid = make_msgid(domain="polybot")
                images.append((cid, gif, "gif"))
                body = (f'<img src="cid:{cid[1:-1]}" width="560" style="width:100%;height:auto;display:block;'
                        f'border-radius:14px;margin-bottom:12px" alt="🎉">') + body
        self._safe_send(subject, _shell("טרייד נסגר" if len(bets) == 1 else f"נסגרו {len(bets)} טריידים", body,
                                        self.url, preheader), images)

    def _mail_night(self, opened, closed, mode, since):
        """When the quiet hours end: one mail with everything that happened during them."""
        label = "סימולציה" if mode == "paper" else "כסף אמיתי"
        total = sum(b["pnl"] for b in closed)
        tz = ZoneInfo(self.cfg.timezone)
        td = f'style="padding:7px 4px;border-top:1px solid {LINE}"'

        def row(b, result):
            when = dt.datetime.fromtimestamp(b["ts"], tz).strftime("%H:%M")
            return (f'<tr><td {td}>{when}</td><td {td}>{_arrow(b["side"])} {html.escape(trade_name(b["asset"], b["minutes"], b["market"]))}'
                    f' · {side_name(b["side"], b["minutes"], b["market"])}</td><td {td}>{_ltr(_cents(b["avg_price"]))}</td><td {td}>{result}</td></tr>')
        closed_rows = "".join(row(b, f'<b style="color:{GOOD if b["pnl"] > 0 else BAD}">{"✅" if b["pnl"] > 0 else "❌"} '
                                     f'{_ltr(_money(b["pnl"], signed=True))}</b>') for b in closed)
        opened_rows = "".join(row(b, f'{_ltr(f"{(b["fair_prob"] or 0) * 100:.0f}%")} סיכוי') for b in opened)
        body = f"""<div style="background:#fff;border:1px solid {LINE};border-radius:14px;padding:20px;text-align:center">
<div style="color:{MUTED};font-size:14px">בשעות השקטות · {label}</div>
<div style="font-size:44px;font-weight:800;color:{GOOD if total >= 0 else BAD};margin:6px 0">{_ltr(_money(total, signed=True))}</div>
<div style="font-size:15px">{len(opened)} טריידים נפתחו · {len(closed)} נסגרו</div></div>
{f'<h2 style="font-size:16px;margin:20px 0 8px">נסגרו</h2><table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">{closed_rows}</table>' if closed else ""}
{f'<h2 style="font-size:16px;margin:20px 0 8px">נפתחו</h2><table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">{opened_rows}</table>' if opened else ""}"""
        subject = (f"🌙 בשעות השקטות: {len(opened)} נפתחו · {len(closed)} נסגרו"
                   + (f" · {_iso(_money(total, signed=True))}" if closed else "") + _real(mode))
        self._safe_send(subject, _shell("מה קרה בשעות השקטות", body, self.url,
                                        f"{len(opened)} נפתחו, {len(closed)} נסגרו, {_money(total, signed=True)}"))

    def _maybe_daily(self, now):  # (held during quiet hours; tick sends it once they end)
        tz = ZoneInfo(self.cfg.timezone)
        local = dt.datetime.fromtimestamp(now, tz)
        hh, mm = (int(x) for x in self.cfg.daily_email_time.split(":"))
        today = local.strftime("%Y-%m-%d")
        if (local.hour, local.minute) < (hh, mm) or self.store.get_setting("notify_daily_day") == today:
            return
        self.store.set_setting("notify_daily_day", today)
        self._safe_send(*self.daily_mail(now))

    def send_summary_now(self):
        """The settings page's "send a summary now": today's summary, sent immediately."""
        if not (self.configured() and self.settings()["email"]):
            raise ValueError("אין כתובת מייל או ששרת המייל לא מוגדר")
        self.send(*self.daily_mail(time.time()))

    def trades_table(self, db, mode, since):
        """Every trade of the day, newest first, each marked simulation or real money."""
        rows = db.execute(
            "SELECT ts, asset, minutes, side, avg_price, cost + fee AS paid, pnl, market FROM bets"
            " WHERE mode = ? AND status = 'filled' AND ts >= ? ORDER BY ts DESC", (mode, since)).fetchall()
        tag_text, tag_bg, tag_fg = (("סימולציה", "#eef2ff", ACCENT) if mode == "paper" else ("כסף אמיתי", "#fdecec", BAD))
        tz = ZoneInfo(self.cfg.timezone)
        cells = []
        for r in rows:
            when = dt.datetime.fromtimestamp(r["ts"], tz).strftime("%H:%M")
            if r["pnl"] is None:
                result, color = "פתוח", MUTED
            else:
                result, color = (("✅ " if r["pnl"] > 0 else "❌ ") + _money(r["pnl"], signed=True)), (GOOD if r["pnl"] > 0 else BAD)
            td = f'style="padding:7px 4px;border-top:1px solid {LINE}"'
            cells.append(
                f'<tr><td {td}>{when}</td><td {td}>{"📈" if r["side"] == "up" else "📉"} '
                f'{html.escape(trade_name(r["asset"], r["minutes"], r["market"]))}</td><td {td}>{_ltr(_cents(r["avg_price"]))}</td>'
                f'<td {td}>{_ltr(_money(r["paid"]))}</td><td style="padding:7px 4px;border-top:1px solid {LINE};'
                f'color:{color};font-weight:700">{_ltr(result) if r["pnl"] is not None else result}</td>'
                f'<td {td}><span style="background:{tag_bg};color:{tag_fg};border-radius:999px;padding:2px 8px;'
                f'font-size:11px;font-weight:700">{tag_text}</span></td></tr>')
        if not cells:
            return f'<p style="color:{MUTED};font-size:14px">אין טריידים היום.</p>'
        head = "".join(f'<th style="text-align:right;padding:6px 4px;color:{MUTED};font-weight:600">{h}</th>'
                       for h in ("שעה", "טרייד", "מחיר", "סכום", "תוצאה", ""))
        return (f'<table width="100%" cellpadding="0" cellspacing="0" style="font-size:13px">'
                f'<tr>{head}</tr>{"".join(cells)}</table>')

    def daily_mail(self, now):
        from . import stats
        from .store import connect

        mode = "live" if self.engine.live.armed() else "paper"
        starts = period_starts(now, self.cfg.timezone)
        db = connect(self.cfg.db_path)
        try:
            which = self.engine.strategy_filter()
            day = stats.bet_summary(db, mode, since=starts["day"], strategy=which)
            month = stats.bet_summary(db, mode, since=starts["month"], strategy=which)
            costs = self.engine.costs_since(db, starts["month"], now)
            learned = self.learned(db, starts)
            trades = self.trades_table(db, mode, starts["day"])
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
 ({_ltr(f"{(month['roi'] or 0) * 100:+.1f}%")}) על {month['settled']} טריידים, אחרי עמלות פולימרקט</div>
<div style="font-size:13px;color:{MUTED};margin-top:4px">אחרי עלויות תפעול (שרת {_ltr(_money(costs['hosting']))}, Claude {_ltr(_money(costs['claude']))}):
 <b style="color:{GOOD if month['pnl'] - costs['total'] >= 0 else BAD}">{_ltr(_money(month['pnl'] - costs['total'], signed=True))}</b></div></div>
<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px;font-size:14px">
<tr><td style="padding:8px 0;border-top:1px solid {LINE}">יתרה בחשבון פולימרקט</td><td style="padding:8px 0;border-top:1px solid {LINE};font-weight:700">
{_ltr(_money(balance)) if balance is not None else "החשבון עוד לא מחובר"}</td></tr>
<tr><td style="padding:8px 0;border-top:1px solid {LINE}">מוכנות לכסף אמיתי</td><td style="padding:8px 0;border-top:1px solid {LINE};font-weight:700">{passed} מתוך 6 בדיקות</td></tr>
</table>
<h2 style="font-size:16px;margin:22px 0 8px">כל הטריידים של היום</h2>
{trades}
<h2 style="font-size:16px;margin:22px 0 8px">מה המערכת למדה היום</h2>
<ul style="margin:0;padding-inline-start:20px;font-size:14px;line-height:1.7">{"".join(f"<li>{html.escape(x)}</li>" for x in learned)}</ul>"""
        return (f"📊 סיכום היום: {_iso(_money(day['pnl'], signed=True))}{_real(mode)}",
                _shell(f"סיכום יומי · {date}", body, self.url,
                       f"החודש {_money(month['pnl'], signed=True)} · {day['settled']} טריידים נסגרו היום · {label}"))

    def learned(self, db, starts):
        """A few plain sentences on what the data showed. Claude's research brief when one was
        written today; otherwise straight from the measured numbers."""
        from . import research, stats

        latest = research.latest(db)
        if latest and latest.get("data") and latest["ts"] >= starts["day"]:
            d = latest["data"]
            return [d["headline"]] + d["key_findings"][:3]
        out = []
        mix = stats.strategy_mix(db, self.cfg, starts["day"], self.engine.current_variants)
        if mix["bets"]:
            out.append(f"המעבדה הריצה היום {mix['bets']} הימורים מדומים שהוכרעו, "
                       f"{'ברווח' if mix['pnl'] >= 0 else 'בהפסד'} של {(mix['roi'] or 0) * 100:+.1f}% על כל דולר.")
            if mix["top"]:
                t = mix["top"][0]
                out.append(f"הכי טובה היום: {NAMES.get(t['asset'], t['asset'])} בחלונות של {t['minutes']} דקות, "
                           f"{t['rule']}: {t['roi'] * 100:+.0f}% על {t['bets']} הימורים.")
            c = mix["counts"]
            out.append(f"{c['hot']} אסטרטגיות הרוויחו יפה היום, {c['swinging']} התנדנדו, {c['losing']} הפסידו, "
                       f"ול-{c['few']} עוד אין מספיק נתונים.")
        else:
            out.append("המעבדה עוד לא סגרה הימורים היום.")
        board = (self.engine.lab or {}).get("board", [])
        proven = sum(1 for b in board if b["status"] == "proven")
        out.append(f"{proven} אסטרטגיות עמדו עד עכשיו במבחן הסטטיסטי המחמיר." if proven
                   else "עדיין אין אסטרטגיה שעמדה במבחן הסטטיסטי המחמיר. זה דורש ימים של נתונים.")
        cal = (self.engine.analysis or {}).get("calibration")
        if cal and cal["bets"] >= 50:
            gap = cal["said"] - cal["happened"]
            verdict = "בטוח בעצמו יותר מדי" if gap > 0.05 else "זהיר מדי" if gap < -0.05 else "מכויל היטב"
            out.append(f"כיול: המודל נתן בממוצע {cal['said'] * 100:.0f}% סיכוי, וזה קרה ב-{cal['happened'] * 100:.0f}% "
                       f"מהמקרים ({cal['bets']} הימורים). כרגע הוא {verdict}.")
        readiness = self.engine.readiness
        if readiness:
            names = {"hours": "זמן הקלטה", "feed": "התאמה להכרעת פולימרקט", "bets": "מספר הימורים",
                     "roi": "רווח אחרי עמלות", "luck": "רווח שאינו מזל", "recent": "רווח לאחרונה"}
            missing = [names.get(c["id"], c["id"]) for c in readiness["checks"] if not c["passed"]]
            out.append("כל בדיקות המוכנות לכסף אמיתי עברו." if not missing
                       else "עוד חסר לכסף אמיתי: " + ", ".join(missing) + ".")
        return out

    def _safe_send(self, subject, body, images=()):
        try:
            self.send(subject, body, images)
        except Exception:
            pass  # recorded in last_error and shown on the dashboard
