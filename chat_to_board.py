#!/usr/bin/env python3
"""Show a YouTube live chat on the canvas: the latest messages, in a chat panel you place on the
board. Needs paint_flag.py, pixfont.py and guard_flags.py in the same folder.

    pip install -U yt-dlp pillow numpy requests

    python3 chat_to_board.py URL --dry-run                  # finds a spot, previews, paints nothing
    python3 chat_to_board.py URL                            # paint and keep it updated
    python3 chat_to_board.py URL --cookies-from-browser firefox   # if YouTube asks to sign in
    python3 chat_to_board.py --demo --dry-run               # fake messages

Where: a "chat" zone drawn in flag_ui.py (re-read every round: move it there and the panel
follows), else --rect X0 Y0 X1 Y1, else (--rect auto, the default) the spot covering the least
art, chosen at the first start and kept in chat_rect.txt (--new-spot to choose again).

How it works:
  - Messages are read with yt-dlp (default; keep it updated with pip install -U yt-dlp), which
    follows the live chat as it happens. If YouTube answers "Sign in to confirm you're not a bot"
    or shows its cookie-consent page, add --cookies-from-browser firefox (or chrome, ...) to use
    your own browser's YouTube cookies, or --cookies FILE. Other readers: --reader
    chat-downloader, or the official YouTube Data API with --api-key (a Google Cloud key with
    the YouTube Data API v3 enabled).
  - The last --lines messages (6) are shown, one line "author: message" each, in a 5x7 pixel
    font (8 px per line, see pixfont.py; --font ttf --font-size N for a normal font): author in
    French blue, message in black, on white. The panel is a ring of line slots: a new message replaces the
    oldest line and gets the ">" marker, so each message costs about one line of pixels (a
    scrolling chat would repaint the whole panel for every message).
  - Every message is shown, no word is filtered. Links become [link], and characters the font
    cannot draw (most emoji) are dropped.
  - New messages are picked up continuously: yt-dlp keeps polling YouTube (as often as YouTube
    asks, every few seconds) and appends them to its file, which this script reads every second.
    Each message is printed as it arrives, and once a minute a "listening" line says how many
    came in and when the last one did, so a quiet chat is told apart from a stuck reader.
  - Every --interval seconds the panel is compared with the board and the differing pixels are
    painted, so it is also repaired if someone paints over it.
"""
import argparse, os, queue, random, re, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import paint_flag as engine
import pixfont
from guard_flags import CHAT_LINES

AUTHOR, TEXT, BG, MARK = (0x00, 0x23, 0x95), (0, 0, 0), (255, 255, 255), (0xed, 0x29, 0x39)


def font(size):
    for name in ("DejaVu Sans", "Liberation Sans", "Arial"):
        try:
            path = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True,
                                  timeout=5).stdout
            if path and os.path.exists(path):
                return ImageFont.truetype(path, size)
        except (OSError, subprocess.SubprocessError):
            break
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def clean(text):
    """Links become [link], undrawable characters (emoji, ...) are dropped, spaces collapsed."""
    text = re.sub(r"https?://\S+|www\.\S+", "[link]", text or "")
    text = "".join(ch for ch in text if ch.isprintable() and ord(ch) < 0x2500)
    return re.sub(r"\s+", " ", text).strip()


# chat sources: each puts (author, message) tuples into a queue
def chat_items(obj):
    """(author, message) of every chat message renderer found in a YouTube chat JSON object."""
    if isinstance(obj, dict):
        for key in ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer"):
            r = obj.get(key)
            if isinstance(r, dict):
                runs = r.get("message", {}).get("runs", [])
                text = "".join(run.get("text") or (run.get("emoji", {}).get("shortcuts") or [""])[0] for run in runs)
                yield r.get("authorName", {}).get("simpleText", ""), text
        for v in obj.values():
            yield from chat_items(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from chat_items(v)


def read_ytdlp(url, out, cookie_args):
    """Run yt-dlp on the live chat and follow the JSON lines it writes as messages arrive."""
    import json, shutil, sys, tempfile
    while True:
        tmp = tempfile.mkdtemp(prefix="chat_")
        cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--write-subs", "--sub-langs", "live_chat",
               "--no-warnings", *cookie_args, "-o", os.path.join(tmp, "chat"), url]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        pos, path = 0, None
        while True:
            alive = proc.poll() is None
            for name in ("chat.live_chat.json.part", "chat.live_chat.json"):
                if os.path.exists(os.path.join(tmp, name)):
                    if path != name:
                        path, pos = name, (pos if path else 0)
                    break
            if path and os.path.exists(os.path.join(tmp, path)):
                with open(os.path.join(tmp, path), encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    data = f.read()
                end = data.rfind("\n") + 1  # complete lines only
                pos += len(data[:end].encode("utf-8"))
                for line in data[:end].splitlines():
                    try:
                        for item in chat_items(json.loads(line)):
                            out.put(item)
                    except ValueError:
                        pass
            if not alive:
                break
            time.sleep(1)
        err = (proc.stderr.read() or "").strip().splitlines()
        print(f"chat reader (yt-dlp) stopped: {err[-1] if err else 'stream ended?'} (retrying in 15 s)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        time.sleep(15)


def read_chat_downloader(url, out, cookies=None):
    from chat_downloader import ChatDownloader
    while True:
        try:
            for m in ChatDownloader(cookies=cookies).get_chat(url):
                out.put((m.get("author", {}).get("name", ""), m.get("message", "")))
        except Exception as e:
            print(f"chat reader: {type(e).__name__}: {e} (retrying in 15 s)", flush=True)
        time.sleep(15)


def read_youtube_api(url, key, out):
    import requests
    vid = re.search(r"(?:v=|youtu\.be/|/live/)([\w-]{11})", url)
    vid = vid.group(1) if vid else url
    api = "https://www.googleapis.com/youtube/v3"
    info = requests.get(f"{api}/videos", params={"part": "liveStreamingDetails", "id": vid, "key": key},
                        timeout=20).json()
    chat_id = (info.get("items") or [{}])[0].get("liveStreamingDetails", {}).get("activeLiveChatId")
    if not chat_id:
        print("YouTube API: no active live chat for this video", flush=True)
        return
    token = None
    while True:
        params = {"liveChatId": chat_id, "part": "snippet,authorDetails", "key": key}
        if token:
            params["pageToken"] = token
        r = requests.get(f"{api}/liveChat/messages", params=params, timeout=20).json()
        if "error" in r:
            print(f"YouTube API error: {r['error'].get('message')}", flush=True)
            time.sleep(30)
            continue
        for it in r.get("items", []):
            out.put((it["authorDetails"]["displayName"], it["snippet"].get("displayMessage", "")))
        token = r.get("nextPageToken")
        time.sleep(max(5.0, r.get("pollingIntervalMillis", 5000) / 1000))


def read_demo(out):
    """Fake chat: a few quick messages, then one every few seconds, forever."""
    samples = [("Marie", "Vive la France !"), ("Tom_42", "the croissant flag was so good"),
               ("Lukas", "hello from Germany"), ("Brenno", "oh damn they got live chat"),
               ("Ana", "bonjour tout le monde \U0001f1eb\U0001f1f7"), ("rix", "MERCI"),
               ("Jules", "look https://example.com/x"), ("Zoé", "allez les bleus")]
    for i in range(10 ** 9):
        out.put(samples[i % len(samples)])
        time.sleep(random.uniform(0.2, 0.6) if i < len(samples) else random.uniform(2, 4))


class Panel:
    """A ring of text lines: new messages replace the oldest line and get the '>' marker."""

    def __init__(self, w, rows, font_kind="pixel", size=10):
        self.w, self.rows, self.kind = w, rows, font_kind
        if font_kind == "pixel":
            self.line_h = pixfont.HEIGHT + 1
        else:
            self.font = font(size)
            asc, desc = self.font.getmetrics()
            self.line_h = asc + desc + 1
        self.h = rows * self.line_h + 1
        self.slots, self.newest = [None] * rows, -1

    def add(self, author, msg):
        self.newest = (self.newest + 1) % self.rows
        self.slots[self.newest] = (author, msg)

    def length(self, text):
        return pixfont.width(text) if self.kind == "pixel" else self.font.getlength(text)

    def fit(self, text, width):
        if self.kind == "pixel":
            text = pixfont.to_ascii(text)
        if self.length(text) <= width:
            return text
        while text and self.length(text + "..") > width:
            text = text[:-1]
        return text + ".."

    def render(self):
        img = Image.new("RGB", (self.w, self.h), BG)
        arr = np.array(img)
        d = ImageDraw.Draw(img)
        d.fontmode = "1"  # crisp pixels, no anti-aliasing
        mark_w = self.length("> ")
        for i, slot in enumerate(self.slots):
            if not slot:
                continue
            y = 1 + i * self.line_h
            author = self.fit(slot[0], self.w * 0.3) + ": "
            x = 2 + mark_w + self.length(author)
            msg = self.fit(slot[1], self.w - x - 2)
            if self.kind == "pixel":
                if i == self.newest:
                    pixfont.draw(arr, 2, y, ">", MARK)
                pixfont.draw(arr, 2 + mark_w, y, author, AUTHOR)
                pixfont.draw(arr, x, y, msg, TEXT)
            else:
                if i == self.newest:
                    d.text((2, y), ">", font=self.font, fill=MARK)
                d.text((2 + mark_w, y), author, font=self.font, fill=AUTHOR)
                d.text((x, y), msg, font=self.font, fill=TEXT)
        return arr if self.kind == "pixel" else np.array(img)


SPOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_rect.txt")


def find_spot(board, w, h, step=2):
    """(x0, y0, x1, y1) of the w x h spot that is the most white already (cheapest to paint, covers
    no art: letter strokes, lines and drawings count 8x more than noise dots), away from guarded.txt
    zones and our own art."""
    H, W = board.shape[:2]
    nonwhite = (board != 255).any(axis=2)
    # pixels that are part of something (a letter stroke, a line, a drawing: 2+ non-white neighbours)
    # weigh 8x more than isolated noise dots, so text and art are avoided, noise is not
    from scipy.ndimage import convolve
    k = np.ones((3, 3), np.int32)
    k[1, 1] = 0
    structure = nonwhite & (convolve(nonwhite.astype(np.int32), k, mode="constant") >= 2)
    busy = nonwhite.astype(np.int32) + 7 * structure
    try:
        from guard_flags import load_guarded, our_art
        for z in load_guarded():
            x0, y0, x1, y1 = z.rect
            busy[max(0, y0 - 3):y1 + 4, max(0, x0 - 3):x1 + 4] = 10 ** 6  # never on (or touching) a zone
        for (x, y) in our_art():
            busy[y, x] = 10 ** 6
    except ImportError:
        pass
    ii = np.pad(busy.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = None
    for y in range(2, H - h - 2, step):
        row = ii[y + h, w + 2:W - 1] - ii[y, w + 2:W - 1] - ii[y + h, 2:W - w - 1] + ii[y, 2:W - w - 1]
        x = int(np.argmin(row[::step]) * step)
        if best is None or row[x] < best[0]:
            best = (int(row[x]), x + 2, y)
    return (best[1], best[2], best[1] + w - 1, best[2] + h - 1), best[0]


def main():
    ap = argparse.ArgumentParser(description="Show a YouTube live chat in a panel on the canvas.")
    ap.add_argument("url", nargs="?", default="https://www.youtube.com/watch?v=cyZS5gfil0E")
    ap.add_argument("--rect", nargs="+", default=["auto"], metavar="X0 Y0 X1 Y1",
                    help="panel position (inclusive), or 'auto' (default): whitest free spot, kept in chat_rect.txt")
    ap.add_argument("--new-spot", action="store_true", help="with --rect auto: choose a new spot")
    ap.add_argument("--lines", type=int, default=CHAT_LINES, help="how many messages to show")
    ap.add_argument("--width", type=int, default=300, help="panel width for --rect auto")
    ap.add_argument("--font", choices=["pixel", "ttf"], default="pixel", help="5x7 pixel font, or a normal font")
    ap.add_argument("--font-size", type=int, default=10, help="size of the normal font (--font ttf)")
    ap.add_argument("--reader", choices=["yt-dlp", "chat-downloader", "api"], default="yt-dlp",
                    help="how to read the chat (api needs --api-key)")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER",
                    help="yt-dlp: use this browser's YouTube cookies (firefox, chrome, ...)")
    ap.add_argument("--cookies", metavar="FILE", help="cookies.txt file for yt-dlp / chat-downloader")
    ap.add_argument("--api-key", help="key for the official YouTube Data API (with --reader api)")
    ap.add_argument("--demo", action="store_true", help="fake messages instead of the live chat")
    ap.add_argument("--dry-run", action="store_true", help="no painting: save chat_preview.png after a few messages")
    ap.add_argument("--interval", type=float, default=3, help="seconds between panel updates")
    ap.add_argument("--preview-after", type=float, default=40, help="--dry-run: seconds of chat to collect")
    ap.add_argument("--workers", type=int, default=engine.default_workers())
    ap.add_argument("--max-bits", type=int, default=11)
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    from guard_flags import load_guarded

    def chat_zone():
        """The chat zone drawn in flag_ui.py, if any (with --rect auto only)."""
        if args.rect != ["auto"]:
            return None
        return next((z.rect for z in load_guarded() if z.kind == "chat" and z.enabled), None)

    if args.rect != ["auto"]:
        x0, y0, x1, y1 = (int(v) for v in args.rect)
        panel = Panel(x1 - x0 + 1, args.lines, args.font, args.font_size)
        y1 = y0 + panel.h - 1  # the height follows the lines and the font
    elif chat_zone():
        x0, y0, x1, _ = chat_zone()
        panel = Panel(x1 - x0 + 1, args.lines, args.font, args.font_size)
        y1 = y0 + panel.h - 1
        print("using the chat zone from guarded.txt (move it in flag_ui.py, the panel follows)", flush=True)
    else:
        panel = Panel(args.width, args.lines, args.font, args.font_size)
        saved = None
        if os.path.exists(SPOT_FILE) and not args.new_spot:
            try:
                saved = tuple(int(v) for v in open(SPOT_FILE).read().split()[:4])
            except ValueError:
                saved = None
        if saved and saved[2] - saved[0] + 1 == panel.w and saved[3] - saved[1] + 1 == panel.h:
            x0, y0, x1, y1 = saved
            print(f"reusing the spot in {SPOT_FILE}", flush=True)
        else:
            w, raw = engine.fetch_board()
            (x0, y0, x1, y1), busy = find_spot(np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3), panel.w, panel.h)
            print(f"chose the whitest free spot ({busy} non-white pixels under it)", flush=True)
            if not args.dry_run:
                with open(SPOT_FILE, "w") as f:
                    f.write(f"{x0} {y0} {x1} {y1}\n")
    print(f"chat panel {panel.w}x{panel.h} at ({x0},{y0})-({x1},{y1}): {panel.rows} lines "
          f"(--rect {x0} {y0} {x1} {y1} to reuse it)", flush=True)

    msgs = queue.Queue()
    cookie_args = (["--cookies-from-browser", args.cookies_from_browser] if args.cookies_from_browser else []) + \
                  (["--cookies", args.cookies] if args.cookies else [])
    if args.demo:
        reader = threading.Thread(target=read_demo, args=(msgs,), daemon=True)
    elif args.reader == "api" or args.api_key:
        if not args.api_key:
            raise SystemExit("--reader api needs --api-key")
        reader = threading.Thread(target=read_youtube_api, args=(args.url, args.api_key, msgs), daemon=True)
    elif args.reader == "chat-downloader":
        reader = threading.Thread(target=read_chat_downloader, args=(args.url, msgs, args.cookies), daemon=True)
    else:
        reader = threading.Thread(target=read_ytdlp, args=(args.url, msgs, cookie_args), daemon=True)
    reader.start()

    if not args.dry_run:
        cfg = engine.requests.get(f"{engine.BASE}/api/config", timeout=20).json()
        engine._init(cfg["pow_kib"], args.max_bits, args.name)
        print(f"hashing with {engine.backend()}, {args.workers} workers", flush=True)
    pool = ThreadPoolExecutor(args.workers)
    shown = skipped = received = 0
    started = beat = time.time()
    last_msg = None
    try:
        while True:
            while True:
                try:
                    author, msg = msgs.get_nowait()
                except queue.Empty:
                    break
                received, last_msg = received + 1, time.time()
                author, msg = clean(author), clean(msg)
                if not msg:  # nothing the font can draw (only emoji, for example)
                    skipped += 1
                    print(f"  skipped {author!r}: nothing drawable (emoji only?)", flush=True)
                    continue
                panel.add(author or "?", msg)
                shown += 1
                print(f"  {author}: {msg}", flush=True)
            if time.time() - beat >= 60:  # heartbeat: tells a quiet chat from a stuck reader
                beat = time.time()
                gap = beat - last_msg if last_msg else 0
                ago = (f"last one {gap:.0f} s ago" if gap < 120 else f"last one {gap / 60:.0f} min ago") \
                    if last_msg else "none yet"
                print(f"[{time.strftime('%H:%M:%S')}] listening to the chat: {received} messages so far "
                      f"({shown} shown, {skipped} emoji-only), {ago}", flush=True)
            moved = chat_zone()
            if moved and (moved[0], moved[1], moved[2]) != (x0, y0, x1):
                x0, y0, x1 = moved[0], moved[1], moved[2]
                panel.w, y1 = x1 - x0 + 1, moved[1] + panel.h - 1
                print(f"chat zone moved: panel now at ({x0},{y0})-({x1},{y1})", flush=True)
            target = panel.render()
            w, raw = engine.fetch_board()
            board = np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3)
            area = board[y0:y1 + 1, x0:x1 + 1]
            ys, xs = np.nonzero((area != target).any(axis=2))
            todo = [(int(x0 + x), int(y0 + y), target[y, x].tobytes().hex()) for y, x in zip(ys, xs)]
            if args.dry_run:
                if time.time() - started > args.preview_after or not reader.is_alive():
                    prev = board.copy()
                    prev[y0:y1 + 1, x0:x1 + 1] = target
                    Image.fromarray(prev[max(0, y0 - 20):y1 + 21, max(0, x0 - 20):x1 + 21]).resize(
                        ((x1 - x0 + 41) * 3, (y1 - y0 + 41) * 3), Image.NEAREST).save("chat_preview.png")
                    print(f"{shown} shown, {skipped} emoji-only; {len(todo)} px would be painted; "
                          f"preview saved to chat_preview.png", flush=True)
                    return
            elif todo:
                counts = engine.run_pass(pool, todo, f"[{time.strftime('%H:%M:%S')}] updating the chat panel")
                print(f"  painted {counts.get('ok', 0)}, deferred {counts.get('deferred', 0)}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nstopped", flush=True)
        os._exit(130)


if __name__ == "__main__":
    main()
