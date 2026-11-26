#!/usr/bin/env python3
"""Show a YouTube live chat on the canvas: the latest messages, in a chat panel you place on the
board. Needs paint_flag.py and hate_words.py in the same folder.

    pip install chat-downloader pillow numpy requests better-profanity

    python3 chat_to_board.py URL --rect 600 860 1010 1010 --dry-run   # preview, no painting
    python3 chat_to_board.py URL --rect 600 860 1010 1010             # paint and keep it updated
    python3 chat_to_board.py --demo --rect 600 860 1010 1010 --dry-run   # fake messages

How it works:
  - Messages are read with chat-downloader (no API key), or with the official YouTube Data API
    if you pass --api-key (needs a Google Cloud key with the YouTube Data API v3 enabled).
  - Each message becomes one line "author: message" in small crisp text: author in French blue,
    message in black, on white. The panel is a ring of line slots: a new message replaces the
    oldest line and gets the ">" marker, so each message costs about one line of pixels (a
    scrolling chat would repaint the whole panel for every message).
  - Messages that are hateful by the same rules as scan_hate.py (profanity, slurs, hostile
    patterns) are skipped, links become [link], and characters the font cannot draw (most
    emoji) are dropped.
  - Every --interval seconds the panel is compared with the board and the differing pixels are
    painted, so it is also repaired if someone paints over it.
"""
import argparse, os, queue, random, re, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import paint_flag as engine
from hate_words import hateful

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
def read_chat_downloader(url, out):
    from chat_downloader import ChatDownloader
    while True:
        try:
            for m in ChatDownloader().get_chat(url):
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
    samples = [("Marie", "Vive la France !"), ("Tom_42", "the croissant flag was so good"),
               ("Lukas", "hello from Germany"), ("hater99", "france sucks"),
               ("Ana", "bonjour tout le monde \U0001f1eb\U0001f1f7"), ("rix", "MERCI"),
               ("Jules", "look https://example.com/x"), ("Zoé", "allez les bleus")]
    for author, msg in samples:
        out.put((author, msg))
        time.sleep(random.uniform(0.2, 0.6))


class Panel:
    """A ring of text lines: new messages replace the oldest line and get the '>' marker."""

    def __init__(self, w, h, size):
        self.w, self.h, self.font = w, h, font(size)
        asc, desc = self.font.getmetrics()
        self.line_h = asc + desc + 1
        self.rows = max(1, (h - 2) // self.line_h)
        self.slots, self.newest = [None] * self.rows, -1

    def add(self, author, msg):
        self.newest = (self.newest + 1) % self.rows
        self.slots[self.newest] = (author, msg)

    def fit(self, text, width):
        if self.font.getlength(text) <= width:
            return text
        while text and self.font.getlength(text + "..") > width:
            text = text[:-1]
        return text + ".."

    def render(self):
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        d.fontmode = "1"  # crisp pixels, no anti-aliasing
        mark_w = self.font.getlength("> ")
        for i, slot in enumerate(self.slots):
            if not slot:
                continue
            y = 1 + i * self.line_h
            if i == self.newest:
                d.text((2, y), ">", font=self.font, fill=MARK)
            author = self.fit(slot[0], self.w * 0.35) + ": "
            d.text((2 + mark_w, y), author, font=self.font, fill=AUTHOR)
            x = 2 + mark_w + self.font.getlength(author)
            d.text((x, y), self.fit(slot[1], self.w - x - 2), font=self.font, fill=TEXT)
        return np.array(img)


def main():
    ap = argparse.ArgumentParser(description="Show a YouTube live chat in a panel on the canvas.")
    ap.add_argument("url", nargs="?", default="https://www.youtube.com/watch?v=cyZS5gfil0E")
    ap.add_argument("--rect", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"),
                    help="chat panel on the board (inclusive); flag_ui.py helps to find a free area")
    ap.add_argument("--font-size", type=int, default=10)
    ap.add_argument("--api-key", help="use the official YouTube Data API instead of chat-downloader")
    ap.add_argument("--demo", action="store_true", help="fake messages instead of the live chat")
    ap.add_argument("--dry-run", action="store_true", help="no painting: save chat_preview.png after a few messages")
    ap.add_argument("--interval", type=float, default=3, help="seconds between panel updates")
    ap.add_argument("--workers", type=int, default=min(128, 8 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=11)
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    x0, y0, x1, y1 = args.rect
    panel = Panel(x1 - x0 + 1, y1 - y0 + 1, args.font_size)
    print(f"chat panel {panel.w}x{panel.h} at ({x0},{y0}): {panel.rows} lines", flush=True)

    msgs = queue.Queue()
    if args.demo:
        reader = threading.Thread(target=read_demo, args=(msgs,), daemon=True)
    elif args.api_key:
        reader = threading.Thread(target=read_youtube_api, args=(args.url, args.api_key, msgs), daemon=True)
    else:
        reader = threading.Thread(target=read_chat_downloader, args=(args.url, msgs), daemon=True)
    reader.start()

    if not args.dry_run:
        cfg = engine.requests.get(f"{engine.BASE}/api/config", timeout=20).json()
        engine._init(cfg["pow_kib"], args.max_bits, args.name)
        print(f"hashing with {engine.backend()}, {args.workers} workers", flush=True)
    pool = ThreadPoolExecutor(args.workers)
    shown = skipped = 0
    started = time.time()
    try:
        while True:
            while True:
                try:
                    author, msg = msgs.get_nowait()
                except queue.Empty:
                    break
                author, msg = clean(author), clean(msg)
                if not msg or hateful(msg) or hateful(author):
                    skipped += 1
                    print(f"  skipped {author!r}: filtered", flush=True)
                    continue
                panel.add(author or "?", msg)
                shown += 1
                print(f"  {author}: {msg}", flush=True)
            target = panel.render()
            w, raw = engine.fetch_board()
            board = np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3)
            area = board[y0:y1 + 1, x0:x1 + 1]
            ys, xs = np.nonzero((area != target).any(axis=2))
            todo = [(int(x0 + x), int(y0 + y), target[y, x].tobytes().hex()) for y, x in zip(ys, xs)]
            if args.dry_run:
                if time.time() - started > 6 or not reader.is_alive():
                    prev = board.copy()
                    prev[y0:y1 + 1, x0:x1 + 1] = target
                    Image.fromarray(prev[max(0, y0 - 20):y1 + 21, max(0, x0 - 20):x1 + 21]).resize(
                        ((x1 - x0 + 41) * 3, (y1 - y0 + 41) * 3), Image.NEAREST).save("chat_preview.png")
                    print(f"{shown} shown, {skipped} filtered; {len(todo)} px would be painted; "
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
