#!/usr/bin/env python3
"""Watch the board's live feed, record for every painter name the colour of each pixel they
paint, and find their rectangular artworks (at least 80x80). flag_ui.py lists them, and can place
a copy of one anywhere on the board (guard_flags.py then paints and keeps it).

    pip install websockets numpy scipy pillow requests
    python3 watch_art.py              # until Ctrl-C; leave it running in the background
    python3 watch_art.py --min 100    # only artworks of at least 100x100
    python3 watch_art.py --reset      # forget everything recorded so far

Only paints seen while this runs are known: an artwork fills up as its painter paints or
repairs it. The UI shows how much of each one is known, and "complete" once every pixel is.

How an artwork is found: a painter's known pixels, with gaps of up to 6 px closed (pixels not
painted yet, or skipped by a bot because they already had the right colour), make solid
rectangles; each one at least --min px wide and high, with nothing else of that painter just
outside at least 3 of its sides, is an artwork. Sparse drawings (the worm patterns) and shapes
that are not rectangles (a character on a transparent background) are left out; two artworks
of one painter that touch are told apart.

Files, in artworks/ next to this script:
  index.json                     the artworks: painter, rectangle, known pixels, picture file
  <name>_<x>_<y>_<w>x<h>.png      each artwork, transparent where a pixel is not known yet
  state.npz                      everything recorded (saved every minute, reloaded at start)
  placed/                        copies placed with flag_ui.py (never deleted by this script)
"""
import argparse, asyncio, json, os, re, time

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

import paint_flag as engine

HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(HERE, "artworks")
INDEX = os.path.join(ART_DIR, "index.json")
STATE = os.path.join(ART_DIR, "state.npz")
WS_URL = engine.BASE.replace("https://", "wss://").replace("http://", "ws://") + "/api/ws"


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "noname"


class Painter:
    """Colours one name painted (last one wins): a dict while small, a whole-board array once
    big (-1 = not seen)."""

    def __init__(self, size):
        self.size, self.small, self.arr, self.count, self.changed = size, {}, None, 0, True

    def paint(self, idx, colour):
        if self.arr is None:
            self.count += idx not in self.small
            self.small[idx] = colour
            if len(self.small) > 4000:
                self.arr = np.full(self.size, -1, np.int32)
                self.arr[np.fromiter(self.small, np.int64)] = np.fromiter(self.small.values(), np.int64)
                self.small = None
        else:
            self.count += int(self.arr[idx] < 0)
            self.arr[idx] = colour
        self.changed = True

    def array(self):
        """A copy as a whole-board int32 array (-1 = not seen)."""
        if self.arr is not None:
            return self.arr.copy()
        a = np.full(self.size, -1, np.int32)
        if self.small:
            a[np.fromiter(self.small, np.int64)] = np.fromiter(self.small.values(), np.int64)
        return a


def largest_rect(mask, min_w, min_h, accept):
    """Biggest all-True rectangle at least min_w x min_h for which accept(x0, y0, x1, y1) is true:
    (x0, y0, x1, y1) inclusive, or None. Every maximal rectangle is tried (histogram method)."""
    H, W = mask.shape
    heights = np.zeros(W, np.int64)
    best, best_area = None, 0
    for y in range(H):
        heights = np.where(mask[y], heights + 1, 0)
        if heights.max() < min_h:
            continue
        stack = []  # (start x, height), heights increasing
        for x, h in enumerate(heights.tolist() + [0]):
            start = x
            while stack and stack[-1][1] >= h:
                sx, sh = stack.pop()
                if sh >= min_h and x - sx >= min_w and sh * (x - sx) > best_area \
                        and accept(sx, y - sh + 1, x - 1, y):
                    best_area, best = sh * (x - sx), (sx, y - sh + 1, x - 1, y)
                start = sx
            stack.append((start, h))
    return best


GAP = 4  # how far outside a side we look for "nothing of the same shape there"


def side_checker(shape):
    """exposed(rect) -> [top, bottom, left, right]: True where the line GAP px outside that side
    is (almost) free of `shape`. A real artwork is free on (nearly) all sides; a piece cut out of
    a bigger shape (two touching artworks, a disc) is not."""
    P = np.pad(shape, GAP + 1).astype(np.int32)
    rowc = np.zeros((P.shape[0], P.shape[1] + 1), np.int32)
    rowc[:, 1:] = P.cumsum(1)
    colc = np.zeros((P.shape[0] + 1, P.shape[1]), np.int32)
    colc[1:] = P.cumsum(0)
    o = GAP + 1

    def exposed(x0, y0, x1, y1):
        w, h = x1 - x0 + 1, y1 - y0 + 1
        top = rowc[y0 + o - GAP, x1 + o + 1] - rowc[y0 + o - GAP, x0 + o]
        bottom = rowc[y1 + o + GAP, x1 + o + 1] - rowc[y1 + o + GAP, x0 + o]
        left = colc[y1 + o + 1, x0 + o - GAP] - colc[y0 + o, x0 + o - GAP]
        right = colc[y1 + o + 1, x1 + o + GAP] - colc[y0 + o, x1 + o + GAP]
        return [top <= 0.2 * w, bottom <= 0.2 * w, left <= 0.2 * h, right <= 0.2 * h]
    return exposed


def grow(rect, known, free):
    """Push the free sides out over ragged edges (a half-painted artwork's border rows)."""
    x0, y0, x1, y1 = rect
    H, W = known.shape
    dens = known[y0:y1 + 1, x0:x1 + 1].mean()
    need = max(0.1, 0.5 * dens)
    for _ in range(GAP):
        if free[0] and y0 > 0 and known[y0 - 1, x0:x1 + 1].mean() >= need:
            y0 -= 1
        if free[1] and y1 < H - 1 and known[y1 + 1, x0:x1 + 1].mean() >= need:
            y1 += 1
        if free[2] and x0 > 0 and known[y0:y1 + 1, x0 - 1].mean() >= need:
            x0 -= 1
        if free[3] and x1 < W - 1 and known[y0:y1 + 1, x1 + 1].mean() >= need:
            x1 += 1
    return x0, y0, x1, y1


def find_artworks(arr2d, min_size, max_per_shape=6):
    """Rectangles (x0, y0, x1, y1) of solid known pixels, at least min_size x min_size, free of
    the same painter's other pixels on at least 3 sides."""
    known = arr2d >= 0
    if known.sum() < min_size * min_size * 0.25:
        return []
    pad = 4  # so the closing does not eat into artworks touching the board's edge
    m = np.pad(known, pad)
    closed = ndimage.binary_closing(m, structure=np.ones((3, 3), bool), iterations=3)[pad:-pad, pad:-pad] | known
    lab, _ = ndimage.label(closed)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None or sl[0].stop - sl[0].start < min_size or sl[1].stop - sl[1].start < min_size:
            continue
        shape = lab[sl] == i
        exposed = side_checker(shape)
        free_rect = shape.copy()  # where later rectangles may still go (no overlaps)
        for _ in range(max_per_shape):
            r = largest_rect(free_rect, min_size, min_size, lambda *r: sum(exposed(*r)) >= 3)
            if r is None:
                break
            x0, y0, x1, y1 = r
            free_rect[y0:y1 + 1, x0:x1 + 1] = False
            x0, y0, x1, y1 = grow(r, known[sl], exposed(*r))
            if known[sl][y0:y1 + 1, x0:x1 + 1].mean() >= 0.25:  # at least a quarter really seen
                out.append((sl[1].start + x0, sl[0].start + y0, sl[1].start + x1, sl[0].start + y1))
            if free_rect.sum() < min_size * min_size:
                break
    return out


class Watcher:
    def __init__(self, min_size, reset):
        cfg = requests.get(f"{engine.BASE}/api/config", timeout=20).json()
        self.W, self.H = int(cfg["width"]), int(cfg["height"])
        self.min = min_size
        self.painters, self.events, self.index = {}, 0, {}
        os.makedirs(ART_DIR, exist_ok=True)
        if reset:
            for f in (STATE, INDEX):
                if os.path.exists(f):
                    os.remove(f)
        self.load()

    # ------------------------------------------------------------- state
    def load(self):
        try:
            with np.load(STATE, allow_pickle=False) as z:
                names = [str(n) for n in z["names"]]
                for i, name in enumerate(names):
                    p = Painter(self.W * self.H)
                    for idx, col in zip(z[f"i{i}"].tolist(), z[f"c{i}"].tolist()):
                        p.paint(idx, col)
                    self.painters[name] = p
            print(f"reloaded {sum(p.count for p in self.painters.values()):,} recorded pixels of "
                  f"{len(self.painters)} painters from {STATE}", flush=True)
        except (OSError, KeyError, ValueError):
            pass
        try:
            with open(INDEX) as f:
                for e in json.load(f):
                    self.index.setdefault(e["name"], []).append(e)
        except (OSError, ValueError, KeyError):
            pass

    def save_state(self, snap):
        data = {"names": np.array(list(snap), dtype=str)}
        for i, a in enumerate(snap.values()):
            idx = np.flatnonzero(a >= 0)
            data[f"i{i}"], data[f"c{i}"] = idx.astype(np.int32), a[idx]
        tmp = STATE + ".tmp.npz"
        np.savez_compressed(tmp, **data)
        os.replace(tmp, STATE)

    # ------------------------------------------------------------- detection
    def detect(self, snap):
        """Find the artworks of the painters in snap ({name: board array}); write pictures + index."""
        news = []
        for name, flat in snap.items():
            arr2d = flat.reshape(self.H, self.W)
            old = {tuple(e["rect"]): e for e in self.index.get(name, [])}
            entries = []
            for rect in find_artworks(arr2d, self.min):
                x0, y0, x1, y1 = rect
                sub = arr2d[y0:y1 + 1, x0:x1 + 1]
                seen, area = int((sub >= 0).sum()), sub.size
                png = f"{safe_name(name)}_{x0}_{y0}_{x1 - x0 + 1}x{y1 - y0 + 1}.png"
                prev = old.get(rect)
                if not prev or prev["seen"] != seen or not os.path.exists(os.path.join(ART_DIR, png)):
                    rgba = np.zeros(sub.shape + (4,), np.uint8)
                    v = np.where(sub >= 0, sub, 0)
                    rgba[..., 0], rgba[..., 1], rgba[..., 2] = v >> 16 & 255, v >> 8 & 255, v & 255
                    rgba[..., 3] = np.where(sub >= 0, 255, 0)
                    Image.fromarray(rgba, "RGBA").save(os.path.join(ART_DIR, png))
                entries.append({"name": name, "rect": list(rect), "w": x1 - x0 + 1, "h": y1 - y0 + 1,
                                "seen": seen, "area": area, "complete": seen == area,
                                "png": f"artworks/{png}",
                                "updated": time.strftime("%Y-%m-%d %H:%M:%S") if not prev or prev["seen"] != seen
                                else prev.get("updated")})
                if not prev:
                    news.append(f"new: {name or '(no name)'} {x1 - x0 + 1}x{y1 - y0 + 1} at ({x0},{y0}), "
                                f"{100 * seen / area:.0f}% known")
                elif seen == area and not prev.get("complete"):
                    news.append(f"complete: {name or '(no name)'} {x1 - x0 + 1}x{y1 - y0 + 1} at ({x0},{y0})")
            self.index[name] = entries
        all_entries = [e for es in self.index.values() for e in es]
        tmp = INDEX + ".tmp"
        with open(tmp, "w") as f:
            json.dump(all_entries, f, indent=1)
        os.replace(tmp, INDEX)
        keep = {os.path.basename(e["png"]) for e in all_entries}
        for f in os.listdir(ART_DIR):  # pictures of rectangles that grew or vanished
            if f.endswith(".png") and f not in keep:
                os.remove(os.path.join(ART_DIR, f))
        return news, len(all_entries)

    # ------------------------------------------------------------- live feed
    async def listen(self):
        import websockets
        while True:
            try:
                async with websockets.connect(WS_URL, open_timeout=20, max_size=None, ping_interval=20) as ws:
                    print("connected to the live feed", flush=True)
                    async for msg in ws:
                        for line in str(msg).splitlines():
                            p = line.split(",", 3)
                            if len(p) < 3 or len(p[2]) != 6:
                                continue
                            try:
                                x, y, colour = int(p[0]), int(p[1]), int(p[2], 16)
                            except ValueError:
                                continue
                            if 0 <= x < self.W and 0 <= y < self.H:
                                name = p[3].strip() if len(p) > 3 else ""
                                if name not in self.painters:
                                    self.painters[name] = Painter(self.W * self.H)
                                self.painters[name].paint(y * self.W + x, colour)
                                self.events += 1
            except Exception as e:  # dropped connection: reconnect
                print(f"live feed lost ({e.__class__.__name__}), reconnecting in 5 s", flush=True)
            await asyncio.sleep(5)

    async def run(self, every):
        loop = asyncio.get_running_loop()
        listener = asyncio.create_task(self.listen())
        last_save, last_events, t_last = 0.0, 0, time.time()
        try:
            while True:
                await asyncio.sleep(every)
                snap = {}
                for name, p in self.painters.items():
                    if p.changed and p.count >= self.min * self.min * 0.25:
                        snap[name] = p.array()  # copied on this thread: consistent
                        p.changed = False
                news, total = await loop.run_in_executor(None, self.detect, snap)
                now = time.time()
                rate = (self.events - last_events) / (now - t_last)
                last_events, t_last = self.events, now
                for n in news:
                    print(f"  {n}", flush=True)
                print(f"[{time.strftime('%H:%M:%S')}] {rate:.0f} paints/s, {len(self.painters)} painters, "
                      f"{sum(p.count for p in self.painters.values()):,} pixels recorded, {total} artworks", flush=True)
                if now - last_save >= 60:
                    full = {name: p.array() for name, p in self.painters.items()}
                    await loop.run_in_executor(None, self.save_state, full)
                    last_save = now
        finally:
            listener.cancel()


def main():
    ap = argparse.ArgumentParser(description="Record painters' artworks from the live feed for flag_ui.py.")
    ap.add_argument("--min", type=int, default=80, help="smallest artwork width and height (default 80)")
    ap.add_argument("--every", type=float, default=30, help="seconds between artwork searches (default 30)")
    ap.add_argument("--reset", action="store_true", help="forget everything recorded so far")
    args = ap.parse_args()
    try:
        import websockets  # noqa: F401
    except ImportError:
        raise SystemExit("needs: pip install websockets")
    w = Watcher(args.min, args.reset)
    print(f"watching {WS_URL}; artworks of at least {args.min}x{args.min} go to {INDEX}", flush=True)
    try:
        asyncio.run(w.run(args.every))
    except KeyboardInterrupt:
        w.save_state({name: p.array() for name, p in w.painters.items()})
        print("\nsaved; stopped", flush=True)


if __name__ == "__main__":
    main()
