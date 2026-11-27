#!/usr/bin/env python3
"""Canvas control room: one window to watch the board (https://canvas.bandwidthyt.com), place and
guard our art, copy other painters' artworks, and run the drawing, all from the UI.

    sudo apt install python3-tk          (Tk for Python; macOS/Windows Pythons already include it)
    pip install requests pillow numpy websockets scipy pynacl
    python3 fastpow/build.py             (optional, ~3x faster hashing: needs clang or gcc)
    python3 canvas.py                    # --file guarded.txt, --draw to start drawing at once

Everything runs inside this window:
  - Board view, live: pixels appear as they are painted. Wheel = zoom at the mouse, middle-drag
    or Shift+drag or the arrow keys = move around, "Fit" = whole board.
  - Zones (guarded.txt): drag on the board to add a French flag, croissant flag, white, text or
    chat zone; drag a zone to move it, its corner handle to resize it; the buttons above the
    selected zone switch it on/off or delete it. Outlines: green intact, red damaged, grey off.
  - Drawing: "Enable drawing" paints every enabled zone and keeps repairing it ("Pause drawing"
    stops it). Name, workers, hashing threads, max bits, batch and interval are set in the second
    toolbar and apply on the next round; they are remembered in canvas_settings.json.
  - Artworks panel (right): "Board" = rectangular artworks of other painters, found by watching
    the live feed (each painter's pixels are recorded until their artwork is known, pieces seen
    at different times are merged); "My library" = our own pictures (French flag, croissant
    flag, VIVE LA FRANCE MERCI flag, the Justice picture with its QR code, the cat, the heart,
    the SONY Computer Entertainment logo, plus any image dropped into art/library/). Select one, then "Place a copy..." and click where its
    top-left corner goes (or "Usual place"): it becomes a zone that the drawing paints and keeps.
  - Stocks tab: search a company or symbol (free prices from Yahoo Finance and Nasdaq) and place a
    live widget (price, change and chart over the last week), updated every 10 minutes.
  - Hate scanner (checkbox): reads the text on the board every 20 s and covers hater words with
    small French flags (needs tesseract: sudo apt install tesseract-ocr; pip install pytesseract
    better-profanity).
  - Logs (button): a side panel with what the drawing, the watcher and the scanner do.
The YouTube chat panel stays a separate program: chat_to_board.py.
"""
import argparse, asyncio, collections, ctypes, functools, json, math, os, queue, random, re, struct, \
    subprocess, sys, threading, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import NamedTuple

import numpy as np
import requests
from PIL import Image, ImageFilter

try:  # the window; chat_to_board.py imports this file for the engine only
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk
except ImportError:
    tk = None
try:
    from PIL import ImageTk
except ImportError:  # some Linux packages split it out: python3-pil.imagetk
    ImageTk = None

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://canvas.bandwidthyt.com"
WS_URL = BASE.replace("https://", "wss://") + "/api/ws"
GUARDED = os.path.join(HERE, "guarded.txt")
SETTINGS = os.path.join(HERE, "canvas_settings.json")
ART_DIR = os.path.join(HERE, "artworks")      # watcher state + copies placed from the UI
LIB_DIR = os.path.join(HERE, "art", "library")
PLACED = "artworks/placed"                      # relative to HERE, as written in guarded.txt


# =============================================================================================
# Engine: Argon2id proof of work and painting
# =============================================================================================
# Backends, fastest first: fastpow/ C library (built by fastpow/build.py, specialised for the
# canvas parameters), libsodium through PyNaCl, then argon2-cffi. All give identical hashes.
try:
    from nacl.pwhash import argon2id as _nacl
except ImportError:
    _nacl = None
try:
    from argon2.low_level import hash_secret_raw, Type
except ImportError:
    hash_secret_raw = None


def _load_fast():
    folder = os.path.join(HERE, "fastpow")
    for name in ("libargon2_canvas.so", "libargon2_canvas.dylib", "argon2_canvas.dll"):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        lib.canvas_impl.restype = ctypes.c_char_p
        lib.canvas_search.restype = ctypes.c_int
        lib.canvas_search.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64,
                                      ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        try:  # libraries built before this function existed simply solve on one thread
            lib.canvas_search_stop.restype = ctypes.c_int
            lib.canvas_search_stop.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64,
                                               ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                                               ctypes.POINTER(ctypes.c_int)]
        except AttributeError:
            pass
        lib.canvas_init()
        return lib
    return None


_fast = _load_fast()
_local = threading.local()  # one HTTP session per worker thread
_kib, _max_bits, _name = 1024, 12, "tricolore"


def have_backend():
    return bool(_fast or _nacl or hash_secret_raw)


def backend():
    if _fast and _kib == 1024:
        return f"fastpow C library ({_fast.canvas_impl().decode()})"
    return "libsodium (PyNaCl)" if _nacl else "argon2-cffi" if hash_secret_raw else "none"


def argon2id(pwd, salt, kib=None):
    kib = kib or _kib
    if _nacl:
        return _nacl.kdf(32, pwd, salt, opslimit=1, memlimit=kib * 1024)
    return hash_secret_raw(pwd, salt, time_cost=1, memory_cost=kib, parallelism=1, hash_len=32, type=Type.ID)


TUNING_FILE = os.path.join(HERE, "fastpow", "tuning.json")


def _tuning():
    """fastpow/tuning.json: hash_threads (written by fastpow/build.py), or {}."""
    try:
        with open(TUNING_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def default_workers():
    """Paints in flight: WORKERS env var, else a measured value in tuning.json, else 16 per CPU thread."""
    try:
        return max(1, int(os.environ.get("WORKERS") or _tuning()["workers"]))
    except (KeyError, ValueError, TypeError):
        return min(256, 16 * (os.cpu_count() or 4))


def _hash_threads():
    """How many threads hash at once: HASH_THREADS env var, else tuning.json (on Hyper-Threading
    CPUs one hash per physical core can be faster), else every CPU thread."""
    if os.environ.get("HASH_THREADS"):
        return max(1, int(os.environ["HASH_THREADS"]))
    try:
        return max(1, int(_tuning()["hash_threads"]))
    except (KeyError, ValueError, TypeError):
        return os.cpu_count() or 4


CORES = _hash_threads()
_hash_slots = threading.BoundedSemaphore(CORES)  # workers wait on the network; at most CORES hash
_helpers = None


def set_hash_threads(n):
    """Change how many threads hash at once (paints already hashing finish with the old count)."""
    global CORES, _hash_slots, _helpers
    n = max(1, int(n))
    if n != CORES:
        old = _helpers
        CORES, _hash_slots, _helpers = n, threading.BoundedSemaphore(n), None
        if old:
            old.shutdown(wait=False)


def _helper_pool():
    global _helpers
    if _helpers is None:
        _helpers = ThreadPoolExecutor(CORES, thread_name_prefix="solve")
    return _helpers


def solve_split(prefix, epoch, bits, nonce, ways):
    """One pixel's search shared by `ways` threads (interleaved nonces); all of them stop within
    one hash when any finds a solution, so splitting wastes (almost) nothing."""
    stop, result, slots = ctypes.c_int(0), [], _hash_slots

    def helper(i):
        found, start = ctypes.c_uint64(), nonce + i
        with slots:
            while not stop.value:
                r = _fast.canvas_search_stop(prefix, epoch, bits, start, ways, 1 << 14, ctypes.byref(found),
                                             ctypes.byref(stop))
                if r == 1:
                    result.append(found.value)
                    return
                if r < 0:
                    raise MemoryError("fastpow: could not allocate work memory")
                start += ways << 14

    for f in [_helper_pool().submit(helper, i) for i in range(ways)]:
        f.result()
    return result[0]


def solve(epoch, x, y, rgb, bits, kib=None):
    """A nonce whose hash has `bits` leading zero bits. kib: the challenge's Argon2 memory (the
    server may change it at any epoch); the fastpow library only does 1024 KiB, the others any."""
    kib = kib or _kib
    fast = _fast and kib == 1024
    prefix = struct.pack("<HH", x, y) + rgb
    nonce = random.getrandbits(62)  # random start: a repeat paint must not find a used solution
    # pricey pixels are shared by several cores: 1 thread at 8 bits, 2 at 9, 4 at 10 ... all cores
    ways = min(CORES, 1 << max(0, bits - 8))
    if fast and ways > 1 and hasattr(_fast, "canvas_search_stop"):
        return solve_split(prefix, epoch, bits, nonce, ways)
    with _hash_slots:
        if fast:  # whole search loop in C, 65536 nonces per call
            found = ctypes.c_uint64()
            while True:
                r = _fast.canvas_search(prefix, epoch, bits, nonce, 1 << 16, ctypes.byref(found))
                if r == 1:
                    return found.value
                if r < 0:
                    raise MemoryError("fastpow: could not allocate work memory")
                nonce += 1 << 16
        limit = 1 << (256 - bits)
        while True:
            if int.from_bytes(argon2id(prefix + struct.pack("<Q", nonce), epoch, kib), "big") < limit:
                return nonce
            nonce += 1


def _init(kib, max_bits, name):
    global _kib, _max_bits, _name
    _kib, _max_bits, _name = kib, max_bits, name


def _sess():
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    return s


_warned = set()


def _warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(f"WARNING: {msg}", flush=True)


def paint_one(task):
    """Paint one pixel (x, y, 'rrggbb'): (x, y, status, bits), status ok / deferred (pricier than
    max bits: it cools down if left alone) / refused / fail."""
    x, y, color = task
    x, y = int(x), int(y)  # numpy integers are not JSON serialisable
    rgb = bytes.fromhex(color)
    session = _sess()
    for attempt in range(12):
        try:
            c = session.get(f"{BASE}/api/challenge", params={"x": x, "y": y}, timeout=20).json()
            if c["bits"] > _max_bits:
                return (x, y, "deferred", c["bits"])
            kib = int(c.get("kib") or _kib)  # may change at any epoch: hashing with another is refused
            if kib != 1024 and not (_nacl or hash_secret_raw):
                _warn_once(f"the server now wants {kib} KiB hashes: the fastpow library only does 1024; "
                           f"pip install pynacl to paint meanwhile")
                return (x, y, "deferred", c["bits"])
            nonce = solve(bytes.fromhex(c["epoch"]), x, y, rgb, c["bits"], kib)
            r = session.post(f"{BASE}/api/paint", timeout=20,
                             json={"x": x, "y": y, "color": color, "epoch": c["epoch"], "nonce": nonce, "name": _name})
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(min(30, 2 ** attempt))
            continue
        if r.status_code == 200:
            return (x, y, "ok", c["bits"])
        if r.status_code == 409:  # epoch rotated or price rose while we worked: ask again
            continue
        if r.status_code == 429:
            time.sleep(61)
            continue
        if r.status_code == 403:  # board closed, or a short refusal: leave it for a later round
            return (x, y, "refused", 0)
        time.sleep(1)
    return (x, y, "fail", 0)


def fetch_board():
    """(width, raw RGB bytes) of the whole board."""
    import gzip
    r = requests.get(f"{BASE}/api/board", timeout=60)
    r.raise_for_status()
    w = int(r.headers.get("x-canvas-size", "1024x1024").split("x")[0])
    data = r.content
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return w, data


def board_array():
    w, raw = fetch_board()
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3)


def damaged(pixels, board=None):
    """The pixels [(x, y, colour)] that the board does not show yet."""
    if board is None:
        w, raw = fetch_board()
        return [(x, y, c) for x, y, c in pixels if raw[(y * w + x) * 3:(y * w + x) * 3 + 3] != bytes.fromhex(c)]
    return [(x, y, c) for x, y, c in pixels if board[y, x].tobytes() != bytes.fromhex(c)]


def prioritize(tasks):
    """Densely damaged areas (a pasted picture or text) before isolated noise dots, top to bottom
    within each density band, so a pasted message is usually wiped from its first line down."""
    bad = {(x, y) for x, y, _ in tasks}
    near = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]

    def key(t):
        dens = sum((t[0] + dx, t[1] + dy) in bad for dx, dy in near)
        return (-(dens // 5), t[1], t[0])
    return sorted(tasks, key=key)


def run_pass(pool, tasks, label):
    """Paint tasks with a thread pool, printing progress (used by chat_to_board.py)."""
    total, counts, t0 = len(tasks), {}, time.time()
    print(f"{label}: {total} pixels to paint", flush=True)
    for i, fut in enumerate(as_completed([pool.submit(paint_one, t) for t in tasks]), 1):
        status = fut.result()[2]
        counts[status] = counts.get(status, 0) + 1
        if i % 250 == 0 or i == total:
            el = time.time() - t0
            print(f"  {i}/{total} {counts} {i / el:.1f} px/s eta {(total - i) / (i / el):.0f}s", flush=True)
    return counts


# =============================================================================================
# Croissant flag (procedural, any width)
# =============================================================================================
FLAG = ["002395", "ffffff", "ed2939"]  # official French blue, white, red
# 1 outline, 2 shadow, 3 base, 4 light, 5 highlight
CROISSANT_COLORS = {1: "4a2a0e", 2: "9c5518", 3: "cf8429", 4: "eaae4c", 5: "fbe0a0"}
# Tilted croissant rising to the right, in coordinates of a 147x98 flag: a crescent along an arc
# of radius R around (cx, cy), tapering at both tips, cut into n rolled bands shaded like
# cylinders, lit from the top-left.
CROISSANT = dict(cx=80, cy=72, R=40, phi=92, h=15, n=7, rot=-22, band_warp=1.18, taper=1.8, curve=0.28)


def _sample(X, Y, c, k, lw):
    """Palette index at a sub-pixel position (0 = outside). k: flag scale, lw: line width."""
    dx, dy = X - c["cx"] * k, Y - c["cy"] * k
    phi = math.atan2(dx, -dy) - math.radians(c["rot"])
    phi = (phi + math.pi) % (2 * math.pi) - math.pi
    PHI = math.radians(c["phi"])
    if abs(phi) >= PHI:
        return 0
    R = c["R"] * k
    rho = math.hypot(dx, dy)
    t = phi / PHI                                            # -1 .. 1 along the body
    h = c["h"] * k * (1 - abs(t) ** c["taper"]) ** 1.1
    g = math.copysign(abs(t) ** c["band_warp"], t)           # middle bands wider
    nrm0 = max(-1.0, min(1.0, (rho - R) / max(h, 0.5 * k)))
    # band boundaries bow towards the tips at the edges: rolled-dough look
    pos = (g + 1) / 2 * c["n"] + c["curve"] * (1 - nrm0 ** 2) * -math.copysign(1, t) * (abs(t) > 0.08)
    f = pos - math.floor(pos)                                # 0..1 inside a band
    h_eff = h * (0.80 + 0.20 * math.sin(math.pi * f)) + 0.3 * k
    off = rho - R
    if abs(off) > h_eff:
        return 0
    if abs(off) > h_eff - 1.15 * lw:                         # outline
        return 1
    dpos = min(f, 1 - f)                                     # separator between bands
    band_px = 2 * PHI * R / c["n"] * (0.6 if abs(t) > 0.6 else 1.0)
    if dpos * band_px < 0.75 * lw and abs(off) < h_eff - 1.5 * lw:
        return 1 if dpos * band_px < 0.45 * lw else 2
    nrm = off / h_eff                                        # -1 inner .. +1 outer
    sc = 0.75 * math.sin(math.pi * f) ** 0.8 + 0.45 * nrm - 0.25 * (f - 0.5) - 0.2 * t - 0.2
    return 5 if sc > 0.62 else 4 if sc > 0.3 else 3 if sc > -0.05 else 2


@functools.lru_cache(maxsize=16)
def croissant_pixels(width, ss=4):
    """{(dx, dy): colour} of the croissant relative to the flag's top-left corner, for a flag
    `width` wide (and 2/3 as high)."""
    c = CROISSANT
    k = width / 147
    lw = max(1.0, 0.8 * k)
    height = round(width * 2 / 3)
    R, hmax, PHI = c["R"] * k, c["h"] * k + 0.3 * k, math.radians(c["phi"])
    subs = [(i + 0.5) / ss for i in range(ss)]
    out = {}
    for y in range(height):
        for x in range(width):
            dx, dy = x + 0.5 - c["cx"] * k, y + 0.5 - c["cy"] * k
            if abs(math.hypot(dx, dy) - R) > hmax + 2:      # quick reject: far from the body
                continue
            phi = (math.atan2(dx, -dy) - math.radians(c["rot"]) + math.pi) % (2 * math.pi) - math.pi
            if abs(phi) > PHI + 2 / R:
                continue
            votes = [_sample(x + ox, y + oy, c, k, lw) for oy in subs for ox in subs]
            nz = [v for v in votes if v]
            if len(nz) < len(votes) / 2:
                continue
            v = 1 if votes.count(1) / len(votes) >= 0.3 else max(set(nz), key=nz.count)
            out[(x, y)] = CROISSANT_COLORS[v]
    return out


# =============================================================================================
# Zones (guarded.txt): what the drawing paints and keeps
# =============================================================================================
KINDS = ("flag", "white", "text", "chat", "art", "croissant", "stock")
OPT_KEYS = ("fg", "ol", "bg", "file", "sym", "src", "cls")
TEXT_DEFAULTS = {"fg": "ffffff", "ol": "000000", "bg": "none"}
CHAT_LINES = 6  # messages chat_to_board.py shows; a chat zone is that many 8 px lines (+1) high
GUARDED_HELP = """\
# Zones painted and kept by canvas.py (and the chat panel of chat_to_board.py), one per line,
# board coordinates (inclusive); everything after '#' is a comment:
#   x0 y0 x1 y1 [off] flag|white|croissant note           French flag, plain white, flag with croissant
#   x0 y0 x1 y1 [off] text fg=ffffff ol=000000 bg=none TEXT  text as big as fits ("|" = new line)
#   x0 y0 x1 y1 [off] art file=artworks/placed/X.png note    a picture, top-left corner at x0 y0
#   x0 y0 x1 y1 [off] stock sym=AAPL src=yahoo note           a live stock widget (price, last week)
#   x0 y0 x1 y1 [off] chat note                             where chat_to_board.py shows the chat
# "off" keeps a zone listed without painting it. Where zones overlap, the later line wins.
"""


class Zone(NamedTuple):
    rect: tuple       # (x0, y0, x1, y1), inclusive
    note: str         # the text of a text zone, a free note otherwise
    kind: str = "flag"
    opts: dict = {}
    enabled: bool = True  # False: listed ("off") but not painted

    def key(self):
        return (self.rect, self.note, self.kind, tuple(sorted(self.opts.items())), self.enabled)


def parse_line(line):
    parts = line.split("#")[0].split()
    try:
        x0, y0, x1, y1 = (int(v) for v in parts[:4])
    except ValueError:
        return None
    if x0 > x1 or y0 > y1:
        return None
    rest, kind, opts, enabled = parts[4:], "flag", {}, True
    if rest and rest[0].lower() == "off":
        enabled = False
        rest.pop(0)
    if rest and rest[0].lower() in KINDS:
        kind = rest.pop(0).lower()
    while rest and "=" in rest[0] and rest[0].split("=")[0].lower() in OPT_KEYS:
        k, v = rest.pop(0).split("=", 1)
        opts[k.lower()] = v if k.lower() in ("file", "sym") else v.lower()  # file names, symbols keep their case
    return Zone((x0, y0, x1, y1), " ".join(rest), kind, opts, enabled)


def load_guarded(path=GUARDED):
    """[Zone] from guarded.txt; unreadable lines are skipped."""
    try:
        with open(path) as f:
            return [z for z in map(parse_line, f) if z]
    except FileNotFoundError:
        return []


def add_guarded(rect, note, path=GUARDED, kind="flag", opts=None):
    head = GUARDED_HELP if not os.path.exists(path) else ""
    words = [str(v) for v in rect] + ([] if kind == "flag" else [kind])
    words += [f"{k}={v}" for k, v in (opts or {}).items()] + [note.replace("#", " ").strip() or "manual"]
    with open(path, "a") as f:  # one write per line, so a reader never sees half a line
        f.write(f"{head}{' '.join(words)}  # {time.strftime('%Y-%m-%d %H:%M')}\n")


def _rect_of(line):
    try:
        return tuple(int(v) for v in line.split("#")[0].split()[:4])
    except ValueError:
        return None


def _rewrite(path, change):
    """Apply change(lines) to the file: read just before (keeps lines another program appended)
    and replace it atomically (no reader ever sees half a file)."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(change(lines))
    os.replace(tmp, path)


def remove_guarded(rect, path=GUARDED):
    _rewrite(path, lambda lines: [l for l in lines if _rect_of(l) != tuple(rect)])


def move_guarded(old, new, path=GUARDED):
    """Give the zone at `old` the rectangle `new` (moved or resized), keeping the rest of its line."""
    def change(lines):
        out, done = [], False
        for line in lines:
            if not done and _rect_of(line) == tuple(old):
                body, sep, comment = line.partition("#")
                line = " ".join([str(v) for v in new] + body.split()[4:]) + (f"  #{comment}" if sep else "\n")
                done = True
            out.append(line)
        return out
    _rewrite(path, change)


def set_enabled(rect, enabled, path=GUARDED):
    """Switch painting of the zone at `rect` on or off (an "off" word after its coordinates)."""
    def change(lines):
        out = []
        for line in lines:
            if _rect_of(line) == tuple(rect):
                body, sep, comment = line.partition("#")
                words = body.split()
                rest = words[5:] if words[4:5] and words[4].lower() == "off" else words[4:]  # only the marker
                line = " ".join(words[:4] + ([] if enabled else ["off"]) + rest) + (f"  #{comment}" if sep else "\n")
            out.append(line)
        return out
    _rewrite(path, change)


def overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def flag_target(rect, kind="flag"):
    """[(x, y, colour)]: a French flag (plain white for "white", with a croissant for "croissant")."""
    x0, y0, x1, y1 = rect
    w = x1 - x0 + 1
    if kind == "white":
        return [(x, y, "ffffff") for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    art = croissant_pixels(w) if kind == "croissant" else {}
    return [(x, y, art.get((x - x0, y - y0)) or FLAG[min(2, 3 * (x - x0) // w)])
            for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


@functools.lru_cache(maxsize=1)
def _font_path():
    for name in ("DejaVu Sans:bold", "Liberation Sans:bold", "Arial:bold"):
        try:
            path = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True,
                                  timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            break
        if path and os.path.exists(path):
            return path
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"):
        if os.path.exists(path):
            return path
    return None


@functools.lru_cache(maxsize=512)
def text_masks(text, w, h, outline):
    """(letters, ring): sets of (dx, dy) for `text` ("|" = new line) in the largest bold size that
    fits w x h with its outline, centred, in crisp pixels (no anti-aliasing)."""
    from PIL import ImageDraw, ImageFont
    lines, path = "\n".join(text.split("|")), _font_path()

    def font_for(size):
        if path:
            return ImageFont.truetype(path, size)
        try:
            return ImageFont.load_default(size)
        except TypeError:  # Pillow < 10.1: fixed-size default font
            return ImageFont.load_default()

    def layout(size):
        font, ow = font_for(size), (max(1, size // 16) if outline else 0)
        box = ImageDraw.Draw(Image.new("L", (1, 1))).multiline_textbbox(
            (0, 0), lines, font=font, stroke_width=ow, align="center", spacing=max(1, size // 8))
        return font, ow, size, box

    lo, hi, best = 4, max(4, 2 * h), None
    while lo <= hi:  # binary search for the biggest size that fits
        mid = (lo + hi) // 2
        font, ow, size, box = layout(mid)
        if box[2] - box[0] <= w and box[3] - box[1] <= h:
            best, lo = (font, ow, size, box), mid + 1
        else:
            hi = mid - 1
    font, ow, size, box = best or layout(4)  # box too small: smallest size, cropped
    ox, oy = (w - (box[2] - box[0])) // 2 - box[0], (h - (box[3] - box[1])) // 2 - box[1]
    im = Image.new("L", (w, h), 0)  # letters (255) and outline (128) in one call: they line up
    d = ImageDraw.Draw(im)
    d.fontmode = "1"
    d.multiline_text((ox, oy), lines, font=font, fill=255, align="center", spacing=max(1, size // 8),
                     stroke_width=ow, stroke_fill=128)
    px = list(im.getdata())
    letters = frozenset((i % w, i // w) for i, v in enumerate(px) if v > 191)
    ring = frozenset((i % w, i // w) for i, v in enumerate(px) if 64 < v <= 191)
    return letters, ring


@functools.lru_cache(maxsize=64)
def _picture(path, mtime):
    """RGBA uint8 array of a picture file (cached until the file changes)."""
    return np.asarray(Image.open(path).convert("RGBA"))


def art_path(name):
    """A zone's file= value, relative to this script's folder unless absolute."""
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def art_target(zone):
    """Pixels of an art zone's picture (alpha >= 128), top-left at the zone's corner."""
    path = art_path(zone.opts.get("file", ""))
    try:
        img = _picture(path, os.path.getmtime(path))
    except (OSError, ValueError):
        return []
    x0, y0, x1, y1 = zone.rect
    img = img[:y1 - y0 + 1, :x1 - x0 + 1]
    ys, xs = np.nonzero(img[..., 3] >= 128)
    return [(x0 + int(x), y0 + int(y), "%02x%02x%02x" % tuple(c)) for x, y, c in zip(xs, ys, img[ys, xs, :3].tolist())]


def zone_target(zone, wait=False):
    """[(x, y, colour)] the zone should show. wait: a stock widget may wait for its data."""
    if zone.kind == "chat":  # chat_to_board.py paints it
        return []
    if zone.kind == "art":
        return art_target(zone)
    if zone.kind == "stock":
        return stock_target(zone, wait)
    if zone.kind != "text":
        return flag_target(zone.rect, zone.kind)
    o = {**TEXT_DEFAULTS, **zone.opts}
    x0, y0, x1, y1 = zone.rect
    w, h = x1 - x0 + 1, y1 - y0 + 1
    letters, ring = text_masks(zone.note or " ", w, h, o["ol"] != "none")
    out = []
    for dy in range(h):
        for dx in range(w):
            if (dx, dy) in letters:
                c = o["fg"]
            elif (dx, dy) in ring:
                c = o["ol"]
            elif o["bg"] != "none":
                c = o["bg"]
            else:
                continue
            out.append((x0 + dx, y0 + dy, c))
    return out


def zones_pixels(path=GUARDED):
    """[(x, y, colour)] of every enabled zone; where zones overlap the later one wins."""
    want = {}
    for zone in load_guarded(path):
        if zone.enabled:
            for x, y, c in zone_target(zone, wait=True):
                want[(x, y)] = c
    return [(x, y, c) for (x, y), c in want.items()]


# =============================================================================================
# My library: our own pictures, placed from the UI
# =============================================================================================
LIBRARY = [  # kind: flag / croissant (drawn at any size) or image (file in art/library)
    dict(id="flag", title="French flag", kind="flag", width=150),
    dict(id="croissant", title="French flag with croissant", kind="croissant", width=237, place=(787, 98)),
    dict(id="merci", title="VIVE LA FRANCE MERCI flag", kind="image", file="merci.png", width=320, place=(105, 62)),
    dict(id="duo", title="Justice, with QR code", kind="image", file="duo.png", width=252, place=(743, 256)),
    dict(id="cat", title="Cat", kind="image", file="cat.jpg", width=250),
    dict(id="sony", title="SONY Computer Entertainment logo", kind="image", file="sony.png", width=144, bg="ffffff"),
    dict(id="heart", title="Tricolour heart", kind="image", file="heart.png", width=20, place=(1001, 494)),
]


def library_items():
    """The built-in items, plus every other picture found in art/library/ (at its own size, at most 200 wide)."""
    items, known = [dict(d) for d in LIBRARY], {d.get("file") for d in LIBRARY}
    try:
        files = sorted(os.listdir(LIB_DIR))
    except OSError:
        files = []
    for f in files:
        if f not in known and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
            try:
                w = Image.open(os.path.join(LIB_DIR, f)).size[0]
            except OSError:
                continue
            items.append(dict(id=f, title=os.path.splitext(f)[0], kind="image", file=f, width=min(200, w)))
    return items


def library_size(item, width):
    """(w, h) of an item drawn `width` wide."""
    if item["kind"] in ("flag", "croissant"):
        return width, round(width * 2 / 3)
    w, h = Image.open(os.path.join(LIB_DIR, item["file"])).size
    return width, max(1, round(h * width / w))


@functools.lru_cache(maxsize=32)
def library_has_alpha(file):
    """Does this library picture have transparent parts?"""
    try:
        return bool((np.asarray(Image.open(os.path.join(LIB_DIR, file)).convert("RGBA"))[..., 3] < 255).any())
    except OSError:
        return False


@functools.lru_cache(maxsize=32)
def _library_render(item_key, width, bg):
    kind, file = item_key
    if kind in ("flag", "croissant"):
        h = round(width * 2 / 3)
        a = np.zeros((h, width, 3), np.uint8)
        for x, y, c in flag_target((0, 0, width - 1, h - 1), kind):
            a[y, x] = list(bytes.fromhex(c))
        return a
    im = Image.open(os.path.join(LIB_DIR, file)).convert("RGBA")
    if bg:  # transparent parts filled with this colour
        im = Image.alpha_composite(Image.new("RGBA", im.size, "#" + bg), im)
    if im.size[0] != width:  # smooth resize, colours lightly sharpened so details stay crisp
        im = im.resize(library_size({"kind": kind, "file": file}, width), Image.LANCZOS)
        rgb = im.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
        im = Image.merge("RGBA", (*rgb.split(), im.getchannel("A")))
    a = np.asarray(im)
    return a if (a[..., 3] < 255).any() else np.ascontiguousarray(a[..., :3])


def library_render(item, width, bg=None):
    """The item drawn `width` wide: RGB uint8 array, or RGBA where it stays transparent (those
    pixels are not painted: the board shows through). bg: 'rrggbb' to fill transparent parts."""
    return _library_render((item["kind"], item.get("file")), int(width), bg or None)


# =============================================================================================
# Stocks: a live widget (price, change and chart over the last week), from public APIs
# =============================================================================================
# Two free sources, no key needed: Yahoo Finance (any market: AAPL, AIR.PA, ^FCHI, BTC-USD...;
# it refuses some networks) and Nasdaq (US stocks, ETFs, some indexes and crypto). Search asks
# both; a US symbol falls back to Nasdaq when Yahoo refuses.
STOCK_REFRESH = 600  # seconds between updates of a widget's data
_WEB = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 "
                      "Safari/537.36", "Accept": "application/json"}


def _num(text):
    return float(str(text).replace("$", "").replace(",", "").strip())


def search_stocks(query):
    """([dict(sym, name, exch, kind, src, cls)], [problems]) for a company name or symbol."""
    found, problems = [], []
    try:
        r = requests.get("https://query1.finance.yahoo.com/v1/finance/search", headers=_WEB, timeout=15,
                         params={"q": query, "quotesCount": 12, "newsCount": 0})
        if r.status_code != 200:
            raise ValueError(f"answered {r.status_code}")
        for q in r.json().get("quotes", []):
            if q.get("symbol"):
                found.append(dict(sym=q["symbol"], name=q.get("shortname") or q.get("longname") or "",
                                  exch=q.get("exchDisp", ""), kind=q.get("quoteType", "").lower(), src="yahoo", cls=""))
    except Exception as e:
        problems.append(f"Yahoo {e}")
    try:
        r = requests.get("https://api.nasdaq.com/api/autocomplete/slookup/12", headers=_WEB, timeout=15,
                         params={"search": query})
        if r.status_code != 200:
            raise ValueError(f"answered {r.status_code}")
        for d in r.json().get("data") or []:
            if d.get("symbol"):
                found.append(dict(sym=d["symbol"], name=d.get("name", ""), exch=d.get("exchange") or "",
                                  kind=(d.get("asset") or "").lower(), src="nasdaq", cls=(d.get("asset") or "stocks").lower()))
    except Exception as e:
        problems.append(f"Nasdaq {e}")
    seen, out = set(), []
    for f in found:  # the same symbol from both: keep the first (Yahoo: finer chart)
        if f["sym"].upper() not in seen:
            seen.add(f["sym"].upper())
            out.append(f)
    return out, problems


def yahoo_chart(sym):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}", headers=_WEB,
                     timeout=15, params={"range": "5d", "interval": "30m"})
    if r.status_code != 200:
        raise ValueError(f"Yahoo answered {r.status_code}")
    res = r.json()["chart"]["result"][0]
    meta, ts = res["meta"], res.get("timestamp") or []
    pts = [(t, c) for t, c in zip(ts, res["indicators"]["quote"][0]["close"]) if c is not None]
    if not pts:
        raise ValueError("no prices")
    price = float(meta.get("regularMarketPrice") or pts[-1][1])
    return dict(sym=meta.get("symbol", sym), name=meta.get("shortName") or meta.get("longName") or sym,
                currency=meta.get("currency") or "", price=price, points=[p for _, p in pts] + [price],
                asof=time.strftime("%m/%d %H:%M", time.localtime(meta.get("regularMarketTime") or pts[-1][0])),
                src="yahoo")


def nasdaq_chart(sym, cls="stocks"):
    import datetime
    today = datetime.date.today()
    r = requests.get(f"https://api.nasdaq.com/api/quote/{sym}/chart", headers=_WEB, timeout=15,
                     params={"assetclass": cls or "stocks", "fromdate": (today - datetime.timedelta(days=7)).isoformat(),
                             "todate": today.isoformat()})
    r.raise_for_status()
    d = r.json().get("data")
    if not d:
        raise ValueError("Nasdaq does not know this symbol")
    closes = [float(p["y"]) for p in d.get("chart") or []]  # daily closes of the last week
    price = _num(d.get("lastSalePrice") or (closes[-1] if closes else "nan"))
    return dict(sym=d.get("symbol", sym), name=d.get("company") or sym, currency="USD",
                price=price, points=closes + [price], asof=(d.get("timeAsOf") or "").replace(" ET", "").strip(),
                src="nasdaq")


def fetch_stock(sym, src="yahoo", cls=""):
    """dict(sym, name, currency, price, points, asof, src): price now and over the last week."""
    if src == "yahoo":
        try:
            return yahoo_chart(sym)
        except Exception:
            if not re.fullmatch(r"[A-Z]{1,5}", sym):  # not a plain US symbol: nothing to fall back to
                raise
    return nasdaq_chart(sym, cls or "stocks")


class StockFeed:
    """Cached stock data, refreshed in the background every STOCK_REFRESH s (after a failure the
    old data stays and a new try comes after a minute). version changes at each update."""

    def __init__(self):
        self.cache, self.pending, self.errors, self.version = {}, {}, {}, 0
        self.lock = threading.Lock()

    def data(self, sym, src="yahoo", cls="", wait=False):
        key = (sym, src, cls)
        with self.lock:
            when, d = self.cache.get(key, (0, None))
            if time.time() - when > (STOCK_REFRESH if d else 60) and key not in self.pending:
                t = self.pending[key] = threading.Thread(target=self._fetch, args=(key,), daemon=True)
                t.start()
            t = self.pending.get(key)
        if wait and t:
            t.join(20)
        with self.lock:
            return self.cache.get(key, (0, None))[1]

    def stamp(self, sym, src="yahoo", cls=""):
        return self.cache.get((sym, src, cls), (0, None))[0]

    def _fetch(self, key):
        try:
            d, err = fetch_stock(*key), None
        except Exception as e:
            d, err = None, f"{e.__class__.__name__}: {e}"
        with self.lock:
            old = self.cache.get(key, (0, None))[1]
            self.cache[key] = (time.time(), d or old)
            self.errors[key] = err
            self.pending.pop(key, None)
            self.version += 1


STOCKS = StockFeed()
UP, DOWN, INK, DIM, BG, EDGE = (63, 185, 80), (248, 81, 73), (230, 237, 243), (139, 148, 158), (13, 17, 23), (48, 54, 61)


def _fmt_price(p):  # no thousands separators: in a 5x7 font a comma reads like a decimal point
    return f"{p:.0f}" if p >= 10000 else f"{p:.2f}" if p >= 1 else f"{p:.4f}"


def _draw_text(img, x, y, text, colour, s=1):
    """pixfont text, s times bigger; returns the x after it."""
    import pixfont
    if s == 1:
        return pixfont.draw(img, x, y, text, colour)
    w = pixfont.width(text)
    m = np.zeros((pixfont.HEIGHT, max(1, w), 3), np.uint8)
    pixfont.draw(m, 0, 0, text, (255, 255, 255))
    m = m.repeat(s, 0).repeat(s, 1)[..., 0] > 0
    H, W = img.shape[:2]
    m = m[:max(0, H - y), :max(0, W - x)]
    img[y:y + m.shape[0], x:x + m.shape[1]][m] = colour
    return x + w * s


def _line(img, x0, y0, x1, y1, colour, t=1):
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(n + 1):
        x, y = round(x0 + (x1 - x0) * i / n), round(y0 + (y1 - y0) * i / n)
        img[max(0, y - t // 2):y + (t + 1) // 2, max(0, x - t // 2):x + (t + 1) // 2] = colour


def render_stock(d, w, h):
    """The widget, w x h: symbol and price, the week's change, and the week's chart."""
    import pixfont
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    img[0, :], img[-1, :], img[:, 0], img[:, -1] = EDGE, EDGE, EDGE, EDGE
    s = 2 if h >= 100 and w >= 220 else 1
    lh = (pixfont.HEIGHT + 2) * s
    pts = [p for p in d["points"] if p == p] or [d["price"]]
    change = d["price"] - pts[0]
    pct = 100 * change / pts[0] if pts[0] else 0.0
    col = UP if change >= 0 else DOWN
    price = _fmt_price(d["price"])
    cur = f" {d['currency']}" if d.get("currency") and d["currency"] != "USD" else ""
    x = _draw_text(img, 3, 3, pixfont.to_ascii(d["sym"])[:12], INK, s)
    if pixfont.width(price + cur) * s > w - 3 - (x + 4):
        cur = ""  # no room for the currency
    px = max(x + 4, w - 3 - pixfont.width(price + cur) * s)  # price on the right
    _draw_text(img, _draw_text(img, px, 3, price, INK, s), 3, cur, DIM, s)
    y = 3 + lh
    for r in range(3 * s):  # a small triangle: up or down
        half = r if change >= 0 else 3 * s - 1 - r
        img[y + 2 * s + r, 3 + 2 * s - half:3 + 2 * s + half + 1] = col
    ax = 3
    _draw_text(img, ax + 6 * s, y, f"{change:+.2f} {pct:+.2f}% 7d" if w >= 150 * s else f"{pct:+.2f}% 7d", col, s)
    y += lh
    if h >= 76 * s and d.get("name"):
        name = pixfont.to_ascii(d["name"])
        while name and pixfont.width(name) * s > w - 6:
            name = name[:-1]
        _draw_text(img, 3, y, name, DIM, s)
        y += lh
    top, bottom, left, right = y + 1, h - 4, 3, w - 4
    if bottom - top >= 8 and len(pts) >= 2:
        lo, hi = min(pts), max(pts)
        pad = (hi - lo) * 0.1 or abs(hi) * 0.01 or 1
        lo, hi = lo - pad, hi + pad
        X = [left + (right - left) * i / (len(pts) - 1) for i in range(len(pts))]
        Y = [bottom - (bottom - top) * (p - lo) / (hi - lo) for p in pts]
        fill = tuple(int(c * 0.28 + b * 0.72) for c, b in zip(col, BG))
        for xi in range(left, right + 1):  # area under the line
            k = min(len(X) - 2, max(0, int((xi - left) / max(1e-9, right - left) * (len(X) - 1))))
            f = (xi - X[k]) / max(1e-9, X[k + 1] - X[k])
            yy = round(Y[k] + (Y[k + 1] - Y[k]) * min(1, max(0, f)))
            img[yy:bottom + 1, xi] = fill
        y0 = round(bottom - (bottom - top) * (pts[0] - lo) / (hi - lo))  # where the week started
        img[y0, left:right + 1:3] = DIM
        for i in range(len(X) - 1):
            _line(img, round(X[i]), round(Y[i]), round(X[i + 1]), round(Y[i + 1]), col, s)
        img[max(0, round(Y[-1]) - s):round(Y[-1]) + s + 1, max(0, right - s):right + s + 1] = INK
    return img


def stock_target(zone, wait=False):
    o = zone.opts
    d = STOCKS.data(o.get("sym", ""), o.get("src", "yahoo"), o.get("cls", ""), wait)
    if not d:
        return []
    x0, y0, x1, y1 = zone.rect
    img = render_stock(d, x1 - x0 + 1, y1 - y0 + 1)
    hexes = [bytes(c).hex() for c in img.reshape(-1, 3).tolist()]
    w = x1 - x0 + 1
    return [(x0 + i % w, y0 + i // w, c) for i, c in enumerate(hexes)]


# =============================================================================================
# Artwork watcher: other painters' rectangular artworks, from the live feed
# =============================================================================================
class Recorder:
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


def maximal_rects(mask, min_w, min_h, accept):
    """Every maximal all-True rectangle (can grow in no direction) at least min_w x min_h for
    which accept(x0, y0, x1, y1) is true, as (x0, y0, x1, y1) inclusive (histogram method)."""
    H, W = mask.shape
    rowc = np.zeros((H, W + 1), np.int64)
    rowc[:, 1:] = mask.cumsum(1)
    heights = np.zeros(W, np.int64)
    found = []
    for y in range(H):
        heights = np.where(mask[y], heights + 1, 0)
        if heights.max() < min_h:
            continue
        stack = []  # (start x, height), heights strictly increasing
        for x, h in enumerate(heights.tolist() + [0]):
            start = x
            while stack and stack[-1][1] > h:
                sx, sh = stack.pop()
                # widest at this height; maximal downwards if the row below is not full under it
                if sh >= min_h and x - sx >= min_w and (y == H - 1 or rowc[y + 1, x] - rowc[y + 1, sx] < x - sx):
                    r = (sx, y - sh + 1, x - 1, y)
                    if accept(*r):
                        found.append(r)
                start = sx
            if h and (not stack or stack[-1][1] < h):
                stack.append((start, h))
    return found


GAP = 4  # how far outside a side we look for "nothing of the same shape there"


def side_checker(shape):
    """exposed(rect) -> [top, bottom, left, right]: True where the line GAP px outside that side
    is (almost) free of `shape`."""
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


def _rgb_diff(a, b):
    """Mean per-channel difference of two arrays of 0xRRGGBB colours."""
    return (np.abs((a >> 16 & 255) - (b >> 16 & 255)) + np.abs((a >> 8 & 255) - (b >> 8 & 255))
            + np.abs((a & 255) - (b & 255))) / 3.0


def edge_checker(cols):
    """edges(rect) -> [top, bottom, left, right]: True where the colours change sharply across that
    side (the painter's colours just outside vs just inside), as at the border of a picture pasted
    over the same painter's other art. cols: the painter's colours (-1 = not seen)."""
    H, W = cols.shape

    def one(inside, outside, inner):
        ok = (inside >= 0) & (outside >= 0) & (inner >= 0)
        if ok.mean() < 0.3:
            return False
        d_out = _rgb_diff(inside[ok], outside[ok]).mean()
        d_in = _rgb_diff(inside[ok], inner[ok]).mean()
        return d_out >= max(25.0, 2 * d_in + 10)

    def edges(x0, y0, x1, y1):
        if y1 - y0 < 2 or x1 - x0 < 2:
            return [False] * 4
        return [y0 > 0 and one(cols[y0, x0:x1 + 1], cols[y0 - 1, x0:x1 + 1], cols[y0 + 1, x0:x1 + 1]),
                y1 < H - 1 and one(cols[y1, x0:x1 + 1], cols[y1 + 1, x0:x1 + 1], cols[y1 - 1, x0:x1 + 1]),
                x0 > 0 and one(cols[y0:y1 + 1, x0], cols[y0:y1 + 1, x0 - 1], cols[y0:y1 + 1, x0 + 1]),
                x1 < W - 1 and one(cols[y0:y1 + 1, x1], cols[y0:y1 + 1, x1 + 1], cols[y0:y1 + 1, x1 - 1])]
    return edges


def grow(rect, known, free):
    """Push the free sides out over ragged edges (a half-painted artwork's border rows)."""
    x0, y0, x1, y1 = rect
    H, W = known.shape
    need = max(0.1, 0.5 * known[y0:y1 + 1, x0:x1 + 1].mean())
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


def find_pieces(arr2d, min_piece):
    """Rectangles (x0, y0, x1, y1) of solid known pixels (gaps up to 6 px closed), at least
    min_piece wide and high, bounded on 3 sides by nothing of the painter's, or on all 4 by
    nothing or by a sharp colour edge. They may overlap (a picture over the painter's own flag)."""
    from scipy import ndimage
    known = arr2d >= 0
    if known.sum() < min_piece * min_piece:
        return []
    pad = 4  # so the closing does not eat into artworks touching the board's edge
    closed = ndimage.binary_closing(np.pad(known, pad), structure=np.ones((3, 3), bool),
                                    iterations=3)[pad:-pad, pad:-pad] | known
    lab, _ = ndimage.label(closed)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None or sl[0].stop - sl[0].start < min_piece or sl[1].stop - sl[1].start < min_piece:
            continue
        shape, kn = lab[sl] == i, known[sl]
        exposed, edges = side_checker(shape), edge_checker(arr2d[sl])

        def accept(*r):
            free = exposed(*r)
            if sum(free) >= 3:
                return True
            return sum(free) >= 2 and all(f or g for f, g in zip(free, edges(*r)))

        cands = sorted(set(maximal_rects(shape, min_piece, min_piece, accept)), key=lambda r: -_area(r))
        taken = np.zeros_like(shape)
        for r in cands[:40]:
            x0, y0, x1, y1 = r
            if taken[y0:y1 + 1, x0:x1 + 1].sum() > 0.7 * _area(r):  # mostly inside one already kept
                continue
            taken[y0:y1 + 1, x0:x1 + 1] = True
            x0, y0, x1, y1 = grow(r, kn, exposed(*r))
            if kn[y0:y1 + 1, x0:x1 + 1].mean() >= 0.25:  # at least a quarter really seen
                out.append((sl[1].start + x0, sl[0].start + y0, sl[1].start + x1, sl[0].start + y1))
    return out


def _area(r):
    return (r[2] - r[0] + 1) * (r[3] - r[1] + 1)


def mergeable(a, b, gap=32, fill=0.9):
    """Two pieces of one artwork: they overlap or nearly touch, and together they (almost) fill
    the rectangle around both."""
    dx = max(a[0], b[0]) - min(a[2], b[2]) - 1  # gap between them (negative: they overlap)
    dy = max(a[1], b[1]) - min(a[3], b[3]) - 1
    if dx > gap or dy > gap:
        return False
    both = _area(a) + _area(b) - max(0, -dx) * max(0, -dy)
    return both >= fill * _area((min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])))


def merge_rects(rects):
    """Merge pieces of the same artwork (see mergeable) until none are left to merge."""
    rects = list(dict.fromkeys(tuple(r) for r in rects))
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if mergeable(a, b):
                    rects[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del rects[j]
                    changed = True
                    break
            if changed:
                break
    return rects


MIN_PIECE = 24  # smaller pieces count only once merged into an artwork of at least the minimum


def find_artworks(arr2d, min_size, known_rects=()):
    """A painter's artworks: pieces found now, merged with each other and with the artworks
    already known (known_rects: they only grow), at least min_size x min_size."""
    merged = merge_rects(list(known_rects) + find_pieces(arr2d, min(MIN_PIECE, min_size)))
    keep = set(map(tuple, known_rects))
    return [r for r in merged if (r[2] - r[0] + 1 >= min_size and r[3] - r[1] + 1 >= min_size) or r in keep]


class ArtWatcher:
    """Records, for every painter name in the live feed, the colour of each pixel they paint, and
    every `every` seconds finds their rectangular artworks (at least min_size x min_size). Only
    paints seen while the window is open are known: an artwork fills up as its painter paints or
    repairs it; artworks are kept (artworks/) and only grow, merging pieces seen later."""
    STATE, INDEX = os.path.join(ART_DIR, "state.npz"), os.path.join(ART_DIR, "index.json")

    def __init__(self, log, size=(1024, 1024), min_size=80, every=30):
        self.log, self.W, self.H, self.min, self.every = log, size[0], size[1], min_size, every
        self.lock = threading.Lock()
        self.painters, self.index, self.arts, self.version = {}, {}, [], 0
        self.enabled, self.ready, self.problem = True, False, None
        threading.Thread(target=self.loop, daemon=True, name="watcher").start()

    def feed(self, x, y, colour, name):
        if self.enabled and self.ready and 0 <= x < self.W and 0 <= y < self.H:
            with self.lock:
                p = self.painters.get(name)
                if p is None:
                    p = self.painters[name] = Recorder(self.W * self.H)
                p.paint(y * self.W + x, colour)

    def load(self):
        try:
            with np.load(self.STATE, allow_pickle=False) as z:
                for i, name in enumerate(str(n) for n in z["names"]):
                    p = Recorder(self.W * self.H)
                    for idx, col in zip(z[f"i{i}"].tolist(), z[f"c{i}"].tolist()):
                        p.paint(idx, col)
                    self.painters[name] = p
        except (OSError, KeyError, ValueError):
            pass
        try:
            with open(self.INDEX) as f:
                for e in json.load(f):
                    self.index.setdefault(e["name"], []).append(e)
        except (OSError, ValueError, KeyError):
            pass
        self.publish()
        if self.painters:
            self.log(f"watcher: reloaded {sum(p.count for p in self.painters.values()):,} recorded pixels of "
                     f"{len(self.painters)} painters, {len(self.arts)} artworks")

    def save(self):
        with self.lock:
            snap = {name: p.array() for name, p in self.painters.items()}
        os.makedirs(ART_DIR, exist_ok=True)
        data = {"names": np.array(list(snap), dtype=str)}
        for i, a in enumerate(snap.values()):
            idx = np.flatnonzero(a >= 0)
            data[f"i{i}"], data[f"c{i}"] = idx.astype(np.int32), a[idx]
        tmp = self.STATE + ".tmp.npz"
        np.savez_compressed(tmp, **data)
        os.replace(tmp, self.STATE)

    def publish(self):
        arts = [e for es in self.index.values() for e in es]
        arts.sort(key=lambda a: (not a.get("complete"), a["name"].lower(), -a["area"]))
        self.arts = arts
        self.version += 1

    def detect(self, snap):
        for name, flat in snap.items():
            arr2d = flat.reshape(self.H, self.W)
            old = {tuple(e["rect"]): e for e in self.index.get(name, [])}
            entries = []
            for rect in find_artworks(arr2d, self.min, list(old)):
                x0, y0, x1, y1 = rect
                sub = arr2d[y0:y1 + 1, x0:x1 + 1]
                seen, area = int((sub >= 0).sum()), int(sub.size)
                prev = old.get(rect)
                who = name or "(no name)"
                if not prev:
                    grown = sum(1 for r in old if r[0] >= x0 and r[1] >= y0 and r[2] <= x1 and r[3] <= y1)
                    self.log(f"watcher: {'grown' if grown else 'new'}: {who} {x1 - x0 + 1}x{y1 - y0 + 1} at "
                             f"({x0},{y0}), {100 * seen / area:.0f}% known")
                elif seen == area and not prev.get("complete"):
                    self.log(f"watcher: complete: {who} {x1 - x0 + 1}x{y1 - y0 + 1} at ({x0},{y0})")
                entries.append({"name": name, "rect": list(rect), "w": x1 - x0 + 1, "h": y1 - y0 + 1,
                                "seen": seen, "area": area, "complete": seen == area})
            self.index[name] = entries
        os.makedirs(ART_DIR, exist_ok=True)
        tmp = self.INDEX + ".tmp"
        with open(tmp, "w") as f:
            json.dump([e for es in self.index.values() for e in es], f, indent=1)
        os.replace(tmp, self.INDEX)
        self.publish()

    def rgba(self, entry):
        """The artwork as an RGBA array (alpha 0 where a pixel is not known yet)."""
        x0, y0, x1, y1 = entry["rect"]
        with self.lock:
            p = self.painters.get(entry["name"])
            flat = p.array() if p else np.full(self.W * self.H, -1, np.int32)
        sub = flat.reshape(self.H, self.W)[y0:y1 + 1, x0:x1 + 1]
        out = np.zeros(sub.shape + (4,), np.uint8)
        v = np.where(sub >= 0, sub, 0)
        out[..., 0], out[..., 1], out[..., 2] = v >> 16 & 255, v >> 8 & 255, v & 255
        out[..., 3] = np.where(sub >= 0, 255, 0)
        return out

    def loop(self):
        self.load()
        self.ready = True
        last_save = time.time()
        while True:
            time.sleep(self.every)
            try:
                import scipy  # noqa: F401
            except ImportError:
                self.problem = "pip install scipy to find artworks (pixels are recorded meanwhile)"
                continue
            self.problem = None
            snap = {}
            with self.lock:
                for name, p in self.painters.items():
                    if p.changed and p.count >= MIN_PIECE * MIN_PIECE:
                        snap[name] = p.array()
                        p.changed = False
            try:
                if snap:
                    self.detect(snap)
                if time.time() - last_save >= 60:
                    self.save()
                    last_save = time.time()
            except Exception as e:  # never let a bad round stop the recording
                self.log(f"watcher: error {e.__class__.__name__}: {e}")


# =============================================================================================
# Hate scanner: OCR of the board, flags over hater words
# =============================================================================================
HATE_KEYS = ["DISGUSTING", "PATHETIC", "FUCKING", "GARBAGE", "FUCKED", "FUCKER", "HATERS", "LOSERS", "PHUCK", "SUCKS",
             "HATER", "HATES", "HATED", "DEATH", "LOSER", "TRASH", "FUCK", "SUCK", "HATE", "KILL", "STFU", "SCUM",
             "SHIT", "FCK", "FUK", "SUX", "DIE", "KYS"]
HOSTILE = [r"\bshit", r"\bgarbage\b", r"\bsux\b", r"\bhate[sd]?\b", r"\bhating\b", r"\bhaters?\b", r"\bsucks?\b",
           r"\bdeath\s+to\b", r"\bkill\b", r"\bdie\b", r"\bf+[\W_]*[u*]*[\W_]*c+[\W_]*k+", r"\bstfu\b", r"\bkys\b",
           r"\bgo\s+home\b", r"\bscum\b", r"\btrash\b", r"\bpathetic\b", r"\bloser?s?\b", r"\bdisgusting\b"]
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "|": "l"})
SCALE = 3
# identity words are not hate on their own (and may well be pride messages): never covered
IDENTITY = ["gay", "gays", "lesbian", "lesbians", "queer", "queers", "trans", "transgender", "bi", "bisexual",
            "lgbt", "lgbtq", "homosexual", "pride", "nonbinary"]


def ocr_missing():
    """What the hate scanner still needs, or "" when it can run."""
    missing = []
    for mod, pkg in (("pytesseract", "pytesseract"), ("better_profanity", "better-profanity"), ("scipy", "scipy")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"pip install {pkg}")
    if not missing:
        import pytesseract
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            missing.append("tesseract (sudo apt install tesseract-ocr / brew install tesseract)")
    return ", ".join(missing)


@functools.lru_cache(maxsize=1)
def _profanity():
    from better_profanity import profanity
    profanity.load_censor_words(whitelist_words=IDENTITY)
    return profanity


def norm(text):
    return re.sub(r"[^a-z\s*]", " ", text.lower().translate(LEET))


def hateful(text):
    t = norm(text)
    return _profanity().contains_profanity(t) or any(re.search(p, t) for p in HOSTILE)


def luminance(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def letter_mask(sel, min_area=6, max_area=5000, min_h=4, max_h=60):
    """Keep connected blobs of `sel` shaped like letters (or a few letters run together)."""
    from scipy import ndimage
    lab, n = ndimage.label(sel, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == i).sum())
        if min_area <= area <= max_area and min_h <= h <= max_h and w <= 6 * h and area < 0.85 * h * w:
            keep[i] = True
    return keep[lab]


def ocr_lines(mask):
    """Tesseract over a binary mask (True = ink): [(line text, [(word, conf, box)])]."""
    import pytesseract
    img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).resize(
        (mask.shape[1] * SCALE, mask.shape[0] * SCALE), Image.NEAREST)
    d = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    lines = {}
    for i, word in enumerate(d["text"]):
        word, conf = word.strip(), float(d["conf"][i])
        if word and conf >= 0:
            x0, y0 = d["left"][i] // SCALE, d["top"][i] // SCALE
            box = (x0, y0, x0 + d["width"][i] // SCALE, y0 + d["height"][i] // SCALE)
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append((word, conf, box))
    return [(" ".join(w for w, _, _ in ws), ws) for ws in lines.values()]


def char_boxes(mask, box):
    """Tesseract character boxes for one line region: [(char, (x0, y0, x1, y1))] in board coords."""
    import pytesseract
    x0, y0 = max(0, box[0] - 4), max(0, box[1] - 4)
    crop = mask[y0:box[3] + 5, x0:box[2] + 5]
    img = Image.fromarray(np.where(crop, 0, 255).astype(np.uint8)).resize(
        (crop.shape[1] * SCALE, crop.shape[0] * SCALE), Image.NEAREST)
    out, H = [], crop.shape[0] * SCALE
    for row in pytesseract.image_to_boxes(img, config="--psm 7").splitlines():
        c, a, b, cc, dd, _ = row.split(" ")
        out.append((c, (x0 + int(a) // SCALE, y0 + (H - int(dd)) // SCALE, x0 + int(cc) // SCALE,
                        y0 + (H - int(b)) // SCALE)))
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def find_hate_words(chars):
    """[(key, box)] of hater words in a line's characters. Best matches win: exact before one
    misread letter (allowed for words of 4+ letters), longer words before shorter ones, and
    matches never overlap, so "I HATE FRANCE" gives HATE (not "HATE F" read as HATER)."""
    s = [c.upper().translate(str.maketrans("0135", "OIES")) for c, _ in chars]
    cands = []
    for key in HATE_KEYS:
        tol = 0 if len(key) <= 3 else 1
        for n in {len(key), len(key) - 1, len(key) + 1}:
            for i in range(0, len(s) - n + 1):
                d = edit_distance("".join(s[i:i + n]), key)
                if d <= tol:
                    cands.append((d, -len(key), i, i + n, key))
    used, out = set(), []
    for d, _, i, j, key in sorted(cands):
        if used.intersection(range(i, j)):
            continue
        used.update(range(i, j))
        boxes = [b for _, b in chars[i:j]]
        out.append((key, (min(b[0] for b in boxes), min(b[1] for b in boxes),
                          max(b[2] for b in boxes), max(b[3] for b in boxes))))
    return out


def word_rect(board, ink, box, mode):
    """The word's letters plus their outline (light text) or 1 px (dark text): (x0, y0, x1, y1).
    Whole letters are taken, even where tesseract's box cuts through one (e.g. a last "s")."""
    from scipy import ndimage
    H, W = board.shape[:2]
    pad = max(4, box[3] - box[1])
    wx0, wy0 = max(0, box[0] - pad), max(0, box[1] - pad)
    win = ink[wy0:box[3] + pad + 1, wx0:box[2] + pad + 1]
    lab, n = ndimage.label(win, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    bx0, by0, bx1, by1 = box[0] - wx0 - 1, box[1] - wy0 - 1, box[2] - wx0 + 1, box[3] - wy0 + 1
    touch = [i for i, o in enumerate(objs, 1) if o is not None and o[1].start <= bx1 and o[1].stop > bx0
             and o[0].start <= by1 and o[0].stop > by0]
    if not touch:
        return None
    tallest = max(objs[i - 1][0].stop - objs[i - 1][0].start for i in touch)  # letters, not specks
    letters = [i for i in touch if objs[i - 1][0].stop - objs[i - 1][0].start >= 0.6 * tallest]
    ys, xs = np.nonzero(np.isin(lab, letters))
    if len(xs) < 6:
        return None
    m = (2 if ys.max() - ys.min() >= 26 else 1) + 1 if mode == "light" else 1
    return (int(max(0, wx0 + xs.min() - m)), int(max(0, wy0 + ys.min() - m)),
            int(min(W - 1, wx0 + xs.max() + m)), int(min(H - 1, wy0 + ys.max() + m)))


def find_words(board, min_conf=60):
    """[(word, rect)] of hater words currently readable on the board: letter-shaped blobs (dark
    letters, and light letters with a dark outline) read by tesseract; in hateful lines only,
    each hater word gets a rectangle over its letters."""
    lum = luminance(board.astype(np.float32))
    found = []
    for mode, sel in (("dark", lum < 100), ("light", lum > 200)):
        ink = letter_mask(sel)
        for text, words in ocr_lines(ink):
            good = [w for w, c, _ in words if c >= min_conf and len(re.sub(r"\W", "", w)) >= 2]
            if not good or not hateful(text):
                continue
            words = [(w, c, b) for w, c, b in words if b[3] - b[1] <= 80 and b[2] - b[0] <= 600]  # sanity
            if not words:
                continue
            line_box = (min(b[0] for _, _, b in words), min(b[1] for _, _, b in words),
                        max(b[2] for _, _, b in words), max(b[3] for _, _, b in words))
            targets = find_hate_words(char_boxes(ink, line_box))
            for w, c, b in words:  # slurs / profanity that are not in HATE_KEYS: the whole word
                if (_profanity().contains_profanity(norm(w)) or hateful(w)) and \
                        not any(overlap(b, tb) for _, tb in targets):
                    targets.append((w.upper(), b))
            for key, box in targets:
                rect = word_rect(board, ink, box, mode)
                if rect:
                    found.append((key, rect))
    return found


class HateScanner:
    """While enabled: every `interval` s, read the board and add a flag zone over each new hater word."""

    def __init__(self, path, log, interval=20):
        self.path, self.log, self.interval = path, log, interval
        self.enabled, self.wake, self.failed = threading.Event(), threading.Event(), False
        threading.Thread(target=self.loop, daemon=True, name="scanner").start()

    def loop(self):
        while True:
            self.enabled.wait()
            missing = ocr_missing()
            if missing:
                self.log(f"hate scanner needs: {missing}")
                self.enabled.clear()
                self.failed = True
                continue
            t = time.time()
            try:
                board = board_array()
                known = [z.rect for z in load_guarded(self.path)]
                new = []
                for word, rect in find_words(board):
                    if not any(overlap(rect, r) for r in known + [n for n, _ in new]):
                        new.append((rect, word))
                for rect, word in new:
                    add_guarded(rect, word, self.path)
                    self.log(f"hate scanner: {word!r} at {rect}: flag zone added")
                self.log(f"hate scanner: scan took {time.time() - t:.0f} s, {len(new)} new word(s)")
            except Exception as e:
                self.log(f"hate scanner: error {e.__class__.__name__}: {e}")
            self.wake.wait(max(1.0, self.interval - (time.time() - t)))
            self.wake.clear()


# =============================================================================================
# Drawing: paints the enabled zones and keeps repairing them, while enabled
# =============================================================================================
class Drawer:
    def __init__(self, path, settings, log):
        self.path, self.s, self.log = path, settings, log
        self.enabled, self.wake = threading.Event(), threading.Event()
        self.pool, self.pool_size = None, 0
        self.state, self.rate, self.left, self.round = "paused", 0.0, 0, 0
        threading.Thread(target=self.loop, daemon=True, name="drawer").start()

    def start(self):
        self.enabled.set()
        self.wake.set()

    def pause(self):
        self.enabled.clear()
        self.wake.set()

    def sleep(self, seconds):
        self.wake.wait(seconds)
        self.wake.clear()

    def loop(self):
        while True:
            if not self.enabled.is_set():
                self.state, self.rate = "paused", 0.0
                self.enabled.wait()
            try:
                self.one_round()
            except Exception as e:  # network trouble or a bad zone: say so, try again a bit later
                self.state = f"error: {e.__class__.__name__}"
                self.log(f"drawing: error {e.__class__.__name__}: {e}")
                self.sleep(10)

    def one_round(self):
        s = self.s
        cfg = requests.get(f"{BASE}/api/config", timeout=20).json()
        if cfg.get("read_only"):
            self.log("drawing: the board is read-only now: paused")
            self.pause()
            return
        _init(int(cfg.get("pow_kib", 1024)), int(s["max_bits"]), s["name"])
        set_hash_threads(s["hash_threads"])
        if self.pool_size != int(s["workers"]):  # new size: a new pool (the old one finishes its paints)
            if self.pool:
                self.pool.shutdown(wait=False)
            self.pool_size = int(s["workers"])
            self.pool = ThreadPoolExecutor(self.pool_size, thread_name_prefix="paint")
        pixels = zones_pixels(self.path)
        if not pixels:
            self.state, self.left = "no zone to paint", 0
            self.sleep(float(s["interval"]))
            return
        board = board_array()
        H, W = board.shape[:2]
        pixels = [p for p in pixels if 0 <= p[0] < W and 0 <= p[1] < H]
        tasks = damaged(pixels, board)
        self.left = len(tasks)
        intact = 100 * (len(pixels) - len(tasks)) / len(pixels)
        if not tasks:
            self.state, self.rate = f"all {len(pixels):,} px intact", 0.0
            self.sleep(float(s["interval"]))
            return
        self.round += 1
        batch = int(s["batch"])
        tasks = prioritize(tasks)[:batch] if batch else prioritize(tasks)
        self.log(f"[{time.strftime('%H:%M:%S')}] round {self.round}: {intact:.1f}% of {len(pixels):,} px intact, "
                 f"painting {len(tasks):,} of {self.left:,} ({s['workers']} workers, {CORES} hashing threads, "
                 f"max {s['max_bits']} bits, {backend()})")
        self.paint(tasks)

    def paint(self, tasks):
        workers = self.pool_size
        counts, done, t0 = collections.Counter(), 0, time.time()
        todo, flying = iter(tasks), set()

        def fill():
            while self.enabled.is_set() and len(flying) < 2 * workers:
                t = next(todo, None)
                if t is None:
                    return
                flying.add(self.pool.submit(paint_one, t))

        fill()
        while flying:
            finished, flying = wait(flying, timeout=1.0, return_when=FIRST_COMPLETED)
            for f in finished:
                try:
                    counts[f.result()[2]] += 1
                except Exception:
                    counts["error"] += 1
                done += 1
                if done % 250 == 0 or done == len(tasks):
                    self.log(f"  {done:,}/{len(tasks):,}  {dict(counts)}  {done / (time.time() - t0):.1f} px/s")
            el = time.time() - t0
            self.rate = counts["ok"] / el if el > 0 else 0.0
            self.state = f"painting {done:,}/{len(tasks):,}"
            fill()
        if not self.enabled.is_set() and done < len(tasks):
            self.log(f"  paused after {done:,} of {len(tasks):,} px")
        else:
            self.log(f"  round done: {counts['ok']:,} painted, {counts['deferred']:,} too hot for now "
                     f"(over {self.s['max_bits']} bits), {time.time() - t0:.0f} s")


# =============================================================================================
# Settings (canvas_settings.json)
# =============================================================================================
def load_settings():
    s = dict(name=os.environ.get("NAME", "tricolore"), workers=default_workers(), hash_threads=CORES,
             max_bits=12, batch=3000, interval=10, record=True, scanner=False, logs=False)
    try:
        with open(SETTINGS) as f:
            s.update({k: v for k, v in json.load(f).items() if k in s})
    except (OSError, ValueError):
        pass
    return s


def save_settings(s):
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, SETTINGS)


# =============================================================================================
# The window
# =============================================================================================
COLOURS = {"white": "ffffff", "black": "000000", "French blue": "002395", "French red": "ed2939",
           "gold": "ffd700", "none": "none"}
DASH = {"flag": "", "croissant": "", "white": (6, 3), "text": (2, 2), "chat": (8, 2, 2, 2), "art": (12, 4),
        "stock": (12, 4)}
ZOOMS = (0.5, 0.75, 1, 2, 3, 4, 6, 8)
SPIN = dict(workers=(1, 2048, 8), hash_threads=(1, 4 * (os.cpu_count() or 4), 1), max_bits=(8, 24, 1),
            batch=(100, 100000, 500), interval=(1, 600, 5))


class TextDialog(simpledialog.Dialog if tk else object):
    """Asks for a text zone's text and colours; result = (text, opts) or None."""

    def body(self, master):
        tk.Label(master, text="Text  ( | = new line ):").grid(row=0, column=0, sticky="w")
        self.text = tk.Entry(master, width=36)
        self.text.grid(row=0, column=1, columnspan=2, pady=2)
        self.vars = {}
        for row, (key, label, choices, default) in enumerate((
                ("fg", "Text colour", ["white", "black", "French blue", "French red", "gold"], "white"),
                ("ol", "Outline", ["black", "white", "none"], "black"),
                ("bg", "Background", ["none", "white", "black", "French blue", "French red"], "none")), start=1):
            tk.Label(master, text=label).grid(row=row, column=0, sticky="w")
            self.vars[key] = tk.StringVar(value=default)
            tk.OptionMenu(master, self.vars[key], *choices).grid(row=row, column=1, sticky="w")
        tk.Label(master, text="The text takes the biggest size that fits the box.", fg="#666").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        return self.text

    def validate(self):
        return bool(self.text.get().strip())

    def apply(self):
        self.result = (self.text.get().strip(), {k: COLOURS[v.get()] for k, v in self.vars.items()})


def checker_thumb(rgba, max_w=250, max_h=190):
    """A PIL image of an RGBA array on a grey checkerboard (unknown pixels show through), scaled to fit."""
    h, w = rgba.shape[:2]
    if rgba.shape[2] == 3:
        rgba = np.dstack([rgba, np.full((h, w), 255, np.uint8)])
    yy, xx = np.mgrid[0:h, 0:w]
    back = np.where(((yy // 4 + xx // 4) % 2)[..., None] == 0, 90, 130).astype(np.uint8).repeat(3, 2)
    al = rgba[..., 3:4] / 255.0
    img = Image.fromarray((rgba[..., :3] * al + back * (1 - al)).astype(np.uint8))
    k = min(max_w / w, max_h / h, 3)
    return img.resize((max(1, int(w * k)), max(1, int(h * k))), Image.NEAREST if k >= 1 else Image.LANCZOS)


class CanvasUI:
    def __init__(self, root, path, draw_now=False):
        self.root, self.path = root, path
        self.s = load_settings()
        self.zoom, self.board, self.view = 1, None, None
        self.zones, self.cache, self.state = [], {}, {}
        self.sel, self.drag, self.last, self.ant = None, None, "-", 0
        self.photo, self.img_item, self.render_pending, self.buttons = None, None, False, []
        self.live, self.live_buf, self.lock = False, [], threading.Lock()
        self.queue, self.wake, self.logq = queue.Queue(), threading.Event(), queue.Queue()
        self.placing, self.ghost, self.ghost_zoom, self.thumbs = None, None, None, {}
        self.art_sel, self.lib_sel, self.arts_version = None, None, -1
        self.calls = queue.Queue()  # work from other threads, run on the Tk thread by poll()
        self.stock_results, self.stock_sel, self.stock_seen = [], None, -1

        self.drawer = Drawer(path, self.s, self.log)
        self.watcher = ArtWatcher(self.log)
        self.watcher.enabled = bool(self.s["record"])
        self.scanner = HateScanner(path, self.log)

        root.title("Canvas")
        root.geometry("1450x950")
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.build_toolbars()
        self.status = tk.Label(root, anchor="w", relief="sunken")
        self.status.pack(side="bottom", fill="x")
        main = tk.Frame(root)
        main.pack(fill="both", expand=True)
        self.build_logs(main)
        self.build_art_panel(main)
        self.canvas = c = tk.Canvas(main, bg="#303030", highlightthickness=0, xscrollincrement=40,
                                    yscrollincrement=40)
        c.pack(side="left", fill="both", expand=True)
        if self.s["logs"]:
            self.toggle_logs(show=True)
        self.bind_events()

        threading.Thread(target=self.fetch_loop, daemon=True).start()
        threading.Thread(target=self.live_loop, daemon=True).start()
        root.after(100, self.poll)
        root.after(90, self.animate)
        root.after(500, self.poll_side)
        if self.s["scanner"]:
            self.scanner.enabled.set()
        if not have_backend():
            self.log("no Argon2 backend: pip install pynacl (or argon2-cffi), or run fastpow/build.py")
        elif draw_now:
            self.toggle_drawing()

    # ---------------------------------------------------------------- layout
    def build_toolbars(self):
        root = self.root
        bar = tk.Frame(root)
        bar.pack(fill="x")
        self.draw_btn = tk.Button(bar, text="Enable drawing", width=15, fg="white", bg="#2e7d32",
                                  activebackground="#3d9b42", command=self.toggle_drawing)
        self.draw_btn.pack(side="left", padx=(2, 8))
        tk.Button(bar, text="Refresh", command=self.wake.set).pack(side="left")
        tk.Button(bar, text="Zoom -", command=lambda: self.zoom_step(-1)).pack(side="left")
        tk.Button(bar, text="Zoom +", command=lambda: self.zoom_step(1)).pack(side="left")
        tk.Button(bar, text="Fit", command=self.fit).pack(side="left")
        self.kind = tk.StringVar(value="flag")
        tk.Label(bar, text="   New zone:").pack(side="left")
        for kind in ("flag", "croissant", "white", "text", "chat"):
            tk.Radiobutton(bar, text=kind.capitalize(), variable=self.kind, value=kind).pack(side="left")
        self.preview = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Preview", variable=self.preview, command=self.recompose).pack(side="left", padx=6)
        self.logs_btn = tk.Button(bar, text="Logs", command=self.toggle_logs)
        self.logs_btn.pack(side="right", padx=2)

        bar2 = tk.Frame(root)
        bar2.pack(fill="x", pady=(2, 0))
        self.vars = {}
        tk.Label(bar2, text="Name").pack(side="left", padx=(4, 2))
        self.vars["name"] = v = tk.StringVar(value=self.s["name"])
        tk.Entry(bar2, textvariable=v, width=14).pack(side="left")
        v.trace_add("write", lambda *a: self.setting_changed("name"))
        for key, label in (("workers", "Workers"), ("hash_threads", "Hash threads"), ("max_bits", "Max bits"),
                           ("batch", "Batch"), ("interval", "Interval s")):
            lo, hi, step = SPIN[key]
            tk.Label(bar2, text=label).pack(side="left", padx=(10, 2))
            self.vars[key] = v = tk.StringVar(value=str(self.s[key]))
            tk.Spinbox(bar2, from_=lo, to=hi, increment=step, textvariable=v, width=6).pack(side="left")
            v.trace_add("write", lambda *a, k=key: self.setting_changed(k))
        self.rec_var = tk.BooleanVar(value=bool(self.s["record"]))
        tk.Checkbutton(bar2, text="Record artworks", variable=self.rec_var, command=self.toggle_record).pack(
            side="left", padx=(14, 0))
        self.scan_var = tk.BooleanVar(value=bool(self.s["scanner"]))
        tk.Checkbutton(bar2, text="Hate scanner", variable=self.scan_var, command=self.toggle_scanner).pack(side="left")
        tk.Label(root, anchor="w", fg="#555", text="drag empty board = new zone   drag a zone = move   corner handle = "
                 "resize   wheel = zoom   middle-drag / Shift+drag / arrows = move around").pack(fill="x")

    def build_logs(self, parent):
        self.logs = tk.Frame(parent, width=380)
        head = tk.Frame(self.logs)
        head.pack(fill="x")
        tk.Label(head, text="Logs", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        tk.Button(head, text="Clear", command=lambda: self.log_text.delete("1.0", "end")).pack(side="right")
        self.log_text = tk.Text(self.logs, width=56, wrap="word", font=("TkFixedFont", 8), bg="#111", fg="#ddd")
        sb = tk.Scrollbar(self.logs, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    def toggle_logs(self, show=None):
        show = not self.logs.winfo_ismapped() if show is None else show
        if show:
            self.logs.pack(side="left", fill="y", before=self.canvas)
        else:
            self.logs.pack_forget()
        self.logs_btn.configure(relief="sunken" if show else "raised")
        if self.s["logs"] != show:
            self.s["logs"] = show
            save_settings(self.s)
        self.request_render()

    def build_art_panel(self, parent):
        side = tk.Frame(parent, padx=4)
        side.pack(side="right", fill="y")
        tk.Label(side, text="Artworks", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        nb = ttk.Notebook(side)
        nb.pack(fill="both", expand=True)
        self.tabs = nb
        # board artworks (watcher)
        t1 = tk.Frame(nb)
        nb.add(t1, text="Board (watched)")
        box = tk.Frame(t1)
        box.pack(fill="both", expand=True)
        self.art_list = tk.Listbox(box, width=38, height=14, exportselection=False, font=("TkFixedFont", 9))
        sb = tk.Scrollbar(box, command=self.art_list.yview)
        self.art_list.configure(yscrollcommand=sb.set)
        self.art_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.art_list.bind("<<ListboxSelect>>", lambda e: self.on_art_select())
        self.art_list.bind("<Double-Button-1>", lambda e: self.goto(self.art_sel and self.art_sel["rect"]))
        self.art_thumb = tk.Label(t1)
        self.art_thumb.pack(pady=4)
        self.art_info = tk.Label(t1, justify="left", anchor="w", wraplength=260)
        self.art_info.pack(fill="x")
        row = tk.Frame(t1)
        row.pack(fill="x", pady=4)
        tk.Button(row, text="Go to", command=lambda: self.goto(self.art_sel and self.art_sel["rect"])).pack(side="left")
        tk.Button(row, text="Place a copy...", command=self.place_board_art).pack(side="left", padx=4)
        # my library
        t2 = tk.Frame(nb)
        nb.add(t2, text="My library")
        self.lib_items = library_items()
        self.lib_list = tk.Listbox(t2, width=38, height=8, exportselection=False, font=("TkFixedFont", 9))
        self.lib_list.pack(fill="x")
        for it in self.lib_items:
            self.lib_list.insert("end", it["title"])
        self.lib_list.bind("<<ListboxSelect>>", lambda e: self.on_lib_select())
        size = tk.Frame(t2)
        size.pack(fill="x", pady=4)
        tk.Label(size, text="Width").pack(side="left")
        self.lib_width = tk.StringVar(value="")
        tk.Spinbox(size, from_=8, to=1024, increment=10, textvariable=self.lib_width, width=6).pack(side="left", padx=4)
        self.lib_width.trace_add("write", lambda *a: self.show_lib())
        self.lib_size = tk.Label(size, text="")
        self.lib_size.pack(side="left")
        self.lib_bg = tk.BooleanVar(value=False)
        self.lib_bg_box = tk.Checkbutton(t2, text="White background (transparent parts painted white)",
                                         variable=self.lib_bg, command=self.show_lib)
        self.lib_bg_box.pack(anchor="w")
        self.lib_thumb = tk.Label(t2)
        self.lib_thumb.pack(pady=4)
        self.lib_info = tk.Label(t2, justify="left", anchor="w", wraplength=260)
        self.lib_info.pack(fill="x")
        row = tk.Frame(t2)
        row.pack(fill="x", pady=4)
        tk.Button(row, text="Place a copy...", command=self.place_lib).pack(side="left")
        self.usual_btn = tk.Button(row, text="Usual place", command=self.place_lib_usual)
        self.usual_btn.pack(side="left", padx=4)
        tk.Label(t2, fg="#666", justify="left", wraplength=260,
                 text="Drop more pictures (png, jpg...) into art/library/ and restart to add them.").pack(
            anchor="w", pady=(8, 0))
        # stocks
        t3 = tk.Frame(nb)
        nb.add(t3, text="Stocks")
        row = tk.Frame(t3)
        row.pack(fill="x", pady=(4, 2))
        self.stock_q = tk.StringVar()
        entry = tk.Entry(row, textvariable=self.stock_q, width=24)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.stock_search())
        tk.Button(row, text="Search", command=self.stock_search).pack(side="left", padx=4)
        self.stock_list = tk.Listbox(t3, width=38, height=9, exportselection=False, font=("TkFixedFont", 9))
        self.stock_list.pack(fill="x")
        self.stock_list.insert("end", "type a company or a symbol, then Search")
        self.stock_list.bind("<<ListboxSelect>>", lambda e: self.on_stock_select())
        size = tk.Frame(t3)
        size.pack(fill="x", pady=4)
        tk.Label(size, text="Size").pack(side="left")
        self.stock_w, self.stock_h = tk.StringVar(value="160"), tk.StringVar(value="60")
        tk.Spinbox(size, from_=60, to=600, increment=10, textvariable=self.stock_w, width=5).pack(side="left", padx=2)
        tk.Label(size, text="x").pack(side="left")
        tk.Spinbox(size, from_=24, to=400, increment=10, textvariable=self.stock_h, width=5).pack(side="left", padx=2)
        for v in (self.stock_w, self.stock_h):
            v.trace_add("write", lambda *a: self.show_stock())
        self.stock_thumb = tk.Label(t3)
        self.stock_thumb.pack(pady=4)
        self.stock_info = tk.Label(t3, justify="left", anchor="w", wraplength=260)
        self.stock_info.pack(fill="x")
        tk.Button(t3, text="Place widget...", command=self.place_stock).pack(anchor="w", pady=4)
        tk.Label(t3, fg="#666", justify="left", wraplength=260,
                 text=f"Free, delayed prices from Yahoo Finance (any market) and Nasdaq (US). A placed widget "
                      f"updates every {STOCK_REFRESH // 60} min; the drawing repaints what changed.").pack(anchor="w")

    def bind_events(self):
        c, root = self.canvas, self.root
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Shift-ButtonPress-1>", lambda e: c.scan_mark(e.x, e.y))
        c.bind("<Shift-B1-Motion>", lambda e: (c.scan_dragto(e.x, e.y, gain=1), self.request_render()))
        c.bind("<Button-3>", self.on_right)
        c.bind("<MouseWheel>", lambda e: self.zoom_step(1 if e.delta > 0 else -1, (e.x, e.y)))
        c.bind("<Button-4>", lambda e: self.zoom_step(1, (e.x, e.y)))  # wheel on Linux
        c.bind("<Button-5>", lambda e: self.zoom_step(-1, (e.x, e.y)))
        c.bind("<ButtonPress-2>", lambda e: c.scan_mark(e.x, e.y))
        c.bind("<B2-Motion>", lambda e: (c.scan_dragto(e.x, e.y, gain=1), self.request_render()))
        c.bind("<Configure>", lambda e: self.request_render())
        c.bind("<Motion>", self.on_motion)
        for key, dx, dy in (("<Left>", -1, 0), ("<Right>", 1, 0), ("<Up>", 0, -1), ("<Down>", 0, 1)):
            root.bind(key, lambda e, dx=dx, dy=dy: self.pan(dx, dy))
        for key in ("<Delete>", "<BackSpace>"):
            root.bind(key, lambda e: self.delete_selected() if not isinstance(e.widget, (tk.Entry, tk.Spinbox)) else None)
        root.bind("<Escape>", lambda e: self.cancel_placing())

    # ---------------------------------------------------------------- settings, drawing, logs
    def setting_changed(self, key):
        raw = self.vars[key].get().strip()
        if key == "name":
            if not re.fullmatch(r"[A-Za-z0-9 _.\-]{1,16}", raw):
                return  # the API only takes 1-16 of A-Z a-z 0-9 space _ . -
            value = raw
        else:
            lo, hi, _ = SPIN[key]
            try:
                value = max(lo, min(hi, int(float(raw))))
            except ValueError:
                return
        if self.s.get(key) != value:
            self.s[key] = value  # the drawing reads it at its next round
            save_settings(self.s)

    def toggle_drawing(self):
        if self.drawer.enabled.is_set():
            self.drawer.pause()
            self.log("drawing: paused by you (paints in flight finish)")
        else:
            if not have_backend():
                messagebox.showerror("Drawing", "No Argon2 backend: pip install pynacl (or argon2-cffi), or run "
                                                "fastpow/build.py", parent=self.root)
                return
            self.drawer.start()
            self.log("drawing: enabled")
        self.update_draw_button()

    def update_draw_button(self):
        on = self.drawer.enabled.is_set()
        self.draw_btn.configure(text="Pause drawing" if on else "Enable drawing", bg="#c26a00" if on else "#2e7d32",
                                activebackground="#d97c10" if on else "#3d9b42")

    def toggle_record(self):
        self.s["record"] = self.watcher.enabled = bool(self.rec_var.get())
        save_settings(self.s)

    def toggle_scanner(self):
        self.s["scanner"] = on = bool(self.scan_var.get())
        save_settings(self.s)
        if on:
            self.scanner.failed = False
            self.scanner.enabled.set()
            self.scanner.wake.set()
        else:
            self.scanner.enabled.clear()

    def log(self, msg):
        """Thread-safe: shown in the logs panel (and printed)."""
        print(msg, flush=True)
        self.logq.put(msg)

    def flush_logs(self):
        lines = []
        while True:
            try:
                lines.append(self.logq.get_nowait())
            except queue.Empty:
                break
        if lines:
            at_end = self.log_text.yview()[1] > 0.999
            self.log_text.insert("end", "\n".join(lines) + "\n")
            extra = int(self.log_text.index("end-1c").split(".")[0]) - 3000
            if extra > 0:
                self.log_text.delete("1.0", f"{extra}.0")
            if at_end:
                self.log_text.see("end")

    def quit(self):
        try:
            self.watcher.save()
        except Exception as e:
            print(f"could not save the watcher's recording: {e}")
        self.root.destroy()

    # ---------------------------------------------------------------- board data (threads)
    def fetch_loop(self):
        """Full board download: at start, every 10 s without the live feed, every 60 s with it."""
        while True:
            try:
                self.queue.put(board_array().copy())
            except Exception as e:  # network trouble: show it, try again next time
                self.queue.put(e)
            self.wake.wait(60 if self.live else 10)
            self.wake.clear()

    def live_loop(self):
        """Pixels as they are painted, from the board's live feed ("x,y,rrggbb,name" lines); they
        also feed the artwork watcher."""
        try:
            import websockets
        except ImportError:
            self.log("pip install websockets for the live view and the artwork watcher (reloading every 10 s)")
            return

        async def listen():
            async with websockets.connect(WS_URL, open_timeout=20, max_size=None, ping_interval=20) as ws:
                self.live = True
                async for msg in ws:
                    for line in str(msg).splitlines():
                        if line == "reload":
                            self.wake.set()
                            continue
                        p = line.split(",", 3)
                        if len(p) >= 3 and len(p[2]) == 6:
                            try:
                                x, y, col = int(p[0]), int(p[1]), int(p[2], 16)
                            except ValueError:
                                continue
                            with self.lock:
                                self.live_buf.append((x, y, col))
                            self.watcher.feed(x, y, col, p[3].strip() if len(p) > 3 else "")

        while True:
            try:
                asyncio.run(listen())
            except Exception:
                pass
            self.live = False
            time.sleep(5)

    def poll(self):
        """Apply new data on the Tk thread, a few times per second."""
        changed = False
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, Exception):
                self.log(f"board fetch failed: {item}")
            else:
                first = self.board is None
                self.board, self.last, changed = item, time.strftime("%H:%M:%S"), True
                if first:
                    self.reload_zones()
                    self.fit()
        with self.lock:
            buf, self.live_buf = self.live_buf, []
        if buf and self.board is not None:
            H, W = self.board.shape[:2]
            buf = [(x, y, c) for x, y, c in buf if 0 <= x < W and 0 <= y < H]
            if buf:
                xs, ys, cs = (np.array(v) for v in zip(*buf))
                self.board[ys, xs] = np.stack([cs >> 16 & 255, cs >> 8 & 255, cs & 255], 1).astype(np.uint8)
                changed = True
        if changed and self.board is not None:
            old = dict(self.state)
            self.update_state()
            self.recompose()
            if old != self.state:
                self.draw_overlays()
        while True:
            try:
                self.calls.get_nowait()()
            except queue.Empty:
                break
        if STOCKS.version != self.stock_seen:  # new stock data: widgets re-drawn, preview updated
            self.stock_seen = STOCKS.version
            self.cache = {k: v for k, v in self.cache.items() if k[2] != "stock"}
            if self.board is not None and any(z.kind == "stock" for z in self.zones):
                self.update_state()
                self.recompose()
            self.show_stock()
        self.flush_logs()
        self.update_status()
        self.root.after(250, self.poll)

    # ---------------------------------------------------------------- zones
    @staticmethod
    def chat_sample(zone):
        """What a chat zone will look like (chat_to_board.py paints it): sample lines."""
        import pixfont
        x0, y0, x1, _ = zone.rect
        w, hh = x1 - x0 + 1, CHAT_LINES * (pixfont.HEIGHT + 1) + 1
        img = np.full((hh, w, 3), 255, np.uint8)
        sample = (("Marie", "Vive la France !"), ("Tom_42", "the croissant flag was so good"),
                  ("Zoe", "allez les bleus"), ("Lucas", "on lache rien"), ("Ines", "trop beau le drapeau"),
                  ("Max", "who painted the QR code?"))
        for i in range(CHAT_LINES):
            who, msg = sample[i % len(sample)]
            if i == CHAT_LINES - 1:
                pixfont.draw(img, 2, 1 + i * 8, ">", (0xed, 0x29, 0x39))
            x = pixfont.draw(img, 14, 1 + i * 8, who + ": ", (0x00, 0x23, 0x95))
            pixfont.draw(img, x, 1 + i * 8, msg, (0, 0, 0))
        ys, xs = np.mgrid[0:hh, 0:w]
        return [(x0 + int(x), y0 + int(y), img[y, x].tobytes().hex()) for y, x in zip(ys.ravel(), xs.ravel())]

    def arrays(self, zone):
        """(xs, ys, colours) numpy arrays of the zone's target, cached per zone."""
        k = zone.key()
        if zone.kind == "stock":  # re-drawn when its data is updated
            k += (STOCKS.stamp(zone.opts.get("sym", ""), zone.opts.get("src", "yahoo"), zone.opts.get("cls", "")),)
        if k not in self.cache:
            t = self.chat_sample(zone) if zone.kind == "chat" else zone_target(zone)
            H, W = self.board.shape[:2] if self.board is not None else (1024, 1024)
            t = [p for p in t if 0 <= p[0] < W and 0 <= p[1] < H]
            if t:
                xs, ys, cs = zip(*t)
                self.cache[k] = (np.array(xs), np.array(ys), np.frombuffer(bytes.fromhex("".join(cs)), np.uint8).reshape(-1, 3))
            else:
                self.cache[k] = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), np.uint8))
        return self.cache[k]

    def chat_h(self):
        return CHAT_LINES * 8 + 1

    def reload_zones(self):
        """Re-read guarded.txt and refresh everything that depends on it."""
        self.zones = load_guarded(self.path)
        H = self.board.shape[0] if self.board is not None else 1024
        fixed = False
        for z in self.zones:  # a chat zone drawn for another number of lines: the panel's real height
            x0, y0, x1, y1 = z.rect
            if z.kind == "chat" and y1 - y0 + 1 != self.chat_h():
                y0 = max(0, min(y0, H - self.chat_h()))
                move_guarded(z.rect, (x0, y0, x1, y0 + self.chat_h() - 1), self.path)
                fixed = True
        if fixed:
            self.zones = load_guarded(self.path)
        keys = {z.key() for z in self.zones}
        self.cache = {k: v for k, v in self.cache.items() if k in keys}
        if self.sel not in {z.rect for z in self.zones}:
            self.sel = None
        if self.board is not None:
            self.update_state()
            self.recompose()
        self.draw_overlays()

    def update_state(self):
        self.state = {}
        for z in self.zones:
            if z.enabled and z.kind != "chat":  # the chat panel's content is not known here
                xs, ys, cs = self.arrays(z)
                self.state[z.rect] = bool((self.board[ys, xs] == cs).all()) if len(xs) else True

    def recompose(self):
        """The image to show: the board, with the zones' targets drawn over it in preview mode."""
        if self.board is None:
            return
        if self.preview.get():
            self.view = self.board.copy()
            for z in self.zones:
                if z.enabled:
                    xs, ys, cs = self.arrays(z)
                    self.view[ys, xs] = cs
        else:
            self.view = self.board
        self.request_render()

    # ---------------------------------------------------------------- drawing (visible part only)
    def request_render(self):
        if not self.render_pending:
            self.render_pending = True
            self.root.after_idle(self.render)

    def render(self):
        self.render_pending = False
        if self.view is None:
            return
        z, c = self.zoom, self.canvas
        H, W = self.view.shape[:2]
        c.configure(scrollregion=(0, 0, W * z, H * z))
        vx, vy = c.canvasx(0), c.canvasy(0)
        vw, vh = max(1, c.winfo_width()), max(1, c.winfo_height())
        if z >= 1:
            z = int(z)
            bx0, by0 = max(0, int(vx // z) - 1), max(0, int(vy // z) - 1)
            bx1, by1 = min(W, int((vx + vw) // z) + 2), min(H, int((vy + vh) // z) + 2)
            crop = self.view[by0:by1, bx0:bx1]
            if z > 1:
                crop = crop.repeat(z, axis=0).repeat(z, axis=1)
        else:  # zoomed out: shrink the visible part smoothly
            bx0, by0 = max(0, int(vx / z) - 2), max(0, int(vy / z) - 2)
            bx1, by1 = min(W, int((vx + vw) / z) + 2), min(H, int((vy + vh) / z) + 2)
            crop = self.view[by0:by1, bx0:bx1]
        if crop.size == 0:
            return
        img = Image.fromarray(np.ascontiguousarray(crop))
        if z < 1:
            img = img.resize((max(1, round(img.width * z)), max(1, round(img.height * z))), Image.BOX)
        self.photo = ImageTk.PhotoImage(img)
        if self.img_item is None:
            self.img_item = c.create_image(0, 0, anchor="nw", tags="img")
        c.itemconfigure(self.img_item, image=self.photo)
        c.coords(self.img_item, bx0 * self.zoom, by0 * self.zoom)
        c.tag_lower("img")

    def draw_overlays(self):
        """Zone outlines, labels, and the selected zone's animated border, handle and buttons."""
        c, z = self.canvas, self.zoom
        c.delete("zone")
        for b in self.buttons:
            b.destroy()
        self.buttons = []
        for zone in self.zones:
            x0, y0, x1, y1 = zone.rect
            box = (x0 * z - 1, y0 * z - 1, (x1 + 1) * z, (y1 + 1) * z)
            col = "#9a9a9a" if not zone.enabled else {True: "#00e000", False: "#ff3030"}.get(self.state.get(zone.rect), "white")
            label = zone.note if zone.kind == "flag" else f"{zone.kind}: {zone.note}"
            if zone.rect == self.sel:
                c.create_rectangle(*box, outline="white", width=2, tags="zone")
                c.create_rectangle(*box, outline="black", width=2, dash=(4, 4), dashoffset=self.ant, tags=("zone", "ants"))
                if zone.kind != "art":  # a picture keeps its size: no resize handle
                    hx, hy = box[2], box[3]
                    c.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, fill="yellow", outline="black", tags="zone")
                self.add_buttons(zone, box)
            else:
                c.create_rectangle(*box, outline=col, width=2, dash=DASH.get(zone.kind, ""), tags="zone")
            if label and (z >= 2 or zone.rect == self.sel):
                c.create_text(box[0] + 1, box[1] - (24 if zone.rect == self.sel else 2), text=label, anchor="sw",
                              fill=col, tags="zone")
        self.draw_art_source()

    def add_buttons(self, zone, box):
        c = self.canvas
        on = zone.enabled
        toggle = tk.Button(c, text="Drawing: ON" if on else "Drawing: OFF", font=("TkDefaultFont", 8), padx=3, pady=0,
                           bg="#2e7d32" if on else "#8e2b2b", fg="white", activebackground="#555",
                           command=lambda: self.toggle_zone(zone))
        delete = tk.Button(c, text="Delete", font=("TkDefaultFont", 8), padx=3, pady=0, command=self.delete_selected)
        above = box[1] - 3 >= self.canvas.canvasy(0) + 22  # room above, else put them under the box
        y, anchor = (box[1] - 3, "sw") if above else (box[3] + 3, "nw")
        c.create_window(box[0], y, window=toggle, anchor=anchor, tags="zone")
        c.create_window(box[0] + toggle.winfo_reqwidth() + 3, y, window=delete, anchor=anchor, tags="zone")
        self.buttons = [toggle, delete]

    def animate(self):
        self.ant = (self.ant - 1) % 8
        self.canvas.itemconfigure("ants", dashoffset=self.ant)
        self.root.after(90, self.animate)

    def update_status(self):
        if self.placing:
            self.status.config(text="click where the top-left corner goes (Esc cancels)")
            return
        bad = sum(1 for v in self.state.values() if v is False)
        off = sum(1 for zn in self.zones if not zn.enabled)
        mode = "live" if self.live else "reload every 10 s"
        d = self.drawer
        drawing = (f"drawing: {d.state}" + (f", {d.rate:.0f} px/s" if d.rate else "") +
                   (f", {d.left:,} px to repair" if d.left and d.enabled.is_set() else ""))
        self.status.config(text=f"  {drawing}     |     {len(self.zones)} zone(s), {bad} damaged, {off} off   "
                                f"{mode}, full {self.last}   zoom {self.zoom:g}x")
        if self.drawer.enabled.is_set() != (self.draw_btn.cget("text") == "Pause drawing"):
            self.update_draw_button()
        if self.scanner.failed and self.scan_var.get():
            self.scan_var.set(False)
            self.s["scanner"] = False
            save_settings(self.s)

    # ---------------------------------------------------------------- navigation
    def board_xy(self, e):
        return int(self.canvas.canvasx(e.x) // self.zoom), int(self.canvas.canvasy(e.y) // self.zoom)

    def zoom_step(self, d, anchor=None):
        i = min(range(len(ZOOMS)), key=lambda k: abs(ZOOMS[k] - self.zoom))
        self.set_zoom(ZOOMS[max(0, min(len(ZOOMS) - 1, i + d))], anchor)

    def set_zoom(self, z, anchor=None):
        """Zoom keeping the board point under `anchor` (window x, y; default: view centre) in place."""
        if z == self.zoom or self.board is None:
            return
        c = self.canvas
        ax, ay = anchor if anchor else (c.winfo_width() // 2, c.winfo_height() // 2)
        bx, by = c.canvasx(ax) / self.zoom, c.canvasy(ay) / self.zoom
        self.zoom = z
        H, W = self.board.shape[:2]
        c.configure(scrollregion=(0, 0, W * z, H * z))
        c.xview_moveto(max(0.0, bx * z - ax) / (W * z))
        c.yview_moveto(max(0.0, by * z - ay) / (H * z))
        self.ghost_zoom = None
        self.draw_overlays()
        self.request_render()

    def fit(self):
        """The biggest zoom that shows the whole board, from its top-left corner."""
        if self.board is None:
            return
        H, W = self.board.shape[:2]
        c = self.canvas
        c.update_idletasks()
        fits = [z for z in ZOOMS if W * z <= c.winfo_width() and H * z <= c.winfo_height()]
        self.zoom = fits[-1] if fits else ZOOMS[0]
        c.configure(scrollregion=(0, 0, W * self.zoom, H * self.zoom))
        c.xview_moveto(0)
        c.yview_moveto(0)
        self.ghost_zoom = None
        self.draw_overlays()
        self.request_render()

    def pan(self, dx, dy):
        if dx:
            self.canvas.xview_scroll(dx, "units")
        if dy:
            self.canvas.yview_scroll(dy, "units")
        self.draw_overlays()
        self.request_render()

    def goto(self, rect):
        if not rect or self.board is None:
            return
        x0, y0, x1, y1 = rect
        c, z = self.canvas, self.zoom
        H, W = self.board.shape[:2]
        c.xview_moveto(max(0.0, ((x0 + x1) / 2 * z - c.winfo_width() / 2) / (W * z)))
        c.yview_moveto(max(0.0, ((y0 + y1) / 2 * z - c.winfo_height() / 2) / (H * z)))
        self.request_render()
        self.draw_overlays()

    # ---------------------------------------------------------------- mouse on zones
    def zone_at(self, x, y):
        for zone in reversed(self.zones):
            r = zone.rect
            if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                return zone
        return None

    def on_handle(self, e):
        if not self.sel:
            return False
        hx, hy = (self.sel[2] + 1) * self.zoom, (self.sel[3] + 1) * self.zoom
        return abs(self.canvas.canvasx(e.x) - hx) <= 7 and abs(self.canvas.canvasy(e.y) - hy) <= 7

    def clamp(self, r):
        H, W = self.board.shape[:2]
        return (max(0, r[0]), max(0, r[1]), min(W - 1, r[2]), min(H - 1, r[3]))

    def on_press(self, e):
        self.canvas.focus_set()
        if self.board is None:
            return
        if self.placing:
            self.finish_placing(e)
            return
        x, y = self.board_xy(e)
        if self.on_handle(e):
            self.drag = ("resize", self.sel, (x, y))
        else:
            zone = self.zone_at(x, y)
            new_sel = zone.rect if zone else None
            self.drag = ("move", zone.rect, (x, y)) if zone else ("create", None, (x, y))
            if new_sel != self.sel:
                self.sel = new_sel
                self.draw_overlays()

    def zone_kind(self, rect):
        return next((z.kind for z in self.zones if z.rect == rect), None)

    def drag_rect(self, e):
        mode, rect, (sx, sy) = self.drag
        x, y = self.board_xy(e)
        if mode == "create":
            return (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
        if mode == "move":
            H, W = self.board.shape[:2]
            dx = max(-rect[0], min(W - 1 - rect[2], x - sx))
            dy = max(-rect[1], min(H - 1 - rect[3], y - sy))
            return (rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy)
        kind = self.zone_kind(rect)
        if kind == "art":  # a picture keeps its size
            return rect
        if kind == "chat":  # a chat zone only changes width: its height follows the lines
            return (rect[0], rect[1], max(rect[0] + 40, x), rect[1] + self.chat_h() - 1)
        if kind == "stock":  # the widget re-lays itself out, down to 60 x 24
            return (rect[0], rect[1], max(rect[0] + 59, x), max(rect[1] + 23, y))
        return (rect[0], rect[1], max(rect[0] + 2, x), max(rect[1] + 2, y))  # resize from the corner

    def on_drag(self, e):
        if not self.drag or self.board is None:
            return
        r, z = self.clamp(self.drag_rect(e)), self.zoom
        self.canvas.delete("rubber")
        self.canvas.create_rectangle(r[0] * z, r[1] * z, (r[2] + 1) * z, (r[3] + 1) * z, outline="yellow",
                                     dash=(3, 3), width=2, tags="rubber")

    def on_release(self, e):
        if not self.drag or self.board is None:
            return
        mode, old, _ = self.drag
        rect = self.clamp(self.drag_rect(e))
        self.drag = None
        self.canvas.delete("rubber")
        if mode in ("move", "resize"):
            if rect != old:
                move_guarded(old, rect, self.path)
                self.sel = rect
                self.reload_zones()
            return
        if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
            return
        kind = self.kind.get()
        if kind == "chat":  # one chat panel: replace the previous one
            for z in self.zones:
                if z.kind == "chat":
                    remove_guarded(z.rect, self.path)
            rect = self.clamp((rect[0], rect[1], max(rect[2], rect[0] + 40), rect[1] + self.chat_h() - 1))
            add_guarded(rect, "chat", self.path, "chat")
        elif kind == "text":
            got = TextDialog(self.root, f"New text zone {rect}").result
            if not got:
                return
            add_guarded(rect, got[0], self.path, "text", got[1])
        else:
            note = simpledialog.askstring(f"New {kind} zone", f"{kind.capitalize()} zone {rect}\nNote (optional):",
                                          parent=self.root)
            if note is None:  # cancelled
                return
            add_guarded(rect, note, self.path, kind)
        self.sel = rect
        self.reload_zones()

    def on_right(self, e):
        if self.placing:
            self.cancel_placing()
            return
        zone = self.zone_at(*self.board_xy(e))
        if zone:
            self.sel = zone.rect
            self.draw_overlays()
            self.delete_selected()

    def toggle_zone(self, zone):
        set_enabled(zone.rect, not zone.enabled, self.path)
        self.reload_zones()

    def delete_selected(self):
        if not self.sel:
            return
        if messagebox.askyesno("Delete zone", f"Stop guarding the zone at {self.sel}?", parent=self.root):
            remove_guarded(self.sel, self.path)
            self.sel = None
            self.reload_zones()

    # ---------------------------------------------------------------- artworks panel
    def poll_side(self):
        """Refresh the watcher's list when it changed."""
        w = self.watcher
        if w.version != self.arts_version:
            self.arts_version = w.version
            key = (self.art_sel["name"], tuple(self.art_sel["rect"])) if self.art_sel else None
            self.art_list.delete(0, "end")
            for a in w.arts:
                x0, y0 = a["rect"][:2]
                pct = 100 * a["seen"] / max(1, a["area"])
                self.art_list.insert("end", f"{(a['name'] or '(no name)')[:14]:14s} {a['w']:>3}x{a['h']:<3} "
                                            f"({x0},{y0}) {'done' if a.get('complete') else f'{pct:.0f}%'}")
            if not w.arts:
                self.art_list.insert("end", "none yet: watching the live feed..." if w.enabled else
                                     "Record artworks is off")
            match = [i for i, a in enumerate(w.arts) if (a["name"], tuple(a["rect"])) == key]
            if match:
                self.art_list.selection_set(match[0])
                self.art_sel = w.arts[match[0]]
            else:
                self.art_sel = None
            self.show_art()
        if w.problem and not self.art_info.cget("text"):
            self.art_info.configure(text=w.problem)
        self.root.after(2000, self.poll_side)

    def on_art_select(self):
        sel = self.art_list.curselection()
        arts = self.watcher.arts
        self.art_sel = arts[sel[0]] if sel and sel[0] < len(arts) else None
        self.show_art()

    def show_art(self):
        a = self.art_sel
        if not a:
            self.art_thumb.configure(image="")
            self.art_info.configure(text="")
            self.draw_art_source()
            return
        self.thumbs["art"] = ImageTk.PhotoImage(checker_thumb(self.watcher.rgba(a)))
        self.art_thumb.configure(image=self.thumbs["art"])
        x0, y0, x1, y1 = a["rect"]
        pct = 100 * a["seen"] / max(1, a["area"])
        self.art_info.configure(text=f"{a['name'] or '(no name)'}\n{a['w']}x{a['h']} at ({x0},{y0})-({x1},{y1})\n"
                                     f"{a['seen']:,} of {a['area']:,} px known ({pct:.1f}%)\n" +
                                     ("complete" if a.get("complete") else
                                      "not complete yet: unknown pixels are copied from the board when placed"))
        self.draw_art_source()

    def draw_art_source(self):
        c, z = self.canvas, self.zoom
        c.delete("artsrc")
        if self.art_sel and self.tabs.index("current") == 0:
            x0, y0, x1, y1 = self.art_sel["rect"]
            c.create_rectangle(x0 * z - 2, y0 * z - 2, (x1 + 1) * z + 1, (y1 + 1) * z + 1, outline="cyan", width=2,
                               dash=(6, 3), tags="artsrc")

    def on_lib_select(self):
        sel = self.lib_list.curselection()
        self.lib_sel = self.lib_items[sel[0]] if sel else None
        if self.lib_sel:
            self.lib_bg.set(bool(self.lib_sel.get("bg")))
            self.lib_width.set(str(self.lib_sel["width"]))  # triggers show_lib
        self.show_lib()

    def lib_bg_colour(self):
        it = self.lib_sel
        if not it or it["kind"] != "image" or not library_has_alpha(it["file"]) or not self.lib_bg.get():
            return None
        return it.get("bg") or "ffffff"

    def lib_w(self):
        try:
            return max(8, min(1024, int(float(self.lib_width.get()))))
        except ValueError:
            return self.lib_sel["width"] if self.lib_sel else 100

    def show_lib(self):
        it = self.lib_sel
        if not it:
            self.lib_thumb.configure(image="")
            self.lib_info.configure(text="")
            return
        w = self.lib_w()
        alpha = it["kind"] == "image" and library_has_alpha(it["file"])
        self.lib_bg_box.configure(state="normal" if alpha else "disabled")
        try:
            img = library_render(it, w, self.lib_bg_colour())
        except OSError as e:
            self.lib_info.configure(text=f"cannot read {it.get('file')}: {e}")
            return
        self.thumbs["lib"] = ImageTk.PhotoImage(checker_thumb(img))
        self.lib_thumb.configure(image=self.thumbs["lib"])
        h = img.shape[0]
        n = int((img[..., 3] >= 128).sum()) if img.shape[2] == 4 else w * h
        self.lib_size.configure(text=f"{w}x{h}, {n:,} px to paint")
        place = it.get("place")
        self.lib_info.configure(text=f"{it['title']}\n" + (f"usual place: ({place[0]},{place[1]})\n" if place else "") +
                                     ("drawn at any size; it stays resizable on the board" if it["kind"] != "image"
                                      else "a picture: its size is fixed once placed") +
                                     ("\ntransparent parts are not painted: the board shows through"
                                      if img.shape[2] == 4 else ""))
        self.usual_btn.configure(state="normal" if place else "disabled")

    # ---------------------------------------------------------------- stocks
    def stock_search(self):
        q = self.stock_q.get().strip()
        if not q:
            return
        self.stock_list.delete(0, "end")
        self.stock_list.insert("end", f"searching '{q}'...")

        def work():
            try:
                res, probs = search_stocks(q)
            except Exception as e:
                res, probs = [], [str(e)]
            self.calls.put(lambda: self.show_search(res, probs))
        threading.Thread(target=work, daemon=True).start()

    def show_search(self, res, probs):
        self.stock_results, self.stock_sel = res, None
        self.stock_list.delete(0, "end")
        for r in res:
            self.stock_list.insert("end", f"{r['sym'][:9]:9s} {r['name'][:21]:21s} {r['exch'][:6]}")
        if not res:
            self.stock_list.insert("end", "nothing found")
        yahoo_down = any(p.startswith("Yahoo") for p in probs)
        self.stock_info.configure(text=("Yahoo Finance refuses this network right now: US markets only (Nasdaq)."
                                        if yahoo_down and res else "; ".join(probs)))
        self.show_stock()

    def on_stock_select(self):
        sel = self.stock_list.curselection()
        self.stock_sel = self.stock_results[sel[0]] if sel and sel[0] < len(self.stock_results) else None
        self.show_stock()

    def stock_size(self):
        try:
            return (max(60, min(600, int(float(self.stock_w.get())))), max(24, min(400, int(float(self.stock_h.get())))))
        except ValueError:
            return 160, 60

    def show_stock(self):
        r = self.stock_sel
        if not r:
            self.stock_thumb.configure(image="")
            return
        d = STOCKS.data(r["sym"], r["src"], r["cls"])  # in the background; poll() calls back on news
        if not d:
            self.stock_thumb.configure(image="")
            err = STOCKS.errors.get((r["sym"], r["src"], r["cls"]))
            self.stock_info.configure(text=f"{r['sym']}: {err or 'loading...'}")
            return
        w, h = self.stock_size()
        self.thumbs["stock"] = ImageTk.PhotoImage(checker_thumb(render_stock(d, w, h)))
        self.stock_thumb.configure(image=self.thumbs["stock"])
        pts = d["points"]
        pct = 100 * (d["price"] - pts[0]) / pts[0] if pts and pts[0] else 0
        self.stock_info.configure(text=f"{d['name']} ({d['sym']})\n{_fmt_price(d['price'])} {d['currency']}, "
                                       f"{pct:+.2f}% over the last week\nas of {d['asof']}, from "
                                       f"{'Yahoo Finance' if d['src'] == 'yahoo' else 'Nasdaq'}\nwidget {w}x{h} = {w * h:,} px")

    def place_stock(self):
        r = self.stock_sel
        d = r and STOCKS.data(r["sym"], r["src"], r["cls"])
        if not d:
            messagebox.showinfo("Stock widget", "Search, select a stock and wait for its price first.", parent=self.root)
            return
        w, h = self.stock_size()

        def commit(x, y):
            rect = (x, y, x + w - 1, y + h - 1)
            opts = {k: v for k, v in (("sym", r["sym"]), ("src", r["src"]), ("cls", r["cls"])) if v}
            add_guarded(rect, f"{r['sym']} stock", self.path, "stock", opts)
            self.sel = rect
            self.reload_zones()
        self.start_placing(render_stock(d, w, h), commit, f"the {r['sym']} widget",
                           f"It updates every {STOCK_REFRESH // 60} min; the drawing repaints what changed.\n\n")

    # ---------------------------------------------------------------- placing a copy
    def place_board_art(self):
        a = self.art_sel
        if not a:
            messagebox.showinfo("Place a copy", "Select an artwork in the list first.", parent=self.root)
            return
        rgba = self.watcher.rgba(a)
        unknown = rgba[..., 3] < 128
        who = a["name"] or "noname"

        def commit(x, y):
            img = rgba[..., :3].copy()
            if unknown.any() and self.board is not None:  # still unknown: as the board shows it now
                sx0, sy0 = a["rect"][:2]
                img[unknown] = self.board[sy0:sy0 + img.shape[0], sx0:sx0 + img.shape[1]][unknown]
            self.add_art_zone(img, re.sub(r"[^A-Za-z0-9_.-]", "_", who), x, y, f"copy of {who} art")

        extra = (f"{int(unknown.sum()):,} pixels are not known yet: they are taken from the board where the "
                 f"artwork is now.\n\n" if unknown.any() else "")
        self.start_placing(rgba, commit, f"a copy of {who}'s artwork", extra)

    def lib_commit(self):
        it, w = self.lib_sel, self.lib_w()
        img = library_render(it, w, self.lib_bg_colour())
        h = img.shape[0]

        def commit(x, y):
            if it["kind"] in ("flag", "croissant"):  # a flag zone: drawn at any size, resizable later
                add_guarded((x, y, x + w - 1, y + h - 1), it["title"], self.path, it["kind"])
                self.sel = (x, y, x + w - 1, y + h - 1)
                self.reload_zones()
            else:
                self.add_art_zone(img, it["id"].split(".")[0], x, y, it["title"])
        return img, commit

    def place_lib(self):
        if not self.lib_sel:
            messagebox.showinfo("Place a copy", "Select an item of the library first.", parent=self.root)
            return
        img, commit = self.lib_commit()
        self.start_placing(img, commit, self.lib_sel["title"])

    def place_lib_usual(self):
        it = self.lib_sel
        if not it or not it.get("place"):
            return
        img, commit = self.lib_commit()
        h, w = img.shape[:2]
        x, y = it["place"]
        if messagebox.askyesno("Usual place", f"Place {it['title']} ({w}x{h}) at ({x},{y})?\n\nThe drawing paints it "
                                              f"while it is enabled.", parent=self.root):
            commit(x, y)
            self.goto((x, y, x + w - 1, y + h - 1))

    def add_art_zone(self, img, stem, x, y, note):
        h, w = img.shape[:2]
        rel = f"{PLACED}/{stem}_{x}_{y}_{time.strftime('%Y%m%d-%H%M%S')}.png"
        os.makedirs(os.path.join(HERE, PLACED), exist_ok=True)
        Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(HERE, rel))  # RGBA keeps transparent parts
        rect = (x, y, x + w - 1, y + h - 1)
        add_guarded(rect, note, self.path, "art", {"file": rel})
        self.sel = rect
        self.reload_zones()

    def start_placing(self, img, commit, label, extra=""):
        if self.board is None:
            return
        self.placing = (img, commit, label, extra)
        self.ghost_zoom = None
        self.canvas.configure(cursor="crosshair")
        self.update_status()

    def cancel_placing(self):
        self.placing = None
        self.canvas.delete("ghost")
        self.canvas.configure(cursor="")
        self.update_status()

    def place_xy(self, e):
        """Top-left corner for the copy under the mouse, kept inside the board."""
        img = self.placing[0]
        h, w = img.shape[:2]
        H, W = self.board.shape[:2]
        x, y = self.board_xy(e)
        return max(0, min(W - w, x)), max(0, min(H - h, y))

    def on_motion(self, e):
        if not self.placing or self.board is None:
            return
        img = self.placing[0]
        z, c = self.zoom, self.canvas
        h, w = img.shape[:2]
        if self.ghost_zoom != z:  # the picture under the mouse, at the current zoom
            rgba = img if img.shape[2] == 4 else np.dstack([img, np.full((h, w), 255, np.uint8)])
            ghost = rgba.copy()
            ghost[..., 3] = np.where(rgba[..., 3] > 0, 200, 60)
            self.ghost = ImageTk.PhotoImage(Image.fromarray(ghost, "RGBA").resize(
                (max(1, int(w * z)), max(1, int(h * z))), Image.NEAREST))
            self.ghost_zoom = z
            c.delete("ghost")
            c.create_image(0, 0, anchor="nw", image=self.ghost, tags=("ghost", "ghostimg"))
            c.create_rectangle(0, 0, w * z, h * z, outline="yellow", width=2, tags=("ghost", "ghostbox"))
        x, y = self.place_xy(e)
        c.coords("ghostimg", x * z, y * z)
        c.coords("ghostbox", x * z - 1, y * z - 1, (x + w) * z, (y + h) * z)
        c.tag_raise("ghost")

    def finish_placing(self, e):
        img, commit, label, extra = self.placing
        x, y = self.place_xy(e)
        h, w = img.shape[:2]
        under = [z.note or z.kind for z in self.zones if z.enabled and overlap(z.rect, (x, y, x + w - 1, y + h - 1))]
        if not messagebox.askyesno(
                "Place a copy", f"Place {label} ({w}x{h}) with its top-left corner at ({x},{y})?\n\n{extra}" +
                (f"It covers {len(under)} zone(s) ({', '.join(under[:3])}): the new one wins there.\n\n" if under else "") +
                "The drawing paints it while it is enabled.", parent=self.root):
            return
        self.cancel_placing()
        commit(x, y)


def main():
    ap = argparse.ArgumentParser(description="The canvas control room: board, zones, artworks and drawing in one window.")
    ap.add_argument("--file", default=GUARDED, help="zone list (default guarded.txt next to this script)")
    ap.add_argument("--draw", action="store_true", help="start with drawing enabled")
    args = ap.parse_args()
    if tk is None:
        sys.exit("Tk is missing: sudo apt install python3-tk (macOS/Windows Pythons include it)")
    if ImageTk is None:
        sys.exit("Pillow's Tk support is missing: pip install -U pillow (or sudo apt install python3-pil.imagetk)")
    root = tk.Tk()
    CanvasUI(root, args.file, args.draw)
    root.mainloop()


if __name__ == "__main__":
    main()
