#!/usr/bin/env python3
"""Paint (and guard) a French flag on the canvas (https://canvas.bandwidthyt.com/api.html).

    pip install requests pynacl          (or argon2-cffi instead of pynacl)
    python3 fastpow/build.py             # optional, ~3x faster hashing: needs clang or gcc

    python3 paint_flag.py --bench         # show which hash backend is used and its speed
    python3 paint_flag.py                 # paint every pixel of the flag once
    python3 paint_flag.py --fix           # repaint only pixels that differ on the board
    python3 paint_flag.py --guard         # keep repairing damaged pixels until Ctrl-C

Options: --x 877 --y 491 --width 147   position/size (height = width * 2/3)
         --no-fox                      also paint the fox's sparkle area (see FOX_MASK)
         --workers N                   parallel workers (default: 6 per CPU, max 48)
         --max-bits 12                 leave pixels pricier than this for a later round
         --interval 30                 seconds between guard rounds
         --name tricolore              leaderboard name (or NAME env var)

The flag is 3:2 (width:height), split into three equal vertical bands:
blue 0055A4, white FFFFFF, red EF4135.

Why --guard only repairs and never repaints good pixels: every repaint of a pixel inside
the heat window (120 s) adds a bit to its price for *everyone*, us included. Repainting
pixels nobody touched doubles our own cost for nothing, and hot pixels cool by one bit per
window when left alone, so pricey fights are cheaper to win a little later.
"""
import argparse, ctypes, gzip, multiprocessing as mp, os, random, struct, sys, time

import requests

# Argon2id backends, fastest first: fastpow/ C library (built by fastpow/build.py, specialised for
# the canvas parameters), libsodium through PyNaCl, then argon2-cffi. All give identical hashes.
try:
    from nacl.pwhash import argon2id as _nacl
except ImportError:
    _nacl = None
try:
    from argon2.low_level import hash_secret_raw, Type
except ImportError:
    hash_secret_raw = None


def _load_fast():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastpow")
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
        lib.canvas_init()
        return lib
    return None


_fast = _load_fast()
if not (_fast or _nacl or hash_secret_raw):
    sys.exit("No Argon2 backend: pip install pynacl (or argon2-cffi), or run fastpow/build.py")


def backend():
    if _fast and _kib == 1024:
        return f"fastpow C library ({_fast.canvas_impl().decode()})"
    return "libsodium (PyNaCl)" if _nacl else "argon2-cffi"


def argon2id(pwd, salt):
    if _nacl:
        return _nacl.kdf(32, pwd, salt, opslimit=1, memlimit=_kib * 1024)
    return hash_secret_raw(pwd, salt, time_cost=1, memory_cost=_kib, parallelism=1, hash_len=32, type=Type.ID)

BASE = "https://canvas.bandwidthyt.com"
COLORS = ["0055a4", "ffffff", "ef4135"]

_session = None
_kib = 1024
_max_bits = 12
_name = "tricolore"


# Pixels left to the neighbouring fox art, whose sparkle reaches into the flag's bottom-left
# corner: the sparkle plus a 1 px margin. '#' = theirs, '.' = flag. The top-left character
# sits at FOX_AT, relative to the flag's top-left corner (board x 877, y 570 by default).
FOX_AT = (0, 79)
FOX_MASK = [
    "#........",
    "##.......",
    "###......",
    "########.",
    "########.",
    "########.",
    "#########",
    "..#######",
    "..#######",
    "..#######",
    "...#####.",
    "...#####.",
    "....###..",
]
FOX_PIXELS = {(FOX_AT[0] + dx, FOX_AT[1] + dy)
              for dy, row in enumerate(FOX_MASK) for dx, ch in enumerate(row) if ch == "#"}


def flag_pixels(ox, oy, width, skip=FOX_PIXELS):
    """Flag pixels, minus the (x, y) offsets in `skip`."""
    height = round(width * 2 / 3)
    band = width / 3
    for y in range(height):
        for x in range(width):
            if (x, y) in skip:
                continue
            yield ox + x, oy + y, COLORS[min(2, int(x // band))]


def solve(epoch, x, y, rgb, bits):
    prefix = struct.pack("<HH", x, y) + rgb
    nonce = random.getrandbits(62)  # random start: a repeat paint must not find a used solution
    if _fast and _kib == 1024:  # whole search loop in C, 65536 nonces per call
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
        if int.from_bytes(argon2id(prefix + struct.pack("<Q", nonce), epoch), "big") < limit:
            return nonce
        nonce += 1


def bench(seconds=2.0):
    """Single-core hash speed of each available backend (all must agree on a test vector)."""
    salt, pwd = bytes.fromhex("d972d4f67c1260e10efcfcdecef7e331"), bytes.fromhex("e903ee010055a415cd5b0700000000")
    want = bytes.fromhex("c2ecfec2fc63c1e3")
    rows = []
    if _fast:
        out, mem = ctypes.create_string_buffer(32), None
        _fast.canvas_alloc.restype = ctypes.c_void_p
        _fast.canvas_hash.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_void_p]
        mem = _fast.canvas_alloc()
        _fast.canvas_hash(pwd, 15, salt, out, mem)
        found, n, t = ctypes.c_uint64(), 0, time.perf_counter()
        while time.perf_counter() - t < seconds:
            _fast.canvas_search(pwd[:7], salt, 256, n, 500, ctypes.byref(found))
            n += 500
        rows.append((f"fastpow C library ({_fast.canvas_impl().decode()})", n / (time.perf_counter() - t),
                     out.raw[:8] == want))
    if _nacl:
        rows.append(("libsodium (PyNaCl)",) + _time(lambda p: _nacl.kdf(32, p, salt, opslimit=1,
                                                                         memlimit=1 << 20), pwd, want, seconds))
    if hash_secret_raw:
        rows.append(("argon2-cffi",) + _time(lambda p: hash_secret_raw(p, salt, time_cost=1, memory_cost=1024,
                                             parallelism=1, hash_len=32, type=Type.ID), pwd, want, seconds))
    cores = os.cpu_count() or 1
    for name, hps, ok in rows:
        print(f"{name:34s} {hps:7.0f} hashes/s per core  ~{hps * cores / 256:5.0f} fresh px/s on {cores} cores"
              f"  {'ok' if ok else 'WRONG RESULT'}")
    print("used:", backend())


def _time(f, pwd, want, seconds):
    ok = f(pwd)[:8] == want
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < seconds:
        f(pwd[:7] + struct.pack("<Q", n))
        n += 1
    return n / (time.perf_counter() - t), ok


def _init(kib, max_bits, name):
    global _session, _kib, _max_bits, _name
    _kib, _max_bits, _name = kib, max_bits, name
    _session = requests.Session()
    random.seed(os.urandom(16))


def paint_one(task):
    x, y, color = task
    rgb = bytes.fromhex(color)
    for attempt in range(12):
        try:
            c = _session.get(f"{BASE}/api/challenge", params={"x": x, "y": y}, timeout=20).json()
            if c["bits"] > _max_bits:  # too hot right now; it cools if left alone
                return (x, y, "deferred", c["bits"])
            nonce = solve(bytes.fromhex(c["epoch"]), x, y, rgb, c["bits"])
            r = _session.post(f"{BASE}/api/paint", timeout=20,
                              json={"x": x, "y": y, "color": color, "epoch": c["epoch"],
                                    "nonce": nonce, "name": _name})
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
        if r.status_code == 403:
            return (x, y, "closed", 0)
        time.sleep(1)
    return (x, y, "fail", 0)


def fetch_board():
    r = requests.get(f"{BASE}/api/board", timeout=60)
    r.raise_for_status()
    w = int(r.headers.get("x-canvas-size", "1024x1024").split("x")[0])
    data = r.content
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return w, data


def damaged(pixels):
    w, board = fetch_board()
    return [(x, y, c) for x, y, c in pixels
            if board[(y * w + x) * 3:(y * w + x) * 3 + 3] != bytes.fromhex(c)]


def run_pass(pool, tasks, label):
    random.shuffle(tasks)
    total, counts, t0 = len(tasks), {}, time.time()
    print(f"{label}: {total} pixels to paint", flush=True)
    for i, (x, y, status, bits) in enumerate(pool.imap_unordered(paint_one, tasks, chunksize=4), 1):
        counts[status] = counts.get(status, 0) + 1
        if status == "closed":
            sys.exit("board closed")
        if i % 250 == 0 or i == total:
            el = time.time() - t0
            print(f"  {i}/{total} {counts} {i / el:.1f} px/s eta {(total - i) / (i / el):.0f}s", flush=True)
    return counts


def add_common_args(ap, x, y):
    """Options shared by paint_flag.py and paint_croissant_flag.py."""
    ap.add_argument("--x", type=int, default=x)
    ap.add_argument("--y", type=int, default=y)
    ap.add_argument("--width", type=int, default=147)
    ap.add_argument("--no-fox", action="store_true",
                    help="also paint over the areas left to neighbours (fox art, Exil_S's heart)")
    ap.add_argument("--workers", type=int, default=min(48, 6 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    ap.add_argument("--fix", action="store_true", help="only repaint pixels that differ on the board")
    ap.add_argument("--guard", action="store_true", help="repair damaged pixels in a loop until Ctrl-C")
    ap.add_argument("--interval", type=float, default=30)


def main():
    ap = argparse.ArgumentParser(description="Paint and guard a French flag on the canvas.")
    add_common_args(ap, x=877, y=491)
    ap.add_argument("--bench", action="store_true", help="compare the Argon2 backends' speed and exit")
    args = ap.parse_args()
    if args.bench:
        return bench()
    skip = set() if args.no_fox else FOX_PIXELS
    pixels = list(flag_pixels(args.x, args.y, args.width, skip))
    run(pixels, args, f"flag {args.width}x{round(args.width * 2 / 3)} at ({args.x},{args.y}) "
                      f"minus {len(skip)} px left to the fox")


def run(pixels, args, what):
    """Paint `pixels` [(x, y, 'rrggbb')] once (--fix: only damaged ones), or guard them (--guard)."""
    cfg = requests.get(f"{BASE}/api/config", timeout=20).json()
    if cfg.get("read_only"):
        sys.exit("board is read-only")
    global _kib
    _kib = cfg["pow_kib"]
    print(f"{what}, name {args.name!r}, {args.workers} workers, hashing with {backend()}", flush=True)

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(cfg["pow_kib"], args.max_bits, args.name)) as pool:
        try:
            if not args.guard:
                tasks = damaged(pixels) if args.fix else list(pixels)
                print("done", run_pass(pool, tasks, "fix" if args.fix else "paint"), flush=True)
                return
            rnd = 0
            while True:
                rnd += 1
                try:
                    tasks = damaged(pixels)
                except requests.RequestException as e:
                    print(f"round {rnd}: board fetch failed ({e}), retrying", flush=True)
                    time.sleep(args.interval)
                    continue
                intact = 100 * (len(pixels) - len(tasks)) / len(pixels)
                stamp = time.strftime("%H:%M:%S")
                if tasks:
                    counts = run_pass(pool, tasks, f"[{stamp}] round {rnd}, {intact:.1f}% intact")
                    print(f"  repaired {counts.get('ok', 0)}, deferred {counts.get('deferred', 0)}", flush=True)
                else:
                    print(f"[{stamp}] round {rnd}: flag 100% intact", flush=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pool.terminate()
            print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
