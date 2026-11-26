#!/usr/bin/env python3
"""Paint a French flag on the canvas (https://canvas.bandwidthyt.com/api.html).

    pip install argon2-cffi requests numpy
    python3 paint_flag.py                      # paint the whole flag
    python3 paint_flag.py --fix                # repaint only pixels that differ on the board
    python3 paint_flag.py --x 877 --y 491 --width 147 --workers 4

The flag is 3:2 (width:height), split into three equal vertical bands:
blue 0055A4, white FFFFFF, red EF4135.
"""
import argparse, gzip, json, multiprocessing as mp, os, random, struct, sys, time

import requests
from argon2.low_level import hash_secret_raw, Type

BASE = "https://canvas.bandwidthyt.com"
COLORS = ["0055a4", "ffffff", "ef4135"]
NAME = os.environ.get("NAME", "tricolore")

_session = None
_kib = 1024


def flag_pixels(ox, oy, width):
    height = round(width * 2 / 3)
    band = width / 3
    for y in range(height):
        for x in range(width):
            yield ox + x, oy + y, COLORS[min(2, int(x // band))]


def solve(epoch, x, y, rgb, bits):
    prefix = struct.pack("<HH", x, y) + rgb
    limit = 1 << (256 - bits)
    nonce = random.getrandbits(62)
    while True:
        h = hash_secret_raw(prefix + struct.pack("<Q", nonce), epoch, time_cost=1, memory_cost=_kib,
                            parallelism=1, hash_len=32, type=Type.ID)
        if int.from_bytes(h, "big") < limit:
            return nonce
        nonce += 1


def _init(kib):
    global _session, _kib
    _kib = kib
    _session = requests.Session()
    random.seed(os.urandom(16))


def paint_one(task):
    x, y, color = task
    rgb = bytes.fromhex(color)
    for attempt in range(12):
        try:
            c = _session.get(f"{BASE}/api/challenge", params={"x": x, "y": y}, timeout=20).json()
            if c["bits"] > 14:  # far too contested; leave it for a later pass
                return (x, y, "skip", c["bits"])
            nonce = solve(bytes.fromhex(c["epoch"]), x, y, rgb, c["bits"])
            r = _session.post(f"{BASE}/api/paint", timeout=20,
                              json={"x": x, "y": y, "color": color, "epoch": c["epoch"],
                                    "nonce": nonce, "name": NAME})
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(min(30, 2 ** attempt))
            continue
        if r.status_code == 200:
            return (x, y, "ok", c["bits"])
        if r.status_code == 409:
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
    w, h = map(int, r.headers.get("x-canvas-size", "1024x1024").split("x"))
    data = r.content
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return w, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=int, default=877)
    ap.add_argument("--y", type=int, default=491)
    ap.add_argument("--width", type=int, default=147)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--fix", action="store_true", help="only repaint pixels that differ on the board")
    args = ap.parse_args()

    cfg = requests.get(f"{BASE}/api/config", timeout=20).json()
    if cfg.get("read_only"):
        sys.exit("board is read-only")
    tasks = list(flag_pixels(args.x, args.y, args.width))
    if args.fix:
        w, board = fetch_board()
        tasks = [(x, y, c) for x, y, c in tasks
                 if board[(y * w + x) * 3:(y * w + x) * 3 + 3] != bytes.fromhex(c)]
    random.shuffle(tasks)
    total = len(tasks)
    print(f"{total} pixels to paint as {NAME!r} with {args.workers} workers", flush=True)
    if not total:
        return

    counts, t0 = {}, time.time()
    with mp.Pool(args.workers, initializer=_init, initargs=(cfg["pow_kib"],)) as pool:
        for i, (x, y, status, bits) in enumerate(pool.imap_unordered(paint_one, tasks, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if status == "closed":
                pool.terminate()
                sys.exit("board closed")
            if i % 250 == 0 or i == total:
                el = time.time() - t0
                print(f"{i}/{total} {counts} {i / el:.1f} px/s eta {(total - i) / (i / el):.0f}s", flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
