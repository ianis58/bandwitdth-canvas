#!/usr/bin/env python3
"""Find the best --workers for this computer and connection, and save it: every paint script
then uses it by default (in fastpow/tuning.json, next to the hashing-thread count that
fastpow/build.py saves there). Run fastpow/build.py first if you use the fast C library.
Stop the other paint scripts while it runs: they share the CPU and would skew the test.

    python3 tune.py                      # ~8-10 min: measure, 2 min live trial per worker count, save
    python3 tune.py --levels 128 256 400 # compare exactly these worker counts
    python3 tune.py --rounds 2           # every count twice, in alternating order (cancels drift)
    python3 tune.py --quick              # ~10 s: estimate from CPU and network alone, paints nothing
    python3 tune.py --no-save            # show the result without saving it

What a worker is: one pixel in flight. Each pixel asks the server for its price (one round trip),
hashes until it finds a solution (CPU: at most one hash per hashing thread at a time, other
workers wait their turn), then sends it (a second round trip). While a worker waits on the
network, the CPU needs other workers' hashes to do. Too few workers: the CPU sits idle. Too
many: idle threads that only cost memory (about 1 MiB each once they have hashed).

Steps:
  1. CPU: hashes per second with every hashing thread busy, i.e. the most fresh pixels (8 bits)
     per second this computer can pay for.
  2. Network: round-trip time of a price request, alone and with many in flight.
  3. Estimate: hashing threads + pixels/s x time on the network per pixel, x2 (a paint request
     takes longer than a price request: the server checks the hash; too many workers costs
     little, too few costs speed).
  4. Live test (not with --quick): paints for --seconds (default 120) per worker count, keeping
     that many workers busy the whole time, and counts the pixels per second. Default counts:
     half, once and twice the estimate, plus more while the speed still climbs at an end.
     Before each trial 10 s are not counted (threads and connections start); before the first
     one 45 s, so a laptop CPU has dropped from its short turbo boost to the speed it keeps.
     Each trial is also cut into 30 s windows: their lowest and highest speeds show how much it
     varies by itself (other players' load on the server, the network), so differences smaller
     than that are noise.
     The pixels painted are intact pixels of our art (Justice picture, croissant flag, heart,
     what is left of the MERCI flag), with the colour they already have (nothing visibly changes;
     they get a bit of heat for a few minutes, which only makes them dearer for others to change).
     --area X0 Y0 X1 Y1 uses another area, with its current colours.
  5. Keeps the fewest workers within 5 % of the best speed.

Afterwards --workers N on any script, or the WORKERS environment variable, still overrides it.
Run it again if your connection changes (other place, other network) or after fastpow/build.py.
"""
import argparse, ctypes, json, os, random, statistics, struct, threading, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests

import paint_flag as engine

MAX_WORKERS = 1024


def hash_rate(threads, seconds=4.0):
    """Hashes per second with `threads` threads hashing at the same time."""
    salt, counts = os.urandom(16), [0] * threads
    start = threading.Barrier(threads + 1)

    def work(i):
        prefix, n = struct.pack("<HH", i, 0) + b"\x01\x02\x03", 0
        found = ctypes.c_uint64()
        start.wait()
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            if engine._fast:  # bits=256 never succeeds: pure hashing, 256 nonces per call
                engine._fast.canvas_search(prefix, salt, 256, n, 256, ctypes.byref(found))
                n += 256
            else:
                engine.argon2id(prefix + struct.pack("<Q", n), salt, 1024)
                n += 1
        counts[i] = n

    pool = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(threads)]
    for t in pool:
        t.start()
    start.wait()
    t0 = time.perf_counter()
    for t in pool:
        t.join()
    return sum(counts) / (time.perf_counter() - t0)


def round_trips(in_flight, total, size):
    """(median s, 90th percentile s, requests/s) of price requests with `in_flight` at once."""
    def one(_):
        t = time.perf_counter()
        try:
            engine._sess().get(f"{engine.BASE}/api/challenge", timeout=20,
                               params={"x": random.randrange(size), "y": random.randrange(size)}).raise_for_status()
        except requests.RequestException:
            return None
        return time.perf_counter() - t

    t0 = time.perf_counter()
    with ThreadPoolExecutor(in_flight) as pool:
        got = sorted(r for r in pool.map(one, range(total)) if r is not None)
    if not got:
        raise SystemExit("the server does not answer price requests")
    return statistics.median(got), got[int(0.9 * (len(got) - 1))], len(got) / (time.perf_counter() - t0)


def test_pixels(area):
    """Pixels to repaint with the colour they already show on the board."""
    w, raw = engine.fetch_board()

    def now(x, y):
        return raw[(y * w + x) * 3:(y * w + x) * 3 + 3].hex()

    if area:
        x0, y0, x1, y1 = area
        return [(x, y, now(x, y)) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    return [(x, y, c) for (x, y), c in our_art().items() if now(x, y) == c]  # intact ones only


def our_art():
    """{(x, y): colour} of all our art at its default place (later entries win where they overlap)."""
    import paint_croissant_flag as croissant, paint_heart, paint_image as image, paint_merci_flag
    art = {(x, y): c for x, y, c in paint_merci_flag.merci_pixels()}
    art.update({(x, y): c for x, y, c in paint_heart.heart_pixels(1001, 494)})
    art.update({(x, y): c for x, y, c in croissant.croissant_flag_pixels(787, 98, 237)})
    art.update({(x, y): c for x, y, c in image.image_pixels(image.DEFAULT_IMAGE, image.DEFAULT_RECT)})
    art.update({(x, y): c for x, y, c in image.qr_pixels(image.qr_matrix(image.DEFAULT_QR), image.DEFAULT_RECT)})
    return art


class Feed:
    """Endless supply of test pixels, shuffled; the list is re-read from the board when used up."""

    def __init__(self, area):
        self.area, self.items, self.i, self.loads = area, [], 0, 0

    def __call__(self):
        if self.i >= len(self.items):
            self.items, self.i = test_pixels(self.area), 0
            random.Random(self.loads).shuffle(self.items)
            self.loads += 1
            if not self.items:
                raise SystemExit("no intact test pixels left: use --area X0 Y0 X1 Y1")
        self.i += 1
        return self.items[self.i - 1]


def live_trial(workers, feed, seconds, warm, base_bits, window):
    """Paint for warm + seconds with `workers` always busy. Returns (px/s over the counted
    `seconds`, [px/s of each window], status counts). A pixel at b bits counts 2^(b - base)
    fresh ones, since hot pixels cost more work."""
    done, counts = [], {}
    t0 = time.perf_counter()
    start, end = t0 + warm, t0 + warm + seconds
    flying = set()
    with ThreadPoolExecutor(workers) as pool:
        while True:
            if time.perf_counter() < end:
                while len(flying) < 2 * workers:  # a queue a bit deeper than the workers: none waits for work
                    flying.add(pool.submit(engine.paint_one, feed()))
            else:  # time is up: drop the queued pixels; those being painted finish, uncounted
                for f in flying:
                    f.cancel()
                flying = {f for f in flying if not f.cancelled()}
                if not flying:
                    break
            finished, flying = wait(flying, timeout=1.0, return_when=FIRST_COMPLETED)
            now = time.perf_counter()
            for f in finished:
                _, _, status, bits = f.result()
                if start <= now < end:
                    counts[status] = counts.get(status, 0) + 1
                    if status == "ok":
                        done.append((now - start, 2.0 ** (bits - base_bits)))
    n = max(1, int(seconds // window))
    per = [0.0] * n
    for t, v in done:
        per[min(n - 1, int(t * n / seconds))] += v
    return sum(v for _, v in done) / seconds, [p * n / seconds for p in per], counts


def nice(n):
    return max(engine.CORES, min(MAX_WORKERS, int(round(n / 8.0)) * 8 or 8))


def main():
    ap = argparse.ArgumentParser(description="Find and save the best --workers for this computer.")
    ap.add_argument("--quick", action="store_true", help="estimate only (CPU + network), paint nothing")
    ap.add_argument("--no-save", action="store_true", help="do not write fastpow/tuning.json")
    ap.add_argument("--seconds", type=float, default=120, help="counted length of each live trial (default 120)")
    ap.add_argument("--levels", type=int, nargs="+", metavar="N", help="worker counts to compare (default: "
                    "half, once and twice the estimate, extended while the speed still climbs at an end)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="run every count this many times, in alternating order (cancels slow drift)")
    ap.add_argument("--window", type=float, default=30, help="seconds per window for the spread (default 30)")
    ap.add_argument("--preheat", type=float, default=45, help="uncounted seconds before the first trial")
    ap.add_argument("--area", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="live test on this area (repainted with its current colours) instead of the MERCI flag")
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()

    cfg = requests.get(f"{engine.BASE}/api/config", timeout=20).json()
    base, kib, size = int(cfg["base_bits"]), int(cfg["pow_kib"]), int(cfg["width"])
    engine._init(kib, base + 2, args.name)
    cores, current = engine.CORES, engine.default_workers()
    src = "HASH_THREADS" if os.environ.get("HASH_THREADS") else \
        ("fastpow/tuning.json" if "hash_threads" in engine._tuning() else "all CPU threads")
    print(f"hashing: {engine.backend()}, {cores} hashing threads ({src}); workers now: {current}", flush=True)

    # 1. CPU
    hps = hash_rate(cores)
    cpu_px = hps / 2 ** base
    print(f"1. CPU: {hps:,.0f} hashes/s -> at most {cpu_px:,.0f} fresh px/s ({base} bits); "
          f"at {base + 2} bits {cpu_px / 4:,.0f} px/s, at {base + 4} bits {cpu_px / 16:,.0f} px/s", flush=True)

    # 2. network
    alone = round_trips(1, 12, size)
    busy_n = nice(min(MAX_WORKERS, max(64, cpu_px)))
    busy = round_trips(busy_n, 3 * busy_n, size)
    print(f"2. network: price request {alone[0] * 1000:.0f} ms alone; with {busy_n} in flight "
          f"{busy[0] * 1000:.0f} ms (90 % under {busy[1] * 1000:.0f} ms), {busy[2]:,.0f} requests/s", flush=True)

    # 3. estimate: a pixel spends two round trips on the network; meanwhile `cores` workers hash
    per_px = 2 * busy[0]
    est = nice(cores + cpu_px * per_px * 2)
    print(f"3. estimate: {cores} hashing + {cpu_px:,.0f} px/s x {per_px:.2f} s on the network, x2 "
          f"-> {est} workers", flush=True)
    best, results = est, {}

    # 4. live test
    if not args.quick:
        feed = Feed(args.area)
        fixed = bool(args.levels)
        levels = sorted({max(1, min(MAX_WORKERS, n)) for n in args.levels}) if fixed else \
            sorted({nice(est * f) for f in (0.5, 1, 2)})
        trial_s = args.seconds + 10
        mins = (len(levels) * args.rounds * trial_s + args.preheat - 10) / 60
        print(f"4. live test: {levels} workers, {args.seconds:.0f} s counted each, {args.rounds} round(s): "
              f"about {mins:.0f} min{'' if fixed else ' (more if the speed still climbs at an end)'}; "
              f"repainting intact {'area' if args.area else 'art'} pixels with the colour they already have",
              flush=True)
        runs = {}  # workers -> [(px/s, [window px/s])]
        first = True

        def trial(w):
            nonlocal first
            warm = args.preheat if first else 10.0
            first = False
            rate, wins, counts = live_trial(w, feed, args.seconds, warm, base, args.window)
            runs.setdefault(w, []).append((rate, wins))
            print(f"   [{time.strftime('%H:%M:%S')}] {w:5d} workers: {rate:6.1f} px/s   "
                  f"({args.window:.0f} s windows: {min(wins):.0f}-{max(wins):.0f})   {counts}", flush=True)

        def mean(w):
            return statistics.mean(r for r, _ in runs[w])

        todo = list(levels)
        for extension in range(4):
            for r in range(args.rounds):
                for w in (todo if r % 2 == 0 else todo[::-1]):
                    trial(w)
            if fixed:
                break
            ranked = sorted(runs)
            top = max(ranked, key=mean)
            if top == ranked[-1] and top < MAX_WORKERS and mean(top) > 1.05 * mean(ranked[-2]):
                todo = [nice(top * 2)]  # still climbing at the top: try more
            elif top == ranked[0] and top > cores and mean(top) > 1.05 * mean(ranked[1]):
                todo = [nice(top / 2)]  # still climbing at the bottom: try fewer
            else:
                break
            if todo[0] in runs:
                break
            print(f"   speed still climbing at {top} workers: also trying {todo[0]}", flush=True)

        results = {w: round(mean(w), 1) for w in sorted(runs)}
        spread = statistics.median((max(wins) - min(wins)) / 2 / max(1e-9, statistics.mean(wins))
                                   for rs in runs.values() for _, wins in rs)
        top_rate = max(results.values())
        best = min(w for w, r in results.items() if r >= 0.95 * top_rate)  # fewest within 5 % of the best
        print("5. summary (mean px/s per worker count):", flush=True)
        for w, r in results.items():
            wins = [x for _, ws in runs[w] for x in ws]
            print(f"   {w:5d} workers: {r:6.1f} px/s  (windows {min(wins):.0f}-{max(wins):.0f})"
                  f"{'   <- best' if w == best else ''}", flush=True)
        print(f"   speed varies by about +-{spread * 100:.0f} % by itself inside a trial: smaller differences "
              f"between counts are noise", flush=True)
        if results[best] < 0.8 * cpu_px:
            print(f"   (below the CPU's {cpu_px:.0f} px/s measured in a short burst: the CPU slowing down "
                  f"under long load, the network or the server is the limit, not the worker count)", flush=True)

    if args.no_save:
        print(f"not saved; use --workers {best}")
        return
    tuning = engine._tuning()
    tuning.update({"workers": best, "workers_tuned": {
        "date": time.strftime("%Y-%m-%d %H:%M"), "estimate": est, "live_px_per_s": results,
        "trial_s": None if args.quick else args.seconds, "round_trip_ms": round(busy[0] * 1000),
        "cpu_fresh_px_per_s": round(cpu_px)}})
    os.makedirs(os.path.dirname(engine.TUNING_FILE), exist_ok=True)
    with open(engine.TUNING_FILE, "w") as f:
        json.dump(tuning, f, indent=1)
    print(f"saved workers = {best} in {engine.TUNING_FILE}: every script now uses it by default "
          f"(--workers N or WORKERS=N still override it)")


if __name__ == "__main__":
    main()
