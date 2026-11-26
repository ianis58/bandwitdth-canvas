#!/usr/bin/env python3
"""Find the best --workers for this computer and connection, and save it: every paint script
then uses it by default (in fastpow/tuning.json, next to the hashing-thread count that
fastpow/build.py saves there). Run fastpow/build.py first if you use the fast C library.

    python3 tune.py            # ~2-3 min: measure, live test, save
    python3 tune.py --quick    # ~20 s: estimate from CPU and network alone, paints nothing
    python3 tune.py --no-save  # show the result without saving it

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
  4. Live test (not with --quick): repaints pixels of our MERCI flag that are intact with the
     colour they already have (nothing visibly changes), with several worker counts around the
     estimate, ~15 s each, and measures the pixels per second really reached. Those pixels get
     one bit of heat for a few minutes, which only makes them dearer for others to change.
     --area X0 Y0 X1 Y1 tests on another area instead (repainted with its current colours).
  5. Keeps the fewest workers within 5 % of the best speed.

Afterwards --workers N on any script, or the WORKERS environment variable, still overrides it.
Run it again if your connection changes (other place, other network) or after fastpow/build.py.
"""
import argparse, ctypes, json, os, random, statistics, struct, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    import paint_merci_flag
    return [(x, y, c) for x, y, c in paint_merci_flag.merci_pixels() if now(x, y) == c]  # intact ones only


def live_trial(workers, tasks, base_bits):
    """(fresh-pixel equivalents per second in the steady part, status counts)."""
    done, counts = [], {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(engine.paint_one, t) for t in tasks]):
            _, _, status, bits = fut.result()
            counts[status] = counts.get(status, 0) + 1
            # a pixel at b bits is worth 2^(b - base) fresh ones (hot pixels cost more)
            done.append((time.perf_counter() - t0, 2.0 ** (bits - base_bits) if status == "ok" else 0.0))
    # steady state only: between the 10 % and 85 % completion marks (not the start-up, not the tail)
    a, b = len(done) // 10, int(len(done) * 0.85)
    span = done[b][0] - done[a][0]
    return (sum(v for _, v in done[a + 1:b + 1]) / span if span > 0 else 0.0), counts


def nice(n):
    return max(engine.CORES, min(MAX_WORKERS, int(round(n / 8.0)) * 8 or 8))


def main():
    ap = argparse.ArgumentParser(description="Find and save the best --workers for this computer.")
    ap.add_argument("--quick", action="store_true", help="estimate only (CPU + network), paint nothing")
    ap.add_argument("--no-save", action="store_true", help="do not write fastpow/tuning.json")
    ap.add_argument("--seconds", type=float, default=15, help="length of each live trial")
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
        pool = test_pixels(args.area)
        random.Random(1).shuffle(pool)
        levels = sorted({nice(est * f) for f in (0.25, 0.5, 1, 2)})
        print(f"4. live test on {len(pool):,} {'area' if args.area else 'intact MERCI flag'} pixels, "
              f"repainted with the colour they already have, ~{args.seconds:.0f} s per trial:", flush=True)
        used = 0
        i = 0
        while i < len(levels):
            w = levels[i]
            expect = min(cpu_px, w / per_px)  # Little's law: w in flight, each ~per_px seconds
            n = int(min(6000, max(3 * w, expect * args.seconds)))
            if used + n > len(pool):  # area used up: start again (those pixels are a bit hotter)
                used = 0
                pool = test_pixels(args.area)
                random.Random(i + 2).shuffle(pool)
            rate, counts = live_trial(w, pool[used:used + n], base)
            used += n
            results[w] = rate
            print(f"   {w:5d} workers: {rate:6.1f} px/s   {counts}", flush=True)
            i += 1
            if i == len(levels):  # still rising at the top (or bottom) of the range: look further
                top = max(results, key=results.get)
                ranked = sorted(results)
                if top == ranked[-1] and len(ranked) > 1 and results[top] > 1.05 * results[ranked[-2]] \
                        and top < MAX_WORKERS:
                    levels.append(nice(top * 2))
                elif top == ranked[0] and top > cores and len(ranked) > 1 \
                        and results[top] > 1.05 * results[ranked[1]]:
                    levels.append(nice(top / 2))
                levels = list(dict.fromkeys(levels))
        top_rate = max(results.values())
        best = min(w for w, r in results.items() if r >= 0.95 * top_rate)  # fewest within 5 % of the best
        print(f"5. best: {best} workers ({results[best]:.0f} px/s; top speed {top_rate:.0f} px/s)", flush=True)
        if results[best] < 0.8 * cpu_px:
            print(f"   (below the CPU's {cpu_px:.0f} px/s: the network or the server is the limit here, "
                  f"not your CPU; more workers did not help)", flush=True)

    if args.no_save:
        print(f"not saved; use --workers {best}")
        return
    tuning = engine._tuning()
    tuning.update({"workers": best, "workers_tuned": {
        "date": time.strftime("%Y-%m-%d %H:%M"), "estimate": est, "live_px_per_s": results,
        "round_trip_ms": round(busy[0] * 1000), "cpu_fresh_px_per_s": round(cpu_px)}})
    os.makedirs(os.path.dirname(engine.TUNING_FILE), exist_ok=True)
    with open(engine.TUNING_FILE, "w") as f:
        json.dump(tuning, f, indent=1)
    print(f"saved workers = {best} in {engine.TUNING_FILE}: every script now uses it by default "
          f"(--workers N or WORKERS=N still override it)")


if __name__ == "__main__":
    main()
