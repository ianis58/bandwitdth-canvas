#!/usr/bin/env python3
"""Build the canvas-specialised Argon2id library for this machine's CPU, check it, benchmark it.

    python3 fastpow/build.py          (needs clang or gcc; CC=... to choose)

Every available compiler is built with and without reference-block prefetching (and, on AVX-512
CPUs, also AVX2-only, since some clock down under AVX-512), each build is timed on this CPU with
all threads busy, and the fastest is kept. Then 1 vs 2 hashing threads per core is measured
(Hyper-Threading) and the better count saved in tuning.json, which the paint scripts follow. The library is only kept
if its hashes match a reference implementation (PyNaCl or argon2-cffi).
paint_flag.py and the other scripts pick it up automatically from this folder.
"""
import ctypes, json, os, platform, random, shutil, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
LIBNAME = {"Darwin": "libargon2_canvas.dylib", "Windows": "argon2_canvas.dll"}.get(
    platform.system(), "libargon2_canvas.so")


def physical_cores():
    """Physical cores (Hyper-Threading counts 2 threads per core in os.cpu_count())."""
    try:
        out = subprocess.run(["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True, timeout=5).stdout
        cores = {line for line in out.splitlines() if line and not line.startswith("#")}
        if cores:
            return len(cores)
    except (OSError, subprocess.SubprocessError):
        pass
    return os.cpu_count() or 1


def _speed(path, threads, seconds=1.5):
    """Total hashes/s of a built library with `threads` threads hashing at once, each used the
    way the scripts use it (one search per 8-bit pixel). Under full load is what matters: some
    CPUs lower their clock for AVX-512 on all cores, and two threads on one core share its cache."""
    lib = ctypes.CDLL(path)
    lib.canvas_search.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64,
                                  ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    lib.canvas_init()
    salt, done = os.urandom(16), [0] * threads
    stop_at = time.perf_counter() + seconds

    def work(i):
        found, start, n = ctypes.c_uint64(), 1 + (i << 40), 0
        while time.perf_counter() < stop_at:
            lib.canvas_search(b"\xe9\x03\xee\x01\x00\x55\xa4", salt, 8, start, 1 << 20, ctypes.byref(found))
            n += found.value - start + 1
            start = found.value + 1
        done[i] = n

    t = time.perf_counter()
    pool = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for th in pool:
        th.start()
    for th in pool:
        th.join()
    return sum(done) / (time.perf_counter() - t)


def compile_lib():
    """Build every compiler x instruction set (native, and AVX2-only on AVX-512 CPUs) x prefetch,
    time each with all threads busy, keep the fastest; then choose how many threads should hash
    at once (physical cores or all threads) and save it in tuning.json for the paint scripts."""
    out = os.path.join(HERE, LIBNAME)
    srcs = [os.path.join(HERE, "argon2_canvas.c"), os.path.join(HERE, "libargon2", "src", "blake2", "blake2b.c")]
    incs = ["-I" + os.path.join(HERE, *p) for p in ([], ["libargon2", "include"], ["libargon2", "src"])]
    x86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686", "x86")
    pic = [] if platform.system() == "Windows" else ["-fPIC"]
    compilers, seen = [], set()
    for cc in ([os.environ["CC"]] if os.environ.get("CC") else ["clang", "gcc", "cc"]):
        real = shutil.which(cc) and os.path.realpath(shutil.which(cc))
        if real and real not in seen:  # "cc" is often just gcc again
            seen.add(real)
            compilers.append(cc)
    isas = {"native": ["-march=native"] if x86 else []}  # non-x86 (e.g. Apple Silicon): portable C path
    built = []

    def build(cc, isa, prefetch):
        path = os.path.join(HERE, f".try_{cc}_{isa}_{int(prefetch)}_{LIBNAME}")
        cmd = [cc, "-O3", *isas[isa], *(["-DCANVAS_PREFETCH"] if prefetch else []), "-shared", *pic, "-o", path,
               *incs, *srcs]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            built.append((cc, isa, prefetch, path))
        elif isa == "native" and not prefetch:
            print(f"{cc} failed:\n{r.stderr[-1500:]}")

    for cc in compilers:
        build(cc, "native", False)
    if not built:
        sys.exit("No working C compiler found: install clang or gcc (on Windows: MSYS2/MinGW-w64 or WSL).")
    probe = ctypes.CDLL(built[0][3])
    probe.canvas_impl.restype = ctypes.c_char_p
    if probe.canvas_impl() == b"avx512":
        isas["avx2"] = ["-march=native", "-mno-avx512f"]  # some CPUs clock down under AVX-512: compare
    for cc in compilers:
        for isa in isas:
            for prefetch in ([False, True] if x86 else [False]):
                if not (isa == "native" and not prefetch):
                    build(cc, isa, prefetch)

    logical, physical = os.cpu_count() or 1, physical_cores()
    print(f"timing {len(built)} builds with {logical} threads busy ({physical} physical cores)...")
    speeds = {b: [] for b in built}
    for _ in range(2):  # interleaved rounds: fairer than timing one build after the other
        for b in built:
            speeds[b].append(_speed(b[3], logical))
    best = max(built, key=lambda b: sum(speeds[b]))
    for b in built:
        mark = "  <- kept" if b == best else ""
        print(f"  {b[0]:6s} {b[1]:6s} prefetch {'on ' if b[2] else 'off'}  {sum(speeds[b]) / 2:8.0f} hashes/s{mark}")
    threads = logical
    if physical < logical:  # Hyper-Threading: do two hashes per core help, or fight over its cache?
        per = {n: max(_speed(best[3], n, 2.0) for _ in range(2)) for n in (physical, logical)}
        threads = max(per, key=per.get)
        print(f"  hashing threads: {physical} -> {per[physical]:.0f} hashes/s, {logical} -> {per[logical]:.0f} hashes/s"
              f"  -> using {threads}")
    try:  # keep what other tools saved there (tune.py's workers)
        with open(os.path.join(HERE, "tuning.json")) as f:
            tuning = json.load(f)
    except (OSError, ValueError):
        tuning = {}
    tuning.update({"hash_threads": threads, "logical_cpus": logical, "physical_cores": physical,
                   "build": f"{best[0]} {best[1]}{' prefetch' if best[2] else ''}",
                   "hashes_per_s": round(sum(speeds[best]) / 2)})
    with open(os.path.join(HERE, "tuning.json"), "w") as f:
        json.dump(tuning, f, indent=1)
    shutil.copyfile(best[3], out)
    for b in built:
        try:
            os.remove(b[3])
        except OSError:
            pass  # Windows keeps loaded DLLs locked; harmless leftovers
    return out, f"{best[0]}, {best[1]}{', prefetch' if best[2] else ''}, {threads} hashing threads"


def reference():
    try:
        from nacl.pwhash import argon2id
        return "PyNaCl", lambda p, s: argon2id.kdf(32, p, s, opslimit=1, memlimit=1024 * 1024)
    except ImportError:
        pass
    try:
        from argon2.low_level import hash_secret_raw, Type
        return "argon2-cffi", lambda p, s: hash_secret_raw(p, s, time_cost=1, memory_cost=1024, parallelism=1,
                                                         hash_len=32, type=Type.ID)
    except ImportError:
        sys.exit("Install pynacl or argon2-cffi to check the library: pip install pynacl")


def main():
    path, cc = compile_lib()
    lib = ctypes.CDLL(path)
    lib.canvas_impl.restype = ctypes.c_char_p
    lib.canvas_alloc.restype = ctypes.c_void_p
    lib.canvas_hash.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    lib.canvas_search.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                                  ctypes.POINTER(ctypes.c_uint64)]
    lib.canvas_init()
    impl = lib.canvas_impl().decode()
    print(f"built {os.path.basename(path)} with {cc} ({impl})")

    name, ref = reference()
    mem, out = lib.canvas_alloc(), ctypes.create_string_buffer(32)
    for i in range(40):
        pwd = os.urandom(15 if i % 4 else random.randrange(40))
        salt = os.urandom(16)
        lib.canvas_hash(pwd, len(pwd), salt, out, mem)
        if out.raw != ref(pwd, salt):
            os.remove(path)
            sys.exit(f"MISMATCH against {name}: library removed, the scripts will use {name} instead.")
    print(f"check: 40 random hashes identical to {name}")

    found, n = ctypes.c_uint64(), 2000
    t = time.perf_counter()
    lib.canvas_search(b"\xe9\x03\xee\x01\x00\x55\xa4", os.urandom(16), 256, 0, n, ctypes.byref(found))
    fast = n / (time.perf_counter() - t)
    t, m = time.perf_counter(), 200
    for i in range(m):
        ref(os.urandom(15), salt)
    slow = m / (time.perf_counter() - t)
    print(f"speed, 1 core: {fast:.0f} hashes/s  vs {name} {slow:.0f} hashes/s  ({fast / slow:.1f}x)")
    print(f"~{fast * (os.cpu_count() or 1) / 256:.0f} fresh pixels/s with all {os.cpu_count()} cores (8 bits each)")


if __name__ == "__main__":
    main()
