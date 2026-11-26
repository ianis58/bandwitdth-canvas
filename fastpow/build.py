#!/usr/bin/env python3
"""Build the canvas-specialised Argon2id library for this machine's CPU, check it, benchmark it.

    python3 fastpow/build.py          (needs clang or gcc; CC=... to choose)

Every available compiler is built with and without reference-block prefetching, each build is
timed on this CPU the way the scripts use it, and the fastest is kept. The library is only kept
if its hashes match a reference implementation (PyNaCl or argon2-cffi).
paint_flag.py and the other scripts pick it up automatically from this folder.
"""
import ctypes, os, platform, random, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LIBNAME = {"Darwin": "libargon2_canvas.dylib", "Windows": "argon2_canvas.dll"}.get(
    platform.system(), "libargon2_canvas.so")


def _speed(path, seconds=1.2):
    """Hashes/s of a built library, used the way the scripts use it: one search per 8-bit pixel."""
    lib = ctypes.CDLL(path)
    lib.canvas_search.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64,
                                  ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    lib.canvas_init()
    found, start, hashes, salt = ctypes.c_uint64(), 1, 0, os.urandom(16)
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        lib.canvas_search(b"\xe9\x03\xee\x01\x00\x55\xa4", salt, 8, start, 1 << 20, ctypes.byref(found))
        hashes += found.value - start + 1
        start = found.value + 1
    return hashes / (time.perf_counter() - t)


def compile_lib():
    """Build every available compiler x (prefetch off/on), time each here, keep the fastest."""
    out = os.path.join(HERE, LIBNAME)
    srcs = [os.path.join(HERE, "argon2_canvas.c"), os.path.join(HERE, "libargon2", "src", "blake2", "blake2b.c")]
    incs = ["-I" + os.path.join(HERE, *p) for p in ([], ["libargon2", "include"], ["libargon2", "src"])]
    x86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686", "x86")
    arch = ["-march=native"] if x86 else []  # non-x86 (e.g. Apple Silicon): portable C path
    pic = [] if platform.system() == "Windows" else ["-fPIC"]
    compilers, seen = [], set()
    for cc in ([os.environ["CC"]] if os.environ.get("CC") else ["clang", "gcc", "cc"]):
        real = shutil.which(cc) and os.path.realpath(shutil.which(cc))
        if real and real not in seen:  # "cc" is often just gcc again
            seen.add(real)
            compilers.append(cc)
    built = []
    for cc in compilers:
        for prefetch in ([False, True] if x86 else [False]):
            path = os.path.join(HERE, f".try_{cc}_{int(prefetch)}_{LIBNAME}")
            cmd = [cc, "-O3", *arch, *(["-DCANVAS_PREFETCH"] if prefetch else []), "-shared", *pic, "-o", path,
                   *incs, *srcs]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                built.append((cc, prefetch, path))
            elif not prefetch:
                print(f"{cc} failed:\n{r.stderr[-1500:]}")
    if not built:
        sys.exit("No working C compiler found: install clang or gcc (on Windows: MSYS2/MinGW-w64 or WSL).")
    speeds = {b: [] for b in built}
    for _ in range(3):  # interleaved rounds: fairer than timing one build after the other
        for b in built:
            speeds[b].append(_speed(b[2]))
    best = max(built, key=lambda b: sorted(speeds[b])[1])
    for b in built:
        mark = "  <- kept" if b == best else ""
        print(f"  {b[0]:6s} prefetch {'on ' if b[1] else 'off'}  {sorted(speeds[b])[1]:7.0f} hashes/s{mark}")
    shutil.copyfile(best[2], out)
    for b in built:
        try:
            os.remove(b[2])
        except OSError:
            pass  # Windows keeps loaded DLLs locked; harmless leftovers
    return out, f"{best[0]}{', prefetch' if best[1] else ''}"


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
