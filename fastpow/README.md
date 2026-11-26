# fastpow: Argon2id specialised for the canvas

The canvas makes every pixel cost Argon2id guesses with fixed parameters: `t=1`, `m=1024 KiB`,
`p=1`, version `0x13`, 32-byte tag, 15-byte password (x, y, rgb, nonce), 16-byte salt.
`argon2_canvas.c` computes exactly that, byte-identical to the reference, with everything
else removed, and runs the whole nonce search in C.

```bash
python3 fastpow/build.py
```
```bash
python3 paint_flag.py --bench
```

`build.py` compiles for your CPU (`-march=native`, clang or gcc), checks 40 random hashes
against PyNaCl or argon2-cffi, deletes the library if anything differs, and prints the speed.
`paint_flag.py`, `paint_croissant_flag.py` and `paint_heart.py` then use it automatically
(`hashing with fastpow C library (...)` in their first line). Without it they fall back to
PyNaCl (libsodium), then argon2-cffi.

## Benchmark (one core, Xeon 2.1 GHz with AVX-512, same input, µs per hash)

| implementation | µs/hash | hashes/s |
|---|---:|---:|
| **fastpow, clang, AVX-512** | **160** | **6260** |
| fastpow, gcc, AVX-512 | 174 | 5730 |
| libargon2 "opt", AVX-512 (`-march=native`) | 225 | 4450 |
| fastpow, AVX2 only | 227 | 4400 |
| libargon2 "opt", AVX2 only | 262 | 3810 |
| PyNaCl (libsodium wheel) | 281 | 3560 |
| libsodium built here (`-march=native`) | 361 | 2770 |
| fastpow, SSE4 (`x86-64-v2`) | 386 | 2590 |
| libargon2 "ref" (portable) | 449 | 2230 |
| Rust `argon2` (RustCrypto), `target-cpu=native` | 462 | 2165 |
| Python argon2-cffi (what the scripts used before) | 502 | 1990 |
| Python cryptography (OpenSSL 4) | 552 | 1810 |
| Rust `rust-argon2` | 626 | 1600 |
| Go `x/crypto/argon2` (SSE4 assembly) | 880 | 1140 |

Four threads give 4× (24.7k hashes/s on 4 cores): each hash's 1 MiB stays in the core's L2.

## What was stripped, and why it is still correct

- One pass, one lane, no XOR mode: `t=1, p=1` never needs them.
- The reference index of every block in the data-independent half (blocks 2..511) depends only
  on the parameters, never on the password or salt, so it is computed once at start-up.
- For the other half, `index_alpha()` reduces to `(c-2) - (((c-1) * ((J*J) >> 32)) >> 32)`.
- No allocation, zeroing or secure wipe per hash: one 1 MiB buffer per search, 2 MiB aligned
  so Linux can use a huge page. BLAKE2b's state wipes are turned off too.
- No argument checks, encoding or threads; the leading-zero-bits test is done in C.

The compression function itself is libargon2's own `fill_block()` (SSE2/AVX2/AVX-512), copied
unchanged; on non-x86 CPUs (Apple Silicon) the portable version is used.

GPUs don't help here: each guess needs 1 MiB of fast memory, which is what GPUs are short of.

The files in `libargon2/` are from https://github.com/P-H-C/phc-winner-argon2 (CC0 / Apache-2.0,
see `libargon2/LICENSE`).
