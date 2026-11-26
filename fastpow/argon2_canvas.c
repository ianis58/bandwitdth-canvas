/* Argon2id specialised for the canvas proof of work: t=1, m=1024 KiB, p=1, v=0x13, 32-byte tag,
 * empty secret and associated data. Output is identical to argon2id_hash_raw() with those
 * parameters (checked by test_canvas.c).
 *
 * Built on libargon2 (https://github.com/P-H-C/phc-winner-argon2, CC0/Apache-2.0, vendored in
 * libargon2/): its BLAKE2b and its fill_block() compression, copied unchanged from opt.c
 * (SSE2/AVX2/AVX-512, fill_block_opt.inc) and ref.c (portable, fill_block_ref.inc).
 * What is removed versus argon2id_hash_raw():
 *   - argument validation, encoding, threads, lanes, multiple passes, XOR mode (pass 0 only)
 *   - the 1 MiB allocation per hash, and secure wipes (of that memory and of BLAKE2b state)
 *   - address-block generation: with t=1 the reference indices of the data-independent half
 *     (blocks 2..511) never change, so they are computed once in canvas_init()
 * canvas_search() runs the whole nonce loop in C, including the leading-zero-bits test.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "argon2.h"
#include "core.h"
#include "blake2/blake2.h"

#define MBLOCKS 1024u
#define SEGLEN (MBLOCKS / ARGON2_SYNC_POINTS)

#if defined(__SSE2__) || defined(__x86_64__)
#include "blake2/blamka-round-opt.h"
#include "fill_block_opt.inc"
#if defined(__AVX512F__)
typedef __m512i vec_t;
#define NVEC ARGON2_512BIT_WORDS_IN_BLOCK
#elif defined(__AVX2__)
typedef __m256i vec_t;
#define NVEC ARGON2_HWORDS_IN_BLOCK
#else
typedef __m128i vec_t;
#define NVEC ARGON2_OWORDS_IN_BLOCK
#endif
#define SIMD 1
#else
#include "blake2/blamka-round-ref.h"
#include "fill_block_ref.inc"
#define SIMD 0
#endif

static uint16_t REF[SEGLEN * 2];
static volatile int ref_ready;

static inline uint32_t ref_index(uint32_t c, uint32_t j1) {
    /* index_alpha() for pass 0, one lane: reference area = all blocks but the previous one */
    uint64_t rel = (uint64_t)j1 * j1 >> 32;
    return (c - 2) - (uint32_t)(((uint64_t)(c - 1) * rel) >> 32);
}

static void next_addresses(block *address, block *input) {
    input->v[6]++;
#if SIMD
    vec_t z1[NVEC], z2[NVEC];
    memset(z1, 0, sizeof z1);
    memset(z2, 0, sizeof z2);
    fill_block(z1, input, address, 0);
    fill_block(z2, address, address, 0);
#else
    block zero;
    memset(&zero, 0, sizeof zero);
    fill_block(&zero, input, address, 0);
    fill_block(&zero, address, address, 0);
#endif
}

const char *canvas_impl(void) {
#if defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__SSSE3__)
    return "ssse3";
#elif SIMD
    return "sse2";
#else
    return "portable";
#endif
}

#ifdef CANVAS_WITH_LIBARGON2
extern int FLAG_clear_internal_memory;  /* libargon2 core.c: wipes BLAKE2b state and buffers */
#else
/* Standalone build: libargon2's BLAKE2b calls this to wipe its state; nothing to protect here. */
int FLAG_clear_internal_memory = 0;
void clear_internal_memory(void *v, size_t n) { (void)v; (void)n; }
#endif

void canvas_init(void) {
    if (ref_ready) return;
    FLAG_clear_internal_memory = 0;
    block address, input;
    for (uint32_t slice = 0; slice < 2; slice++) {
        memset(&input, 0, sizeof input);
        input.v[2] = slice;
        input.v[3] = MBLOCKS;
        input.v[4] = 1;          /* passes */
        input.v[5] = Argon2_id;
        uint32_t start = slice == 0 ? 2 : 0;
        if (slice == 0) next_addresses(&address, &input);
        for (uint32_t i = start; i < SEGLEN; i++) {
            if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) next_addresses(&address, &input);
            uint32_t c = slice * SEGLEN + i;
            REF[c] = (uint16_t)ref_index(c, (uint32_t)address.v[i % ARGON2_ADDRESSES_IN_BLOCK]);
        }
    }
    ref_ready = 1;
}

/* Work memory: 1 MiB, 2 MiB-aligned so the kernel can back it with one huge page. */
void *canvas_alloc(void) {
    void *p = NULL;
    if (posix_memalign(&p, 2u << 20, 2u << 20)) return NULL;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    madvise(p, 2u << 20, MADV_HUGEPAGE);
#endif
    memset(p, 0, MBLOCKS * ARGON2_BLOCK_SIZE);
    return p;
}

void canvas_free(void *p) { free(p); }

static inline void put32(blake2b_state *S, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    blake2b_update(S, b, 4);
}

/* mem: from canvas_alloc(). Host must be little-endian (x86, ARM). */
void canvas_hash(const uint8_t *pwd, uint32_t pwdlen, const uint8_t salt[16], uint8_t out[32], void *memv) {
    block *mem = (block *)memv;
    uint8_t seed[ARGON2_PREHASH_SEED_LENGTH];
    blake2b_state S;
    blake2b_init(&S, ARGON2_PREHASH_DIGEST_LENGTH);
    put32(&S, 1);                 /* lanes */
    put32(&S, 32);                /* tag length */
    put32(&S, MBLOCKS);           /* m_cost */
    put32(&S, 1);                 /* t_cost */
    put32(&S, ARGON2_VERSION_13);
    put32(&S, Argon2_id);
    put32(&S, pwdlen);
    blake2b_update(&S, pwd, pwdlen);
    put32(&S, 16);
    blake2b_update(&S, salt, 16);
    put32(&S, 0);                 /* secret */
    put32(&S, 0);                 /* associated data */
    blake2b_final(&S, seed, ARGON2_PREHASH_DIGEST_LENGTH);
    memset(seed + 64, 0, 8);
    blake2b_long(mem[0].v, ARGON2_BLOCK_SIZE, seed, sizeof seed);
    seed[64] = 1;
    blake2b_long(mem[1].v, ARGON2_BLOCK_SIZE, seed, sizeof seed);

#if SIMD
    vec_t state[NVEC];
    memcpy(state, mem[1].v, ARGON2_BLOCK_SIZE);
    for (uint32_t c = 2; c < 2 * SEGLEN; c++)
        fill_block(state, &mem[REF[c]], &mem[c], 0);
    for (uint32_t c = 2 * SEGLEN; c < MBLOCKS; c++)
        fill_block(state, &mem[ref_index(c, (uint32_t)mem[c - 1].v[0])], &mem[c], 0);
#else
    for (uint32_t c = 2; c < 2 * SEGLEN; c++)
        fill_block(&mem[c - 1], &mem[REF[c]], &mem[c], 0);
    for (uint32_t c = 2 * SEGLEN; c < MBLOCKS; c++)
        fill_block(&mem[c - 1], &mem[ref_index(c, (uint32_t)mem[c - 1].v[0])], &mem[c], 0);
#endif
    blake2b_long(out, 32, mem[MBLOCKS - 1].v, ARGON2_BLOCK_SIZE);
}

static inline int zero_bits_ok(const uint8_t *h, int bits) {
    int i = 0;
    for (; bits >= 8; bits -= 8, i++)
        if (h[i]) return 0;
    return bits == 0 || (h[i] >> (8 - bits)) == 0;
}

/* Try nonces start .. start+count-1 for password prefix (x u16le, y u16le, r, g, b).
 * Returns 1 and sets *found on success, 0 if none of them works. Thread-safe. */
int canvas_search(const uint8_t prefix[7], const uint8_t salt[16], int bits,
                  uint64_t start, uint64_t count, uint64_t *found) {
    canvas_init();
    void *mem = canvas_alloc();
    if (!mem) return -1;
    uint8_t pwd[15], out[32];
    memcpy(pwd, prefix, 7);
    int ok = 0;
    for (uint64_t n = start; n - start < count; n++) {
        for (int i = 0; i < 8; i++) pwd[7 + i] = (uint8_t)(n >> (8 * i));
        canvas_hash(pwd, 15, salt, out, mem);
        if (zero_bits_ok(out, bits)) {
            *found = n;
            ok = 1;
            break;
        }
    }
    canvas_free(mem);
    return ok;
}
