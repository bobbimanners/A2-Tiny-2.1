/* Unit tests for the assembly kernels, run in the LLVM-MOS simulator.
 * Each routine is compared with plain C arithmetic on 64-bit integers.
 * Build and run from the v0.2 directory:
 *   mos-sim-clang -mcpu=mos65c02 -Os -mreserve-zp=30 -Wa,--defsym,TINY_MAC_GENERAL=1 -Wa,--defsym,TINY_MAC_SIM=1 -I. tests/tiny-asm-test.c \
 *     tiny-wide-mos.s tiny-mac65-mos.s tiny-qsquare-mos.s -o asm-test.sim
 *   mos-sim --cmos asm-test.sim
 * The simulator has flat memory. TINY_MAC_SIM directs the Apple II
 * soft-switch accesses to a private buffer in that memory. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "tiny-wide.h"

const int8_t *tiny_mac_a, *tiny_mac_b;
uint16_t tiny_mac_n;
int32_t tiny_mac_result;
int8_t tiny_mac_coeff;
int32_t *tiny_mac_acc;
extern void tiny_mac65_prepare(void);
extern void tiny_mac65_prepared_dot(void);
extern void tiny_mac65_aux_dot(void);
extern void tiny_mac65_aux_rows(void);
extern void tiny_att_batch(void);
typedef struct { uint16_t m; int16_t exp; } PackedScale;
TinyScoreArgs tiny_att_args;
Wide *tiny_att_out;
int16_t *tiny_att_exps;
PackedScale *tiny_att_ks;
int16_t tiny_att_base, tiny_att_hi;
uint8_t tiny_att_first;
extern void tiny_mac65_axpy_begin(void);
extern void tiny_mac65_aux_axpy(void);
extern void tiny_mac65_axpy_end(void);
uint16_t tiny_mac_stride;
uint8_t tiny_mac_rows;
extern int8_t tiny_mac65_work[512];
extern void tiny_mac65_init(void);

static uint32_t state = 0x6502c0deUL;
static uint32_t rnd(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static uint64_t rnd64(void) { return ((uint64_t)rnd() << 32) | rnd(); }
static uint64_t u64(const Wide *a) { return ((uint64_t)a->hi << 32) | a->lo; }
static void set(Wide *a, uint64_t v) { a->lo = (uint32_t)v; a->hi = (uint32_t)(v >> 32); }
static unsigned long fails, checks;
static void expect(const char *what, uint64_t got, uint64_t want, long tag) {
    checks++;
    if (got == want) return;
    if (++fails <= 12)
        printf("FAIL %s #%ld: got %08lx%08lx want %08lx%08lx\n", what, tag,
               (unsigned long)(got >> 32), (unsigned long)got,
               (unsigned long)(want >> 32), (unsigned long)want);
}

static void test_wm(void) {
    static const uint8_t widths[] = {1, 8, 9, 16, 24, 31, 32, 33, 48, 63, 64};
    Wide a, b, r;
    unsigned i, j, k;
    for (i = 0; i < sizeof widths; i++)
        for (j = 0; j < sizeof widths; j++)
            for (k = 0; k < 6; k++) {
                uint64_t x = rnd64(), y = rnd64();
                if (widths[i] < 64) x &= ((uint64_t)1 << widths[i]) - 1;
                if (widths[j] < 64) y &= ((uint64_t)1 << widths[j]) - 1;
                if (k == 1) x = ((uint64_t)1 << (widths[i] - 1)) | (x >> 1) | 0xff;
                if (k & 2) x = 0 - x;
                if (k & 4) y = 0 - y;
                set(&a, x); set(&b, y);
                wm(&r, &a, &b);
                expect("wm", u64(&r), x * y, (long)(i * 100 + j * 10 + k));
                wm(&a, &a, &b);                 /* exact alias r == a */
                expect("wm alias", u64(&a), x * y, (long)(i * 100 + j * 10 + k));
            }
    /* Long carry chains. */
    set(&a, 0xffffffffffffffffULL); set(&b, 0xffffffffffffffffULL); wm(&r, &a, &b);
    expect("wm carry", u64(&r), 1, 0);
    set(&a, 0x00ffffffffffffffULL); set(&b, 0x0101010101010101ULL); wm(&r, &a, &b);
    expect("wm carry", u64(&r), 0x00ffffffffffffffULL * 0x0101010101010101ULL, 1);
}

static void test_rescale(void) {
    TinyRescaleArgs args;
    uint8_t scale[4];
    Wide r;
    unsigned i;
    for (i = 0; i < 600; i++) {
        int32_t dot = (int32_t)(rnd() >> (9 + i % 17));   /* signed 24 bits */
        uint32_t s = rnd() >> (8 + (i / 17) % 9);
        uint16_t peak = (uint16_t)(1 + rnd() % 32767u);
        if (i % 3 == 0) dot = -dot;
        if (i < 4) dot = (i & 1) ? 8388607L : -8388607L;
        if (i < 4) { s = 0xffffffUL; peak = 32767; }
        if (i == 5) dot = 0;
        if (i == 6) s = 0;
        scale[0] = (uint8_t)s; scale[1] = (uint8_t)(s >> 8); scale[2] = (uint8_t)(s >> 16); scale[3] = 0;
        args.dot = dot; args.peak = peak;
        tiny_rescale(&r, scale, &args);
        expect("rescale", u64(&r), (uint64_t)((int64_t)dot * (int64_t)s * (int64_t)peak), (long)i);
    }
}

static int16_t vec[258];
static void test_sumsq(void) {
    static const uint16_t sizes[] = {1, 2, 3, 24, 96, 127, 128, 255, 256};
    Wide r;
    unsigned i, k, n;
    for (i = 0; i < sizeof sizes / sizeof sizes[0]; i++)
        for (k = 0; k < 4; k++) {
            uint64_t want = 0;
            n = sizes[i];
            for (unsigned j = 0; j < n; j++) {
                int16_t v = (int16_t)(rnd() >> (16 + (k == 2 ? 6 : 0)));
                if (k == 0) v = (j & 1) ? 32767 : -32767;
                if (k == 3 && (j % 5) == 0) v = 0;
                if (k == 3 && (j % 7) == 0) v = -32768;
                vec[j + 1] = v;
                want += (uint64_t)((int32_t)v * (int32_t)v);
            }
            vec[0] = vec[n + 1] = 12345;
            set(&r, 0x1122334455667788ULL);
            tiny_sumsq(&r, vec + 1, n);
            expect("sumsq", u64(&r), want, (long)(i * 10 + k));
            expect("sumsq guard", (uint64_t)(vec[0] == 12345 && vec[n + 1] == 12345), 1, (long)n);
        }
}

static void test_score(void) {
    TinyScoreArgs args;
    Wide r;
    unsigned i;
    for (i = 0; i < 900; i++) {
        int32_t dot = (int32_t)(rnd() >> (9 + i % 23));
        uint16_t m = (uint16_t)(rnd() >> (16 + (i / 23) % 3));
        uint16_t csm = (uint16_t)(rnd() >> (16 + (i / 69) % 3));
        uint32_t factor = 1 + rnd() % 1090717716UL;
        int64_t raw, t;
        uint64_t mag, want;
        int shift = 0, top = -1, got;
        if (i % 3 == 0) dot = -dot;
        if (i == 1) dot = 0;
        if (i == 2) m = 0;
        if (i == 4) { dot = 8388607L; m = 32767; csm = 65535; factor = 1090717716UL; }
        if (i == 5) { dot = -8388607L; m = 32767; csm = 65535; factor = 1090717716UL; }
        if (i % 37 == 6) { m = 1; csm = 1; }                 /* small: exercises shift 0 */
        raw = (int64_t)dot * m * csm;
        mag = (uint64_t)(raw < 0 ? -raw : raw);
        if (mag) { int b = 63; while (!(mag >> b)) b--; shift = b - 29; if (shift < 0) shift = 0; }
        if (shift) mag = (mag >> shift) + ((mag >> (shift - 1)) & 1);
        t = raw < 0 ? -(int64_t)mag : (int64_t)mag;
        want = (uint64_t)(t * (int64_t)factor);
        if (mag) { uint64_t o = mag * factor; top = 63; while (!(o >> top)) top--; }
        args.dot = dot; args.m = m; args.csm = csm; args.factor = factor; args.top = 0x77;
        set(&r, 0x0123456789abcdefULL);
        got = tiny_att_score(&r, &args);
        expect("score", u64(&r), want, (long)i);
        expect("score shift", (uint64_t)got, (uint64_t)shift, (long)i);
        expect("score top", (uint64_t)args.top, (uint64_t)(top < 0 ? 0xff : top), (long)i);
    }
}

/* One record as stored in auxiliary memory: four scale bytes, then weights. */
static int8_t record[4 + 256 + 2];
static uint8_t scale_out[6];
static int8_t act[256];
static void test_prepared(void) {
    static const uint16_t sizes[] = {1, 2, 24, 96, 97, 255, 256};
    unsigned i, k, j, n;
    for (i = 0; i < sizeof sizes / sizeof sizes[0]; i++)
        for (k = 0; k < 9; k++) {
            int32_t want = 0;
            n = sizes[i];
            for (j = 0; j < n; j++) {
                int8_t a = (int8_t)((int)(rnd() % 255u) - 127);
                int8_t w = (int8_t)(rnd() >> 24);
                if (k == 1) a = 0;                                  /* all zero */
                if (k == 2 && (rnd() % 100u) < 82) a = 0;           /* ReLU-like */
                if (k == 3) { a = (j & 1) ? 127 : -127; w = (j & 2) ? -128 : 127; }
                if (k == 4) { a = 127; w = 127; }                   /* largest positive sum */
                if (k == 5) { a = 127; w = -128; }                  /* most negative sum */
                if (k == 6 && j != 0) a = 0;                        /* only column 0 */
                if (k == 7 && j != n - 1) a = 0;                    /* only the last column */
                if (k == 8 && j == 0) a = 0;                        /* column 0 zero */
                act[j] = a; record[4 + j] = w;
                want += (int32_t)a * (int32_t)w;
            }
            for (j = 0; j < 4; j++) record[j] = (int8_t)(0x40 + j + k);
            record[4 + n] = 0x55;
            /* Plain prepared path. */
            memcpy(tiny_mac65_work, act, n);
            tiny_mac_a = record + 4; tiny_mac_b = tiny_mac65_work; tiny_mac_n = (uint16_t)n;
            tiny_mac65_prepare();
            tiny_mac65_prepared_dot();
            expect("prepared", (uint64_t)(int64_t)tiny_mac_result, (uint64_t)(int64_t)want, (long)(i * 10 + k));
            tiny_mac65_prepared_dot();                               /* state survives */
            expect("prepared again", (uint64_t)(int64_t)tiny_mac_result, (uint64_t)(int64_t)want, (long)(i * 10 + k));
            /* Direct path: scale bytes copied, weights read in place. */
            memset(scale_out, 0xee, sizeof scale_out);
            tiny_mac_a = record; tiny_mac_b = (const int8_t *)(scale_out + 1); tiny_mac_coeff = 3;
            tiny_mac65_aux_dot();
            expect("aux", (uint64_t)(int64_t)tiny_mac_result, (uint64_t)(int64_t)want, (long)(i * 10 + k));
            expect("aux scale", (uint64_t)(memcmp(scale_out + 1, record, 4) == 0 &&
                                           scale_out[0] == 0xee && scale_out[5] == 0xee), 1, (long)(i * 10 + k));
        }
}

/* Batched rows: eight records share one pass over the columns. */
static int8_t records[8 * 300 + 8];
static uint8_t rows_out[8 * 8 + 2];
static void test_rows(void) {
    static const uint16_t sizes[] = {1, 24, 96, 255, 256};
    static const uint16_t gaps[] = {0, 4, 40};      /* stride beyond 4+n */
    unsigned i, k, j, n, rows, g, r;
    for (i = 0; i < sizeof sizes / sizeof sizes[0]; i++)
        for (g = 0; g < 3; g++)
            for (rows = 1; rows <= 8; rows++)
                for (k = 0; k < 4; k++) {
                    unsigned stride;
                    int32_t want[8];
                    n = sizes[i];
                    stride = 4 + n + gaps[g];
                    if (8 * stride > sizeof records) continue;
                    for (j = 0; j < n; j++) {
                        int8_t a = (int8_t)((int)(rnd() % 255u) - 127);
                        if (k == 1 && (rnd() % 100u) < 82) a = 0;
                        if (k == 2) a = (j & 1) ? 127 : -127;
                        if (k == 3) a = 0;
                        act[j] = a;
                    }
                    for (r = 0; r < rows; r++) {
                        int8_t *rec = records + r * stride;
                        want[r] = 0;
                        for (j = 0; j < 4; j++) rec[j] = (int8_t)(rnd() >> 24);
                        for (j = 0; j < n; j++) {
                            int8_t w = (int8_t)(rnd() >> 24);
                            if (k == 2) w = (r & 1) ? -128 : 127;
                            rec[4 + j] = (int8_t)((uint8_t)w ^ 0x80u);
                            want[r] += (int32_t)act[j] * (int32_t)w;
                        }
                    }
                    memcpy(tiny_mac65_work, act, n);
                    tiny_mac_a = records + 4; tiny_mac_b = tiny_mac65_work; tiny_mac_n = (uint16_t)n;
                    tiny_mac65_prepare();
                    memset(rows_out, 0xee, sizeof rows_out);
                    tiny_mac_a = records; tiny_mac_b = (const int8_t *)(rows_out + 1);
                    tiny_mac_coeff = 2; tiny_mac_rows = (uint8_t)rows; tiny_mac_stride = (uint16_t)stride;
                    tiny_mac65_aux_rows();
                    for (r = 0; r < rows; r++) {
                        const uint8_t *o = rows_out + 1 + 8 * r;
                        int32_t got = (int32_t)((uint32_t)o[4] | (uint32_t)o[5] << 8 | (uint32_t)o[6] << 16 | (uint32_t)o[7] << 24);
                        expect("rows dot", (uint64_t)(int64_t)got, (uint64_t)(int64_t)want[r], (long)(i * 1000 + rows * 10 + k));
                        expect("rows scale", (uint64_t)(memcmp(o, records + r * stride, 4) == 0), 1, (long)r);
                    }
                    expect("rows guard", (uint64_t)(rows_out[0] == 0xee && rows_out[1 + 8 * rows] == 0xee), 1, (long)rows);
                }
}

static void test_axpy(void) {
    static int8_t values[130][24];
    static int32_t out[26];
    unsigned k, t, i, terms;
    for (k = 0; k < 6; k++) {
        int32_t want[24];
        terms = k == 0 ? 1 : k == 1 ? 128 : 1 + rnd() % 128u;
        memset(want, 0, sizeof want);
        out[0] = out[25] = 0x5a5a5a5aL;
        tiny_mac_acc = out + 1;
        tiny_mac65_axpy_begin();
        for (t = 0; t < terms; t++) {
            int8_t c = (int8_t)(1 + rnd() % 127u);
            if (k == 1) c = 127;
            for (i = 0; i < 24; i++) {
                int8_t v = (int8_t)(rnd() >> 24);
                if (k == 1) v = (i & 1) ? -128 : 127;
                values[t][i] = (int8_t)((uint8_t)v ^ 0x80u);
                want[i] += (int32_t)c * (int32_t)v;
            }
            tiny_mac_a = values[t]; tiny_mac_rows = 5; tiny_mac_coeff = c;
            tiny_mac65_aux_axpy();
        }
        tiny_mac65_axpy_end();
        for (i = 0; i < 24; i++)
            expect("axpy", (uint64_t)(int64_t)out[1 + i], (uint64_t)(int64_t)want[i], (long)(k * 100 + i));
        expect("axpy guard", (uint64_t)(out[0] == 0x5a5a5a5aL && out[25] == 0x5a5a5a5aL), 1, (long)k);
    }
}

/* Batch post-processing against the single-call routines. */
static void test_batches(void) {
    static uint8_t recs[8 * 8];
    static Wide out[9], want[8];
    static int16_t exps[9];
    static PackedScale ks[9];
    unsigned round, i, k;
    for (round = 0; round < 60; round++) {
        uint16_t rargs[2];
        TinyRescaleArgs ra;
        TinyScoreArgs sa;
        int16_t hi = -32768, base = (int16_t)((int)(rnd() % 200u) - 100);
        int first = round & 1;
        k = 1 + round % 8;
        for (i = 0; i < 64; i++) recs[i] = (uint8_t)(rnd() >> 24);
        for (i = 0; i < k; i++) {
            int32_t dot = (int32_t)(rnd() >> (9 + (round + i) % 20));
            if ((round + i) % 3 == 0) dot = -dot;
            if (round == 7) dot = 0;
            recs[8 * i + 3] = 0;                                    /* scale is 24 bits */
            recs[8 * i + 4] = (uint8_t)dot; recs[8 * i + 5] = (uint8_t)(dot >> 8);
            recs[8 * i + 6] = (uint8_t)(dot >> 16); recs[8 * i + 7] = (uint8_t)(dot >> 24);
        }
        /* rescale */
        ra.peak = (uint16_t)(1 + rnd() % 32767u);
        for (i = 0; i < k; i++) {
            ra.dot = (int32_t)((uint32_t)recs[8*i+4] | (uint32_t)recs[8*i+5] << 8 | (uint32_t)recs[8*i+6] << 16 | (uint32_t)recs[8*i+7] << 24);
            tiny_rescale(&want[i], recs + 8 * i, &ra);
        }
        set(&out[k], 0x7777777777777777ULL);
        rargs[0] = (uint16_t)k; rargs[1] = ra.peak;
        tiny_rescale_rows(out, recs, rargs);
        for (i = 0; i < k; i++) expect("rescale rows", u64(&out[i]), u64(&want[i]), (long)(round * 10 + i));
        expect("rescale rows guard", u64(&out[k]), 0x7777777777777777ULL, (long)round);
        /* scores */
        sa.m = (uint16_t)(rnd() >> 17); sa.factor = 1 + rnd() % 1090717716UL;
        for (i = 0; i < 9; i++) { ks[i].m = (uint16_t)(rnd() >> 17); ks[i].exp = (int16_t)((int)(rnd() % 60u) - 30); exps[i] = 0x1234; }
        tiny_att_args.m = sa.m; tiny_att_args.factor = sa.factor;
        tiny_att_out = out; tiny_att_exps = exps; tiny_att_ks = ks;
        tiny_att_base = base; tiny_att_hi = -32768; tiny_att_first = (uint8_t)first;
        tiny_mac_b = (const int8_t *)recs; tiny_mac_rows = (uint8_t)k;
        set(&out[k], 0x7777777777777777ULL);
        tiny_att_batch();
        for (i = 0; i < k; i++) {
            Wide w; int shift, e;
            if (first) {
                expect("batch scale", (uint64_t)(ks[i].m == (uint16_t)(recs[8*i] | recs[8*i+1] << 8) &&
                       ks[i].exp == (int16_t)(uint16_t)(recs[8*i+2] | recs[8*i+3] << 8)), 1, (long)(round * 10 + i));
            }
            sa.dot = (int32_t)((uint32_t)recs[8*i+4] | (uint32_t)recs[8*i+5] << 8 | (uint32_t)recs[8*i+6] << 16 | (uint32_t)recs[8*i+7] << 24);
            sa.csm = ks[i].m;
            shift = tiny_att_score(&w, &sa);
            e = base + ks[i].exp + shift;
            expect("batch score", u64(&out[i]), u64(&w), (long)(round * 10 + i));
            expect("batch exp", (uint64_t)(uint16_t)exps[i], (uint64_t)(uint16_t)e, (long)(round * 10 + i));
            if (sa.top != 0xff && sa.top + e > hi) hi = (int16_t)(sa.top + e);
        }
        expect("batch hi", (uint64_t)(uint16_t)tiny_att_hi, (uint64_t)(uint16_t)hi, (long)round);
        expect("batch guard", (uint64_t)(u64(&out[k]) == 0x7777777777777777ULL && exps[k] == 0x1234), 1, (long)round);
        expect("batch pointers", (uint64_t)(tiny_att_out == out + k && tiny_att_exps == exps + k && tiny_att_ks == ks + k), 1, (long)round);
    }
}

/* F1 discards negative lanes but must preserve the exponent originally
 * selected by pack. Check both signs, ties and the winning row across calls. */
static void test_f1(void) {
    static uint8_t records[16][8];
    static Wide results[16];
    static int64_t expected[16];
    unsigned trial, i, winner;
    uint16_t args[2] = {8, 0};
    for (trial = 0; trial < 36; ++trial) {
        int64_t minimum = 0;
        winner = 16;
        args[1] = trial == 0 ? 0 : trial == 1 ? 32767 : (uint16_t)(1 + rnd() % 32767);
        for (i = 0; i < 16; ++i) {
            uint32_t scale = rnd() % 8388609UL;
            int32_t dot = (int32_t)(rnd() % 8388608UL);
            if (trial % 3 == 1 || (trial % 3 == 2 && (i & 1))) dot = -dot;
            if (trial == 1 && (i == 8 || i == 10)) {
                dot = -8388608L;
                scale = 8388608UL;
            }
            memcpy(records[i], &scale, 4);
            memcpy(records[i]+4, &dot, 4);
            expected[i] = (int64_t)dot * scale * args[1];
            if (expected[i] < minimum) { minimum = expected[i]; winner = i; }
        }
        tiny_f1_begin();
        tiny_rescale_rows(results, records[0], args);
        tiny_rescale_rows(results+8, records[8], args);
        tiny_f1_finish();
        for (i = 0; i < 16; ++i) {
            uint64_t want = expected[i] >= 0 || i == winner ? (uint64_t)expected[i] : 0;
            expect("F1 ReLU rows", u64(&results[i]), want, (long)(16*trial+i));
        }
        tiny_rescale_rows(results, records[0], args);
        for (i = 0; i < 8; ++i)
            expect("F1 mode restored", u64(&results[i]), (uint64_t)expected[i], (long)(16*trial+i));
    }
}

static Wide wides[258];
static int16_t packed[258], va[98], vb[98];
static uint64_t mag64(uint64_t v) { return (v >> 63) ? 0 - v : v; }
static int64_t shift16(int16_t v, int s) {
    int64_t m;
    if (s >= 0) return (int64_t)v * ((int64_t)1 << s);
    m = v < 0 ? -(int64_t)v : v;
    m = (m >> -s) + ((m >> (-s - 1)) & 1);
    return v < 0 ? -m : m;
}
static void test_pack_residual(void) {
    static const uint16_t sizes[] = {1, 2, 96, 97, 255, 256};
    static const int8_t shifts[] = {16, 15, 9, 8, 7, 1, 0, -1, -7, -8, -9, -15, -16};
    unsigned i, k, j, n;
    for (i = 0; i < sizeof sizes / sizeof sizes[0]; i++)
        for (k = 0; k < 8; k++) {
            uint64_t best = 0;
            Wide r;
            int16_t args[2];
            int shift = (int)(k * 7) - 14, width = shift + 15;       /* results fit 16 bits */
            n = sizes[i];
            for (j = 0; j < n; j++) {
                uint64_t v = rnd64();
                if (width < 1) width = 1;
                v &= ((uint64_t)1 << width) - 1;
                if (shift > 0 && (v >> shift) >= 32767) v = (uint64_t)32766 << shift;
                if (shift <= 0 && v > (uint64_t)(32767 >> -shift)) v = (uint64_t)(32767 >> -shift);
                if (rnd() & 1) v = 0 - v;
                if (j % 11 == 3) v = 0;
                set(&wides[j], v);
                if (mag64(v) > best) best = mag64(v);
            }
            if (k == 7) { set(&wides[n - 1], 0x8000000000000000ULL); best = 0x8000000000000000ULL; }
            tiny_wmaxmag(&r, wides, n);
            expect("maxmag", u64(&r), best, (long)(i * 10 + k));
            if (k == 7) continue;
            packed[0] = packed[n + 1] = 0x5a5a;
            args[0] = (int16_t)n; args[1] = (int16_t)shift;
            tiny_pack_rows(packed + 1, wides, args);
            for (j = 0; j < n; j++)
                expect("pack rows", (uint64_t)(uint16_t)packed[1 + j], (uint64_t)(uint16_t)tiny_pack16(&wides[j], shift), (long)(i * 1000 + k * 300 + j));
            expect("pack guard", (uint64_t)(packed[0] == 0x5a5a && packed[n + 1] == 0x5a5a), 1, (long)n);
        }
    for (i = 0; i < sizeof shifts; i++)
        for (k = 0; k < sizeof shifts; k++) {
            TinyResidualArgs ra;
            n = 96;
            for (j = 0; j < n; j++) {
                va[j + 1] = (int16_t)(rnd() >> 16); vb[j + 1] = (int16_t)(rnd() >> 16);
                if (j == 0) { va[1] = 32767; vb[1] = 32767; }
                if (j == 1) { va[2] = -32767; vb[2] = -32767; }
                if (j == 2) { va[3] = 0; vb[3] = -1; }
                if (j == 3) { va[4] = 1; vb[4] = 16384; }
            }
            ra.b = vb + 1; ra.n = (uint16_t)n; ra.sa = shifts[i]; ra.sb = shifts[k];
            set(&wides[n], 0x7777777777777777ULL);
            tiny_residual_rows(wides, va + 1, &ra);
            for (j = 0; j < n; j++)
                expect("residual", u64(&wides[j]), (uint64_t)(shift16(va[j + 1], shifts[i]) + shift16(vb[j + 1], shifts[k])), (long)(i * 1000 + k * 100 + j));
            expect("residual guard", u64(&wides[n]), 0x7777777777777777ULL, (long)i);
        }
}

int main(void) {
    static uint8_t rescale_code[1026];
    rescale_code[0] = rescale_code[1025] = 0xa5;
    tiny_rescale_init(rescale_code + 1);
    expect("rescale code guard", rescale_code[0] == 0xa5 && rescale_code[1025] == 0xa5, 1, 0);
    tiny_mac65_init();
    test_wm();        printf("wm: %lu checks\n", checks);
    test_rescale();   printf("rescale: %lu checks\n", checks);
    test_sumsq();     printf("sumsq: %lu checks\n", checks);
    test_score();     printf("score: %lu checks\n", checks);
    test_prepared();  printf("prepared: %lu checks\n", checks);
    test_rows();      printf("rows: %lu checks\n", checks);
    test_axpy();      printf("axpy: %lu checks\n", checks);
    test_batches();   printf("batches: %lu checks\n", checks);
    test_f1();        printf("F1: %lu checks\n", checks);
    test_pack_residual(); printf("pack/residual: %lu checks\n", checks);
    if (fails) { printf("FAILED: %lu of %lu\n", fails, checks); return 1; }
    printf("PASS: %lu checks\n", checks);
    return 0;
}
