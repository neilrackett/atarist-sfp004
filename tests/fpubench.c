/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * fpubench.c - Soft-float / integer vs SFP-004 68882 micro-benchmark.
 *
 * Times three workloads, each two ways, with the 200 Hz system clock and reports
 * per-op cost. (With a correct response poll the memory-mapped 68882 turns out
 * to BEAT soft-float even on scalar ops, and crush it on transcendentals.)
 *
 *   A) arithmetic mix (mul/add/sub/div), single precision: 68882 vs libgcc
 *      soft-float.
 *   B) square root: 68882 FSQRT vs libm soft-float sqrt.
 *   C) FixedDiv (16.16): the renderer's hot divide. The soft path is Doom's
 *      integer long-division (m_fixed.c FixedDiv2, ~32 shift/subtract steps);
 *      the 68882 path feeds 32-bit *integer* operands to the FPU (FMOVE.L /
 *      FDIV.L / FMUL.L 65536 / FMOVE.L — the FPU works in 80-bit extended
 *      internally, so no doubles are needed) and reads a 32-bit result back.
 *      This is the one that could actually accelerate STDOOM.
 *
 * On a Mega STE it runs everything three times, at 8 MHz, 16 MHz and 16 MHz
 * with cache, then puts the CPU back as it found it; elsewhere it runs once.
 * The results end in one table, us/op per CPU speed, sized for a 40 column
 * screen.
 *
 * Build (from repo root):
 *   make -C tests fpubench     -> tests/FPUBENCH.TOS
 */
#include <math.h>
#include <mint/osbind.h>
#include <stdio.h>

#include "atari_sfp004.h"
#include "atari_megaste.h"

/* Iterations, chosen so that each timing spans at least ~40 ticks of the
 * 200 Hz clock even at 16 MHz with cache (a tick either way is then <3%). */
/* Arithmetic mix: x4 ops per iteration. */
#define SOFT_ITERS 5000
#define FPU_ITERS  2000
/* Square root: 1 op per iteration. */
#define SQRT_SOFT_ITERS 1000
#define SQRT_FPU_ITERS  8000
/* FixedDiv: 1 op per iteration. */
#define FDIV_INT_ITERS 5000
#define FDIV_FPU_ITERS 5000

typedef long fixed_t;
#define FRACUNIT 65536L

/* float <-> raw 32-bit pattern without aliasing UB (also used to carry a
 * fixed_t result into report()). */
typedef union { float f; unsigned long u; } fbits;

/* Volatile so the compiler cannot constant-fold the loops away. acc stays ~1.0
 * across an arithmetic iteration (mul/add/sub/div cancel). */
static volatile float v_mul = 1.01f;
static volatile float v_add = 0.5f;
static volatile float v_sub = 0.5f;
static volatile float v_div = 1.01f;
static volatile float v_sqrtin = 2.0f;
static volatile float v_sink;
static volatile fixed_t v_a = 160L * FRACUNIT;  /* FixedDiv operands */
static volatile fixed_t v_b = 7L * FRACUNIT;

/* ── SFP-004 CIR registers, for the raw FixedDiv path (mirror atari_sfp004.c) ── */
#define FB_RESP ((volatile short *)0x00fffa40UL) /* Response, 16-bit */
#define FB_REST ((volatile short *)0x00fffa46UL) /* Restore,  16-bit */
#define FB_CMD  ((volatile short *)0x00fffa4aUL) /* Command,  16-bit */
#define FB_OPL  ((volatile long  *)0x00fffa50UL) /* Operand,  32-bit */

static void fb_wait(void)
{
    long g = 0x4000L;
    while ((unsigned short)*FB_RESP == 0x8900 && --g)
        ;
}

/* 200 Hz system tick counter ($04BA); one tick = 5 ms. Supervisor read. */
static unsigned long ticks(void)
{
    return *(volatile unsigned long *)0x04baUL;
}

static float bench_soft(int iters)
{
    float acc = 1.0f;
    int i;
    for (i = 0; i < iters; i++) {
        acc = acc * v_mul;
        acc = acc + v_add;
        acc = acc - v_sub;
        acc = acc / v_div;
    }
    return acc;
}

static float bench_fpu(int iters)
{
    float acc = 1.0f;
    int i;
    for (i = 0; i < iters; i++) {
        acc = sfp004_mul(acc, v_mul);
        acc = sfp004_add(acc, v_add);
        acc = sfp004_sub(acc, v_sub);
        acc = sfp004_div(acc, v_div);
    }
    return acc;
}

static float bench_sqrt_soft(int iters)
{
    float r = 0.0f;
    int i;
    for (i = 0; i < iters; i++)
        r = (float)sqrt((double)v_sqrtin);
    return r;
}

static float bench_sqrt_fpu(int iters)
{
    float r = 0.0f;
    int i;
    for (i = 0; i < iters; i++)
        r = sfp004_sqrt(v_sqrtin);
    return r;
}

/* Doom's integer FixedDiv2 (m_fixed.c soft path). Assumes a, b > 0 and the
 * result fits (the FixedDiv wrapper's guard). */
static fixed_t fixdiv_int(fixed_t a, fixed_t b)
{
    unsigned short ibit = 1;
    short          ch = 0;
    unsigned short cl = 0;

    while (b < a) { b <<= 1; ibit <<= 1; }
    for (; ibit != 0; ibit >>= 1) {
        if (a >= b) { a -= b; ch |= ibit; }
        a <<= 1;
    }
    if (a>=b){a-=b;cl|=0x8000;} a<<=1;
    if (a>=b){a-=b;cl|=0x4000;} a<<=1;
    if (a>=b){a-=b;cl|=0x2000;} a<<=1;
    if (a>=b){a-=b;cl|=0x1000;} a<<=1;
    if (a>=b){a-=b;cl|=0x0800;} a<<=1;
    if (a>=b){a-=b;cl|=0x0400;} a<<=1;
    if (a>=b){a-=b;cl|=0x0200;} a<<=1;
    if (a>=b){a-=b;cl|=0x0100;} a<<=1;
    if (a>=b){a-=b;cl|=0x0080;} a<<=1;
    if (a>=b){a-=b;cl|=0x0040;} a<<=1;
    if (a>=b){a-=b;cl|=0x0020;} a<<=1;
    if (a>=b){a-=b;cl|=0x0010;} a<<=1;
    if (a>=b){a-=b;cl|=0x0008;} a<<=1;
    if (a>=b){a-=b;cl|=0x0004;} a<<=1;
    if (a>=b){a-=b;cl|=0x0002;} a<<=1;
    if (a>=b){a-=b;cl|=0x0001;}
    return (((fixed_t)ch) << 16) | cl;
}

/* 16.16 a/b on the 68882 with 32-bit integer operands: FP0 = a; FP0 /= b;
 * FP0 *= 65536; result = (long)FP0. */
static fixed_t fixdiv_fpu(fixed_t a, fixed_t b)
{
    *FB_REST = 0x0000;                          /* FRESTORE null     */
    *FB_CMD = 0x4000; fb_wait(); *FB_OPL = a;   /* FMOVE.L a, FP0    */
    *FB_CMD = 0x4020; fb_wait(); *FB_OPL = b;   /* FDIV.L  b, FP0    */
    *FB_CMD = 0x4023; fb_wait(); *FB_OPL = FRACUNIT; /* FMUL.L 65536 */
    *FB_CMD = 0x0003; fb_wait();                /* FINTRZ (trunc->0) */
    *FB_CMD = 0x6000; fb_wait();                /* FMOVE.L FP0, <ea> */
    return (fixed_t)*FB_OPL;
}

static fixed_t bench_fdiv_int(int iters)
{
    fixed_t r = 0;
    int i;
    for (i = 0; i < iters; i++)
        r = fixdiv_int(v_a, v_b);
    return r;
}

static fixed_t bench_fdiv_fpu(int iters)
{
    fixed_t r = 0;
    int i;
    for (i = 0; i < iters; i++)
        r = fixdiv_fpu(v_a, v_b);
    return r;
}

/* The six timings of a run */
enum { ARITH_SOFT, ARITH_FPU, SQRT_SOFT, SQRT_FPU, FDIV_INT, FDIV_FPU, NUM_TIMES };

static const unsigned long s_ops[NUM_TIMES] = {
    SOFT_ITERS * 4UL, FPU_ITERS * 4UL,
    SQRT_SOFT_ITERS, SQRT_FPU_ITERS,
    FDIV_INT_ITERS, FDIV_FPU_ITERS
};

typedef struct {
    unsigned long t[NUM_TIMES];   /* 200 Hz ticks */
    fbits         r[NUM_TIMES];   /* results, to check the FPU agrees */
} bench_run;

/* Mega STE CPU speeds to run at */
static const struct {
    unsigned char mode;
    const char   *name;
} s_modes[] = {
    { 0,                             "8MHz" },
    { MEGASTE_16MHZ,                 "16MHz" },
    { MEGASTE_16MHZ | MEGASTE_CACHE, "16+cache" },
};
#define MAX_MODES ((int)(sizeof(s_modes) / sizeof(s_modes[0])))

/* Time every workload at the current CPU speed (supervisor mode). */
static void run_bench(bench_run *b, int have)
{
    unsigned long t0;
    int i;

    for (i = 0; i < NUM_TIMES; i++) {
        b->t[i] = 0;
        b->r[i].u = 0;
    }

    t0 = ticks(); b->r[ARITH_SOFT].f = bench_soft(SOFT_ITERS); b->t[ARITH_SOFT] = ticks() - t0;
    t0 = ticks(); b->r[SQRT_SOFT].f = bench_sqrt_soft(SQRT_SOFT_ITERS); b->t[SQRT_SOFT] = ticks() - t0;
    t0 = ticks(); b->r[FDIV_INT].u = (unsigned long)bench_fdiv_int(FDIV_INT_ITERS); b->t[FDIV_INT] = ticks() - t0;
    if (have) {
        t0 = ticks(); b->r[ARITH_FPU].f = bench_fpu(FPU_ITERS); b->t[ARITH_FPU] = ticks() - t0;
        t0 = ticks(); b->r[SQRT_FPU].f = bench_sqrt_fpu(SQRT_FPU_ITERS); b->t[SQRT_FPU] = ticks() - t0;
        t0 = ticks(); b->r[FDIV_FPU].u = (unsigned long)bench_fdiv_fpu(FDIV_FPU_ITERS); b->t[FDIV_FPU] = ticks() - t0;
    }
    v_sink = b->r[ARITH_SOFT].f + b->r[ARITH_FPU].f + b->r[SQRT_SOFT].f + b->r[SQRT_FPU].f;
}

/* Tenths of a microsecond per op: ticks * 5 ms * 10000 / ops. */
static unsigned long per_op(const bench_run *b, int i)
{
    return b->t[i] ? b->t[i] * 50000UL / s_ops[i] : 0;
}

/* One table row: us/op of timing i for each run */
static void print_time(const char *label, const bench_run *runs, int n, int i)
{
    int m;

    printf("%-12s", label);
    for (m = 0; m < n; m++) {
        unsigned long u = per_op(&runs[m], i);
        printf("%7lu.%lu", u / 10, u % 10);
    }
    printf("\n");
}

/* One table row: how many times faster timing b is than timing a */
static void print_speedup(const bench_run *runs, int n, int a, int b)
{
    int m;

    printf("%-12s", "  x faster");
    for (m = 0; m < n; m++) {
        unsigned long ua = per_op(&runs[m], a), ub = per_op(&runs[m], b);
        unsigned long x = ub ? ua * 100UL / ub : 0;
        printf("%6lu.%02lu", x / 100, x % 100);
    }
    printf("\n");
}

int main(void)
{
    bench_run runs[MAX_MODES];
    unsigned char orig = 0;
    int have, megaste, n, m;
    long ssp;

    printf("SFP-004 68882 benchmark\n");
    sfp004_init();
    have = sfp004_available();
    megaste = is_megaste();
    n = megaste ? MAX_MODES : 1;
    printf("68882 detected: %s\n\n", have ? "YES" : "no");

    ssp = Super(0L);  /* CIR, tick and Mega STE speed reads are supervisor-only */
    if (megaste)
        orig = megaste_cpu_mode();

    for (m = 0; m < n; m++) {
        bench_run *b = &runs[m];

        if (megaste)
            megaste_set_cpu_mode(s_modes[m].mode);
        printf("%s... ", megaste ? s_modes[m].name : "running");
        run_bench(b, have);
        if (have && (b->r[ARITH_SOFT].u != b->r[ARITH_FPU].u
                     || b->r[SQRT_SOFT].u != b->r[SQRT_FPU].u
                     || b->r[FDIV_INT].u != b->r[FDIV_FPU].u))
            printf("results differ:\n %08lX/%08lX %08lX/%08lX\n %08lX/%08lX\n",
                   b->r[ARITH_SOFT].u, b->r[ARITH_FPU].u,
                   b->r[SQRT_SOFT].u, b->r[SQRT_FPU].u,
                   b->r[FDIV_INT].u, b->r[FDIV_FPU].u);
        else
            printf("ok\n");
    }

    if (megaste)
        megaste_set_cpu_mode(orig);
    Super((void *)ssp);

    printf("\n%-12s", "us/op");
    for (m = 0; m < n; m++)
        printf("%9s", megaste ? s_modes[m].name : "now");
    printf("\n");
    print_time("arith soft", runs, n, ARITH_SOFT);
    if (have) {
        print_time("      68882", runs, n, ARITH_FPU);
        print_speedup(runs, n, ARITH_SOFT, ARITH_FPU);
    }
    print_time("sqrt  soft", runs, n, SQRT_SOFT);
    if (have) {
        print_time("      68882", runs, n, SQRT_FPU);
        print_speedup(runs, n, SQRT_SOFT, SQRT_FPU);
    }
    print_time("FixDiv int", runs, n, FDIV_INT);
    if (have) {
        print_time("      68882", runs, n, FDIV_FPU);
        print_speedup(runs, n, FDIV_INT, FDIV_FPU);
    }

    printf("\nPress any key to quit.\n");
    Cconin();
    return 0;
}
