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
 * Build (from repo root):
 *   make -C tests fpubench     -> tests/FPUBENCH.TOS
 */
#include <math.h>
#include <mint/osbind.h>
#include <stdio.h>

#include "atari_sfp004.h"

/* Arithmetic mix: x4 ops per iteration. */
#define SOFT_ITERS 5000
#define FPU_ITERS  500
/* Square root: 1 op per iteration. */
#define SQRT_SOFT_ITERS 1000
#define SQRT_FPU_ITERS  2000
/* FixedDiv: 1 op per iteration. */
#define FDIV_INT_ITERS 5000
#define FDIV_FPU_ITERS 1000

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

/* us per op = ticks * 5 ms * 1000 / ops. */
static unsigned long per_op_us(unsigned long t, unsigned long ops)
{
    return (ops && t) ? (t * 5000UL / ops) : 0;
}

/* Print one A-vs-B comparison; the verdict is stated for side B relative to A. */
static void report(const char *name, const char *la, const char *lb,
                   unsigned long ta, unsigned long opa, fbits ra,
                   unsigned long tb, unsigned long opb, fbits rb)
{
    unsigned long ua = per_op_us(ta, opa);
    unsigned long ub = per_op_us(tb, opb);

    printf("\n[%s]\n", name);
    printf(" %s: %lu ops %lu ms %lu us/op  =%08lX\n", la, opa, ta * 5UL, ua, ra.u);
    printf(" %s: %lu ops %lu ms %lu us/op  =%08lX\n", lb, opb, tb * 5UL, ub, rb.u);
    if (!ua || !ub)
        printf(" (too fast to time — raise iters)\n");
    else if (ub >= ua)
        printf(" => %s %lu.%02lu x SLOWER than %s\n", lb,
               (ub * 100UL / ua) / 100, (ub * 100UL / ua) % 100, la);
    else
        printf(" => %s %lu.%02lu x FASTER than %s\n", lb,
               (ua * 100UL / ub) / 100, (ua * 100UL / ub) % 100, la);
}

int main(void)
{
    long ssp;
    unsigned long t0, t1;
    unsigned long at_s, at_f = 0, qt_s, qt_f = 0, dt_i, dt_f = 0;
    fbits ar_s, ar_f, qr_s, qr_f, dr_i, dr_f;
    int have;

    ar_f.u = qr_f.u = dr_f.u = 0;

    printf("SFP-004 68882 benchmark\n");
    sfp004_init();
    have = sfp004_available();
    printf("68882 detected: %s\n", have ? "YES" : "no");

    ssp = Super(0L);  /* CIR + tick reads are supervisor-only */

    printf("\narithmetic: soft %lu / fpu %lu ops...\n",
           (unsigned long)SOFT_ITERS * 4, (unsigned long)FPU_ITERS * 4);
    t0 = ticks(); ar_s.f = bench_soft(SOFT_ITERS); t1 = ticks(); at_s = t1 - t0;
    if (have) { t0 = ticks(); ar_f.f = bench_fpu(FPU_ITERS); t1 = ticks(); at_f = t1 - t0; }

    printf("sqrt: soft %d / fpu %d ops...\n", SQRT_SOFT_ITERS, SQRT_FPU_ITERS);
    t0 = ticks(); qr_s.f = bench_sqrt_soft(SQRT_SOFT_ITERS); t1 = ticks(); qt_s = t1 - t0;
    if (have) { t0 = ticks(); qr_f.f = bench_sqrt_fpu(SQRT_FPU_ITERS); t1 = ticks(); qt_f = t1 - t0; }

    printf("FixedDiv: int %d / fpu %d ops...\n", FDIV_INT_ITERS, FDIV_FPU_ITERS);
    t0 = ticks(); dr_i.u = (unsigned long)bench_fdiv_int(FDIV_INT_ITERS); t1 = ticks(); dt_i = t1 - t0;
    if (have) { t0 = ticks(); dr_f.u = (unsigned long)bench_fdiv_fpu(FDIV_FPU_ITERS); t1 = ticks(); dt_f = t1 - t0; }

    v_sink = ar_s.f + ar_f.f + qr_s.f + qr_f.f;

    Super((void *)ssp);

    if (!have) {
        printf("\nno FPU: soft/integer only\n");
        printf(" arith %lu us/op  sqrt %lu us/op  FixedDiv %lu us/op\n",
               per_op_us(at_s, (unsigned long)SOFT_ITERS * 4),
               per_op_us(qt_s, SQRT_SOFT_ITERS),
               per_op_us(dt_i, FDIV_INT_ITERS));
    } else {
        report("arithmetic mul/add/sub/div", "soft ", "68882",
               at_s, (unsigned long)SOFT_ITERS * 4, ar_s,
               at_f, (unsigned long)FPU_ITERS * 4, ar_f);
        report("sqrt", "soft ", "68882",
               qt_s, SQRT_SOFT_ITERS, qr_s,
               qt_f, SQRT_FPU_ITERS, qr_f);
        report("FixedDiv 16.16 (160.0 / 7.0)", "int  ", "68882",
               dt_i, FDIV_INT_ITERS, dr_i,
               dt_f, FDIV_FPU_ITERS, dr_f);
        if (dr_i.u != dr_f.u)
            printf(" note: results differ by %ld (rounding) int=%08lX fpu=%08lX\n",
                   (long)dr_f.u - (long)dr_i.u, dr_i.u, dr_f.u);
    }

    printf("\nPress any key to quit.\n");
    Cconin();
    return 0;
}
