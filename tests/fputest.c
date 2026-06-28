/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * fputest.c - Standalone correctness harness for the SFP-004 68882 dispatch.
 *
 * Exercises linuxdoom-1.10/atari_sfp004.c: for each test case it computes the
 * result two ways and compares the raw IEEE-754 single bit patterns —
 *
 *   fpu = sfp004_<op>(a, b)   dispatched to the 68882 when one is detected,
 *                               otherwise the C-operator soft-float fallback;
 *   ref = a <op> b              always the C-operator soft-float result (this
 *                               app is built for the 68000, so it has no FPU).
 *
 * With no FPU present both paths are the same soft-float code, so every case
 * PASSes — that confirms detection, wiring and the fallback are sound, and the
 * harness itself works. On a machine with the memory-mapped 68882 (SFP004-style,
 * and SFP004_CIR_BASE/offsets matching the GAL decode in atari_sfp004.c),
 * `fpu` comes from the coprocessor and the comparison becomes a real
 * hardware-vs-soft-float check.
 *
 * Build (from repo root):
 *   make -C tests fputest      -> tests/FPUTEST.TOS
 */
#include <mint/osbind.h>
#include <stdio.h>

#include "atari_sfp004.h"

enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV };

typedef struct {
    const char *name;
    int         op;
    float       a;
    float       b;
} fpu_case;

/* Spread of cases: whole numbers, fractions that round, negatives, a large
 * magnitude, and the classic 0.1+0.2. */
static const fpu_case s_cases[] = {
    { "ADD 3.5+2.0",    OP_ADD,    3.5f,    2.0f },
    { "SUB 10-7.25",    OP_SUB,   10.0f,    7.25f },
    { "MUL 2.0*0.5",    OP_MUL,    2.0f,    0.5f },
    { "DIV 1/3",        OP_DIV,    1.0f,    3.0f },
    { "ADD -4.5+1.25",  OP_ADD,   -4.5f,    1.25f },
    { "MUL 1234.5*6.78",OP_MUL, 1234.5f,    6.78f },
    { "ADD 0.1+0.2",    OP_ADD,    0.1f,    0.2f },
    { "DIV 355/113",    OP_DIV,  355.0f,  113.0f },
};
#define NUM_CASES ((int)(sizeof(s_cases) / sizeof(s_cases[0])))

/* float <-> raw 32-bit pattern without aliasing UB. */
typedef union { float f; unsigned long u; } fbits;

/* SFP004 memory-mapped CIR registers (must match atari_sfp004.c). Used only by
 * the diagnostic trace below. */
#define PRB_RESP ((volatile short *)0x00fffa40UL) /* Response, 16-bit */
#define PRB_REST ((volatile short *)0x00fffa46UL) /* Restore,  16-bit */
#define PRB_CMD  ((volatile short *)0x00fffa4aUL) /* Command,  16-bit */
#define PRB_OPND ((volatile long  *)0x00fffa50UL) /* Operand,  32-bit */

/*
 * Trace one full working op (1.0 + 2.0 -> 40400000), polling "while RESPONSE ==
 * $8900" exactly as the dispatch does, and report for each command: the RESPONSE
 * read immediately after the command, how many $8900 (busy) reads the poll saw
 * before the FPU presented a transfer primitive, and that settled RESPONSE value
 * — plus the result. This confirms the completion condition and shows how many
 * poll iterations a real op actually needs (vs the old guard-spin).
 */
static void fpu_probe(void)
{
    short ra0, ra1, ra2, rs0, rs1, rs2;
    int   n0, n1, n2;
    unsigned long result;

    *PRB_REST = 0x0000;                                 /* FRESTORE null */

    *PRB_CMD = 0x4400; ra0 = *PRB_RESP;                 /* FMOVE.S in */
    n0 = 0; while ((unsigned short)*PRB_RESP == 0x8900 && ++n0 < 4000) ;
    rs0 = *PRB_RESP; *PRB_OPND = 0x3F800000UL;          /* 1.0 */

    *PRB_CMD = 0x4422; ra1 = *PRB_RESP;                 /* FADD.S */
    n1 = 0; while ((unsigned short)*PRB_RESP == 0x8900 && ++n1 < 4000) ;
    rs1 = *PRB_RESP; *PRB_OPND = 0x40000000UL;          /* 2.0 */

    *PRB_CMD = 0x6400; ra2 = *PRB_RESP;                 /* FMOVE.S out */
    n2 = 0; while ((unsigned short)*PRB_RESP == 0x8900 && ++n2 < 4000) ;
    rs2 = *PRB_RESP; result = (unsigned long)*PRB_OPND;

    printf("CIR trace (1.0+2.0 -> want 40400000):\n");
    printf(" in : resp=%04X poll=%d ->%04X\n", (unsigned short)ra0, n0, (unsigned short)rs0);
    printf(" add: resp=%04X poll=%d ->%04X\n", (unsigned short)ra1, n1, (unsigned short)rs1);
    printf(" out: resp=%04X poll=%d ->%04X\n", (unsigned short)ra2, n2, (unsigned short)rs2);
    printf(" result=%08lX\n", result);
}

/* Dispatched result (FPU when available, else soft-float). */
static float run_dispatch(int op, float a, float b)
{
    switch (op) {
        case OP_ADD: return sfp004_add(a, b);
        case OP_SUB: return sfp004_sub(a, b);
        case OP_MUL: return sfp004_mul(a, b);
        default:     return sfp004_div(a, b);
    }
}

/* Pure soft-float reference. noinline keeps the compiler from constant-folding
 * the call sites away, so this is a genuine runtime __addsf3/etc. result. */
static float __attribute__((noinline)) run_ref(int op, float a, float b)
{
    switch (op) {
        case OP_ADD: return a + b;
        case OP_SUB: return a - b;
        case OP_MUL: return a * b;
        default:     return a / b;
    }
}

/* Run all cases once and return the pass count; print each line when verbose. */
static int run_table(int verbose)
{
    int i, passed = 0;

    for (i = 0; i < NUM_CASES; i++) {
        fbits got, ref;
        int   ok;

        got.f = run_dispatch(s_cases[i].op, s_cases[i].a, s_cases[i].b);
        ref.f = run_ref(s_cases[i].op, s_cases[i].a, s_cases[i].b);
        ok = (got.u == ref.u);
        if (ok)
            passed++;
        if (verbose)
            printf("%-16s fpu=%08lX ref=%08lX %s\n",
                   s_cases[i].name, got.u, ref.u, ok ? "PASS" : "FAIL");
    }
    return passed;
}

int main(void)
{
    int  have_fpu, passed;
    long ssp = 0;
    int  ssp_valid = 0;

    printf("SFP-004 68882 FPU dispatch test\n");

    sfp004_init();
    have_fpu = sfp004_available();
    if (have_fpu) {
        printf("68882 detected: YES (testing CIR dispatch)\n");
        /* CIR access is supervisor-only, mirror STDOOM's runtime mode. */
        ssp = Super(0L);
        ssp_valid = 1;
    } else {
        printf("68882 detected: no (soft-float fallback)\n");
    }
    printf("comparing IEEE-754 single bit patterns\n\n");

    passed = run_table(1);
    printf("\n%d/%d passed\n", passed, NUM_CASES);

    /* Trace one op's response sequence (always, to confirm the poll condition
     * and show how few iterations a real op needs). */
    if (have_fpu) {
        printf("\n");
        fpu_probe();
    }

    if (ssp_valid)
        Super((void *)ssp);

    if (!have_fpu)
        printf("(no FPU: PASS only confirms the fallback)\n");

    printf("\nPress any key to quit.\n");
    Cconin();
    return 0;
}
