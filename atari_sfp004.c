/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/**
 * File: atari_sfp004.c
 * Description: Optional 68881/68882 FPU dispatch via the SFP-004 memory-mapped
 *              coprocessor interface.
 *
 * Built -msoft-float (never -m68881), so a stock 68000 ST/STE runs every float
 * through libgcc's soft-float routines (__addsf3 etc.). This module adds a
 * faster path for machines fitted with an SFP-004-style memory-mapped MC68881/
 * 68882 — a card, or a 68882 added to the motherboard with a GAL address decoder
 * (the case this was developed on: a Mega STE). It detects the coprocessor via
 * the _FPU cookie at startup and, when present, dispatches single-precision
 * arithmetic to the coprocessor's memory-mapped coprocessor interface registers
 * (CIR) with inline asm. When absent, the same entry points fall straight
 * through to the C operator, i.e. soft-float, so the binary still runs on a
 * plain ST/STE.
 *
 * A 68000 has no coprocessor bus, so the FPU cannot be driven with FPU
 * instructions (FADD etc.) — those F-line opcodes would trap. Instead the host
 * pokes the command/operand registers and reads the result back, as the SFP-004
 * memory-mapped FPU does on the ST bus. This is what "dispatch via inline asm"
 * means here: there are no -m68881 instructions anywhere.
 */

#include <math.h>

#include <mint/osbind.h>
#include <mint/cookie.h>

#include "atari_sfp004.h"

#ifndef C__FPU
#define C__FPU 0x5f465055L /* '_FPU' */
#endif

/*
 * _FPU cookie bit that flags the memory-mapped (SFP004-style) FPU.
 *
 * This is _FPU bit 16 (0x00010000), the "SFP-004" flag — the same bit the
 * startup detector in atari_checkcpu.c prints as "FPU: SFP-004". A 68882 added
 * to the motherboard with a GAL decoder reports through this bit, because it is
 * driven through the SFP004 memory-mapped register interface rather than a
 * coprocessor bus. (atari_checkcpu.c also decodes bits 17-18 as an external
 * 68881/68882 and bit 19 as an integrated FPU; those are the coprocessor-bus
 * cases this module does not handle.)
 */
#define SFP004_COOKIE 0x00010000L

/*
 * Base address of the 68882 coprocessor interface registers (CIR).
 *
 * BOARD-SPECIFIC: this must match where the GAL decodes the 68882 CIR. The
 * default below is the standard SFP004 base ($FFFA40, just past the MFP). The
 * register offsets that follow assume the GAL presents the registers as the
 * standard contiguous MC68881/MC68882 CIR map at that base (response at +$00,
 * command at +$0A, operand at +$10). If the GAL decode differs, adjust the base
 * and/or the offsets to match it before relying on the dispatch path.
 */
#define SFP004_CIR_BASE 0x00fffa40UL

#define SFP004_RESPONSE ((volatile short *)(SFP004_CIR_BASE + 0x00)) /* w */
#define SFP004_CONTROL  ((volatile short *)(SFP004_CIR_BASE + 0x02)) /* w */
#define SFP004_SAVE     ((volatile short *)(SFP004_CIR_BASE + 0x04)) /* w */
#define SFP004_RESTORE  ((volatile short *)(SFP004_CIR_BASE + 0x06)) /* w */
#define SFP004_COMMAND  ((volatile short *)(SFP004_CIR_BASE + 0x0a)) /* w */
#define SFP004_OPERAND  ((volatile long  *)(SFP004_CIR_BASE + 0x10)) /* l */

/*
 * RESPONSE value the coprocessor returns while it is still working: the Null/
 * come-again primitive $8900. The host spins while RESPONSE == $8900 and
 * proceeds (transfer operand / read result) as soon as it changes to a
 * transfer-data primitive — this is the exact-match condition the CHZ-Soft
 * SFP004 doc's working example uses (cmpiw #$8900 / beq wait), and unlike a
 * bit-8 (CA) test it works regardless of which bits the transfer primitive sets
 * on a given board. Reading RESPONSE is also what *starts* the coprocessor
 * processing the command just written.
 */
#define SFP004_RESP_BUSY 0x8900

/*
 * MC68882 cpGEN command words. Format (verified against the CHZ-Soft doc's
 * known-good FACOS word $541C): bit 15 = 0, bit 14 = R/M, bits 13-11 = source
 * specifier, bits 10-8 = destination FP register, bits 6-0 = opcode. So a
 * memory-source single-precision op into FP0 is R/M=1 (bit 14), src spec = 001
 * (single), dest = 000 (FP0): base $4400 | opcode.
 *
 * NOTE: R/M is bit 14, NOT bit 15 — a $88xx encoding (bit 15 set) is malformed
 * and the coprocessor returns garbage without executing.
 */
#define SFP004_CMD_FMOVE_IN 0x4400 /* FMOVE.S <ea>,FP0  (load FP0)        */
#define SFP004_CMD_FADD     0x4422 /* FADD.S  <ea>,FP0                    */
#define SFP004_CMD_FSUB     0x4428 /* FSUB.S  <ea>,FP0                    */
#define SFP004_CMD_FMUL     0x4423 /* FMUL.S  <ea>,FP0                    */
#define SFP004_CMD_FDIV     0x4420 /* FDIV.S  <ea>,FP0                    */
#define SFP004_CMD_FSQRT    0x4404 /* FSQRT.S <ea>,FP0  (monadic)         */
/* FMOVE.S FP0,<ea> (result out): bits 15-13 = 011, format = single (001),
 * source reg = FP0 (000), k-factor = 0. (cf. the doc's $7400 = FMOVE.D out.) */
#define SFP004_CMD_FMOVE_OUT 0x6400

/* Long-integer-format command words for sfp004_fixdiv (source spec = 000 = Long
 * Word Integer): the FPU converts the 32-bit integer operands to its internal
 * 80-bit extended, so a 16.16 divide needs no doubles. FINTRZ is a register-only
 * op (R/M=0, FP0->FP0, opcode $03) that rounds toward zero — see sfp004_fixdiv. */
#define SFP004_CMD_FMOVEL_IN  0x4000 /* FMOVE.L <ea>,FP0  (load integer)    */
#define SFP004_CMD_FDIVL      0x4020 /* FDIV.L  <ea>,FP0                    */
#define SFP004_CMD_FMULL      0x4023 /* FMUL.L  <ea>,FP0                    */
#define SFP004_CMD_FINTRZ     0x0003 /* FINTRZ  FP0,FP0  (trunc toward 0)   */
#define SFP004_CMD_FMOVEL_OUT 0x6000 /* FMOVE.L FP0,<ea>  (integer result)  */

static int s_fpu_available = 0;
static int s_fpu_armed     = 0;

/* float <-> raw 32-bit pattern without aliasing UB. */
typedef union { float f; unsigned long u; } sfp004_word;

void sfp004_init(void)
{
    long cookie = 0;

    s_fpu_available = 0;
    if (C_FOUND == Getcookie(C__FPU, &cookie)) {
        if (cookie & SFP004_COOKIE)
            s_fpu_available = 1;
    }
}

int sfp004_available(void)
{
    return s_fpu_available;
}

/*
 * Arm/active gate for *implicit* dispatch (i.e. callers that can't manage their
 * own supervisor mode, such as FixedDiv). sfp004_init() detects in user mode
 * (cookie only), but every CIR access is supervisor-only, so a hot path like
 * FixedDiv must not dispatch until the program is in supervisor. The host calls
 * sfp004_arm() once it has entered (and will stay in) supervisor mode; until
 * then sfp004_active() is false and such callers use their software path.
 * (The direct sfp004_add/.../fixdiv entry points still assume the caller is in
 * supervisor — sfp004_arm() is just the signal for implicit users.)
 */
void sfp004_arm(void)
{
    s_fpu_armed = 1;
}

int sfp004_active(void)
{
    return s_fpu_available && s_fpu_armed;
}

/* Memory-mapped CIR accessors — every coprocessor register touch is an explicit
 * inline-asm move of the correct width (word for the control registers, long for
 * the operand), which is the whole point of "dispatch via inline asm" here. */
static inline void  cir_ww(volatile short *a, short v) { __asm__ __volatile__("movew %1,%0" : "=m"(*a) : "d"(v) : "memory"); }
static inline short cir_rw(volatile short *a)          { short v; __asm__ __volatile__("movew %1,%0" : "=d"(v) : "m"(*a)); return v; }
static inline void  cir_wl(volatile long  *a, long  v) { __asm__ __volatile__("movel %1,%0" : "=m"(*a) : "d"(v) : "memory"); }
static inline long  cir_rl(volatile long  *a)          { long  v; __asm__ __volatile__("movel %1,%0" : "=d"(v) : "m"(*a)); return v; }

/* FRESTORE a null state frame (format byte $00) to bring the coprocessor to a
 * known idle state. A board whose FPU was not reset at boot can power up in a
 * state where the response register never reflects command processing; this
 * resets it. Harmless if the FPU is already idle. */
static void sfp004_reset(void)
{
    cir_ww(SFP004_RESTORE, 0x0000);
}

/*
 * Spin while the coprocessor is still working (RESPONSE == the $8900 busy
 * primitive); return as soon as it presents a transfer-data primitive. The
 * first RESPONSE read also starts the command processing. The guard is just a
 * hard upper bound so a missing/wedged coprocessor can never hang the game —
 * generous enough for slow ops (e.g. FSQRT) but, since the loop exits the
 * moment the FPU is ready, it normally runs only a handful of iterations.
 */
static void sfp004_wait(void)
{
    long guard = 0x4000L;
    while ((unsigned short)cir_rw(SFP004_RESPONSE) == SFP004_RESP_BUSY && --guard)
        ;
}

/*
 * Run one memory-source single-precision instruction on FP0: write the command
 * word, wait for the coprocessor to ask for the source operand, then hand over
 * the 32-bit operand. Used both to load FP0 (FMOVE_IN) and to apply an op.
 */
static void sfp004_op(unsigned short cmd, unsigned long operand)
{
    cir_ww(SFP004_COMMAND, (short)cmd);
    sfp004_wait();
    cir_wl(SFP004_OPERAND, (long)operand);
}

/* Read FP0 back as a single via FMOVE.S FP0,<ea>: issue the command, wait for
 * the coprocessor to present the result, then read it from the operand reg. */
static unsigned long sfp004_read_fp0(void)
{
    cir_ww(SFP004_COMMAND, (short)SFP004_CMD_FMOVE_OUT);
    sfp004_wait();
    return (unsigned long)cir_rl(SFP004_OPERAND);
}

/*
 * Common dyadic helper: FP0 = a <op> b, single precision, on the 68882.
 *
 * No cache handling is needed: the dialog was once bracketed with a Mega STE
 * cache disable/restore on the theory the 16 MHz cache would serve the polled
 * RESPONSE reads stale, but the real dead-FPU symptom was a missing FRESTORE
 * reset, and the dialog was confirmed correct at 16 MHz + cache without the
 * guard (the $FFFAxx CIR region evidently isn't cached, same as the MFP). The
 * FRESTORE reset, by contrast, is required.
 */
static float sfp004_dyadic(unsigned short opcmd, float a, float b)
{
    sfp004_word wa, wb, wr;

    wa.f = a;
    wb.f = b;
    sfp004_reset();                         /* known idle state   */
    sfp004_op(SFP004_CMD_FMOVE_IN, wa.u); /* FP0 = a            */
    sfp004_op(opcmd, wb.u);                 /* FP0 = FP0 <op> b   */
    wr.u = sfp004_read_fp0();               /* result = FP0       */
    return wr.f;
}

float sfp004_add(float a, float b)
{
    if (s_fpu_available)
        return sfp004_dyadic(SFP004_CMD_FADD, a, b);
    return a + b;
}

float sfp004_sub(float a, float b)
{
    if (s_fpu_available)
        return sfp004_dyadic(SFP004_CMD_FSUB, a, b);
    return a - b;
}

float sfp004_mul(float a, float b)
{
    if (s_fpu_available)
        return sfp004_dyadic(SFP004_CMD_FMUL, a, b);
    return a * b;
}

float sfp004_div(float a, float b)
{
    if (s_fpu_available)
        return sfp004_dyadic(SFP004_CMD_FDIV, a, b);
    return a / b;
}

/* Monadic FSQRT: FP0 = sqrt(x), single precision, on the 68882. Same dialog as
 * a dyadic op minus the FP0 preload — FSQRT.S takes its source from memory. */
float sfp004_sqrt(float x)
{
    sfp004_word wx, wr;

    if (!s_fpu_available)
        return (float)sqrt((double)x); /* libm has sqrt(double), not sqrtf */

    wx.f = x;
    sfp004_reset();                          /* known idle state   */
    sfp004_op(SFP004_CMD_FSQRT, wx.u);     /* FP0 = sqrt(x)      */
    wr.u = sfp004_read_fp0();                /* result = FP0       */
    return wr.f;
}

/*
 * 16.16 fixed-point divide on the 68882: result = a / b in 16.16, computed as
 * trunc(a/b * 65536). Drops in for Doom's FixedDiv — same overflow clamp, and
 * **bit-exact** with the integer long-division it replaces.
 *
 * Why bit-exact (do not "simplify" away the FINTRZ): Doom's simulation is
 * deterministic and replays from recorded inputs, so a 1-ULP difference in any
 * FixedDiv result desyncs demos — including the title-screen attract demos and
 * -timedemo (which we use to benchmark this very change). It also keeps the
 * runtime FPU divides consistent with the lookup tables R_Init builds in user
 * mode with the *integer* FixedDiv (before supervisor is entered). FMOVE.L on
 * its own rounds to nearest; FINTRZ forces truncation toward zero, matching the
 * integer divide (and the C double cast in the fallback) exactly.
 *
 * The dialog feeds 32-bit *integer* operands (FMOVE.L / FDIV.L / FMUL.L) — the
 * FPU works in 80-bit extended internally, so no doubles are needed — then
 * FINTRZ truncates and FMOVE.L reads the integer result. The overflow guard
 * mirrors Doom's FixedDiv. Supervisor mode required (CIR access).
 */
long sfp004_fixdiv(long a, long b)
{
    long aa = a < 0 ? -a : a;
    long ab = b < 0 ? -b : b;

    if ((aa >> 14) >= ab)               /* would overflow 16.16 — clamp */
        return (a ^ b) < 0 ? (long)0x80000000L : (long)0x7FFFFFFFL;

    if (!s_fpu_available)
        return (long)((double)a / (double)b * 65536.0); /* truncates toward 0 */

    sfp004_reset();                                  /* known idle state    */
    sfp004_op(SFP004_CMD_FMOVEL_IN, (unsigned long)a); /* FP0 = (long)a     */
    sfp004_op(SFP004_CMD_FDIVL,     (unsigned long)b); /* FP0 /= (long)b    */
    sfp004_op(SFP004_CMD_FMULL,     65536UL);          /* FP0 *= 65536      */
    cir_ww(SFP004_COMMAND, (short)SFP004_CMD_FINTRZ);  /* trunc toward zero */
    sfp004_wait();
    cir_ww(SFP004_COMMAND, (short)SFP004_CMD_FMOVEL_OUT);
    sfp004_wait();
    return (long)cir_rl(SFP004_OPERAND);             /* result = (long)FP0  */
}
