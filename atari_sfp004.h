/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/**
 * File: atari_sfp004.h
 * Description: Optional 68881/68882 FPU dispatch via the SFP-004 memory-mapped
 *              coprocessor interface — public header.
 *
 * Detects a memory-mapped MC68881/68882 floating-point coprocessor of the
 * SFP-004 type (the standard $FFFA40 register interface — a card, or a 68882
 * fitted to the motherboard with a GAL address decoder) via the _FPU cookie at
 * startup. When present, single-precision floating-point operations are
 * dispatched to the coprocessor through its memory-mapped coprocessor interface
 * registers (CIR) with inline asm. When absent, the same entry points fall back
 * to the C operators, which the toolchain compiles to the libgcc soft-float
 * routines (__addsf3 etc.) when built -msoft-float (never -m68881).
 *
 * Self-contained, so it drops into any -msoft-float Atari ST/STE/TT project:
 * set SFP004_CIR_BASE for the target board (defaults to the standard $FFFA40),
 * call sfp004_init() once at startup, then gate FPU use on sfp004_available().
 */

#ifndef ATARI_SFP004_H
#define ATARI_SFP004_H

/* Detect the coprocessor and latch availability. Call once at startup (as part
 * of the existing FPU-detection process in atari_checkcpu.c). Safe to call from
 * user or supervisor mode; only the cookie jar is touched here. The CIR itself
 * lives in supervisor-only address space, so the dispatch entry points below
 * must only be used once the program is in supervisor mode (STDOOM stays in
 * supervisor from I_Init onwards). */
void sfp004_init(void);

/* 1 if a usable 68882 was detected by sfp004_init(), else 0. Cheap to call;
 * use it to choose the hardware path over soft-float at runtime. */
int sfp004_available(void);

/* Arm/active gate for callers that can't manage their own supervisor mode (e.g.
 * FixedDiv). Detection runs in user mode, but the CIR dispatch is supervisor
 * only — call sfp004_arm() once after entering (and staying in) supervisor mode;
 * sfp004_active() is then true and such callers may dispatch. */
void sfp004_arm(void);
int  sfp004_active(void);

/* Single-precision dispatch entry points. Each runs on the 68882 when
 * sfp004_available(), otherwise via the C operator (libgcc soft-float). */
float sfp004_add(float a, float b);
float sfp004_sub(float a, float b);
float sfp004_mul(float a, float b);
float sfp004_div(float a, float b);

/* Single-precision square root: one FSQRT on the 68882 when available, else
 * sqrt() (libm soft-float). This is where the coprocessor earns its keep — a
 * transcendental is one FPU instruction but hundreds of cycles in software.
 * NOTE: the fallback pulls in libm (sqrt), so link with -lm where there's no FPU. */
float sfp004_sqrt(float x);

/* 16.16 fixed-point divide on the 68882, bit-exact (truncate toward zero) with an
 * integer long-division FixedDiv, including the same overflow clamp. Drop-in for
 * Doom's FixedDiv; falls back to a double divide when no FPU. Supervisor mode
 * required when an FPU is present (CIR access) — see the note in atari_sfp004.c. */
long sfp004_fixdiv(long a, long b);

/* -------------------------------------------------------------------------
 * Fused-dispatch (session) layer
 *
 * Every scalar entry point above pays for a complete coprocessor dialog per
 * flop: FRESTORE, FMOVE.S in, the op, FMOVE.S out — four command/response
 * exchanges plus the call itself. A hot inner loop (a raymarcher step, a dot
 * product) spends most of its time in that per-call overhead. This layer
 * exposes the dialog primitives so a caller can keep intermediates in FP0-FP7
 * across many operations and read back only final results: begin a session
 * (one FRESTORE), issue any mix of memory-source and register-register
 * commands, then read out what's needed.
 *
 * Rules: supervisor mode only, and only when sfp004_available() — these
 * primitives don't check, callers gate themselves. A session owns the FPU
 * until its last read-out; don't call the scalar entry points mid-session
 * (they FRESTORE, clobbering the FP register file). Operands and results are
 * raw single-precision bit patterns (use a float/unsigned long union).
 */

/* Extension-word opcodes (bits 6-0 of the cpGEN command word). FSGLDIV and
 * FSGLMUL are the 68882's fast single-precision variants (results rounded to
 * single) — 15-30 FPU cycles cheaper than FDIV/FMUL, and exact for pipelines
 * whose data are singles anyway. */
#define SFP004_FOP_FMOVE   0x00 /* FMOVE  (register copy / memory load)    */
#define SFP004_FOP_FINTRZ  0x03 /* FINTRZ (truncate toward zero)           */
#define SFP004_FOP_FSQRT   0x04 /* FSQRT  (monadic)                        */
#define SFP004_FOP_FABS    0x18 /* FABS   (monadic)                        */
#define SFP004_FOP_FNEG    0x1A /* FNEG   (monadic)                        */
#define SFP004_FOP_FDIV    0x20 /* FDIV                                    */
#define SFP004_FOP_FADD    0x22 /* FADD                                    */
#define SFP004_FOP_FMUL    0x23 /* FMUL                                    */
#define SFP004_FOP_FSGLDIV 0x24 /* FSGLDIV (fast single divide)            */
#define SFP004_FOP_FSGLMUL 0x27 /* FSGLMUL (fast single multiply)          */
#define SFP004_FOP_FSUB    0x28 /* FSUB                                    */

/* Command-word builders (cpGEN format, per the command-word notes in
 * atari_sfp004.c: bit 14 = R/M, bit 13 = 0, bits 12-10 = source specifier
 * (R/M=1: operand format; R/M=0: source FP register), bits 9-7 = destination
 * FP register, bits 6-0 = opcode. FMOVE-out uses bits 15-13 = 011, bits
 * 12-10 = result format, bits 9-7 = source FP register.)
 *
 *   SFP004_C_MEM_S(fop, dst) —  op.S  #operand,FPdst   (memory single in)
 *   SFP004_C_REG(fop, s, d)  —  op.X  FPs,FPd          (register-register)
 *   SFP004_C_OUT_S(src)      —  FMOVE.S FPsrc,#result  (single read-out)
 *   SFP004_C_OUT_L(src)      —  FMOVE.L FPsrc,#result  (long-int read-out,
 *                               pair with FINTRZ to replace soft __fixsfsi)
 */
#define SFP004_C_MEM_S(fop, dst) (0x4400 | ((dst) << 7) | (fop))
#define SFP004_C_REG(fop, src, dst) (((src) << 10) | ((dst) << 7) | (fop))
#define SFP004_C_OUT_S(src) (0x6400 | ((src) << 7))
#define SFP004_C_OUT_L(src) (0x6000 | ((src) << 7))

/*
 * The session primitives are static inline: a fused kernel issues dozens of
 * dialogs, and a jsr/rts plus argument marshalling per dialog is measurable
 * against the ~150-cycle dialog itself. A host-side test build can define
 * SFP004_MOCK before including this header to get plain prototypes instead
 * and supply its own (e.g. simulated-68881) implementations.
 */
#ifdef SFP004_MOCK

void sfp004_begin(void);       /* FRESTORE null: start a session from idle  */
void sfp004_cmd(unsigned short cmd);              /* register-register op   */
void sfp004_cmd_in(unsigned short cmd, unsigned long operand); /* mem source */
unsigned long sfp004_cmd_out(unsigned short cmd); /* read a result register */

#else /* !SFP004_MOCK */

/* CIR register map — see the base-address note in atari_sfp004.c (the map is
 * defined here so both the .c dialogs and these inlines share one copy). */
#define SFP004_CIR_BASE 0x00fffa40UL
#define SFP004_RESPONSE ((volatile short *)(SFP004_CIR_BASE + 0x00)) /* w */
#define SFP004_CONTROL  ((volatile short *)(SFP004_CIR_BASE + 0x02)) /* w */
#define SFP004_SAVE     ((volatile short *)(SFP004_CIR_BASE + 0x04)) /* w */
#define SFP004_RESTORE  ((volatile short *)(SFP004_CIR_BASE + 0x06)) /* w */
#define SFP004_COMMAND  ((volatile short *)(SFP004_CIR_BASE + 0x0a)) /* w */
#define SFP004_OPERAND  ((volatile long  *)(SFP004_CIR_BASE + 0x10)) /* l */
#define SFP004_RESP_BUSY 0x8900

/* Explicit-width CIR accessors, same dialog as atari_sfp004.c's private
 * statics: write the command, spin while RESPONSE reads the $8900 busy
 * primitive (the guard only bounds a wedged/missing coprocessor), then
 * transfer data. */
static inline void sfp004_i_ww(volatile short *a, short v) { __asm__ __volatile__("movew %1,%0" : "=m"(*a) : "d"(v) : "memory"); }
static inline short sfp004_i_rw(volatile short *a)          { short v; __asm__ __volatile__("movew %1,%0" : "=d"(v) : "m"(*a)); return v; }
static inline void sfp004_i_wl(volatile long *a, long v)    { __asm__ __volatile__("movel %1,%0" : "=m"(*a) : "d"(v) : "memory"); }
static inline long sfp004_i_rl(volatile long *a)            { long v; __asm__ __volatile__("movel %1,%0" : "=d"(v) : "m"(*a)); return v; }

static inline void sfp004_i_wait(void)
{
    long guard = 0x4000L;
    while ((unsigned short)sfp004_i_rw(SFP004_RESPONSE) == SFP004_RESP_BUSY && --guard)
        ;
}

/* FRESTORE null: start a session from idle */
static inline void sfp004_begin(void)
{
    sfp004_i_ww(SFP004_RESTORE, 0x0000);
}

/* register-register op */
static inline void sfp004_cmd(unsigned short cmd)
{
    sfp004_i_ww(SFP004_COMMAND, (short)cmd);
    sfp004_i_wait();
}

/* memory-source op */
static inline void sfp004_cmd_in(unsigned short cmd, unsigned long operand)
{
    sfp004_i_ww(SFP004_COMMAND, (short)cmd);
    sfp004_i_wait();
    sfp004_i_wl(SFP004_OPERAND, (long)operand);
}

/* read a result register */
static inline unsigned long sfp004_cmd_out(unsigned short cmd)
{
    sfp004_i_ww(SFP004_COMMAND, (short)cmd);
    sfp004_i_wait();
    return (unsigned long)sfp004_i_rl(SFP004_OPERAND);
}

#endif /* SFP004_MOCK */

#endif /* ATARI_SFP004_H */
