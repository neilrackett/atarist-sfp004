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

#endif /* ATARI_SFP004_H */
