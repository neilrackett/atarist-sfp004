# SFP-004 for Atari ST

A small C library that enables you to use a memory-mapped MC68881/68882 FPU (the SFP-004
interface) with any computer in the Atari ST family.

By [Neil Rackett](https://x.com/neilrackett).

## Introduction

A 68000 has no coprocessor interface, so you can't drive an FPU with `-m68881`
coprocessor instructions (`FADD` etc.) — they trap. The SFP-004 design maps
the 68881/68882's coprocessor interface registers (CIR) into the ST's address
space at `$FFFA40`, and software drives the FPU by hand: writing command and
operand registers and polling a response register. This is exactly how a 68882
fitted to a Mega STE motherboard with a GAL chip, or an SFP-004 card, is reached.

This library:

- **Detects** the FPU via the `_FPU` cookie (the SFP-004 bit);
- **Dispatches** single-precision `add`/`sub`/`mul`/`div`/`sqrt` and a 16.16
  fixed-point divide to the FPU through the memory-mapped CIR, in hand-written
  inline asm;
- **Falls back** transparently to soft-float (libgcc / libm) when no FPU is
  present, so the same binary runs everywhere.

Build everything `-msoft-float` (the toolchain default for the 68000) — never
`-m68881`.

## Why bother?

It's a lot faster!

Even though every operation goes through the memory-mapped bus handshake, on real
hardware (e.g. a Mega STE with a 68882 installed) it still beats soft-float:

| workload                         | vs                    | speed-up      |
| -------------------------------- | --------------------- | ------------- |
| `add`/`sub`/`mul`/`div` (single) | libgcc soft-float     | **~1.6–2.0×** |
| `sqrt` (single)                  | libm soft-float       | **~75–100×**  |
| `FixedDiv` (16.16)               | integer long-division | **~1.7–2.2×** |

`sqrt` (and any transcendental) is the big win — one FPU instruction versus
hundreds of cycles in software. `FixedDiv` matters because it's the hot divide in
fixed-point renderers like Doom.

## API

```c
#include "atari_sfp004.h"

void  sfp004_init(void);        /* detect the FPU (reads the _FPU cookie)        */
int   sfp004_available(void);   /* 1 if a usable SFP-004 FPU was detected         */

void  sfp004_arm(void);         /* enable implicit dispatch (call in supervisor)  */
int   sfp004_active(void);      /* available() && armed                           */

float sfp004_add(float a, float b);
float sfp004_sub(float a, float b);
float sfp004_mul(float a, float b);
float sfp004_div(float a, float b);
float sfp004_sqrt(float x);

long  sfp004_fixdiv(long a, long b);  /* 16.16 a/b, bit-exact with integer FixedDiv */
```

### Supervisor mode

The CIR lives in supervisor-only address space (`$FFFAxx`). `sfp004_init()` only
reads the cookie and is safe from user mode, but the **dispatch entry points must
be called in supervisor mode**. Either ensure you're in supervisor (`Super()`),
or use the arm/active gate for code that can't manage its own mode (see below).

### `-lm`

`sfp004_sqrt`'s fallback uses `sqrt()`, so link with `-lm` on machines without an FPU.

## Usage

### Implicit dispatch (e.g. accelerating an existing `FixedDiv`)

For a hot routine that's called from code which starts in user mode and only
later enters supervisor, gate on `sfp004_active()` and flip the arm once you're
safely in supervisor:

```c
/* once, after you've entered (and will stay in) supervisor mode: */
sfp004_arm();

/* in the hot path: */
fixed_t FixedDiv(fixed_t a, fixed_t b) {
    if (sfp004_active())
        return (fixed_t)sfp004_fixdiv(a, b);   /* FPU */
    return /* ... your software divide ... */;
}
```

`sfp004_fixdiv` truncates toward zero, so it is **bit-exact** with a classic
integer long-division `FixedDiv` (and applies the same overflow clamp). That
matters for anything deterministic — e.g. Doom demo playback desyncs on a 1-ULP
difference.

### Direct use (you manage supervisor mode)

```c
#include <mint/osbind.h>
#include "atari_sfp004.h"

sfp004_init();

long ssp = Super(0L);                       /* CIR access needs supervisor */
if (sfp004_available()) {
    float r = sfp004_sqrt(2.0f);            /* on the FPU */
    float s = sfp004_mul(r, r);             /* ~2.0 */
}
Super((void *)ssp);
```

When no FPU is present, the same calls return the soft-float result, so you can
call them unconditionally if you don't care which path is taken.

## Configuring the base address

`SFP004_CIR_BASE` in `atari_sfp004.c` defaults to the standard SFP-004 base
`$FFFA40` (Response `$FFFA40`, Command `$FFFA4A`, Operand `$FFFA50`). If your
board decodes the CIR elsewhere, change it there; the register offsets are the
fixed MC68881/68882 map and don't move.

## Building

This library is designed for use with MintLib's `m68k-atari-mint-gcc` compiler.

If you don't have it locally, the easiest route is [atarist-toolkit-docker](https://github.com/sidecartridge/atarist-toolkit-docker); with that, prefix the commands with `stcmd` (e.g. `stcmd make`).

```sh
make            # libsfp004.a + tests/FPUTEST.TOS + tests/FPUBENCH.TOS
make lib        # libsfp004.a only
make tests      # the test apps only
make clean
```

Or just drop `atari_sfp004.c` / `atari_sfp004.h` straight into your project and
compile them with your own build (`-msoft-float`, link `-lm`).

## Tests

- **`tests/FPUTEST.TOS`** — correctness: runs `add`/`sub`/`mul`/`div` on the FPU
  and against soft-float and compares the IEEE-754 bit patterns; prints a CIR
  dialog trace.
- **`tests/FPUBENCH.TOS`** — benchmark: times the FPU against soft-float for the
  arithmetic mix, `sqrt`, and `FixedDiv`, reporting µs/op and the speed-up.

You can run them in any resolution. They report `68882 detected: YES/no`; with no FPU
they exercise (and confirm) the soft-float fallback.

## Credits & licence

Library and tests by [Neil Rackett](https://x.com/neilrackett). Developed on a real
Mega STE with a 68882 and GAL chip, with thanks to the CHZ-Soft "Atari SFP-004 by example"
documentation for the memory-mapped CIR protocol.

Licensed under the GNU GPL v3 — see [LICENSE](LICENSE).
