# atarist-sfp004 — SFP-004 (memory-mapped 68881/68882) FPU library for the Atari ST
# by Neil Rackett
#
# Build with the m68k-atari-mint-gcc cross toolchain (MintLib) on your PATH:
#
#   make            build the static library and the test/benchmark apps
#   make lib        build libsfp004.a only
#   make tests      build tests/FPUTEST.TOS and tests/FPUBENCH.TOS only
#   make clean      remove build products
#
# Using the atarist-toolkit-docker image instead of a local toolchain? Just
# prefix make with the stcmd wrapper, e.g. `stcmd make`.

CC := m68k-atari-mint-gcc
AR := m68k-atari-mint-ar

# Build for the plain 68000 with soft-float (the toolchain default). This is
# REQUIRED: a 68000 has no coprocessor bus, so the FPU is driven by hand through
# memory-mapped registers — never with -m68881 coprocessor instructions. The
# library transparently falls back to soft-float when no SFP-004 is present.
CFLAGS := -O2 -Wall -I.

.PHONY: all lib tests clean

all: lib tests

## Static library -> libsfp004.a
lib: libsfp004.a

libsfp004.a: atari_sfp004.c atari_sfp004.h
	$(CC) $(CFLAGS) -c atari_sfp004.c -o atari_sfp004.o
	$(AR) rcs libsfp004.a atari_sfp004.o

## Standalone correctness + benchmark apps (.TOS).
## -lm: the sqrt fallback pulls in libm (the toolchain has sqrt, not sqrtf).
tests: tests/FPUTEST.TOS tests/FPUBENCH.TOS

tests/FPUTEST.TOS: tests/fputest.c atari_sfp004.c atari_sfp004.h
	$(CC) $(CFLAGS) -o tests/FPUTEST.TOS tests/fputest.c atari_sfp004.c -lm

tests/FPUBENCH.TOS: tests/fpubench.c atari_sfp004.c atari_sfp004.h
	$(CC) $(CFLAGS) -o tests/FPUBENCH.TOS tests/fpubench.c atari_sfp004.c -lm

clean:
	rm -f atari_sfp004.o libsfp004.a tests/FPUTEST.TOS tests/FPUBENCH.TOS
