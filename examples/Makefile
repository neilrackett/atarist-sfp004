# atarist-sfp004 examples — demos for the SFP-004
# by Neil Rackett
#
#   make            build all the demos
#   make lores      TORUS.TOS     raymarcher, 80x50 chunky blocks
#   make hires      TORUSHI.TOS   raymarcher, 160x100 chunky blocks
#   make mesh       KNOTMESH.TOS  polygon mesh, full 320x200
#   make cube       CUBEMESH.TOS  classic spinning cube (spec benchmark)
#   make jet        JETMESH.TOS   F-22 model from 3DOS, full 320x200
#   make mandel     MANDEL.TOS    endless shining Mandelbrot zoom
#   make clean      remove build products
#
# Same toolchain notes as the library Makefile. With atarist-toolkit-docker,
# run from the repository root: `STCMD_NO_TTY=1 stcmd make -C examples`.

CC := m68k-atari-mint-gcc

# Soft-float 68000 code as required by the library (the toolchain default);
# the demos push the frame rate, hence -fomit-frame-pointer.
CFLAGS := -O2 -fomit-frame-pointer -Wall -I. -I..

SFP  := ../atari_sfp004.c
DEPS := ../atari_sfp004.c ../atari_sfp004.h sintab.h atari_megaste.c atari_megaste.h

.PHONY: all lores hires mesh cube jet mandel clean

all: lores hires mesh cube jet mandel

lores: TORUS.TOS

hires: TORUSHI.TOS

mesh: KNOTMESH.TOS

cube: CUBEMESH.TOS

jet: JETMESH.TOS

mandel: MANDEL.TOS

## Real-time SDF raymarcher: every pixel is sphere-traced signed-distance
## work on the 68882 (fused CIR sessions). Slow and glorious.
TORUS.TOS: torus.c $(DEPS)
	$(CC) $(CFLAGS) -o $@ torus.c atari_megaste.c $(SFP) -lm

TORUSHI.TOS: torus.c $(DEPS)
	$(CC) $(CFLAGS) -DHIRES=1 -o $@ torus.c atari_megaste.c $(SFP) -lm

## Faceted mesh: the 68882 does vertex transform/projection, the 68000 does
## integer cull/sort/span-fill. Built for frame rate (-O3 on the demo code).
KNOTMESH.TOS: knotmesh.c $(DEPS)
	$(CC) $(CFLAGS) -O3 -o $@ knotmesh.c atari_megaste.c $(SFP) -lm

## The traditional spinning cube: minimum geometry, same pipeline - a
## friendly benchmark for comparing STs of different specs.
CUBEMESH.TOS: cubemesh.c $(DEPS)
	$(CC) $(CFLAGS) -O3 -o $@ cubemesh.c atari_megaste.c $(SFP) -lm

## The low-poly F-22 from 3DOS (model data in f22.h): the mesh pipeline
## with four materials and a 200-triangle painter's sort.
JETMESH.TOS: jetmesh.c f22.h $(DEPS)
	$(CC) $(CFLAGS) -O3 -o $@ jetmesh.c atari_megaste.c $(SFP) -lm

## Endless Mandelbrot zoom: the 68882 iterates in its 80-bit registers and
## tests escape with its conditional predicates; a VBL shines the palette.
MANDEL.TOS: mandel.c $(DEPS)
	$(CC) $(CFLAGS) -O3 -o $@ mandel.c atari_megaste.c $(SFP) -lm

clean:
	rm -f TORUS.TOS TORUSHI.TOS KNOTMESH.TOS CUBEMESH.TOS JETMESH.TOS MANDEL.TOS
