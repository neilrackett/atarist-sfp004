/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * MANDEL.C - an endless, shining Mandelbrot zoom for Atari ST/STE/Mega STE
 *
 * Dives into the Mandelbrot set on its own, forever. Each frame zooms in 2x
 * on a Misiurewicz point, where the set repeats its spirals and branches at
 * every depth, down to where the FPU's 64-bit mantissa runs out (2^46 in);
 * then the next dive starts, at another point. The 68882 (via the
 * atarist-sfp004 session layer, its hot loop in inline asm) iterates
 * z = z^2 + c in its 80-bit registers and tests for escape with its own
 * conditional predicates; the 68000 turns where z escaped into a smooth
 * colour value and dithers it into 13 colour bands. A VBL interrupt rotates
 * those 13 palette entries all the while, so the bands shine outwards even
 * while a frame is coming in.
 *
 * Zooming in 2x on the screen's centre pixel puts every other pixel of the
 * new frame on a pixel of the old one, so a frame starts as the old one's
 * middle, magnified, and only the pixels in between are new. Those whose
 * neighbours agree to within a colour step are guessed rather than worked
 * out; guesses are never carried into the next frame, so a filament that
 * slips between neighbours is back a frame later.
 *
 * Any key shows or hides the info bar, ESC quits. Falls back to soft-float
 * without the FPU (correct, but each frame then takes many minutes). On a
 * Mega STE the 16 MHz clock + cache are enabled at startup.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mint/osbind.h>

#include "atari_sfp004.h"
#include "atari_megaste.h"

/* ---------- screen ---------- */
#define SCREEN_W       320
#define SCREEN_H       200
#define BAR_Y          184      /* the info bar: the last two text rows   */
#define CENTRE_X       160      /* the pixel on the target                */
#define CENTRE_Y       100
#define SCANLINE_BYTES 160

/* ---------- the zoom ---------- */
#define START_W   3.36          /* width of the first frame of a dive     */
#define MAX_STEP  46            /* 2^46 in: about where 80 bits run out   */
#define BAILOUT   256           /* |z|^2 to escape, big for smooth colours */
#define GUESS_T   4             /* neighbours within one colour step       */

/* ---------- colours ---------- */
#define RAMP      13            /* shining colours: palette entries 1-13  */
#define SUBSTEPS  4             /* dithered steps from one to the next     */
#define FINE      (RAMP * SUBSTEPS)
#define PER_ITER  2             /* quarter colour steps per iteration      */
#define COL_BAR   14            /* info bar background                     */
#define COL_TEXT  15            /* info bar text                           */
#define CYCLE_VBLS 2            /* VBLs per quarter step of the shine      */

/* s_val: quarter colour steps, with these special values */
#define INSIDE    0x7ffe        /* a point in the set                      */
#define UNKNOWN   0xffff        /* not worked out yet                      */
#define GUESSED   0x8000        /* flag: guessed, not worked out           */
#define VALUE(v)  ((v) & 0x7fff)

/* Circular key colours (4 bits per channel) that the shine runs through */
typedef struct {
    const char *name;
    int n;
    unsigned char rgb[8][3];
} palette_t;

static const palette_t s_palettes[] = {
    { "ICE",     6, { {1,2,8}, {3,6,13}, {7,11,15}, {15,15,15}, {8,12,15}, {3,6,12} } },
    { "SUNSET",  6, { {3,0,8}, {9,0,9}, {15,3,4}, {15,9,0}, {15,15,6}, {14,6,2} } },
    { "GOLD",    6, { {4,1,0}, {10,4,0}, {15,10,2}, {15,15,9}, {13,9,2}, {8,3,0} } },
    { "NEON",    6, { {5,0,8}, {12,0,12}, {15,6,10}, {10,15,4}, {15,15,10}, {6,13,2} } },
    { "PINK",    6, { {6,0,4}, {12,2,8}, {15,8,12}, {15,14,15}, {12,6,14}, {7,2,10} } },
};

/* Misiurewicz points: the orbit of 0 falls onto a repelling cycle, so the
 * set looks much the same around them at every depth, and escape counts
 * grow only slowly as the zoom goes in */
typedef struct {
    const char *name;
    double re, im;
    int pal;
} target_t;

static const target_t s_targets[] = {
    { "NORTH ANTENNA",  -0.10109636384562219, 0.95628651080914151, 0 },
    { "ELEPHANT VALLEY", 0.31323493159149535, 0.02548432002954753, 1 },
    { "SPIKE",          -1.54368901269207637, 0.00000000000000000, 2 },
    { "QUARTER BULB",    0.29705677781801204, 0.61042818300878698, 3 },
    { "WEST SPIRAL",    -1.41911618645799287, 0.12569543584899998, 4 },
};
#define NUM_TARGETS ((int)(sizeof(s_targets) / sizeof(s_targets[0])))

/* ---------- state ---------- */
static unsigned short s_val[SCREEN_H][SCREEN_W];
static unsigned char *s_screen;
static int  use_fpu;
static int  s_maxiter;
static int  s_bar;                      /* info bar shown */
static int  s_quit;

/* the view: c(x, y) = (re0 + x * dx, im0 - y * dx) */
typedef union { double d; unsigned long u[2]; } dbits;
typedef union { float f; unsigned long u; } fbits;
static dbits s_re0, s_im0, s_dx, s_ndx;

/* ---------- shine: the VBL rotates palette entries 1-13 ---------- */
static unsigned short s_fine[2][FINE];          /* STE-encoded colours */
static const unsigned short * volatile s_ramp;  /* the one being shown */
static volatile int s_phase;

static void shine_vbl(void)
{
    static int tick;
    const unsigned short *ramp = (const unsigned short *)s_ramp;
    volatile unsigned short *pal = (volatile unsigned short *)0xFFFF8242UL;
    int k, i;

    if (!ramp || ++tick < CYCLE_VBLS)
        return;
    tick = 0;
    if (++s_phase == FINE)
        s_phase = 0;
    i = s_phase;
    for (k = 0; k < RAMP; k++) {
        pal[k] = ramp[i];
        i += SUBSTEPS;
        if (i >= FINE)
            i -= FINE;
    }
}

/* A 4-bit channel in the STE's bit order (the ST ignores the low bit) */
static unsigned short ste(int v)
{
    return (unsigned short)(((v & 1) << 3) | (v >> 1));
}

/* Interpolate a palette's key colours round into FINE steps, and show it */
static void set_palette(const palette_t *p)
{
    static int which;
    unsigned short *fine = s_fine[which ^= 1];
    int i, c;

    for (i = 0; i < FINE; i++) {
        int pos = i * p->n * 16 / FINE;
        int k = pos >> 4, f = pos & 15;
        const unsigned char *a = p->rgb[k], *b = p->rgb[(k + 1) % p->n];
        unsigned short rgb = 0;
        for (c = 0; c < 3; c++)
            rgb = (unsigned short)(rgb << 4 | ste((a[c] * (16 - f) + b[c] * f + 8) >> 4));
        fine[i] = rgb;
    }
    s_ramp = fine;          /* one long write: the VBL sees old or new */
}

/* ---------- smooth colour ---------- */
static unsigned char s_log2frac[256];   /* 256 * log2(1 + i/256) */

/*
 * The smooth escape value n + 2 - log2(log2 |z|^2), in quarter colour steps,
 * for z escaping at iteration n. |z|^2 comes as a single's bits; each log2
 * is its exponent plus a table lookup on the top bits of its mantissa, in
 * 8.8 fixed point, plenty for colours.
 */
static unsigned short smooth(int n, unsigned long zz)
{
    long l = ((long)((zz >> 23) & 0xff) - 127) * 256 + s_log2frac[(zz >> 15) & 0xff];
    long ll, v;
    int p = 14;

    while (!(l >> p))                   /* l >= 8 * 256: |z|^2 > 256 */
        p--;
    ll = (long)(p - 8) * 256 + s_log2frac[(l << (15 - p) >> 7) & 0xff];
    v = (((long)(n + 2) * 256 - ll) * PER_ITER) >> 8;
    return (unsigned short)(v < 0 ? 0 : v > INSIDE - 1 ? INSIDE - 1 : v);
}

/* ---------- the fractal on the 68882 ---------- */
#define C_MEM_L(fop, dst)    SFP004_C_MEM_L(SFP004_FOP_##fop, dst)
#define C_MEM_D(fop, dst)    SFP004_C_MEM_D(SFP004_FOP_##fop, dst)
#define C_REG(fop, src, dst) SFP004_C_REG(SFP004_FOP_##fop, src, dst)

/* The hot loop's dialogs, three instructions each: write the command, then
 * read the response until it isn't busy. cmd, resp and busy ($8900) are in
 * registers. */
#define FAST_RR(fop, src, dst)                                          \
    __asm__ __volatile__("movew %1,%0@\n1:\tcmpw %2@,%3\n\tjeq 1b"      \
                         : : "a"(cmd), "i"(C_REG(fop, src, dst)),      \
                             "a"(resp), "d"(busy) : "cc", "memory")

/* One iteration, zr in FPa and zi in FPb: FPf = zr^2, FPa = zr zi,
 * FPb = zi^2, then zi' = 2 zr zi + ci in FPa and zr' = zr^2 - zi^2 + cr in
 * FPf. The new z is (FPf, FPa) and FPb is free, so three iterations take z
 * round FP0, FP1 and FP4 and back, with no register copied back. */
#define SQUARE(a, b, f)                                                 \
    FAST_RR(FMOVE, a, f); FAST_RR(FMUL, a, f);                          \
    FAST_RR(FMUL, b, a); FAST_RR(FMUL, b, b)
#define ADD_C(a, b, f)                                                  \
    FAST_RR(FADD, a, a); FAST_RR(FADD, 3, a);                           \
    FAST_RR(FSUB, b, f); FAST_RR(FADD, 2, f)

static int s_fpu_y;             /* the row FP3 holds ci for */

/* FP6 holds the bailout for the whole dive; begin a session for it */
static void fpu_start(void)
{
    sfp004_begin();
    sfp004_cmd_in(C_MEM_L(FMOVE, 6), BAILOUT);
    s_fpu_y = -1;
}

/*
 * Iterate z = z^2 + c from z = c, all in the FPU's registers: z in two of
 * FP0, FP1 and FP4, c in FP2/FP3, the bailout in FP6 and |z|^2 in FP7.
 * Escape is tested every third iteration, which the smooth value doesn't
 * notice: |z|^2 is further past the bailout by then, and its log-log has
 * gone up by one for each iteration over.
 */
static unsigned short sample_fpu(int x, int y)
{
    volatile short *cmd = SFP004_COMMAND, *resp = SFP004_RESPONSE;
    short busy = (short)SFP004_RESP_BUSY;
    int n, maxiter = s_maxiter;

    if (y != s_fpu_y) {                 /* ci = im0 - y * dx */
        sfp004_cmd_in(C_MEM_L(FMOVE, 3), (unsigned long)y);
        sfp004_cmd_in_d(C_MEM_D(FMUL, 3), s_ndx.u[0], s_ndx.u[1]);
        sfp004_cmd_in_d(C_MEM_D(FADD, 3), s_im0.u[0], s_im0.u[1]);
        s_fpu_y = y;
    }
    sfp004_cmd_in(C_MEM_L(FMOVE, 2), (unsigned long)x);   /* cr = re0 + x * dx */
    sfp004_cmd_in_d(C_MEM_D(FMUL, 2), s_dx.u[0], s_dx.u[1]);
    sfp004_cmd_in_d(C_MEM_D(FADD, 2), s_re0.u[0], s_re0.u[1]);

    FAST_RR(FMOVE, 2, 0);
    FAST_RR(FMOVE, 3, 1);
    for (n = 1; n < maxiter; n += 3) {
        SQUARE(0, 1, 4);
        FAST_RR(FMOVE, 4, 7); FAST_RR(FADD, 1, 7);     /* |z|^2 > bailout? */
        FAST_RR(FCMP, 6, 7);
        if (sfp004_cond(SFP004_P_OGT))
            return smooth(n, sfp004_cmd_out(SFP004_C_OUT_S(7)));
        ADD_C(0, 1, 4);                 /* z in FP4, FP0 */
        SQUARE(4, 0, 1); ADD_C(4, 0, 1); /* z in FP1, FP4 */
        SQUARE(1, 4, 0); ADD_C(1, 4, 0); /* z in FP0, FP1 */
    }
    return INSIDE;
}

/* The same in soft-float, for machines without the FPU */
static unsigned short sample_soft(int x, int y)
{
    double cr = s_re0.d + x * s_dx.d, ci = s_im0.d - y * s_dx.d;
    double zr = cr, zi = ci;
    int n;

    for (n = 1; n < s_maxiter; n++) {
        double t = zr * zr, u = zi * zi;
        if (t + u > BAILOUT) {
            fbits zz;
            zz.f = (float)(t + u);
            return smooth(n, zz.u);
        }
        zi = 2 * zr * zi + ci;
        zr = t - u + cr;
    }
    return INSIDE;
}

static unsigned short sample(int x, int y)
{
    return use_fpu ? sample_fpu(x, y) : sample_soft(x, y);
}

/* ---------- drawing ---------- */
static const unsigned char s_bayer[4][4] = {
    {  0,  8,  2, 10 }, { 12,  4, 14,  6 }, {  3, 11,  1,  9 }, { 15,  7, 13,  5 },
};

/* s_mod[v]: a value's place on the ramp, v % FINE, or FINE for black.
 * s_bits01/23[y & 3][place * 4 + (x & 3)]: its dithered colour's bitplane
 * bits, planes 0 and 2 in bit 0 and planes 1 and 3 in bit 16, so a line is
 * built two planes to a long. */
static unsigned char s_mod[0x8000];
static unsigned long s_bits01[4][(FINE + 1) * 4], s_bits23[4][(FINE + 1) * 4];

static void init_tables(void)
{
    int v, f, r, i, place = 0;

    for (v = 0; v < 0x8000; v++) {
        s_mod[v] = (unsigned char)place;
        if (++place == FINE)
            place = 0;
    }
    s_mod[INSIDE] = s_mod[VALUE(UNKNOWN)] = FINE;

    /* inside black; outside the band colour, dithered towards the next by
     * its quarter steps */
    for (r = 0; r < 4; r++)
        for (f = 0; f <= FINE; f++)
            for (i = 0; i < 4; i++) {
                int c = 0;
                if (f < FINE) {
                    int c0 = 1 + (f >> 2), c1 = c0 == RAMP ? 1 : c0 + 1;
                    c = ((f & 3) << 2) > s_bayer[r][i] ? c1 : c0;
                }
                s_bits01[r][f << 2 | i] = (unsigned long)(c & 1) | (unsigned long)(c & 2) << 15;
                s_bits23[r][f << 2 | i] = (unsigned long)(c >> 2 & 1) | (unsigned long)(c & 8) << 13;
            }

    for (i = 0; i < 256; i++)
        s_log2frac[i] = (unsigned char)(256.0 * log(1.0 + i / 256.0) / M_LN2 + 0.5);
}

/* Draw screen line y from a row of values, pixel x showing src[x & mask]
 * (a mask of ~1 doubles them across). Not over the info bar. */
static void draw_line(int y, const unsigned short *src, unsigned short mask)
{
    const unsigned long *b01 = s_bits01[y & 3], *b23 = s_bits23[y & 3];
    unsigned short *dst;
    int x, i;

    if (s_bar && y >= BAR_Y)
        return;
    dst = (unsigned short *)(s_screen + y * SCANLINE_BYTES);
    for (x = 0; x < SCREEN_W; x += 16) {
        unsigned long p01 = 0, p23 = 0;
        for (i = 0; i < 16; i++) {
            int k = s_mod[VALUE(src[(x + i) & mask])] << 2 | (i & 3);
            p01 = p01 << 1 | b01[k];
            p23 = p23 << 1 | b23[k];
        }
        *dst++ = (unsigned short)p01;
        *dst++ = (unsigned short)(p01 >> 16);
        *dst++ = (unsigned short)p23;
        *dst++ = (unsigned short)(p23 >> 16);
    }
}

/* ---------- info bar ---------- */
static int  s_info_t, s_info_step;
static double s_info_w;
static long s_info_secs;

static void print_at(int row, const char *s)
{
    char esc[5] = { 27, 'Y', (char)(32 + row), 32, 0 };
    (void)Cconws(esc);
    (void)Cconws(s);
}

/* [-]i.dddddddd */
static char *fmt_coord(char *p, double v)
{
    double a = v < 0 ? -v : v;
    long i = (long)a;
    long f = (long)((a - i) * 100000000.0 + 0.5);
    if (f >= 100000000L) { i++; f -= 100000000L; }
    return p + sprintf(p, "%c%ld.%08ld", v < 0 ? '-' : '+', i, f);
}

/* d.dE-ee */
static char *fmt_sci(char *p, double v)
{
    int e = 0;
    long m;
    while (v < 1.0) { v *= 10.0; e--; }
    while (v >= 10.0) { v /= 10.0; e++; }
    m = (long)(v * 10.0 + 0.5);
    if (m >= 100) { m = 10; e++; }
    return p + sprintf(p, "%ld.%ldE%c%02d", m / 10, m % 10, e < 0 ? '-' : '+', e < 0 ? -e : e);
}

/* Show the current dive and depth, and the last frame's time (or ...) */
static void show_info(void)
{
    char line[64], *p = line;
    const target_t *tg = &s_targets[s_info_t];

    if (!s_bar)
        return;
    p += sprintf(p, "%d %-15s ZOOM 2^%-2d %-7s", s_info_t, tg->name, s_info_step,
                 s_palettes[tg->pal].name);
    if (s_info_secs >= 0)
        sprintf(p, "%4ldS", s_info_secs > 9999 ? 9999 : s_info_secs);
    else
        sprintf(p, "  ...");
    print_at(23, line);

    p = line;
    p += sprintf(p, "RE ");
    p = fmt_coord(p, tg->re);
    p += sprintf(p, " IM ");
    p = fmt_coord(p, tg->im);
    p += sprintf(p, " W ");
    p = fmt_sci(p, s_info_w);
    sprintf(p, "%-*s", (int)(40 - (p - line)), "");
    print_at(24, line);
}

/* Any key shows or hides the info bar; ESC quits */
static void poll_keys(void)
{
    unsigned short row[SCREEN_W];
    int x, y;

    while (Bconstat(2)) {
        if ((Bconin(2) & 0xFF) == 27) {
            s_quit = 1;
            continue;
        }
        s_bar = !s_bar;
        if (s_bar) {
            show_info();
            continue;
        }
        /* the fractal comes back from under the bar; pixels not worked
         * out yet show the block they're in */
        for (y = BAR_Y; y < SCREEN_H; y++) {
            for (x = 0; x < SCREEN_W; x++) {
                int m = 0;
                while (m < 8 && s_val[y & ~m][x & ~m] == UNKNOWN)
                    m = m * 2 + 1;
                row[x] = m < 8 ? s_val[y & ~m][x & ~m] : INSIDE;
            }
            draw_line(y, row, 0xffff);
        }
    }
}

/* ---------- frames ---------- */
#define HZ200 (*(volatile unsigned long *)0x4BAL)

/* The value at (x, y), between neighbours a-d: if they agree to within a
 * colour step, guess their average; otherwise work it out */
static unsigned short guess(unsigned short a, unsigned short b,
                            unsigned short c, unsigned short d, int x, int y)
{
    unsigned short lo, hi;

    a = VALUE(a); b = VALUE(b); c = VALUE(c); d = VALUE(d);
    lo = hi = a;
    if (b < lo) lo = b;
    if (b > hi) hi = b;
    if (c < lo) lo = c;
    if (c > hi) hi = c;
    if (d < lo) lo = d;
    if (d > hi) hi = d;
    if (hi == INSIDE ? lo == INSIDE : hi - lo <= GUESS_T)
        return (unsigned short)(GUESSED | (hi == INSIDE ? INSIDE : (a + b + c + d + 2) >> 2));
    return sample(x, y);
}

/* The first frame of a dive: blocks of 8, then 4, 2 and 1, each pixel
 * between the corners of the block of twice its size */
static void render_full(void)
{
    static const int steps[] = { 8, 4, 2, 1 };
    int p, x, y, i;

    memset(s_val, 0xff, sizeof(s_val));     /* all UNKNOWN */
    for (p = 0; p < 4 && !s_quit; p++) {
        int s = steps[p], s2 = s * 2;
        for (y = 0; y < SCREEN_H && !s_quit; y += s) {
            int y0 = y & ~(s2 - 1);
            int y1 = y0 == y || y0 + s2 >= SCREEN_H ? y0 : y0 + s2;
            for (x = 0; x < SCREEN_W; x += s) {
                int x0 = x & ~(s2 - 1), x1;
                if (p == 0) {
                    s_val[y][x] = sample(x, y);
                    continue;
                }
                if (x0 == x && y0 == y)
                    continue;               /* there from a pass before */
                x1 = x0 == x || x0 + s2 >= SCREEN_W ? x0 : x0 + s2;
                s_val[y][x] = guess(s_val[y0][x0], s_val[y0][x1],
                                    s_val[y1][x0], s_val[y1][x1], x, y);
            }
            for (i = 0; i < s && y + i < SCREEN_H; i++)
                draw_line(y + i, s_val[y], (unsigned short)~(s - 1));
            poll_keys();
        }
    }
}

/* Move the old frame's middle onto every other pixel of the new one. Each
 * new pixel's source is nearer the centre than it is, so going in from the
 * edges never reads a pixel already moved. */
static void magnify(void)
{
    int y, x, i;

    for (i = 0; i < SCREEN_H / 2; i += 2) {
        int rows[2] = { i, SCREEN_H - 2 - i };      /* from the top and bottom in */
        for (y = 0; y < 2; y++) {
            unsigned short *dst = s_val[rows[y]];
            const unsigned short *src = s_val[rows[y] / 2 + CENTRE_Y / 2];
            for (x = 0; x < CENTRE_X; x += 2)       /* left half, from the left */
                dst[x] = src[x / 2 + CENTRE_X / 2];
            for (x = SCREEN_W - 2; x >= CENTRE_X; x -= 2)   /* right, from the right */
                dst[x] = src[x / 2 + CENTRE_X / 2];
            for (x = 1; x < SCREEN_W; x += 2)
                dst[x] = UNKNOWN;
        }
    }
    for (y = 1; y < SCREEN_H; y += 2)
        memset(s_val[y], 0xff, sizeof(s_val[y]));
}

/* Zoom in 2x on the centre: the old frame's middle, magnified, gives every
 * other pixel; the ones in between are new */
static void render_zoom(void)
{
    int x, y;

    magnify();
    for (y = 0; y < SCREEN_H; y += 2) {
        draw_line(y, s_val[y], 0xfffe);
        draw_line(y + 1, s_val[y], 0xfffe);
    }

    /* work out the guesses carried over, and points the old frame gave
     * up on, which may escape with more iterations */
    for (y = 0; y < SCREEN_H && !s_quit; y += 2) {
        unsigned short *row = s_val[y];
        int changed = 0;
        for (x = 0; x < SCREEN_W; x += 2) {
            unsigned short v = row[x];
            if ((v & GUESSED) || v == INSIDE) {
                row[x] = sample(x, y);
                changed |= row[x] != VALUE(v);
            }
        }
        if (changed) {
            draw_line(y, row, 0xfffe);
            draw_line(y + 1, row, 0xfffe);
        }
        poll_keys();
    }

    /* then the pixels in between, from the worked-out ones around them */
    for (y = 0; y < SCREEN_H && !s_quit; y++) {
        unsigned short *row = s_val[y];
        if (!(y & 1)) {
            for (x = 1; x < SCREEN_W; x += 2) {
                unsigned short a = row[x - 1], b = x + 1 < SCREEN_W ? row[x + 1] : a;
                row[x] = guess(a, b, a, b, x, y);
            }
        } else {
            const unsigned short *up = s_val[y - 1];
            const unsigned short *dn = y + 1 < SCREEN_H ? s_val[y + 1] : up;
            for (x = 0; x < SCREEN_W; x++) {
                int l = x & ~1, r = (x & 1) && x + 1 < SCREEN_W ? x + 1 : l;
                row[x] = guess(up[l], up[r], dn[l], dn[r], x, y);
            }
        }
        draw_line(y, row, 0xffff);
        poll_keys();
    }
}

/* Dive into one target, from the first frame down to MAX_STEP */
static void dive(int t)
{
    const target_t *tg = &s_targets[t];
    int step;

    set_palette(&s_palettes[tg->pal]);
    if (use_fpu)
        fpu_start();

    for (step = 0; step <= MAX_STEP && !s_quit; step++) {
        double w = ldexp(START_W, -step);
        unsigned long t0 = HZ200;

        s_dx.d = w / SCREEN_W;
        s_ndx.d = -s_dx.d;
        s_re0.d = tg->re - CENTRE_X * s_dx.d;
        s_im0.d = tg->im + CENTRE_Y * s_dx.d;
        s_maxiter = 100 + 10 * step;
        s_fpu_y = -1;

        s_info_t = t;
        s_info_step = step;
        s_info_w = w;
        s_info_secs = -1;
        show_info();
        if (step == 0)
            render_full();
        else
            render_zoom();
        if (!s_quit) {
            s_info_secs = (long)((HZ200 - t0) / 200);
            show_info();
        }
    }
}

/* ---------- main ---------- */
int main(void)
{
    volatile unsigned short *pal = (volatile unsigned short *)0xFFFF8240UL;
    void (**slot)(void) = NULL;
    unsigned short old_pal[16];
    int old_rez, i, t;
    long ssp;

    sfp004_init();
    use_fpu = sfp004_available();
    old_rez = Getrez();
    init_tables();

    ssp = Super(0L);
    megaste_enable_16mhz_cache();
    sfp004_arm();
    for (i = 0; i < 16; i++)
        old_pal[i] = pal[i];

    Setscreen((void *)-1, (void *)-1, 0);           /* ST low res */
    s_screen = (unsigned char *)Logbase();
    pal[0] = 0x000;
    pal[COL_BAR] = 0x119;
    pal[COL_TEXT] = 0xFFF;
    /* clear, no cursor, no wrap, text in colour 15 on 14 */
    (void)Cconws("\033E\033f\033w\033b?\033c>");

    /* the shine runs in the VBL queue, beside TOS's own */
    {
        void (**queue)(void) = *(void (***)(void))0x456;
        int n = *(volatile short *)0x454;
        for (i = 0; i < n; i++)
            if (!queue[i]) {
                slot = &queue[i];
                *slot = shine_vbl;
                break;
            }
    }

    for (t = 0; !s_quit; t = (t + 1) % NUM_TARGETS)
        dive(t);

    if (slot)
        *slot = NULL;
    for (i = 0; i < 16; i++)
        pal[i] = old_pal[i];
    Setscreen((void *)-1, (void *)-1, old_rez);
    (void)Cconws("\033E\033e\033v");
    Super((void *)ssp);
    return 0;
}
