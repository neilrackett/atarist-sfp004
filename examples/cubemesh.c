/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * CUBEMESH.C - flat-shaded spinning cube for Atari ST/STE/Mega STE
 *
 * The traditional one: a benchmark-friendly companion to knotmesh.c for
 * comparing STs of different specs. Same pipeline, minimum geometry: the
 * 68882 (via the atarist-sfp004 session layer) transforms 8 vertices per
 * frame in fused coprocessor sessions; the 68000 does integer backface
 * culling, fixed-point lighting, and span fills at full 320x200. A cube is
 * convex, so backface culling alone gives correct visibility - no depth
 * sort, and no per-vertex depth either.
 *
 * White numbers top-left: total frame ms, FPU transform ms, paint ms
 * (200 Hz system tick). Falls back to scalar soft-float without the FPU;
 * on a Mega STE the 16 MHz clock + cache are enabled at startup.
 */

#include <string.h>
#include <mint/osbind.h>

#include "atari_sfp004.h"
#include "atari_megaste.h"
#include "sintab.h"

#define NV 8
#define NF 6

/* ---------- geometry ---------- */
#define CUBE_S   0.75f    /* half edge: bounding sphere r = 1.30            */
#define CAM_DIST 3.0f
#define PROJ_F   180.0f   /* max |x'|/zc over that sphere is 0.481: +/-87px */

/* ---------- screen constants ---------- */
#define SCREEN_W    320
#define SCREEN_H    200
#define CENTRE_X    (SCREEN_W / 2)
#define CENTRE_Y    (SCREEN_H / 2)
#define PLANE_WORDS (SCREEN_W / 16)
#define PLANES        4
#define SCANLINE_BYTES (PLANE_WORDS * PLANES * 2)
#define SCREEN_BYTES  (SCANLINE_BYTES * SCREEN_H)

/* ---------- palette: same cyan->white ramp as the other demos ---------- */
static const unsigned short palette[16] = {
    0x000, 0x019, 0x092, 0x023, 0x834, 0x8BC, 0x145, 0x1CD,
    0x956, 0x2DE, 0xA67, 0xBEF, 0xC7F, 0xD7F, 0xEFF, 0xFFF,
};

/* ---------- double-buffer ---------- */
static unsigned char screen_buf[2][SCREEN_BYTES + 256];
static unsigned char *screens[2];
static int cur_buf = 0;

/* ---------- vec3 / float bits ---------- */
typedef struct { float x, y, z; } Vec3;

typedef union { float f; unsigned long u; long l; } FloatBits;
static inline long fbits(float x)          { FloatBits v; v.f = x; return v.l; }
static inline unsigned long fu(float x)    { FloatBits v; v.f = x; return v.u; }

static int use_fpu = 0;

static inline float fp_add(float a, float b) { return sfp004_add(a, b); }
static inline float fp_sub(float a, float b) { return sfp004_sub(a, b); }
static inline float fp_mul(float a, float b) { return sfp004_mul(a, b); }
static inline float fp_div(float a, float b) { return sfp004_div(a, b); }

/* ---------- mesh ---------- */
static Vec3 overts[NV];
static short fnorm_fix[NF][3];        /* 8.8 object-space face normals */
static unsigned short fvert[NF][4];   /* quad vertex indices, outward winding */

static short vsx[NV], vsy[NV];        /* per-frame screen coordinates */

static void build_mesh(void)
{
    /* vertex i: bit 0 = +x, bit 1 = +y, bit 2 = +z */
    static const unsigned char face[NF][4] = {
        { 0, 1, 3, 2 },   /* z- */
        { 4, 5, 7, 6 },   /* z+ */
        { 0, 1, 5, 4 },   /* y- */
        { 2, 3, 7, 6 },   /* y+ */
        { 0, 2, 6, 4 },   /* x- */
        { 1, 3, 7, 5 },   /* x+ */
    };
    int i, f, k;

    for (i = 0; i < NV; i++) {
        overts[i].x = (i & 1) ? CUBE_S : -CUBE_S;
        overts[i].y = (i & 2) ? CUBE_S : -CUBE_S;
        overts[i].z = (i & 4) ? CUBE_S : -CUBE_S;
    }

    for (f = 0; f < NF; f++) {
        Vec3 e1, e2, n, v0;
        for (k = 0; k < 4; k++)
            fvert[f][k] = face[f][k];
        /* geometric normal from the diagonals; the cube is centred on the
         * origin, so n.v0 > 0 means outward - flip winding to match */
        v0 = overts[fvert[f][0]];
        e1.x = fp_sub(overts[fvert[f][2]].x, v0.x);
        e1.y = fp_sub(overts[fvert[f][2]].y, v0.y);
        e1.z = fp_sub(overts[fvert[f][2]].z, v0.z);
        e2.x = fp_sub(overts[fvert[f][3]].x, overts[fvert[f][1]].x);
        e2.y = fp_sub(overts[fvert[f][3]].y, overts[fvert[f][1]].y);
        e2.z = fp_sub(overts[fvert[f][3]].z, overts[fvert[f][1]].z);
        n.x = fp_sub(fp_mul(e1.y, e2.z), fp_mul(e1.z, e2.y));
        n.y = fp_sub(fp_mul(e1.z, e2.x), fp_mul(e1.x, e2.z));
        n.z = fp_sub(fp_mul(e1.x, e2.y), fp_mul(e1.y, e2.x));
        if (fbits(fp_add(fp_add(fp_mul(n.x, v0.x), fp_mul(n.y, v0.y)),
                         fp_mul(n.z, v0.z))) < 0) {
            unsigned short t = fvert[f][1];
            fvert[f][1] = fvert[f][3];
            fvert[f][3] = t;
            n.x = -n.x; n.y = -n.y; n.z = -n.z;
        }
        /* the cross product of an axis-aligned face's diagonals is itself
         * axis-aligned, so the unit normal is just the component signs -
         * testing the magnitude bits, because the zero components come out
         * of the cross product as IEEE negative zero */
        fnorm_fix[f][0] = (short)(((fbits(n.x) & 0x7FFFFFFFL) == 0) ? 0
                                  : (fbits(n.x) > 0 ? 256 : -256));
        fnorm_fix[f][1] = (short)(((fbits(n.y) & 0x7FFFFFFFL) == 0) ? 0
                                  : (fbits(n.y) > 0 ? 256 : -256));
        fnorm_fix[f][2] = (short)(((fbits(n.z) & 0x7FFFFFFFL) == 0) ? 0
                                  : (fbits(n.z) > 0 ? 256 : -256));
    }
}

/* ---------- per-frame rotation matrix (Rx * Ry) ---------- */
static float mrot[9];

static void build_matrix(int ay, int ax)
{
    float cy = lut_cos(ay), sy = lut_sin(ay);
    float cx = lut_cos(ax), sx = lut_sin(ax);
    mrot[0] = cy;                 mrot[1] = 0.0f; mrot[2] = sy;
    mrot[3] = fp_mul(sx, sy);     mrot[4] = cx;   mrot[5] = fp_mul(-sx, cy);
    mrot[6] = fp_mul(-cx, sy);    mrot[7] = sx;   mrot[8] = fp_mul(cx, cy);
}

/* ---------- vertex transform + perspective, fused ---------- */
#define C_MEM(fop, dst)      SFP004_C_MEM_S(SFP004_FOP_##fop, dst)
#define C_REG(fop, src, dst) SFP004_C_REG(SFP004_FOP_##fop, src, dst)
#define C_OUTL(src)          SFP004_C_OUT_L(src)

static void xform_vertex_fpu(const Vec3 *v, int idx)
{
    sfp004_begin();
    sfp004_cmd_in(C_MEM(FMOVE, 0), fu(v->x));
    sfp004_cmd_in(C_MEM(FMOVE, 1), fu(v->y));
    sfp004_cmd_in(C_MEM(FMOVE, 2), fu(v->z));
    /* FP3 = x'  (mrot[1] is structurally 0 for Rx*Ry: term dropped) */
    sfp004_cmd(C_REG(FMOVE, 0, 3));
    sfp004_cmd_in(C_MEM(FSGLMUL, 3), fu(mrot[0]));
    sfp004_cmd(C_REG(FMOVE, 2, 4));
    sfp004_cmd_in(C_MEM(FSGLMUL, 4), fu(mrot[2]));
    sfp004_cmd(C_REG(FADD, 4, 3));
    /* FP5 = y' */
    sfp004_cmd(C_REG(FMOVE, 0, 5));
    sfp004_cmd_in(C_MEM(FSGLMUL, 5), fu(mrot[3]));
    sfp004_cmd(C_REG(FMOVE, 1, 4));
    sfp004_cmd_in(C_MEM(FSGLMUL, 4), fu(mrot[4]));
    sfp004_cmd(C_REG(FADD, 4, 5));
    sfp004_cmd(C_REG(FMOVE, 2, 4));
    sfp004_cmd_in(C_MEM(FSGLMUL, 4), fu(mrot[5]));
    sfp004_cmd(C_REG(FADD, 4, 5));
    /* FP6 = zc = CAM_DIST - z' */
    sfp004_cmd(C_REG(FMOVE, 0, 6));
    sfp004_cmd_in(C_MEM(FSGLMUL, 6), fu(mrot[6]));
    sfp004_cmd(C_REG(FMOVE, 1, 4));
    sfp004_cmd_in(C_MEM(FSGLMUL, 4), fu(mrot[7]));
    sfp004_cmd(C_REG(FADD, 4, 6));
    sfp004_cmd(C_REG(FMOVE, 2, 4));
    sfp004_cmd_in(C_MEM(FSGLMUL, 4), fu(mrot[8]));
    sfp004_cmd(C_REG(FADD, 4, 6));
    sfp004_cmd(C_REG(FNEG, 6, 6));
    sfp004_cmd_in(C_MEM(FADD, 6), fu(CAM_DIST));
    /* perspective: FP7 = PROJ_F / zc, scale, truncate - no depth needed:
     * a convex solid is fully visibility-sorted by backface culling */
    sfp004_cmd_in(C_MEM(FMOVE, 7), fu(PROJ_F));
    sfp004_cmd(C_REG(FSGLDIV, 6, 7));
    sfp004_cmd(C_REG(FSGLMUL, 7, 3));
    sfp004_cmd(C_REG(FSGLMUL, 7, 5));
    sfp004_cmd(C_REG(FINTRZ, 3, 3));
    sfp004_cmd(C_REG(FINTRZ, 5, 5));
    vsx[idx] = (short)(CENTRE_X + (long)sfp004_cmd_out(C_OUTL(3)));
    vsy[idx] = (short)(CENTRE_Y - (long)sfp004_cmd_out(C_OUTL(5)));
}

static void xform_vertex_soft(const Vec3 *v, int idx)
{
    float xp = fp_add(fp_mul(v->x, mrot[0]), fp_mul(v->z, mrot[2]));
    float yp = fp_add(fp_add(fp_mul(v->x, mrot[3]), fp_mul(v->y, mrot[4])),
                      fp_mul(v->z, mrot[5]));
    float zp = fp_add(fp_add(fp_mul(v->x, mrot[6]), fp_mul(v->y, mrot[7])),
                      fp_mul(v->z, mrot[8]));
    float zc = fp_add(-zp, CAM_DIST);
    float fz = fp_div(PROJ_F, zc);
    vsx[idx] = (short)(CENTRE_X + (int)fp_mul(fz, xp));
    vsy[idx] = (short)(CENTRE_Y - (int)fp_mul(fz, yp));
}

/* ---------- per-face lighting, integer (as knotmesh.c) ---------- */
static const Vec3 LIGHT = { 0.577f, 0.577f, 0.577f };
static short lobj_fix[3];        /* 8.8 light in object space */

static void build_light(void)
{
    Vec3 l;
    l.x = fp_add(fp_add(fp_mul(mrot[0], LIGHT.x), fp_mul(mrot[3], LIGHT.y)),
                 fp_mul(mrot[6], LIGHT.z));
    l.y = fp_add(fp_add(fp_mul(mrot[1], LIGHT.x), fp_mul(mrot[4], LIGHT.y)),
                 fp_mul(mrot[7], LIGHT.z));
    l.z = fp_add(fp_add(fp_mul(mrot[2], LIGHT.x), fp_mul(mrot[5], LIGHT.y)),
                 fp_mul(mrot[8], LIGHT.z));
    lobj_fix[0] = (short)(int)fp_mul(l.x, 256.0f);
    lobj_fix[1] = (short)(int)fp_mul(l.y, 256.0f);
    lobj_fix[2] = (short)(int)fp_mul(l.z, 256.0f);
}

static int face_colour(int f)
{
    long d = (long)fnorm_fix[f][0] * lobj_fix[0]
           + (long)fnorm_fix[f][1] * lobj_fix[1]
           + (long)fnorm_fix[f][2] * lobj_fix[2];   /* 16.16 */
    long idx;
    if (d <= 0)
        return 2;
    idx = 2 + ((d * 13L) >> 16);
    if (idx > 15) idx = 15;
    return (int)idx;
}

/* ---------- span rasteriser (as knotmesh.c) ---------- */
static short span_min[SCREEN_H], span_max[SCREEN_H];

static const unsigned long edge_lmask[16] = {
    0xFFFFFFFFUL, 0x7FFF7FFFUL, 0x3FFF3FFFUL, 0x1FFF1FFFUL,
    0x0FFF0FFFUL, 0x07FF07FFUL, 0x03FF03FFUL, 0x01FF01FFUL,
    0x00FF00FFUL, 0x007F007FUL, 0x003F003FUL, 0x001F001FUL,
    0x000F000FUL, 0x00070007UL, 0x00030003UL, 0x00010001UL,
};
static const unsigned long edge_rmask[16] = {
    0x80008000UL, 0xC000C000UL, 0xE000E000UL, 0xF000F000UL,
    0xF800F800UL, 0xFC00FC00UL, 0xFE00FE00UL, 0xFF00FF00UL,
    0xFF80FF80UL, 0xFFC0FFC0UL, 0xFFE0FFE0UL, 0xFFF0FFF0UL,
    0xFFF8FFF8UL, 0xFFFCFFFCUL, 0xFFFEFFFEUL, 0xFFFFFFFFUL,
};

static void trace_edge(int xa, int ya, int xb, int yb)
{
    long x, dx;
    int y;

    if (ya == yb) {
        if ((unsigned)ya < SCREEN_H) {
            if (xa > xb) { int t = xa; xa = xb; xb = t; }
            if (xa < span_min[ya]) span_min[ya] = (short)xa;
            if (xb > span_max[ya]) span_max[ya] = (short)xb;
        }
        return;
    }
    if (ya > yb) {
        int t = xa; xa = xb; xb = t;
        t = ya; ya = yb; yb = t;
    }
    x  = (long)xa << 16;
    dx = ((long)(xb - xa) << 16) / (yb - ya);
    for (y = ya; y <= yb; y++, x += dx) {
        if ((unsigned)y < SCREEN_H) {
            int xi = (int)(x >> 16);
            if (xi < span_min[y]) span_min[y] = (short)xi;
            if (xi > span_max[y]) span_max[y] = (short)xi;
        }
    }
}

static void fill_quad(unsigned char *scr, int f, int c)
{
    int xs[4], ys[4], ymin = SCREEN_H, ymax = -1;
    int k, y;
    unsigned long pat01, pat23;
    unsigned char *base;

    for (k = 0; k < 4; k++) {
        xs[k] = vsx[fvert[f][k]];
        ys[k] = vsy[fvert[f][k]];
        if (ys[k] < ymin) ymin = ys[k];
        if (ys[k] > ymax) ymax = ys[k];
    }
    if (ymax < 0 || ymin > SCREEN_H - 1) return;
    for (y = ymin; y <= ymax; y++) { span_min[y] = SCREEN_W; span_max[y] = -1; }
    for (k = 0; k < 4; k++)
        trace_edge(xs[k], ys[k], xs[(k + 1) & 3], ys[(k + 1) & 3]);

    pat01 = ((c & 1) ? 0xFFFF0000UL : 0UL) | ((c & 2) ? 0x0000FFFFUL : 0UL);
    pat23 = ((c & 4) ? 0xFFFF0000UL : 0UL) | ((c & 8) ? 0x0000FFFFUL : 0UL);
    base = scr + ymin * SCANLINE_BYTES;
    for (y = ymin; y <= ymax; y++, base += SCANLINE_BYTES) {
        int x0 = span_min[y], x1 = span_max[y];
        int w0, w1, w;
        unsigned long m0, m1;
        unsigned long *row = (unsigned long *)base;

        if (x1 < x0)
            continue;
        w0 = x0 >> 4; w1 = x1 >> 4;
        m0 = edge_lmask[x0 & 15];
        m1 = edge_rmask[x1 & 15];
        if (w0 == w1) {
            unsigned long m = m0 & m1;
            row[w0 * 2]     = (row[w0 * 2]     & ~m) | (pat01 & m);
            row[w0 * 2 + 1] = (row[w0 * 2 + 1] & ~m) | (pat23 & m);
        } else {
            row[w0 * 2]     = (row[w0 * 2]     & ~m0) | (pat01 & m0);
            row[w0 * 2 + 1] = (row[w0 * 2 + 1] & ~m0) | (pat23 & m0);
            for (w = w0 + 1; w < w1; w++) {
                row[w * 2]     = pat01;
                row[w * 2 + 1] = pat23;
            }
            row[w1 * 2]     = (row[w1 * 2]     & ~m1) | (pat01 & m1);
            row[w1 * 2 + 1] = (row[w1 * 2 + 1] & ~m1) | (pat23 & m1);
        }
    }
}

/* ---------- screen plumbing (as knotmesh.c) ---------- */
static void clear_screen(unsigned char *scr) { memset(scr, 0, SCREEN_BYTES); }

#define BLT_EM1  (*(volatile unsigned short *)0xFFFF8A28UL)
#define BLT_EM2  (*(volatile unsigned short *)0xFFFF8A2AUL)
#define BLT_EM3  (*(volatile unsigned short *)0xFFFF8A2CUL)
#define BLT_DXI  (*(volatile short          *)0xFFFF8A2EUL)
#define BLT_DYI  (*(volatile short          *)0xFFFF8A30UL)
#define BLT_DADR (*(volatile unsigned long  *)0xFFFF8A32UL)
#define BLT_XCNT (*(volatile unsigned short *)0xFFFF8A36UL)
#define BLT_YCNT (*(volatile unsigned short *)0xFFFF8A38UL)
#define BLT_HOP  (*(volatile unsigned char  *)0xFFFF8A3AUL)
#define BLT_OP   (*(volatile unsigned char  *)0xFFFF8A3BUL)
#define BLT_CTRL (*(volatile unsigned char  *)0xFFFF8A3CUL)
#define BLT_SKEW (*(volatile unsigned char  *)0xFFFF8A3DUL)

static int have_blitter = 0;

static void blit_clear(unsigned char *scr)
{
    BLT_EM1 = 0xFFFF; BLT_EM2 = 0xFFFF; BLT_EM3 = 0xFFFF;
    BLT_DXI = 2; BLT_DYI = 2;
    BLT_DADR = (unsigned long)scr;
    BLT_XCNT = PLANE_WORDS * PLANES;
    BLT_YCNT = SCREEN_H;
    BLT_HOP = 0; BLT_OP = 0; BLT_SKEW = 0;
    BLT_CTRL = 0xC0;                   /* busy + hog: run to completion */
    while (BLT_CTRL & 0x80)
        ;
}

static void wait_vbl(void) {
    volatile long *vbl_count = (volatile long *)0x462L;
    long t = *vbl_count;
    while (*vbl_count == t) {}
}

static void set_screen(void *addr) {
    unsigned long a = (unsigned long)addr;
    *((volatile unsigned char *)0xFF8201L) = (unsigned char)(a >> 16);
    *((volatile unsigned char *)0xFF8203L) = (unsigned char)(a >> 8);
    *((volatile unsigned char *)0xFF820DL) = (unsigned char)(a);
}

/* ---------- ms readouts (as knotmesh.c) ---------- */
#define HZ200 ((volatile unsigned long *)0x4BAL)

static const unsigned char digfont[10][5] = {
    { 7, 5, 5, 5, 7 }, { 2, 6, 2, 2, 7 }, { 7, 1, 7, 4, 7 }, { 7, 1, 7, 1, 7 },
    { 5, 5, 7, 1, 1 }, { 7, 4, 7, 1, 7 }, { 7, 4, 7, 5, 7 }, { 7, 1, 2, 2, 2 },
    { 7, 5, 7, 5, 7 }, { 7, 5, 7, 1, 7 },
};

static void put_pixel(unsigned char *scr, int x, int y) {
    unsigned short mask = (unsigned short)(0x8000U >> (x & 15));
    unsigned short *w = (unsigned short *)(scr + y * SCANLINE_BYTES) + (x >> 4) * PLANES;
    w[0] |= mask; w[1] |= mask; w[2] |= mask; w[3] |= mask;
}

static void draw_ms(unsigned char *scr, unsigned long ms, int row) {
    int col = 2 + 5 * 4;
    int digits = 0;
    if (ms > 99999UL) ms = 99999UL;
    do {
        const unsigned char *glyph = digfont[ms % 10];
        int gx, gy;
        col -= 4;
        for (gy = 0; gy < 5; gy++)
            for (gx = 0; gx < 3; gx++)
                if (glyph[gy] & (4 >> gx))
                    put_pixel(scr, col + gx, row + gy);
        ms /= 10;
        digits++;
    } while (ms && digits < 5);
}

/* ---------- render one frame ---------- */
static unsigned long xform_ms;
static unsigned long paint_ms;

static void render_frame(unsigned char *scr, int angle_y, int angle_x)
{
    int v, f;
#ifndef SFP004_MOCK
    unsigned long t0 = *HZ200;
#endif

    build_matrix(angle_y, angle_x);
    build_light();

    for (v = 0; v < NV; v++) {
        if (use_fpu) xform_vertex_fpu(&overts[v], v);
        else         xform_vertex_soft(&overts[v], v);
    }
#ifndef SFP004_MOCK
    xform_ms = (*HZ200 - t0) * 5UL;
#endif

    if (have_blitter)
        blit_clear(scr);
    else
        clear_screen(scr);

#ifndef SFP004_MOCK
    { unsigned long t1 = *HZ200;
#endif
    /* convex: backface culling is the whole visibility story */
    for (f = 0; f < NF; f++) {
        int v0 = fvert[f][0], v1 = fvert[f][1], v2 = fvert[f][2];
        long area = (long)(vsx[v1] - vsx[v0]) * (vsy[v2] - vsy[v0])
                  - (long)(vsy[v1] - vsy[v0]) * (vsx[v2] - vsx[v0]);
        if (area >= 0)
            continue;
        fill_quad(scr, f, face_colour(f));
    }
#ifndef SFP004_MOCK
    paint_ms = (*HZ200 - t1) * 5UL; }
#endif
}

/* ---------- main ---------- */
int main(void)
{
    int i;

    screens[0] = (unsigned char *)(((unsigned long)screen_buf[0] + 255) & ~255UL);
    screens[1] = (unsigned char *)(((unsigned long)screen_buf[1] + 255) & ~255UL);

    sfp004_init();
    use_fpu = sfp004_available();
    have_blitter = (int)(Blitmode(-1) & 1);
    void *old_phys = Physbase();

    long ssp = Super(0L);
    megaste_enable_16mhz_cache();
    sfp004_arm();
    build_mesh();

    unsigned short old_pal[16];
    unsigned char old_res = *((volatile unsigned char *)0xFF8260L);
    for (i = 0; i < 16; i++)
        old_pal[i] = *((volatile unsigned short *)(0xFF8240L + i * 2));

    for (i = 0; i < 16; i++)
        *((volatile unsigned short *)(0xFF8240L + i * 2)) = palette[i];
    *((volatile unsigned short *)0xFF8260L) = 0;

    clear_screen(screens[0]);
    clear_screen(screens[1]);

    int angle_y = 0;
    int angle_x = 0;

    while (1) {
        unsigned char *draw = screens[cur_buf];
        unsigned long tick = *HZ200;

        render_frame(draw, angle_y, angle_x);
        draw_ms(draw, (*HZ200 - tick) * 5UL, 2);   /* total frame ms      */
        draw_ms(draw, xform_ms, 10);               /* FPU transform phase */
        draw_ms(draw, paint_ms, 18);               /* fills + lighting    */

        /* base register latches at the NEXT VBL: point it at the finished
         * buffer BEFORE waiting, so when the wait returns the swap has
         * actually happened and the other buffer is safely off-screen */
        set_screen(draw);
        wait_vbl();
        cur_buf ^= 1;

        angle_y = (angle_y + 2) & 511;
        angle_x = (angle_x + 1) & 511;

        if (Bconstat(2)) {
            if ((Bconin(2) & 0xFF) == 27) break;   /* ESC exits */
        }
    }

    for (i = 0; i < 16; i++)
        *((volatile unsigned short *)(0xFF8240L + i * 2)) = old_pal[i];
    *((volatile unsigned char *)0xFF8260L) = old_res;
    set_screen(old_phys);

    Super((void *)ssp);
    return 0;
}
