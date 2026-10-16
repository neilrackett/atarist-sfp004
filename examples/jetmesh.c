/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * JETMESH.C - flat-shaded F-22 for Atari ST/STE/Mega STE
 *
 * The low-poly F-22 from xms0g's 3DOS (an MS-DOS VGA renderer; model data
 * and its BSD licence in f22.h), flown on the knotmesh.c pipeline: the 68882
 * (via the atarist-sfp004 session layer) transforms and projects its 102
 * vertices in fused coprocessor sessions; the 68000 does integer backface
 * culling, a depth-bucket painter's sort of the 200 triangles, fixed-point
 * lighting with a ramp per material, and span fills at full 320x200.
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
#include "f22.h"

#define NV F22_NV
#define NF F22_NF

/* ---------- geometry ---------- */
#define JET_CX   0.1f     /* x of the model's bounding-sphere centre...     */
#define JET_R    1.92f    /* ...and its radius                              */
#define CAM_DIST 4.0f
#define PROJ_F   180.0f   /* max |x'|/zc over that sphere is 0.547: +/-98px */

/* painter's depth: a triangle's three zc*64 values sum to 399..1134 over
 * zc = CAM_DIST -/+ JET_R; less the near end, >> 2 gives buckets 0..183 */
#define DEPTH_NEAR 399

/* ---------- motion: turntable spin, camera swinging above the wings ---------- */
#define ELEV_MID   24      /* LUT units: 17 degrees...                     */
#define ELEV_SWING 40.0f   /* ...+/-28, from just below the wings to 45    */

/* ---------- screen constants ---------- */
#define SCREEN_W    320
#define SCREEN_H    200
#define CENTRE_X    (SCREEN_W / 2)
#define CENTRE_Y    (SCREEN_H / 2)
#define PLANE_WORDS (SCREEN_W / 16)
#define PLANES        4
#define SCANLINE_BYTES (PLANE_WORDS * PLANES * 2)
#define SCREEN_BYTES  (SCANLINE_BYTES * SCREEN_H)

/* ---------- palette: 3DOS's teal sky and paint scheme ---------- */
static const unsigned short palette[16] = {
    0x844,                                 /* 0      sky                 */
    0xB88, 0xE11,                          /* 1-2    engine red          */
    0x81B, 0x12D, 0xABF,                   /* 3-5    canopy blue         */
    0x992, 0x22A, 0xAA3, 0x33B, 0x44C,     /* 6-15   grey ramp, shared   */
    0xCC5, 0xDDD, 0x666, 0x777, 0xFFF,     /*        by hull and wings   */
};

/* ---------- double-buffer ---------- */
static unsigned char screen_buf[2][SCREEN_BYTES + 256];
static unsigned char *screens[2];
static int cur_buf = 0;

/* ---------- vec3 / float bits ---------- */
typedef struct { float x, y, z; } Vec3;

typedef union { float f; unsigned long u; long l; } FloatBits;
static inline unsigned long fu(float x)    { FloatBits v; v.f = x; return v.u; }

static int use_fpu = 0;

static inline float fp_add(float a, float b) { return sfp004_add(a, b); }
static inline float fp_sub(float a, float b) { return sfp004_sub(a, b); }
static inline float fp_mul(float a, float b) { return sfp004_mul(a, b); }
static inline float fp_div(float a, float b) { return sfp004_div(a, b); }
static inline float fp_sqrt(float x)         { return sfp004_sqrt(x); }

static Vec3 v_sub(Vec3 a, Vec3 b) {
    Vec3 r = { fp_sub(a.x, b.x), fp_sub(a.y, b.y), fp_sub(a.z, b.z) };
    return r;
}

static Vec3 v_cross(Vec3 a, Vec3 b) {
    Vec3 r = {
        fp_sub(fp_mul(a.y, b.z), fp_mul(a.z, b.y)),
        fp_sub(fp_mul(a.z, b.x), fp_mul(a.x, b.z)),
        fp_sub(fp_mul(a.x, b.y), fp_mul(a.y, b.x))
    };
    return r;
}

static float v_dot(Vec3 a, Vec3 b) {
    return fp_add(fp_add(fp_mul(a.x, b.x), fp_mul(a.y, b.y)), fp_mul(a.z, b.z));
}

static Vec3 v_norm(Vec3 v) {
    float inv = fp_div(1.0f, fp_sqrt(v_dot(v, v)));
    Vec3 r = { fp_mul(v.x, inv), fp_mul(v.y, inv), fp_mul(v.z, inv) };
    return r;
}

/* ---------- mesh (centred and lit once at startup) ---------- */
static Vec3 overts[NV];               /* object-space vertices, centred */
static short fnorm_fix[NF][3];        /* 8.8 object-space face normals  */

/* per-frame results */
static short vsx[NV], vsy[NV], vsz[NV];   /* screen x, y, depth (zc * 64)   */

static void build_mesh(void)
{
    int v, f;

    for (v = 0; v < NV; v++) {
        overts[v].x = fp_sub(f22_verts[v][0], JET_CX);
        overts[v].y = f22_verts[v][1];
        overts[v].z = f22_verts[v][2];
    }

    /* 3DOS's mesh is closed and consistently wound (every edge shared by
     * two triangles, counter-clockwise from outside), so the edge cross
     * product is the outward normal and screen-space culling needs no
     * winding fix-ups */
    for (f = 0; f < NF; f++) {
        const unsigned char *t = f22_faces[f];
        Vec3 n = v_norm(v_cross(v_sub(overts[t[1]], overts[t[0]]),
                                v_sub(overts[t[2]], overts[t[0]])));
        fnorm_fix[f][0] = (short)(int)fp_mul(n.x, 256.0f);
        fnorm_fix[f][1] = (short)(int)fp_mul(n.y, 256.0f);
        fnorm_fix[f][2] = (short)(int)fp_mul(n.z, 256.0f);
    }
}

/* ---------- per-frame rotation matrix (Rx * Ry) ---------- */
/* Ry spins the jet about its own vertical axis, then Rx tilts the result
 * towards the camera: a turntable, viewed from angle_x above the wings */
static float mrot[9];

static void build_matrix(int ay, int ax)
{
    float cy = lut_cos(ay), sy = lut_sin(ay);
    float cx = lut_cos(ax), sx = lut_sin(ax);
    mrot[0] = cy;                 mrot[1] = 0.0f; mrot[2] = sy;
    mrot[3] = fp_mul(sx, sy);     mrot[4] = cx;   mrot[5] = fp_mul(-sx, cy);
    mrot[6] = fp_mul(-cx, sy);    mrot[7] = sx;   mrot[8] = fp_mul(cx, cy);
}

/* ---------- vertex transform + perspective, fused (as knotmesh.c) ---------- */
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
    /* perspective: FP7 = PROJ_F / zc, scale x'/y', truncate to ints */
    sfp004_cmd_in(C_MEM(FMOVE, 7), fu(PROJ_F));
    sfp004_cmd(C_REG(FSGLDIV, 6, 7));
    sfp004_cmd(C_REG(FSGLMUL, 7, 3));
    sfp004_cmd(C_REG(FSGLMUL, 7, 5));
    sfp004_cmd(C_REG(FINTRZ, 3, 3));
    sfp004_cmd(C_REG(FINTRZ, 5, 5));
    vsx[idx] = (short)(CENTRE_X + (long)sfp004_cmd_out(C_OUTL(3)));
    vsy[idx] = (short)(CENTRE_Y - (long)sfp004_cmd_out(C_OUTL(5)));
    /* depth = trunc(zc * 64) for the painter buckets */
    sfp004_cmd_in(C_MEM(FSGLMUL, 6), fu(64.0f));
    sfp004_cmd(C_REG(FINTRZ, 6, 6));
    vsz[idx] = (short)(long)sfp004_cmd_out(C_OUTL(6));
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
    vsz[idx] = (short)(int)fp_mul(zc, 64.0f);
}

/* ---------- per-face lighting, integer ---------- */
/* As knotmesh.c, three 68000 muls of 8.8 normals against the light rotated
 * into object space. Half-Lambert: n.l spreads over a material's whole
 * ramp from -1 to 1 rather than clamping at 0, so faces turned from the
 * light still show the airframe's shape. */
static const Vec3 LIGHT = { 0.577f, 0.577f, 0.577f };
static short lobj_fix[3];        /* 8.8 light in object space     */

static const unsigned char ramp[4][9] = {
    {  8,  9, 10, 11, 12, 12, 13, 14, 15 },   /* F22_HULL:   light grey */
    {  6,  6,  7,  7,  8,  8,  9,  9, 10 },   /* F22_WING:   dark grey  */
    {  3,  3,  3,  4,  4,  4,  5,  5,  5 },   /* F22_CANOPY: blue       */
    {  1,  1,  1,  1,  2,  2,  2,  2,  2 },   /* F22_ENGINE: red        */
};

static void build_light(void)
{
    /* L_obj = R^T * L (rows of R dotted as columns), then to 8.8 */
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
           + (long)fnorm_fix[f][2] * lobj_fix[2];   /* 16.16, -1..1 */
    int lvl = (int)((d + 0x10000L) >> 14);         /* 0..8          */
    if (lvl < 0) lvl = 0;
    if (lvl > 8) lvl = 8;
    return ramp[f22_faces[f][3]][lvl];
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

/* Rasterise one triangle: DDA the edges into span_min/max, then fill spans
 * with patterns hoisted per face. No clamps: vertices are provably
 * on-screen (PROJ_F above), so spans are too. */
static void fill_tri(unsigned char *scr, int f, int c)
{
    int xs[3], ys[3], ymin = SCREEN_H, ymax = -1;
    int k, y;
    unsigned long pat01, pat23;
    unsigned char *base;

    for (k = 0; k < 3; k++) {
        xs[k] = vsx[f22_faces[f][k]];
        ys[k] = vsy[f22_faces[f][k]];
        if (ys[k] < ymin) ymin = ys[k];
        if (ys[k] > ymax) ymax = ys[k];
    }
    if (ymax < 0 || ymin > SCREEN_H - 1) return;
    for (y = ymin; y <= ymax; y++) { span_min[y] = SCREEN_W; span_max[y] = -1; }
    trace_edge(xs[0], ys[0], xs[1], ys[1]);
    trace_edge(xs[1], ys[1], xs[2], ys[2]);
    trace_edge(xs[2], ys[2], xs[0], ys[0]);

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

/* ---------- painter's sort: depth buckets, far-to-near ---------- */
#define NBUCKET 256
static short bucket_head[NBUCKET];
static short face_next[NF];

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
static unsigned long xform_ms;   /* vertex-pipeline share of the frame */
static unsigned long paint_ms;   /* lighting + span-fill share          */

static void render_frame(unsigned char *scr, int angle_y, int angle_x)
{
    int v, f, b;
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

    /* cull + depth-bucket the visible triangles (integer only) */
    for (b = 0; b < NBUCKET; b++) bucket_head[b] = -1;
    for (f = 0; f < NF; f++) {
        int v0 = f22_faces[f][0], v1 = f22_faces[f][1], v2 = f22_faces[f][2];
        long area = (long)(vsx[v1] - vsx[v0]) * (vsy[v2] - vsy[v0])
                  - (long)(vsy[v1] - vsy[v0]) * (vsx[v2] - vsx[v0]);
        int depth;
        if (area >= 0)
            continue;                     /* back-facing */
        depth = (vsz[v0] + vsz[v1] + vsz[v2] - DEPTH_NEAR) >> 2;
        if (depth < 0) depth = 0;
        if (depth > NBUCKET - 1) depth = NBUCKET - 1;
        face_next[f] = bucket_head[depth];
        bucket_head[depth] = (short)f;
    }

    if (have_blitter)
        blit_clear(scr);
    else
        clear_screen(scr);

#ifndef SFP004_MOCK
    { unsigned long t1 = *HZ200;
#endif
    /* painter: far (large zc) first */
    for (b = NBUCKET - 1; b >= 0; b--) {
        short f2 = bucket_head[b];
        while (f2 >= 0) {
            fill_tri(scr, f2, face_colour(f2));
            f2 = face_next[f2];
        }
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

    int spin = 0;   /* yaw (LUT units, 512 = 360 degrees)  */
    int bob = 0;    /* phase of the camera-elevation swing */

    while (1) {
        unsigned char *draw = screens[cur_buf];
        unsigned long tick = *HZ200;
        int elev = ELEV_MID + (int)fp_mul(lut_sin(bob), ELEV_SWING);

        render_frame(draw, spin, elev);
        draw_ms(draw, (*HZ200 - tick) * 5UL, 2);   /* total frame ms      */
        draw_ms(draw, xform_ms, 10);               /* FPU transform phase */
        draw_ms(draw, paint_ms, 18);               /* lighting + fills    */

        /* base register latches at the NEXT VBL: point it at the finished
         * buffer BEFORE waiting, so when the wait returns the swap has
         * actually happened and the other buffer is safely off-screen */
        set_screen(draw);
        wait_vbl();
        cur_buf ^= 1;

        spin = (spin + 2) & LUT_MASK;
        bob  = (bob + 3) & LUT_MASK;

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
