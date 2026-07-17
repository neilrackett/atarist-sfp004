/*
 * Copyright (C) 2026 Neil Rackett
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * TORUS.C - (2,3) torus-knot raymarcher for Atari ST/STE/Mega STE
 *
 * Uses the SFP-004 memory-mapped 68882 via atarist-sfp004 for hardware float
 * math, falling back to soft-float transparently on machines without it.
 * On a Mega STE the 16 MHz clock + cache are switched on automatically.
 *
 * The knot is two strand circles per cross-section of a carrier torus, with
 * the pair pattern turning 1.5x per revolution (period pi, so it closes
 * seamlessly into a trefoil). The SDF needs cos/sin of 1.5*theta around the
 * main axis but uses NO atan2: cos(theta) and sin(theta) fall out of x/rho
 * and z/rho, the half angle comes from sqrt((1 +/- cos)/2) with sin's sign,
 * and the angle-sum formulas combine them - all mul/sqrt, exactly the ops
 * the 68882 dispatch provides.
 *
 * When the FPU is present, ray setup and each march step run as single fused
 * coprocessor sessions (see the session layer in atari_sfp004.h): operands
 * load once, intermediates stay in FP0-FP7, and only final results are read
 * back - roughly a quarter of the CIR traffic of per-op scalar dispatch.
 *
 * Rendering: chunky-sim in ST low res (320x200, 16 colours), double-buffered.
 *   RENDER_W/H = logical pixels, BLOCK_W/H = physical per logical.
 * Default 80x50 @ 4x4 blocks; HIRES=1 for 160x100 @ 2x2.
 */

#include <string.h>
#include <mint/osbind.h>

#include "atari_sfp004.h"
#include "atari_megaste.h"
#include "sintab.h"

/* ---------- resolution ---------- */
#ifndef HIRES
#define HIRES 0
#endif

#if HIRES
#  define RENDER_W  160
#  define RENDER_H  100
#  define BLOCK_W     2
#  define BLOCK_H     2
#  define MAX_STEPS  12   /* fewer steps to keep fps alive at 4x pixels */
#else
#  define RENDER_W   80
#  define RENDER_H   50
#  define BLOCK_W     4
#  define BLOCK_H     4
#  define MAX_STEPS  16
#endif

/* ---------- screen constants ---------- */
#define SCREEN_W    320
#define SCREEN_H    200
#define PLANE_WORDS (SCREEN_W / 16)   /* 20 words per scanline per plane */
#define PLANES        4
#define SCANLINE_BYTES (PLANE_WORDS * PLANES * 2)  /* 160 bytes */
#define SCREEN_BYTES  (SCANLINE_BYTES * SCREEN_H)  /* 32000 bytes */

/* ---------- palette: cyan->white ramp for the knot, black background ----------
 * STE colour nibbles carry the LSB of the 4-bit gun in bit 3, so a logical
 * level v (0-15) encodes as ((v&1)<<3)|(v>>1). On a plain ST bit 3 is ignored
 * and the same words degrade gracefully to the 3-bit gun (v>>1). Comments
 * give the logical (r,g,b) levels. */
static const unsigned short palette[16] = {
    0x000,  /*  0 background        ( 0, 0, 0) */
    0x019,  /*  1 deepest teal      ( 0, 2, 3) */
    0x092,  /*  2                   ( 0, 3, 4) */
    0x023,  /*  3                   ( 0, 4, 6) */
    0x834,  /*  4                   ( 1, 6, 8) */
    0x8BC,  /*  5                   ( 1, 7, 9) */
    0x145,  /*  6                   ( 2, 8,10) */
    0x1CD,  /*  7                   ( 2, 9,11) */
    0x956,  /*  8 mid cyan          ( 3,10,12) */
    0x2DE,  /*  9                   ( 4,11,13) */
    0xA67,  /* 10                   ( 5,12,14) */
    0xBEF,  /* 11                   ( 7,13,15) */
    0xC7F,  /* 12                   ( 9,14,15) */
    0xD7F,  /* 13                   (11,14,15) */
    0xEFF,  /* 14                   (13,15,15) */
    0xFFF,  /* 15 specular white    (15,15,15) */
};

/* ---------- double-buffer ---------- */
static unsigned char screen_buf[2][SCREEN_BYTES + 256];
static unsigned char *screens[2];
static int cur_buf = 0;

/* ---------- vec3 ---------- */
typedef struct { float x, y, z; } Vec3;

/* ---------- geometry ---------- */
#define TORUS_R    1.0f   /* major radius: the knot's carrier circle       */
#define KNOT_W     0.35f  /* winding radius: strand offset in cross-section */
#define TUBE_R     0.16f  /* strand tube radius                            */
#define SDF_SCALE  0.85f  /* Lipschitz safety: the twisting strand pattern
                             makes the cross-section field over-estimate by
                             up to sqrt(1 + (1.5*W)^2) ~= 1.13, so 1/1.13
                             with a little margin                           */
#define CAM_DIST   3.0f
#define BOUND_R    1.55f  /* R + W + tube + margin: bounding sphere        */
#define BOUND_C    (CAM_DIST * CAM_DIST - BOUND_R * BOUND_R)
#define MARCH_HIT  0.01f  /* on the scaled distance                        */

/* Two-level marching: the knot lies inside a shell of radius SHELL_R around
 * the carrier circle, and the plain-torus shell SDF is a valid lower bound
 * for the knot SDF at a third of the cost - and exact (Lipschitz 1), so
 * shell steps skip SDF_SCALE too. Only within BOUND_NEAR of the shell is the
 * full knot field evaluated. */
#define SHELL_R    (KNOT_W + TUBE_R)
#define BOUND_NEAR 0.06f

/* Temporal warm start: the rotation moves any surface point by at most
 * ~0.06/frame, so a pixel that hit last frame can start marching just short
 * of last frame's hit distance instead of at the bounding sphere. The
 * backoff covers two frames of motion because of the checkerboard below. */
#define WARM_BACKOFF 0.15f

/* Checkerboard: march only alternate pixels each frame (lattice parity flips
 * per frame) and hold the other half from last frame's colours - halves the
 * per-frame FPU work for a one-frame lag on alternate pixels. Set to 0 for
 * every-pixel rendering. */
#ifndef CHECKERBOARD
#define CHECKERBOARD 1
#endif

/* ---------- float bit tricks ---------- */
/* IEEE singles that are non-negative (or NaN, which safely fails every
 * test below) order the same as their bit patterns read as signed longs,
 * and a negative float reads as a negative long - so the march-loop
 * comparisons run in the integer ALU instead of __cmpsf2. */
typedef union { float f; unsigned long u; long l; } FloatBits;
static inline long fbits(float x)            { FloatBits v; v.f = x; return v.l; }
static inline unsigned long fu(float x)      { FloatBits v; v.f = x; return v.u; }
static inline float bitsf(unsigned long u)   { FloatBits v; v.u = u; return v.f; }

/* ---------- FPU-accelerated scalar helpers ---------- */
static int use_fpu = 0; /* sfp004_available(), latched once in main() */

static inline float fp_add(float a, float b) { return sfp004_add(a, b); }
static inline float fp_sub(float a, float b) { return sfp004_sub(a, b); }
static inline float fp_mul(float a, float b) { return sfp004_mul(a, b); }
static inline float fp_div(float a, float b) { return sfp004_div(a, b); }
static inline float fp_sqrt(float x)         { return sfp004_sqrt(x); }

static inline float fp_dot(Vec3 a, Vec3 b) {
    return fp_add(fp_add(fp_mul(a.x, b.x), fp_mul(a.y, b.y)), fp_mul(a.z, b.z));
}

static inline Vec3 fp_normalize(Vec3 v) {
    float inv = fp_div(1.0f, fp_sqrt(fp_dot(v, v)));
    Vec3 r = { fp_mul(v.x, inv), fp_mul(v.y, inv), fp_mul(v.z, inv) };
    return r;
}

/* ---------- rotation helpers (LUT-based, per-frame only) ---------- */
static void rot_y(Vec3 *v, int angle) {
    float s = lut_sin(angle);
    float c = lut_cos(angle);
    float nx = fp_add(fp_mul(c, v->x), fp_mul(s, v->z));
    float nz = fp_sub(fp_mul(c, v->z), fp_mul(s, v->x));  /* -s for right-hand */
    v->x = nx;
    v->z = nz;
}

static void rot_x(Vec3 *v, int angle) {
    float s = lut_sin(angle);
    float c = lut_cos(angle);
    float ny = fp_sub(fp_mul(c, v->y), fp_mul(s, v->z));
    float nz = fp_add(fp_mul(s, v->y), fp_mul(c, v->z));
    v->y = ny;
    v->z = nz;
}

/* ---------- (2,3) torus-knot SDF, scalar path ---------- */
/* Cross-section decomposition of p: strand centres sit at +/-W*(cos psi,
 * sin psi) in (radial, y) cross-section coords, psi = 1.5*theta. Returns the
 * scaled SDF; if centre is non-NULL also returns the nearer strand centre in
 * 3D (used for the shading normal). Used by the soft-float fallback march
 * and by shade(); the FPU march runs the fused version below instead. */
static float knot_eval(Vec3 p, Vec3 *centre)
{
    float rho = fp_sqrt(fp_add(fp_mul(p.x, p.x), fp_mul(p.z, p.z)));
    float inv = fp_div(1.0f, rho);
    float ct  = fp_mul(p.x, inv);                          /* cos(theta)      */
    float st  = fp_mul(p.z, inv);                          /* sin(theta)      */
    float s   = fp_sub(rho, TORUS_R);
    /* Half angle via two square roots, sign of sin(theta/2) copied from
     * sin(theta). (Not sin/(2*cos(theta/2)): dividing by the ill-conditioned
     * small root amplifies the cancellation error near theta = pi.) */
    float h   = fp_mul(0.5f, ct);
    float ch  = fp_sqrt(fp_add(0.5f, h));                  /* cos(theta/2)>=0 */
    float sh  = bitsf(fu(fp_sqrt(fp_sub(0.5f, h)))
                      | (fu(st) & 0x80000000UL));          /* sin(theta/2)    */
    float cp  = fp_sub(fp_mul(ct, ch), fp_mul(st, sh));    /* cos(1.5 theta)  */
    float sp  = fp_add(fp_mul(st, ch), fp_mul(ct, sh));    /* sin(1.5 theta)  */
    float ux  = fp_mul(KNOT_W, cp);
    float uy  = fp_mul(KNOT_W, sp);
    float ax  = fp_sub(s, ux),   ay = fp_sub(p.y, uy);
    float bx  = fp_add(s, ux),   by = fp_add(p.y, uy);
    float d0  = fp_add(fp_mul(ax, ax), fp_mul(ay, ay));
    float d1  = fp_add(fp_mul(bx, bx), fp_mul(by, by));
    int near0 = fbits(d0) < fbits(d1);
    float m   = near0 ? d0 : d1;

    if (centre) {
        float su = near0 ? ux : -ux;
        float sv = near0 ? uy : -uy;
        float rr = fp_add(TORUS_R, su);
        centre->x = fp_mul(rr, ct);
        centre->y = sv;
        centre->z = fp_mul(rr, st);
    }
    return fp_mul(SDF_SCALE, fp_sub(fp_sqrt(m), TUBE_R));
}

/* ---------- fused coprocessor kernels ---------- */
/* Shorthand for the session builders. */
#define C_MEM(fop, dst)      SFP004_C_MEM_S(SFP004_FOP_##fop, dst)
#define C_REG(fop, src, dst) SFP004_C_REG(SFP004_FOP_##fop, src, dst)
#define C_OUT(src)           SFP004_C_OUT_S(src)
#define C_OUTL(src)          SFP004_C_OUT_L(src)

/*
 * Ray setup, one session: rd = normalize(row + fx*right), then the bounding
 * sphere test. For an orthonormal camera basis dot(fwd, row + fx*right) == 1,
 * and ro = -CAM_DIST*fwd, so -b = -dot(ro, rd) = CAM_DIST/|rd_unnorm| - the
 * sphere test needs no second dot product. Returns 0 on a miss (most pixels:
 * ~30 dialogs and out), else fills rd and the march interval [t0, t1].
 */
static int ray_setup_fpu(const Vec3 *right, const Vec3 *row, float fx,
                         Vec3 *rd, float *t0, float *t1)
{
    unsigned long disc;

    sfp004_begin();
    sfp004_cmd_in(C_MEM(FMOVE, 0), fu(fx));           /* FP0 = fx            */
    sfp004_cmd_in(C_MEM(FMOVE, 1), fu(right->x));
    sfp004_cmd(C_REG(FSGLMUL, 0, 1));
    sfp004_cmd_in(C_MEM(FADD, 1), fu(row->x));        /* FP1 = rd_un.x       */
    sfp004_cmd_in(C_MEM(FMOVE, 2), fu(right->y));
    sfp004_cmd(C_REG(FSGLMUL, 0, 2));
    sfp004_cmd_in(C_MEM(FADD, 2), fu(row->y));        /* FP2 = rd_un.y       */
    sfp004_cmd_in(C_MEM(FMOVE, 3), fu(right->z));
    sfp004_cmd(C_REG(FSGLMUL, 0, 3));
    sfp004_cmd_in(C_MEM(FADD, 3), fu(row->z));        /* FP3 = rd_un.z       */
    sfp004_cmd(C_REG(FMOVE, 1, 4));
    sfp004_cmd(C_REG(FSGLMUL, 4, 4));                    /* x^2                 */
    sfp004_cmd(C_REG(FMOVE, 2, 5));
    sfp004_cmd(C_REG(FSGLMUL, 5, 5));
    sfp004_cmd(C_REG(FADD, 5, 4));                    /* + y^2               */
    sfp004_cmd(C_REG(FMOVE, 3, 5));
    sfp004_cmd(C_REG(FSGLMUL, 5, 5));
    sfp004_cmd(C_REG(FADD, 5, 4));                    /* FP4 = |rd_un|^2     */
    sfp004_cmd(C_REG(FSQRT, 4, 4));                   /* FP4 = |rd_un|       */
    sfp004_cmd_in(C_MEM(FMOVE, 0), fu(1.0f));
    sfp004_cmd(C_REG(FSGLDIV, 4, 0));                    /* FP0 = 1/|rd_un|     */
    sfp004_cmd(C_REG(FSGLMUL, 0, 1));                    /* FP1..3 = rd         */
    sfp004_cmd(C_REG(FSGLMUL, 0, 2));
    sfp004_cmd(C_REG(FSGLMUL, 0, 3));
    sfp004_cmd_in(C_MEM(FSGLMUL, 0), fu(CAM_DIST));      /* FP0 = -b            */
    sfp004_cmd(C_REG(FMOVE, 0, 4));
    sfp004_cmd(C_REG(FSGLMUL, 4, 4));                    /* b^2                 */
    sfp004_cmd_in(C_MEM(FSUB, 4), fu((float)BOUND_C));/* FP4 = discriminant  */
    disc = sfp004_cmd_out(C_OUT(4));
    if ((long)disc <= 0)
        return 0;
    /* session continues: FP0 = -b and FP4 = disc are still live */
    sfp004_cmd(C_REG(FSQRT, 4, 4));                   /* FP4 = sq            */
    sfp004_cmd(C_REG(FMOVE, 0, 5));
    sfp004_cmd(C_REG(FSUB, 4, 5));                    /* FP5 = -b - sq = t0  */
    sfp004_cmd(C_REG(FADD, 4, 0));                    /* FP0 = -b + sq = t1  */
    rd->x = bitsf(sfp004_cmd_out(C_OUT(1)));
    rd->y = bitsf(sfp004_cmd_out(C_OUT(2)));
    rd->z = bitsf(sfp004_cmd_out(C_OUT(3)));
    *t0 = bitsf(sfp004_cmd_out(C_OUT(5)));
    *t1 = bitsf(sfp004_cmd_out(C_OUT(0)));
    return 1;
}

/*
 * One march step, one session: d at ro + t*rd and t_next = t + d.
 * Two-level: first the cheap carrier-shell lower bound (leaving rho live in
 * FP7); only if the point is within BOUND_NEAR of the shell does the session
 * continue into the full knot decomposition, same maths as knot_eval() with
 * intermediates held in FP0-FP7. Only the shell distance, the two squared
 * strand distances (host-side min - no FPU compares) and the final d /
 * t_next cross the bus.
 */
static float knot_step_fpu(const Vec3 *ro, const Vec3 *rd, float t, float *t_next)
{
    unsigned long db, q2, st_bits, d_bits;

    sfp004_begin();
    sfp004_cmd_in(C_MEM(FMOVE, 0), fu(t));            /* FP0 = t             */
    sfp004_cmd_in(C_MEM(FMOVE, 1), fu(rd->x));
    sfp004_cmd(C_REG(FSGLMUL, 0, 1));
    sfp004_cmd_in(C_MEM(FADD, 1), fu(ro->x));         /* FP1 = p.x           */
    sfp004_cmd_in(C_MEM(FMOVE, 2), fu(rd->y));
    sfp004_cmd(C_REG(FSGLMUL, 0, 2));
    sfp004_cmd_in(C_MEM(FADD, 2), fu(ro->y));         /* FP2 = p.y           */
    sfp004_cmd_in(C_MEM(FMOVE, 3), fu(rd->z));
    sfp004_cmd(C_REG(FSGLMUL, 0, 3));
    sfp004_cmd_in(C_MEM(FADD, 3), fu(ro->z));         /* FP3 = p.z           */
    sfp004_cmd(C_REG(FMOVE, 1, 4));
    sfp004_cmd(C_REG(FSGLMUL, 4, 4));                 /* x^2                 */
    sfp004_cmd(C_REG(FMOVE, 3, 5));
    sfp004_cmd(C_REG(FSGLMUL, 5, 5));                 /* z^2                 */
    sfp004_cmd(C_REG(FADD, 5, 4));
    sfp004_cmd(C_REG(FSQRT, 4, 4));                   /* FP4 = rho           */
    sfp004_cmd(C_REG(FMOVE, 4, 7));                   /* FP7 = rho (kept)    */
    sfp004_cmd_in(C_MEM(FSUB, 4), fu(TORUS_R));       /* FP4 = s             */
    /* shell lower bound: sqrt(s^2 + y^2) - SHELL_R, keeping q^2 in FP6 */
    sfp004_cmd(C_REG(FMOVE, 4, 5));
    sfp004_cmd(C_REG(FSGLMUL, 5, 5));
    sfp004_cmd(C_REG(FMOVE, 2, 6));
    sfp004_cmd(C_REG(FSGLMUL, 6, 6));
    sfp004_cmd(C_REG(FADD, 5, 6));                    /* FP6 = q^2 (kept)    */
    sfp004_cmd(C_REG(FMOVE, 6, 5));
    sfp004_cmd(C_REG(FSQRT, 5, 5));
    sfp004_cmd_in(C_MEM(FSUB, 5), fu((float)SHELL_R));/* FP5 = shell dist    */
    db = sfp004_cmd_out(C_OUT(5));
    if ((long)db > fbits(BOUND_NEAR)) {               /* far: bound suffices */
        sfp004_cmd(C_REG(FADD, 5, 0));                /* FP0 = t + db        */
        *t_next = bitsf(sfp004_cmd_out(C_OUT(0)));
        return bitsf(db);
    }
    /*
     * Near the shell: knot field, session continues. Instead of building the
     * strand offsets, uses d{0,1}^2 = q^2 + W^2 -/+ 2W*(c.u): with
     * e = s*cos(psi) + y*sin(psi) regrouped by half-angle terms as
     * e = ct*(s*ch + y*sh) + st*(y*ch - s*sh), the nearer strand is simply
     * min = q^2 + W^2 - |2W*e| - one FABS instead of two distance builds
     * and a host-side min.
     */
    sfp004_cmd(C_REG(FSGLDIV, 7, 1));                 /* FP1 = cos(theta)    */
    sfp004_cmd(C_REG(FSGLDIV, 7, 3));                 /* FP3 = sin(theta)    */
    q2 = sfp004_cmd_out(C_OUT(6));                    /* frees FP6           */
    st_bits = sfp004_cmd_out(C_OUT(3));
    /* half angle via two square roots (see knot_eval): FP5 = ch, FP7 = sh */
    sfp004_cmd(C_REG(FMOVE, 1, 5));
    sfp004_cmd_in(C_MEM(FSGLMUL, 5), fu(0.5f));       /* FP5 = cos/2         */
    sfp004_cmd(C_REG(FMOVE, 5, 7));
    sfp004_cmd(C_REG(FNEG, 7, 7));
    sfp004_cmd_in(C_MEM(FADD, 5), fu(0.5f));
    sfp004_cmd(C_REG(FSQRT, 5, 5));                   /* FP5 = cos(theta/2)  */
    sfp004_cmd_in(C_MEM(FADD, 7), fu(0.5f));
    sfp004_cmd(C_REG(FSQRT, 7, 7));                   /* FP7 = |sin(th/2)|   */
    if (st_bits & 0x80000000UL)
        sfp004_cmd(C_REG(FNEG, 7, 7));                /* FP7 = sin(theta/2)  */
    sfp004_cmd(C_REG(FMOVE, 4, 0));
    sfp004_cmd(C_REG(FSGLMUL, 5, 0));                 /* s*ch                */
    sfp004_cmd(C_REG(FMOVE, 2, 6));
    sfp004_cmd(C_REG(FSGLMUL, 7, 6));                 /* y*sh                */
    sfp004_cmd(C_REG(FADD, 6, 0));                    /* FP0 = u             */
    sfp004_cmd(C_REG(FMOVE, 2, 6));
    sfp004_cmd(C_REG(FSGLMUL, 5, 6));                 /* y*ch                */
    sfp004_cmd(C_REG(FSGLMUL, 7, 4));                 /* s*sh (s dead)       */
    sfp004_cmd(C_REG(FSUB, 4, 6));                    /* FP6 = v             */
    sfp004_cmd(C_REG(FSGLMUL, 1, 0));                 /* ct*u                */
    sfp004_cmd(C_REG(FSGLMUL, 3, 6));                 /* st*v                */
    sfp004_cmd(C_REG(FADD, 6, 0));                    /* FP0 = e             */
    sfp004_cmd_in(C_MEM(FSGLMUL, 0), fu(2.0f * KNOT_W));
    sfp004_cmd(C_REG(FABS, 0, 0));
    sfp004_cmd(C_REG(FNEG, 0, 0));                    /* -|2We|              */
    sfp004_cmd_in(C_MEM(FADD, 0), q2);
    sfp004_cmd_in(C_MEM(FADD, 0), fu(KNOT_W * KNOT_W));
    sfp004_cmd(C_REG(FSQRT, 0, 0));                   /* nearer strand dist  */
    sfp004_cmd_in(C_MEM(FSUB, 0), fu(TUBE_R));
    sfp004_cmd_in(C_MEM(FSGLMUL, 0), fu(SDF_SCALE));  /* FP0 = d             */
    d_bits = sfp004_cmd_out(C_OUT(0));
    sfp004_cmd_in(C_MEM(FADD, 0), fu(t));             /* FP0 = t + d         */
    *t_next = bitsf(sfp004_cmd_out(C_OUT(0)));
    return bitsf(d_bits);
}

/* Scalar twin of knot_step_fpu for the soft-float fallback (and the host
 * test harness): same two-level structure, same branch decisions. */
static float knot_step_soft(const Vec3 *ro, const Vec3 *rd, float t, float *t_next)
{
    Vec3 p = {
        fp_add(ro->x, fp_mul(rd->x, t)),
        fp_add(ro->y, fp_mul(rd->y, t)),
        fp_add(ro->z, fp_mul(rd->z, t))
    };
    float rho = fp_sqrt(fp_add(fp_mul(p.x, p.x), fp_mul(p.z, p.z)));
    float s   = fp_sub(rho, TORUS_R);
    float q2  = fp_add(fp_mul(s, s), fp_mul(p.y, p.y));
    float db  = fp_sub(fp_sqrt(q2), (float)SHELL_R);
    float d;

    if (fbits(db) > fbits(BOUND_NEAR)) {
        d = db;
    } else {
        /* mirror of the fused near path: min strand distance^2 as
         * q^2 + W^2 - |2W*e| with e regrouped by half-angle terms */
        float ct = fp_div(p.x, rho);
        float st = fp_div(p.z, rho);
        float h  = fp_mul(0.5f, ct);
        float ch = fp_sqrt(fp_add(0.5f, h));
        float sh = bitsf(fu(fp_sqrt(fp_sub(0.5f, h)))
                         | (fu(st) & 0x80000000UL));
        float u  = fp_add(fp_mul(s, ch), fp_mul(p.y, sh));
        float v  = fp_sub(fp_mul(p.y, ch), fp_mul(s, sh));
        float e  = fp_add(fp_mul(ct, u), fp_mul(st, v));
        float f  = bitsf(fu(fp_mul(2.0f * KNOT_W, e)) & 0x7FFFFFFFUL);
        float m  = fp_add(fp_add(-f, q2), KNOT_W * KNOT_W);
        d = fp_mul(SDF_SCALE, fp_sub(fp_sqrt(m), TUBE_R));
    }
    *t_next = fp_add(t, d);
    return d;
}

/* ---------- scalar ray setup (soft-float fallback) ---------- */
static int ray_setup_soft(const Vec3 *right, const Vec3 *row, float fx,
                          Vec3 *rd, float *t0, float *t1)
{
    Vec3 r = {
        fp_add(row->x, fp_mul(fx, right->x)),
        fp_add(row->y, fp_mul(fx, right->y)),
        fp_add(row->z, fp_mul(fx, right->z))
    };
    float inv = fp_div(1.0f, fp_sqrt(fp_dot(r, r)));
    float nb, disc, sq;

    rd->x = fp_mul(r.x, inv);
    rd->y = fp_mul(r.y, inv);
    rd->z = fp_mul(r.z, inv);
    nb   = fp_mul(CAM_DIST, inv);                     /* -b                  */
    disc = fp_sub(fp_mul(nb, nb), BOUND_C);
    if (fbits(disc) <= 0)
        return 0;
    sq  = fp_sqrt(disc);
    *t0 = fp_sub(nb, sq);
    *t1 = fp_add(nb, sq);
    return 1;
}

/* ---------- raymarch over [t0, t1] ---------- */
static int raymarch(const Vec3 *ro, const Vec3 *rd, float t, float t1, float *hit_t)
{
    long t1b = fbits(t1);
    int i;

    for (i = 0; i < MAX_STEPS; i++) {
        float tn, d;
        d = use_fpu ? knot_step_fpu(ro, rd, t, &tn)
                    : knot_step_soft(ro, rd, t, &tn);
        if (fbits(d) < fbits(MARCH_HIT)) { *hit_t = t; return 1; }
        t = tn;
        if (fbits(t) > t1b) return 0;
    }
    return 0;
}

/* ---------- shade: diffuse + specular, map to palette index 1..15 ---------- */
static const Vec3 LIGHT = { 0.577f, 0.577f, 0.577f }; /* normalised */

static int shade_soft(Vec3 hit, Vec3 rd) {
    /* Nearest-strand-centre normal: exact for an untwisted tube and
     * indistinguishable at 15 shades - avoids the 6 SDF evaluations of
     * central differences. */
    Vec3 c, n;
    float diff, spec, intensity;
    int idx;

    (void)knot_eval(hit, &c);
    n.x = fp_sub(hit.x, c.x);
    n.y = fp_sub(hit.y, c.y);
    n.z = fp_sub(hit.z, c.z);
    n = fp_normalize(n);

    diff = fp_dot(n, LIGHT);
    if (fbits(diff) < 0) diff = 0.0f;

    /* Blinn-Phong specular; the 0.5 scale before normalising is a no-op */
    {
        Vec3 h = {
            fp_sub(LIGHT.x, rd.x),
            fp_sub(LIGHT.y, rd.y),
            fp_sub(LIGHT.z, rd.z)
        };
        h = fp_normalize(h);
        spec = fp_dot(n, h);
    }
    if (fbits(spec) < 0) spec = 0.0f;
    /* spec^8 via repeated multiply */
    spec = fp_mul(spec, spec);
    spec = fp_mul(spec, spec);
    spec = fp_mul(spec, spec);

    intensity = fp_add(fp_mul(0.7f, diff), fp_mul(0.3f, spec));
    idx = (int)fp_mul(intensity, 14.0f) + 1;
    if (idx < 1)  idx = 1;
    if (idx > 15) idx = 15;
    return idx;
}

/*
 * Fused shade, one session. Once per hit pixel: recompute the cross-section
 * decomposition at ro + t*rd, pick the nearer strand (host-side min again),
 * then work the normal in cylindrical coordinates - n = (a*ct, b, a*st) with
 * a = s -/+ ux, b = y -/+ uy, and |n|^2 = a^2 + b^2, so nothing leaves the
 * cross-section plane until the light dots. Diffuse and n.h read out for the
 * host-side clamps (feeding clamped bits back as memory operands), and the
 * final palette index comes back through FINTRZ + FMOVE.L - no soft-float
 * __fixsfsi anywhere.
 */
static int shade_fpu(const Vec3 *ro, const Vec3 *rd, float t)
{
    unsigned long st_bits, m0, m1, diff, hy, hz, nh;
    int near0;
    long idx;

    sfp004_begin();
    sfp004_cmd_in(C_MEM(FMOVE, 0), fu(t));            /* FP0 = t             */
    sfp004_cmd_in(C_MEM(FMOVE, 1), fu(rd->x));
    sfp004_cmd(C_REG(FSGLMUL, 0, 1));
    sfp004_cmd_in(C_MEM(FADD, 1), fu(ro->x));         /* FP1 = p.x           */
    sfp004_cmd_in(C_MEM(FMOVE, 2), fu(rd->y));
    sfp004_cmd(C_REG(FSGLMUL, 0, 2));
    sfp004_cmd_in(C_MEM(FADD, 2), fu(ro->y));         /* FP2 = p.y           */
    sfp004_cmd_in(C_MEM(FMOVE, 3), fu(rd->z));
    sfp004_cmd(C_REG(FSGLMUL, 0, 3));
    sfp004_cmd_in(C_MEM(FADD, 3), fu(ro->z));         /* FP3 = p.z           */
    sfp004_cmd(C_REG(FMOVE, 1, 4));
    sfp004_cmd(C_REG(FSGLMUL, 4, 4));
    sfp004_cmd(C_REG(FMOVE, 3, 5));
    sfp004_cmd(C_REG(FSGLMUL, 5, 5));
    sfp004_cmd(C_REG(FADD, 5, 4));
    sfp004_cmd(C_REG(FSQRT, 4, 4));                   /* FP4 = rho           */
    sfp004_cmd(C_REG(FMOVE, 4, 7));                   /* FP7 = rho (kept)    */
    sfp004_cmd_in(C_MEM(FSUB, 4), fu(TORUS_R));       /* FP4 = s             */
    sfp004_cmd(C_REG(FSGLDIV, 7, 1));                 /* FP1 = cos(theta)    */
    sfp004_cmd(C_REG(FSGLDIV, 7, 3));                 /* FP3 = sin(theta)    */
    st_bits = sfp004_cmd_out(C_OUT(3));
    sfp004_cmd(C_REG(FMOVE, 1, 5));
    sfp004_cmd_in(C_MEM(FSGLMUL, 5), fu(0.5f));
    sfp004_cmd(C_REG(FMOVE, 5, 6));
    sfp004_cmd(C_REG(FNEG, 6, 6));
    sfp004_cmd_in(C_MEM(FADD, 5), fu(0.5f));
    sfp004_cmd(C_REG(FSQRT, 5, 5));                   /* FP5 = cos(theta/2)  */
    sfp004_cmd_in(C_MEM(FADD, 6), fu(0.5f));
    sfp004_cmd(C_REG(FSQRT, 6, 6));
    if (st_bits & 0x80000000UL)
        sfp004_cmd(C_REG(FNEG, 6, 6));                /* FP6 = sin(theta/2)  */
    /* psi products into FP7 (rho dead) and FP0 (t dead), keeping ct/st */
    sfp004_cmd(C_REG(FMOVE, 1, 7));
    sfp004_cmd(C_REG(FSGLMUL, 5, 7));                 /* ct*ch               */
    sfp004_cmd(C_REG(FMOVE, 3, 0));
    sfp004_cmd(C_REG(FSGLMUL, 6, 0));                 /* st*sh               */
    sfp004_cmd(C_REG(FSUB, 0, 7));                    /* FP7 = cos(1.5 t)    */
    sfp004_cmd(C_REG(FMOVE, 3, 0));
    sfp004_cmd(C_REG(FSGLMUL, 5, 0));                 /* st*ch (ch dead)     */
    sfp004_cmd(C_REG(FSGLMUL, 1, 6));                 /* FP6 = ct*sh         */
    sfp004_cmd(C_REG(FADD, 6, 0));                    /* FP0 = sin(1.5 t)    */
    sfp004_cmd_in(C_MEM(FSGLMUL, 7), fu(KNOT_W));     /* FP7 = ux            */
    sfp004_cmd_in(C_MEM(FSGLMUL, 0), fu(KNOT_W));     /* FP0 = uy            */
    /* nearer strand via the two squared distances (FP5 scratch twice) */
    sfp004_cmd(C_REG(FMOVE, 4, 5));
    sfp004_cmd(C_REG(FSUB, 7, 5));
    sfp004_cmd(C_REG(FSGLMUL, 5, 5));
    sfp004_cmd(C_REG(FMOVE, 2, 6));
    sfp004_cmd(C_REG(FSUB, 0, 6));
    sfp004_cmd(C_REG(FSGLMUL, 6, 6));
    sfp004_cmd(C_REG(FADD, 6, 5));
    m0 = sfp004_cmd_out(C_OUT(5));                    /* d0^2                */
    sfp004_cmd(C_REG(FMOVE, 4, 5));
    sfp004_cmd(C_REG(FADD, 7, 5));
    sfp004_cmd(C_REG(FSGLMUL, 5, 5));
    sfp004_cmd(C_REG(FMOVE, 2, 6));
    sfp004_cmd(C_REG(FADD, 0, 6));
    sfp004_cmd(C_REG(FSGLMUL, 6, 6));
    sfp004_cmd(C_REG(FADD, 6, 5));
    m1 = sfp004_cmd_out(C_OUT(5));                    /* d1^2                */
    near0 = (long)m0 <= (long)m1;
    /* cylindrical normal components: a = s -/+ ux (FP5), b = y -/+ uy (FP6) */
    {
        unsigned short strand_op = near0 ? SFP004_FOP_FSUB : SFP004_FOP_FADD;
        sfp004_cmd(C_REG(FMOVE, 4, 5));
        sfp004_cmd(SFP004_C_REG(strand_op, 7, 5));
        sfp004_cmd(C_REG(FMOVE, 2, 6));
        sfp004_cmd(SFP004_C_REG(strand_op, 0, 6));
    }
    /* normalise: |n|^2 = a^2 + b^2 (ux/uy dead: FP7/FP0 reused) */
    sfp004_cmd(C_REG(FMOVE, 5, 7));
    sfp004_cmd(C_REG(FSGLMUL, 7, 7));
    sfp004_cmd(C_REG(FMOVE, 6, 0));
    sfp004_cmd(C_REG(FSGLMUL, 0, 0));
    sfp004_cmd(C_REG(FADD, 0, 7));
    sfp004_cmd(C_REG(FSQRT, 7, 7));
    sfp004_cmd_in(C_MEM(FMOVE, 0), fu(1.0f));
    sfp004_cmd(C_REG(FSGLDIV, 7, 0));                 /* FP0 = 1/|n|         */
    sfp004_cmd(C_REG(FSGLMUL, 0, 5));                 /* FP5 = a'            */
    sfp004_cmd(C_REG(FSGLMUL, 0, 6));                 /* FP6 = b'            */
    /* diffuse = a'*(ct*Lx + st*Lz) + b'*Ly */
    sfp004_cmd(C_REG(FMOVE, 1, 7));
    sfp004_cmd_in(C_MEM(FSGLMUL, 7), fu(LIGHT.x));
    sfp004_cmd(C_REG(FMOVE, 3, 0));
    sfp004_cmd_in(C_MEM(FSGLMUL, 0), fu(LIGHT.z));
    sfp004_cmd(C_REG(FADD, 0, 7));
    sfp004_cmd(C_REG(FSGLMUL, 5, 7));
    sfp004_cmd(C_REG(FMOVE, 6, 0));
    sfp004_cmd_in(C_MEM(FSGLMUL, 0), fu(LIGHT.y));
    sfp004_cmd(C_REG(FADD, 0, 7));
    diff = sfp004_cmd_out(C_OUT(7));                  /* host clamp below    */
    if ((long)diff < 0) diff = fu(0.0f);
    /* h = LIGHT - rd (FP0/FP2/FP4: t, p.y, s all dead) */
    sfp004_cmd_in(C_MEM(FMOVE, 0), fu(LIGHT.x));
    sfp004_cmd_in(C_MEM(FSUB, 0), fu(rd->x));         /* hx                  */
    sfp004_cmd_in(C_MEM(FMOVE, 2), fu(LIGHT.y));
    sfp004_cmd_in(C_MEM(FSUB, 2), fu(rd->y));         /* hy                  */
    sfp004_cmd_in(C_MEM(FMOVE, 4), fu(LIGHT.z));
    sfp004_cmd_in(C_MEM(FSUB, 4), fu(rd->z));         /* hz                  */
    hy = sfp004_cmd_out(C_OUT(2));
    hz = sfp004_cmd_out(C_OUT(4));
    /* |h|^2 in FP7 (free after diffuse read-out), squaring hy/hz in place */
    sfp004_cmd(C_REG(FMOVE, 0, 7));
    sfp004_cmd(C_REG(FSGLMUL, 7, 7));
    sfp004_cmd(C_REG(FSGLMUL, 2, 2));
    sfp004_cmd(C_REG(FADD, 2, 7));
    sfp004_cmd(C_REG(FSGLMUL, 4, 4));
    sfp004_cmd(C_REG(FADD, 4, 7));
    sfp004_cmd(C_REG(FSQRT, 7, 7));                   /* FP7 = |h|           */
    /* n.h = a'*(ct*hx + st*hz) + b'*hy, then /|h| */
    sfp004_cmd(C_REG(FSGLMUL, 1, 0));                 /* ct*hx (hx dead)     */
    sfp004_cmd_in(C_MEM(FMOVE, 2), hz);
    sfp004_cmd(C_REG(FSGLMUL, 3, 2));                 /* st*hz               */
    sfp004_cmd(C_REG(FADD, 2, 0));
    sfp004_cmd(C_REG(FSGLMUL, 5, 0));
    sfp004_cmd_in(C_MEM(FMOVE, 2), hy);
    sfp004_cmd(C_REG(FSGLMUL, 6, 2));                 /* b'*hy               */
    sfp004_cmd(C_REG(FADD, 2, 0));
    sfp004_cmd(C_REG(FSGLDIV, 7, 0));                 /* FP0 = n.h_hat       */
    nh = sfp004_cmd_out(C_OUT(0));
    if ((long)nh < 0)                                 /* clamp before ^8     */
        sfp004_cmd_in(C_MEM(FMOVE, 0), fu(0.0f));
    sfp004_cmd(C_REG(FSGLMUL, 0, 0));                 /* spec^8              */
    sfp004_cmd(C_REG(FSGLMUL, 0, 0));
    sfp004_cmd(C_REG(FSGLMUL, 0, 0));
    /* intensity*14 -> palette index via FINTRZ + FMOVE.L */
    sfp004_cmd_in(C_MEM(FSGLMUL, 0), fu(0.3f));
    sfp004_cmd_in(C_MEM(FMOVE, 2), diff);
    sfp004_cmd_in(C_MEM(FSGLMUL, 2), fu(0.7f));
    sfp004_cmd(C_REG(FADD, 2, 0));
    sfp004_cmd_in(C_MEM(FSGLMUL, 0), fu(14.0f));
    sfp004_cmd(C_REG(FINTRZ, 0, 0));
    idx = (long)sfp004_cmd_out(C_OUTL(0)) + 1;
    if (idx < 1)  idx = 1;
    if (idx > 15) idx = 15;
    return (int)idx;
}

static int shade(const Vec3 *ro, const Vec3 *rd, float hit_t)
{
    if (use_fpu)
        return shade_fpu(ro, rd, hit_t);
    {
        Vec3 hit = {
            fp_add(ro->x, fp_mul(rd->x, hit_t)),
            fp_add(ro->y, fp_mul(rd->y, hit_t)),
            fp_add(ro->z, fp_mul(rd->z, hit_t))
        };
        return shade_soft(hit, *rd);
    }
}

/* ---------- chunky->planar block writer ---------- */
/* BLOCK_W is 2 or 4 and px = cx*BLOCK_W, so a block never straddles a word
 * and one precomputed mask serves all four planes. Only set bits are written:
 * the buffer was cleared, so colour 0 never calls this. */
static void put_block(unsigned char *scr, int cx, int cy, int c) {
    int px = cx * BLOCK_W;
    unsigned short mask =
        (unsigned short)(((1 << BLOCK_W) - 1) << (16 - BLOCK_W - (px & 15)));
    unsigned short *w =
        (unsigned short *)(scr + cy * BLOCK_H * SCANLINE_BYTES) + (px >> 4) * PLANES;
    int by;

    for (by = 0; by < BLOCK_H; by++) {
        if (c & 1) w[0] |= mask;
        if (c & 2) w[1] |= mask;
        if (c & 4) w[2] |= mask;
        if (c & 8) w[3] |= mask;
        w += SCANLINE_BYTES / 2;
    }
}

/* ---------- clear screen buffer ---------- */
static void clear_screen(unsigned char *scr) {
    memset(scr, 0, SCREEN_BYTES);
}

/* ---------- ms/frame readout (can't max out what you can't measure) ---------- */
/* 3x5 digit font, one 3-bit row per byte, drawn as logical pixels with
 * put_block in the top-left corner after each frame. Timing comes from the
 * 200 Hz system tick (_hz_200 at $4BA, supervisor-readable). */
#define HZ200 ((volatile unsigned long *)0x4BAL)

static const unsigned char digfont[10][5] = {
    { 7, 5, 5, 5, 7 },  /* 0 */
    { 2, 6, 2, 2, 7 },  /* 1 */
    { 7, 1, 7, 4, 7 },  /* 2 */
    { 7, 1, 7, 1, 7 },  /* 3 */
    { 5, 5, 7, 1, 1 },  /* 4 */
    { 7, 4, 7, 1, 7 },  /* 5 */
    { 7, 4, 7, 5, 7 },  /* 6 */
    { 7, 1, 2, 2, 2 },  /* 7 */
    { 7, 5, 7, 5, 7 },  /* 8 */
    { 7, 5, 7, 1, 7 },  /* 9 */
};

/* single physical pixel, colour 15 (all planes set) */
static void put_pixel(unsigned char *scr, int x, int y) {
    unsigned short mask = (unsigned short)(0x8000U >> (x & 15));
    unsigned short *w = (unsigned short *)(scr + y * SCANLINE_BYTES) + (x >> 4) * PLANES;
    w[0] |= mask; w[1] |= mask; w[2] |= mask; w[3] |= mask;
}

static void draw_ms(unsigned char *scr, unsigned long ms) {
    int col = 2 + 5 * 4;   /* 5 digits, right-aligned, drawn low digit first */
    int digits = 0;
    if (ms > 99999UL) ms = 99999UL;
    do {
        const unsigned char *glyph = digfont[ms % 10];
        int gx, gy;
        col -= 4;
        for (gy = 0; gy < 5; gy++)
            for (gx = 0; gx < 3; gx++)
                if (glyph[gy] & (4 >> gx))
                    put_pixel(scr, col + gx, 2 + gy);
        ms /= 10;
        digits++;
    } while (ms && digits < 5);
}

/* ---------- VBL wait ---------- */
static void wait_vbl(void) {
    volatile long *vbl_count = (volatile long *)0x462L;
    long t = *vbl_count;
    while (*vbl_count == t) {}
}

/* ---------- set hardware screen address ---------- */
static void set_screen(void *addr) {
    unsigned long a = (unsigned long)addr;
    /* write to video base registers */
    *((volatile unsigned char *)0xFF8201L) = (unsigned char)(a >> 16);
    *((volatile unsigned char *)0xFF8203L) = (unsigned char)(a >> 8);
    *((volatile unsigned char *)0xFF820DL) = (unsigned char)(a);
}

/* ---------- per-pixel ray fan, precomputed once ---------- */
static float fx_tab[RENDER_W];
static float fy_tab[RENDER_H];

/* ---------- temporal state: warm-start distances and colour cache ---------- */
static float warm_t[RENDER_H][RENDER_W];          /* last hit t (0 = miss)  */
static unsigned char colour_cache[RENDER_H][RENDER_W]; /* for checkerboard  */

static void reset_warm(void) {
    memset(warm_t, 0, sizeof warm_t);
    memset(colour_cache, 0, sizeof colour_cache);
}

static void init_ray_tables(void) {
    float aspect = fp_div((float)RENDER_W, (float)RENDER_H);
    int i;
    for (i = 0; i < RENDER_W; i++)
        fx_tab[i] = fp_mul(fp_div((float)(i - RENDER_W / 2), (float)RENDER_W), aspect);
    for (i = 0; i < RENDER_H; i++)
        fy_tab[i] = fp_div((float)(RENDER_H / 2 - i), (float)RENDER_H);
}

/* ---------- render one frame ---------- */
static void render_frame(unsigned char *scr, int angle_y, int angle_x, int parity) {
    /* Rotate the camera basis once per frame; every ray is a linear
     * combination of it, replacing the old per-pixel vector rotations. */
    Vec3 right = { 1.0f, 0.0f, 0.0f };
    Vec3 up    = { 0.0f, 1.0f, 0.0f };
    Vec3 fwd   = { 0.0f, 0.0f, -1.0f };
    Vec3 ro;
    int cx, cy;

    rot_y(&right, angle_y); rot_x(&right, angle_x);
    rot_y(&up,    angle_y); rot_x(&up,    angle_x);
    rot_y(&fwd,   angle_y); rot_x(&fwd,   angle_x);
    ro.x = fp_mul(-CAM_DIST, fwd.x);   /* camera at (0,0,CAM_DIST), rotated */
    ro.y = fp_mul(-CAM_DIST, fwd.y);
    ro.z = fp_mul(-CAM_DIST, fwd.z);

    clear_screen(scr);

    for (cy = 0; cy < RENDER_H; cy++) {
        float fy = fy_tab[cy];
        Vec3 row = {
            fp_add(fwd.x, fp_mul(fy, up.x)),
            fp_add(fwd.y, fp_mul(fy, up.y)),
            fp_add(fwd.z, fp_mul(fy, up.z))
        };
        for (cx = 0; cx < RENDER_W; cx++) {
            Vec3 rd;
            float t0, t1, hit_t;
            float tw;
            int ok;
#if CHECKERBOARD
            if (((cx + cy + parity) & 1)) {       /* held pixel: replay     */
                int cc = colour_cache[cy][cx];
                if (cc)
                    put_block(scr, cx, cy, cc);
                continue;
            }
#else
            (void)parity;
#endif
            ok = use_fpu
                ? ray_setup_fpu(&right, &row, fx_tab[cx], &rd, &t0, &t1)
                : ray_setup_soft(&right, &row, fx_tab[cx], &rd, &t0, &t1);
            if (!ok) {
                colour_cache[cy][cx] = 0;
                continue;   /* missed the bounding sphere: stays background */
            }
            tw = warm_t[cy][cx];
            if (fbits(tw) > 0) {
                tw = fp_sub(tw, WARM_BACKOFF);
                if (fbits(tw) > fbits(t0))
                    t0 = tw;
            }
            if (!raymarch(&ro, &rd, t0, t1, &hit_t)) {
                warm_t[cy][cx] = 0.0f;
                colour_cache[cy][cx] = 0;
                continue;
            }
            warm_t[cy][cx] = hit_t;
            {
                int c = shade(&ro, &rd, hit_t);
                colour_cache[cy][cx] = (unsigned char)c;
                put_block(scr, cx, cy, c);
            }
        }
    }
}

/* ---------- main ---------- */
int main(void) {
    int i;

    /* align screen buffers to 256-byte boundary */
    screens[0] = (unsigned char *)(((unsigned long)screen_buf[0] + 255) & ~255UL);
    screens[1] = (unsigned char *)(((unsigned long)screen_buf[1] + 255) & ~255UL);

    /* init FPU (cookie check, user mode) */
    sfp004_init();
    use_fpu = sfp004_available();
    void *old_phys = Physbase();

    /* enter supervisor mode for FPU CIR access and hardware register writes */
    long ssp = Super(0L);
    megaste_enable_16mhz_cache();   /* no-op on anything but a Mega STE */
    sfp004_arm();
    init_ray_tables();
    reset_warm();

    /* remember TOS's screen state: without restoring it on exit the desktop
     * comes back blind behind our last frame */
    unsigned short old_pal[16];
    unsigned char old_res = *((volatile unsigned char *)0xFF8260L);
    for (i = 0; i < 16; i++)
        old_pal[i] = *((volatile unsigned short *)(0xFF8240L + i * 2));

    /* set palette */
    for (i = 0; i < 16; i++) {
        *((volatile unsigned short *)(0xFF8240L + i * 2)) = palette[i];
    }

    /* set ST low-res mode */
    *((volatile unsigned short *)0xFF8260L) = 0;

    /* pre-clear both buffers */
    clear_screen(screens[0]);
    clear_screen(screens[1]);

    int angle_y = 0;
    int angle_x = 0;
    int parity  = 0;

    /* main loop — ESC (scancode $01 in ikbd) exits */
    while (1) {
        /* render off-screen, then show the buffer we just drew */
        unsigned char *draw = screens[cur_buf];
        unsigned long tick = *HZ200;

        render_frame(draw, angle_y, angle_x, parity);
        parity ^= 1;
        draw_ms(draw, (*HZ200 - tick) * 5UL);

        /* base register latches at the NEXT VBL: point it at the finished
         * buffer BEFORE waiting, so when the wait returns the swap has
         * actually happened and the other buffer is safely off-screen */
        set_screen(draw);
        wait_vbl();
        cur_buf ^= 1;

        /* advance rotation: ~360°/512 steps per frame */
        angle_y = (angle_y + 2) & 511;
        angle_x = (angle_x + 1) & 511;

        /* ESC exits; poll through BIOS - TOS's keyboard ISR consumes the
         * ACIA byte, so reading $FFFC00/02 directly never sees it */
        if (Bconstat(2)) {
            if ((Bconin(2) & 0xFF) == 27) break;
        }
    }

    /* hand the display back to TOS */
    for (i = 0; i < 16; i++)
        *((volatile unsigned short *)(0xFF8240L + i * 2)) = old_pal[i];
    *((volatile unsigned char *)0xFF8260L) = old_res;
    set_screen(old_phys);

    Super((void *)ssp);
    return 0;
}
