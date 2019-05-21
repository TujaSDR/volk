/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

/* Copyright (C) 2011  Julien Pommier
   Copyright (C) 2019  Albin Stigo
 
 This software is provided 'as-is', without any express or implied
 warranty.  In no event will the authors be held liable for any damages
 arising from the use of this software.
 
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it
 freely, subject to the following restrictions:
 
 1. The origin of this software must not be misrepresented; you must not
 claim that you wrote the original software. If you use this software
 in a product, an acknowledgment in the product documentation would be
 appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be
 misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 
 (this is the zlib license)
 */

/*
 * This file is intended to hold NEON intrinsics of intrinsics.
 * They should be used in VOLK kernels to avoid copy-pasta.
 */

#ifndef INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_

#include <arm_neon.h>

/*
static inline float32x4_t _vinvsqrtq_neonv8_f32(float32x4_t x) {
    // vsqrtq_f32 is new in armv8
    float32x4_t sqrt_reciprocal = _vinvq_f32(vsqrtq_f32(x));
    return sqrt_reciprocal;
}
*/

static inline float32x4_t _vinvq_f32(float32x4_t x)
{
    // Newton's method
    float32x4_t recip = vrecpeq_f32(x);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    return recip;
}

static inline float32x4_t
_vmagnitudesquaredq_f32(float32x4x2_t cmplxValue)
{
    float32x4_t iValue, qValue, result;
    iValue = vmulq_f32(cmplxValue.val[0], cmplxValue.val[0]); // Square the values
    qValue = vmulq_f32(cmplxValue.val[1], cmplxValue.val[1]); // Square the values
    result = vaddq_f32(iValue, qValue); // Add the I2 and Q2 values
    return result;
}

static inline float32x4_t _vinvsqrtq_f32(float32x4_t x)
{
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    
    return sqrt_reciprocal;
}

/* armv8 has vsqrtq_f32
 WARNING: does not handle 0 well, needs work */
static inline float32x4_t _vsqrtq_f32(float32x4_t x)
{
    return _vinvq_f32(_vinvsqrtq_f32(x));
}

/* Adapted from ARM Compute Library MIT license */
static inline float32x4_t _vtaylor_polyq_f32(float32x4_t x, const float32x4_t coeffs[8])
{
    float32x4_t cA   = vmlaq_f32(coeffs[0], coeffs[4], x);
    float32x4_t cB   = vmlaq_f32(coeffs[2], coeffs[6], x);
    float32x4_t cC   = vmlaq_f32(coeffs[1], coeffs[5], x);
    float32x4_t cD   = vmlaq_f32(coeffs[3], coeffs[7], x);
    float32x4_t x2  = vmulq_f32(x, x);
    float32x4_t x4  = vmulq_f32(x2, x2);
    float32x4_t res = vmlaq_f32(vmlaq_f32(cA, cB, x2), vmlaq_f32(cC, cD, x2), x4);
    return res;
}

/* Adapted from ARM Compute Library MIT license */
static inline float32x4_t _vexpq_f32(float32x4_t x)
{
    const float32x4_t exp_tab[8] = {
        vdupq_n_f32(1.f),
        vdupq_n_f32(0.0416598916054f),
        vdupq_n_f32(0.500000596046f),
        vdupq_n_f32(0.0014122662833f),
        vdupq_n_f32(1.00000011921f),
        vdupq_n_f32(0.00833693705499f),
        vdupq_n_f32(0.166665703058f),
        vdupq_n_f32(0.000195780929062f),
    };
    
    const float32x4_t CONST_LN2          = vdupq_n_f32(0.6931471805f); // ln(2)
    const float32x4_t CONST_INV_LN2      = vdupq_n_f32(1.4426950408f); // 1/ln(2)
    const float32x4_t CONST_0            = vdupq_n_f32(0.f);
    const int32x4_t   CONST_NEGATIVE_126 = vdupq_n_s32(-126);
    
    // Perform range reduction [-ln(2),ln(2)]
    int32x4_t   m   = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
    float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);
    
    // Polynomial Approximation
    float32x4_t poly = _vtaylor_polyq_f32(val, exp_tab);
    
    // Reconstruct
    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_126), CONST_0, poly);
    
    return poly;
}

// floor a float32x4_t
static inline float32x4_t vfloorq_f32(float32x4_t val)
{
    const float32x4_t CONST_1 = vdupq_n_f32(1.f);
    const int32x4_t   z = vcvtq_s32_f32(val);
    const float32x4_t r = vcvtq_f32_s32(z);
    return vbslq_f32(vcgtq_f32(r, val), vsubq_f32(r, CONST_1), r);
}

// https://stackoverflow.com/questions/46974513/code-for-acos-with-avx256
static inline float32x4_t _vacos2q_f32(float32x4_t x) {
    const float32x4_t CONST_0    = vdupq_n_f32(0.f);
    const float32x4_t CONST_1    = vdupq_n_f32(1.f);
    const float32x4_t CONST_PI   = vdupq_n_f32(M_PI);
    const float32x4_t abs_x      = vabsq_f32(x);
    
    const uint32x4_t sign = vcgtq_f32(x, CONST_0);
    
    float32x4_t t = vsqrtq_f32(CONST_1 - abs_x);
    
    // Polynomial correction factor based on xp
    // TODO: try to find better coefficients
    // This has to be the best way to evaluate polynomials on NEON
    const float32x4_t c3 = vdupq_n_f32(-0.02007522f);
    const float32x4_t c2 = vmlaq_f32(vdupq_n_f32(0.07590315f), abs_x, c3);
    const float32x4_t c1 = vmlaq_f32(vdupq_n_f32(-0.2126757f), abs_x, c2);
    const float32x4_t c0 = vmlaq_f32(vdupq_n_f32(1.5707963267948966f), abs_x, c1);
    
    // Positive
    float32x4_t p = t * c0;
    // Correct for negative
    float32x4_t n = CONST_PI - p;
    
    return vbslq_f32(sign, p, n);
}

/* arcsin using arccos */
static inline float32x4_t _vasinq_f32(float32x4_t x) {
    const float32x4_t CONST_PI_2 = vdupq_n_f32(M_PI_2);
    // asin(x) = acos(-x) - pi/2
    return _vacos2q_f32(vnegq_f32(x)) - CONST_PI_2;
}

/* Yet another method for arctan via arcsin */
static inline float32x4_t _vatanfast2q_f32(float32x4_t x) {
    const float32x4_t CONST_1 = vdupq_n_f32(1.f);
    x = vmulq_f32(x, _vinvsqrtq_f32((CONST_1 + x * x)));
    return _vasinq_f32(x);
}

/* Efficient Approximations for the Arctangent Function [dsp TIPS&TRICKS] */
static inline float32x4_t _vatanfastq_f32(float32x4_t x) {
    const float32x4_t CONST_1 = vdupq_n_f32(1.f);
    const float32x4_t CONST_0273 = vdupq_n_f32(0.273);
    const float32x4_t CONST_PI_4 = vdupq_n_f32(0.78539816339f);
    const float32x4_t y = CONST_PI_4 * x + CONST_0273 * x * (CONST_1 - vabsq_f32(x));
    return y;
}

/* atan */
static inline float32x4_t _vatanq_f32(float32x4_t x) {
    const float32x4_t CONST_0          = vdupq_n_f32(0.f);
    const float32x4_t CONST_1 = vdupq_n_f32(1.f);
    const float32x4_t CONST_TAN_3_PI_8 = vdupq_n_f32(2.41421356237f);
    const float32x4_t CONST_TAN_PI_8   = vdupq_n_f32(0.41421356237f);
    const float32x4_t CONST_PI_2       = vdupq_n_f32(M_PI_2);
    const float32x4_t CONST_PI_4       = vdupq_n_f32(M_PI/4);

    float32x4_t y = vdupq_n_f32(0.f);
    float32x4_t x1, x2;
    uint32x4_t gt_mask;
    
    // x < 0
    const uint32x4_t sign_mask = vcltq_f32(x, CONST_0);
    x = vabsq_f32(x);

    x1  = vnegq_f32(_vinvq_f32(x));
    x2 = vmulq_f32(vsubq_f32(x, CONST_1),
                   _vinvq_f32(vaddq_f32(x, CONST_1)));
    
    // x > 0.41421356237
    gt_mask = vcgtq_f32(x, CONST_TAN_PI_8);
    y = vbslq_f32(gt_mask, CONST_PI_4, y);
    x = vbslq_f32(gt_mask, x2, x);
    
    // x > 2.41421356237
    gt_mask = vcgtq_f32(x, CONST_TAN_3_PI_8);
    y = vbslq_f32(gt_mask, CONST_PI_2, y);
    x = vbslq_f32(gt_mask, x1, x);
    
    float32x4_t z = x * x;
    
    const float32x4_t c3 = vdupq_n_f32(8.05374449538e-2);
    const float32x4_t c2 = vdupq_n_f32(1.38776856032e-1);
    const float32x4_t c1 = vdupq_n_f32(1.99777106478e-1);
    const float32x4_t c0 = vdupq_n_f32(3.33329491539e-1);

    // Evaluate polynom
    // Change this to vmlaq
    y += ((( c3 * z - c2) * z + c1) * z - c0) * z * x + x;
    
    y = vbslq_f32(sign_mask, vnegq_f32(y), y);

    return y;
}

static inline float32x4_t _vatan2q_f32(float32x4_t y, float32x4_t x) {
    // Adjusts atan to the right quadrant
    const float32x4_t CONST_0    = vdupq_n_f32(0.f);
    const float32x4_t CONST_PI   = vdupq_n_f32(M_PI);
    const float32x4_t CONST_PI_2 = vdupq_n_f32(M_PI_2);
    const float32x4_t CONST_NEG_PI_2 = vdupq_n_f32(-M_PI_2);

    //y = vmulq_n_f32(y, 0.5);
    //x = vmulq_n_f32(x, 0.5);
    
    const float32x4_t atan_y_x = _vatanq_f32(vmulq_f32(y, _vinvq_f32(x)));
    
    const float32x4_t atan_y_x_p_pi = atan_y_x + CONST_PI;
    const float32x4_t atan_y_x_m_pi = atan_y_x - CONST_PI;

    uint32x4_t x_gt_0 = vcgtq_f32(x, CONST_0);
    uint32x4_t x_lt_0 = vcltq_f32(x, CONST_0);
    uint32x4_t y_ge_0 = vcgeq_f32(y, CONST_0);
    
    // Select the right value based on quadrant
    // This is a bit tricky but look at the definition of atan2
    float32x4_t res = vbslq_f32(x_gt_0, atan_y_x,
                                vbslq_f32(x_lt_0,
                                          vbslq_f32(y_ge_0, atan_y_x_p_pi, atan_y_x_m_pi),
                                          vbslq_f32(y_ge_0, CONST_PI_2, CONST_NEG_PI_2)));
    
    return res;
}

/* evaluation of 4 sines & cosines at once.
 * Optimized from here:
 * http://gruntthepeon.free.fr/ssemath/
 *
 */
static inline void _vsincosq_f32(float32x4_t x,
                                 float32x4_t *ysin,
                                 float32x4_t *ycos) { // any x
    //float32x4_t xmm1, xmm2, xmm3, y;
    float32x4_t y;
    
    const float32x4_t c_minus_cephes_DP1 = vdupq_n_f32(-0.78515625);
    const float32x4_t c_minus_cephes_DP2 = vdupq_n_f32(-2.4187564849853515625e-4);
    const float32x4_t c_minus_cephes_DP3 = vdupq_n_f32(-3.77489497744594108e-8);
    const float32x4_t c_sincof_p0 = vdupq_n_f32(-1.9515295891e-4);
    const float32x4_t c_sincof_p1  = vdupq_n_f32(8.3321608736e-3);
    const float32x4_t c_sincof_p2 = vdupq_n_f32(-1.6666654611e-1);
    const float32x4_t c_coscof_p0 = vdupq_n_f32(2.443315711809948e-005);
    const float32x4_t c_coscof_p1 = vdupq_n_f32(-1.388731625493765e-003);
    const float32x4_t c_coscof_p2 = vdupq_n_f32(4.166664568298827e-002);
    const float32x4_t c_cephes_FOPI = vdupq_n_f32(1.27323954473516); // 4 / M_PI

    const float32x4_t CONST_1 = vdupq_n_f32(1.f);
    const float32x4_t CONST_1_2 = vdupq_n_f32(0.5f);
    const float32x4_t CONST_0 = vdupq_n_f32(0.f);
    const uint32x4_t  CONST_2 = vdupq_n_u32(2);
    const uint32x4_t  CONST_4 = vdupq_n_u32(4);
    
    uint32x4_t emm2;
    
    uint32x4_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcltq_f32(x, CONST_0);
    x = vabsq_f32(x);
    // scale by 4/pi
    y = vmulq_f32(x, c_cephes_FOPI);
    
    // store the integer part of y in mm0
    emm2 = vcvtq_u32_f32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);
    
    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2
     Both branches will be computed.
     */
    uint32x4_t poly_mask = vtstq_u32(emm2, CONST_2);
    
    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    /*xmm1 = vmulq_f32(y, c_minus_cephes_DP1);
    xmm2 = vmulq_f32(y, c_minus_cephes_DP2);
    xmm3 = vmulq_f32(y, c_minus_cephes_DP3);
    x = vaddq_f32(x, xmm1);
    x = vaddq_f32(x, xmm2);
    x = vaddq_f32(x, xmm3);*/
    
    // The magic pass: "Extended precision modular arithmetic"
    x = vmlaq_f32(x, y, c_minus_cephes_DP1);
    x = vmlaq_f32(x, y, c_minus_cephes_DP2);
    x = vmlaq_f32(x, y, c_minus_cephes_DP3);
    
    sign_mask_sin = veorq_u32(sign_mask_sin, vtstq_u32(emm2, CONST_4));
    sign_mask_cos = vtstq_u32(vsubq_u32(emm2, CONST_2), CONST_4);
    
    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     and the second polynom      (Pi/4 <= x <= 0) in y2 */

    float32x4_t y1, y2;
    float32x4_t z = vmulq_f32(x,x);

    y1 = vmlaq_f32(c_coscof_p1, z, c_coscof_p0);
    y1 = vmlaq_f32(c_coscof_p2, z, y1);
    y1 = vmulq_f32(y1, z);
    y1 = vmulq_f32(y1, z);
    y1 = vmlsq_f32(y1, z, CONST_1_2);
    y1 = vaddq_f32(y1, CONST_1);

    y2 = vmlaq_f32(c_sincof_p1, z, c_sincof_p0);
    y2 = vmlaq_f32(c_sincof_p2, z, y2);
    y2 = vmulq_f32(y2, z);
    y2 = vmlaq_f32(x, x, y2);
    
    /* select the correct result from the two polynoms */
    float32x4_t ys = vbslq_f32(poly_mask, y1, y2);
    float32x4_t yc = vbslq_f32(poly_mask, y2, y1);
    *ysin = vbslq_f32(sign_mask_sin, vnegq_f32(ys), ys);
    *ycos = vbslq_f32(sign_mask_cos, yc, vnegq_f32(yc));
}


/* Adapted from ARM Compute Library MIT license */
static inline float32x4_t _vtanhq_f32(float32x4_t val)
{
    const float32x4_t CONST_1        = vdupq_n_f32(1.f);
    const float32x4_t CONST_2        = vdupq_n_f32(2.f);
    const float32x4_t CONST_MIN_TANH = vdupq_n_f32(-10.f);
    const float32x4_t CONST_MAX_TANH = vdupq_n_f32(10.f);
    
    float32x4_t x     = vminq_f32(vmaxq_f32(val, CONST_MIN_TANH), CONST_MAX_TANH);
    float32x4_t exp2x = _vexpq_f32(vmulq_f32(CONST_2, x));
    float32x4_t num   = vsubq_f32(exp2x, CONST_1);
    float32x4_t den   = vaddq_f32(exp2x, CONST_1);
    float32x4_t tanh  = vmulq_f32(num, _vinvq_f32(den));
    return tanh;
}

static inline float
_vsumq_f32(float32x4_t vec)
{
    // Sum lanes in vec and return value
    const float32x2_t tmp1 = vadd_f32(vget_high_f32(vec), vget_low_f32(vec));
    const float32x2_t tmp2 = vpadd_f32(tmp1, tmp1);
    return vget_lane_f32(tmp2, 0);
}

static inline float32x4_t
_vclampq_f32(float32x4_t vec, float32x4_t min_vec, float32x4_t max_vec)
{
    return vminq_f32(vmaxq_f32(vec, min_vec), max_vec);
}

static inline int32x4_t
_vclampq_s32(int32x4_t vec, int32x4_t min_vec, int32x4_t max_vec)
{
    return vminq_s32(vmaxq_s32(vec, min_vec), max_vec);
}

static inline float32x4x2_t
_vmultiply_complexq_f32(float32x4x2_t a_val, float32x4x2_t b_val)
{
    float32x4x2_t tmp_real;
    float32x4x2_t tmp_imag;
    float32x4x2_t c_val;
    
    // multiply the real*real and imag*imag to get real result
    // a0r*b0r|a1r*b1r|a2r*b2r|a3r*b3r
    tmp_real.val[0] = vmulq_f32(a_val.val[0], b_val.val[0]);
    // a0i*b0i|a1i*b1i|a2i*b2i|a3i*b3i
    tmp_real.val[1] = vmulq_f32(a_val.val[1], b_val.val[1]);
    // Multiply cross terms to get the imaginary result
    // a0r*b0i|a1r*b1i|a2r*b2i|a3r*b3i
    tmp_imag.val[0] = vmulq_f32(a_val.val[0], b_val.val[1]);
    // a0i*b0r|a1i*b1r|a2i*b2r|a3i*b3r
    tmp_imag.val[1] = vmulq_f32(a_val.val[1], b_val.val[0]);
    // combine the products
    c_val.val[0] = vsubq_f32(tmp_real.val[0], tmp_real.val[1]);
    c_val.val[1] = vaddq_f32(tmp_imag.val[0], tmp_imag.val[1]);
    return c_val;
}

/* Adapted from ARM Compute Library MIT license
 * Natural logarithm */
static inline float32x4_t _vlogq_f32(float32x4_t x)
{
    const float32x4_t log_tab[8] = {
        vdupq_n_f32(-2.29561495781f),
        vdupq_n_f32(-2.47071170807f),
        vdupq_n_f32(-5.68692588806f),
        vdupq_n_f32(-0.165253549814f),
        vdupq_n_f32(5.17591238022f),
        vdupq_n_f32(0.844007015228f),
        vdupq_n_f32(4.58445882797f),
        vdupq_n_f32(0.0141278216615f),
    };
    
    const int32x4_t   CONST_127 = vdupq_n_s32(127);           // 127
    const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f); // ln(2)
    
    // Extract exponent
    int32x4_t m = vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)), CONST_127);
    float32x4_t val = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));
    
    // Polynomial Approximation
    float32x4_t poly = _vtaylor_polyq_f32(val, log_tab);
    
    // Reconstruct
    poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);
    
    return poly;
}

static inline float32x4_t _vlog2q_f32(float32x4_t x) {
    const float32x4_t CONST_1_LN2 = vdupq_n_f32(1.44269504089); // 1/ln(2)
    return vmulq_f32(_vlogq_f32(x), CONST_1_LN2);
}

/* Adapted from ARM Compute Library MIT license */
static inline float32x4_t _vpowq_f32(float32x4_t val, float32x4_t n)
{
    return _vexpq_f32(vmulq_f32(n, _vlogq_f32(val)));
}

static inline float32x4_t _vsinq_f32(float32x4_t x) {
    float32x4_t ysin, ycos;
    _vsincosq_f32(x, &ysin, &ycos);
    return ysin;
}

static inline float32x4_t _vcosq_f32(float32x4_t x) {
    float32x4_t ysin, ycos;
    _vsincosq_f32(x, &ysin, &ycos);
    return ycos;
}

static inline float32x4_t _vtanq_f32(float32x4_t x) {
    float32x4_t ysin, ycos;
    _vsincosq_f32(x, &ysin, &ycos);
    return vmulq_f32(ysin, _vinvq_f32(ycos));
}

static inline float32x4_t _vtruncateq_f32(float32x4_t x) {
    // To int and back without rounding
    return vcvtq_f32_s32(vcvtq_s32_f32(x));
}

#endif /* INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_ */
