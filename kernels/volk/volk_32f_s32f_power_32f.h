/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
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

/*!
 * \page volk_32f_s32f_power_32f
 *
 * \b Overview
 *
 * Takes each input vector value to the specified power and stores the
 * results in the return vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_power_32f(float* cVector, const float* aVector, const float power, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li power: The power to raise the input value to.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * Square the numbers (0,9)
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *   }
 *
 *   // Normalize by the smallest delta (0.2 in this example)
 *   float scale = 2.0f;
 *
 *   volk_32f_s32f_power_32f(out, increasing, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_power_32f_a_H
#define INCLUDED_volk_32f_s32f_power_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <math.h>


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_s32f_power_32f_neon(float* cVector, const float* aVector,
                             const float power, unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number;
    const unsigned int quarter_points = num_points / 4;
    
    const float32x4_t power_vec = vdupq_n_f32(power);
    
    for(number = 0; number < quarter_points; number++) {
        // load f32
        const float32x4_t a_vec = vld1q_f32(aPtr);
        // Prefetch next 4
        __VOLK_PREFETCH(aPtr+4);
        const float32x4_t c_vec = _vpowq_f32(a_vec, power_vec);
        vst1q_f32(cPtr, c_vec);
        // move pointers ahead
        cPtr+=4;
        aPtr+=4;
    }
    
    // deal with the rest
    for(number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = powf(*aPtr++, power);
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_SSE4_1
#include <tmmintrin.h>

#ifdef LV_HAVE_LIB_SIMDMATH
#include <simdmath.h>
#endif /* LV_HAVE_LIB_SIMDMATH */

static inline void
volk_32f_s32f_power_32f_a_sse4_1(float* cVector, const float* aVector,
                                 const float power, unsigned int num_points)
{
  unsigned int number = 0;

  float* cPtr = cVector;
  const float* aPtr = aVector;

#ifdef LV_HAVE_LIB_SIMDMATH
  const unsigned int quarterPoints = num_points / 4;
  __m128 vPower = _mm_set_ps1(power);
  __m128 zeroValue = _mm_setzero_ps();
  __m128 signMask;
  __m128 negatedValues;
  __m128 negativeOneToPower = _mm_set_ps1(powf(-1, power));
  __m128 onesMask = _mm_set_ps1(1);

  __m128 aVal, cVal;
  for(;number < quarterPoints; number++){

    aVal = _mm_load_ps(aPtr);
    signMask = _mm_cmplt_ps(aVal, zeroValue);
    negatedValues = _mm_sub_ps(zeroValue, aVal);
    aVal = _mm_blendv_ps(aVal, negatedValues, signMask);

    // powf4 doesn't support negative values in the base, so we mask them off and then apply the negative after
    cVal = powf4(aVal, vPower); // Takes each input value to the specified power

    cVal = _mm_mul_ps( _mm_blendv_ps(onesMask, negativeOneToPower, signMask), cVal);

    _mm_store_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
#endif /* LV_HAVE_LIB_SIMDMATH */

  for(;number < num_points; number++){
    *cPtr++ = powf((*aPtr++), power);
  }
}

#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

#ifdef LV_HAVE_LIB_SIMDMATH
#include <simdmath.h>
#endif /* LV_HAVE_LIB_SIMDMATH */

static inline void
volk_32f_s32f_power_32f_a_sse(float* cVector, const float* aVector,
                              const float power, unsigned int num_points)
{
  unsigned int number = 0;

  float* cPtr = cVector;
  const float* aPtr = aVector;

#ifdef LV_HAVE_LIB_SIMDMATH
  const unsigned int quarterPoints = num_points / 4;
  __m128 vPower = _mm_set_ps1(power);
  __m128 zeroValue = _mm_setzero_ps();
  __m128 signMask;
  __m128 negatedValues;
  __m128 negativeOneToPower = _mm_set_ps1(powf(-1, power));
  __m128 onesMask = _mm_set_ps1(1);

  __m128 aVal, cVal;
  for(;number < quarterPoints; number++){

    aVal = _mm_load_ps(aPtr);
    signMask = _mm_cmplt_ps(aVal, zeroValue);
    negatedValues = _mm_sub_ps(zeroValue, aVal);
    aVal = _mm_or_ps(_mm_andnot_ps(signMask, aVal), _mm_and_ps(signMask, negatedValues) );

    // powf4 doesn't support negative values in the base, so we mask them off and then apply the negative after
    cVal = powf4(aVal, vPower); // Takes each input value to the specified power

    cVal = _mm_mul_ps( _mm_or_ps( _mm_andnot_ps(signMask, onesMask), _mm_and_ps(signMask, negativeOneToPower) ), cVal);

    _mm_store_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
#endif /* LV_HAVE_LIB_SIMDMATH */

  for(;number < num_points; number++){
    *cPtr++ = powf((*aPtr++), power);
  }
}

#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_s32f_power_32f_generic(float* cVector, const float* aVector,
                                const float power, unsigned int num_points)
{
  float* cPtr = cVector;
  const float* aPtr = aVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = powf((*aPtr++), power);
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_s32f_power_32f_a_H */
