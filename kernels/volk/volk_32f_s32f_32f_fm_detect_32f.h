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
 * \page volk_32f_s32f_32f_fm_detect_32f
 *
 * \b Overview
 *
 * Performs FM-detect differentiation on the input vector and stores
 * the results in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_32f_fm_detect_32f(float* outputVector, const float* inputVector, const float bound, float* saveValue, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector containing phase data (must be on the interval (-bound, bound]).
 * \li bound: The interval that the input phase data is in, which is used to modulo the differentiation.
 * \li saveValue: A pointer to a float which contains the phase value of the sample before the first input sample.
 * \li num_points The number of data points.
 *
 * \b Outputs
 * \li outputVector: The vector where the results will be stored.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * <FIXME>
 *
 * volk_32f_s32f_32f_fm_detect_32f();
 *
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_32f_fm_detect_32f_a_H
#define INCLUDED_volk_32f_s32f_32f_fm_detect_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_a_avx(float* outputVector, const float* inputVector, const float bound, float* saveValue, unsigned int num_points){
  if (num_points < 1) {
    return;
  }
  unsigned int number = 1;
  unsigned int j = 0;
  // num_points-1 keeps Fedora 7's gcc from crashing...
  // num_points won't work.  :(
  const unsigned int eighthPoints = (num_points-1) / 8;

  float* outPtr = outputVector;
  const float* inPtr = inputVector;
  __m256 upperBound = _mm256_set1_ps(bound);
  __m256 lowerBound = _mm256_set1_ps(-bound);
  __m256 next3old1;
  __m256 next4;
  __m256 boundAdjust;
  __m256 posBoundAdjust = _mm256_set1_ps(-2*bound); // Subtract when we're above.
  __m256 negBoundAdjust = _mm256_set1_ps(2*bound); // Add when we're below.
  // Do the first 8 by hand since we're going in from the saveValue:
  *outPtr = *inPtr - *saveValue;
  if (*outPtr >  bound) *outPtr -= 2*bound;
  if (*outPtr < -bound) *outPtr += 2*bound;
  inPtr++;
  outPtr++;
  for (j = 1; j < ( (8 < num_points) ? 8 : num_points); j++) {
    *outPtr = *(inPtr) - *(inPtr-1);
    if (*outPtr >  bound) *outPtr -= 2*bound;
    if (*outPtr < -bound) *outPtr += 2*bound;
    inPtr++;
    outPtr++;
  }

  for (; number < eighthPoints; number++) {
    // Load data
    next3old1 = _mm256_loadu_ps((float*) (inPtr-1));
    next4 = _mm256_load_ps(inPtr);
    inPtr += 8;
    // Subtract and store:
    next3old1 = _mm256_sub_ps(next4, next3old1);
    // Bound:
    boundAdjust = _mm256_cmp_ps(next3old1, upperBound, 14);
    boundAdjust = _mm256_and_ps(boundAdjust, posBoundAdjust);
    next4 = _mm256_cmp_ps(next3old1, lowerBound, 1);
    next4 = _mm256_and_ps(next4, negBoundAdjust);
    boundAdjust = _mm256_or_ps(next4, boundAdjust);
    // Make sure we're in the bounding interval:
    next3old1 = _mm256_add_ps(next3old1, boundAdjust);
    _mm256_store_ps(outPtr,next3old1); // Store the results back into the output
    outPtr += 8;
  }

  for (number = (8 > (eighthPoints*8) ? 8 : (8 * eighthPoints)); number < num_points; number++) {
    *outPtr = *(inPtr) - *(inPtr-1);
    if (*outPtr >  bound) *outPtr -= 2*bound;
    if (*outPtr < -bound) *outPtr += 2*bound;
    inPtr++;
    outPtr++;
  }

  *saveValue = inputVector[num_points-1];
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_a_sse(float* outputVector, const float* inputVector, const float bound, float* saveValue, unsigned int num_points){
  if (num_points < 1) {
    return;
  }
  unsigned int number = 1;
  unsigned int j = 0;
  // num_points-1 keeps Fedora 7's gcc from crashing...
  // num_points won't work.  :(
  const unsigned int quarterPoints = (num_points-1) / 4;

  float* outPtr = outputVector;
  const float* inPtr = inputVector;
  __m128 upperBound = _mm_set_ps1(bound);
  __m128 lowerBound = _mm_set_ps1(-bound);
  __m128 next3old1;
  __m128 next4;
  __m128 boundAdjust;
  __m128 posBoundAdjust = _mm_set_ps1(-2*bound); // Subtract when we're above.
  __m128 negBoundAdjust = _mm_set_ps1(2*bound); // Add when we're below.
  // Do the first 4 by hand since we're going in from the saveValue:
  *outPtr = *inPtr - *saveValue;
  if (*outPtr >  bound) *outPtr -= 2*bound;
  if (*outPtr < -bound) *outPtr += 2*bound;
  inPtr++;
  outPtr++;
  for (j = 1; j < ( (4 < num_points) ? 4 : num_points); j++) {
    *outPtr = *(inPtr) - *(inPtr-1);
    if (*outPtr >  bound) *outPtr -= 2*bound;
    if (*outPtr < -bound) *outPtr += 2*bound;
    inPtr++;
    outPtr++;
  }

  for (; number < quarterPoints; number++) {
    // Load data
    next3old1 = _mm_loadu_ps((float*) (inPtr-1));
    next4 = _mm_load_ps(inPtr);
    inPtr += 4;
    // Subtract and store:
    next3old1 = _mm_sub_ps(next4, next3old1);
    // Bound:
    boundAdjust = _mm_cmpgt_ps(next3old1, upperBound);
    boundAdjust = _mm_and_ps(boundAdjust, posBoundAdjust);
    next4 = _mm_cmplt_ps(next3old1, lowerBound);
    next4 = _mm_and_ps(next4, negBoundAdjust);
    boundAdjust = _mm_or_ps(next4, boundAdjust);
    // Make sure we're in the bounding interval:
    next3old1 = _mm_add_ps(next3old1, boundAdjust);
    _mm_store_ps(outPtr,next3old1); // Store the results back into the output
    outPtr += 4;
  }

  for (number = (4 > (quarterPoints*4) ? 4 : (4 * quarterPoints)); number < num_points; number++) {
    *outPtr = *(inPtr) - *(inPtr-1);
    if (*outPtr >  bound) *outPtr -= 2*bound;
    if (*outPtr < -bound) *outPtr += 2*bound;
    inPtr++;
    outPtr++;
  }

  *saveValue = inputVector[num_points-1];
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_32f_fm_detect_32f_generic(float* outputVector, const float* inputVector, const float bound, float* saveValue, unsigned int num_points){
  if (num_points < 1) {
    return;
  }
  unsigned int number = 0;
  float* outPtr = outputVector;
  const float* inPtr = inputVector;

  // Do the first 1 by hand since we're going in from the saveValue:
  *outPtr = *inPtr - *saveValue;
  if (*outPtr >  bound) *outPtr -= 2*bound;
  if (*outPtr < -bound) *outPtr += 2*bound;
  inPtr++;
  outPtr++;

  for (number = 1; number < num_points; number++) {
    *outPtr = *(inPtr) - *(inPtr-1);
    if (*outPtr >  bound) *outPtr -= 2*bound;
    if (*outPtr < -bound) *outPtr += 2*bound;
    inPtr++;
    outPtr++;
  }

  *saveValue = inputVector[num_points-1];
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_32f_s32f_32f_fm_detect_32f_a_H */


#ifndef INCLUDED_volk_32f_s32f_32f_fm_detect_32f_u_H
#define INCLUDED_volk_32f_s32f_32f_fm_detect_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_u_avx(float* outputVector, const float* inputVector, const float bound, float* saveValue, unsigned int num_points){
  if (num_points < 1) {
    return;
  }
  unsigned int number = 1;
  unsigned int j = 0;
  // num_points-1 keeps Fedora 7's gcc from crashing...
  // num_points won't work.  :(
  const unsigned int eighthPoints = (num_points-1) / 8;

  float* outPtr = outputVector;
  const float* inPtr = inputVector;
  __m256 upperBound = _mm256_set1_ps(bound);
  __m256 lowerBound = _mm256_set1_ps(-bound);
  __m256 next3old1;
  __m256 next4;
  __m256 boundAdjust;
  __m256 posBoundAdjust = _mm256_set1_ps(-2*bound); // Subtract when we're above.
  __m256 negBoundAdjust = _mm256_set1_ps(2*bound); // Add when we're below.
  // Do the first 8 by hand since we're going in from the saveValue:
  *outPtr = *inPtr - *saveValue;
  if (*outPtr >  bound) *outPtr -= 2*bound;
  if (*outPtr < -bound) *outPtr += 2*bound;
  inPtr++;
  outPtr++;
  for (j = 1; j < ( (8 < num_points) ? 8 : num_points); j++) {
    *outPtr = *(inPtr) - *(inPtr-1);
    if (*outPtr >  bound) *outPtr -= 2*bound;
    if (*outPtr < -bound) *outPtr += 2*bound;
    inPtr++;
    outPtr++;
  }

  for (; number < eighthPoints; number++) {
    // Load data
    next3old1 = _mm256_loadu_ps((float*) (inPtr-1));
    next4 = _mm256_loadu_ps(inPtr);
    inPtr += 8;
    // Subtract and store:
    next3old1 = _mm256_sub_ps(next4, next3old1);
    // Bound:
    boundAdjust = _mm256_cmp_ps(next3old1, upperBound, 14);
    boundAdjust = _mm256_and_ps(boundAdjust, posBoundAdjust);
    next4 = _mm256_cmp_ps(next3old1, lowerBound, 1);
    next4 = _mm256_and_ps(next4, negBoundAdjust);
    boundAdjust = _mm256_or_ps(next4, boundAdjust);
    // Make sure we're in the bounding interval:
    next3old1 = _mm256_add_ps(next3old1, boundAdjust);
    _mm256_storeu_ps(outPtr,next3old1); // Store the results back into the output
    outPtr += 8;
  }

  for (number = (8 > (eighthPoints*8) ? 8 : (8 * eighthPoints)); number < num_points; number++) {
    *outPtr = *(inPtr) - *(inPtr-1);
    if (*outPtr >  bound) *outPtr -= 2*bound;
    if (*outPtr < -bound) *outPtr += 2*bound;
    inPtr++;
    outPtr++;
  }

  *saveValue = inputVector[num_points-1];
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_neon(float* outputVector, const float* inputVector, const float bound, float* saveValue, unsigned int num_points){
    if (num_points < 1) {
        return;
    }
    unsigned int number = 0;
    unsigned int quarter_points = (num_points-1) / 4;
    
    float* outPtr = outputVector;
    const float* inPtr = inputVector;
    
    const float32x4_t bound_vec = vdupq_n_f32(bound);
    const float32x4_t neg_bound_vec = vdupq_n_f32(-bound);
    const float32x4_t bound_adj_vec = vdupq_n_f32(2.*bound);
    
    // Do the first 4 by hand since we're going in from the saveValue:
    *outPtr = *inPtr - *saveValue;
    if (*outPtr >  bound) *outPtr -= 2*bound;
    if (*outPtr < -bound) *outPtr += 2*bound;
    inPtr++;
    outPtr++;
    for (unsigned int j = 1; j < ( (4 < num_points) ? 4 : num_points); j++) {
        *outPtr = *(inPtr) - *(inPtr-1);
        if (*outPtr >  bound) *outPtr -= 2*bound;
        if (*outPtr < -bound) *outPtr += 2*bound;
        inPtr++;
        outPtr++;
    }
    
    for(number = 1; number < quarter_points; number++) {
        // Load
        const float32x4_t next3old1_vec = vld1q_f32(inPtr-1);
        const float32x4_t next4_vec = vld1q_f32(inPtr);
        __VOLK_PREFETCH(inPtr+4);
        // Differentiate
        float32x4_t out_vec = vsubq_f32(next4_vec, next3old1_vec);
        // Constrain to bounds
        const uint32x4_t gt_bound = vcgtq_f32(out_vec, bound_vec);
        const uint32x4_t lt_neg_bound = vcltq_f32(out_vec, neg_bound_vec);
        out_vec = vbslq_f32(gt_bound, out_vec - bound_adj_vec, out_vec);
        out_vec = vbslq_f32(lt_neg_bound, out_vec + bound_adj_vec, out_vec);
        // Store
        vst1q_f32(outPtr, out_vec);
        
        inPtr+=4;
        outPtr+=4;
    }
    
    for (number = (4 > (quarter_points*4) ? 4 : (4 * quarter_points)); number < num_points; number++) {
        *outPtr = *(inPtr) - *(inPtr-1);
        if (*outPtr >  bound) *outPtr -= 2*bound;
        if (*outPtr < -bound) *outPtr += 2*bound;
        inPtr++;
        outPtr++;
    }
    
    // Continuous
    *saveValue = inputVector[num_points-1];
}
#endif /* LV_HAVE_NEON */


#endif /* INCLUDED_volk_32f_s32f_32f_fm_detect_32f_u_H */
