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


#ifndef INCLUDED_volk_32i_convert_32f_H
#define INCLUDED_volk_32i_convert_32f_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32i_convert_32f_generic(float* outputVector, const int32_t* inputVector, unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int32_t* inputVectorPtr = inputVector;
  unsigned int number = 0;
  const float iScalar = 1.0 / INT32_MAX;

  for(number = 0; number < num_points; number++){
     *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32i_convert_32f_neon(float* outputVector,
                               const int32_t* inputVector,
                               unsigned int num_points) {
    float* outputVectorPtr = outputVector;
    const int32_t* inputVectorPtr = inputVector;
    const float scalar = INT32_MAX;
    const float iScalar = 1.0 / scalar;
    unsigned int number;
    unsigned int quarter_points = num_points / 4;
    int32x4_t input_vec;
    float32x4_t output_vec;
    
    const float32x4_t iscalar_vec = vdupq_n_f32(iScalar);
    
    for(number = 0; number < quarter_points; number++) {
        // load s32
        input_vec = vld1q_s32(inputVectorPtr);
        // Prefetch next 4
        __VOLK_PREFETCH(inputVectorPtr+4);
        // convert s32 to f32
        output_vec = vcvtq_f32_s32(input_vec);
        // scale
        output_vec = vmulq_f32(output_vec, iscalar_vec);
        // store
        vst1q_f32(outputVectorPtr, output_vec);
        // move pointers ahead
        outputVectorPtr+=4;
        inputVectorPtr+=4;
    }
    
    // deal with the rest
    for(number = quarter_points * 4; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
extern void
volk_32i_convert_32f_a_neonv8asm(float* outputVector, const int32_t* inputVector, unsigned int num_points);
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32i_convert_32f_H */
