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


#ifndef INCLUDED_volk_32f_convert_32i_H
#define INCLUDED_volk_32f_convert_32i_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_convert_32i_generic(int32_t* outputVector, const float* inputVector, unsigned int num_points)
{
    int32_t* outputVectorPtr = outputVector;
    const float* inputVectorPtr = inputVector;
    unsigned int number = 0;
    const float scalar = (float) INT32_MAX;
    float r;
    
    for(number = 0; number < num_points; number++){
        r = *inputVectorPtr++ * scalar;
        /*if(r > max_val)
            r = max_val;
        else if(r < min_val)
            r = min_val;*/
        *outputVectorPtr++ = (int32_t)rintf(r);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEONV8
static inline void
volk_32f_convert_32i_neonv8(int32_t* outputVector, const float* inputVector, unsigned int num_points) {
    int32_t* outputVectorPtr = outputVector;
    const float* inputVectorPtr = inputVector;
    const float scalar = INT32_MAX;
    unsigned int number;
    unsigned int quarter_points = num_points / 4;
    float32x4_t input_vec;
    int32x4_t   output_vec;
    
    const float32x4_t scalar_vec = vdupq_n_f32(scalar);
    
    for(number = 0; number < quarter_points; number++) {
        // load floats
        input_vec = vld1q_f32(inputVectorPtr);
        // Prefetch next 4
        __VOLK_PREFETCH(inputVectorPtr+4);
        const float32x4_t scaled_vec = vmulq_f32(input_vec, scalar_vec);
        // convert f32 to s32
        // think about rounding?
        output_vec = vcvtq_s32_f32(scaled_vec);
        // store
        vst1q_s32(outputVectorPtr, output_vec);
        // move pointers ahead
        outputVectorPtr+=4;
        inputVectorPtr+=4;
    }
    
    // deal with the rest
    for(number = quarter_points * 4; number < num_points; number++) {
        // rounding...?
        *outputVectorPtr++ = (int32_t)((*inputVectorPtr++) * scalar);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_NEONV8
extern void
volk_32f_convert_32i_a_neonv8asm(int32_t* outputVector, const float* inputVector, unsigned int num_points);
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32f_convert_32i_H */
