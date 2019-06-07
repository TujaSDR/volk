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


#ifndef INCLUDED_volk_32fc_add_real_imag_32f_H
#define INCLUDED_volk_32fc_add_real_imag_32f_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_add_real_imag_32f_generic(float* outputVector, const lv_32fc_t* inputVector, unsigned int num_points)
{
    float* outputVectorPtr = outputVector;
    const lv_32fc_t* inputVectorPtr = inputVector;
    unsigned int number = 0;
    
    for(number = 0; number < num_points; number++){
        *outputVectorPtr = lv_creal(*inputVectorPtr) + lv_cimag(*inputVectorPtr);
        outputVectorPtr++;
        inputVectorPtr++;
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static void inline
volk_32fc_add_real_imag_32f_neon(float* outputVector,
                                 const lv_32fc_t* inputVector,
                                 unsigned int num_points) {
    
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;
    float* outputVectorPtr = outputVector;
    const lv_32fc_t* inputVectorPtr = inputVector;
    
    for(number = 0; number < quarter_points; number++) {
        const float32x4x2_t input_vec = vld2q_f32((float*)inputVectorPtr);
        // Prefetch next one, speeds things up
        __VOLK_PREFETCH(inputVectorPtr+4);
        // Sum parts
        const float32x4_t output_vec = vaddq_f32(input_vec.val[0], input_vec.val[1]);
        // Store result
        vst1q_f32(outputVectorPtr, output_vec);
        // move pointers ahead
        outputVectorPtr+=4;
        inputVectorPtr+=4;
    }
    
    // Deal with the rest
    for(number = quarter_points * 4; number < num_points; number++) {
        *outputVectorPtr = lv_creal(*inputVectorPtr) + lv_cimag(*inputVectorPtr);
        outputVectorPtr++;
        inputVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_32fc_add_real_imag_32f_H */
