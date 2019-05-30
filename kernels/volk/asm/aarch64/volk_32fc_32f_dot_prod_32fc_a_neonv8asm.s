// static inline void volk_32fc_32f_dot_prod_32fc_a_neonv8asm(lv_32fc_t* result, const lv_32fc_t* input, const float * taps, unsigned int num_points)

.global	volk_32fc_32f_dot_prod_32fc_a_neonv8asm
.align 4

volk_32fc_32f_dot_prod_32fc_a_neonv8asm:
    // x0 - cVector: pointer to output
    // x1 - aVector: pointer to input array 1
    // x2 - bVector: pointer to input array 1
    // x3 - num_points: number of items to process

    // Callee must preserve r19-r29, sp
    // Can use  x0-x7, x9-x15 without storing anything on the stack
    // Can use v0-v7, v16-v31 without storing anything on the stack

    // quarter_points = num_points / 4;
    mov x4, x3, lsr #2
    // the rest
    and x5, x3, #0x03
    // zero accumulators
    eor v19.16b, v19.16b, v19.16b
    eor v20.16b, v20.16b, v20.16b
.loop1:
    cbz x4, .loop2
    sub x4, x4, #1
    // load aVector and bVector
    ld2 {v16.4s, v17.4s}, [x1], #32 // load and increment x1 by 4 * 4 * 2
    ld1 {v18.4s}, [x2], #16 // load and increment x1 by 4 * 4
    // multiply accumulate
    fmla v19.4s, v16.4s, v18.4s
    fmla v20.4s, v17.4s, v18.4s
    b .loop1
.loop2:
    cbz x5, .done
    sub x5, x5, 1
    // load aVector and bVector
    ldr s18, [x2], #4
    ldr s16, [x1], #4
    fmla v19.4s, v16.4s, v18.4s
    ldr s17, [x1], #4
    fmla v20.4s, v17.4s, v18.4s
    b .loop2
.done:
    // reduce lanes
    faddp v21.4s, v19.4s, v19.4s
    faddp v22.2s, v21.2s, v21.2s
    faddp v23.4s, v20.4s, v20.4s
    faddp v24.2s, v23.2s, v23.2s
    str s22, [x0], #4
    str s24, [x0]
    ret
