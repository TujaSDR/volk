// static inline void volk_32f_s32f_multiply_32f_a_neonv8asm(float* cVector, const float* aVector, const float scalar, unsigned int num_points);

.global	volk_32f_s32f_multiply_32f_a_neonv8asm
.align 4

volk_32f_s32f_multiply_32f_a_neonv8asm:
    // x0 - cVector: pointer to output array
    // x1 - aVector: pointer to input array 1
    // v0 - scalar
    // x2 - num_points: number of items to process

    // Can use  x0-x7, x9-x15 without storing anything on the stack
    // Can use v0-v7, v16-v31 without storing anything on the stack

    // quarter_points = num_points / 4;
    mov x4, x2, lsr 2
    // the rest
    and x5, x2, 0x03
.loop1:
    cbz x4, .loop2
    ld1 {v16.4s}, [x1], 16 // load and increment x1 by 4 * 4
    sub x4, x4, 1
    // multiply
    fmul v17.4s, v16.4s, v0.s[0]
    // store
    st1 {v17.4s}, [x0], 16
    b .loop1
.loop2:
    cbz x5, .done
    // load 1
    ld1 {v16.s}[0], [x1], 4
    sub x5, x5, 1
    fmul v17.4s, v16.4s, v0.s[0]
    st1 {v17.s}[0], [x0], 4
    b .loop2
.done:
    ret
