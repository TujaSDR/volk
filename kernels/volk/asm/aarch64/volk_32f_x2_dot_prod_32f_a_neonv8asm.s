// static inline void volk_32f_x2_dot_prod_32f_a_neonv8asm(float * result, const float * input, const float * taps, unsigned int num_points)

.global	volk_32f_x2_dot_prod_32f_a_neonv8asm
.align 4

volk_32f_x2_dot_prod_32f_a_neonv8asm:
    // x0 - cVector: pointer to output array
    // x1 - aVector: pointer to input array 1
    // x2 - bVector: pointer to input array 1
    // x3 - num_points: number of items to process

    // Can use  x0-x7, x9-x15 without storing anything on the stack
    // Can use v0-v7, v16-v31 without storing anything on the stack

    // quarter_points = num_points / 4;
    mov x4, x3, lsr 2
    // the rest
    and x5, x3, 0x03
.loop1:
    cbz x4, .loop2
    sub x4, x4, 1
    // load aVector and bVector
    ld1 {v16.4s}, [x1], 16 // load and increment x1 by 4 * 4
    ld1 {v17.4s}, [x2], 16 // load and increment x1 by 4 * 4
    // add
    fadd v18.4s, v16.4s, v17.4s
    // store to cVector
    st1 {v18.4s}, [x0], 16

    b .loop1
.loop2:
    cbz x5, .done
    sub x5, x5, 1
    // load aVector and bVector
    ldr s16, [x1], 4
    ldr s17, [x2], 4
    // add
    fadd s18, s16, s17
    str s18, [x0], 4
    b .loop2
.done:
    ret
