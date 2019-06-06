// static inline void volk_32i_convert_32f_a_neonv8asm(float* outputVector, const float* inputVector, unsigned int num_points)

.global	volk_32i_convert_32f_a_neonv8asm
.align 4

volk_32i_convert_32f_a_neonv8asm:
    // x0 - cVector: pointer to output array
    // x1 - aVector: pointer to input array
    // x2 - num_points: number of items to process

    // Callee must preserve r19-r29, sp
    // Can use  x0-x7, x9-x15 without storing anything on the stack
    // Can use v0-v7, v16-v31 without storing anything on the stack

    // quarter_points = num_points / 4;
    mov x4, x2, lsr 2
    // the rest
    and x5, x2, 0x03
.loop1:
    cbz x4, .loop2
    sub x4, x4, 1
    // load aVector and bVector
    ld1 {v16.4s}, [x1], 16 // 4 * 4
    // convert to fixed poing q1.31
    scvtf v17.4s, v16.4s, 31
    // store and increment
    st1 {v17.4s}, [x0], 16 // 4 * 4
    b .loop1
.loop2:
    cbz x5, .done
    sub x5, x5, 1
    // load aVector
    ldr s16, [x1], 4
    scvtf s17, s16, 31
    str s17, [x0], 4
    b .loop2
.done:
    ret
