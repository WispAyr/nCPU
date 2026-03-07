/*
 * arm64_start.S — Startup code for ARM64 Metal GPU kernel programs.
 *
 * Sets up the stack pointer, calls main(), and exits with the return value.
 */

.global _start
.section .text

_start:
    /* Set stack pointer high in 16 MB GPU memory */
    /* 0xF00000 = 15 MB mark, leaving ~12 MB for data/BSS below */
    movz x0, #0xF0, lsl #16
    mov sp, x0

    /* Call main() */
    bl main

    /* Exit with return value from main (in x0) */
    mov x8, #93        /* SYS_EXIT */
    svc #0

    /* Should never reach here */
    hlt #0
