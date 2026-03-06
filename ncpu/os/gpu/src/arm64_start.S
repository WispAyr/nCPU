/*
 * arm64_start.S — Startup code for ARM64 Metal GPU kernel programs.
 *
 * Sets up the stack pointer, calls main(), and exits with the return value.
 */

.global _start
.section .text

_start:
    /* Set stack pointer to top of stack area (16 MB layout) */
    mov x0, #0xFF000
    mov sp, x0

    /* Call main() */
    bl main

    /* Exit with return value from main (in x0) */
    mov x8, #93        /* SYS_EXIT */
    svc #0

    /* Should never reach here */
    hlt #0
