/* Linker script for ARM64 Metal GPU kernel */
/* Memory layout (16 MB GPU memory):
 *   0x000000 - 0x00FFFF: Reserved
 *   0x010000 - 0x04FFFF: .text + .rodata (256 KB code)
 *   0x050000 - 0x09FFFF: .data + .bss (256 KB globals + heap start)
 *   0x0A0000 - 0x0FEFFF: Heap (brk)
 *   0x0FF000           : Stack top (grows down)
 *
 *   Multi-process backing stores (1 MB each):
 *   0x100000 - 0x1FFFFF: Process 1 backing store
 *   0x200000 - 0x2FFFFF: Process 2 backing store
 *   ...up to Process 15 at 0xF00000
 */

SECTIONS {
    . = 0x10000;
    .text : { *(.text*) }
    .rodata : { *(.rodata*) }
    . = 0x50000;
    .data : { *(.data*) }
    .bss : { *(.bss*) *(COMMON) }
}
