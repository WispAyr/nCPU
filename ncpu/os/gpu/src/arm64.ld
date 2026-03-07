/* Linker script for ARM64 Metal GPU kernel */
/* Memory layout (16 MB GPU memory):
 *   0x000000 - 0x00FFFF: Reserved
 *   0x010000 - 0x04FFFF: .text + .rodata (256 KB code)
 *   0x050000 - 0xEFFFFF: .data + .bss (up to ~14 MB for globals/heap)
 *   0xF00000           : Stack top (grows down)
 *   0xF00000 - 0xFFFFFF: Stack (1 MB)
 */

SECTIONS {
    . = 0x10000;
    .text : { *(.text*) }
    .rodata : { *(.rodata*) }
    . = 0x50000;
    .data : { *(.data*) }
    .bss : { *(.bss*) *(COMMON) }
}
