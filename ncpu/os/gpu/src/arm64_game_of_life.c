/*
 * Conway's Game of Life — Freestanding C for ARM64 Metal GPU kernel.
 *
 * 20x20 toroidal grid, double-buffered. Runs 30 generations.
 * Uses SVC syscalls for output (fd=3 signals grid dump to Python handler).
 *
 * No libc. No malloc. Pure freestanding C.
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/arm64_game_of_life.c
 *          -o /tmp/life.elf
 */

#include "arm64_syscalls.h"

#define ROWS 20
#define COLS 20

/* Double-buffered grids as static arrays in .bss */
static char grid_a[ROWS * COLS];
static char grid_b[ROWS * COLS];

/* Pointer to "current generation address" — written to fd=3 for Python */
static long grid_info[2];  /* [grid_ptr, generation] */

static inline char get_cell(const char *grid, int r, int c) {
    /* Toroidal wrapping */
    if (r < 0) r += ROWS;
    if (r >= ROWS) r -= ROWS;
    if (c < 0) c += COLS;
    if (c >= COLS) c -= COLS;
    return grid[r * COLS + c];
}

static int count_neighbors(const char *grid, int r, int c) {
    int count = 0;
    count += get_cell(grid, r - 1, c - 1);
    count += get_cell(grid, r - 1, c    );
    count += get_cell(grid, r - 1, c + 1);
    count += get_cell(grid, r    , c - 1);
    count += get_cell(grid, r    , c + 1);
    count += get_cell(grid, r + 1, c - 1);
    count += get_cell(grid, r + 1, c    );
    count += get_cell(grid, r + 1, c + 1);
    return count;
}

static void step(const char *src, char *dst) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            int n = count_neighbors(src, r, c);
            char alive = src[r * COLS + c];
            if (alive) {
                /* Survive on 2 or 3 neighbors */
                dst[r * COLS + c] = (n == 2 || n == 3) ? 1 : 0;
            } else {
                /* Birth on exactly 3 neighbors */
                dst[r * COLS + c] = (n == 3) ? 1 : 0;
            }
        }
    }
}

static void init_grid(char *grid) {
    /* Clear */
    for (int i = 0; i < ROWS * COLS; i++) {
        grid[i] = 0;
    }

    /* Place a glider at (1,1) */
    grid[1 * COLS + 2] = 1;  /*   .X. */
    grid[2 * COLS + 3] = 1;  /*   ..X */
    grid[3 * COLS + 1] = 1;  /*   XX. */
    grid[3 * COLS + 2] = 1;
    grid[3 * COLS + 3] = 1;

    /* Place a blinker at (10,10) */
    grid[10 * COLS + 9]  = 1;
    grid[10 * COLS + 10] = 1;
    grid[10 * COLS + 11] = 1;

    /* Place a block (still life) at (15,15) */
    grid[15 * COLS + 15] = 1;
    grid[15 * COLS + 16] = 1;
    grid[16 * COLS + 15] = 1;
    grid[16 * COLS + 16] = 1;
}

int main(void) {
    init_grid(grid_a);

    char *current = grid_a;
    char *next = grid_b;

    for (int gen = 0; gen <= 30; gen++) {
        /* Signal grid state to Python handler via fd=3 */
        grid_info[0] = (long)current;
        grid_info[1] = gen;
        sys_write(3, (const char *)grid_info, sizeof(grid_info));

        if (gen < 30) {
            step(current, next);
            /* Swap buffers */
            char *tmp = current;
            current = next;
            next = tmp;
        }
    }

    return 0;
}
