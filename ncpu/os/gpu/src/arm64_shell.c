/*
 * Interactive Shell — Freestanding C for ARM64 Metal GPU kernel.
 *
 * Commands: echo, add, mul, fib, fact, help, info, exit
 * Uses SVC syscalls for I/O. No libc.
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/arm64_shell.c
 *          -o /tmp/shell.elf
 */

#include "arm64_syscalls.h"

/* ═══════════════════════════════════════════════════════════════════════════ */
/* STRING UTILITIES                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int my_strcmp(const char *a, const char *b) {
    while (*a && *b && *a == *b) { a++; b++; }
    return *a - *b;
}

static int starts_with(const char *str, const char *prefix) {
    while (*prefix) {
        if (*str != *prefix) return 0;
        str++;
        prefix++;
    }
    return 1;
}

static long atoi(const char *s) {
    long n = 0;
    int neg = 0;
    while (*s == ' ') s++;
    if (*s == '-') { neg = 1; s++; }
    while (*s >= '0' && *s <= '9') {
        n = n * 10 + (*s - '0');
        s++;
    }
    return neg ? -n : n;
}

/* Skip to next whitespace-separated token */
static const char *next_token(const char *s) {
    while (*s && *s != ' ' && *s != '\t') s++;
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

/* Skip to next space-separated number */
static const char *skip_to_arg(const char *cmd) {
    /* Skip command name */
    while (*cmd && *cmd != ' ' && *cmd != '\t') cmd++;
    /* Skip spaces */
    while (*cmd == ' ' || *cmd == '\t') cmd++;
    return cmd;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* COMMANDS                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void cmd_help(void) {
    print("Commands:\n");
    print("  echo <text>    Print text\n");
    print("  add <a> <b>    Add two numbers\n");
    print("  mul <a> <b>    Multiply two numbers\n");
    print("  fib <n>        Fibonacci(n)\n");
    print("  fact <n>       Factorial(n)\n");
    print("  help           Show this help\n");
    print("  info           System info\n");
    print("  exit           Exit shell\n");
}

static void cmd_info(void) {
    print("ARM64 Shell v1.0\n");
    print("Running on: Metal GPU (Apple Silicon)\n");
    print("ISA: ARM64 (125 instructions)\n");
    print("Kernel: MLXKernelCPUv2 (double-buffer memory)\n");
    print("Compiler: aarch64-elf-gcc 15.2.0\n");
}

static void cmd_add(const char *args) {
    long a = atoi(args);
    const char *b_str = next_token(args);
    long b = atoi(b_str);
    print_int(a);
    print(" + ");
    print_int(b);
    print(" = ");
    print_int(a + b);
    print("\n");
}

static void cmd_mul(const char *args) {
    long a = atoi(args);
    const char *b_str = next_token(args);
    long b = atoi(b_str);
    print_int(a);
    print(" * ");
    print_int(b);
    print(" = ");
    print_int(a * b);
    print("\n");
}

static void cmd_fib(const char *args) {
    long n = atoi(args);
    if (n < 0) {
        print("Error: n must be >= 0\n");
        return;
    }
    long a = 0, b = 1;
    for (long i = 0; i < n; i++) {
        long tmp = a + b;
        a = b;
        b = tmp;
    }
    print("fib(");
    print_int(n);
    print(") = ");
    print_int(a);
    print("\n");
}

static void cmd_fact(const char *args) {
    long n = atoi(args);
    if (n < 0) {
        print("Error: n must be >= 0\n");
        return;
    }
    long result = 1;
    for (long i = 2; i <= n; i++) {
        result *= i;
    }
    print("fact(");
    print_int(n);
    print(") = ");
    print_int(result);
    print("\n");
}

static void cmd_echo(const char *args) {
    print(args);
    print("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MAIN LOOP                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

static char input_buf[256];

int main(void) {
    print("ARM64 Shell running on Metal GPU\n");
    print("Type 'help' for commands, 'exit' to quit.\n\n");

    while (1) {
        print("> ");

        /* Read a line from stdin */
        ssize_t n = sys_read(0, input_buf, sizeof(input_buf) - 1);
        if (n <= 0) break;

        /* Null-terminate, strip trailing newline */
        input_buf[n] = '\0';
        if (n > 0 && input_buf[n - 1] == '\n') {
            input_buf[n - 1] = '\0';
            n--;
        }
        if (n == 0) continue;

        /* Dispatch command */
        const char *args = skip_to_arg(input_buf);

        if (my_strcmp(input_buf, "exit") == 0) {
            print("Goodbye!\n");
            break;
        } else if (my_strcmp(input_buf, "help") == 0) {
            cmd_help();
        } else if (my_strcmp(input_buf, "info") == 0) {
            cmd_info();
        } else if (starts_with(input_buf, "echo ")) {
            cmd_echo(args);
        } else if (starts_with(input_buf, "add ")) {
            cmd_add(args);
        } else if (starts_with(input_buf, "mul ")) {
            cmd_mul(args);
        } else if (starts_with(input_buf, "fib ")) {
            cmd_fib(args);
        } else if (starts_with(input_buf, "fact ")) {
            cmd_fact(args);
        } else {
            print("Unknown command: ");
            print(input_buf);
            print("\nType 'help' for commands.\n");
        }
    }

    return 0;
}
