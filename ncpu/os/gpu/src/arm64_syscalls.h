/*
 * arm64_syscalls.h — Freestanding syscall wrappers for ARM64 Metal GPU kernel.
 *
 * These use SVC #0 to trap to the Python syscall handler.
 * Register convention follows Linux/ARM64: x8=syscall number, x0-x5=args.
 */

#ifndef ARM64_SYSCALLS_H
#define ARM64_SYSCALLS_H

typedef long ssize_t;
typedef unsigned long size_t;

/* Syscall numbers (Linux/ARM64 convention) */
#define SYS_GETCWD    17
#define SYS_MKDIRAT   34
#define SYS_UNLINKAT  35
#define SYS_CHDIR     49
#define SYS_OPENAT    56
#define SYS_CLOSE     57
#define SYS_GETDENTS64 61
#define SYS_LSEEK     62
#define SYS_READ      63
#define SYS_WRITE     64
#define SYS_FSTAT     80
#define SYS_EXIT      93
#define SYS_BRK      214

/* Custom syscalls for GPU OS */
#define SYS_COMPILE  300
#define SYS_EXEC     301
#define SYS_GETCHAR  302
#define SYS_CLOCK    303
#define SYS_SLEEP    304
#define SYS_SOCKET   305
#define SYS_BIND     306
#define SYS_LISTEN   307
#define SYS_ACCEPT   308
#define SYS_CONNECT  309
#define SYS_SEND     310
#define SYS_RECV     311

/* Process management syscalls */
#define SYS_DUP3     24
#define SYS_PIPE2    59
#define SYS_GETPID  172
#define SYS_GETPPID 173
#define SYS_FORK    220
#define SYS_WAIT4   260
#define SYS_PS      312
#define SYS_FLUSH_FB 313
#define SYS_KILL    314
#define SYS_GETENV  315
#define SYS_SETENV  316

/* Signal numbers */
#define SIGTERM  15
#define SIGKILL  9

static inline ssize_t sys_write(int fd, const char *buf, size_t len) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = (long)buf;
    register long x2 __asm__("x2") = (long)len;
    register long x8 __asm__("x8") = SYS_WRITE;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return x0;
}

static inline ssize_t sys_read(int fd, char *buf, size_t max) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = (long)buf;
    register long x2 __asm__("x2") = (long)max;
    register long x8 __asm__("x8") = SYS_READ;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return x0;
}

static inline void sys_exit(int code) {
    register long x0 __asm__("x0") = code;
    register long x8 __asm__("x8") = SYS_EXIT;
    __asm__ volatile("svc #0" : : "r"(x0), "r"(x8));
    __builtin_unreachable();
}

static inline int sys_flush_fb(int width, int height, void *addr) {
    register long x0 __asm__("x0") = width;
    register long x1 __asm__("x1") = height;
    register long x2 __asm__("x2") = (long)addr;
    register long x8 __asm__("x8") = SYS_FLUSH_FB;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

/* Helper: compute string length */
static inline int strlen(const char *s) {
    int n = 0;
    while (s[n]) n++;
    return n;
}

/* Helper: print null-terminated string to stdout */
static inline void print(const char *s) {
    sys_write(1, s, strlen(s));
}

/* Helper: print a single character */
static inline void putchar(char c) {
    sys_write(1, &c, 1);
}

/* Helper: print an integer in decimal */
static inline void print_int(long n) {
    char buf[21];
    int i = 0;
    int neg = 0;

    if (n < 0) {
        neg = 1;
        n = -n;
    }
    if (n == 0) {
        buf[i++] = '0';
    } else {
        while (n > 0) {
            buf[i++] = '0' + (n % 10);
            n /= 10;
        }
    }
    if (neg) buf[i++] = '-';

    /* Reverse */
    char out[21];
    for (int j = 0; j < i; j++) {
        out[j] = buf[i - 1 - j];
    }
    sys_write(1, out, i);
}

#endif /* ARM64_SYSCALLS_H */
