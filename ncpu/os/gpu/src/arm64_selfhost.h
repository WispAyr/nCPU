/*
 * arm64_selfhost.h — Self-hosting C runtime for cc.c on ARM64 Metal GPU.
 *
 * This header provides the same API as arm64_libc.h but without inline
 * assembly, so that cc.c can compile itself. Syscalls use the __syscall()
 * compiler intrinsic which cc.c emits as MOV X8,nr; SVC #0.
 *
 * Only included when __CCGPU__ is defined (i.e., compiled by cc.c itself).
 */

#ifndef ARM64_SELFHOST_H
#define ARM64_SELFHOST_H

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TYPES                                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef long ssize_t;
typedef unsigned long size_t;

#ifndef NULL
#define NULL 0
#endif

/* ═══════════════════════════════════════════════════════════════════════════ */
/* FILE I/O CONSTANTS                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define O_RDONLY  0
#define O_WRONLY  1
#define O_RDWR   2
#define O_CREAT   64
#define O_TRUNC  512
#define O_APPEND 1024

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

#define AT_FDCWD -100

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SYSCALL NUMBERS (Linux/ARM64)                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define SYS_OPENAT    56
#define SYS_CLOSE     57
#define SYS_LSEEK     62
#define SYS_READ      63
#define SYS_WRITE     64
#define SYS_EXIT      93
#define SYS_BRK      214

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SYSCALL WRAPPERS (via __syscall intrinsic)                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

long __syscall(long nr, long a0, long a1, long a2, long a3, long a4);

int open(const char *path, int flags) {
    return (int)__syscall(SYS_OPENAT, AT_FDCWD, (long)path, (long)flags, 0644, 0);
}

int close(int fd) {
    return (int)__syscall(SYS_CLOSE, (long)fd, 0, 0, 0, 0);
}

long read(int fd, void *buf, long len) {
    return __syscall(SYS_READ, (long)fd, (long)buf, len, 0, 0);
}

long write(int fd, const void *buf, long len) {
    return __syscall(SYS_WRITE, (long)fd, (long)buf, len, 0, 0);
}

long lseek(int fd, long offset, int whence) {
    return __syscall(SYS_LSEEK, (long)fd, offset, (long)whence, 0, 0);
}

void exit(int code) {
    __syscall(SYS_EXIT, (long)code, 0, 0, 0, 0);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* STRING OPERATIONS                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

int strlen(const char *s) {
    int n = 0;
    while (s[n]) n++;
    return n;
}

char *strcpy(char *dst, const char *src) {
    char *d = dst;
    while (*src) {
        *d = *src;
        d++;
        src++;
    }
    *d = 0;
    return dst;
}

char *strncpy(char *dst, const char *src, int n) {
    int i;
    for (i = 0; i < n && src[i]; i++)
        dst[i] = src[i];
    for (; i < n; i++)
        dst[i] = 0;
    return dst;
}

char *strcat(char *dst, const char *src) {
    char *d = dst;
    while (*d) d++;
    while (*src) {
        *d = *src;
        d++;
        src++;
    }
    *d = 0;
    return dst;
}

int strcmp(const char *a, const char *b) {
    while (*a && *b && *a == *b) { a++; b++; }
    return *a - *b;
}

int strncmp(const char *a, const char *b, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (a[i] != b[i]) return a[i] - b[i];
        if (a[i] == 0) return 0;
    }
    return 0;
}

char *strchr(const char *s, int c) {
    while (*s) {
        if (*s == (char)c) return (char *)s;
        s++;
    }
    return NULL;
}

void *memcpy(void *dst, const void *src, int n) {
    char *d = (char *)dst;
    char *s = (char *)src;
    int i;
    for (i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void *memset(void *s, int c, int n) {
    char *p = (char *)s;
    int i;
    for (i = 0; i < n; i++) p[i] = (char)c;
    return s;
}

long strtol(const char *s, char *endp, int base) {
    long val = 0;
    int neg = 0;
    while (*s == ' ') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') s++;

    if (base == 0) {
        if (*s == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s += 2; }
        else if (*s == '0') { base = 8; s++; }
        else base = 10;
    } else if (base == 16 && *s == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
    }

    while (*s) {
        int d = -1;
        if (*s >= '0' && *s <= '9') d = *s - '0';
        else if (*s >= 'a' && *s <= 'f') d = *s - 'a' + 10;
        else if (*s >= 'A' && *s <= 'F') d = *s - 'A' + 10;
        if (d < 0 || d >= base) break;
        val = val * base + d;
        s++;
    }
    return neg ? -val : val;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PRINT HELPERS                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

void print(const char *s) {
    write(1, s, strlen(s));
}

void putchar(char c) {
    write(1, &c, 1);
}

void print_int(long n) {
    char buf[21];
    int i = 0;
    int neg = 0;
    if (n < 0) { neg = 1; n = -n; }
    if (n == 0) buf[i++] = '0';
    else {
        while (n > 0) {
            buf[i++] = '0' + (int)(n % 10);
            n = n / 10;
        }
    }
    if (neg) buf[i++] = '-';
    /* Reverse into output */
    char out[21];
    int j;
    for (j = 0; j < i; j++) out[j] = buf[i - 1 - j];
    write(1, out, i);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PRINTF — Simplified (handles %s, %d, %ld, %x, %c, %%)                    */
/*                                                                           */
/* Declared with explicit params since cc.c does not support variadic (...). */
/* The ARM64 calling convention puts args in X0-X7 in order, so callers      */
/* passing fewer args is safe — unused params hold stale register values.    */
/* ═══════════════════════════════════════════════════════════════════════════ */

int printf(const char *fmt, long a0, long a1, long a2, long a3, long a4) {
    long args[6];
    args[0] = a0;
    args[1] = a1;
    args[2] = a2;
    args[3] = a3;
    args[4] = a4;
    args[5] = 0;
    int ai = 0;
    int total = 0;
    int fi = 0;

    while (fmt[fi]) {
        if (fmt[fi] == '%' && fmt[fi + 1]) {
            fi++;
            /* Skip 'l' modifier */
            if (fmt[fi] == 'l') fi++;
            if (fmt[fi] == 'd') {
                print_int(args[ai]);
                ai++;
            } else if (fmt[fi] == 's') {
                char *s = (char *)args[ai];
                if (s) print(s);
                ai++;
            } else if (fmt[fi] == 'x') {
                /* Hex output */
                long v = args[ai];
                ai++;
                char hex[17];
                int hi = 0;
                if (v == 0) { hex[hi++] = '0'; }
                else {
                    char tmp[17];
                    int ti = 0;
                    while (v > 0) {
                        int d = (int)(v & 15);
                        if (d < 10) tmp[ti++] = '0' + d;
                        else tmp[ti++] = 'a' + d - 10;
                        v = v >> 4;
                    }
                    while (ti > 0) hex[hi++] = tmp[--ti];
                }
                write(1, hex, hi);
            } else if (fmt[fi] == 'c') {
                char ch = (char)args[ai];
                ai++;
                putchar(ch);
            } else if (fmt[fi] == '%') {
                putchar('%');
            } else {
                /* Unknown format: print raw */
                putchar('%');
                putchar(fmt[fi]);
            }
            fi++;
        } else {
            putchar(fmt[fi]);
            fi++;
        }
        total++;
    }
    return total;
}

#endif /* ARM64_SELFHOST_H */
