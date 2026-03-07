"""
Alpine Linux Rootfs Builder — Create a complete Alpine Linux filesystem for GPU execution.

Builds a GPUFilesystem populated with the Alpine Linux FHS directory tree,
identity files, configuration, synthetic /proc, and BusyBox applet registry.

The resulting filesystem is served to a real BusyBox binary (Alpine's core userspace)
running on the Metal GPU compute shader via SVC-trapped syscalls.

Author: Robert Price
Date: March 2026
"""

from ncpu.os.gpu.filesystem import GPUFilesystem


# BusyBox applets known to be available in the Alpine BusyBox binary.
# These map to argv[0] when invoking BusyBox as a multi-call binary.
BUSYBOX_APPLETS = [
    # Core utils
    "ash", "sh", "busybox",
    # File operations
    "ls", "cat", "cp", "mv", "rm", "mkdir", "rmdir", "touch",
    "chmod", "chown", "chgrp", "ln", "readlink", "stat",
    # Text processing
    "grep", "egrep", "fgrep", "sed", "awk", "sort", "uniq",
    "wc", "head", "tail", "cut", "tr", "tee", "fold",
    # File finding
    "find", "xargs", "which",
    # Output / display
    "echo", "printf", "cat", "more", "less",
    # Path manipulation
    "basename", "dirname", "realpath",
    # System info
    "uname", "hostname", "whoami", "id", "uptime", "free",
    "df", "du", "mount", "umount",
    # Process control
    "ps", "kill", "sleep", "true", "false", "test", "[",
    "env", "printenv", "expr", "seq", "yes",
    # Archiving
    "tar", "gzip", "gunzip", "zcat",
    # Networking
    "wget", "ping", "ifconfig", "route", "netstat",
    # Admin
    "adduser", "addgroup", "passwd", "su", "login",
    "init", "halt", "reboot", "poweroff",
    # Package management
    "apk",
    # Editors
    "vi", "ed",
    # Misc
    "date", "cal", "clear", "reset", "tty",
    "md5sum", "sha256sum", "sha1sum",
    "nslookup", "traceroute",
    "diff", "cmp", "patch",
    "install", "mktemp",
    "od", "hexdump", "xxd",
]


def create_alpine_rootfs() -> GPUFilesystem:
    """Create a complete Alpine Linux v3.20 filesystem for GPU execution.

    Returns:
        GPUFilesystem populated with Alpine FHS, identity, config, and /proc.
    """
    fs = GPUFilesystem()

    # ═══════════════════════════════════════════════════════════════════
    # DIRECTORY TREE (Alpine FHS)
    # ═══════════════════════════════════════════════════════════════════
    alpine_dirs = [
        "/bin", "/sbin", "/usr/bin", "/usr/sbin",
        "/etc", "/etc/init.d", "/etc/apk", "/etc/network",
        "/home", "/home/user", "/root",
        "/tmp", "/var", "/var/log", "/var/run", "/var/cache", "/var/cache/apk",
        "/proc", "/sys", "/dev", "/dev/pts", "/dev/shm",
        "/lib", "/usr/lib", "/usr/share", "/usr/local",
        "/run", "/mnt", "/media", "/opt", "/srv",
    ]
    for d in alpine_dirs:
        fs.directories.add(d)

    # ═══════════════════════════════════════════════════════════════════
    # IDENTITY FILES
    # ═══════════════════════════════════════════════════════════════════
    fs.write_file("/etc/alpine-release", "3.20.0\n")
    fs.write_file("/etc/os-release",
        'NAME="Alpine Linux"\n'
        'ID=alpine\n'
        'VERSION_ID=3.20.0\n'
        'PRETTY_NAME="Alpine Linux v3.20 (nCPU GPU)"\n'
        'HOME_URL="https://alpinelinux.org/"\n'
        'BUG_REPORT_URL="https://gitlab.alpinelinux.org/alpine/aports/-/issues"\n'
    )
    fs.write_file("/etc/issue", "Welcome to Alpine Linux v3.20 (nCPU GPU)\\n\\l\n\n")

    # ═══════════════════════════════════════════════════════════════════
    # USER / AUTH FILES
    # ═══════════════════════════════════════════════════════════════════
    fs.write_file("/etc/passwd",
        "root:x:0:0:root:/root:/bin/ash\n"
        "bin:x:1:1:bin:/bin:/sbin/nologin\n"
        "daemon:x:2:2:daemon:/sbin:/sbin/nologin\n"
        "nobody:x:65534:65534:Nobody:/:/sbin/nologin\n"
        "user:x:1000:1000:Linux User:/home/user:/bin/ash\n"
    )
    fs.write_file("/etc/group",
        "root:x:0:root\n"
        "bin:x:1:root,bin,daemon\n"
        "daemon:x:2:root,bin,daemon\n"
        "wheel:x:10:root\n"
        "nobody:x:65534:\n"
        "users:x:1000:\n"
    )
    fs.write_file("/etc/shadow",
        "root:!::0:::::\n"
        "bin:!::0:::::\n"
        "daemon:!::0:::::\n"
        "nobody:!::0:::::\n"
        "user:!::0:::::\n"
    )

    # ═══════════════════════════════════════════════════════════════════
    # SYSTEM CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    fs.write_file("/etc/hostname", "ncpu-gpu\n")
    fs.write_file("/etc/hosts",
        "127.0.0.1\tlocalhost\n"
        "::1\t\tlocalhost\n"
        "127.0.1.1\tncpu-gpu\n"
    )
    fs.write_file("/etc/resolv.conf",
        "nameserver 8.8.8.8\n"
        "nameserver 8.8.4.4\n"
    )
    fs.write_file("/etc/profile",
        'export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\n'
        'export HOME="/root"\n'
        'export TERM="xterm"\n'
        'export PS1="\\u@\\h:\\w\\$ "\n'
        'export CHARSET="UTF-8"\n'
        'export LANG="C.UTF-8"\n'
        'umask 022\n'
    )
    fs.write_file("/etc/shells",
        "/bin/ash\n"
        "/bin/sh\n"
    )
    fs.write_file("/etc/motd",
        "Welcome to Alpine Linux v3.20 (nCPU GPU)\n"
        "Running on Apple Silicon Metal compute shader\n"
        "\n"
        "All ARM64 instructions execute natively on GPU.\n"
        "Syscalls are trapped via SVC and handled by Python.\n"
    )
    fs.write_file("/etc/inittab",
        "::sysinit:/sbin/openrc sysinit\n"
        "::sysinit:/sbin/openrc boot\n"
        "::wait:/sbin/openrc default\n"
        "tty1::respawn:/sbin/getty 38400 tty1\n"
        "::ctrlaltdel:/sbin/reboot\n"
        "::shutdown:/sbin/openrc shutdown\n"
    )
    fs.write_file("/etc/network/interfaces",
        "auto lo\n"
        "iface lo inet loopback\n"
        "\n"
        "auto eth0\n"
        "iface eth0 inet dhcp\n"
    )

    # APK (package manager) stubs
    fs.write_file("/etc/apk/repositories",
        "https://dl-cdn.alpinelinux.org/alpine/v3.20/main\n"
        "https://dl-cdn.alpinelinux.org/alpine/v3.20/community\n"
    )
    fs.write_file("/etc/apk/world",
        "alpine-base\n"
        "busybox\n"
    )

    # ═══════════════════════════════════════════════════════════════════
    # SYNTHETIC /proc
    # ═══════════════════════════════════════════════════════════════════
    fs.write_file("/proc/version",
        "Linux version 6.1.0-ncpu (gcc 13.2) #1 SMP Metal GPU aarch64\n"
    )
    fs.write_file("/proc/cpuinfo",
        "processor\t: 0\n"
        "BogoMIPS\t: 48.00\n"
        "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
        "CPU implementer\t: 0x61\n"
        "CPU architecture: 8\n"
        "CPU variant\t: 0x1\n"
        "CPU part\t: 0xd07\n"
        "CPU revision\t: 4\n"
        "\n"
        "Hardware\t: Apple Silicon (nCPU Metal Compute)\n"
    )
    fs.write_file("/proc/meminfo",
        "MemTotal:         262144 kB\n"
        "MemFree:          131072 kB\n"
        "MemAvailable:     196608 kB\n"
        "Buffers:               0 kB\n"
        "Cached:            65536 kB\n"
        "SwapTotal:             0 kB\n"
        "SwapFree:              0 kB\n"
    )
    fs.write_file("/proc/uptime", "3600.00 3600.00\n")
    fs.write_file("/proc/loadavg", "0.00 0.00 0.00 1/1 1\n")
    fs.write_file("/proc/stat",
        "cpu  0 0 0 0 0 0 0 0 0 0\n"
        "cpu0 0 0 0 0 0 0 0 0 0 0\n"
    )
    fs.write_file("/proc/mounts",
        "rootfs / rootfs rw 0 0\n"
        "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0\n"
        "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0\n"
    )
    fs.write_file("/proc/filesystems",
        "nodev\tproc\n"
        "nodev\tsysfs\n"
        "nodev\ttmpfs\n"
        "\text4\n"
    )

    # ═══════════════════════════════════════════════════════════════════
    # SAMPLE USER FILES
    # ═══════════════════════════════════════════════════════════════════
    fs.write_file("/root/.ash_history", "")
    fs.write_file("/root/.profile",
        'export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\n'
        'export HOME="/root"\n'
    )
    fs.write_file("/home/user/hello.txt",
        "Hello from Alpine Linux on GPU!\n"
        "This file lives in Python memory,\n"
        "served to BusyBox via syscalls.\n"
    )
    fs.write_file("/tmp/data.txt",
        "apple\n"
        "banana\n"
        "cherry\n"
        "date\n"
        "elderberry\n"
        "fig\n"
        "grape\n"
    )
    fs.write_file("/var/log/messages", "")

    return fs
