#!/usr/bin/env python3
"""
Alpine Linux on GPU — Full distro running on Metal compute shader.

Real Alpine Linux userspace (BusyBox + musl libc, aarch64) executing
entirely on Apple Silicon GPU via Metal compute shaders.

Each command spawns a fresh BusyBox ELF invocation on the GPU with a
shared Python-side filesystem that persists across commands — exactly
like a real Linux system where /bin/busybox is the multi-call binary
behind every core utility.

Usage:
    python demos/alpine_gpu.py                # Interactive Alpine shell
    python demos/alpine_gpu.py --demo         # Automated demo suite
    python demos/alpine_gpu.py cat /etc/motd  # Run single command

Author: Robert Price
Date: March 2026
"""

import io
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.elf_loader import load_and_run_elf
from ncpu.os.gpu.alpine import create_alpine_rootfs

BUSYBOX = str(Path(__file__).parent / "busybox.elf")


def run_command(argv, filesystem, quiet=True):
    """Run a BusyBox command on GPU with shared filesystem."""
    return load_and_run_elf(
        BUSYBOX,
        argv=argv,
        max_cycles=500_000_000,
        quiet=quiet,
        filesystem=filesystem,
    )


def run_and_capture(argv, filesystem):
    """Run command and capture stdout output, returning (output_str, results)."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        results = run_command(argv, filesystem, quiet=True)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return output, results


def interactive_mode():
    """Interactive Alpine Linux shell on GPU."""
    fs = create_alpine_rootfs()
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root",
        "TERM": "xterm",
        "USER": "root",
        "LOGNAME": "root",
        "SHELL": "/bin/ash",
        "HOSTNAME": "ncpu-gpu",
        "LANG": "C.UTF-8",
    }

    print("Alpine Linux v3.20 (nCPU GPU)")
    print("Running on Apple Silicon Metal compute shader")
    print("Type 'exit' to quit. Each command executes BusyBox on GPU.\n")

    while True:
        # Build prompt
        cwd = fs.cwd
        if cwd == "/root":
            cwd_display = "~"
        elif cwd.startswith("/root/"):
            cwd_display = "~" + cwd[5:]
        else:
            cwd_display = cwd
        prompt = f"root@ncpu-gpu:{cwd_display}# "

        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue
        if line in ("exit", "quit", "logout"):
            break

        # Handle cd locally (no ELF invocation needed)
        parts = line.split()
        if parts[0] == "cd":
            target = parts[1] if len(parts) > 1 else env.get("HOME", "/")
            if target == "-":
                target = env.get("OLDPWD", fs.cwd)
            old_cwd = fs.cwd
            result = fs.chdir(target)
            if result == 0:
                env["OLDPWD"] = old_cwd
                env["PWD"] = fs.cwd
            else:
                print(f"ash: cd: can't cd to {target}: No such file or directory")
            continue

        # Handle export locally
        if parts[0] == "export" and len(parts) > 1:
            for assignment in parts[1:]:
                if "=" in assignment:
                    key, val = assignment.split("=", 1)
                    env[key] = val
            continue

        # Parse pipes
        pipe_segments = _split_pipes(line)

        if len(pipe_segments) == 1:
            # Single command — handle redirection
            argv, redir_file, redir_append = _parse_redirects(pipe_segments[0])
            if not argv:
                continue

            if redir_file:
                output, results = run_and_capture(argv, fs)
                path = fs.resolve_path(redir_file)
                if redir_append:
                    existing = fs.read_file(path) or b""
                    fs.write_file(path, existing + output.encode("utf-8", errors="replace"))
                else:
                    fs.write_file(path, output.encode("utf-8", errors="replace"))
            else:
                run_command(argv, fs, quiet=True)
        else:
            # Pipeline: run each segment, feeding stdout → stdin
            _run_pipeline(pipe_segments, fs, env)


def _split_pipes(line):
    """Split command line on pipe characters."""
    segments = []
    current = []
    for token in line.split():
        if token == "|":
            if current:
                segments.append(" ".join(current))
                current = []
        else:
            current.append(token)
    if current:
        segments.append(" ".join(current))
    return segments


def _parse_redirects(cmd_str):
    """Parse a command string for output redirection. Returns (argv, file, append)."""
    tokens = cmd_str.split()
    redir_file = None
    redir_append = False

    for i, tok in enumerate(tokens):
        if tok == ">>" and i + 1 < len(tokens):
            redir_file = tokens[i + 1]
            redir_append = True
            tokens = tokens[:i]
            break
        elif tok == ">" and i + 1 < len(tokens):
            redir_file = tokens[i + 1]
            redir_append = False
            tokens = tokens[:i]
            break
        elif tok.startswith(">>"):
            redir_file = tok[2:]
            redir_append = True
            tokens = tokens[:i]
            break
        elif tok.startswith(">") and len(tok) > 1:
            redir_file = tok[1:]
            redir_append = False
            tokens = tokens[:i]
            break

    return tokens, redir_file, redir_append


def _run_pipeline(segments, fs, env):
    """Execute a pipeline by chaining commands through intermediate files."""
    prev_output = None

    for i, segment in enumerate(segments):
        argv, redir_file, redir_append = _parse_redirects(segment)
        if not argv:
            continue

        is_last = (i == len(segments) - 1)

        # If there's input from previous command, write it to a temp file
        # and have the command read from it via stdin redirection
        if prev_output is not None:
            # Write previous output to temp pipe file
            pipe_path = f"/tmp/.pipe_{i}"
            fs.write_file(pipe_path, prev_output.encode("utf-8", errors="replace"))
            # BusyBox commands that read stdin will use /dev/stdin
            # We simulate by adding the pipe file as an argument for filter commands
            filter_cmds = {"grep", "sort", "uniq", "wc", "head", "tail", "cut",
                           "tr", "tee", "awk", "sed", "cat"}
            if argv[0] in filter_cmds and not any(
                not a.startswith("-") and a != argv[0] for a in argv[1:]
                if not a.startswith("-")
            ):
                # No file argument — add pipe file
                argv.append(pipe_path)

        if is_last and not redir_file:
            # Last command — output goes to terminal
            run_command(argv, fs, quiet=True)
            prev_output = None
        else:
            # Capture output for next stage or redirection
            output, results = run_and_capture(argv, fs)
            if redir_file:
                path = fs.resolve_path(redir_file)
                if redir_append:
                    existing = fs.read_file(path) or b""
                    fs.write_file(path, existing + output.encode("utf-8", errors="replace"))
                else:
                    fs.write_file(path, output.encode("utf-8", errors="replace"))
                prev_output = None
            else:
                prev_output = output

    # Clean up temp pipe files
    to_clean = [p for p in fs.files if p.startswith("/tmp/.pipe_")]
    for p in to_clean:
        del fs.files[p]


def demo_suite():
    """Automated Alpine Linux demo suite."""
    print("=" * 64)
    print("  Alpine Linux v3.20 on GPU — Metal Compute Shader Demo")
    print("=" * 64)
    print()

    fs = create_alpine_rootfs()

    demos = [
        # Identity
        ("System Identity", [
            (["cat", "/etc/alpine-release"], "Alpine version"),
            (["cat", "/etc/hostname"], "Hostname"),
            (["uname", "-a"], "Kernel info"),
        ]),
        # Core utils
        ("Core Utilities", [
            (["echo", "Hello from Alpine on GPU!"], "echo"),
            (["basename", "/usr/local/bin/busybox"], "basename"),
            (["dirname", "/usr/local/bin/busybox"], "dirname"),
            (["true"], "true (exit 0)"),
            (["false"], "false (exit 1)"),
        ]),
        # Filesystem
        ("Filesystem Operations", [
            (["cat", "/etc/passwd"], "cat /etc/passwd"),
            (["cat", "/etc/os-release"], "cat /etc/os-release"),
            (["cat", "/proc/cpuinfo"], "cat /proc/cpuinfo"),
            (["cat", "/proc/meminfo"], "cat /proc/meminfo"),
        ]),
        # Text processing (depend on LDR literal working)
        ("Text Processing", [
            (["cat", "/tmp/data.txt"], "cat data"),
            (["wc", "-l", "/tmp/data.txt"], "wc -l"),
        ]),
    ]

    total_time = 0
    total_passed = 0
    total_tests = 0

    for section_name, tests in demos:
        print(f"  --- {section_name} ---")
        for argv, label in tests:
            total_tests += 1
            sys.stdout.write(f"  {label:30s} → ")
            sys.stdout.flush()

            t = time.perf_counter()
            try:
                output, results = run_and_capture(argv, fs)
                dt = time.perf_counter() - t
                total_time += dt
                cycles = results["total_cycles"]

                # Show first line of output (trimmed)
                first_line = output.strip().split("\n")[0][:50] if output.strip() else "(no output)"
                print(f"{first_line:50s}  ({cycles:>10,} cyc, {dt:.1f}s)")
                total_passed += 1
            except Exception as e:
                dt = time.perf_counter() - t
                total_time += dt
                print(f"ERROR: {e}")
        print()

    # Summary
    binary_size = Path(BUSYBOX).stat().st_size if Path(BUSYBOX).exists() else 0
    print(f"  {total_passed}/{total_tests} commands executed")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Binary: {binary_size:,} bytes ({binary_size // 1024} KB)")
    print(f"  Architecture: aarch64, musl libc, static")
    print(f"  Filesystem: {len(fs.files)} files, {len(fs.directories)} dirs")
    print(f"  Distro: Alpine Linux v3.20")
    print("=" * 64)

    return total_passed, total_tests


if __name__ == "__main__":
    if not Path(BUSYBOX).exists():
        print(f"Error: BusyBox ELF not found at {BUSYBOX}")
        print("Cross-compile with: aarch64-linux-musl-gcc -static -mgeneral-regs-only")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_suite()
    elif len(sys.argv) > 1:
        # Run single command
        fs = create_alpine_rootfs()
        run_command(sys.argv[1:], fs, quiet=True)
    else:
        interactive_mode()
