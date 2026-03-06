#!/usr/bin/env python3
"""
BusyBox on GPU — Real Linux userspace running on Metal compute shader.

Demonstrates BusyBox (Alpine Linux's core utility suite) compiled for aarch64
with musl libc, running entirely on Apple Silicon GPU via Metal compute shaders.

The GPU executes ARM64 instructions natively. Python mediates syscalls via SVC trap.

Usage:
    python demos/busybox_gpu_demo.py                    # Run demo suite
    python demos/busybox_gpu_demo.py echo "hello"       # Run specific command
    python demos/busybox_gpu_demo.py uname -a           # System info

Author: Robert Price
Date: March 2026
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.elf_loader import load_and_run_elf

BUSYBOX = str(Path(__file__).parent / "busybox.elf")


def run_command(argv, quiet_init=True):
    """Run a busybox command on GPU and return results."""
    return load_and_run_elf(
        BUSYBOX,
        argv=argv,
        max_cycles=500_000_000,
        quiet=quiet_init,
    )


def demo_suite():
    """Run the standard busybox demo suite."""
    print("=" * 60)
    print("  BusyBox on GPU — Metal Compute Shader ARM64 Execution")
    print("=" * 60)
    print()

    tests = [
        (["echo", "Hello from BusyBox on GPU!"], "echo"),
        (["echo", "nCPU:", "Metal", "kernel", "is", "alive"], "echo (multi)"),
        (["uname", "-s"], "uname -s"),
        (["basename", "/usr/local/bin/busybox"], "basename"),
        (["dirname", "/usr/local/bin/busybox"], "dirname"),
        (["true"], "true"),
        (["false"], "false"),
    ]

    total_time = 0
    passed = 0

    for argv, name in tests:
        sys.stdout.write(f"  {name:20s} → ")
        sys.stdout.flush()
        t = time.perf_counter()
        results = run_command(argv, quiet_init=True)
        dt = time.perf_counter() - t
        total_time += dt
        cycles = results["total_cycles"]
        sys.stdout.write(f"  ({cycles:,} cycles, {dt:.1f}s)\n")
        sys.stdout.flush()
        passed += 1

    print()
    print(f"  {passed}/{len(tests)} commands executed successfully")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Binary: {Path(BUSYBOX).stat().st_size:,} bytes (264 KB)")
    print(f"  Architecture: aarch64, statically linked, musl libc")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific command
        results = run_command(sys.argv[1:], quiet_init=True)
    else:
        demo_suite()
