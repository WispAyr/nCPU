#!/usr/bin/env python3
"""Demo: Tensor-based nCPU running ARM64 machine code on GPU.

Shows the tensor-native nCPU executing real ARM64 binaries with all
state (registers, memory, PC, flags) living as GPU tensors.

Requires: torch
"""

import struct
from pathlib import Path
import sys

# Allow running this file directly from the repo without pip install -e .
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def demo_hello_world():
    """Run ARM64 Hello World via tensor-native kernel with syscall handling."""
    print("=" * 60)
    print("DEMO: ARM64 Hello World (tensor-native kernel)")
    print("=" * 60)

    from ncpu.tensor import TensorKernel, get_device
    from ncpu.tensor.kernel import create_hello_world

    print(f"  Device: {get_device()}")

    kernel = TensorKernel(mem_size=1 * 1024 * 1024)
    binary = create_hello_world()
    kernel.load_binary(binary)
    kernel.setup_stack()

    result = kernel.run(max_instructions=1000, batch_size=8)

    print(f"  Output: {result.output.strip()}")
    print(f"  Instructions: {result.instructions}")
    print(f"  Time: {result.time_seconds*1000:.2f}ms")
    print(f"  Syscalls: {result.syscalls_handled}")
    print(f"  Exit code: {result.exit_code}")
    passed = "Hello, World!" in result.output
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()


def demo_alu_benchmark():
    """Benchmark straight-line ALU throughput."""
    print("=" * 60)
    print("DEMO: ALU throughput benchmark")
    print("=" * 60)

    import torch
    from ncpu.tensor import TensorCPU, get_device

    device = get_device()
    print(f"  Device: {device}")

    cpu = TensorCPU(mem_size=1 * 1024 * 1024)

    # Generate 1000 ADD instructions + SVC
    code = []
    for i in range(1000):
        rd = (i % 30) + 1
        imm = i % 4096
        inst = 0x91000000 | rd | (imm << 10)
        code.extend([inst & 0xFF, (inst >> 8) & 0xFF, (inst >> 16) & 0xFF, (inst >> 24) & 0xFF])
    code.extend([0x01, 0x00, 0x00, 0xD4])  # SVC

    cpu.memory[:len(code)] = torch.tensor(code, dtype=torch.uint8, device=device)
    cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
    cpu.regs.zero_()

    # Step-by-step baseline
    stats_step = cpu.run(max_instructions=100)
    print(f"  Step-by-step (100 inst): {stats_step.ips:,.0f} IPS")

    # Batch execution
    cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
    cpu.regs.zero_()
    cpu.halted = False

    for batch_size in [64, 256]:
        cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
        cpu.regs.zero_()
        cpu.halted = False
        stats = cpu.run_batch(max_instructions=1010, batch_size=batch_size)
        print(f"  Batch-{batch_size}: {stats.instructions_executed:,} inst, "
              f"{stats.ips:,.0f} IPS, {stats.time_seconds*1000:.1f}ms")

    print()


def demo_loop():
    """Run a counted loop in ARM64."""
    print("=" * 60)
    print("DEMO: ARM64 counted loop (100 iterations)")
    print("=" * 60)

    import torch
    from ncpu.tensor import TensorCPU, get_device

    device = get_device()
    cpu = TensorCPU(mem_size=1 * 1024 * 1024)

    loop_count = 100
    mov_inst = 0xD2800000 | (loop_count << 5)  # MOV X0, #100

    program = bytes([
        mov_inst & 0xFF, (mov_inst >> 8) & 0xFF,
        (mov_inst >> 16) & 0xFF, (mov_inst >> 24) & 0xFF,
        0x00, 0x04, 0x00, 0xD1,  # SUB X0, X0, #1
        0xC0, 0xFF, 0xFF, 0xB5,  # CBNZ X0, -4
        0x01, 0x00, 0x00, 0xD4,  # SVC #0
    ])

    cpu.memory[:len(program)] = torch.tensor(list(program), dtype=torch.uint8, device=device)
    cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
    cpu.regs.zero_()

    stats = cpu.run(max_instructions=loop_count * 3 + 10)
    print(f"  Instructions: {stats.instructions_executed:,}")
    print(f"  Branches taken: {stats.branches_taken}")
    print(f"  Time: {stats.time_seconds*1000:.1f}ms")
    print(f"  IPS: {stats.ips:,.0f}")
    print()


if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required for tensor CPU demos.")
        print("Install with: pip install torch")
        exit(1)

    demo_hello_world()
    demo_alu_benchmark()
    demo_loop()
