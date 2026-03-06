"""
MLX Metal Kernel for ARM64 CPU Emulation.

This package provides a custom Metal GPU kernel for high-performance ARM64
CPU emulation on Apple Silicon. It eliminates the GPU-CPU synchronization
bottleneck that limits PyTorch-based execution.

QUICK START:
============

    from mlx_kernel import MLXKernelCPU, StopReason

    # Create CPU
    cpu = MLXKernelCPU(memory_size=4*1024*1024)

    # Load program
    program = [0xD2800000, 0x91000400, 0xD4400000]  # MOVZ, ADD, HLT
    cpu.load_program(program, address=0)
    cpu.set_pc(0)

    # Execute
    result = cpu.execute(max_cycles=100000)
    print(f"Executed {result.cycles} instructions at {result.ips:,.0f} IPS")

PERFORMANCE TARGET:
===================

10M-100M+ IPS on Apple Silicon (vs ~120K IPS with PyTorch batched execution)

PACKAGE CONTENTS:
=================

- MLXKernelCPU: Main CPU emulator class
- StopReason: Enum for execution stop reasons
- ExecutionResult: Dataclass for execution results
- create_cpu(): Convenience function to create CPU instance
- KERNEL_HEADER, KERNEL_SOURCE: Raw Metal shader source code
- NCPUComputeKernel: nCPU ISA compute kernel (qemu-style GPU execution)
- ComputeResult: Dataclass for compute kernel results

Author: KVRM Project
Date: 2024
"""

from .cpu_kernel import (
    MLXKernelCPU,
    StopReason,
    ExecutionResult,
    create_cpu,
)

from .cpu_kernel_source import (
    KERNEL_HEADER,
    KERNEL_SOURCE,
    STOP_RUNNING,
    STOP_HALT,
    STOP_SYSCALL,
    STOP_MAX_CYCLES,
    get_kernel_source,
    get_full_kernel_source,
)

from .cpu_kernel_v2 import (
    MLXKernelCPUv2,
    StopReasonV2,
    ExecutionResultV2,
    create_cpu_v2,
)

from .ncpu_kernel import (
    NCPUComputeKernel,
    ComputeResult,
)

from .ncpu_kernel_source import (
    NCPU_KERNEL_HEADER,
    NCPU_KERNEL_SOURCE,
    NCPU_STOP_RUNNING,
    NCPU_STOP_HALT,
    NCPU_STOP_MAX_CYCLES,
)

from .muxleq_kernel import (
    MuxleqVM,
    MuxleqResult,
)

from .muxleq_kernel_source import (
    MUXLEQ_KERNEL_HEADER,
    MUXLEQ_KERNEL_SOURCE,
    MUXLEQ_STOP_RUNNING,
    MUXLEQ_STOP_HALT,
    MUXLEQ_STOP_MAX_CYCLES,
    MUXLEQ_STOP_IO_READ,
    MUXLEQ_STOP_IO_WRITE,
)

__all__ = [
    # ARM64 kernel classes
    'MLXKernelCPU',
    'StopReason',
    'ExecutionResult',
    'create_cpu',
    # ARM64 kernel source
    'KERNEL_HEADER',
    'KERNEL_SOURCE',
    'STOP_RUNNING',
    'STOP_HALT',
    'STOP_SYSCALL',
    'STOP_MAX_CYCLES',
    'get_kernel_source',
    'get_full_kernel_source',
    # ARM64 kernel V2 (125-instruction, double-buffer memory)
    'MLXKernelCPUv2',
    'StopReasonV2',
    'ExecutionResultV2',
    'create_cpu_v2',
    # nCPU ISA compute kernel
    'NCPUComputeKernel',
    'ComputeResult',
    'NCPU_KERNEL_HEADER',
    'NCPU_KERNEL_SOURCE',
    'NCPU_STOP_RUNNING',
    'NCPU_STOP_HALT',
    'NCPU_STOP_MAX_CYCLES',
    # MUXLEQ VM
    'MuxleqVM',
    'MuxleqResult',
    'MUXLEQ_KERNEL_HEADER',
    'MUXLEQ_KERNEL_SOURCE',
    'MUXLEQ_STOP_RUNNING',
    'MUXLEQ_STOP_HALT',
    'MUXLEQ_STOP_MAX_CYCLES',
    'MUXLEQ_STOP_IO_READ',
    'MUXLEQ_STOP_IO_WRITE',
]

__version__ = '1.1.0'
