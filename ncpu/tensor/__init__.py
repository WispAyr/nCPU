"""Tensor-based nCPU: GPU-native ARM64 emulation.

All execution happens on tensors -- registers, memory, PC, and flags
are PyTorch tensors. Instruction fetch, decode, and execute use vectorized
tensor operations for maximum throughput.

Requires: torch

Quick start:
    from ncpu.tensor import TensorCPU
    cpu = TensorCPU()
    # Load ARM64 machine code into cpu.memory, set cpu.pc
    stats = cpu.run_batch(max_instructions=1000)
"""

from .cpu import TensorCPU, ExecutionStats, get_device
from .kernel import TensorKernel, RunResult

__all__ = [
    "TensorCPU", "ExecutionStats", "get_device",
    "TensorKernel", "RunResult",
]
