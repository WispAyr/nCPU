"""nCPU ISA Compute Kernel — Python Wrapper.

Provides NCPUComputeKernel: a qemu-style fetch-decode-execute emulator
that runs nCPU ISA programs entirely on the GPU via Metal compute shaders.

The GPU's native ALU does the actual arithmetic — no neural nets, no tensor
ops. Programs that run through neural networks at ~5K IPS instead run through
raw GPU ALU at millions of IPS.

Usage:
    from kernels.mlx.ncpu_kernel import NCPUComputeKernel

    kernel = NCPUComputeKernel()
    kernel.load_program_from_asm('''
        MOV R0, 7
        MOV R1, 6
        MUL R2, R0, R1
        HALT
    ''')
    result = kernel.execute()
    print(kernel.get_register(2))  # 42
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from .ncpu_kernel_source import (
    NCPU_KERNEL_HEADER,
    NCPU_KERNEL_SOURCE,
    NCPU_STOP_RUNNING,
    NCPU_STOP_HALT,
    NCPU_STOP_MAX_CYCLES,
)


@dataclass
class ComputeResult:
    """Result of compute kernel execution."""
    cycles: int
    elapsed_seconds: float
    stop_reason: int  # 0=running, 1=halt, 2=max_cycles
    final_pc: int

    @property
    def ips(self) -> float:
        """Instructions per second."""
        if self.elapsed_seconds > 0:
            return self.cycles / self.elapsed_seconds
        return 0.0

    @property
    def stop_reason_name(self) -> str:
        return {
            NCPU_STOP_RUNNING: "RUNNING",
            NCPU_STOP_HALT: "HALT",
            NCPU_STOP_MAX_CYCLES: "MAX_CYCLES",
        }.get(self.stop_reason, "UNKNOWN")


class NCPUComputeKernel:
    """nCPU ISA compute kernel — runs programs on GPU via Metal shaders.

    This is a qemu-style fetch-decode-execute interpreter running entirely
    as a Metal compute shader. The GPU's silicon does native integer
    add/sub/mul/shift — no neural nets, no tensor ops.

    Attributes:
        registers: 8 int64 registers (R0-R7)
        pc: Program counter (word-addressed)
        flags: [ZF, SF] as float32
        program: List of 32-bit instruction words
    """

    def __init__(self):
        # Register file: 8 int64 registers
        self._registers = np.zeros(8, dtype=np.int64)
        self._pc = 0
        self._flags = np.array([0.0, 0.0], dtype=np.float32)  # [ZF, SF]
        self._program: Optional[np.ndarray] = None

        # Compile Metal kernel
        self._kernel = mx.fast.metal_kernel(
            name="ncpu_isa_execute",
            input_names=[
                "program_in", "registers_in", "pc_in",
                "flags_in", "max_cycles_in", "program_size_in",
            ],
            output_names=[
                "registers_out", "pc_out", "flags_out",
                "cycles_out", "stop_reason_out",
            ],
            source=NCPU_KERNEL_SOURCE,
            header=NCPU_KERNEL_HEADER,
            ensure_row_contiguous=True,
            atomic_outputs=False,
        )

    def load_program_from_asm(self, source: str) -> None:
        """Assemble nCPU ISA source and load into kernel.

        Uses the ClassicalAssembler from neurOS to assemble source into
        32-bit binary words, then loads them for GPU execution.

        Args:
            source: Assembly source code (nCPU ISA)
        """
        # Add project root to path for imports
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from ncpu.os.assembler import ClassicalAssembler

        asm = ClassicalAssembler()
        result = asm.assemble(source)

        if not result.success:
            raise ValueError(f"Assembly failed: {result.errors}")

        self.load_program(result.binary)

    def load_program(self, binary: Union[list[int], np.ndarray]) -> None:
        """Load pre-assembled binary instruction words.

        Args:
            binary: List of 32-bit instruction words
        """
        if isinstance(binary, np.ndarray):
            self._program = binary.astype(np.int64)
        else:
            self._program = np.array(binary, dtype=np.int64)
        self._pc = 0

    def execute(self, max_cycles: int = 1_000_000) -> ComputeResult:
        """Execute loaded program on GPU via Metal compute shader.

        Args:
            max_cycles: Maximum instructions to execute

        Returns:
            ComputeResult with cycles, elapsed time, and stop reason
        """
        if self._program is None:
            raise RuntimeError("No program loaded")

        start_time = time.perf_counter()

        # Prepare MLX inputs
        program_mx = mx.array(self._program, dtype=mx.int64)
        registers_mx = mx.array(self._registers, dtype=mx.int64)
        pc_mx = mx.array([self._pc], dtype=mx.int64)
        flags_mx = mx.array(self._flags, dtype=mx.float32)
        max_cycles_mx = mx.array([max_cycles], dtype=mx.uint32)
        program_size_mx = mx.array([len(self._program)], dtype=mx.uint32)

        # Execute Metal kernel
        outputs = self._kernel(
            inputs=[
                program_mx, registers_mx, pc_mx,
                flags_mx, max_cycles_mx, program_size_mx,
            ],
            output_shapes=[(8,), (1,), (2,), (1,), (1,)],
            output_dtypes=[mx.int64, mx.int64, mx.float32, mx.uint32, mx.uint8],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            verbose=False,
        )

        # Force evaluation
        mx.eval(outputs)

        registers_out, pc_out, flags_out, cycles_out, stop_reason_out = outputs

        # Update state
        self._registers = np.array(registers_out)
        self._pc = int(pc_out[0].item())
        self._flags = np.array(flags_out)
        cycles = int(cycles_out[0].item())
        stop_reason = int(stop_reason_out[0].item())

        elapsed = time.perf_counter() - start_time

        return ComputeResult(
            cycles=cycles,
            elapsed_seconds=elapsed,
            stop_reason=stop_reason,
            final_pc=self._pc,
        )

    def get_register(self, idx: int) -> int:
        """Get value of register Rn.

        Args:
            idx: Register index (0-7)

        Returns:
            Register value as Python int
        """
        if not 0 <= idx <= 7:
            raise IndexError(f"Register index must be 0-7, got {idx}")
        return int(self._registers[idx])

    def get_registers_dict(self) -> dict[str, int]:
        """Get all registers as {R0: val, R1: val, ...} dict."""
        return {f"R{i}": int(self._registers[i]) for i in range(8)}

    def get_flags(self) -> dict[str, bool]:
        """Get flags as {ZF: bool, SF: bool} dict."""
        return {
            "ZF": bool(self._flags[0] > 0.5),
            "SF": bool(self._flags[1] > 0.5),
        }

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._registers = np.zeros(8, dtype=np.int64)
        self._pc = 0
        self._flags = np.array([0.0, 0.0], dtype=np.float32)
        self._program = None
