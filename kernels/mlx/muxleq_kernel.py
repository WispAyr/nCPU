"""MUXLEQ VM — Three execution modes on nCPU.

A MUXLEQ virtual machine (2-instruction Turing-complete computer) that
runs in three modes:

  1. compute — Metal GPU shader, millions of IPS potential
  2. fast    — native Python integer ops, ~100K+ IPS
  3. neural  — every SUB through arithmetic.pt (Kogge-Stone CLA),
               every MUX through logical.pt (neural truth tables),
               100% exact neural arithmetic running eForth

MUXLEQ is the minimal proof that neural networks can execute a Turing-
complete computer exactly: if they handle 130+ ARM64 instructions, they
certainly handle 2.

Usage:
    from kernels.mlx.muxleq_kernel import MuxleqVM

    vm = MuxleqVM(mode="fast")
    vm.load_dec("muxleq.dec")
    vm.run()  # interactive eForth REPL
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .muxleq_kernel_source import (
    MUXLEQ_KERNEL_HEADER,
    MUXLEQ_KERNEL_SOURCE,
    MUXLEQ_STOP_RUNNING,
    MUXLEQ_STOP_HALT,
    MUXLEQ_STOP_MAX_CYCLES,
    MUXLEQ_STOP_IO_READ,
    MUXLEQ_STOP_IO_WRITE,
)

# Constants
MEMORY_SIZE = 65536
WORD_MASK = 0xFFFF
SENTINEL = 0xFFFF
HALT_THRESHOLD = 32768
MUX_FLAG = 0x8000
MUX_ADDR_MASK = 0x7FFF


@dataclass
class MuxleqResult:
    """Result from a MUXLEQ execution run."""
    cycles: int
    elapsed_seconds: float
    stop_reason: int
    final_pc: int

    @property
    def ips(self) -> float:
        if self.elapsed_seconds > 0:
            return self.cycles / self.elapsed_seconds
        return 0.0

    @property
    def stop_reason_name(self) -> str:
        return {
            MUXLEQ_STOP_RUNNING: "RUNNING",
            MUXLEQ_STOP_HALT: "HALT",
            MUXLEQ_STOP_MAX_CYCLES: "MAX_CYCLES",
            MUXLEQ_STOP_IO_READ: "IO_READ",
            MUXLEQ_STOP_IO_WRITE: "IO_WRITE",
        }.get(self.stop_reason, "UNKNOWN")


class MuxleqVM:
    """MUXLEQ virtual machine with three execution modes.

    Modes:
        "compute" — Metal GPU shader (fastest, requires MLX)
        "fast"    — native Python integer ops
        "neural"  — neural network arithmetic via NeuralOps
    """

    def __init__(self, mode: str = "fast"):
        if mode not in ("compute", "fast", "neural"):
            raise ValueError(f"Mode must be 'compute', 'fast', or 'neural', got '{mode}'")
        self.mode = mode
        self.memory = np.zeros(MEMORY_SIZE, dtype=np.uint16)
        self.pc = 0
        self.total_cycles = 0

        # Neural ops (lazy loaded)
        self._neural_ops = None

        # Metal kernel (lazy compiled)
        self._kernel = None

    def load_dec(self, path: str) -> int:
        """Load a .dec file (comma-separated signed decimals) into memory.

        Args:
            path: Path to .dec file

        Returns:
            Number of words loaded
        """
        data = Path(path).read_text()
        words = []
        for token in data.replace(",", " ").split():
            token = token.strip()
            if not token:
                continue
            val = int(token)
            words.append(val & WORD_MASK)

        for i, w in enumerate(words):
            if i >= MEMORY_SIZE:
                break
            self.memory[i] = w

        self.pc = 0
        self.total_cycles = 0
        return len(words)

    def load_program(self, words: list[int]) -> None:
        """Load program from a list of 16-bit words."""
        self.memory[:] = 0
        for i, w in enumerate(words):
            if i >= MEMORY_SIZE:
                break
            self.memory[i] = w & WORD_MASK
        self.pc = 0
        self.total_cycles = 0

    def _ensure_neural_ops(self):
        """Lazy-load neural models for neural mode."""
        if self._neural_ops is not None:
            return
        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from ncpu.model.neural_ops import NeuralOps
        self._neural_ops = NeuralOps(models_dir=str(project_root / "models"))
        avail = self._neural_ops.load()
        loaded = [k for k, v in avail.items() if v]
        print(f"[neural] Loaded models: {', '.join(loaded) if loaded else 'none (fallback to Python)'}")

    def _ensure_compute_kernel(self):
        """Lazy-compile Metal kernel for compute mode."""
        if self._kernel is not None:
            return
        import mlx.core as mx
        self._kernel = mx.fast.metal_kernel(
            name="muxleq_execute",
            input_names=[
                "memory_in", "pc_in", "max_cycles_in",
            ],
            output_names=[
                "memory_out", "pc_out", "cycles_out",
                "stop_reason_out", "io_out",
            ],
            source=MUXLEQ_KERNEL_SOURCE,
            header=MUXLEQ_KERNEL_HEADER,
            ensure_row_contiguous=True,
            atomic_outputs=False,
        )

    # ─── Fast mode execution ───

    def _step_fast(self, max_cycles: int = 100_000,
                   input_fn=None, output_fn=None) -> MuxleqResult:
        """Execute in fast mode (native Python ops)."""
        m = self.memory
        pc = self.pc
        cycles = 0
        start = time.perf_counter()

        while cycles < max_cycles:
            if pc >= HALT_THRESHOLD:
                elapsed = time.perf_counter() - start
                self.pc = pc
                self.total_cycles += cycles
                return MuxleqResult(cycles, elapsed, MUXLEQ_STOP_HALT, pc)

            a, b, c = int(m[pc]), int(m[pc + 1]), int(m[pc + 2])
            pc += 3

            if a == SENTINEL:
                # INPUT
                if input_fn:
                    byte_val = input_fn()
                else:
                    ch = sys.stdin.buffer.read(1)
                    byte_val = ch[0] if ch else SENTINEL
                m[b] = byte_val & WORD_MASK
            elif b == SENTINEL:
                # OUTPUT
                if output_fn:
                    output_fn(int(m[a]) & 0xFF)
                else:
                    sys.stdout.buffer.write(bytes([int(m[a]) & 0xFF]))
                    sys.stdout.buffer.flush()
            elif (c & MUX_FLAG) and c != SENTINEL:
                # MUX
                mask_addr = c & MUX_ADDR_MASK
                mask = int(m[mask_addr])
                m[b] = ((int(m[a]) & ~mask) | (int(m[b]) & mask)) & WORD_MASK
            else:
                # SUBLEQ
                r = (int(m[b]) - int(m[a])) & WORD_MASK
                m[b] = r
                if r == 0 or (r & MUX_FLAG):
                    pc = c

            cycles += 1

        elapsed = time.perf_counter() - start
        self.pc = pc
        self.total_cycles += cycles
        return MuxleqResult(cycles, elapsed, MUXLEQ_STOP_MAX_CYCLES, pc)

    # ─── Neural mode execution ───

    def _step_neural(self, max_cycles: int = 100_000,
                     input_fn=None, output_fn=None) -> MuxleqResult:
        """Execute in neural mode — SUB via CLA, MUX via neural logic."""
        self._ensure_neural_ops()
        ops = self._neural_ops

        m = self.memory
        pc = self.pc
        cycles = 0
        start = time.perf_counter()

        while cycles < max_cycles:
            if pc >= HALT_THRESHOLD:
                elapsed = time.perf_counter() - start
                self.pc = pc
                self.total_cycles += cycles
                return MuxleqResult(cycles, elapsed, MUXLEQ_STOP_HALT, pc)

            a, b, c = int(m[pc]), int(m[pc + 1]), int(m[pc + 2])
            pc += 3

            if a == SENTINEL:
                # INPUT
                if input_fn:
                    byte_val = input_fn()
                else:
                    ch = sys.stdin.buffer.read(1)
                    byte_val = ch[0] if ch else SENTINEL
                m[b] = byte_val & WORD_MASK
            elif b == SENTINEL:
                # OUTPUT
                if output_fn:
                    output_fn(int(m[a]) & 0xFF)
                else:
                    sys.stdout.buffer.write(bytes([int(m[a]) & 0xFF]))
                    sys.stdout.buffer.flush()
            elif (c & MUX_FLAG) and c != SENTINEL:
                # MUX — neural AND, OR, NOT (bitwise)
                mask_addr = c & MUX_ADDR_MASK
                mask = int(m[mask_addr])
                val_a = int(m[a])
                val_b = int(m[b])

                # m[b] = (m[a] & ~mask) | (m[b] & mask)
                not_mask = (~mask) & WORD_MASK
                part1 = ops.neural_and(val_a, not_mask) & WORD_MASK
                part2 = ops.neural_and(val_b, mask) & WORD_MASK
                m[b] = ops.neural_or(part1, part2) & WORD_MASK
            else:
                # SUBLEQ — neural subtraction via CLA
                val_b = int(m[b])
                val_a = int(m[a])
                r = ops.neural_sub(val_b, val_a) & WORD_MASK
                m[b] = r
                if r == 0 or (r & MUX_FLAG):
                    pc = c

            cycles += 1

        elapsed = time.perf_counter() - start
        self.pc = pc
        self.total_cycles += cycles
        return MuxleqResult(cycles, elapsed, MUXLEQ_STOP_MAX_CYCLES, pc)

    # ─── Compute mode execution ───

    def _step_compute(self, max_cycles: int = 1_000_000,
                      input_fn=None, output_fn=None) -> MuxleqResult:
        """Execute in compute mode — Metal GPU shader."""
        import mlx.core as mx
        self._ensure_compute_kernel()

        start = time.perf_counter()

        # Prepare inputs
        memory_mx = mx.array(self.memory.astype(np.uint16), dtype=mx.uint16)
        pc_mx = mx.array([self.pc], dtype=mx.int64)
        max_cycles_mx = mx.array([max_cycles], dtype=mx.uint32)

        # Execute
        outputs = self._kernel(
            inputs=[memory_mx, pc_mx, max_cycles_mx],
            output_shapes=[
                (MEMORY_SIZE,),  # memory_out
                (1,),            # pc_out
                (1,),            # cycles_out
                (1,),            # stop_reason_out
                (2,),            # io_out [addr, data]
            ],
            output_dtypes=[
                mx.uint16, mx.int64, mx.uint32, mx.uint8, mx.uint32,
            ],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            verbose=False,
        )
        mx.eval(outputs)

        memory_out, pc_out, cycles_out, stop_reason_out, io_out = outputs

        # Update state
        self.memory = np.array(memory_out, dtype=np.uint16)
        self.pc = int(pc_out[0].item())
        cycles = int(cycles_out[0].item())
        stop_reason = int(stop_reason_out[0].item())
        io_addr = int(io_out[0].item())
        io_data = int(io_out[1].item())

        self.total_cycles += cycles
        elapsed = time.perf_counter() - start

        # Handle I/O traps
        if stop_reason == MUXLEQ_STOP_IO_READ:
            if input_fn:
                byte_val = input_fn()
            else:
                ch = sys.stdin.buffer.read(1)
                byte_val = ch[0] if ch else SENTINEL
            self.memory[io_addr] = byte_val & WORD_MASK
            # Advance PC past the instruction (it was rewound)
            self.pc += 3
            return MuxleqResult(cycles, elapsed, MUXLEQ_STOP_IO_READ, self.pc)

        elif stop_reason == MUXLEQ_STOP_IO_WRITE:
            if output_fn:
                output_fn(io_data & 0xFF)
            else:
                sys.stdout.buffer.write(bytes([io_data & 0xFF]))
                sys.stdout.buffer.flush()
            return MuxleqResult(cycles, elapsed, MUXLEQ_STOP_IO_WRITE, self.pc)

        return MuxleqResult(cycles, elapsed, stop_reason, self.pc)

    # ─── Unified interface ───

    def step(self, max_cycles: int = 100_000,
             input_fn=None, output_fn=None) -> MuxleqResult:
        """Execute up to max_cycles instructions (or until halt/IO).

        Args:
            max_cycles: Maximum instructions before pausing
            input_fn: Optional callable returning int (byte value) for input
            output_fn: Optional callable(int) for output bytes

        Returns:
            MuxleqResult with cycles, timing, and stop reason
        """
        if self.mode == "compute":
            return self._step_compute(max_cycles, input_fn, output_fn)
        elif self.mode == "neural":
            return self._step_neural(max_cycles, input_fn, output_fn)
        else:
            return self._step_fast(max_cycles, input_fn, output_fn)

    def run(self, max_total_cycles: int = 100_000_000) -> MuxleqResult:
        """Run until halt, dispatching I/O as needed.

        This is the main entry point for interactive use (e.g. eForth REPL).
        The VM runs in bursts, pausing for I/O, until it halts or hits the
        total cycle limit.
        """
        batch = 1_000_000 if self.mode == "compute" else 100_000
        total_start = time.perf_counter()

        while self.total_cycles < max_total_cycles:
            result = self.step(max_cycles=batch)

            if result.stop_reason == MUXLEQ_STOP_HALT:
                total_elapsed = time.perf_counter() - total_start
                return MuxleqResult(
                    self.total_cycles, total_elapsed,
                    MUXLEQ_STOP_HALT, result.final_pc,
                )
            elif result.stop_reason == MUXLEQ_STOP_MAX_CYCLES:
                # Just continue — no I/O needed
                continue
            # IO_READ / IO_WRITE — already handled in step(), continue
            continue

        total_elapsed = time.perf_counter() - total_start
        return MuxleqResult(
            self.total_cycles, total_elapsed,
            MUXLEQ_STOP_MAX_CYCLES, self.pc,
        )

    def read_memory(self, addr: int) -> int:
        """Read a 16-bit word from memory."""
        return int(self.memory[addr & WORD_MASK])

    def write_memory(self, addr: int, value: int) -> None:
        """Write a 16-bit word to memory."""
        self.memory[addr & WORD_MASK] = value & WORD_MASK

    def reset(self) -> None:
        """Reset VM state."""
        self.memory[:] = 0
        self.pc = 0
        self.total_cycles = 0


# ─── CLI entry point ───

def main():
    """Run MUXLEQ VM from command line.

    Usage:
        python -m kernels.mlx.muxleq_kernel [--mode fast|neural|compute] file.dec [file2.dec ...]
    """
    import argparse
    parser = argparse.ArgumentParser(description="MUXLEQ VM on nCPU")
    parser.add_argument("files", nargs="+", help=".dec files to load")
    parser.add_argument("--mode", choices=["fast", "neural", "compute"],
                        default="fast", help="Execution mode")
    parser.add_argument("--max-cycles", type=int, default=100_000_000,
                        help="Maximum total cycles")
    args = parser.parse_args()

    vm = MuxleqVM(mode=args.mode)

    # Load all .dec files contiguously (like the C implementation)
    offset = 0
    for f in args.files:
        data = Path(f).read_text()
        for token in data.replace(",", " ").split():
            token = token.strip()
            if not token:
                continue
            val = int(token)
            vm.memory[offset] = val & WORD_MASK
            offset += 1

    print(f"[muxleq] Loaded {offset} words, mode={args.mode}", file=sys.stderr)

    result = vm.run(max_total_cycles=args.max_cycles)

    print(f"\n[muxleq] {result.stop_reason_name} after {result.cycles:,} cycles "
          f"({result.elapsed_seconds:.2f}s, {result.ips:,.0f} IPS)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
