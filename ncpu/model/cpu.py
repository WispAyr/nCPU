"""CPU: Main orchestrator for the model-based nCPU.

Execution pipeline:
    MEMORY -> FETCH -> DECODE (LLM or regex) -> KEY -> REGISTRY -> EXECUTE -> STATE

Execution backends:
    - mock:   Regex decode + Python arithmetic (fast, no GPU)
    - real:   LLM decode + Python arithmetic (requires torch)
    - neural: Regex decode + trained neural models for ALU (requires torch + .pt models)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from .state import CPUState, create_initial_state
from .registry import CPURegistry, get_registry
from .decode import Decoder, DecodeResult, parse_program


@dataclass
class ExecutionTraceEntry:
    """Single entry in the execution trace."""
    cycle: int
    instruction: str
    decode_result: DecodeResult
    pre_state: dict
    post_state: dict
    error: Optional[str] = None


class CPU:
    """Model-based nCPU with semantic instruction decode.

    Args:
        mock_mode: Use rule-based decoder (True) or trained LLM (False)
        model_path: Path to trained decode model (required if mock_mode=False)
        max_cycles: Maximum cycles before forced halt (safety limit)
        neural_execution: Use trained neural models for ALU ops (default: False)
        models_dir: Path to directory containing trained .pt models
    """

    DEFAULT_MAX_CYCLES = 10000

    def __init__(
        self,
        mock_mode: bool = True,
        model_path: Optional[str] = None,
        max_cycles: int = DEFAULT_MAX_CYCLES,
        neural_execution: bool = False,
        models_dir: str = "models",
    ):
        self.decoder = Decoder(mock_mode=mock_mode, model_path=model_path)
        self.neural_execution = neural_execution
        self._neural_registry = None

        if neural_execution:
            from .neural_ops import NeuralRegistry
            self._neural_registry = NeuralRegistry(models_dir=models_dir)
            available = self._neural_registry.load()
            self.registry = self._neural_registry
        else:
            self.registry = get_registry()

        self.state: Optional[CPUState] = None
        self.trace: List[ExecutionTraceEntry] = []
        self.max_cycles = max_cycles
        self._labels: Dict[str, int] = {}

    def load(self) -> None:
        """Load the decoder model (only needed for real mode)."""
        self.decoder.load()

    def unload(self) -> None:
        """Unload the decoder model to free memory."""
        self.decoder.unload()

    def load_program(self, source: str) -> None:
        """Load an assembly program from source code."""
        instructions, labels = parse_program(source)
        self._labels = labels
        self.decoder.set_labels(labels)
        self.state = create_initial_state(instructions)
        self.trace = []

    def load_instructions(self, instructions: List[str], labels: Optional[Dict[str, int]] = None) -> None:
        """Load a program from a pre-parsed instruction list."""
        self._labels = labels or {}
        self.decoder.set_labels(self._labels)
        self.state = create_initial_state(instructions)
        self.trace = []

    def step(self) -> ExecutionTraceEntry:
        """Execute a single fetch-decode-execute cycle."""
        if self.state is None:
            raise RuntimeError("No program loaded")
        if self.state.halted:
            raise RuntimeError("CPU is halted")
        if self.state.cycle_count >= self.max_cycles:
            self.state = self.state.set_halted(True)
            raise RuntimeError(f"Max cycles ({self.max_cycles}) exceeded")

        pc = self.state.pc
        if pc >= len(self.state.memory):
            self.state = self.state.set_halted(True)
            entry = ExecutionTraceEntry(
                cycle=self.state.cycle_count,
                instruction="<END OF PROGRAM>",
                decode_result=DecodeResult("OP_HALT", {}, True),
                pre_state=self.state.snapshot(),
                post_state=self.state.snapshot(),
                error="PC past end of program"
            )
            self.trace.append(entry)
            return entry

        instruction = self.state.memory[pc]
        pre_state = self.state.snapshot()
        decode_result = self.decoder.decode(instruction)

        error = None
        if decode_result.valid:
            try:
                self.state = self.registry.execute(
                    self.state, decode_result.key, decode_result.params
                )
            except Exception as e:
                error = str(e)
                self.state = self.state.set_halted(True)
        else:
            error = decode_result.error
            self.state = self.state.set_halted(True)

        post_state = self.state.snapshot()
        entry = ExecutionTraceEntry(
            cycle=self.state.cycle_count,
            instruction=instruction,
            decode_result=decode_result,
            pre_state=pre_state,
            post_state=post_state,
            error=error
        )
        self.trace.append(entry)
        return entry

    def run(self, max_cycles: Optional[int] = None) -> List[ExecutionTraceEntry]:
        """Run until HALT or max cycles."""
        if self.state is None:
            raise RuntimeError("No program loaded")
        limit = max_cycles if max_cycles is not None else self.max_cycles
        while not self.state.halted and self.state.cycle_count < limit:
            self.step()
        if not self.state.halted and self.state.cycle_count >= limit:
            raise RuntimeError(f"Max cycles ({limit}) exceeded")
        return self.trace

    def get_register(self, reg: str) -> int:
        if self.state is None:
            raise RuntimeError("No program loaded")
        return self.state.get_register(reg)

    def dump_registers(self) -> Dict[str, int]:
        if self.state is None:
            raise RuntimeError("No program loaded")
        return self.state.dump_registers()

    def get_flags(self) -> Dict[str, bool]:
        if self.state is None:
            raise RuntimeError("No program loaded")
        return dict(self.state.flags)

    def get_pc(self) -> int:
        if self.state is None:
            raise RuntimeError("No program loaded")
        return self.state.pc

    def get_cycle_count(self) -> int:
        if self.state is None:
            return 0
        return self.state.cycle_count

    def is_halted(self) -> bool:
        if self.state is None:
            return True
        return self.state.halted

    def print_trace(self) -> None:
        """Print execution trace in human-readable format."""
        print("=" * 70)
        print("nCPU EXECUTION TRACE")
        print("=" * 70)

        for entry in self.trace:
            status = "OK" if not entry.error else f"ERROR: {entry.error}"
            print(f"\n[Cycle {entry.cycle}] {status}")
            print(f"  Instruction: {entry.instruction}")
            print(f"  Decoded Key: {entry.decode_result.key}")
            print(f"  Params: {entry.decode_result.params}")

            pre_regs = entry.pre_state.get("registers", {})
            post_regs = entry.post_state.get("registers", {})
            changes = []
            for reg in sorted(pre_regs.keys()):
                if pre_regs[reg] != post_regs.get(reg, pre_regs[reg]):
                    changes.append(f"{reg}: {pre_regs[reg]} -> {post_regs[reg]}")
            if changes:
                print(f"  Changes: {', '.join(changes)}")

            pre_pc = entry.pre_state.get("pc", 0)
            post_pc = entry.post_state.get("pc", 0)
            if pre_pc != post_pc:
                print(f"  PC: {pre_pc} -> {post_pc}")

        print("\n" + "=" * 70)
        print("FINAL STATE")
        print("=" * 70)
        if self.state:
            print(f"  Registers: {self.dump_registers()}")
            print(f"  Flags: {self.get_flags()}")
            print(f"  PC: {self.get_pc()}")
            print(f"  Cycles: {self.get_cycle_count()}")
            print(f"  Halted: {self.is_halted()}")

    def get_summary(self) -> Dict:
        """Get execution summary."""
        return {
            "cycles": self.get_cycle_count(),
            "halted": self.is_halted(),
            "registers": self.dump_registers() if self.state else {},
            "flags": self.get_flags() if self.state else {},
            "pc": self.get_pc() if self.state else 0,
            "trace_length": len(self.trace),
            "errors": [e.error for e in self.trace if e.error],
        }
