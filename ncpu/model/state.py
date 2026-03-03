"""CPUState: Immutable state representation for the model-based nCPU.

State Components:
    - Registers: R0-R7 (8 general-purpose 32-bit signed integers)
    - PC: Program counter
    - Flags: ZF (zero), SF (sign/negative)
    - Memory: Program instructions (list of strings)
    - Halted: Execution termination flag
    - Cycle count: Total executed cycles

All state mutations return new state objects, preserving immutability
for full execution trace capability.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from copy import deepcopy


INT32_MIN = -(2**31)
INT32_MAX = (2**31) - 1


@dataclass
class CPUState:
    """Immutable CPU state representation.

    Attributes:
        registers: Dictionary mapping register names (R0-R7) to 32-bit signed values
        pc: Program counter (current instruction address)
        flags: Dictionary of CPU flags (ZF=zero, SF=sign/negative)
        memory: List of instruction strings (program loaded in memory)
        halted: Whether the CPU has executed HALT
        cycle_count: Number of execution cycles completed
    """
    registers: Dict[str, int] = field(default_factory=lambda: {
        "R0": 0, "R1": 0, "R2": 0, "R3": 0,
        "R4": 0, "R5": 0, "R6": 0, "R7": 0
    })
    pc: int = 0
    flags: Dict[str, bool] = field(default_factory=lambda: {
        "ZF": False,
        "SF": False
    })
    memory: List[str] = field(default_factory=list)
    halted: bool = False
    cycle_count: int = 0

    def snapshot(self) -> dict:
        """Create a deep-copy snapshot for tracing."""
        return {
            "registers": deepcopy(self.registers),
            "pc": self.pc,
            "flags": deepcopy(self.flags),
            "halted": self.halted,
            "cycle_count": self.cycle_count,
        }

    def validate(self) -> bool:
        """Validate state integrity."""
        expected_regs = {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}
        if set(self.registers.keys()) != expected_regs:
            return False

        for value in self.registers.values():
            if not isinstance(value, int):
                return False
            if value < INT32_MIN or value > INT32_MAX:
                return False

        if self.pc < 0:
            return False
        if self.memory and self.pc > len(self.memory):
            return False

        if set(self.flags.keys()) != {"ZF", "SF"}:
            return False
        for flag_value in self.flags.values():
            if not isinstance(flag_value, bool):
                return False

        if self.cycle_count < 0:
            return False

        return True

    def get_register(self, reg: str) -> int:
        """Get value of a register (case insensitive)."""
        reg_upper = reg.upper()
        if reg_upper not in self.registers:
            raise KeyError(f"Invalid register: {reg}")
        return self.registers[reg_upper]

    def set_register(self, reg: str, value: int) -> "CPUState":
        """Return new state with updated register value (clamped to 32-bit)."""
        reg_upper = reg.upper()
        if reg_upper not in self.registers:
            raise KeyError(f"Invalid register: {reg}")

        clamped_value = max(INT32_MIN, min(INT32_MAX, value))
        new_registers = deepcopy(self.registers)
        new_registers[reg_upper] = clamped_value

        return CPUState(
            registers=new_registers,
            pc=self.pc,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def set_flags(self, value: int) -> "CPUState":
        """Return new state with flags derived from value."""
        new_flags = {
            "ZF": value == 0,
            "SF": value < 0
        }
        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc,
            flags=new_flags,
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def set_flags_direct(self, zf: bool, sf: bool) -> "CPUState":
        """Return new state with explicitly set flags."""
        new_flags = {"ZF": zf, "SF": sf}
        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc,
            flags=new_flags,
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def increment_pc(self) -> "CPUState":
        """Return new state with PC + 1."""
        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc + 1,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def set_pc(self, new_pc: int) -> "CPUState":
        """Return new state with specified PC."""
        return CPUState(
            registers=deepcopy(self.registers),
            pc=new_pc,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def set_halted(self, halted: bool = True) -> "CPUState":
        """Return new state with halted flag."""
        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=halted,
            cycle_count=self.cycle_count
        )

    def increment_cycle(self) -> "CPUState":
        """Return new state with cycle_count + 1."""
        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count + 1
        )

    def dump_registers(self) -> Dict[str, int]:
        """Get a copy of all register values."""
        return deepcopy(self.registers)

    def __str__(self) -> str:
        regs = " ".join(f"{k}={v}" for k, v in sorted(self.registers.items()))
        flags = " ".join(f"{k}={int(v)}" for k, v in self.flags.items())
        return f"[Cycle {self.cycle_count}] PC={self.pc} {regs} {flags} {'HALTED' if self.halted else ''}"


def create_initial_state(program: List[str]) -> CPUState:
    """Create initial CPU state with a loaded program."""
    return CPUState(
        registers={"R0": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R6": 0, "R7": 0},
        pc=0,
        flags={"ZF": False, "SF": False},
        memory=program,
        halted=False,
        cycle_count=0
    )
