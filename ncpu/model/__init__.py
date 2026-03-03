"""Model-based nCPU: LLM-powered instruction decode with neural execution.

Architecture:
    MEMORY -> FETCH -> DECODE (LLM or regex) -> KEY -> REGISTRY -> EXECUTE -> STATE

Three execution modes:
    mock:   Regex decode + Python arithmetic (fast, no GPU needed)
    real:   LLM decode + Python arithmetic (requires torch + decode_llm model)
    neural: Regex decode + trained neural models for ALU (requires torch + .pt models)

Quick start:
    from ncpu.model import CPU

    # Mock mode (default)
    cpu = CPU()
    cpu.load_program("MOV R0, 42\\nHALT")
    cpu.run()
    print(cpu.get_register("R0"))  # 42

    # Neural mode (all ALU ops use trained neural networks)
    cpu = CPU(neural_execution=True)
    cpu.load_program("MOV R0, 10\\nMOV R1, 20\\nADD R2, R0, R1\\nHALT")
    cpu.run()
    print(cpu.get_register("R2"))  # 30 (computed by neural full adder)
"""

from .cpu import CPU
from .state import CPUState, create_initial_state
from .registry import CPURegistry, get_registry
from .decode import Decoder, DecodeResult, parse_program

__all__ = [
    "CPU", "CPUState", "create_initial_state",
    "CPURegistry", "get_registry",
    "Decoder", "DecodeResult", "parse_program",
]
