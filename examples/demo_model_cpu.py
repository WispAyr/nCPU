#!/usr/bin/env python3
"""Demo: Model-based nCPU running assembly programs.

Shows the model-based nCPU executing programs with full trace output.
Uses the trained semantic decoder model.
"""

from pathlib import Path
import sys

# Allow running this file directly from the repo without pip install -e .
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ncpu.model import CPU


def demo_sum():
    """Calculate sum of 1 to 10."""
    print("=" * 60)
    print("DEMO: Sum of 1 to 10")
    print("=" * 60)

    cpu = CPU()
    cpu.load_program("""
        MOV R0, 0       ; sum = 0
        MOV R1, 1       ; counter = 1
        MOV R2, 11      ; limit (exclusive)
        MOV R3, 1       ; increment
    loop:
        ADD R0, R0, R1  ; sum += counter
        ADD R1, R1, R3  ; counter++
        CMP R1, R2      ; compare to limit
        JNZ loop        ; loop if not equal
        HALT
    """)
    cpu.run()

    print(f"  Result: R0 = {cpu.get_register('R0')} (expected 55)")
    print(f"  Cycles: {cpu.get_cycle_count()}")
    print(f"  Halted: {cpu.is_halted()}")
    print()


def demo_fibonacci():
    """Calculate 10th Fibonacci number."""
    print("=" * 60)
    print("DEMO: Fibonacci (10 iterations)")
    print("=" * 60)

    cpu = CPU()
    cpu.load_program("""
        MOV R0, 0       ; fib(0)
        MOV R1, 1       ; fib(1)
        MOV R2, 10      ; iterations
        MOV R3, 0       ; counter
        MOV R4, 1       ; constant 1
    loop:
        MOV R5, R1      ; temp = current
        ADD R1, R0, R1  ; current = prev + current
        MOV R0, R5      ; prev = temp
        ADD R3, R3, R4  ; counter++
        CMP R3, R2
        JNZ loop
        HALT
    """)
    cpu.run()

    print(f"  Result: R1 = {cpu.get_register('R1')} (expected 89)")
    print(f"  Cycles: {cpu.get_cycle_count()}")
    print()


def demo_multiply():
    """Multiply via repeated addition."""
    print("=" * 60)
    print("DEMO: 7 x 6 via repeated addition")
    print("=" * 60)

    cpu = CPU()
    cpu.load_program("""
        MOV R0, 0       ; result
        MOV R1, 7       ; multiplicand
        MOV R2, 6       ; multiplier (countdown)
        MOV R3, 1       ; decrement
        MOV R4, 0       ; zero for comparison
    loop:
        ADD R0, R0, R1  ; result += multiplicand
        SUB R2, R2, R3  ; multiplier--
        CMP R2, R4
        JNZ loop
        HALT
    """)
    cpu.run()

    print(f"  Result: R0 = {cpu.get_register('R0')} (expected 42)")
    print(f"  Cycles: {cpu.get_cycle_count()}")
    print()


def demo_trace():
    """Show execution trace for a simple program."""
    print("=" * 60)
    print("DEMO: Execution trace")
    print("=" * 60)

    cpu = CPU()
    cpu.load_program("""
        MOV R0, 10
        MOV R1, 20
        ADD R2, R0, R1
        HALT
    """)
    cpu.run()
    cpu.print_trace()
    print()


if __name__ == "__main__":
    demo_sum()
    demo_fibonacci()
    demo_multiply()
    demo_trace()
