#!/usr/bin/env python3
"""nCPU — Neural CPU Command Line Interface.

Run programs on a CPU where every ALU operation is a trained neural network.

Usage:
    python main.py --program programs/sum_1_to_10.asm
    python main.py --program programs/fibonacci.asm --trace
    python main.py --inline "MOV R0, 42; HALT"
    python main.py --binary firmware.bin --fast
"""

import argparse
import sys
from pathlib import Path


def run_neural(args):
    """Run a text assembly program with neural ALU (all ops through trained .pt models)."""
    from ncpu.model import CPU

    cpu = CPU(
        mock_mode=True,
        neural_execution=True,
        models_dir=args.models_dir,
        max_cycles=args.max_cycles,
    )

    if args.program:
        source = Path(args.program).read_text()
        if not args.quiet:
            print(f"Loading program: {args.program}")
    else:
        source = args.inline.replace(";", "\n")
        if not args.quiet:
            print("Running inline assembly")

    cpu.load_program(source)

    if not args.quiet:
        print("Neural ALU: all operations through trained .pt models")
        print("-" * 60)
        print("Executing...")
        print("-" * 60)

    try:
        cpu.run()
    except RuntimeError as e:
        print(f"Execution error: {e}")

    if args.trace:
        cpu.print_trace()
    elif not args.quiet:
        summary = cpu.get_summary()
        print(f"\nCycles: {summary['cycles']}")
        print(f"Halted: {summary['halted']}")
        print(f"Registers: {summary['registers']}")
        print(f"Flags: {summary['flags']}")
        if summary['errors']:
            print(f"Errors: {summary['errors']}")
    else:
        regs = cpu.dump_registers()
        for reg in sorted(regs.keys()):
            if regs[reg] != 0:
                print(f"{reg}={regs[reg]}")

    return 0 if cpu.is_halted() else 1


def run_fast(args):
    """Run on the GPU tensor CPU (NeuralCPU) — native tensor ops, maximum speed."""
    from ncpu.neural import NeuralCPU

    device = args.device or "cpu"  # auto-detect later if needed

    cpu = NeuralCPU(device_override=device, fast_mode=True)

    if args.binary:
        binary_data = Path(args.binary).read_bytes()
        cpu.load_binary(binary_data)
        if not args.quiet:
            print(f"Loaded binary: {args.binary} ({len(binary_data)} bytes)")
    elif args.program:
        # For text assembly in fast mode, assemble to ARM64 binary first
        print("Error: --fast mode requires --binary (ARM64 binary). Text assembly uses neural mode.")
        print("Usage: python main.py --program programs/fibonacci.asm  (neural mode)")
        print("       python main.py --binary firmware.bin --fast       (GPU tensor mode)")
        return 1
    else:
        print("Error: --fast mode requires --binary")
        return 1

    if not args.quiet:
        print(f"GPU Tensor CPU: native tensor ops on {device}")
        print("-" * 60)
        print("Executing...")
        print("-" * 60)

    # Run
    cycles = 0
    max_cycles = args.max_cycles
    while cycles < max_cycles:
        result = cpu.step()
        cycles += 1
        if not result:
            break

    if not args.quiet:
        print(f"\nCycles: {cycles}")
        # Show non-zero registers
        for i in range(31):
            val = int(cpu.regs[i].item())
            if val != 0:
                print(f"  X{i} = {val}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="nCPU: A CPU where every component is a trained neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with neural ALU (default — all ops through trained .pt models)
    python main.py --program programs/sum_1_to_10.asm

    # Run with trace output
    python main.py --program programs/fibonacci.asm --trace

    # Run inline assembly
    python main.py --inline "MOV R0, 42; HALT"

    # Run ARM64 binary on GPU tensor CPU (maximum speed)
    python main.py --binary firmware.bin --fast
        """
    )

    parser.add_argument("--program", "-p", type=str, help="Path to assembly program (.asm)")
    parser.add_argument("--inline", "-i", type=str, help="Inline assembly (separate with ;)")
    parser.add_argument("--binary", "-b", type=str, help="Path to ARM64 binary (for --fast mode)")
    parser.add_argument("--fast", action="store_true",
                        help="GPU tensor mode: native tensor ops, maximum speed (requires --binary)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, mps (default: auto-detect)")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Path to trained .pt models")
    parser.add_argument("--max-cycles", type=int, default=10000,
                        help="Max cycles (default: 10000)")
    parser.add_argument("--trace", "-t", action="store_true",
                        help="Print full execution trace")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")

    args = parser.parse_args()

    if not args.program and not args.inline and not args.binary:
        parser.error("One of --program, --inline, or --binary is required")

    if args.fast:
        return run_fast(args)
    else:
        if args.binary:
            parser.error("--binary requires --fast flag")
        if not args.program and not args.inline:
            parser.error("Neural mode requires --program or --inline")
        return run_neural(args)


if __name__ == "__main__":
    sys.exit(main())
