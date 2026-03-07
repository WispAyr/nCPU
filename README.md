<p align="center">
  <img src="assets/logo.png" alt="nCPU" width="400">
</p>

<p align="center">
  <strong>A CPU where every arithmetic operation is a trained neural network.</strong><br>
  Addition uses Kogge-Stone carry-lookahead. Multiplication uses a learned byte-pair lookup table.<br>
  Bitwise ops use neural truth tables. Shifts use attention-based bit routing. No hardcoded arithmetic.<br>
  All state lives on GPU as tensors. All computation stays on-device.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-850%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/models-24%20trained-blue" alt="Models">
  <img src="https://img.shields.io/badge/accuracy-100%25%20integer-green" alt="Accuracy">
  <img src="https://img.shields.io/badge/verified-exhaustive-blueviolet" alt="Verified">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## Why?

**Can neural networks perform exact integer arithmetic?**

Prior systems (Neural GPUs, NALU, Neural Turing Machines) achieve *approximate* results. nCPU demonstrates they can be *exact* --- 100% accurate on 32-bit integers --- when you decompose operations into sub-problems with exhaustively trainable input spaces.

This yields several novel findings:
- **Multiplication is 12x faster than addition** --- inverting the conventional CPU hierarchy
- **Classical hardware algorithms transfer to neural architectures** --- Kogge-Stone CLA works in neural nets
- **Memorization-by-decomposition** --- a general principle for exact neural computation

> See the [research paper](paper/ncpu_paper.md) and [wiki](../../wiki) for detailed analysis.

## Quick Start

```bash
pip install -e ".[dev]"

# Neural mode --- all arithmetic through trained neural networks
python main.py --program programs/fibonacci.asm

# GPU compute mode --- qemu-style Metal shader, ~4M IPS
python main.py --program programs/fibonacci.asm --compute

# GPU UNIX OS --- 25-command shell with fork/pipe/wait on Metal
python ncpu/os/gpu/demo.py --multiproc
```

## Three Execution Tiers

| Tier | ALU Backend | IPS | What It Demonstrates |
|------|-------------|-----|---------------------|
| **Neural** | Trained `.pt` models | ~5K | Exact neural arithmetic (the research contribution) |
| **Fast** | Native tensor ops | ~60K | GPU-resident state with `torch.add`/`torch.mul` |
| **Compute** | Metal compute shaders | ~4M+ | qemu-style GPU execution, zero CPU-GPU sync |

```python
# Neural mode
from ncpu.model import CPU
cpu = CPU(neural_execution=True)
cpu.load_program("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
cpu.run()
print(cpu.get_register("R2"))  # 42 --- computed by neural byte-pair LUT

# GPU compute mode
from kernels.mlx.ncpu_kernel import NCPUComputeKernel
kernel = NCPUComputeKernel()
kernel.load_program_from_asm("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
result = kernel.execute()  # ~4M IPS on Metal
```

## What's Inside

### GPU-Native Multi-Process UNIX OS

A 25-command UNIX shell running as compiled C on Apple Silicon Metal GPU with full multi-process support:

```
gpu:/home/user$ ls | grep .c | sort
fib.c
fork_test.c
hello.c
gpu:/home/user$ cc fork_test.c && run /bin/fork_test
Parent PID: 1
Forked child PID: 2
Child process (PID 2, parent 1)
Child exited, parent done
```

- **25 shell commands** including pipes (`|`), background (`&`), chaining (`;`/`&&`/`||`), redirect (`>`/`>>`)
- **Multi-process**: fork/wait/pipe/dup2/kill via memory swapping, up to 15 concurrent processes
- **28 syscalls**, freestanding C runtime with malloc/printf/fork/pipe/qsort/strtol
- **Robustness**: fork bomb protection, SIGTERM/SIGKILL, orphan reparenting, per-process resource limits

### neurOS: Neural Operating System

Every OS component is a neural network --- 11 trained models, zero fallbacks:

| Component | Accuracy | Component | Accuracy |
|-----------|----------|-----------|----------|
| MMU | 100% | Assembler codegen | **100%** |
| TLB | 99.6% | Assembler tokenizer | 99.4% |
| Cache | 99.7% | Compiler optimizer | 95.2% |
| Scheduler | 99.2% | Watchdog | 100% |
| Prefetch | 97.8% | Block allocator | 98.4% |

Self-compilation verified: nsl source -> neural compiler -> neural assembler -> neural CPU -> correct results (8/8).

### Self-Hosting C Compiler on Metal GPU

A ~3,500-line self-hosting C compiler (`cc.c`) that compiles C source into ARM64 machine code **entirely on the Metal GPU**, then executes the result on the same GPU. Four layers deep:

```
Host GCC compiles cc.c -> compiler₀
  GPU runs compiler₀, self-compiles cc.c -> compiler₁
    GPU runs compiler₁, compiles test.c -> binary
      GPU runs test binary -> correct result
```

| Test Program | Binary | Cycles | Result |
|-------------|--------|--------|--------|
| arithmetic (42+13) | 100 B | 81K | 55 PASS |
| fibonacci (iterative) | 280 B | 100K | 55 PASS |
| factorial (recursive) | 208 B | 85K | 120 PASS |
| bubble sort (5-elem) | 1,292 B | 133K | 12345 PASS |
| enum, typedef, switch | 92-224 B | 82-94K | All PASS |
| funcptr, union, #ifdef | 88-184 B | 77-84K | All PASS |
| i++/++i/i--/--i ops | 140-152 B | 85K | All PASS |
| large stack (>512B) | varies | varies | PASS |
| ...and 24 more | 88-1,292 B | 77-133K | **All PASS** |

Supports: structs (`.`/`->`), pointers, arrays, recursion, for/while/do-while, ternary, sizeof, compound assignment, bitwise ops, short-circuit `&&`/`||`, type casts, `enum`, `typedef`, `switch`/`case`/`default`, `#ifdef`/`#ifndef`/`#endif`, global initializers, function pointers, `union`, `#include`, `__syscall()` intrinsics. **40/40 test programs verified, 14 bugs fixed, self-compilation verified.**

```bash
python ncpu/os/gpu/programs/tools/cc_demo.py
```

### Compiled C on GPU

Full pipeline: C source -> `aarch64-elf-gcc -O2` -> raw binary -> Metal GPU kernel -> Python I/O.

| Demo | Description |
|------|-------------|
| **Crypto** | SHA-256, AES-128 ECB+CBC (6/6 FIPS pass), password vault |
| **Games** | Tetris, Snake, roguelike dungeon crawler, text adventure |
| **VMs** | Brainfuck interpreter, Forth REPL, CHIP-8 emulator |
| **Networking** | HTTP/1.0 server (TCP via Python proxy) |
| **Neural net** | MNIST classifier (Q8.8 fixed-point, 784->128->10) |
| **Tools** | ed line editor, Game of Life, interactive shell |

### Timing Side-Channel Immunity

GPU execution produces **zero cycle-count variance** (sigma=0.0 across 270 runs). Same code on native Apple Silicon shows 47-73% timing variance. AES-128 T-table attacks are structurally impossible --- no data cache, no cache lines, no cache-miss penalty.

## Neural Arithmetic at a Glance

| Instruction | Neural Model | Strategy | Latency |
|-------------|-------------|----------|---------|
| ADD/SUB/CMP | arithmetic.pt + carry_combine.pt | Kogge-Stone CLA (8 passes) | 248 us |
| MUL | multiply.pt | Byte-pair LUT (65,536 entries) | 21 us |
| AND/OR/XOR | logical.pt | Vectorized truth table | 21 us |
| SHL/SHR | lsl.pt / lsr.pt | Attention-based bit routing | 434 us |
| DIV | arithmetic.pt | Restoring division (neural subtraction) | varies |

All sub-components **exhaustively verified** --- every possible input tested, not a sample. This is a mathematical proof, not a statistical argument.

## Tests

```bash
pytest tests/ -v   # 850 tests passing
```

850 tests across 15 files: exhaustive formal verification, neural ops, neurOS (258), compute mode (138), multi-process (41), and more.

## Documentation

- **[Wiki](../../wiki)** --- comprehensive documentation (architecture, models, demos, ISA reference)
- **[Research Paper](paper/ncpu_paper.md)** --- detailed analysis and findings
- **[Model Index](models/MODEL_INDEX.md)** --- complete trained model inventory

## License

MIT
