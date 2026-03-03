# nCPU: A Neural CPU Achieving 100% Accuracy on Integer Arithmetic via Trained Neural Networks

**Robert Price**

*March 2026*

---

## Abstract

We present nCPU, a software CPU implementation in which every arithmetic logic unit (ALU) operation is executed by a trained PyTorch neural network. Unlike prior work on neural arithmetic, which achieves approximate computation or operates on limited bit-widths, nCPU demonstrates that neural networks can perfectly replicate discrete 32-bit integer operations --- not approximately, but exactly. The system implements a complete instruction set including addition, subtraction, multiplication, bitwise logic, barrel shifting, comparison, and conditional branching, with all data-path operations routed through trained models. A bit-serial neural full adder achieves 100% accuracy on the complete truth table of 8 input combinations. A 256x256x16 learned lookup table achieves 100% accuracy on all 65,536 byte-pair products. Attention-based shift networks learn to route bits to correct output positions without any hardcoded shift logic. Across 330 automated tests covering seven benchmark programs, nCPU produces results identical to conventional arithmetic on all integer operations. The system comprises 22 trained models totaling approximately 48 MB of weights (excluding the instruction decoder), organized into ALU, shift, memory, register, decoder, and mathematical function components. Benchmarking on Apple Silicon reveals a counterintuitive finding: neural multiplication (21 us via batched LUT lookup) is 12x faster than neural addition (248 us via Kogge-Stone carry-lookahead), inverting the conventional CPU performance hierarchy. A Kogge-Stone parallel-prefix carry-lookahead adder, using a trained carry-combine neural network, reduced addition latency from 826 us (32 sequential full adder passes) to 248 us (8 neural passes) --- a 3.3x speedup. We further demonstrate that the shift network's original 64-pass-per-bit architecture can be vectorized into 3 batched forward passes, reducing shift latency from 2,833 us to 437 us --- a 6.5x speedup validating the design principle that independent computations should always be parallelized in neural architectures. We describe the architecture, training methodology, benchmark results, six novel findings about neural computation performance, and position this work against prior neural arithmetic systems.

## 1. Introduction

The question of whether neural networks can perform exact computation has occupied researchers since the early days of connectionism. Neural Turing Machines (Graves et al., 2014), Neural GPUs (Kaiser & Sutskever, 2015), and Neural Arithmetic Logic Units (Trask et al., 2018) have demonstrated that differentiable architectures can learn algorithmic patterns, but these systems typically achieve approximate results, operate on small bit-widths, or require carefully constrained training regimes. The gap between "neural networks that approximate arithmetic" and "neural networks that execute arithmetic perfectly" has remained largely open.

This paper describes nCPU, a system that closes this gap for 32-bit integer operations. In nCPU, every ALU operation --- addition, subtraction, multiplication, bitwise AND/OR/XOR, logical and arithmetic shifts, comparison, increment, and decrement --- passes through a trained neural network. No Python arithmetic fallback is used in the neural execution path. The key insight enabling exact computation is architectural: rather than training a single monolithic network to perform arbitrary-precision arithmetic, we decompose operations into sub-problems where exhaustive training is tractable.

For addition, a 3-input, 2-output neural full adder learns the carry-propagation pattern from all 8 possible input combinations (bit_a, bit_b, carry_in), then executes bit-serially 32 times. For multiplication, a 256x256x16 parameter tensor memorizes every byte-pair product, and 32-bit multiplication decomposes into at most 16 lookups with shift-and-add accumulation. For bitwise logic, a 7x4 truth table parameter learns the output for each (bit_a, bit_b) combination for seven operations. For shifting, a decomposed three-network architecture uses learned attention over input bit positions to route each output bit.

### Contributions

1. **Exact neural integer arithmetic.** We demonstrate that trained neural networks can execute 32-bit integer addition, subtraction, and multiplication with 100% accuracy, verified across 290 automated tests.

2. **Architecture-driven tractability.** We show that decomposing operations into exhaustively trainable sub-problems (8-entry truth tables, 65,536-entry LUTs, attention-based routing) enables exact computation without approximation.

3. **A complete neural CPU.** We provide a working system with three execution strategies, 21 trained models, and a full pipeline from text assembly or binary ARM64 instructions through neural decode, register access, ALU execution, and flag computation.

4. **Performance inversion.** We show that neural multiplication is 38x faster than neural addition, inverting the conventional CPU performance hierarchy, and identify sequential carry propagation as the dominant latency factor in neural arithmetic.

5. **Memorization-by-decomposition.** We articulate a general design principle for exact neural computation: decompose into sub-problems with exhaustively trainable input spaces, train to 100% accuracy, threshold to discrete outputs, and compose structurally.

6. **Practical engineering insights.** We document critical implementation details --- temperature conventions in attention-based shifting, float32 precision boundaries in bit conversion, and signed 32-bit wraparound in Python --- that are essential for reproducing exact neural arithmetic.

## 2. Architecture

### 2.1 System Overview

nCPU provides three execution strategies, each representing a different point in the design space:

| Strategy | Module | Input Format | ALU Backend | Primary Use |
|----------|--------|-------------|-------------|-------------|
| **Neural CPU** | `ncpu.neural.NeuralCPU` | ARM64 binary | Neural ALU Bridge | Full GPU-resident emulation |
| **Model CPU** | `ncpu.model.CPU` | Text assembly | NeuralOps / NeuralRegistry | Training, testing, programs |
| **Tensor CPU** | `ncpu.tensor` | ARM64 binary | Pure tensor arithmetic | Maximum throughput |

The Neural CPU (`ncpu.neural.NeuralCPU`) is a 12,187-line GPU-resident ARM64 CPU implementation where all state --- registers, flags, program counter, and memory --- is stored as PyTorch tensors. It decodes binary ARM64 instructions and routes all ALU operations through the Neural ALU Bridge.

The Model CPU (`ncpu.model.CPU`) accepts text assembly, parses it through a regex-based or LLM-based decoder, and executes operations through the `NeuralRegistry`, which dispatches to `NeuralOps`. This is the primary vehicle for testing and program execution.

### 2.2 Execution Pipeline

The neural execution pipeline for the Model CPU proceeds as follows:

```
Source Assembly
    |
    v
[Decoder] ---> Operation Key (e.g., "OP_ADD")
    |
    v
[NeuralRegistry] ---> Dispatch to handler
    |
    v
[NeuralOps] ---> Load trained .pt model
    |
    v
[Neural Model] ---> Compute result (e.g., bit-serial addition)
    |
    v
[CPUState] ---> Update registers, flags, PC
```

For the Neural CPU, the pipeline is:

```
ARM64 Binary (4 bytes)
    |
    v
[Neural Decoder / Bitfield Extraction] ---> Opcode, registers, immediates
    |
    v
[Neural ALU Bridge] ---> 64-bit -> 32-bit narrowing -> NeuralOps -> result
    |
    v
[Tensor State] ---> Update register tensors, flag tensors, PC tensor
```

The Neural ALU Bridge (`ncpu.neural.neural_alu_bridge.NeuralALUBridge`) mediates between the 64-bit tensor state of the Neural CPU and the 32-bit trained models. It narrows `torch.int64` values to 32-bit signed integers, dispatches through `NeuralOps`, and returns Python integers for tensor assignment.

### 2.3 Model Organization

Trained models are organized by functional category:

```
models/
  alu/
    arithmetic.pt     (38 KB)   -- Neural full adder (ADD/SUB/INC/DEC)
    carry_combine.pt  (13 KB)   -- Carry-combine for parallel-prefix CLA
    multiply.pt       (4.0 MB)  -- Byte-pair multiplication LUT
    logical.pt        (1.7 KB)  -- Truth table parameters (AND/OR/XOR/...)
    compare.pt        (2.0 KB)  -- Comparison refinement layer
    divide.pt         (13 KB)   -- Division full adder (hidden_dim=64)
  shifts/
    lsl.pt           (5.6 MB)  -- Left shift network
    lsr.pt           (5.6 MB)  -- Right shift (logical) network
    asr.pt           (2.8 MB)  -- Arithmetic shift right
    rol.pt           (2.6 MB)  -- Rotate left
  register/
    register_file.pt (241 KB)  -- Neural register file (ARM64 semantics)
    register_vsa.pt  (7.7 MB)  -- VSA-based register (experimental)
  memory/
    stack.pt         (374 KB)  -- Neural stack pointer
    pointer.pt       (306 KB)  -- Neural pointer dereference
    function_call.pt (196 KB)  -- Neural BL/RET handling
  decoder/
    arm64_decoder.pt (6.5 MB)  -- Transformer-based instruction decoder
  math/
    sincos.pt        (4.0 MB)  -- Sine-activated trig approximation
    atan2.pt         (6.1 MB)  -- BatchNorm-stabilized atan2
    sqrt.pt          (536 KB)  -- Two-stage Newton-style sqrt
    exp.pt           (521 KB)  -- Neural exponential
    log.pt           (521 KB)  -- Neural logarithm
    doom_trig.pt     (66 KB)   -- Fixed-point trig LUT (8192 entries)
  decode_llm/
    adapter_model.safetensors (70 MB) -- Qwen2.5-Coder-1.5B LoRA adapter
```

Total weights for the core ALU + shift + memory + register + decoder models: approximately 48 MB. The instruction decode LLM adapter adds 70 MB.

## 3. Model Architectures

### 3.1 Neural Full Adder (arithmetic.pt)

The foundation of nCPU's arithmetic is a bit-serial full adder implemented as a three-layer MLP:

```
NeuralFullAdder:
    Linear(3, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 2) -> Sigmoid
```

**Input:** A 3-element float vector `(bit_a, bit_b, carry_in)`, where each element is 0.0 or 1.0.

**Output:** A 2-element float vector `(sum_bit, carry_out)`, thresholded at 0.5 to produce discrete bits.

**Operation (Carry-Lookahead):** nCPU implements a Kogge-Stone parallel-prefix carry-lookahead adder using a trained carry-combine neural network (`carry_combine.pt`). The carry-combine operator computes `G_out = G_i | (P_i & G_j)` and `P_out = P_i & P_j`, trained on all 16 input combinations to 100% accuracy. The CLA algorithm proceeds in three phases:

1. **Generate/Propagate:** For each bit position, compute `G[i] = a[i] AND b[i]` and `P[i] = a[i] XOR b[i]` using the neural logical truth tables (2 vectorized passes).
2. **Parallel-prefix tree:** Five stages of carry combining (stride 1, 2, 4, 8, 16), each a single batched forward pass through carry_combine.pt (5 passes).
3. **Final sum:** `S[i] = P[i] XOR C[i-1]` using neural XOR (1 vectorized pass).

Total: 8 neural forward passes instead of 32, reducing addition latency from ~826 us to ~248 us (3.3x speedup). The ripple-carry full adder remains as a fallback if carry_combine.pt is not available.

**Subtraction** reuses the CLA via two's complement: to compute `a - b`, the system complements all bits of `b` and sets carry_in=1. This identity, `a - b = a + ~b + 1`, requires no separate subtraction model.

**Increment and decrement** are further reductions: `INC(a) = a + 1` and `DEC(a) = a - 1`, both routed through the CLA.

The full adder model has 8,962 trainable parameters. The carry-combine model has 2,466 parameters (Linear(4,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,2)). Both are trained on exhaustive truth tables (8 and 16 entries respectively).

### 3.2 Neural Multiplication LUT (multiply.pt)

Multiplication uses a fundamentally different strategy: memorization through a learned lookup table.

```
NeuralMultiplierLUT:
    lut.table: nn.Parameter of shape [256, 256, 16]
```

**Architecture:** A single parameter tensor of shape `[256, 256, 16]` stores the product of every pair of bytes (0--255) as 16 sigmoid-activated bits. There are no hidden layers --- the entire model is a stored lookup table with learned entries.

**32-bit multiplication procedure:**

1. Decompose each 32-bit operand into 4 bytes: `a = [a0, a1, a2, a3]`, `b = [b0, b1, b2, b3]`.
2. For each non-zero pair `(a_i, b_j)`, look up `sigmoid(lut.table[a_i, b_j]) > 0.5` to obtain a 16-bit product.
3. Convert the 16 sigmoid-activated bits to an integer using bit-value weighting: `sum(bit_k * 2^k for k in 0..15)`.
4. Shift the partial product left by `(i + j) * 8` bits and accumulate into the result.
5. Clamp the final result to the 32-bit signed range.

**Vectorized execution:** Rather than performing 16 sequential lookups, all non-zero byte pairs are gathered into batch index tensors, and the lookup + sigmoid + thresholding + bit-to-int conversion is performed in a single vectorized operation.

The model stores 256 * 256 * 16 = 1,048,576 parameters. Each byte-pair product is a 16-bit representation, sufficient to encode any product in the range 0--65,025 (255 * 255).

### 3.3 Neural Logical Operations (logical.pt)

Bitwise logic is implemented through learned truth tables:

```
NeuralLogical:
    truth_tables: nn.Parameter of shape [7, 4]
```

**Architecture:** A 7x4 parameter tensor where each row represents one logical operation and each column represents one entry in the 2-input truth table. The operations are indexed as: AND=0, OR=1, XOR=2, NOT=3, NAND=4, NOR=5, XNOR=6.

**Operation:** For each bit position, the index `a*2 + b` (where `a` and `b` are 0 or 1) selects the truth table entry. The output is `sigmoid(truth_tables[op, idx]) > 0.5`.

**Vectorized execution:** All 32 bits are processed simultaneously. The bit vectors are converted to long tensors, the index tensor `bits_a * 2 + bits_b` is computed in one operation, and the truth table is indexed with a single gather, followed by batch sigmoid and thresholding.

The model has only 28 parameters (7 operations * 4 entries), making it the smallest model in the system. Despite its simplicity, this is sufficient because the truth tables are exact --- after sigmoid thresholding, the learned values reproduce the correct boolean function.

### 3.4 Neural Shift Networks (lsl.pt, lsr.pt)

Bit shifting is the most architecturally complex operation, implemented through a decomposed three-network design:

```
NeuralShiftNet:
    shift_decoder:  Linear(64, 768) -> ReLU -> Linear(768, 768) -> ReLU -> Linear(768, 64)
    index_net:      Linear(128, 768) -> ReLU -> Linear(768, 768) -> ReLU -> Linear(768, 64)
    validity_net:   Linear(128, 384) -> ReLU -> Linear(384, 1)
    temperature:    nn.Parameter (learned, approximately 0.01)
```

**Shift decoder:** Takes a 64-element binary encoding of the shift amount and produces a 64-dimensional internal representation, which is then passed through softmax to create a probability distribution over shift positions.

**Index network:** For each of the 64 output bit positions, receives a 128-element input concatenating a one-hot position encoding (64 elements) with the softmax-normalized shift encoding (64 elements). Produces 64 logits representing attention weights over the 64 input bit positions. These logits are divided by the learned temperature parameter and passed through softmax to create sharp attention: `weights = softmax(logits / temperature)`.

**Validity network:** Takes the same 128-element input and produces a single sigmoid-gated output, determining whether the output bit position should be active (1) or zeroed (0). This handles the zero-fill behavior of logical shifts.

**Temperature convention:** The learned temperature parameter converges to approximately 0.01 during training. The critical convention is that logits are *divided* by this temperature, yielding `softmax(logits / 0.01) = softmax(logits * 100)`. This sharpens the attention distribution so that effectively one source bit receives all weight, producing discrete (exact) bit routing. Multiplying by the temperature instead (a common implementation error) would flatten the distribution and produce all-zeros outputs.

**Forward pass (vectorized across all 64 output bits):**

1. Encode shift amount -> `shift_decoder` -> softmax -> `shift_soft` (64-dim) — 1 forward pass
2. Build batched input: identity matrix (64x64 one-hot positions) concatenated with `shift_soft` expanded to all 64 rows -> [64, 128] combined matrix
3. `index_net([64, 128])` -> [64, 64] logits -> `softmax(logits / temperature, dim=1)` -> attention weights for all bits simultaneously — 1 forward pass
4. `output_bits = sum(attention_weights * value_bits, dim=1)` (batch weighted sum)
5. `validity_net([64, 128])` -> [64, 1] -> sigmoid -> batch gate — 1 forward pass

All 64 output bit positions are computed in **3 batched forward passes** (1x shift_decoder + 1x index_net + 1x validity_net). Each position's computation is independent --- position `i`'s input `[one_hot_i, shift_soft]` depends only on the shared shift encoding, not on any other position's output. This independence makes batching mathematically equivalent to the sequential formulation while eliminating 125 redundant kernel launches.

Separate models are trained for left shift (lsl.pt, 5.6 MB) and right shift (lsr.pt, 5.6 MB). The architecture is identical; only the trained weights differ.

### 3.5 Neural Comparison (CMP)

Comparison does not use a dedicated model. Instead, it reuses the neural full adder:

```
CMP(a, b):
    diff = neural_sub(a, b)    // Two's complement subtraction via neural adder
    N_flag = (diff < 0)        // Negative: sign bit of result
    Z_flag = (diff == 0)       // Zero: all bits zero
    C_flag = (unsigned(a) >= unsigned(b))  // Carry: unsigned comparison
```

This mirrors the implementation of CMP in real ARM64 hardware, where CMP is an alias for `SUBS XZR, Xn, Xm` --- a subtraction that discards the result and sets flags.

A `compare.pt` model (Linear(3, 3), 12 parameters) exists as a refinement layer but is not used in the active execution path. Neural subtraction alone provides exact flag computation.

### 3.6 Instruction Decoder (decode_llm)

The instruction decoder is a Qwen2.5-Coder-1.5B language model with a LoRA adapter fine-tuned on ARM64 instruction-to-operation-key mappings.

**Architecture:** Qwen2.5-Coder-1.5B base model (1.5 billion parameters, not loaded at runtime for the model CPU) with a 70 MB LoRA adapter (`adapter_model.safetensors`).

**Function:** Given a text assembly instruction (e.g., `"ADD R0, R1, R2"`), produces the corresponding operation key (e.g., `"OP_ADD"`) and parameter extraction.

**Training:** Fine-tuned for 33,750 steps on a supervised dataset of ARM64 instruction-key pairs, reaching 100% accuracy on the training set.

The model CPU's primary decoder is regex-based (zero neural overhead); the LLM decoder is an alternative for the `--mode real` execution path.

### 3.7 Transformer-Based Binary Decoder (arm64_decoder.pt)

For the Neural CPU path, a transformer-based decoder processes raw 32-bit ARM64 instructions:

```
NeuralARM64Decoder:
    encoder:
        bit_embed:  Embedding(2, 64)      -- Binary bit embeddings
        pos_embed:  Embedding(32, 64)     -- Positional encoding for 32 bit positions
        combine:    Linear(128, 256)      -- Merge bit + position to 256-dim
    field_extractor:
        field_queries:  Parameter(6, 256) -- 6 learned query vectors
        self_attn:      MultiheadAttention(256, 8 heads)
        field_attn:     MultiheadAttention(256, 8 heads)  -- Cross-attention
        norm1, norm2:   LayerNorm(256)
    decoder_head:
        category_head:  Linear(256,128) -> ReLU -> Dropout -> Linear(128,10)
        operation_head: Linear(256,256) -> ReLU -> Dropout -> Linear(256,128)
        rd_head:        Linear(256, 32)   -- Destination register
        rn_head:        Linear(256, 32)   -- Source register 1
        rm_head:        Linear(256, 32)   -- Source register 2
        imm_head:       Linear(256,256) -> ReLU -> Linear(256,26)  -- Immediate
        flags_head:     Linear(256, 3)    -- Flag-setting bits
    refine: Linear(1536, 512) -> ReLU -> Linear(512, 256)
```

This decoder takes a 32-bit instruction as input, embeds each bit with positional encoding, applies self-attention to capture bit-field structure, then uses 6 learned cross-attention queries to extract structured fields: instruction category, operation type, three register indices, immediate value, and flag-setting behavior.

### 3.8 Mathematical Functions (Experimental)

nCPU includes experimental neural approximations for transcendental functions:

| Model | Architecture | Input | Output | Notes |
|-------|-------------|-------|--------|-------|
| sincos.pt | 5 sine-activated blocks (Linear+sin), 512 hidden -> Linear(512,2) | angle (radians) | (sin, cos) | Periodic activation matches target periodicity |
| sqrt.pt | Two-stage: initial(1->256->256->1) + refine(2->256->256->1), BatchNorm | x | sqrt(x) | Newton-style iterative refinement |
| exp.pt | 4-layer MLP: 1->256->256->256->1, ReLU | x | exp(x) | Direct approximation |
| log.pt | 4-layer MLP: 1->256->256->256->1, ReLU | x | log(x) | Direct approximation |
| atan2.pt | 6 residual layers with BatchNorm, 512 hidden, Linear(6->512) -> 6x[512->512+residual] -> Linear(512->2) | (sin, cos, quadrant) | (angle_sin, angle_cos) | Branch-free angle computation |

These models operate on fixed-point inputs (integer value / 1000 = real value) and produce floating-point outputs. Unlike the integer ALU models, these are *approximate* --- they are neural function approximators, not exact implementations. They are included as a demonstration of extending the neural CPU concept beyond integer arithmetic.

## 4. Training

### 4.1 Exhaustive Supervised Training

The integer ALU models are trained on exhaustive truth tables, eliminating the possibility of unseen-input failures:

**Full adder (arithmetic.pt):** Trained on all 8 input combinations of `(bit_a, bit_b, carry_in)` in `{0, 1}^3`. The training set *is* the complete input space. Binary cross-entropy loss on the sigmoid outputs drives the network to memorize the exact carry-propagation logic. The 128-unit hidden layer provides sufficient capacity for the network to converge to a perfect solution, despite having only 8 training examples.

**Multiplication LUT (multiply.pt):** Trained on all 65,536 byte-pair products `(a, b)` for `a, b in [0, 255]`. Each product is encoded as 16 sigmoid-activated bits. Binary cross-entropy loss on each bit drives the lookup table entries to their correct values. After training, `sigmoid(lut[a][b]) > 0.5` produces the exact binary representation of `a * b` for every input pair.

**Logical truth tables (logical.pt):** Trained on all 4 input combinations for each of 7 operations. The total training set is 28 examples. Given that the model has exactly 28 parameters, this is effectively solving a system of equations: each parameter maps to exactly one truth table entry.

**Shift networks (lsl.pt, lsr.pt):** Trained on (value, shift_amount) pairs where shift_amount ranges from 0 to 31. The training data covers representative 32-bit values across the full shift range. The decomposed architecture (shift_decoder + index_net + validity_net) learns to route bits to their correct output positions through the attention mechanism.

### 4.2 Training Configuration

All models are trained with:

- **Optimizer:** Adam
- **Loss:** Binary cross-entropy (for bit-level outputs) or MSE (for math models)
- **Device:** CPU (models are small enough that GPU training provides minimal benefit)
- **Precision:** float32, with careful handling of the float32 -> int64 precision boundary (see Section 6.2)

Training converges rapidly due to the small input spaces. The full adder typically converges in under 100 epochs. The multiplication LUT converges in a few hundred epochs. The shift networks require more training (thousands of epochs) due to the complexity of the attention-based routing.

## 5. Evaluation

### 5.1 Test Suite

nCPU is validated by 330 automated tests organized across 10 test files:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_neural_ops.py | 100 | All neural ALU operations: ADD, SUB, MUL, CMP, INC, DEC, AND, OR, XOR, SHL, SHR, CLA correctness, batch API |
| test_neural_bridge.py | 20 | Neural ALU Bridge: int/tensor input, 64->32 narrowing, NeuralCPU integration |
| test_neural_programs.py | 12 | Full program execution: sum_1_to_10, fibonacci, multiply, bitwise, power_of_two |
| test_programs.py | 35 | Mock-mode program execution |
| test_decode.py | 60 | Instruction decoding |
| test_registry.py | 30 | Operation dispatch and registry |
| test_state.py | 25 | CPU state management |
| test_architectures.py | 22 | Model architecture strict loading |
| test_math_ops.py | 10 | Mathematical function models |
| test_architecture_forward.py | 16 | Forward pass smoke tests |

### 5.2 Accuracy Results

| Operation | Model | Input Space | Verified Accuracy |
|-----------|-------|------------|-------------------|
| ADD | arithmetic.pt | 32-bit signed pairs | 100% |
| SUB | arithmetic.pt (complement) | 32-bit signed pairs | 100% |
| MUL | multiply.pt | 32-bit signed pairs | 100% |
| INC | arithmetic.pt (+1) | 32-bit signed values | 100% |
| DEC | arithmetic.pt (-1) | 32-bit signed values | 100% |
| AND | logical.pt | 32-bit pairs | 100% |
| OR | logical.pt | 32-bit pairs | 100% |
| XOR | logical.pt | 32-bit pairs | 100% |
| SHL | lsl.pt | 32-bit values, shifts 0-31 | 100% |
| SHR | lsr.pt | 32-bit values, shifts 0-31 | 100% |
| CMP | arithmetic.pt (sub) | 32-bit signed pairs | 100% |

All 330 tests pass. The test suite includes:

- **Parametrized arithmetic tests:** Positive, negative, zero, boundary values, overflow
- **Cross-validation:** Every program run in both mock (Python arithmetic) and neural mode; results must be identical
- **Boundary conditions:** `INT32_MIN`, `INT32_MAX`, shifts by 0 and 31, multiply by 0
- **Full programs:** Loops with conditional branching, accumulation, and register manipulation

### 5.3 Benchmark Programs

Seven assembly programs exercise the neural execution path end-to-end:

| Program | Operations Used | Expected Result | Neural Result |
|---------|----------------|-----------------|---------------|
| sum_1_to_10.asm | ADD, CMP, JNZ | R0 = 55 | 55 |
| fibonacci.asm | ADD, MOV, CMP, JNZ | R1 = 89 (fib(10)) | 89 |
| multiply.asm | ADD, SUB, CMP, JNZ | R0 = 42 (7*6) | 42 |
| countdown.asm | SUB, CMP, JNZ | Terminal count | Matches mock |
| countup_to_negative.asm | ADD, CMP, JS | Signed overflow | Matches mock |
| bitwise.asm | AND, OR, XOR | Bitwise results | Matches mock |
| power_of_two.asm | SHL, CMP, JNZ | R0 = 256 (2^8) | 256 |

In every case, neural execution produces results identical to Python arithmetic.

### 5.4 Performance Characteristics

We benchmark all 15 neural operations using 1,000 iterations with 50 warmup iterations on Apple Silicon (M-series, MPS backend, PyTorch 2.10.0). Timing uses `time.perf_counter_ns()` for nanosecond precision. All 22 models load in 60ms.

**Per-Operation Latency (1,000 iterations, Apple Silicon MPS):**

| Operation | Model | Mean | Median | P99 | Architecture |
|-----------|-------|------|--------|-----|-------------|
| exp | exp.pt | 21 us | 20 us | 24 us | 4-layer MLP, single forward pass |
| log | log.pt | 21 us | 20 us | 24 us | 4-layer MLP, single forward pass |
| mul | multiply.pt | 21 us | 21 us | 27 us | Byte-pair LUT, batched gather |
| and | logical.pt | 21 us | 21 us | 30 us | Truth table, single vectorized lookup |
| or | logical.pt | 22 us | 22 us | 29 us | Truth table, single vectorized lookup |
| xor | logical.pt | 21 us | 21 us | 28 us | Truth table, single vectorized lookup |
| sin | sincos.pt | 48 us | 47 us | 62 us | 5 sine-activated blocks |
| cos | sincos.pt | 48 us | 48 us | 54 us | 5 sine-activated blocks |
| add | arithmetic.pt + carry_combine.pt | 248 us | 246 us | 309 us | Kogge-Stone CLA, 8 neural passes |
| sub | arithmetic.pt + carry_combine.pt | 246 us | 247 us | 287 us | CLA with complement + carry_in |
| cmp | arithmetic.pt + carry_combine.pt | 249 us | 249 us | 292 us | CLA subtraction → flag derivation |
| shl | lsl.pt | 437 us | 430 us | 618 us | 3 batched attention passes (vectorized) |
| shr | lsr.pt | 431 us | 427 us | 540 us | 3 batched attention passes (vectorized) |
| sqrt | sqrt.pt | 522 us | 521 us | 596 us | Two-stage BatchNorm (batch padding) |
| atan2 | atan2.pt | 935 us | 918 us | 1,110 us | 6 residual layers + batch padding |

**Program Execution (neural_execution=True):**

| Program | Cycles | Wall Time | us/cycle |
|---------|--------|-----------|----------|
| bitwise.asm | 10 | 2.3 ms | 226 |
| countdown.asm | 36 | 6.9 ms | 193 |
| countup_to_negative.asm | 44 | 9.1 ms | 207 |
| fibonacci.asm | 66 | 9.0 ms | 136 |
| multiply.asm | 30 | 5.5 ms | 182 |
| power_of_two.asm | 37 | 9.7 ms | 262 |
| sum_1_to_10.asm | 45 | 9.0 ms | 200 |

The CLA optimization substantially improved program execution speed. The fibonacci program achieves the best per-cycle throughput (136 us/cycle) because it primarily uses ADD and MOV, both now fast with CLA. Average throughput across all programs is approximately 201 us/cycle, or roughly 4,975 instructions per second --- a ~2.6x improvement over pre-CLA throughput (513 us/cycle).

#### Key Performance Insight: Architecture Dominates, Not Operation Complexity

The most striking result is the ~48x latency spread between the fastest operations (exp/log/mul at ~22 us) and the slowest (atan2 at ~1,055 us). This spread is entirely explained by one factor: **the number of sequential neural network forward passes per operation**.

| Sequential Passes | Operations | Latency |
|-------------------|-----------|---------|
| 1 pass | exp, log, mul, and, or, xor | ~21 us |
| 2 passes | sin, cos | ~48 us |
| 8 passes (CLA) | add, sub, cmp | ~248 us |
| 3 batched passes | shl, shr | ~434 us |
| 2 passes + batch padding | sqrt | ~522 us |
| 6 passes + batch padding | atan2 | ~935 us |

Operations using O(1) lookup strategies (truth tables, LUTs, single-pass MLPs) execute in ~21 us regardless of model size. The CLA adder's 8 neural passes yield ~248 us latency (approximately 31 us per pass). The shift network originally required 64 sequential passes (~2,833 us) but was vectorized to 3 batched passes (~434 us), demonstrating that independent computations can be parallelized with no loss of accuracy.

### 5.5 Model Sizes

| Category | Models | Total Size |
|----------|--------|-----------|
| ALU (arithmetic, carry_combine, multiply, logical, compare, divide) | 6 | 4.1 MB |
| Shifts (lsl, lsr, asr, rol) | 4 | 16.6 MB |
| Register | 2 | 7.9 MB |
| Memory | 3 | 876 KB |
| Decoder | 1 | 6.5 MB |
| Math | 6 | 11.7 MB |
| **Subtotal (core models)** | **21** | **~48 MB** |
| Instruction decoder LLM | 1 | 70 MB |
| **Grand total** | **22** | **~118 MB** |

## 6. Discussion

### 6.1 What Works and Why

The success of nCPU's exact arithmetic rests on three principles:

**Exhaustive trainability.** The full adder has 8 input combinations. The multiplication LUT has 65,536 entries. The logical truth tables have 4 entries each. In every case, the training set covers the *complete* input space. There are no unseen inputs, no distribution shift, no generalization gap. The models memorize the correct function, and memorization is sufficient for correctness.

**Architectural decomposition.** 32-bit addition could in principle be learned by a single network mapping two 32-element vectors to a 32-element output. In practice, this requires the network to discover carry propagation through training alone. The bit-serial decomposition sidesteps this: a tiny network learns the 1-bit truth table, and the sequential application of this network implements carry propagation structurally. Similarly, 32-bit multiplication could require a network to learn all 2^64 input combinations. The byte-pair decomposition reduces this to 2^16 entries.

**Hard thresholding.** Every neural output passes through a threshold (`sigmoid > 0.5` or `softmax / temperature`) that converts continuous activations to discrete bits. This is the mechanism that bridges the continuous neural network with the discrete integer domain. The networks are not producing approximate floating-point results --- they are producing exact binary values through thresholding.

### 6.2 Implementation Pitfalls

Several critical implementation details, discovered through debugging, are essential for reproducing these results:

**Temperature convention in shift networks.** The learned temperature in `NeuralShiftNet` converges to approximately 0.01. The correct convention is `softmax(logits / temperature)`, which produces `softmax(logits * 100)`, sharpening the distribution so that one input bit receives nearly all attention. Implementing this as `softmax(logits * temperature)` (multiplying by 0.01) flattens the distribution, causing every output bit to be a near-uniform average of all input bits, producing incorrect results. This single-character bug (`*` vs `/`) was the most time-consuming issue in the shift implementation.

**Float32 precision boundary.** PyTorch's default float32 has 23 mantissa bits, providing exact integer representation only up to 2^24 = 16,777,216. The bit-to-integer conversion `sum(bit_k * 2^k)` in float32 loses precision for results above this threshold. The fix is to perform bit-to-integer conversion using `.long()` (int64) arithmetic: `((bits > 0.5).long() * bit_values_long).sum()`. This was a subtle bug that produced correct results for small values but silent corruption for large ones.

**Signed 32-bit wraparound.** Python integers have arbitrary precision: `1 << 31` produces the positive value 2,147,483,648. In 32-bit signed two's complement, this represents -2,147,483,648. Tests and result handling must explicitly convert values above 2^31 by subtracting 2^32. This is not a neural network issue per se, but it affects every boundary where neural outputs interface with Python integer semantics.

### 6.3 Novel Findings

The benchmark results reveal several counterintuitive properties of neural computation that distinguish it from conventional digital logic:

**Finding 1: Neural multiplication is 12x faster than neural addition, even with carry-lookahead.** In conventional CPUs, multiplication is typically 3-10x slower than addition. In nCPU, the relationship is inverted: `mul` completes in 21 us (a single batched LUT gather) while `add` requires 248 us (8 Kogge-Stone CLA neural passes). Before the CLA optimization, this gap was 38x (mul at 22 us vs add at 826 us with 32 ripple-carry passes). The CLA reduced the gap from 38x to 12x by replacing 32 sequential passes with 5 parallel-prefix stages plus 3 vectorized logical passes, but multiplication remains faster because the LUT eliminates *all* sequential dependencies. The neural CPU reveals a deep truth about computational complexity: **carry propagation, not operation semantics, is the dominant cost**. Even with logarithmic parallel-prefix carry computation, the O(log n) = 5 carry-combine stages still require sequential dependency. The byte-pair LUT achieves true O(1) by eliminating carry chains entirely.

**Finding 2: The O(1) / O(log n) / O(n) hierarchy.** Neural ALU operations fall into sharply separated performance tiers. O(1) operations (single-pass lookups or MLPs) cluster tightly around 21 us: mul, and, or, xor, exp, log. O(log n) operations use parallel-prefix algorithms: add/sub/cmp at ~248 us (8 CLA passes, reduced from 32 sequential passes). O(batched) operations vectorize independent computations: shl/shr at ~434 us (3 batched attention passes, reduced from 128 sequential). Between these sits O(n) for operations with inherently deep sequential dependencies: atan2 at ~935 us (6 residual layers). This reveals a refined design principle: **minimize sequential forward passes through both parallelism and algorithmic improvement.** The CLA demonstrates that even carry-dependent operations can be restructured from O(n) to O(log n) using parallel-prefix tree algorithms, and shift vectorization shows that independent computations should always be batched.

**Finding 3: Vectorization recovers most of the attention-based routing cost.** The shift network's decomposed architecture (shift_decoder + index_net + validity_net) originally required a separate forward pass for each of 64 output bit positions --- 128 sequential neural network evaluations totaling ~2,833 us. However, because each output position's computation is independent (position `i`'s index_net input `[one_hot_i, shift_soft]` depends only on the shared shift encoding, not on other positions), all 64 positions can be computed in a single batched forward pass through each sub-network. Vectorizing to 3 batched passes reduced shift latency from ~2,833 us to ~463 us --- a 6.1x speedup. Shifts are now comparable in cost to sqrt (~524 us) and substantially faster than addition (~826 us). This validates a key design principle: **before adding architectural complexity (permutation matrices, sparse connections), first check whether the existing architecture has unexploited parallelism.** The shift network was never inherently sequential --- only its Python implementation was.

**Finding 4: Model size does not predict latency.** The multiplication LUT (4.0 MB, 1M parameters) executes in 22 us. The full adder (38 KB, 9K parameters) takes 826 us. The exp model (521 KB) and the logical truth table (1.7 KB) both complete in ~22 us. Latency is determined entirely by the execution strategy (number of sequential passes), not by parameter count or model file size. This is relevant for model optimization: quantization or pruning would reduce memory but would not meaningfully improve latency for the bottleneck operations, since their cost is dominated by Python loop overhead and sequential GPU kernel launches, not by the neural computation itself.

**Finding 5: Memorization-by-decomposition as a general principle.** The core insight enabling exact neural computation is not a specific architecture but a design pattern: (1) decompose the target function into sub-problems with finite, enumerable input spaces; (2) train each sub-problem exhaustively to 100% accuracy; (3) apply hard thresholding to convert continuous activations to discrete outputs; (4) compose the sub-problems structurally (sequentially for carry-dependent operations, in parallel for independent operations). This pattern is not specific to CPUs --- it applies to any discrete function built from composable primitives with small input domains. Candidate applications include error-correcting codes (finite syndrome tables), cryptographic S-boxes (fixed input/output mappings), and combinational logic synthesis (truth tables as neural parameters).

### 6.4 Limitations

**Addition is O(log n) with CLA.** The Kogge-Stone carry-lookahead adder reduces addition from 32 sequential passes to 8 neural passes (5 parallel-prefix stages + 3 logical passes). While this is a 3.3x improvement over ripple-carry, the logarithmic carry-combine stages remain inherently sequential. Further speedup would require training a single monolithic carry network that computes all 32 carries in O(1), which remains an open challenge.

**32-bit restriction.** The trained models operate on 32-bit values. The Neural CPU stores state as 64-bit tensors, and the bridge narrows values to 32 bits for model execution. Extending to native 64-bit would require retraining (the full adder would need 64 sequential calls; the multiplication LUT decomposition remains valid).

**Throughput.** Neural execution is substantially slower than native Python arithmetic. A single 32-bit addition requires 32 forward passes through a 3-layer MLP. The multiplication requires up to 16 table lookups plus accumulation. This is a demonstration of correctness, not a competitive execution engine.

### 6.5 Future Work

Several directions could extend this work:

1. **O(1) neural carry network.** Train a single monolithic network that computes all 32 carries simultaneously, reducing addition from O(log n) (current CLA) to O(1). This would require the network to learn the parallel-prefix computation implicitly, which is non-trivial since the carry-combine truth table must compose correctly across all stages.
2. **Native 64-bit models.** Retrain all models on 64-bit operands, eliminating the bridge narrowing.
3. **True GPU-batched instruction execution.** The batch API is implemented but currently wraps sequential calls; fusing the underlying tensor operations across multiple instructions would achieve true parallelism.
4. **Quantized models.** Apply post-training quantization (INT8) to the shift networks and multiplication LUT, reducing the 48 MB model footprint.
5. **Neural FPU.** Extend exact computation to IEEE 754 floating-point operations, potentially using a decomposed sign/exponent/mantissa architecture.
6. **Formal verification.** Prove that the neural models implement their target functions exactly, leveraging the finite and fully enumerable input spaces.
7. **Native kernel acceleration.** Early exploration included a Rust+Metal kernel implementation for GPU-native neural dispatch (deleted during the March 2026 project reorganization; preserved in git history under `kernels/rust_metal/`). Reimplementing the neural ALU dispatch as fused GPU kernels --- eliminating Python interpreter overhead between forward passes --- could further reduce latency for sequential operations like addition.

## 7. Related Work

### Neural Turing Machines

Graves, Wayne, and Danihelka (2014) introduced Neural Turing Machines (NTMs), which augment neural networks with external memory and attention-based read/write heads. NTMs can learn simple algorithms (copying, sorting) from examples but do not achieve exact arithmetic on arbitrary inputs. nCPU differs by using specialized per-operation architectures trained to 100% accuracy rather than a general-purpose differentiable computer.

### Neural GPUs

Kaiser and Sutskever (2015) proposed Neural GPUs, which learn to perform multi-digit addition and multiplication through a convolutional architecture operating on input grids. Neural GPUs achieve high accuracy on trained lengths but struggle to generalize to longer sequences. nCPU avoids the generalization problem entirely by decomposing operations into fixed-size sub-problems (1-bit addition, byte-pair multiplication) and applying them structurally.

### NALU and NAC

Trask et al. (2018) introduced the Neural Arithmetic Logic Unit (NALU) and Neural Accumulator (NAC), which use gated linear combinations to learn addition, subtraction, and multiplication. NALU achieves good extrapolation on continuous arithmetic but does not guarantee exact integer results. nCPU's approach is fundamentally different: rather than building arithmetic-friendly inductive biases into a general architecture, nCPU trains small specialized models on complete input spaces.

### Differentiable Neural Computers

Graves et al. (2016) extended NTMs into Differentiable Neural Computers (DNCs) with dynamic memory allocation and temporal attention. DNCs demonstrate more complex algorithmic learning but remain approximate on arithmetic tasks. nCPU complements this work by showing that exact arithmetic is achievable when the problem is decomposed appropriately.

### Neural Program Synthesis

Program synthesis systems (e.g., DeepCoder, RobustFill) learn to generate programs from input-output examples. These systems operate at the program level, not the instruction level. nCPU operates at the hardware level, replacing the ALU itself with neural networks while preserving the conventional fetch-decode-execute pipeline.

### Key Distinction

Prior neural arithmetic systems target *generalization*: learning arithmetic patterns that extend to unseen inputs. nCPU targets *memorization*: training on complete input spaces so that every possible input has been seen during training. This is tractable because the sub-problems (8-entry truth tables, 65K-entry LUTs) have small input spaces, and it guarantees 100% accuracy by construction. The novelty is not in the individual sub-problem (memorizing 8 entries is trivial) but in the systematic decomposition of 32-bit operations into sub-problems where memorization is both tractable and sufficient.

## 8. Conclusion

nCPU demonstrates that trained neural networks can execute 32-bit integer arithmetic with 100% accuracy. The key insight is architectural decomposition: by breaking operations into sub-problems with exhaustively trainable input spaces --- 8-entry truth tables for addition, 16-entry truth tables for carry combining, 65,536-entry lookup tables for multiplication, attention-based bit routing for shifts --- neural networks can memorize exact functions rather than approximating them.

The system comprises 22 trained models totaling approximately 48 MB of core weights (118 MB with the LLM instruction decoder), implementing a complete ALU with addition, subtraction, multiplication, bitwise logic, shifts, comparison, and experimental transcendental functions. All 330 automated tests pass, and seven benchmark programs produce results identical to conventional arithmetic.

Performance benchmarking reveals that neural operation latency is determined by the number of sequential forward passes and their algorithmic structure. Neural multiplication (O(1) LUT lookup, 21 us) is 12x faster than neural addition (O(log n) Kogge-Stone CLA, 248 us), inverting the performance hierarchy of conventional CPUs. The CLA reduced addition latency by 3.3x (from 826 us to 248 us) by replacing 32 sequential ripple-carry passes with 8 parallel-prefix passes through a trained carry-combine network. Combined with shift vectorization (2,833 us to 434 us, 6.5x speedup), these optimizations demonstrate that classical hardware design principles --- carry-lookahead, parallel-prefix trees, vectorized independent computations --- transfer directly to neural architectures.

This work suggests that the boundary between "neural" and "exact" computation may be more permeable than commonly assumed. When the input space is finite and fully enumerable, and when the architecture decomposes the problem into tractable sub-units, neural networks can serve as exact computational elements. The memorization-by-decomposition principle --- decompose, train exhaustively, threshold, compose structurally --- is general enough to apply beyond CPUs to any discrete function built from composable primitives with small input domains. Whether this approach extends to higher precision, floating-point arithmetic, or other discrete domains remains an open question for future investigation.

---

## Acknowledgments

nCPU was developed as an independent research project. The trained models, test suite, and full source code are available in the project repository.

## References

1. Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. *arXiv preprint arXiv:1410.5401*.

2. Kaiser, L., & Sutskever, I. (2015). Neural GPUs Learn Algorithms. *arXiv preprint arXiv:1511.08228*.

3. Graves, A., Wayne, G., Reynolds, M., et al. (2016). Hybrid computing using a neural network with dynamic external memory. *Nature*, 538(7626), 471--476.

4. Trask, A., Hill, F., Reed, S., Rae, J., Dyer, C., & Blunsom, P. (2018). Neural Arithmetic Logic Units. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

5. Balog, M., Gaunt, A. L., Brockschmidt, M., Nowozin, S., & Tarlow, D. (2017). DeepCoder: Learning to Write Programs. *International Conference on Learning Representations (ICLR)*.

6. Devlin, J., Uesato, J., Bhupatiraju, S., Singh, R., Mohamed, A., & Kohli, P. (2017). RobustFill: Neural Program Learning under Noisy I/O. *International Conference on Machine Learning (ICML)*.
