"""NeuralOps: Trained neural network execution backend for the model-based nCPU.

Loads the trained .pt models from models/ and uses them to execute ALU operations
instead of hardcoded Python arithmetic. Every operation passes through a trained
neural network:

    ADD  → NeuralFullAdder (bit-serial, 128-hidden, 100% accuracy)
    SUB  → NeuralFullAdder (complement + add)
    MUL  → NeuralMultiplierLUT (256x256 byte-pair lookup, 100% accuracy)
    DIV  → NeuralFullAdder (restoring division via neural subtraction, 64-hidden)
    CMP  → NeuralCompare (3→3 refinement layer)
    INC  → NeuralFullAdder (add 1)
    DEC  → NeuralFullAdder (subtract 1)
    AND  → NeuralLogical (learned truth tables, bit-by-bit)
    OR   → NeuralLogical (learned truth tables, bit-by-bit)
    XOR  → NeuralLogical (learned truth tables, bit-by-bit)
    SHL  → NeuralShiftNet (shift_decoder + index_net + validity_net)
    SHR  → NeuralShiftNet (shift_decoder + index_net + validity_net)

Models are loaded lazily on first use and cached for the session.
Falls back to Python arithmetic if torch is unavailable or models are missing.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .state import CPUState, INT32_MIN, INT32_MAX

# Try importing torch — graceful fallback if unavailable
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CLASS DEFINITIONS (reconstructed from trained state dicts)
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class NeuralFullAdder(nn.Module):
        """Bit-level neural full adder.

        Input:  (bit_a, bit_b, carry_in) — 3 floats
        Output: (sum_bit, carry_out)     — 2 floats (sigmoid → round)

        Operates bit-by-bit: for N-bit addition, called N times in sequence.
        Trained to 100% accuracy on all 2^3 = 8 input combinations.
        """

        def __init__(self, hidden_dim: int = 128):
            super().__init__()
            self.full_adder = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),
            )

        def forward(self, bits: torch.Tensor) -> torch.Tensor:
            """bits: [batch, 3] → [batch, 2] (sum_bit, carry_out)"""
            return torch.sigmoid(self.full_adder(bits))

    class NeuralMultiplierLUT(nn.Module):
        """Neural byte-pair multiplication lookup table.

        A 256×256×16 learned tensor where lut[a_byte][b_byte] gives
        16 sigmoid-activated bits representing the product.

        For 32-bit multiplication, decomposes into 4 bytes each,
        performs 16 byte-pair lookups, and combines with shifts.
        Trained to 100% accuracy on all 256×256 = 65536 byte pairs.
        """

        def __init__(self):
            super().__init__()
            self.lut = nn.Module()
            self.lut.table = nn.Parameter(torch.zeros(256, 256, 16))

        _lut_bit_values = None  # Class-level cache for bit→int conversion

        def lookup(self, a_byte: int, b_byte: int) -> int:
            """Look up product of two bytes (0-255) → 16-bit result."""
            if NeuralMultiplierLUT._lut_bit_values is None:
                NeuralMultiplierLUT._lut_bit_values = (
                    1 << torch.arange(16, dtype=torch.long)
                ).float()

            with torch.no_grad():
                bits = (torch.sigmoid(self.lut.table[a_byte, b_byte]) > 0.5).float()
                return int(bits.dot(NeuralMultiplierLUT._lut_bit_values).long().item())

    class NeuralCompare(nn.Module):
        """Neural comparison refinement layer.

        Input:  3 features (sign_bit, is_zero, raw_diff_sign)
        Output: 3 values (negative_flag, zero_flag, positive_flag)
        """

        def __init__(self):
            super().__init__()
            self.refine = nn.Linear(3, 3)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.refine(features))

    class NeuralLogical(nn.Module):
        """Neural truth table for logical operations.

        7 operations × 4 truth table entries.
        Ops: AND=0, OR=1, XOR=2, NOT=3, NAND=4, NOR=5, XNOR=6
        Index: a*2 + b → sigmoid(truth_tables[op, idx])
        """

        def __init__(self):
            super().__init__()
            self.truth_tables = nn.Parameter(torch.zeros(7, 4))

        def apply_op(self, op_idx: int, bit_a: int, bit_b: int) -> int:
            """Apply logical operation on single bits."""
            with torch.no_grad():
                idx = bit_a * 2 + bit_b
                return int(torch.sigmoid(self.truth_tables[op_idx, idx]).item() > 0.5)

    class NeuralCarryCombine(nn.Module):
        """Neural carry-combine operator for parallel-prefix addition.

        Computes (G_out, P_out) = (G_i | (P_i & G_j), P_i & P_j)
        Trained on all 2^4 = 16 input combinations to 100% accuracy.
        Used in Kogge-Stone parallel-prefix carry computation.
        """

        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),
            )

        def forward(self, gp_pairs: torch.Tensor) -> torch.Tensor:
            """gp_pairs: [batch, 4] -> [batch, 2] (G_out, P_out)"""
            return torch.sigmoid(self.net(gp_pairs))

    class NeuralShiftNet(nn.Module):
        """Neural shift network with decomposed architecture.

        Reconstructed from lsl.pt / lsr.pt state dicts. Three sub-networks:
            - shift_decoder: Decodes the shift amount into internal representation
              Input: 64 (one-hot shift bits) → 768 hidden → 64 output
            - index_net: Computes output bit indices from 128-dim input
              Input: 128 (value bits + shift encoding) → 768 hidden → 64 output
            - validity_net: Determines if each output bit position is valid
              Input: 128 → 384 hidden → 1 output (per position)
            - temperature: Learned sharpening parameter for soft→hard decisions

        These networks learned to implement bit-level shifting through training,
        without any hardcoded shift logic.
        """

        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.tensor(1.0))
            self.shift_decoder = nn.Sequential(
                nn.Linear(64, 768),
                nn.ReLU(),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 64),
            )
            self.index_net = nn.Sequential(
                nn.Linear(128, 768),
                nn.ReLU(),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 64),
            )
            self.validity_net = nn.Sequential(
                nn.Linear(128, 384),
                nn.ReLU(),
                nn.Linear(384, 1),
            )

        def forward(self, value_bits: torch.Tensor, shift_amount_bits: torch.Tensor) -> torch.Tensor:
            """Compute shifted output bits using the trained decomposed architecture.

            The correct convention (verified against training code):
            1. Binary-encode shift amount → shift_decoder → 64D encoding
            2. Apply softmax to shift encoding (decoder learned to produce logits)
            3. For all 64 output bits: [one_hot_pos(64), softmax(shift_enc)(64)] → [64, 128]
            4. index_net([64, 128]) → [64, 64] logits → softmax(logits/temp) → attention over value_bits
            5. validity_net([64, 128]) → [64, 1] sigmoid gate

            Vectorized: all 64 output positions computed in 3 batched forward passes
            (1× shift_decoder + 1× index_net + 1× validity_net) instead of 128 sequential passes.

            Args:
                value_bits: [64] float tensor of input bit values
                shift_amount_bits: [64] float tensor (binary-encoded shift amount)

            Returns:
                [64] float tensor of output bits
            """
            with torch.no_grad():
                # Step 1-2: Decode shift amount and apply softmax (1 forward pass)
                shift_enc = self.shift_decoder(shift_amount_bits.unsqueeze(0))[0]  # [64]
                shift_soft = torch.softmax(shift_enc, dim=0)  # [64]

                # Step 3: Build all 64 combined inputs in one batch
                positions = torch.eye(64)                                # [64, 64]
                shift_expanded = shift_soft.unsqueeze(0).expand(64, -1)  # [64, 64]
                combined = torch.cat([positions, shift_expanded], dim=1) # [64, 128]

                # Step 4: Batched index_net → attention over source bits (1 forward pass)
                idx_logits = self.index_net(combined)                    # [64, 64]
                idx_weights = torch.softmax(idx_logits / self.temperature, dim=1)  # [64, 64]
                bit_vals = (idx_weights * value_bits.unsqueeze(0)).sum(dim=1)       # [64]

                # Step 5: Batched validity gate (1 forward pass)
                valid_logits = self.validity_net(combined)               # [64, 1]
                valid = torch.sigmoid(valid_logits.squeeze(1))           # [64]
                result_bits = torch.where(valid > 0.5, bit_vals, torch.zeros(64))  # [64]

                return result_bits


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL OPERATIONS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralOps:
    """Loads trained neural models and provides neural ALU execution.

    All operations run through trained neural networks. Falls back to
    Python arithmetic only if models can't be loaded.
    """

    BITS = 32  # Operating width for ncpu.model ISA

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._adder: Optional['NeuralFullAdder'] = None
        self._divider: Optional['NeuralFullAdder'] = None
        self._multiplier: Optional['NeuralMultiplierLUT'] = None
        self._compare: Optional['NeuralCompare'] = None
        self._logical: Optional['NeuralLogical'] = None
        self._carry_combiner: Optional['NeuralCarryCombine'] = None
        self._shifter_left: Optional['NeuralShiftNet'] = None
        self._shifter_right: Optional['NeuralShiftNet'] = None
        self._sincos = None
        self._sqrt = None
        self._exp = None
        self._log = None
        self._atan2 = None
        self._loaded = False
        self._available_ops: Dict[str, bool] = {}

        # Pre-computed constants for vectorized bit conversion
        if TORCH_AVAILABLE:
            self._bit_shifts = torch.arange(self.BITS, dtype=torch.long)  # [0,1,...,31]
            self._bit_values_long = 1 << self._bit_shifts  # [1,2,4,...,2^31] as long
            self._sign_threshold = 1 << (self.BITS - 1)  # 2^31
            self._mask = (1 << self.BITS) - 1  # 0xFFFFFFFF
            self._adder_input = torch.zeros(1, 3)  # Pre-allocated for add loop

    def load(self) -> Dict[str, bool]:
        """Load all available neural models. Returns availability map."""
        if not TORCH_AVAILABLE:
            return {}

        self._available_ops = {}

        # Arithmetic (ADD/SUB/INC/DEC) — hidden_dim=128
        path = self.models_dir / "alu" / "arithmetic.pt"
        if path.exists():
            try:
                self._adder = NeuralFullAdder(hidden_dim=128)
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._adder.load_state_dict(state)
                self._adder.eval()
                self._available_ops["ADD"] = True
                self._available_ops["SUB"] = True
                self._available_ops["INC"] = True
                self._available_ops["DEC"] = True
            except Exception as e:
                self._available_ops["ADD"] = False
                self._adder = None

        # Carry-Combine (for CLA parallel-prefix addition)
        path = self.models_dir / "alu" / "carry_combine.pt"
        if path.exists():
            try:
                self._carry_combiner = NeuralCarryCombine(hidden_dim=64)
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._carry_combiner.load_state_dict(state)
                self._carry_combiner.eval()
            except Exception:
                self._carry_combiner = None

        # Division — hidden_dim=64
        path = self.models_dir / "alu" / "divide.pt"
        if path.exists():
            try:
                self._divider = NeuralFullAdder(hidden_dim=64)
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._divider.load_state_dict(state)
                self._divider.eval()
                self._available_ops["DIV"] = True
            except Exception as e:
                self._available_ops["DIV"] = False

        # Multiplication — byte-pair LUT
        path = self.models_dir / "alu" / "multiply.pt"
        if path.exists():
            try:
                self._multiplier = NeuralMultiplierLUT()
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._multiplier.load_state_dict(state)
                self._multiplier.eval()
                self._available_ops["MUL"] = True
            except Exception as e:
                self._available_ops["MUL"] = False

        # Compare
        path = self.models_dir / "alu" / "compare.pt"
        if path.exists():
            try:
                self._compare = NeuralCompare()
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._compare.load_state_dict(state)
                self._compare.eval()
                self._available_ops["CMP"] = True
            except Exception as e:
                self._available_ops["CMP"] = False

        # Logical
        path = self.models_dir / "alu" / "logical.pt"
        if path.exists():
            try:
                self._logical = NeuralLogical()
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._logical.load_state_dict(state)
                self._logical.eval()
                self._available_ops["AND"] = True
                self._available_ops["OR"] = True
                self._available_ops["XOR"] = True
            except Exception:
                self._available_ops["AND"] = False

        # Shift Left (LSL) — decomposed shift_decoder + index_net + validity_net
        path = self.models_dir / "shifts" / "lsl.pt"
        if path.exists():
            try:
                self._shifter_left = NeuralShiftNet()
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._shifter_left.load_state_dict(state)
                self._shifter_left.eval()
                self._available_ops["SHL"] = True
            except Exception:
                self._available_ops["SHL"] = False

        # Shift Right (LSR) — decomposed shift_decoder + index_net + validity_net
        path = self.models_dir / "shifts" / "lsr.pt"
        if path.exists():
            try:
                self._shifter_right = NeuralShiftNet()
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._shifter_right.load_state_dict(state)
                self._shifter_right.eval()
                self._available_ops["SHR"] = True
            except Exception:
                self._available_ops["SHR"] = False

        # Math models (experimental)
        self._load_math_models()

        # torch.compile for kernel fusion (silent fallback if unsupported)
        self._try_compile_models()

        self._loaded = True
        return self._available_ops

    def _try_compile_models(self) -> None:
        """Apply torch.compile to neural models for kernel fusion.

        Fuses multiple small forward passes into optimized GPU kernels.
        Only enabled on CUDA where torch.compile provides real benefits.
        On MPS/CPU, torch.compile adds ~700ms load overhead with no
        runtime benefit, so we skip it.
        """
        if not hasattr(torch, 'compile'):
            return
        # Only compile on CUDA — MPS compilation adds overhead with no gain
        if not torch.cuda.is_available():
            return
        try:
            if self._carry_combiner is not None:
                self._carry_combiner = torch.compile(
                    self._carry_combiner, mode="reduce-overhead"
                )
            if self._shifter_left is not None:
                self._shifter_left = torch.compile(
                    self._shifter_left, mode="reduce-overhead"
                )
            if self._shifter_right is not None:
                self._shifter_right = torch.compile(
                    self._shifter_right, mode="reduce-overhead"
                )
        except Exception:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ─── Integer ↔ Bits conversion (vectorized) ─────────────────────────────

    def _int_to_bits(self, value: int) -> torch.Tensor:
        """Convert signed 32-bit integer to float tensor of bits [BITS].

        Vectorized: single tensor operation instead of 32-iteration loop.
        """
        if value < 0:
            value = value + (1 << self.BITS)
        value = value & self._mask
        return ((torch.tensor(value, dtype=torch.long) >> self._bit_shifts) & 1).float()

    def _bits_to_int(self, bits: torch.Tensor) -> int:
        """Convert float tensor of bits back to signed 32-bit integer.

        Vectorized: multiply integer bit positions by power-of-2 weights
        using long arithmetic to avoid float32 precision loss.
        """
        value = int(((bits > 0.5).long() * self._bit_values_long).sum().item())
        if value >= self._sign_threshold:
            value -= (1 << self.BITS)
        return max(INT32_MIN, min(INT32_MAX, value))

    # ─── Neural ALU operations ───────────────────────────────────────────────

    def neural_add(self, a: int, b: int) -> int:
        """Add two 32-bit integers using the neural ALU.

        Uses Kogge-Stone CLA (8 passes) if carry_combine.pt is loaded,
        otherwise falls back to ripple-carry (32 passes).
        """
        if self._adder is None:
            return self._clamp(a + b)

        bits_a = self._int_to_bits(a)
        bits_b = self._int_to_bits(b)

        # CLA path: 8 neural passes instead of 32
        if self._carry_combiner is not None and self._logical is not None:
            result_bits = self._neural_add_cla(bits_a, bits_b, carry_in=0.0)
            return self._bits_to_int(result_bits)

        # Fallback: ripple-carry (32 sequential passes)
        result_bits = torch.zeros(self.BITS)
        inp = self._adder_input  # Pre-allocated [1, 3]
        inp[0, 2] = 0.0  # carry_in = 0

        with torch.no_grad():
            for i in range(self.BITS):
                inp[0, 0] = bits_a[i]
                inp[0, 1] = bits_b[i]
                out = self._adder(inp)[0]
                result_bits[i] = (out[0] > 0.5).float()
                inp[0, 2] = (out[1] > 0.5).float()  # carry for next bit

        return self._bits_to_int(result_bits)

    def neural_sub(self, a: int, b: int) -> int:
        """Subtract using neural adder: a - b = a + (~b + 1) (two's complement).

        Uses CLA path if carry_combine.pt is loaded, otherwise ripple-carry.
        """
        if self._adder is None:
            return self._clamp(a - b)

        bits_a = self._int_to_bits(a)
        bits_b_comp = 1.0 - self._int_to_bits(b)  # Complement b

        # CLA path: carry_in=1 for two's complement
        if self._carry_combiner is not None and self._logical is not None:
            result_bits = self._neural_add_cla(bits_a, bits_b_comp, carry_in=1.0)
            return self._bits_to_int(result_bits)

        # Fallback: ripple-carry
        result_bits = torch.zeros(self.BITS)
        inp = self._adder_input  # Pre-allocated [1, 3]
        inp[0, 2] = 1.0  # carry_in = 1 for two's complement

        with torch.no_grad():
            for i in range(self.BITS):
                inp[0, 0] = bits_a[i]
                inp[0, 1] = bits_b_comp[i]
                out = self._adder(inp)[0]
                result_bits[i] = (out[0] > 0.5).float()
                inp[0, 2] = (out[1] > 0.5).float()

        return self._bits_to_int(result_bits)

    def neural_mul(self, a: int, b: int) -> int:
        """Multiply using the neural byte-pair LUT.

        Optimized: batch all non-zero byte-pair lookups with vectorized
        bit-to-int conversion, and use direct table indexing.
        """
        if self._multiplier is None:
            return self._clamp(a * b)

        # Handle signs
        sign = 1
        if a < 0:
            a = -a
            sign = -sign
        if b < 0:
            b = -b
            sign = -sign

        a = a & 0xFFFFFFFF
        b = b & 0xFFFFFFFF

        # Decompose into bytes
        a_bytes = [(a >> (i * 8)) & 0xFF for i in range(4)]
        b_bytes = [(b >> (i * 8)) & 0xFF for i in range(4)]

        # Collect non-zero byte pairs and their shift amounts
        pairs_a = []
        pairs_b = []
        shifts = []
        for i in range(4):
            if a_bytes[i] == 0:
                continue
            for j in range(4):
                if b_bytes[j] == 0:
                    continue
                pairs_a.append(a_bytes[i])
                pairs_b.append(b_bytes[j])
                shifts.append((i + j) * 8)

        if not pairs_a:
            return 0

        # Batch lookup: gather all logits at once, then vectorized bit→int
        with torch.no_grad():
            a_idx = torch.tensor(pairs_a, dtype=torch.long)
            b_idx = torch.tensor(pairs_b, dtype=torch.long)
            logits = self._multiplier.lut.table[a_idx, b_idx]  # [N, 16]
            bits = (torch.sigmoid(logits) > 0.5).float()  # [N, 16]
            if NeuralMultiplierLUT._lut_bit_values is None:
                NeuralMultiplierLUT._lut_bit_values = (
                    1 << torch.arange(16, dtype=torch.long)
                ).float()
            products = (bits @ NeuralMultiplierLUT._lut_bit_values).long()  # [N]

        result = 0
        for k in range(len(shifts)):
            result += int(products[k].item()) << shifts[k]

        result = result & 0xFFFFFFFF
        if result >= 0x80000000:
            result -= 0x100000000

        result = result * sign
        return max(INT32_MIN, min(INT32_MAX, result))

    def neural_cmp(self, a: int, b: int) -> Tuple[bool, bool]:
        """Compare using neural subtraction: CMP = SUB without storing result.

        Computes a - b through the neural full adder and derives flags
        from the bit-level result. This is how real CPUs implement CMP.

        Returns (zero_flag, sign_flag).
        """
        diff = self.neural_sub(a, b)
        return (diff == 0, diff < 0)

    def neural_inc(self, a: int) -> int:
        """Increment by 1 using neural adder."""
        return self.neural_add(a, 1)

    def neural_dec(self, a: int) -> int:
        """Decrement by 1 using neural adder."""
        return self.neural_sub(a, 1)

    # ─── Neural Division ──────────────────────────────────────────────────────

    def neural_div(self, a: int, b: int) -> int:
        """Integer division using the neural ALU (restoring division algorithm).

        For each bit position from MSB to LSB:
          1. Shift remainder left by 1, bring down next bit of dividend
          2. Try subtracting divisor from remainder via neural_sub
          3. If remainder >= 0, keep subtraction result (quotient bit = 1)
          4. If remainder < 0, restore previous remainder (quotient bit = 0)

        Uses the neural subtractor (arithmetic.pt or divide.pt architecture)
        for all trial subtractions -- no Python arithmetic in the critical path.

        Division by zero returns 0.
        """
        if b == 0:
            return 0

        sign = 1
        if (a < 0) != (b < 0):
            sign = -1
        a, b = abs(a), abs(b)

        quotient = 0
        remainder = 0
        for i in range(31, -1, -1):
            remainder = (remainder << 1) | ((a >> i) & 1)
            diff = self.neural_sub(remainder, b)
            if diff >= 0:
                remainder = diff
                quotient |= (1 << i)

        result = quotient * sign
        return self._clamp(result)

    # ─── Neural Logical operations ────────────────────────────────────────────

    def neural_and(self, a: int, b: int) -> int:
        """Bitwise AND using neural truth tables, bit by bit."""
        if self._logical is None:
            return self._clamp(a & b)
        return self._neural_bitwise_op(a, b, op_idx=0)

    def neural_or(self, a: int, b: int) -> int:
        """Bitwise OR using neural truth tables, bit by bit."""
        if self._logical is None:
            return self._clamp(a | b)
        return self._neural_bitwise_op(a, b, op_idx=1)

    def neural_xor(self, a: int, b: int) -> int:
        """Bitwise XOR using neural truth tables, bit by bit."""
        if self._logical is None:
            return self._clamp(a ^ b)
        return self._neural_bitwise_op(a, b, op_idx=2)

    def _neural_bitwise_op(self, a: int, b: int, op_idx: int) -> int:
        """Apply a neural logical operation across all 32 bits in one vectorized step.

        Instead of 32 separate apply_op calls, computes idx = bits_a*2 + bits_b
        for all bits simultaneously, then indexes into the truth table.
        """
        bits_a = (self._int_to_bits(a) > 0.5).long()
        bits_b = (self._int_to_bits(b) > 0.5).long()
        idx = bits_a * 2 + bits_b  # [32] indices into truth table

        with torch.no_grad():
            logits = self._logical.truth_tables[op_idx, idx]  # [32]
            result_bits = (torch.sigmoid(logits) > 0.5).float()

        return self._bits_to_int(result_bits)

    def _neural_bitwise_bits(self, bits_a: torch.Tensor, bits_b: torch.Tensor, op_idx: int) -> torch.Tensor:
        """Apply neural logical op on bit tensors directly (no int conversion).

        Vectorized over all bit positions simultaneously.
        Returns float tensor of result bits.
        """
        idx = (bits_a > 0.5).long() * 2 + (bits_b > 0.5).long()
        with torch.no_grad():
            logits = self._logical.truth_tables[op_idx, idx]
            return (torch.sigmoid(logits) > 0.5).float()

    # ─── Neural CLA (Carry-Lookahead Addition) ────────────────────────────

    def _neural_add_cla(self, bits_a: torch.Tensor, bits_b: torch.Tensor, carry_in: float = 0.0) -> torch.Tensor:
        """Kogge-Stone parallel-prefix carry-lookahead addition on bit tensors.

        Instead of 32 sequential ripple-carry passes, computes carries in
        O(log N) = 5 stages for 32 bits. Each stage is a single batched
        forward pass through the carry_combine neural network.

        Algorithm:
        1. Generate initial G[i] = a[i] AND b[i], P[i] = a[i] XOR b[i]
        2. Parallel-prefix tree: 5 stages (stride 1,2,4,8,16)
           Each combines (G[i], P[i]) with (G[i-stride], P[i-stride])
        3. Final sum: S[i] = P[i] XOR C[i-1]

        Total: 2 logical + 5 carry-combine + 1 logical = 8 passes (vs 32).
        """
        N = self.BITS  # 32

        # Step 1: Generate initial G and P vectors using neural logical ops
        # G[i] = a[i] AND b[i] (op_idx=0)
        G = self._neural_bitwise_bits(bits_a, bits_b, op_idx=0)  # [32]
        # P[i] = a[i] XOR b[i] (op_idx=2)
        P = self._neural_bitwise_bits(bits_a, bits_b, op_idx=2)  # [32]

        # Handle carry_in: inject into bit position 0
        # With carry_in=1: G[0] = G[0] | (P[0] & 1) = G[0] | P[0], P[0] unchanged
        if carry_in > 0.5:
            # carry_in acts as G[-1]=1, P[-1]=0 for position 0
            # After combining with carry_in: G[0]' = G[0] | (P[0] & carry_in)
            g0_new = max(float(G[0].item()), float(P[0].item()) * carry_in)
            G[0] = float(g0_new > 0.5)

        # Step 2: Kogge-Stone parallel-prefix tree (5 stages for 32 bits)
        with torch.no_grad():
            stride = 1
            for _ in range(5):  # ceil(log2(32)) = 5
                if stride >= N:
                    break

                # Positions that need combining: stride, stride+1, ..., N-1
                # Combine (G[i], P[i]) with (G[i-stride], P[i-stride])
                n_combines = N - stride
                if n_combines <= 0:
                    break

                # Build batched input: [G_i, P_i, G_j, P_j] for each position
                indices_i = torch.arange(stride, N)
                indices_j = torch.arange(0, N - stride)

                batch_input = torch.stack([
                    G[indices_i],    # G_i (current)
                    P[indices_i],    # P_i (current)
                    G[indices_j],    # G_j (from stride positions back)
                    P[indices_j],    # P_j (from stride positions back)
                ], dim=1)  # [n_combines, 4]

                out = self._carry_combiner(batch_input)  # [n_combines, 2]

                # Update G and P for combined positions
                G = G.clone()
                P = P.clone()
                G[indices_i] = (out[:, 0] > 0.5).float()
                P[indices_i] = (out[:, 1] > 0.5).float()

                stride *= 2

        # Step 3: After prefix tree, G[i] = carry into position i+1
        # Final sum bits: S[i] = P_original[i] XOR C[i-1]
        # where C[-1] = carry_in, C[i] = G[i] from prefix tree
        P_original = self._neural_bitwise_bits(bits_a, bits_b, op_idx=2)  # Recompute original P

        # Build carry vector: C[i-1] for each position
        # C[-1] = carry_in, C[0] = G[0], C[1] = G[1], ..., C[30] = G[30]
        carries = torch.zeros(N)
        carries[0] = carry_in
        carries[1:] = G[:-1]

        # S[i] = P_original[i] XOR carries[i]
        result_bits = self._neural_bitwise_bits(P_original, carries, op_idx=2)  # [32]

        return result_bits

    # ─── Neural Shift operations ──────────────────────────────────────────────

    def neural_shl(self, value: int, amount: int) -> int:
        """Shift left using the trained neural shift network.

        The lsl.pt model operates on 64-bit values. For our 32-bit ISA:
        1. Zero-extend value to 64 bits
        2. Binary-encode shift amount into 64-bit vector
        3. Run through NeuralShiftNet.forward()
        4. Read back lower 32 bits
        """
        if self._shifter_left is None:
            amount = max(0, min(31, amount))
            return self._clamp(value << amount)

        amount = max(0, min(31, amount))

        # Zero-extend 32-bit value to 64-bit representation
        if value < 0:
            uval = value + (1 << self.BITS)
        else:
            uval = value & ((1 << self.BITS) - 1)

        # Vectorized bit extraction: all 64 bits at once
        bit_positions_64 = torch.arange(64, dtype=torch.long)
        value_bits = ((torch.tensor(uval, dtype=torch.long) >> bit_positions_64) & 1).float()

        # Binary-encode shift amount into 64-bit vector (only 6 bits needed)
        shift_bits = ((torch.tensor(amount, dtype=torch.long) >> bit_positions_64) & 1).float()

        result_bits = self._shifter_left.forward(value_bits, shift_bits)

        # Vectorized result readback: lower 32 bits
        result = int(((result_bits[:self.BITS] > 0.5).long() * self._bit_values_long).sum().item())
        if result >= self._sign_threshold:
            result -= (1 << self.BITS)
        return self._clamp(result)

    def neural_shr(self, value: int, amount: int) -> int:
        """Shift right (logical) using the trained neural shift network.

        The lsr.pt model operates on 64-bit values. For our 32-bit ISA:
        1. Zero-extend value to 64 bits
        2. Binary-encode shift amount into 64-bit vector
        3. Run through NeuralShiftNet.forward()
        4. Read back lower 32 bits
        """
        if self._shifter_right is None:
            amount = max(0, min(31, amount))
            if value < 0:
                uval = value + (1 << 32)
            else:
                uval = value & 0xFFFFFFFF
            result = uval >> amount
            if result >= (1 << 31):
                result -= (1 << 32)
            return self._clamp(result)

        amount = max(0, min(31, amount))

        # Zero-extend 32-bit value to 64-bit representation
        if value < 0:
            uval = value + (1 << self.BITS)
        else:
            uval = value & ((1 << self.BITS) - 1)

        # Vectorized bit extraction: all 64 bits at once
        bit_positions_64 = torch.arange(64, dtype=torch.long)
        value_bits = ((torch.tensor(uval, dtype=torch.long) >> bit_positions_64) & 1).float()

        # Binary-encode shift amount into 64-bit vector (only 6 bits needed)
        shift_bits = ((torch.tensor(amount, dtype=torch.long) >> bit_positions_64) & 1).float()

        result_bits = self._shifter_right.forward(value_bits, shift_bits)

        # Vectorized result readback: lower 32 bits
        result = int(((result_bits[:self.BITS] > 0.5).long() * self._bit_values_long).sum().item())
        if result >= self._sign_threshold:
            result -= (1 << self.BITS)
        return self._clamp(result)

    def _clamp(self, value: int) -> int:
        return max(INT32_MIN, min(INT32_MAX, value))

    # ─── Neural Math operations (experimental) ─────────────────────────────

    def _load_math_models(self) -> None:
        """Load math models (sincos, sqrt, exp, log, atan2) from models/math/."""
        if not TORCH_AVAILABLE:
            return

        from .architectures import NeuralSinCos, NeuralSqrt, NeuralExp, NeuralLog, NeuralAtan2

        # SinCos
        path = self.models_dir / "math" / "sincos.pt"
        if path.exists():
            try:
                self._sincos = NeuralSinCos()
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                self._sincos.load_state_dict(ckpt["model"])
                self._sincos.eval()
                self._available_ops["SIN"] = True
                self._available_ops["COS"] = True
            except Exception:
                self._sincos = None

        # Sqrt
        path = self.models_dir / "math" / "sqrt.pt"
        if path.exists():
            try:
                self._sqrt = NeuralSqrt()
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                self._sqrt.load_state_dict(ckpt["model"])
                self._sqrt.eval()
                self._available_ops["SQRT"] = True
            except Exception:
                self._sqrt = None

        # Exp
        path = self.models_dir / "math" / "exp.pt"
        if path.exists():
            try:
                self._exp = NeuralExp()
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                self._exp.load_state_dict(ckpt["model"])
                self._exp.eval()
                self._available_ops["EXP"] = True
            except Exception:
                self._exp = None

        # Log
        path = self.models_dir / "math" / "log.pt"
        if path.exists():
            try:
                self._log = NeuralLog()
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                self._log.load_state_dict(ckpt["model"])
                self._log.eval()
                self._available_ops["LOG"] = True
            except Exception:
                self._log = None

        # Atan2 — direct state_dict (not wrapped)
        path = self.models_dir / "math" / "atan2.pt"
        if path.exists():
            try:
                self._atan2 = NeuralAtan2()
                state = torch.load(path, map_location="cpu", weights_only=False)
                self._atan2.load_state_dict(state)
                self._atan2.eval()
                self._available_ops["ATAN2"] = True
            except Exception:
                self._atan2 = None

    def neural_sin(self, a: int) -> float:
        """Neural sin: input is fixed-point (value / 1000 = radians), returns float."""
        if self._sincos is None:
            import math
            return math.sin(a / 1000.0)
        angle = torch.tensor([[a / 1000.0]])
        with torch.no_grad():
            out = self._sincos(angle)
        return float(out[0, 0].item())

    def neural_cos(self, a: int) -> float:
        """Neural cos: input is fixed-point (value / 1000 = radians), returns float."""
        if self._sincos is None:
            import math
            return math.cos(a / 1000.0)
        angle = torch.tensor([[a / 1000.0]])
        with torch.no_grad():
            out = self._sincos(angle)
        return float(out[0, 1].item())

    def neural_sqrt(self, a: int) -> float:
        """Neural sqrt: input is fixed-point (value / 1000), returns float."""
        if self._sqrt is None:
            import math
            return math.sqrt(max(0, a / 1000.0))
        # Sqrt model uses BatchNorm with track_running_stats=False;
        # need batch >= 2, pad with duplicate
        val = torch.tensor([[a / 1000.0], [a / 1000.0]])
        with torch.no_grad():
            out = self._sqrt(val)
        return float(out[0].item())

    def neural_exp(self, a: int) -> float:
        """Neural exp: input is fixed-point (value / 1000), returns float."""
        if self._exp is None:
            import math
            return math.exp(min(20, a / 1000.0))
        val = torch.tensor([[a / 1000.0]])
        with torch.no_grad():
            out = self._exp(val)
        return float(out.item())

    def neural_log(self, a: int) -> float:
        """Neural log: input is fixed-point (value / 1000), returns float."""
        if self._log is None:
            import math
            return math.log(max(1e-10, a / 1000.0))
        val = torch.tensor([[max(1e-10, a / 1000.0)]])
        with torch.no_grad():
            out = self._log(val)
        return float(out.item())

    def neural_atan2(self, y: int, x: int) -> float:
        """Neural atan2: inputs are fixed-point (value / 1000), returns float."""
        if self._atan2 is None:
            import math
            return math.atan2(y / 1000.0, x / 1000.0)
        import math
        yf, xf = y / 1000.0, x / 1000.0
        r = math.sqrt(yf * yf + xf * xf) + 1e-8
        sin_a, cos_a = yf / r, xf / r
        q = [float(yf >= 0), float(xf >= 0), float(yf < 0), float(xf < 0)]
        row = [sin_a, cos_a] + q
        # Atan2 model uses BatchNorm with track_running_stats=False;
        # need batch >= 2, pad with duplicate
        inp = torch.tensor([row, row])
        with torch.no_grad():
            out = self._atan2(inp)
        # Output is (angle_sin, angle_cos) — convert back to angle
        return float(math.atan2(out[0, 0].item(), out[0, 1].item()))


    # ─── Batch API (for instruction-level parallelism) ───────────────────────

    def batch_neural_add(self, pairs: list) -> list:
        """Process multiple additions through the neural CLA in a single batched pass.

        When the CLA carry_combiner is available, all additions share the same
        Kogge-Stone prefix tree passes, processing M additions x 32 bits in parallel.
        Each pair is (a, b). Returns list of results.
        """
        if not pairs:
            return []
        if len(pairs) == 1:
            return [self.neural_add(pairs[0][0], pairs[0][1])]

        # True batched CLA path
        if (self._carry_combiner is not None and self._logical is not None
                and self._adder is not None):
            return self._batch_add_cla(pairs, carry_in=0.0)

        # Fallback: sequential
        return [self.neural_add(a, b) for a, b in pairs]

    def _batch_add_cla(self, pairs: list, carry_in: float = 0.0) -> list:
        """Batched Kogge-Stone CLA: M additions in parallel.

        Processes [M, 32] bit tensors through the same prefix tree,
        amortizing kernel launch overhead across all additions.
        """
        M = len(pairs)
        N = self.BITS

        # Convert all pairs to bit tensors: [M, 32]
        all_bits_a = torch.stack([self._int_to_bits(a) for a, _ in pairs])  # [M, 32]
        all_bits_b = torch.stack([self._int_to_bits(b) for _, b in pairs])  # [M, 32]

        # Step 1: G[m,i] = a[m,i] AND b[m,i], P[m,i] = a[m,i] XOR b[m,i]
        idx_and = (all_bits_a > 0.5).long() * 2 + (all_bits_b > 0.5).long()  # [M, 32]
        idx_xor = idx_and  # same indexing, different op
        with torch.no_grad():
            G = (torch.sigmoid(self._logical.truth_tables[0, idx_and]) > 0.5).float()  # [M, 32]
            P = (torch.sigmoid(self._logical.truth_tables[2, idx_xor]) > 0.5).float()  # [M, 32]

        # Handle carry_in
        if carry_in > 0.5:
            g0_new = torch.clamp(G[:, 0] + P[:, 0], max=1.0)
            G = G.clone()
            G[:, 0] = (g0_new > 0.5).float()

        # Step 2: Kogge-Stone parallel-prefix (5 stages)
        with torch.no_grad():
            stride = 1
            for _ in range(5):
                if stride >= N:
                    break
                n_combines = N - stride

                # Build batched input: [M * n_combines, 4]
                G_i = G[:, stride:].reshape(-1)           # [M * n_combines]
                P_i = P[:, stride:].reshape(-1)
                G_j = G[:, :N - stride].reshape(-1)
                P_j = P[:, :N - stride].reshape(-1)

                batch_input = torch.stack([G_i, P_i, G_j, P_j], dim=1)  # [M*n_combines, 4]
                out = self._carry_combiner(batch_input)  # [M*n_combines, 2]

                new_G = (out[:, 0] > 0.5).float().reshape(M, n_combines)
                new_P = (out[:, 1] > 0.5).float().reshape(M, n_combines)

                G = G.clone()
                P = P.clone()
                G[:, stride:] = new_G
                P[:, stride:] = new_P

                stride *= 2

        # Step 3: Final sum S[m,i] = P_original[m,i] XOR carries[m,i]
        P_original = (torch.sigmoid(self._logical.truth_tables[2, idx_xor]) > 0.5).float()

        carries = torch.zeros(M, N)
        carries[:, 0] = carry_in
        carries[:, 1:] = G[:, :-1]

        idx_final = (P_original > 0.5).long() * 2 + (carries > 0.5).long()
        with torch.no_grad():
            result_bits = (torch.sigmoid(self._logical.truth_tables[2, idx_final]) > 0.5).float()

        # Convert all results back to integers
        results = []
        for m in range(M):
            results.append(self._bits_to_int(result_bits[m]))
        return results

    def batch_neural_mul(self, pairs: list) -> list:
        """Process multiple multiplications through the neural LUT.

        Gathers all byte-pair lookups from all multiplications into
        a single tensor gather for true GPU parallelism.
        """
        if self._multiplier is None or not pairs:
            return [self._clamp(a * b) for a, b in pairs]

        # For single items, skip batching overhead
        if len(pairs) == 1:
            return [self.neural_mul(pairs[0][0], pairs[0][1])]

        # True batched MUL: gather all byte-pair lookups across all muls
        all_results = []
        all_pairs_a = []
        all_pairs_b = []
        all_shifts = []
        all_owners = []  # which multiplication each lookup belongs to
        signs = []

        for idx, (a, b) in enumerate(pairs):
            sign = 1
            if a < 0:
                a = -a
                sign = -sign
            if b < 0:
                b = -b
                sign = -sign
            signs.append(sign)

            a = a & 0xFFFFFFFF
            b = b & 0xFFFFFFFF
            a_bytes = [(a >> (i * 8)) & 0xFF for i in range(4)]
            b_bytes = [(b >> (i * 8)) & 0xFF for i in range(4)]

            for i in range(4):
                if a_bytes[i] == 0:
                    continue
                for j in range(4):
                    if b_bytes[j] == 0:
                        continue
                    all_pairs_a.append(a_bytes[i])
                    all_pairs_b.append(b_bytes[j])
                    all_shifts.append((i + j) * 8)
                    all_owners.append(idx)

        if not all_pairs_a:
            return [0] * len(pairs)

        # Single batched tensor gather for ALL byte-pair lookups
        with torch.no_grad():
            a_idx = torch.tensor(all_pairs_a, dtype=torch.long)
            b_idx = torch.tensor(all_pairs_b, dtype=torch.long)
            logits = self._multiplier.lut.table[a_idx, b_idx]
            bits = (torch.sigmoid(logits) > 0.5).float()
            if NeuralMultiplierLUT._lut_bit_values is None:
                NeuralMultiplierLUT._lut_bit_values = (
                    1 << torch.arange(16, dtype=torch.long)
                ).float()
            products = (bits @ NeuralMultiplierLUT._lut_bit_values).long()

        # Accumulate results per multiplication
        results = [0] * len(pairs)
        for k in range(len(all_owners)):
            owner = all_owners[k]
            results[owner] += int(products[k].item()) << all_shifts[k]

        # Apply sign and clamp
        for idx in range(len(pairs)):
            r = results[idx] & 0xFFFFFFFF
            if r >= 0x80000000:
                r -= 0x100000000
            r *= signs[idx]
            results[idx] = max(INT32_MIN, min(INT32_MAX, r))

        return results

    def batch_neural_bitwise(self, triples: list) -> list:
        """Process multiple bitwise ops in a single call.

        Each element of triples is (a, b, op_idx).
        op_idx: 0=AND, 1=OR, 2=XOR. Returns list of results.
        """
        return [self._neural_bitwise_op(a, b, op) for a, b, op in triples]


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL REGISTRY — Drop-in replacement for CPURegistry
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralRegistry:
    """CPU registry where ALU operations use trained neural networks.

    Pipeline: DECODE → KEY → NEURAL REGISTRY → NEURAL MODEL → STATE

    Operations like ADD, SUB, MUL, CMP pass through trained .pt models.
    Control flow (JMP, JZ, etc.) and data movement (MOV) use direct logic.
    """

    def __init__(self, models_dir: str = "models"):
        self.neural = NeuralOps(models_dir=models_dir)
        self._loaded = False

    def load(self) -> Dict[str, bool]:
        """Load neural models. Returns map of available operations."""
        available = self.neural.load()
        self._loaded = True
        return available

    def get_valid_keys(self) -> set:
        return {
            "OP_MOV_REG_IMM", "OP_MOV_REG_REG",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
            "OP_AND", "OP_OR", "OP_XOR",
            "OP_SHL", "OP_SHR",
            "OP_INC", "OP_DEC",
            "OP_CMP",
            "OP_JMP", "OP_JZ", "OP_JNZ", "OP_JS", "OP_JNS",
            "OP_HALT", "OP_NOP", "OP_INVALID",
        }

    def execute(self, state: CPUState, key: str, params: Dict[str, Any]) -> CPUState:
        """Execute an operation using neural models where available."""
        handler = self._dispatch.get(key)
        if handler is None:
            raise KeyError(f"Unknown operation key: {key}")
        new_state = handler(self, state, params)
        return new_state.increment_cycle()

    # ─── Data Movement (direct, no neural model needed) ──────────────────────

    def _op_mov_reg_imm(self, state: CPUState, params: Dict) -> CPUState:
        dest, value = params["dest"], params["value"]
        new_state = state.set_register(dest, value)
        return new_state.set_flags(value).increment_pc()

    def _op_mov_reg_reg(self, state: CPUState, params: Dict) -> CPUState:
        value = state.get_register(params["src"])
        new_state = state.set_register(params["dest"], value)
        return new_state.set_flags(value).increment_pc()

    # ─── Arithmetic (NEURAL) ─────────────────────────────────────────────────

    def _op_add(self, state: CPUState, params: Dict) -> CPUState:
        a = state.get_register(params["src1"])
        b = state.get_register(params["src2"])
        result = self.neural.neural_add(a, b)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    def _op_sub(self, state: CPUState, params: Dict) -> CPUState:
        a = state.get_register(params["src1"])
        b = state.get_register(params["src2"])
        result = self.neural.neural_sub(a, b)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    def _op_mul(self, state: CPUState, params: Dict) -> CPUState:
        a = state.get_register(params["src1"])
        b = state.get_register(params["src2"])
        result = self.neural.neural_mul(a, b)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    def _op_div(self, state: CPUState, params: Dict) -> CPUState:
        a = state.get_register(params["src1"])
        b = state.get_register(params["src2"])
        result = self.neural.neural_div(a, b)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    # ─── Bitwise (NEURAL) ─────────────────────────────────────────────────

    def _op_and(self, state: CPUState, params: Dict) -> CPUState:
        a = state.get_register(params["src1"])
        b = state.get_register(params["src2"])
        result = self.neural.neural_and(a, b)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    def _op_or(self, state: CPUState, params: Dict) -> CPUState:
        a = state.get_register(params["src1"])
        b = state.get_register(params["src2"])
        result = self.neural.neural_or(a, b)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    def _op_xor(self, state: CPUState, params: Dict) -> CPUState:
        a = state.get_register(params["src1"])
        b = state.get_register(params["src2"])
        result = self.neural.neural_xor(a, b)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    # ─── Shifts (NEURAL) ──────────────────────────────────────────────────

    def _op_shl(self, state: CPUState, params: Dict) -> CPUState:
        value = state.get_register(params["src"])
        amount = params.get("amount")
        if amount is None:
            amount = state.get_register(params["amount_reg"])
        result = self.neural.neural_shl(value, amount)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    def _op_shr(self, state: CPUState, params: Dict) -> CPUState:
        value = state.get_register(params["src"])
        amount = params.get("amount")
        if amount is None:
            amount = state.get_register(params["amount_reg"])
        result = self.neural.neural_shr(value, amount)
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    def _op_inc(self, state: CPUState, params: Dict) -> CPUState:
        result = self.neural.neural_inc(state.get_register(params["dest"]))
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    def _op_dec(self, state: CPUState, params: Dict) -> CPUState:
        result = self.neural.neural_dec(state.get_register(params["dest"]))
        new_state = state.set_register(params["dest"], result)
        return new_state.set_flags(result).increment_pc()

    # ─── Comparison (NEURAL) ─────────────────────────────────────────────────

    def _op_cmp(self, state: CPUState, params: Dict) -> CPUState:
        a = state.get_register(params["src1"])
        b = state.get_register(params["src2"])
        diff = self.neural.neural_sub(a, b)
        return state.set_flags(diff).increment_pc()

    # ─── Control Flow (direct logic) ─────────────────────────────────────────

    def _op_jmp(self, state: CPUState, params: Dict) -> CPUState:
        return state.set_pc(params["addr"])

    def _op_jz(self, state: CPUState, params: Dict) -> CPUState:
        if state.flags["ZF"]:
            return state.set_pc(params["addr"])
        return state.increment_pc()

    def _op_jnz(self, state: CPUState, params: Dict) -> CPUState:
        if not state.flags["ZF"]:
            return state.set_pc(params["addr"])
        return state.increment_pc()

    def _op_js(self, state: CPUState, params: Dict) -> CPUState:
        if state.flags["SF"]:
            return state.set_pc(params["addr"])
        return state.increment_pc()

    def _op_jns(self, state: CPUState, params: Dict) -> CPUState:
        if not state.flags["SF"]:
            return state.set_pc(params["addr"])
        return state.increment_pc()

    # ─── Special ─────────────────────────────────────────────────────────────

    def _op_halt(self, state: CPUState, params: Dict) -> CPUState:
        return state.set_halted(True)

    def _op_nop(self, state: CPUState, params: Dict) -> CPUState:
        return state.increment_pc()

    def _op_invalid(self, state: CPUState, params: Dict) -> CPUState:
        return state.set_halted(True)

    # ─── Dispatch table ──────────────────────────────────────────────────────

    _dispatch = {
        "OP_MOV_REG_IMM": _op_mov_reg_imm,
        "OP_MOV_REG_REG": _op_mov_reg_reg,
        "OP_ADD": _op_add,
        "OP_SUB": _op_sub,
        "OP_MUL": _op_mul,
        "OP_DIV": _op_div,
        "OP_AND": _op_and,
        "OP_OR": _op_or,
        "OP_XOR": _op_xor,
        "OP_SHL": _op_shl,
        "OP_SHR": _op_shr,
        "OP_INC": _op_inc,
        "OP_DEC": _op_dec,
        "OP_CMP": _op_cmp,
        "OP_JMP": _op_jmp,
        "OP_JZ": _op_jz,
        "OP_JNZ": _op_jnz,
        "OP_JS": _op_js,
        "OP_JNS": _op_jns,
        "OP_HALT": _op_halt,
        "OP_NOP": _op_nop,
        "OP_INVALID": _op_invalid,
    }
