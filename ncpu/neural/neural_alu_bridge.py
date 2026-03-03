"""Neural ALU Bridge: routes all ALU operations through trained neural networks.

The NeuralCPU (ncpu.neural) uses this bridge as its sole ALU execution path.
Every arithmetic, logical, shift, and comparison operation passes through a
trained .pt model — no Python arithmetic fallback.

Key details:
    - ncpu.model operates on 32-bit signed integers (Python ints)
    - ncpu.neural operates on 64-bit signed tensors (torch.int64)
    - The trained models are 32-bit; this bridge narrows 64→32 and widens back

Usage:
    from ncpu.neural import NeuralCPU
    cpu = NeuralCPU()  # Neural ALU is always on
"""

import torch
from typing import Tuple


class NeuralALUBridge:
    """Wraps ncpu.model.neural_ops.NeuralOps for use by the 64-bit NeuralCPU.

    Accepts torch.int64 tensors or Python ints, returns Python ints.
    Handles 64-bit → 32-bit narrowing (trained models are 32-bit).
    """

    INT32_MIN = -(2**31)
    INT32_MAX = (2**31) - 1

    def __init__(self, models_dir: str = "models"):
        from ncpu.model.neural_ops import NeuralOps
        self._ops = NeuralOps(models_dir=models_dir)
        self._loaded = False

    def load(self) -> dict:
        """Load neural models. Returns availability map."""
        available = self._ops.load()
        self._loaded = True
        return available

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _to_int32(self, value) -> int:
        """Convert a torch.int64 tensor or Python int to a 32-bit signed int."""
        if isinstance(value, torch.Tensor):
            value = int(value.item())
        # Narrow to 32-bit range (truncate upper bits)
        value = value & 0xFFFFFFFF
        if value >= 0x80000000:
            value -= 0x100000000
        return value

    # ─── Arithmetic ───────────────────────────────────────────────────────

    def add(self, a, b) -> int:
        return self._ops.neural_add(self._to_int32(a), self._to_int32(b))

    def sub(self, a, b) -> int:
        return self._ops.neural_sub(self._to_int32(a), self._to_int32(b))

    def mul(self, a, b) -> int:
        return self._ops.neural_mul(self._to_int32(a), self._to_int32(b))

    def div(self, a, b) -> int:
        """Neural integer division using restoring division algorithm."""
        return self._ops.neural_div(self._to_int32(a), self._to_int32(b))

    # ─── Bitwise ──────────────────────────────────────────────────────────

    def and_(self, a, b) -> int:
        return self._ops.neural_and(self._to_int32(a), self._to_int32(b))

    def or_(self, a, b) -> int:
        return self._ops.neural_or(self._to_int32(a), self._to_int32(b))

    def xor_(self, a, b) -> int:
        return self._ops.neural_xor(self._to_int32(a), self._to_int32(b))

    # ─── Shifts ───────────────────────────────────────────────────────────

    def shl(self, value, amount) -> int:
        return self._ops.neural_shl(self._to_int32(value), self._to_int32(amount))

    def shr(self, value, amount) -> int:
        return self._ops.neural_shr(self._to_int32(value), self._to_int32(amount))

    # ─── Comparison ───────────────────────────────────────────────────────

    def cmp(self, a, b) -> Tuple[float, float, float]:
        """Compare a - b, return (N_flag, Z_flag, C_flag) as floats for tensor assignment.

        N = (diff < 0), Z = (diff == 0), C = (a >= b as unsigned)
        """
        a32 = self._to_int32(a)
        b32 = self._to_int32(b)
        diff = self._ops.neural_sub(a32, b32)

        n_flag = float(diff < 0)
        z_flag = float(diff == 0)

        # Carry flag: unsigned comparison (a >= b)
        ua = a32 & 0xFFFFFFFF
        ub = b32 & 0xFFFFFFFF
        c_flag = float(ua >= ub)

        return n_flag, z_flag, c_flag

    # ─── Math (experimental — trained models, approximate) ───────────────

    def sin(self, a) -> float:
        """Neural sin: input is fixed-point (value / 1000 = radians)."""
        return self._ops.neural_sin(self._to_int32(a))

    def cos(self, a) -> float:
        """Neural cos: input is fixed-point (value / 1000 = radians)."""
        return self._ops.neural_cos(self._to_int32(a))

    def sqrt(self, a) -> float:
        """Neural sqrt: input is fixed-point (value / 1000)."""
        return self._ops.neural_sqrt(self._to_int32(a))

    def exp_(self, a) -> float:
        """Neural exp: input is fixed-point (value / 1000)."""
        return self._ops.neural_exp(self._to_int32(a))

    def log_(self, a) -> float:
        """Neural log: input is fixed-point (value / 1000)."""
        return self._ops.neural_log(self._to_int32(a))

    def atan2(self, y, x) -> float:
        """Neural atan2: inputs are fixed-point (value / 1000)."""
        return self._ops.neural_atan2(self._to_int32(y), self._to_int32(x))

    # ─── Batch API (instruction-level parallelism) ────────────────────────

    def batch_add(self, pairs: list) -> list:
        """Process multiple additions in a single call.

        pairs: list of (a, b) where a, b are torch.int64 tensors or Python ints.
        Returns list of int results.
        """
        converted = [(self._to_int32(a), self._to_int32(b)) for a, b in pairs]
        return self._ops.batch_neural_add(converted)

    def batch_mul(self, pairs: list) -> list:
        """Process multiple multiplications in a single batched tensor gather.

        pairs: list of (a, b) where a, b are torch.int64 tensors or Python ints.
        Returns list of int results.
        """
        converted = [(self._to_int32(a), self._to_int32(b)) for a, b in pairs]
        return self._ops.batch_neural_mul(converted)

    def batch_bitwise(self, triples: list) -> list:
        """Process multiple bitwise ops in a single call.

        triples: list of (a, b, op_idx) where op_idx: 0=AND, 1=OR, 2=XOR.
        Returns list of int results.
        """
        converted = [(self._to_int32(a), self._to_int32(b), op) for a, b, op in triples]
        return self._ops.batch_neural_bitwise(converted)
