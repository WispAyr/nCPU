"""CPURegistry: Verified CPU primitives for the model-based nCPU.

Each operation is a verified, frozen primitive that transforms state in a
predictable, auditable way. The registry is frozen after initialization
to prevent runtime modifications.

Registry Keys:
    OP_MOV_REG_IMM  - Load immediate value into register
    OP_MOV_REG_REG  - Copy register to register
    OP_ADD          - Add two registers
    OP_SUB          - Subtract two registers
    OP_MUL          - Multiply two registers
    OP_DIV          - Integer divide two registers
    OP_AND          - Bitwise AND two registers
    OP_OR           - Bitwise OR two registers
    OP_XOR          - Bitwise XOR two registers
    OP_SHL          - Shift left (by immediate or register)
    OP_SHR          - Shift right (by immediate or register)
    OP_INC          - Increment register by 1
    OP_DEC          - Decrement register by 1
    OP_CMP          - Compare two registers, set flags
    OP_JMP          - Unconditional jump
    OP_JZ           - Jump if zero flag set
    OP_JNZ          - Jump if zero flag not set
    OP_JS           - Jump if sign flag set
    OP_JNS          - Jump if sign flag not set
    OP_HALT         - Stop execution
    OP_NOP          - No operation
    OP_INVALID      - Error handling
"""

from typing import Dict, Callable, Any, Optional
from .state import CPUState, INT32_MIN, INT32_MAX


class CPURegistry:
    """Verified registry of CPU primitives.

    The registry is frozen after initialization to ensure
    no runtime modifications can occur.
    """

    def __init__(self):
        self._primitives: Dict[str, Callable[[CPUState, Dict[str, Any]], CPUState]] = {}
        self._frozen = False
        self._register_all_primitives()
        self.freeze()

    def _register_all_primitives(self) -> None:
        self.register("OP_MOV_REG_IMM", self._op_mov_reg_imm)
        self.register("OP_MOV_REG_REG", self._op_mov_reg_reg)
        self.register("OP_ADD", self._op_add)
        self.register("OP_SUB", self._op_sub)
        self.register("OP_MUL", self._op_mul)
        self.register("OP_DIV", self._op_div)
        self.register("OP_AND", self._op_and)
        self.register("OP_OR", self._op_or)
        self.register("OP_XOR", self._op_xor)
        self.register("OP_SHL", self._op_shl)
        self.register("OP_SHR", self._op_shr)
        self.register("OP_INC", self._op_inc)
        self.register("OP_DEC", self._op_dec)
        self.register("OP_CMP", self._op_cmp)
        self.register("OP_JMP", self._op_jmp)
        self.register("OP_JZ", self._op_jz)
        self.register("OP_JNZ", self._op_jnz)
        self.register("OP_JS", self._op_js)
        self.register("OP_JNS", self._op_jns)
        self.register("OP_HALT", self._op_halt)
        self.register("OP_NOP", self._op_nop)
        self.register("OP_INVALID", self._op_invalid)

    def register(self, key: str, handler: Callable) -> None:
        if self._frozen:
            raise RuntimeError("Cannot register primitives: registry is frozen")
        if key in self._primitives:
            raise ValueError(f"Primitive already registered: {key}")
        self._primitives[key] = handler

    def freeze(self) -> None:
        self._frozen = True

    def is_frozen(self) -> bool:
        return self._frozen

    def get_valid_keys(self) -> set:
        return set(self._primitives.keys())

    def execute(self, state: CPUState, key: str, params: Dict[str, Any]) -> CPUState:
        if key not in self._primitives:
            raise KeyError(f"Unknown operation key: {key}")
        new_state = self._primitives[key](state, params)
        return new_state.increment_cycle()

    # -- Data Movement --

    def _op_mov_reg_imm(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, value = params["dest"], params["value"]
        new_state = state.set_register(dest, value)
        new_state = new_state.set_flags(value)
        return new_state.increment_pc()

    def _op_mov_reg_reg(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src = params["dest"], params["src"]
        value = state.get_register(src)
        new_state = state.set_register(dest, value)
        new_state = new_state.set_flags(value)
        return new_state.increment_pc()

    # -- Arithmetic --

    def _op_add(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src1, src2 = params["dest"], params["src1"], params["src2"]
        result = self._clamp(state.get_register(src1) + state.get_register(src2))
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    def _op_sub(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src1, src2 = params["dest"], params["src1"], params["src2"]
        result = self._clamp(state.get_register(src1) - state.get_register(src2))
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    def _op_mul(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src1, src2 = params["dest"], params["src1"], params["src2"]
        result = self._clamp(state.get_register(src1) * state.get_register(src2))
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    def _op_div(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src1, src2 = params["dest"], params["src1"], params["src2"]
        a = state.get_register(src1)
        b = state.get_register(src2)
        if b == 0:
            result = 0
        else:
            # Python's // truncates toward negative infinity; int division
            # in hardware truncates toward zero, so use int(a/b).
            result = self._clamp(int(a / b))
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    # -- Bitwise --

    def _op_and(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src1, src2 = params["dest"], params["src1"], params["src2"]
        result = self._clamp(state.get_register(src1) & state.get_register(src2))
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    def _op_or(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src1, src2 = params["dest"], params["src1"], params["src2"]
        result = self._clamp(state.get_register(src1) | state.get_register(src2))
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    def _op_xor(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src1, src2 = params["dest"], params["src1"], params["src2"]
        result = self._clamp(state.get_register(src1) ^ state.get_register(src2))
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    # -- Shifts --

    def _op_shl(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src = params["dest"], params["src"]
        amount = params.get("amount")
        if amount is None:
            amount = state.get_register(params["amount_reg"])
        amount = max(0, min(31, amount))
        result = self._clamp(state.get_register(src) << amount)
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    def _op_shr(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest, src = params["dest"], params["src"]
        amount = params.get("amount")
        if amount is None:
            amount = state.get_register(params["amount_reg"])
        amount = max(0, min(31, amount))
        value = state.get_register(src)
        # Logical shift right (treat as unsigned for shifting)
        if value < 0:
            value = value + (1 << 32)
        result = value >> amount
        if result >= (1 << 31):
            result -= (1 << 32)
        result = self._clamp(result)
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    def _op_inc(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest = params["dest"]
        result = self._clamp(state.get_register(dest) + 1)
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    def _op_dec(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        dest = params["dest"]
        result = self._clamp(state.get_register(dest) - 1)
        new_state = state.set_register(dest, result)
        return new_state.set_flags(result).increment_pc()

    # -- Comparison --

    def _op_cmp(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        diff = state.get_register(params["src1"]) - state.get_register(params["src2"])
        return state.set_flags(diff).increment_pc()

    # -- Control Flow --

    def _op_jmp(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        return state.set_pc(params["addr"])

    def _op_jz(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        if state.flags["ZF"]:
            return state.set_pc(params["addr"])
        return state.increment_pc()

    def _op_jnz(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        if not state.flags["ZF"]:
            return state.set_pc(params["addr"])
        return state.increment_pc()

    def _op_js(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        if state.flags["SF"]:
            return state.set_pc(params["addr"])
        return state.increment_pc()

    def _op_jns(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        if not state.flags["SF"]:
            return state.set_pc(params["addr"])
        return state.increment_pc()

    # -- Special --

    def _op_halt(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        return state.set_halted(True)

    def _op_nop(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        return state.increment_pc()

    def _op_invalid(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        return state.set_halted(True)

    def _clamp(self, value: int) -> int:
        return max(INT32_MIN, min(INT32_MAX, value))


_registry: Optional[CPURegistry] = None


def get_registry() -> CPURegistry:
    global _registry
    if _registry is None:
        _registry = CPURegistry()
    return _registry
