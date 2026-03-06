"""Decoder: Instruction decoder for the model-based nCPU.

Three decode modes:
    - Mock: Fast rule-based regex parsing (no GPU, no model download)
    - Neural: Trained CNN classifier (~50K params) for opcode identification
    - Real: Fine-tuned LLM for semantic understanding (legacy, requires torch)

All modes produce identical output format: (operation_key, params).
"""

import re
import json
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DecodeResult:
    """Result of instruction decode."""
    key: str
    params: Dict
    valid: bool
    error: Optional[str] = None
    raw_instruction: str = ""


class Decoder:
    """Instruction decoder with mock (regex), neural (CNN), and real (LLM) modes.

    Mock mode uses deterministic regex patterns — instant, no dependencies.
    Neural mode uses a trained 50K-param CNN for opcode classification,
    then extracts operands deterministically (like a real CPU decoder).
    Real mode uses a fine-tuned LLM (legacy, requires external model download).
    """

    VALID_KEYS: Set[str] = {
        "OP_MOV_REG_IMM", "OP_MOV_REG_REG",
        "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
        "OP_AND", "OP_OR", "OP_XOR",
        "OP_SHL", "OP_SHR",
        "OP_INC", "OP_DEC",
        "OP_CMP",
        "OP_JMP", "OP_JZ", "OP_JNZ", "OP_JS", "OP_JNS",
        "OP_HALT", "OP_NOP", "OP_INVALID",
    }

    REGISTERS: Set[str] = {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}

    def __init__(self, mock_mode: bool = True, model_path: Optional[str] = None,
                 neural_mode: bool = False):
        self.mock_mode = mock_mode
        self.neural_mode = neural_mode
        self.model_path = model_path
        self.labels: Dict[str, int] = {}
        self._model = None
        self._tokenizer = None
        self._neural_model = None

        if not mock_mode and not neural_mode and model_path is None:
            raise ValueError("model_path required when mock_mode=False and neural_mode=False")

    def set_labels(self, labels: Dict[str, int]) -> None:
        self.labels = labels

    def load(self) -> None:
        """Load the trained model (only needed for real mode)."""
        if self.mock_mode:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            adapter_cfg = Path(self.model_path) / "adapter_config.json"
            adapter_weights = Path(self.model_path) / "adapter_model.safetensors"
            is_lora_adapter = adapter_cfg.exists() and adapter_weights.exists()

            if is_lora_adapter:
                try:
                    from peft import PeftModel
                except ImportError as e:
                    raise ImportError(
                        "LoRA adapter requires 'peft'. Install with: pip install peft"
                    ) from e

                with open(adapter_cfg) as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get(
                    "base_model_name_or_path", "Qwen/Qwen2.5-Coder-1.5B"
                )

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, dtype=torch.float32, device_map=None
                )
                self._model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, dtype=torch.float32, device_map=None
                )

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = self._model.to(device)
            self._model.eval()

        except ImportError as e:
            raise ImportError(
                "Real mode requires torch, transformers, and peft. "
                f"Missing: {e}"
            ) from e

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

    def load_neural(self, model_path: Optional[str] = None) -> None:
        """Load the trained neural decoder model (~50K params CNN)."""
        import torch
        from ncpu.model.architectures import InstructionDecoderNet

        raw_path = model_path or self.model_path or "models/decode"
        p = Path(raw_path)
        path = str(p / "decode.pt") if p.is_dir() else str(p)
        model = InstructionDecoderNet()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device).eval()
        self._neural_model = model
        self.neural_mode = True

    def decode(self, instruction: str) -> DecodeResult:
        """Decode an instruction to operation key and parameters."""
        instruction = instruction.strip()

        if not instruction:
            return DecodeResult(
                key="OP_INVALID", params={"raw": ""}, valid=False,
                error="Empty instruction", raw_instruction=""
            )

        if self.mock_mode:
            return self._mock_decode(instruction)
        elif self.neural_mode:
            return self._neural_decode(instruction)
        else:
            return self._llm_decode(instruction)

    # -- Neural Mode (trained CNN classifier) --

    def _neural_decode(self, instruction: str) -> DecodeResult:
        """Decode using the trained neural opcode classifier.

        1. CNN classifies instruction text → opcode
        2. Deterministic extractor pulls operands based on opcode format
        (Exactly like a real CPU: opcode bits → format → operand fields)
        """
        import torch
        from ncpu.model.architectures import InstructionDecoderNet

        if self._neural_model is None:
            self.load_neural()

        x = InstructionDecoderNet.encode_instruction(instruction).unsqueeze(0)
        device = next(self._neural_model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self._neural_model(x)
            pred_idx = logits.argmax(dim=1).item()

        opcode = InstructionDecoderNet.OPCODES[pred_idx]

        # Extract operands deterministically based on opcode format
        params = self._extract_operands(instruction, opcode)
        if params is None:
            return DecodeResult(
                "OP_INVALID", {"raw": instruction}, False,
                error=f"Operand extraction failed for {opcode}",
                raw_instruction=instruction
            )

        return DecodeResult(opcode, params, True, raw_instruction=instruction)

    def _extract_operands(self, instruction: str, opcode: str) -> Optional[Dict]:
        """Extract operands deterministically based on opcode format.

        Once the neural model identifies the opcode, operand positions
        are determined by the instruction format — just like a real CPU
        reads register/immediate fields from fixed bit positions.
        """
        instr = instruction.upper().strip()
        instr = re.sub(r'\s+', ' ', instr)
        instr = re.sub(r'\s*,\s*', ',', instr)

        # Remove the mnemonic to get the operand string
        parts = instr.split(None, 1)
        operands_str = parts[1] if len(parts) > 1 else ""
        operands = [op.strip() for op in operands_str.replace(',', ' ').split()]

        try:
            if opcode in ("OP_HALT", "OP_NOP"):
                return {}

            if opcode == "OP_INVALID":
                return {"raw": instruction}

            if opcode == "OP_MOV_REG_IMM":
                if len(operands) >= 2 and operands[0] in self.REGISTERS:
                    return {"dest": operands[0], "value": self._parse_immediate(operands[1])}

            if opcode == "OP_MOV_REG_REG":
                if len(operands) >= 2 and operands[0] in self.REGISTERS and operands[1] in self.REGISTERS:
                    return {"dest": operands[0], "src": operands[1]}

            if opcode in ("OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
                          "OP_AND", "OP_OR", "OP_XOR"):
                if len(operands) >= 3:
                    return {"dest": operands[0], "src1": operands[1], "src2": operands[2]}

            if opcode in ("OP_SHL", "OP_SHR"):
                if len(operands) >= 3:
                    params = {"dest": operands[0], "src": operands[1]}
                    if operands[2] in self.REGISTERS:
                        params["amount_reg"] = operands[2]
                    else:
                        params["amount"] = self._parse_immediate(operands[2])
                    return params

            if opcode in ("OP_INC", "OP_DEC"):
                if len(operands) >= 1 and operands[0] in self.REGISTERS:
                    return {"dest": operands[0]}

            if opcode == "OP_CMP":
                if len(operands) >= 2:
                    return {"src1": operands[0], "src2": operands[1]}

            if opcode in ("OP_JMP", "OP_JZ", "OP_JNZ", "OP_JS", "OP_JNS"):
                if operands:
                    addr = self._resolve_address(operands[0])
                    if addr is not None:
                        return {"addr": addr}

        except (ValueError, IndexError):
            pass

        return None

    # -- Mock Mode (rule-based regex) --

    def _mock_decode(self, instruction: str) -> DecodeResult:
        instr = instruction.upper().strip()
        instr = re.sub(r'\s+', ' ', instr)
        instr = re.sub(r'\s*,\s*', ',', instr)

        try:
            if instr == "HALT":
                return DecodeResult("OP_HALT", {}, True, raw_instruction=instruction)

            if instr == "NOP":
                return DecodeResult("OP_NOP", {}, True, raw_instruction=instruction)

            # MOV Rd, imm or MOV Rd, Rs
            mov_match = re.match(r'^MOV\s+(R[0-7])[,\s]+(.+)$', instr)
            if mov_match:
                dest = mov_match.group(1)
                src = mov_match.group(2).strip()
                if src in self.REGISTERS:
                    return DecodeResult(
                        "OP_MOV_REG_REG", {"dest": dest, "src": src},
                        True, raw_instruction=instruction
                    )
                else:
                    try:
                        value = self._parse_immediate(src)
                        return DecodeResult(
                            "OP_MOV_REG_IMM", {"dest": dest, "value": value},
                            True, raw_instruction=instruction
                        )
                    except ValueError:
                        return DecodeResult(
                            "OP_INVALID", {"raw": instruction}, False,
                            error=f"Invalid MOV source: {src}",
                            raw_instruction=instruction
                        )

            # ADD Rd, Rs1, Rs2
            add_match = re.match(r'^ADD\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if add_match:
                return DecodeResult(
                    "OP_ADD",
                    {"dest": add_match.group(1), "src1": add_match.group(2), "src2": add_match.group(3)},
                    True, raw_instruction=instruction
                )

            # SUB Rd, Rs1, Rs2
            sub_match = re.match(r'^SUB\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if sub_match:
                return DecodeResult(
                    "OP_SUB",
                    {"dest": sub_match.group(1), "src1": sub_match.group(2), "src2": sub_match.group(3)},
                    True, raw_instruction=instruction
                )

            # MUL Rd, Rs1, Rs2
            mul_match = re.match(r'^MUL\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if mul_match:
                return DecodeResult(
                    "OP_MUL",
                    {"dest": mul_match.group(1), "src1": mul_match.group(2), "src2": mul_match.group(3)},
                    True, raw_instruction=instruction
                )

            # DIV Rd, Rs1, Rs2
            div_match = re.match(r'^DIV\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if div_match:
                return DecodeResult(
                    "OP_DIV",
                    {"dest": div_match.group(1), "src1": div_match.group(2), "src2": div_match.group(3)},
                    True, raw_instruction=instruction
                )

            # AND Rd, Rs1, Rs2
            and_match = re.match(r'^AND\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if and_match:
                return DecodeResult(
                    "OP_AND",
                    {"dest": and_match.group(1), "src1": and_match.group(2), "src2": and_match.group(3)},
                    True, raw_instruction=instruction
                )

            # OR Rd, Rs1, Rs2
            or_match = re.match(r'^OR\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if or_match:
                return DecodeResult(
                    "OP_OR",
                    {"dest": or_match.group(1), "src1": or_match.group(2), "src2": or_match.group(3)},
                    True, raw_instruction=instruction
                )

            # XOR Rd, Rs1, Rs2
            xor_match = re.match(r'^XOR\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if xor_match:
                return DecodeResult(
                    "OP_XOR",
                    {"dest": xor_match.group(1), "src1": xor_match.group(2), "src2": xor_match.group(3)},
                    True, raw_instruction=instruction
                )

            # SHL Rd, Rs, imm_or_reg
            shl_match = re.match(r'^SHL\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(.+)$', instr)
            if shl_match:
                dest = shl_match.group(1)
                src = shl_match.group(2)
                amount = shl_match.group(3).strip()
                if amount in self.REGISTERS:
                    return DecodeResult(
                        "OP_SHL",
                        {"dest": dest, "src": src, "amount_reg": amount},
                        True, raw_instruction=instruction
                    )
                else:
                    try:
                        imm = self._parse_immediate(amount)
                        return DecodeResult(
                            "OP_SHL",
                            {"dest": dest, "src": src, "amount": imm},
                            True, raw_instruction=instruction
                        )
                    except ValueError:
                        return DecodeResult(
                            "OP_INVALID", {"raw": instruction}, False,
                            error=f"Invalid SHL amount: {amount}",
                            raw_instruction=instruction
                        )

            # SHR Rd, Rs, imm_or_reg
            shr_match = re.match(r'^SHR\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(.+)$', instr)
            if shr_match:
                dest = shr_match.group(1)
                src = shr_match.group(2)
                amount = shr_match.group(3).strip()
                if amount in self.REGISTERS:
                    return DecodeResult(
                        "OP_SHR",
                        {"dest": dest, "src": src, "amount_reg": amount},
                        True, raw_instruction=instruction
                    )
                else:
                    try:
                        imm = self._parse_immediate(amount)
                        return DecodeResult(
                            "OP_SHR",
                            {"dest": dest, "src": src, "amount": imm},
                            True, raw_instruction=instruction
                        )
                    except ValueError:
                        return DecodeResult(
                            "OP_INVALID", {"raw": instruction}, False,
                            error=f"Invalid SHR amount: {amount}",
                            raw_instruction=instruction
                        )

            # INC Rd
            inc_match = re.match(r'^INC\s+(R[0-7])$', instr)
            if inc_match:
                return DecodeResult(
                    "OP_INC", {"dest": inc_match.group(1)},
                    True, raw_instruction=instruction
                )

            # DEC Rd
            dec_match = re.match(r'^DEC\s+(R[0-7])$', instr)
            if dec_match:
                return DecodeResult(
                    "OP_DEC", {"dest": dec_match.group(1)},
                    True, raw_instruction=instruction
                )

            # CMP Rs1, Rs2
            cmp_match = re.match(r'^CMP\s+(R[0-7])[,\s]+(R[0-7])$', instr)
            if cmp_match:
                return DecodeResult(
                    "OP_CMP", {"src1": cmp_match.group(1), "src2": cmp_match.group(2)},
                    True, raw_instruction=instruction
                )

            # Jump instructions (JMP, JZ, JNZ, JS, JNS)
            for opcode, op_key in [("JMP", "OP_JMP"), ("JZ", "OP_JZ"), ("JNZ", "OP_JNZ"),
                                   ("JS", "OP_JS"), ("JNS", "OP_JNS")]:
                jmp_match = re.match(rf'^{opcode}\s+(.+)$', instr)
                if jmp_match:
                    target = jmp_match.group(1).strip()
                    addr = self._resolve_address(target)
                    if addr is not None:
                        return DecodeResult(op_key, {"addr": addr}, True, raw_instruction=instruction)
                    else:
                        return DecodeResult(
                            "OP_INVALID", {"raw": instruction}, False,
                            error=f"Unknown label: {target}", raw_instruction=instruction
                        )

            return DecodeResult(
                "OP_INVALID", {"raw": instruction}, False,
                error=f"Unknown instruction format: {instruction}",
                raw_instruction=instruction
            )

        except Exception as e:
            return DecodeResult(
                "OP_INVALID", {"raw": instruction}, False,
                error=str(e), raw_instruction=instruction
            )

    def _parse_immediate(self, value: str) -> int:
        value = value.strip().upper()
        if value.startswith("0X"):
            return int(value, 16)
        if value.startswith("0B"):
            return int(value, 2)
        return int(value)

    def _resolve_address(self, target: str) -> Optional[int]:
        try:
            return int(target)
        except ValueError:
            pass
        target_upper = target.upper()
        for label, addr in self.labels.items():
            if label.upper() == target_upper:
                return addr
        return None

    # -- Real Mode (LLM-based) --

    def _llm_decode(self, instruction: str) -> DecodeResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        prompt = f"### Context:\n{instruction}\n\n### Key:\n"
        inputs = self._tokenizer(
            prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=128
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Key:" in generated:
            key_text = generated.split("### Key:")[-1].strip()
        else:
            key_text = generated.strip()

        try:
            start_idx = key_text.find('{')
            if start_idx != -1:
                depth = 0
                end_idx = start_idx
                for i, char in enumerate(key_text[start_idx:], start_idx):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break

                json_str = key_text[start_idx:end_idx]
                result = json.loads(json_str)
                key = result.get("key", "OP_INVALID")
                params = result.get("params", {k: v for k, v in result.items() if k != "key"})

                if key not in self.VALID_KEYS:
                    return DecodeResult(
                        "OP_INVALID", {"raw": instruction}, False,
                        error=f"Invalid key from LLM: {key}",
                        raw_instruction=instruction
                    )

                if key in ("OP_JMP", "OP_JZ", "OP_JNZ", "OP_JS", "OP_JNS") and "addr" in params:
                    addr = params["addr"]
                    if isinstance(addr, str):
                        resolved = self._resolve_address(addr)
                        if resolved is not None:
                            params["addr"] = resolved
                        else:
                            return DecodeResult(
                                "OP_INVALID", {"raw": instruction}, False,
                                error=f"Unknown label: {addr}",
                                raw_instruction=instruction
                            )

                return DecodeResult(key, params, True, raw_instruction=instruction)
            else:
                return DecodeResult(
                    "OP_INVALID", {"raw": instruction}, False,
                    error=f"No JSON found in LLM output: {key_text}",
                    raw_instruction=instruction
                )

        except json.JSONDecodeError as e:
            return DecodeResult(
                "OP_INVALID", {"raw": instruction}, False,
                error=f"JSON parse error: {e}",
                raw_instruction=instruction
            )


def parse_program(source: str) -> Tuple[list, Dict[str, int]]:
    """Parse assembly source into instructions and labels.

    Handles labels (lines ending with :), comments (; or #), and blank lines.
    """
    instructions = []
    labels = {}

    for line in source.split("\n"):
        line = re.sub(r'[;#].*$', '', line).strip()
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            labels[label] = len(instructions)
        else:
            instructions.append(line)

    return instructions, labels
