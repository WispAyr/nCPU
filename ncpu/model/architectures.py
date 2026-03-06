"""Reconstructed nn.Module architectures for all trained nCPU .pt models.

Each class has the exact layer structure required for strict state_dict loading.
These were reverse-engineered from the trained checkpoint state dict keys and tensor
shapes so that ``model.load_state_dict(state, strict=True)`` succeeds with zero
missing or unexpected keys.

Model categories:
    Register  -- NeuralRegisterFile
    Memory    -- NeuralStack, NeuralPointer, NeuralFunctionCall
    Decoder   -- NeuralARM64Decoder
    Math      -- NeuralAtan2, NeuralSinCos, NeuralSqrt, NeuralExp, NeuralLog, DoomTrigLUT
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    # ------------------------------------------------------------------
    # Shared building blocks
    # ------------------------------------------------------------------

    class _FullAdder(nn.Module):
        """Single-bit neural full adder: (a, b, carry_in) -> (sum, carry_out)."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class _AddrArith(nn.Module):
        """Address arithmetic wrapper around a full adder."""

        def __init__(self) -> None:
            super().__init__()
            self.full_adder = _FullAdder()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.full_adder(x)

    class _MemAddr(nn.Module):
        """Neural memory address encoder/decoder with learned temperature."""

        def __init__(self) -> None:
            super().__init__()
            self.address_encoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.slot_selector = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
            )
            self.temperature = nn.Parameter(torch.tensor(1.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.address_encoder(x)
            return self.slot_selector(encoded)

    # ------------------------------------------------------------------
    # Register models
    # ------------------------------------------------------------------

    class NeuralRegisterFile(nn.Module):
        """Neural register file with SP/XZR detection and W-register handling.

        Trained to emulate ARM64 register semantics including:
        - 31-register index encoding (X0-X30 + SP/XZR)
        - SP vs XZR context switching
        - W-register (32-bit) vs X-register (64-bit) handling
        - NZCV flag selection via learned attention
        """

        def __init__(self) -> None:
            super().__init__()
            self.base = _RegisterBase()
            self.sp_switch = _SPSwitch()
            self.xzr_detector = _XZRDetector()
            self.w_handler = _WHandler()
            self.flag_selector = _FlagSelector()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base.index_encoder(x)

    class _RegisterBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.index_encoder = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
            self.temperature = nn.Parameter(torch.tensor(1.0))
            self.xzr_read_mask = nn.Parameter(torch.zeros(32))
            self.xzr_write_mask = nn.Parameter(torch.zeros(32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.index_encoder(x)

    class _SPSwitch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.is_31_detector = nn.Sequential(
                nn.Linear(5, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            self.switch = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.temp = nn.Parameter(torch.tensor(1.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.switch(x)

    class _XZRDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detector = nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            self.temp = nn.Parameter(torch.tensor(1.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.detector(x)

    class _WHandler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extender = nn.Sequential(
                nn.Linear(64, 96),
                nn.ReLU(),
                nn.Linear(96, 96),
                nn.ReLU(),
                nn.Linear(96, 64),
            )
            self.extractor = nn.Sequential(
                nn.Linear(128, 96),
                nn.ReLU(),
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
            self.extend_temp = nn.Parameter(torch.tensor(1.0))
            self.extract_temp = nn.Parameter(torch.tensor(1.0))
            self.pos_enc_32 = nn.Parameter(torch.zeros(1, 32))
            self.pos_enc_64 = nn.Parameter(torch.zeros(1, 64))
            self.upper_zero_mask = nn.Parameter(torch.zeros(32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.extender(x)

    class _FlagSelector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.key = nn.Linear(4, 16)
            self.query = nn.Linear(4, 16)
            self.temp = nn.Parameter(torch.tensor(1.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.key(x)

    # ------------------------------------------------------------------
    # Memory models
    # ------------------------------------------------------------------

    class NeuralStack(nn.Module):
        """Neural stack with address arithmetic, memory addressing, and push/pop ops.

        Models a CPU stack pointer with neural full-adder address arithmetic,
        learned memory slot selection, and a push/pop operation network.
        """

        def __init__(self) -> None:
            super().__init__()
            self.addr_arith = _AddrArith()
            self.mem_addr = _MemAddr()
            self.op_net = nn.Sequential(
                nn.Linear(65, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.op_net(x)

    class NeuralPointer(nn.Module):
        """Neural pointer dereference with address arithmetic and memory addressing.

        Same address subsystem as NeuralStack but without the stack op network.
        Used for LDR/STR pointer-based memory access.
        """

        def __init__(self) -> None:
            super().__init__()
            self.addr_arith = _AddrArith()
            self.mem_addr = _MemAddr()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mem_addr(x)

    class NeuralFunctionCall(nn.Module):
        """Neural function call/return handling with address arithmetic.

        Models BL/RET semantics: saves return address, selects branch target,
        and processes return values, all through learned networks.
        """

        def __init__(self) -> None:
            super().__init__()
            self.addr_arith = _AddrArith()
            self.return_addr_net = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )
            self.target_selector = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.ret_processor = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.target_selector(x)

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    class NeuralARM64Decoder(nn.Module):
        """Transformer-based ARM64 instruction decoder.

        Encodes a 32-bit instruction as binary embeddings with positional encoding,
        applies self-attention and cross-attention field extraction, then decodes
        instruction category, operation, registers, immediates, and flags through
        specialized head networks.

        Architecture:
            encoder   -- bit + position embedding -> linear combine to 256-d
            field_extractor -- self-attention + cross-attention with 6 learned queries
            decoder_head -- 7 parallel heads for each decoded field
            refine    -- final MLP that merges all head outputs
        """

        def __init__(self) -> None:
            super().__init__()
            self.encoder = _InstructionEncoder()
            self.field_extractor = _FieldExtractor()
            self.decoder_head = _DecoderHead()
            self.refine = nn.Sequential(
                nn.Linear(1536, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.encoder(x)
            fields = self.field_extractor(encoded)
            return fields

    class _InstructionEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bit_embed = nn.Embedding(2, 64)
            self.pos_embed = nn.Embedding(32, 64)
            self.combine = nn.Linear(128, 256)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bit_emb = self.bit_embed(x)
            positions = torch.arange(x.size(-1), device=x.device)
            pos_emb = self.pos_embed(positions)
            combined = torch.cat([bit_emb, pos_emb.unsqueeze(0).expand_as(bit_emb)], dim=-1)
            return self.combine(combined)

    class _FieldExtractor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.field_queries = nn.Parameter(torch.randn(6, 256))
            self.self_attn = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
            self.field_attn = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(256)
            self.norm2 = nn.LayerNorm(256)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attended, _ = self.self_attn(x, x, x)
            normed = self.norm1(attended + x)
            queries = self.field_queries.unsqueeze(0).expand(x.size(0), -1, -1)
            fields, _ = self.field_attn(queries, normed, normed)
            return self.norm2(fields)

    class _DecoderHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.category_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 10),
            )
            self.operation_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
            )
            self.rd_head = nn.Linear(256, 32)
            self.rn_head = nn.Linear(256, 32)
            self.rm_head = nn.Linear(256, 32)
            self.imm_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 26),
            )
            self.flags_head = nn.Linear(256, 3)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "category": self.category_head(x),
                "operation": self.operation_head(x),
                "rd": self.rd_head(x),
                "rn": self.rn_head(x),
                "rm": self.rm_head(x),
                "imm": self.imm_head(x),
                "flags": self.flags_head(x),
            }

    # ------------------------------------------------------------------
    # Math models
    # ------------------------------------------------------------------

    class NeuralAtan2(nn.Module):
        """Neural atan2 approximation with BatchNorm-stabilized deep network.

        6 hidden layers with BatchNorm, trained on (sin, cos, quadrant) -> (angle_sin, angle_cos)
        representation for branch-free angle computation.
        """

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(6, 512),
                nn.BatchNorm1d(512, track_running_stats=False),
                nn.ReLU(),
            )
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512, track_running_stats=False),
                    nn.ReLU(),
                )
                for _ in range(6)
            ])
            self.output = nn.Linear(512, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.encoder(x)
            for layer in self.layers:
                h = layer(h) + h  # residual
            return self.output(h)

    class _SinCosBlock(nn.Module):
        """Custom block with a ``.linear`` attribute, used by NeuralSinCos."""

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.linear(x))

    class NeuralSinCos(nn.Module):
        """Neural sin/cos approximation using sine-activated layers.

        Five _SinCosBlock layers (each wrapping a Linear with sine activation)
        followed by a final Linear output. Trained to map angle -> (sin, cos).

        Note: checkpoint is saved as ``{model: state_dict, max_err, epoch}``.
        """

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.ModuleList([
                _SinCosBlock(1, 512),
                _SinCosBlock(512, 512),
                _SinCosBlock(512, 512),
                _SinCosBlock(512, 512),
                _SinCosBlock(512, 512),
                nn.Linear(512, 2),
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for layer in self.net:
                h = layer(h)
            return h

    class NeuralSqrt(nn.Module):
        """Two-stage neural square root: initial estimate then Newton-style refinement.

        Stage 1 (initial): Linear(1,256) -> BN -> ReLU -> Linear(256,256) -> BN -> ReLU -> Linear(256,1)
        Stage 2 (refine):  Linear(2,256) -> BN -> ReLU -> Linear(256,256) -> BN -> ReLU -> Linear(256,1)

        The refine stage takes (x, initial_estimate) as input for iterative improvement.

        Note: checkpoint is saved as ``{model: state_dict, rel_err, abs_err, epoch}``.
        """

        def __init__(self) -> None:
            super().__init__()
            self.initial = nn.Sequential(
                nn.Linear(1, 256),
                nn.BatchNorm1d(256, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            self.refine = nn.Sequential(
                nn.Linear(2, 256),
                nn.BatchNorm1d(256, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y0 = self.initial(x)
            return self.refine(torch.cat([x, y0], dim=-1))

    class NeuralExp(nn.Module):
        """Neural exponential function approximation.

        Four-layer MLP: Linear(1,256) -> ReLU -> Linear(256,256) -> ReLU ->
        Linear(256,256) -> ReLU -> Linear(256,1).

        Note: checkpoint is saved as ``{model: state_dict, error}``.
        """

        def __init__(self) -> None:
            super().__init__()
            self.core = nn.Sequential(
                nn.Linear(1, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.core(x)

    class NeuralLog(nn.Module):
        """Neural logarithm function approximation.

        Same architecture as NeuralExp: four-layer MLP with ReLU activations.

        Note: checkpoint is saved as ``{model: state_dict, error}``.
        """

        def __init__(self) -> None:
            super().__init__()
            self.core = nn.Sequential(
                nn.Linear(1, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.core(x)

    class DoomTrigLUT(nn.Module):
        """DOOM-compatible trigonometry lookup table (not a neural network).

        Stores 8192-entry sine/cosine tables in 16-bit fixed-point format
        (scale 65536), matching the original DOOM FINESINE table layout.

        Attributes:
            sine_table:   Tensor of shape [8192] -- fixed-point sine values
            cosine_table: Tensor of shape [8192] -- fixed-point cosine values
            n_angles:     int -- number of angle entries (8192)
            format_str:   str -- format description

        Note: this is purely a data container. The checkpoint is a plain dict,
        not a state_dict. Use ``load_from_dict()`` instead of ``load_state_dict()``.
        """

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("sine_table", torch.zeros(8192))
            self.register_buffer("cosine_table", torch.zeros(8192))
            self.n_angles: int = 8192
            self.format_str: str = ""

        def load_from_dict(self, data: dict) -> None:
            """Populate from the raw checkpoint dict."""
            self.sine_table = data["sine_table"].clone()
            self.cosine_table = data["cosine_table"].clone()
            self.n_angles = int(data["n_angles"])
            self.format_str = str(data["format"])

        def forward(self, angle_index: torch.Tensor) -> torch.Tensor:
            """Look up sine values by angle index."""
            idx = angle_index.long() % self.n_angles
            return self.sine_table[idx]

    # ------------------------------------------------------------------
    # Instruction Decoder — character-level CNN for opcode classification
    # ------------------------------------------------------------------

    class InstructionDecoderNet(nn.Module):
        """Neural instruction decoder — classifies assembly text to opcode.

        Character-level CNN encoder → opcode classification.
        After classification, operands are extracted deterministically
        based on the instruction format (like a real CPU decoder).

        Architecture:
            Input:  [B, max_len] character indices (ASCII 0-127)
            Embed:  → [B, embed_dim, max_len]
            Conv:   3 × Conv1d layers (widening receptive field)
            Pool:   Global max pool → [B, hidden_dim]
            Head:   Linear → [B, num_opcodes]

        ~50K parameters. Trains in <60s on CPU.
        """

        # Canonical opcode ordering (must match training)
        OPCODES = [
            "OP_MOV_REG_IMM", "OP_MOV_REG_REG",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
            "OP_AND", "OP_OR", "OP_XOR",
            "OP_SHL", "OP_SHR",
            "OP_INC", "OP_DEC",
            "OP_CMP",
            "OP_JMP", "OP_JZ", "OP_JNZ", "OP_JS", "OP_JNS",
            "OP_HALT", "OP_NOP", "OP_INVALID",
        ]

        def __init__(self, vocab_size: int = 128, embed_dim: int = 32,
                     hidden_dim: int = 128, max_len: int = 64) -> None:
            super().__init__()
            num_opcodes = len(self.OPCODES)
            self.max_len = max_len
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_opcodes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: [B, max_len] char indices → [B, num_opcodes] logits."""
            e = self.embed(x).transpose(1, 2)       # [B, embed_dim, max_len]
            h = torch.relu(self.conv1(e))
            h = torch.relu(self.conv2(h))
            h = torch.relu(self.conv3(h))
            h = h.max(dim=2).values                  # [B, hidden_dim]
            return self.classifier(h)                 # [B, num_opcodes]

        @staticmethod
        def encode_instruction(text: str, max_len: int = 64) -> torch.Tensor:
            """Encode an instruction string as a padded character tensor."""
            chars = [min(ord(c), 127) for c in text[:max_len]]
            chars += [0] * (max_len - len(chars))
            return torch.tensor(chars, dtype=torch.long)


else:
    # Provide no-op stubs so imports don't explode without torch.
    class _Stub:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):  # noqa: D401
            raise ImportError("torch is required for neural architecture classes")

    NeuralRegisterFile = _Stub  # type: ignore[assignment,misc]
    NeuralStack = _Stub  # type: ignore[assignment,misc]
    NeuralPointer = _Stub  # type: ignore[assignment,misc]
    NeuralFunctionCall = _Stub  # type: ignore[assignment,misc]
    NeuralARM64Decoder = _Stub  # type: ignore[assignment,misc]
    NeuralAtan2 = _Stub  # type: ignore[assignment,misc]
    NeuralSinCos = _Stub  # type: ignore[assignment,misc]
    NeuralSqrt = _Stub  # type: ignore[assignment,misc]
    NeuralExp = _Stub  # type: ignore[assignment,misc]
    NeuralLog = _Stub  # type: ignore[assignment,misc]
    DoomTrigLUT = _Stub  # type: ignore[assignment,misc]
    InstructionDecoderNet = _Stub  # type: ignore[assignment,misc]
