"""Metal Shader Source for nCPU ISA Compute Kernel.

This module contains the Metal Shading Language (MSL) source code for
executing nCPU ISA programs directly on the GPU as a compute shader.

This is a qemu-style fetch-decode-execute interpreter that runs entirely
on the GPU. The GPU's native ALU does the actual arithmetic — no neural
networks, no tensor ops. Programs that run through neural networks at
~5K IPS instead run through raw GPU ALU at millions of IPS.

ARCHITECTURE:
=============

The kernel executes instructions in a tight loop on the GPU:
1. FETCH:  Read 32-bit instruction word from memory at PC
2. DECODE: Extract opcode, registers, immediate via bit ops
3. EXECUTE: Switch on opcode, compute result with GPU native ALU
4. FLAGS: Update Z (zero) and N (sign) flags from result
5. WRITEBACK: Store result in register file, advance PC
6. REPEAT: Until HALT or max_cycles

INSTRUCTION ENCODING (32-bit words):
=====================================

    [31:24] opcode (8 bits)
    [23:21] dest register (3 bits, R0-R7)
    [20:18] src1 register (3 bits)
    [17:15] src2 register (3 bits)
    [14:0]  immediate value (15 bits, sign-extended to 64-bit)

REGISTER FILE:
==============

8 general-purpose registers (R0-R7) stored as int64 (64-bit data path).
2 flags: ZF (zero), SF (sign/negative).
PC is word-addressed (PC=4 means instruction at memory index 4).

OPCODES:
========

    0x00 = NOP       0x10 = MOV_IMM   0x20 = ADD   0x30 = AND   0x40 = INC
    0x01 = HALT      0x11 = MOV_REG   0x21 = SUB   0x31 = OR    0x41 = DEC
                                       0x22 = MUL   0x32 = XOR   0x50 = CMP
                                       0x23 = DIV   0x33 = SHL   0x60 = JMP
                                                     0x34 = SHR   0x61 = JZ
                                                                   0x62 = JNZ
                                                                   0x63 = JS
                                                                   0x64 = JNS
"""

# Stop reason constants (shared with Python wrapper)
NCPU_STOP_RUNNING = 0
NCPU_STOP_HALT = 1
NCPU_STOP_MAX_CYCLES = 2

# Metal kernel header — type definitions and constants
NCPU_KERNEL_HEADER = """
// nCPU ISA Compute Kernel — Metal Shading Language
// qemu-style fetch-decode-execute on GPU

// Stop reasons
constant uint8_t STOP_RUNNING    = 0;
constant uint8_t STOP_HALT       = 1;
constant uint8_t STOP_MAX_CYCLES = 2;

// Opcodes
constant uint8_t OP_NOP     = 0x00;
constant uint8_t OP_HALT    = 0x01;
constant uint8_t OP_MOV_IMM = 0x10;
constant uint8_t OP_MOV_REG = 0x11;
constant uint8_t OP_ADD     = 0x20;
constant uint8_t OP_SUB     = 0x21;
constant uint8_t OP_MUL     = 0x22;
constant uint8_t OP_DIV     = 0x23;
constant uint8_t OP_AND     = 0x30;
constant uint8_t OP_OR      = 0x31;
constant uint8_t OP_XOR     = 0x32;
constant uint8_t OP_SHL     = 0x33;
constant uint8_t OP_SHR     = 0x34;
constant uint8_t OP_INC     = 0x40;
constant uint8_t OP_DEC     = 0x41;
constant uint8_t OP_CMP     = 0x50;
constant uint8_t OP_JMP     = 0x60;
constant uint8_t OP_JZ      = 0x61;
constant uint8_t OP_JNZ     = 0x62;
constant uint8_t OP_JS      = 0x63;
constant uint8_t OP_JNS     = 0x64;
"""

# Metal kernel source — the actual compute shader
NCPU_KERNEL_SOURCE = """
    // Thread index (single-threaded execution)
    uint tid = thread_position_in_grid.x;
    if (tid != 0) return;

    // ─── Load state into local variables ───
    int64_t regs[8];
    for (int i = 0; i < 8; i++) {
        regs[i] = registers_in[i];
    }
    uint32_t pc = (uint32_t)pc_in[0];
    bool flag_z = (flags_in[0] > 0.5f);
    bool flag_s = (flags_in[1] > 0.5f);
    uint32_t max_cycles = (uint32_t)max_cycles_in[0];
    uint32_t num_instructions = (uint32_t)program_size_in[0];

    uint32_t cycles = 0;
    uint8_t stop_reason = STOP_RUNNING;

    // ─── Main fetch-decode-execute loop ───
    while (cycles < max_cycles) {
        // FETCH: bounds check then read instruction word
        if (pc >= num_instructions) {
            stop_reason = STOP_HALT;
            break;
        }
        uint32_t inst = (uint32_t)program_in[pc];

        // DECODE: extract fields from 32-bit instruction word
        uint8_t opcode = (uint8_t)((inst >> 24) & 0xFF);
        uint8_t rd     = (uint8_t)((inst >> 21) & 0x7);
        uint8_t rs1    = (uint8_t)((inst >> 18) & 0x7);
        uint8_t rs2    = (uint8_t)((inst >> 15) & 0x7);
        int64_t imm    = (int64_t)(inst & 0x7FFF);
        // Sign-extend 15-bit immediate to 64-bit
        if (imm & 0x4000) {
            imm |= (int64_t)0xFFFFFFFFFFFF8000LL;
        }

        // EXECUTE: dispatch on opcode
        int64_t result;
        bool advance_pc = true;
        bool update_flags = false;

        switch (opcode) {

        case OP_NOP:
            break;

        case OP_HALT:
            stop_reason = STOP_HALT;
            break;

        case OP_MOV_IMM:
            result = imm;
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_MOV_REG:
            result = regs[rs1];
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_ADD:
            result = regs[rs1] + regs[rs2];
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_SUB:
            result = regs[rs1] - regs[rs2];
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_MUL:
            result = regs[rs1] * regs[rs2];
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_DIV:
            if (regs[rs2] == 0) {
                result = 0;
            } else {
                result = regs[rs1] / regs[rs2];
            }
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_AND:
            result = regs[rs1] & regs[rs2];
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_OR:
            result = regs[rs1] | regs[rs2];
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_XOR:
            result = regs[rs1] ^ regs[rs2];
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_SHL: {
            // If rs2 != 0 and imm == 0: shift by register, else by immediate
            int64_t amount;
            if (rs2 != 0 && (inst & 0x7FFF) == 0) {
                amount = regs[rs2];
            } else {
                amount = imm;
            }
            if (amount < 0) amount = 0;
            if (amount > 63) amount = 63;
            result = regs[rs1] << amount;
            regs[rd] = result;
            update_flags = true;
            break;
        }

        case OP_SHR: {
            int64_t amount;
            if (rs2 != 0 && (inst & 0x7FFF) == 0) {
                amount = regs[rs2];
            } else {
                amount = imm;
            }
            if (amount < 0) amount = 0;
            if (amount > 63) amount = 63;
            // Logical shift right (unsigned)
            result = (int64_t)((uint64_t)regs[rs1] >> (uint64_t)amount);
            regs[rd] = result;
            update_flags = true;
            break;
        }

        case OP_INC:
            result = regs[rd] + 1;
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_DEC:
            result = regs[rd] - 1;
            regs[rd] = result;
            update_flags = true;
            break;

        case OP_CMP:
            result = regs[rs1] - regs[rs2];
            update_flags = true;
            break;

        case OP_JMP:
            pc = (uint32_t)imm;
            advance_pc = false;
            break;

        case OP_JZ:
            if (flag_z) {
                pc = (uint32_t)imm;
                advance_pc = false;
            }
            break;

        case OP_JNZ:
            if (!flag_z) {
                pc = (uint32_t)imm;
                advance_pc = false;
            }
            break;

        case OP_JS:
            if (flag_s) {
                pc = (uint32_t)imm;
                advance_pc = false;
            }
            break;

        case OP_JNS:
            if (!flag_s) {
                pc = (uint32_t)imm;
                advance_pc = false;
            }
            break;

        default:
            // Unknown opcode — halt
            stop_reason = STOP_HALT;
            break;
        }

        // Early exit on HALT
        if (stop_reason == STOP_HALT) {
            cycles++;
            break;
        }

        // Update flags from result
        if (update_flags) {
            flag_z = (result == 0);
            flag_s = (result < 0);
        }

        // Advance PC
        if (advance_pc) {
            pc++;
        }

        cycles++;
    }

    // If we exited the loop without halt, it's max_cycles
    if (stop_reason == STOP_RUNNING) {
        stop_reason = STOP_MAX_CYCLES;
    }

    // ─── Write back state ───
    for (int i = 0; i < 8; i++) {
        registers_out[i] = regs[i];
    }
    pc_out[0] = (int64_t)pc;
    flags_out[0] = flag_z ? 1.0f : 0.0f;
    flags_out[1] = flag_s ? 1.0f : 0.0f;
    cycles_out[0] = cycles;
    stop_reason_out[0] = stop_reason;
"""


def get_ncpu_kernel_source() -> str:
    """Get the full Metal kernel source."""
    return NCPU_KERNEL_SOURCE


def get_ncpu_kernel_header() -> str:
    """Get the Metal kernel header."""
    return NCPU_KERNEL_HEADER
