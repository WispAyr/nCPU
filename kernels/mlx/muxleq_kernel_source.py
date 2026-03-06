"""Metal Shader Source for MUXLEQ Compute Kernel.

MUXLEQ is a two-instruction-set computer (multiplexed SUBLEQ):
  1. SUBLEQ: subtract-and-branch-if-less-or-equal-to-zero
  2. MUX: bitwise multiplex (select bits from two sources via mask)

Combined with I/O via sentinel addresses, this is Turing-complete and
runs a full eForth system. The VM has 65,536 words of 16-bit memory.

INSTRUCTION FORMAT:
==================

Each instruction is a triple of 3 consecutive 16-bit words: (a, b, c)

  Case 1 — INPUT:   a == 0xFFFF         → m[b] = input_byte
  Case 2 — OUTPUT:  b == 0xFFFF         → output m[a] & 0xFF
  Case 3 — MUX:     c & 0x8000 and      → mask = m[c & 0x7FFF]
                     c != 0xFFFF           m[b] = (m[a] & ~mask) | (m[b] & mask)
  Case 4 — SUBLEQ:  otherwise           → r = m[b] - m[a]; m[b] = r
                                           if r == 0 or r & 0x8000: pc = c

HALT: pc >= 0x8000 (32768)
I/O:  handled by pausing GPU and returning to Python (like SVC traps)
"""

# Stop reason constants
MUXLEQ_STOP_RUNNING = 0
MUXLEQ_STOP_HALT = 1
MUXLEQ_STOP_MAX_CYCLES = 2
MUXLEQ_STOP_IO_READ = 3   # Paused for input — Python must provide byte
MUXLEQ_STOP_IO_WRITE = 4  # Paused for output — Python reads the byte

# Metal kernel header
MUXLEQ_KERNEL_HEADER = """
// MUXLEQ Compute Kernel — Metal Shading Language
// Two-instruction-set computer on GPU

constant uint8_t STOP_RUNNING    = 0;
constant uint8_t STOP_HALT       = 1;
constant uint8_t STOP_MAX_CYCLES = 2;
constant uint8_t STOP_IO_READ    = 3;
constant uint8_t STOP_IO_WRITE   = 4;

constant uint16_t SENTINEL = 0xFFFF;
constant uint16_t HALT_THRESHOLD = 0x8000;
constant uint16_t MUX_FLAG = 0x8000;
constant uint16_t MUX_ADDR_MASK = 0x7FFF;
"""

# Metal kernel source — tight fetch-decode-execute loop
MUXLEQ_KERNEL_SOURCE = """
    uint tid = thread_position_in_grid.x;
    if (tid != 0) return;

    // Load state
    uint32_t pc = (uint32_t)pc_in[0];
    uint32_t max_cycles = (uint32_t)max_cycles_in[0];
    uint32_t cycles = 0;
    uint8_t stop_reason = STOP_RUNNING;

    // I/O communication slots
    // io_out[0] = address for I/O operation
    // io_out[1] = data byte for output
    io_out[0] = 0;
    io_out[1] = 0;

    // Copy memory_in to memory_out (double-buffer)
    for (uint32_t i = 0; i < 65536; i++) {
        memory_out[i] = memory_in[i];
    }

    // Main fetch-decode-execute loop
    while (cycles < max_cycles) {
        // HALT: pc >= 32768
        if (pc >= HALT_THRESHOLD) {
            stop_reason = STOP_HALT;
            break;
        }

        // FETCH: read triple (a, b, c)
        uint16_t a = (uint16_t)memory_out[pc];
        uint16_t b = (uint16_t)memory_out[pc + 1];
        uint16_t c = (uint16_t)memory_out[pc + 2];
        pc += 3;

        // DECODE + EXECUTE
        if (a == SENTINEL) {
            // Case 1: INPUT — pause and let Python provide input
            io_out[0] = (uint32_t)b;  // destination address
            stop_reason = STOP_IO_READ;
            pc -= 3;  // rewind so Python can re-dispatch after providing input
            cycles++;
            break;
        }
        else if (b == SENTINEL) {
            // Case 2: OUTPUT — pause and let Python read the byte
            io_out[0] = (uint32_t)a;              // source address
            io_out[1] = memory_out[a] & 0xFF;     // the byte to output
            stop_reason = STOP_IO_WRITE;
            cycles++;
            // PC already advanced past this instruction — don't rewind
            break;
        }
        else if ((c & MUX_FLAG) && c != SENTINEL) {
            // Case 3: MUX — bitwise multiplex
            uint16_t mask_addr = c & MUX_ADDR_MASK;
            uint16_t mask = (uint16_t)memory_out[mask_addr];
            memory_out[b] = (uint16_t)((memory_out[a] & ~mask) | (memory_out[b] & mask));
        }
        else {
            // Case 4: SUBLEQ — subtract and branch if <= 0
            uint16_t r = (uint16_t)(memory_out[b] - memory_out[a]);
            memory_out[b] = r;
            if (r == 0 || (r & MUX_FLAG)) {
                pc = (uint32_t)c;
            }
        }

        cycles++;
    }

    // Max cycles reached?
    if (stop_reason == STOP_RUNNING) {
        stop_reason = STOP_MAX_CYCLES;
    }

    // Write back state
    pc_out[0] = (int64_t)pc;
    cycles_out[0] = cycles;
    stop_reason_out[0] = stop_reason;
"""
