"""TensorKernel: Linux syscall emulation for tensor-native ARM64 execution.

Combines the TensorCPU with syscall handling to run real ARM64 binaries.
Supports SYS_WRITE, SYS_EXIT, SYS_BRK, SYS_UNAME, and more.
"""

import time
import struct
from dataclasses import dataclass

import torch

from .cpu import TensorCPU, ExecutionStats, get_device


class Syscalls:
    SYS_READ = 63
    SYS_WRITE = 64
    SYS_EXIT = 93
    SYS_EXIT_GROUP = 94
    SYS_BRK = 214
    SYS_MMAP = 222
    SYS_MUNMAP = 215
    SYS_UNAME = 160
    SYS_GETPID = 172
    SYS_GETTID = 178
    SYS_GETUID = 174
    SYS_GETEUID = 175
    SYS_GETGID = 176
    SYS_GETEGID = 177
    SYS_CLOCK_GETTIME = 113


@dataclass
class RunResult:
    """Result of running a program."""
    instructions: int
    time_seconds: float
    ips: float
    output: str
    exit_code: int
    syscalls_handled: int


class TensorKernel:
    """Kernel that runs ARM64 binaries on the tensor-native CPU.

    Provides Linux syscall emulation for real program execution.

    Args:
        mem_size: Memory size in bytes (default 4MB)
    """

    def __init__(self, mem_size: int = 4 * 1024 * 1024):
        device = get_device()
        self.cpu = TensorCPU(mem_size, device=device)
        self.STACK_TOP = mem_size - 0x10000
        self.HEAP_BASE = 0x100000
        self.brk = self.HEAP_BASE
        self.output_buffer = ""
        self.exit_code = 0
        self.syscall_count = 0

    def load_binary(self, code: bytes, base: int = 0x1000):
        """Load binary code into memory."""
        device = self.cpu.device
        code_tensor = torch.tensor(list(code), dtype=torch.uint8, device=device)
        self.cpu.memory[base:base + len(code)] = code_tensor
        self.cpu.pc = torch.tensor(base, dtype=torch.int64, device=device)

    def setup_stack(self, argv: list = None, envp: list = None):
        """Set up the stack with argc, argv, envp."""
        argv = argv or ["program"]
        envp = envp or []

        sp = self.STACK_TOP
        string_ptrs = []
        for s in argv + envp + [""]:
            s_bytes = s.encode('utf-8') + b'\x00'
            sp -= len(s_bytes)
            sp = sp & ~0x7
            for i, b in enumerate(s_bytes):
                self.cpu.memory[sp + i] = b
            if s:
                string_ptrs.append(sp)

        sp = sp & ~0xF

        sp -= 8
        self.cpu.memory[sp:sp+8] = 0

        for ptr in reversed(string_ptrs[len(argv):]):
            sp -= 8
            for i in range(8):
                self.cpu.memory[sp + i] = (ptr >> (i * 8)) & 0xFF

        sp -= 8
        self.cpu.memory[sp:sp+8] = 0

        for ptr in reversed(string_ptrs[:len(argv)]):
            sp -= 8
            for i in range(8):
                self.cpu.memory[sp + i] = (ptr >> (i * 8)) & 0xFF

        sp -= 8
        self.cpu.memory[sp] = len(argv)
        self.cpu.regs[31] = sp

    def handle_syscall(self) -> bool:
        """Handle Linux syscall. Returns True to continue, False to halt."""
        self.syscall_count += 1

        syscall_num = int(self.cpu.regs[8].item())
        x0 = int(self.cpu.regs[0].item())
        x1 = int(self.cpu.regs[1].item())
        x2 = int(self.cpu.regs[2].item())

        if syscall_num == Syscalls.SYS_WRITE:
            fd, buf, count = x0, x1, x2
            if fd in (1, 2):
                data = bytes(self.cpu.memory[buf:buf + count].cpu().numpy())
                try:
                    text = data.decode('utf-8', errors='replace')
                except Exception:
                    text = str(data)
                self.output_buffer += text
                self.cpu.regs[0] = count
            else:
                self.cpu.regs[0] = -1
            return True

        elif syscall_num in (Syscalls.SYS_EXIT, Syscalls.SYS_EXIT_GROUP):
            self.exit_code = x0
            self.cpu.halted = True
            return False

        elif syscall_num == Syscalls.SYS_BRK:
            if x0 == 0:
                self.cpu.regs[0] = self.brk
            elif x0 > self.brk:
                self.brk = x0
                self.cpu.regs[0] = self.brk
            else:
                self.cpu.regs[0] = self.brk
            return True

        elif syscall_num == Syscalls.SYS_UNAME:
            buf = x0
            fields = [b"Linux", b"neural", b"6.1.0-neural", b"#1 SMP", b"aarch64"]
            offset = 0
            for fld in fields:
                fld = fld.ljust(65, b'\x00')
                for i, b in enumerate(fld[:65]):
                    self.cpu.memory[buf + offset + i] = b
                offset += 65
            self.cpu.regs[0] = 0
            return True

        elif syscall_num == Syscalls.SYS_GETPID:
            self.cpu.regs[0] = 1
            return True

        elif syscall_num == Syscalls.SYS_GETTID:
            self.cpu.regs[0] = 1
            return True

        elif syscall_num in (Syscalls.SYS_GETUID, Syscalls.SYS_GETEUID,
                             Syscalls.SYS_GETGID, Syscalls.SYS_GETEGID):
            self.cpu.regs[0] = 1000
            return True

        elif syscall_num == Syscalls.SYS_CLOCK_GETTIME:
            tp = x1
            t = time.time()
            sec = int(t)
            nsec = int((t - sec) * 1e9)
            for i in range(8):
                self.cpu.memory[tp + i] = (sec >> (i * 8)) & 0xFF
            for i in range(8):
                self.cpu.memory[tp + 8 + i] = (nsec >> (i * 8)) & 0xFF
            self.cpu.regs[0] = 0
            return True

        else:
            self.cpu.regs[0] = -38  # ENOSYS
            return True

    @torch.no_grad()
    def run(self, max_instructions: int = 10_000_000, batch_size: int = 256) -> RunResult:
        """Run loaded program with tensor-native execution and syscall handling."""
        self.output_buffer = ""
        self.exit_code = 0
        self.syscall_count = 0

        start_time = time.perf_counter()
        total_instructions = 0

        while not self.cpu.halted and total_instructions < max_instructions:
            stats = self.cpu.run_batch(
                max_instructions=min(batch_size * 100, max_instructions - total_instructions),
                batch_size=batch_size
            )
            total_instructions += stats.instructions_executed

            if stats.syscalls > 0:
                if not self.handle_syscall():
                    break
                self.cpu.syscall_count = 0
                self.cpu.halted = False

        elapsed = time.perf_counter() - start_time
        ips = total_instructions / elapsed if elapsed > 0 else 0

        return RunResult(
            instructions=total_instructions,
            time_seconds=elapsed,
            ips=ips,
            output=self.output_buffer,
            exit_code=self.exit_code,
            syscalls_handled=self.syscall_count
        )


def create_hello_world() -> bytes:
    """Create a simple ARM64 hello world binary for testing."""
    code = [
        0xD2800020,  # mov x0, #1 (stdout)
    ]
    offset = 28
    imm_lo = offset & 0x3
    imm_hi = offset >> 2
    code.append(0x10000001 | (imm_lo << 29) | (imm_hi << 5))  # adr x1, .+28
    code.extend([
        0xD28001C2,  # mov x2, #14
        0xD2800808,  # mov x8, #64 (SYS_WRITE)
        0xD4000001,  # svc #0
        0xD2800000,  # mov x0, #0
        0xD2800BA8,  # mov x8, #93 (SYS_EXIT)
        0xD4000001,  # svc #0
    ])
    binary = b''.join(struct.pack('<I', inst) for inst in code)
    binary += b"Hello, World!\n"
    return binary
