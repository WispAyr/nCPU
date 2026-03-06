"""neurOS System Call Interface.

POSIX-like syscall table providing the interface between user processes
and the neurOS kernel. Each syscall dispatches to the appropriate OS
component (filesystem, process table, IPC, etc.).

Syscall convention (ARM64-like):
    - Syscall number in X8
    - Arguments in X0-X5
    - Return value in X0
    - SVC instruction triggers syscall

The syscall table is a dispatch dictionary mapping syscall numbers
to handler functions. All handlers operate on GPU tensors.
"""

import torch
import logging
from typing import Optional, Dict, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .boot import NeurOS

from .device import default_device
from .process import ProcessState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Syscall Numbers (Linux ARM64 convention where possible)
# ═══════════════════════════════════════════════════════════════════════════════

SYS_EXIT = 93
SYS_READ = 63
SYS_WRITE = 64
SYS_OPEN = 56
SYS_CLOSE = 57
SYS_LSEEK = 62
SYS_BRK = 214
SYS_MMAP = 222
SYS_MUNMAP = 215
SYS_FORK = 220
SYS_EXEC = 221
SYS_WAIT = 260
SYS_GETPID = 172
SYS_GETPPID = 173
SYS_KILL = 129
SYS_PIPE = 59
SYS_DUP = 23
SYS_MKDIR = 34
SYS_RMDIR = 35
SYS_UNLINK = 87
SYS_STAT = 79
SYS_YIELD = 124
SYS_SLEEP = 101
SYS_SEND = 300     # IPC send (neurOS extension)
SYS_RECV = 301     # IPC receive (neurOS extension)
SYS_SHM_CREATE = 302
SYS_SHM_OPEN = 303
SYS_SHM_CLOSE = 304
SYS_LISTDIR = 305  # Directory listing (neurOS extension)


# ═══════════════════════════════════════════════════════════════════════════════
# Syscall Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

SyscallHandler = Callable[['SyscallInterface', int, list], int]


class SyscallInterface:
    """System call dispatch interface for neurOS.

    Routes syscall numbers to handler methods. Each handler receives
    the calling PID and a list of arguments, and returns a result code.

    Usage:
        result = syscalls.dispatch(pid, syscall_num, [arg0, arg1, ...])
    """

    def __init__(self, os: 'NeurOS'):
        self.os = os
        self.device = os.device if hasattr(os, 'device') else default_device()
        self.total_calls = 0

        # Syscall dispatch table
        self._table: Dict[int, SyscallHandler] = {
            SYS_EXIT: self._sys_exit,
            SYS_READ: self._sys_read,
            SYS_WRITE: self._sys_write,
            SYS_OPEN: self._sys_open,
            SYS_CLOSE: self._sys_close,
            SYS_LSEEK: self._sys_lseek,
            SYS_FORK: self._sys_fork,
            SYS_WAIT: self._sys_wait,
            SYS_GETPID: self._sys_getpid,
            SYS_GETPPID: self._sys_getppid,
            SYS_KILL: self._sys_kill,
            SYS_PIPE: self._sys_pipe,
            SYS_MKDIR: self._sys_mkdir,
            SYS_RMDIR: self._sys_rmdir,
            SYS_UNLINK: self._sys_unlink,
            SYS_STAT: self._sys_stat,
            SYS_YIELD: self._sys_yield,
            SYS_SLEEP: self._sys_sleep,
            SYS_SEND: self._sys_send,
            SYS_RECV: self._sys_recv,
            SYS_LISTDIR: self._sys_listdir,
        }

    def dispatch(self, pid: int, syscall_num: int, args: list) -> int:
        """Dispatch a syscall.

        Args:
            pid: Calling process ID
            syscall_num: Syscall number (from SYS_* constants)
            args: List of arguments

        Returns:
            Result code (0 = success, negative = error)
        """
        self.total_calls += 1

        handler = self._table.get(syscall_num)
        if handler is None:
            logger.warning(f"[SYSCALL] Unknown syscall {syscall_num} from PID {pid}")
            return -1  # ENOSYS

        return handler(pid, args)

    # ─── Process Management ───────────────────────────────────────────────

    def _sys_exit(self, pid: int, args: list) -> int:
        """exit(status) — terminate the calling process."""
        exit_code = args[0] if args else 0
        self.os.scheduler.terminate_process(pid, exit_code)
        return 0

    def _sys_fork(self, pid: int, args: list) -> int:
        """fork() — create a child process.

        Returns child PID to parent, 0 to child.
        """
        parent = self.os.process_table.get(pid)
        if parent is None:
            return -1

        child = self.os.process_table.create_process(
            name=f"{parent.name}.child",
            priority=parent.priority,
            parent_pid=pid,
        )

        # Copy parent's registers to child
        child.registers = parent.registers.clone()
        child.pc = parent.pc.clone()
        child.sp = parent.sp.clone()
        child.flags = parent.flags.clone()

        # Register child for IPC
        self.os.ipc.register_process(child.pid)

        # Return 0 in child's X0, child_pid in parent's X0
        child.registers[0] = 0
        return child.pid

    def _sys_wait(self, pid: int, args: list) -> int:
        """wait(child_pid) — wait for a child process to terminate.

        If child_pid == -1, wait for any child.
        """
        target_pid = args[0] if args else -1

        for p in self.os.process_table.all_processes:
            if p.parent_pid == pid and p.state == ProcessState.ZOMBIE:
                if target_pid == -1 or p.pid == target_pid:
                    exit_code = p.exit_code
                    self.os.process_table.remove(p.pid)
                    return exit_code

        # No zombie children — block
        self.os.scheduler.block_process(pid, f"wait({target_pid})")
        return 0

    def _sys_getpid(self, pid: int, args: list) -> int:
        """getpid() — return calling process's PID."""
        return pid

    def _sys_getppid(self, pid: int, args: list) -> int:
        """getppid() — return parent's PID."""
        pcb = self.os.process_table.get(pid)
        return pcb.parent_pid if pcb else -1

    def _sys_kill(self, pid: int, args: list) -> int:
        """kill(target_pid, signal) — send signal to a process."""
        if len(args) < 2:
            return -1
        target_pid, signal = args[0], args[1]
        self.os.ipc.signal_send(pid, target_pid, signal)
        return 0

    def _sys_yield(self, pid: int, args: list) -> int:
        """yield() — voluntarily yield CPU."""
        pcb = self.os.process_table.get(pid)
        if pcb:
            pcb.state = ProcessState.READY
            pcb.ticks_remaining = 0
        return 0

    def _sys_sleep(self, pid: int, args: list) -> int:
        """sleep(ticks) — sleep for given number of ticks."""
        ticks = args[0] if args else 0
        self.os.scheduler.block_process(pid, f"sleep({ticks})")
        return 0

    # ─── Filesystem ───────────────────────────────────────────────────────

    def _sys_open(self, pid: int, args: list) -> int:
        """open(path, mode) — open a file."""
        if not args:
            return -1
        path = args[0] if isinstance(args[0], str) else f"/file_{args[0]}"
        mode = args[1] if len(args) > 1 else "r"
        if not isinstance(mode, str):
            mode = "r"
        return self.os.fs.open(path, mode)

    def _sys_close(self, pid: int, args: list) -> int:
        """close(fd) — close a file descriptor."""
        if not args:
            return -1
        fd = args[0]
        return 0 if self.os.fs.close(fd) else -1

    def _sys_read(self, pid: int, args: list) -> int:
        """read(fd, size) — read from file descriptor."""
        if len(args) < 2:
            return -1
        fd, size = args[0], args[1]
        data = self.os.fs.read(fd, size)
        return len(data) if data is not None else -1

    def _sys_write(self, pid: int, args: list) -> int:
        """write(fd, data) — write to file descriptor."""
        if len(args) < 2:
            return -1
        fd = args[0]
        data = args[1]
        if isinstance(data, torch.Tensor):
            return self.os.fs.write(fd, data)
        return -1

    def _sys_lseek(self, pid: int, args: list) -> int:
        """lseek(fd, offset, whence) — seek in file."""
        if len(args) < 3:
            return -1
        return self.os.fs.seek(args[0], args[1], args[2])

    def _sys_mkdir(self, pid: int, args: list) -> int:
        """mkdir(path) — create directory."""
        if not args or not isinstance(args[0], str):
            return -1
        return self.os.fs.mkdir(args[0])

    def _sys_rmdir(self, pid: int, args: list) -> int:
        """rmdir(path) — remove empty directory."""
        if not args or not isinstance(args[0], str):
            return -1
        return 0 if self.os.fs.rmdir(args[0]) else -1

    def _sys_unlink(self, pid: int, args: list) -> int:
        """unlink(path) — remove a file."""
        if not args or not isinstance(args[0], str):
            return -1
        return 0 if self.os.fs.unlink(args[0]) else -1

    def _sys_stat(self, pid: int, args: list) -> int:
        """stat(path) — get file metadata. Returns inode number or -1."""
        if not args or not isinstance(args[0], str):
            return -1
        info = self.os.fs.stat(args[0])
        return info["ino"] if info else -1

    def _sys_listdir(self, pid: int, args: list) -> int:
        """listdir(path) — list directory. Returns number of entries or -1."""
        if not args or not isinstance(args[0], str):
            return -1
        entries = self.os.fs.list_dir(args[0])
        return len(entries) if entries is not None else -1

    # ─── IPC ──────────────────────────────────────────────────────────────

    def _sys_send(self, pid: int, args: list) -> int:
        """send(dst_pid, payload, tag) — send IPC message."""
        if len(args) < 2:
            return -1
        dst_pid = args[0]
        payload = args[1] if isinstance(args[1], torch.Tensor) else None
        tag = args[2] if len(args) > 2 else 0
        return 0 if self.os.ipc.send(pid, dst_pid, payload, tag=tag) else -1

    def _sys_recv(self, pid: int, args: list) -> int:
        """recv(tag) — receive IPC message. Returns 0 if message received, -1 if empty."""
        tag = args[0] if args else None
        msg = self.os.ipc.receive(pid, tag)
        return 0 if msg is not None else -1

    def _sys_pipe(self, pid: int, args: list) -> int:
        """pipe(other_pid) — create a pipe to another process."""
        if not args:
            return -1
        other_pid = args[0]
        self.os.ipc.pipe_create(pid, other_pid)
        return 0

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "total_calls": self.total_calls,
            "registered_syscalls": len(self._table),
        }
