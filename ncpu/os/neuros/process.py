"""GPU-Tensor Process Control Blocks (PCBs).

Every process's state lives entirely on GPU as tensors. Context switches
are tensor slice copies — no serialization, no CPU-GPU transfer.

A PCB contains:
    - pid: Process ID
    - registers: [32] int64 tensor (ARM64 general-purpose registers)
    - pc: Program counter (int64 tensor)
    - sp: Stack pointer (int64 tensor)
    - flags: [4] float tensor (N, Z, C, V)
    - state: ProcessState enum
    - priority: Scheduling priority (0-255, lower = higher priority)
    - asid: Address space ID for MMU isolation
    - cpu_time: Total CPU ticks consumed
    - memory_pages: List of allocated virtual page numbers
"""

import torch
import logging
from typing import Optional, List, Dict
from enum import IntEnum
from dataclasses import dataclass, field

from .device import default_device

logger = logging.getLogger(__name__)


class ProcessState(IntEnum):
    """Process lifecycle states."""
    CREATED = 0
    READY = 1
    RUNNING = 2
    BLOCKED = 3      # Waiting on I/O or IPC
    SLEEPING = 4     # Timed wait
    ZOMBIE = 5       # Terminated, awaiting parent wait()
    TERMINATED = 6


@dataclass
class ProcessControlBlock:
    """GPU-native Process Control Block.

    All register/flag state stored as GPU tensors.
    Context switch = tensor slice copy, no CPU round-trip.
    """
    pid: int
    name: str
    priority: int = 128          # Default mid-priority
    asid: int = 0                # Address space ID
    parent_pid: int = 0          # Parent process ID
    state: ProcessState = ProcessState.CREATED

    # These are initialized in __post_init__ as GPU tensors
    registers: Optional[torch.Tensor] = None
    pc: Optional[torch.Tensor] = None
    sp: Optional[torch.Tensor] = None
    flags: Optional[torch.Tensor] = None

    # Scheduling metadata
    cpu_time: int = 0            # Total ticks consumed
    wait_time: int = 0           # Ticks spent waiting
    time_slice: int = 100        # Ticks per scheduling quantum
    ticks_remaining: int = 100   # Ticks left in current quantum

    # Memory
    memory_pages: List[int] = field(default_factory=list)
    stack_base: int = 0
    heap_base: int = 0

    # IPC
    exit_code: int = 0
    blocked_on: Optional[str] = None  # What we're waiting on

    def __post_init__(self):
        device = default_device()
        if self.registers is None:
            self.registers = torch.zeros(32, dtype=torch.int64, device=device)
        if self.pc is None:
            self.pc = torch.tensor(0, dtype=torch.int64, device=device)
        if self.sp is None:
            self.sp = torch.tensor(0, dtype=torch.int64, device=device)
        if self.flags is None:
            self.flags = torch.zeros(4, dtype=torch.float32, device=device)


class ProcessTable:
    """Manages all processes in the system.

    The process table is the central registry. All PCBs are stored here,
    and the scheduler queries this table for runnable processes.
    """

    def __init__(self, max_processes: int = 256,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.max_processes = max_processes
        self._processes: Dict[int, ProcessControlBlock] = {}
        self._next_pid = 1  # PID 0 reserved for kernel/idle

    def create_process(self, name: str, priority: int = 128,
                       parent_pid: int = 0) -> ProcessControlBlock:
        """Create a new process and add it to the process table.

        Returns:
            The new ProcessControlBlock
        Raises:
            RuntimeError if process table is full
        """
        if len(self._processes) >= self.max_processes:
            raise RuntimeError("Process table full")

        pid = self._next_pid
        self._next_pid += 1

        asid = pid % 256  # Simple ASID assignment

        pcb = ProcessControlBlock(
            pid=pid,
            name=name,
            priority=priority,
            asid=asid,
            parent_pid=parent_pid,
            state=ProcessState.READY,
        )

        self._processes[pid] = pcb
        logger.debug(f"[ProcessTable] Created PID {pid}: {name}")
        return pcb

    def get(self, pid: int) -> Optional[ProcessControlBlock]:
        """Get a process by PID."""
        return self._processes.get(pid)

    def remove(self, pid: int) -> bool:
        """Remove a terminated process from the table."""
        if pid in self._processes:
            del self._processes[pid]
            return True
        return False

    def ready_processes(self) -> List[ProcessControlBlock]:
        """Get all processes in READY state, sorted by priority."""
        ready = [p for p in self._processes.values()
                 if p.state == ProcessState.READY]
        ready.sort(key=lambda p: p.priority)
        return ready

    def running_process(self) -> Optional[ProcessControlBlock]:
        """Get the currently running process (if any)."""
        for p in self._processes.values():
            if p.state == ProcessState.RUNNING:
                return p
        return None

    def blocked_processes(self) -> List[ProcessControlBlock]:
        """Get all blocked processes."""
        return [p for p in self._processes.values()
                if p.state == ProcessState.BLOCKED]

    @property
    def count(self) -> int:
        """Number of processes in the table."""
        return len(self._processes)

    @property
    def all_processes(self) -> List[ProcessControlBlock]:
        """All processes, sorted by PID."""
        return sorted(self._processes.values(), key=lambda p: p.pid)

    def context_switch(self, from_pcb: Optional[ProcessControlBlock],
                       to_pcb: ProcessControlBlock) -> None:
        """Perform a context switch between two processes.

        This is a GPU tensor operation — registers are copied
        as tensor slices, no CPU serialization needed.
        """
        if from_pcb is not None:
            # Save: nothing to do, state already in GPU tensors
            if from_pcb.state == ProcessState.RUNNING:
                from_pcb.state = ProcessState.READY

        to_pcb.state = ProcessState.RUNNING
        to_pcb.ticks_remaining = to_pcb.time_slice
        logger.debug(f"[ProcessTable] Switch: "
                     f"PID {from_pcb.pid if from_pcb else '-'} → PID {to_pcb.pid}")

    def stats(self) -> Dict:
        """Process table statistics."""
        states = {}
        for p in self._processes.values():
            name = ProcessState(p.state).name
            states[name] = states.get(name, 0) + 1

        return {
            "total": self.count,
            "max": self.max_processes,
            "states": states,
        }

    def __repr__(self) -> str:
        return f"ProcessTable({self.count}/{self.max_processes} processes)"
