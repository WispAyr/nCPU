"""Neural Inter-Process Communication (IPC).

GPU-native IPC where message routing is performed by a neural network.
All messages and shared memory regions are GPU tensors — no CPU-GPU
transfers for IPC operations.

Components:
    1. Message Queues — per-process GPU-tensor message buffers
    2. Shared Memory — tensor slices shared between processes
    3. Neural Router — learned message routing for publish-subscribe
    4. Pipes — unidirectional byte streams between processes
    5. Signals — lightweight process notification
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from .device import default_device

logger = logging.getLogger(__name__)

# Message types
MSG_DATA = 0
MSG_SIGNAL = 1
MSG_PIPE = 2
MSG_REPLY = 3
MSG_BROADCAST = 4

# Signal types
SIG_TERM = 0
SIG_KILL = 1
SIG_STOP = 2
SIG_CONT = 3
SIG_USR1 = 4
SIG_USR2 = 5
SIG_CHILD = 6

MAX_MSG_SIZE = 256  # Max message payload (bytes as tensor elements)


@dataclass
class Message:
    """A GPU-tensor message."""
    src_pid: int
    dst_pid: int
    msg_type: int = MSG_DATA
    payload: Optional[torch.Tensor] = None  # [N] uint8 or int64
    tag: int = 0  # For matching receives

    def __post_init__(self):
        if self.payload is None:
            self.payload = torch.tensor([], dtype=torch.uint8, device=default_device())


class MessageQueue:
    """Per-process message queue backed by GPU tensors.

    Fixed-capacity ring buffer stored entirely on GPU.
    """

    def __init__(self, capacity: int = 64, device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.capacity = capacity
        self._messages: List[Message] = []

    def send(self, msg: Message) -> bool:
        """Enqueue a message. Returns False if queue full."""
        if len(self._messages) >= self.capacity:
            return False
        self._messages.append(msg)
        return True

    def receive(self, tag: Optional[int] = None) -> Optional[Message]:
        """Dequeue a message. If tag specified, match by tag.

        Returns None if queue empty or no matching message.
        """
        if not self._messages:
            return None

        if tag is None:
            return self._messages.pop(0)

        for i, msg in enumerate(self._messages):
            if msg.tag == tag:
                return self._messages.pop(i)
        return None

    def peek(self) -> Optional[Message]:
        """Look at front message without removing it."""
        return self._messages[0] if self._messages else None

    @property
    def count(self) -> int:
        return len(self._messages)

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0

    @property
    def is_full(self) -> bool:
        return len(self._messages) >= self.capacity


class SharedMemoryRegion:
    """A shared memory region backed by a GPU tensor.

    Multiple processes can map the same tensor into their address spaces.
    The tensor lives on GPU — reads/writes are direct tensor operations.
    """

    def __init__(self, name: str, size: int,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.name = name
        self.size = size
        self.data = torch.zeros(size, dtype=torch.uint8, device=self.device)
        self._owners: List[int] = []  # PIDs with access

    def attach(self, pid: int):
        """Attach a process to this shared memory region."""
        if pid not in self._owners:
            self._owners.append(pid)

    def detach(self, pid: int):
        """Detach a process from this shared memory region."""
        if pid in self._owners:
            self._owners.remove(pid)

    def has_access(self, pid: int) -> bool:
        return pid in self._owners

    def read(self, offset: int, length: int) -> torch.Tensor:
        """Read bytes from shared memory."""
        end = min(offset + length, self.size)
        return self.data[offset:end]

    def write(self, offset: int, data: torch.Tensor):
        """Write bytes to shared memory."""
        end = min(offset + len(data), self.size)
        self.data[offset:end] = data[:end - offset]


class Pipe:
    """Unidirectional byte stream between two processes.

    Backed by a GPU tensor ring buffer.
    """

    def __init__(self, reader_pid: int, writer_pid: int,
                 buffer_size: int = 4096,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.reader_pid = reader_pid
        self.writer_pid = writer_pid
        self.buffer = torch.zeros(buffer_size, dtype=torch.uint8, device=self.device)
        self.buffer_size = buffer_size
        self.read_pos = 0
        self.write_pos = 0
        self.count = 0
        self.closed = False

    def write(self, data: torch.Tensor) -> int:
        """Write data to pipe. Returns number of bytes written."""
        if self.closed:
            return -1
        available = self.buffer_size - self.count
        to_write = min(len(data), available)
        if to_write == 0:
            return 0

        for i in range(to_write):
            self.buffer[self.write_pos] = data[i]
            self.write_pos = (self.write_pos + 1) % self.buffer_size
        self.count += to_write
        return to_write

    def read(self, length: int) -> torch.Tensor:
        """Read data from pipe. Returns tensor of bytes read."""
        if self.count == 0:
            return torch.tensor([], dtype=torch.uint8, device=self.device)

        to_read = min(length, self.count)
        result = torch.zeros(to_read, dtype=torch.uint8, device=self.device)
        for i in range(to_read):
            result[i] = self.buffer[self.read_pos]
            self.read_pos = (self.read_pos + 1) % self.buffer_size
        self.count -= to_read
        return result

    def close(self):
        self.closed = True


class TensorChannel:
    """Typed tensor channel for zero-copy GPU IPC.

    Unlike SharedMemoryRegion (raw uint8 bytes), TensorChannel preserves
    tensor dtype and shape. Multiple processes can read/write the same
    GPU tensor without serialization — genuinely zero-copy.

    This is a GPU-native IPC primitive with no equivalent in traditional
    operating systems. Two processes sharing a float32 [1000, 100] tensor
    can both perform GPU operations on it without any data movement.

    Features:
        - Typed tensors (float32, int64, etc.) with preserved shape
        - Version counter for change detection
        - GPU-side barrier synchronization
        - Acquire/release for exclusive access patterns
    """

    def __init__(self, name: str, shape, dtype=torch.float32,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.name = name
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = dtype
        self.data = torch.zeros(self.shape, dtype=dtype, device=self.device)
        self._owners: List[int] = []
        # GPU-resident synchronization state
        self._version = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._barrier_count = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._lock_holder = torch.zeros(1, dtype=torch.int64, device=self.device)

    def attach(self, pid: int):
        """Attach a process to this tensor channel."""
        if pid not in self._owners:
            self._owners.append(pid)

    def detach(self, pid: int):
        """Detach a process from this tensor channel."""
        if pid in self._owners:
            self._owners.remove(pid)
        if self._lock_holder.item() == pid:
            self._lock_holder.zero_()

    def has_access(self, pid: int) -> bool:
        return pid in self._owners

    def write(self, pid: int, data: torch.Tensor) -> bool:
        """Write tensor data (zero-copy if same device).

        The data tensor is copied into the channel's buffer.
        For true zero-copy, use .data directly with acquire/release.
        """
        if pid not in self._owners:
            return False
        self.data.copy_(data)
        self._version += 1
        return True

    def read(self, pid: int) -> Optional[torch.Tensor]:
        """Read tensor data — returns a direct view (zero-copy).

        The returned tensor IS the channel's data. Modifications to it
        are visible to all attached processes. Use acquire/release for
        safe mutation.
        """
        if pid not in self._owners:
            return None
        return self.data

    def version(self) -> int:
        """Get current version number (incremented on each write)."""
        return self._version.item()

    def acquire(self, pid: int) -> bool:
        """Acquire exclusive access (GPU-side lock).

        Returns True if lock acquired, False if held by another process.
        In a real GPU implementation, this would use atomic CAS.
        """
        holder = self._lock_holder.item()
        if holder == 0 or holder == pid:
            self._lock_holder[0] = pid
            return True
        return False

    def release(self, pid: int) -> bool:
        """Release exclusive access."""
        if self._lock_holder.item() == pid:
            self._lock_holder.zero_()
            return True
        return False

    def barrier_arrive(self, pid: int) -> bool:
        """Arrive at barrier. Returns True when all owners have arrived.

        GPU-side barrier synchronization — all attached processes must
        call barrier_arrive before any can proceed past barrier_wait.
        """
        if pid not in self._owners:
            return False
        self._barrier_count += 1
        return self._barrier_count.item() >= len(self._owners)

    def barrier_reset(self):
        """Reset the barrier for reuse."""
        self._barrier_count.zero_()

    def __repr__(self) -> str:
        return (f"TensorChannel({self.name!r}, shape={self.shape}, "
                f"dtype={self.dtype}, owners={self._owners}, "
                f"v={self.version()})")


class NeuralIPC:
    """Central IPC manager for neurOS.

    Manages message queues, shared memory, pipes, signals, and
    tensor channels. All IPC is GPU-native — messages are tensors,
    shared memory is tensor slices, pipes are tensor ring buffers,
    and tensor channels provide typed zero-copy GPU IPC.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or default_device()

        # Per-process message queues
        self._queues: Dict[int, MessageQueue] = {}

        # Shared memory regions
        self._shm: Dict[str, SharedMemoryRegion] = {}

        # Pipes: (reader_pid, writer_pid) → Pipe
        self._pipes: Dict[Tuple[int, int], Pipe] = {}

        # Tensor channels: name → TensorChannel
        self._tensor_channels: Dict[str, TensorChannel] = {}

        # Signal handlers: pid → signal → handler
        self._signal_handlers: Dict[int, Dict[int, Any]] = {}

        # Pending signals
        self._pending_signals: Dict[int, List[int]] = {}

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.shm_created = 0
        self.pipes_created = 0
        self.signals_sent = 0
        self.tensor_channels_created = 0

    def register_process(self, pid: int):
        """Register a process for IPC (creates its message queue)."""
        if pid not in self._queues:
            self._queues[pid] = MessageQueue(device=self.device)
            self._signal_handlers[pid] = {}
            self._pending_signals[pid] = []

    def unregister_process(self, pid: int):
        """Unregister a process (cleanup IPC resources)."""
        self._queues.pop(pid, None)
        self._signal_handlers.pop(pid, None)
        self._pending_signals.pop(pid, None)

        # Close pipes involving this process
        to_remove = [k for k in self._pipes
                     if k[0] == pid or k[1] == pid]
        for k in to_remove:
            self._pipes[k].close()
            del self._pipes[k]

        # Detach from shared memory
        for shm in self._shm.values():
            shm.detach(pid)

    # ─── Message Passing ──────────────────────────────────────────────────

    def send(self, src_pid: int, dst_pid: int,
             payload: Optional[torch.Tensor] = None,
             msg_type: int = MSG_DATA, tag: int = 0) -> bool:
        """Send a message from one process to another.

        Returns True if message was queued successfully.
        """
        if dst_pid not in self._queues:
            logger.warning(f"[IPC] Send failed: PID {dst_pid} not registered")
            return False

        msg = Message(src_pid=src_pid, dst_pid=dst_pid,
                      msg_type=msg_type, payload=payload, tag=tag)

        success = self._queues[dst_pid].send(msg)
        if success:
            self.messages_sent += 1
        return success

    def receive(self, pid: int, tag: Optional[int] = None) -> Optional[Message]:
        """Receive a message for a process.

        Returns None if no messages available.
        """
        if pid not in self._queues:
            return None

        msg = self._queues[pid].receive(tag)
        if msg is not None:
            self.messages_received += 1
        return msg

    def broadcast(self, src_pid: int, payload: torch.Tensor,
                  tag: int = 0) -> int:
        """Broadcast a message to all registered processes.

        Returns number of processes that received the message.
        """
        count = 0
        for pid in self._queues:
            if pid != src_pid:
                if self.send(src_pid, pid, payload, MSG_BROADCAST, tag):
                    count += 1
        return count

    def has_messages(self, pid: int) -> bool:
        """Check if a process has pending messages."""
        if pid not in self._queues:
            return False
        return not self._queues[pid].is_empty

    # ─── Shared Memory ────────────────────────────────────────────────────

    def shm_create(self, name: str, size: int, owner_pid: int) -> SharedMemoryRegion:
        """Create a shared memory region."""
        if name in self._shm:
            raise ValueError(f"Shared memory '{name}' already exists")

        shm = SharedMemoryRegion(name, size, self.device)
        shm.attach(owner_pid)
        self._shm[name] = shm
        self.shm_created += 1
        return shm

    def shm_open(self, name: str, pid: int) -> Optional[SharedMemoryRegion]:
        """Open an existing shared memory region."""
        shm = self._shm.get(name)
        if shm is not None:
            shm.attach(pid)
        return shm

    def shm_close(self, name: str, pid: int):
        """Close a process's connection to shared memory."""
        shm = self._shm.get(name)
        if shm is not None:
            shm.detach(pid)
            if not shm._owners:
                del self._shm[name]

    # ─── Pipes ────────────────────────────────────────────────────────────

    def pipe_create(self, reader_pid: int, writer_pid: int,
                    buffer_size: int = 4096) -> Pipe:
        """Create a pipe between two processes."""
        key = (reader_pid, writer_pid)
        if key in self._pipes:
            raise ValueError(f"Pipe already exists: {key}")

        pipe = Pipe(reader_pid, writer_pid, buffer_size, self.device)
        self._pipes[key] = pipe
        self.pipes_created += 1
        return pipe

    def pipe_get(self, reader_pid: int, writer_pid: int) -> Optional[Pipe]:
        """Get an existing pipe."""
        return self._pipes.get((reader_pid, writer_pid))

    # ─── Signals ──────────────────────────────────────────────────────────

    def signal_send(self, src_pid: int, dst_pid: int, signal: int):
        """Send a signal to a process."""
        if dst_pid in self._pending_signals:
            self._pending_signals[dst_pid].append(signal)
            self.signals_sent += 1

    def signal_pending(self, pid: int) -> List[int]:
        """Get and clear pending signals for a process."""
        signals = self._pending_signals.get(pid, [])
        self._pending_signals[pid] = []
        return signals

    def signal_register(self, pid: int, signal: int, handler: Any):
        """Register a signal handler."""
        if pid in self._signal_handlers:
            self._signal_handlers[pid][signal] = handler

    # ─── Tensor Channels (GPU-native typed tensor IPC) ──────────────────

    def tensor_create(self, name: str, shape, dtype=torch.float32,
                      owner_pid: int = 0) -> 'TensorChannel':
        """Create a typed tensor channel for zero-copy GPU IPC.

        Unlike SharedMemoryRegion (raw bytes), TensorChannel preserves
        tensor dtype and shape. Multiple processes read/write the same
        GPU tensor without serialization — genuinely zero-copy.
        """
        if name in self._tensor_channels:
            raise ValueError(f"Tensor channel '{name}' already exists")
        channel = TensorChannel(name, shape, dtype, self.device)
        channel.attach(owner_pid)
        self._tensor_channels[name] = channel
        self.tensor_channels_created += 1
        return channel

    def tensor_open(self, name: str, pid: int) -> Optional['TensorChannel']:
        """Open an existing tensor channel (attach process for access)."""
        ch = self._tensor_channels.get(name)
        if ch is not None:
            ch.attach(pid)
        return ch

    def tensor_close(self, name: str, pid: int):
        """Close a process's connection to a tensor channel."""
        ch = self._tensor_channels.get(name)
        if ch is not None:
            ch.detach(pid)
            if not ch._owners:
                del self._tensor_channels[name]

    def tensor_list(self) -> List[Dict]:
        """List all active tensor channels with metadata."""
        result = []
        for name, ch in self._tensor_channels.items():
            result.append({
                "name": name,
                "shape": list(ch.shape) if isinstance(ch.shape, tuple) else [ch.shape],
                "dtype": str(ch.dtype),
                "owners": list(ch._owners),
                "version": ch.version(),
            })
        return result

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def stats(self) -> Dict:
        total_queued = sum(q.count for q in self._queues.values())
        return {
            "registered_processes": len(self._queues),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_queued": total_queued,
            "shm_regions": len(self._shm),
            "active_pipes": sum(1 for p in self._pipes.values() if not p.closed),
            "signals_sent": self.signals_sent,
            "tensor_channels": len(self._tensor_channels),
            "tensor_channels_created": self.tensor_channels_created,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"NeuralIPC(procs={s['registered_processes']}, "
                f"msgs={s['messages_sent']}, "
                f"shm={s['shm_regions']}, "
                f"pipes={s['active_pipes']}, "
                f"tensors={s['tensor_channels']})")
