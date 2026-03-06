"""GPU-Native Concurrency Primitives for neurOS.

All synchronization state lives entirely on GPU as tensors. No CPU-GPU
transfers for lock operations -- compare-and-swap, counting, and barrier
coordination are all tensor operations on the compute device.

Primitives:
    1. TensorMutex     -- GPU-resident mutual exclusion via tensor CAS
    2. TensorSemaphore -- GPU-resident counting semaphore
    3. TensorBarrier   -- GPU-resident N-process barrier synchronization
    4. TensorRWLock    -- GPU-resident reader-writer lock
    5. SyncManager     -- Central registry for named synchronization objects
"""

import torch
import logging
from typing import Optional, Dict, List

from .device import default_device

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TensorMutex -- GPU-resident mutual exclusion
# ═══════════════════════════════════════════════════════════════════════════════

class TensorMutex:
    """GPU-resident mutex using tensor-based compare-and-swap.

    The lock state is a single int32 tensor on GPU. Acquiring the lock
    checks the tensor value and atomically sets it -- all on-device.
    Ownership is tracked by a second tensor holding the PID of the lock
    holder.

    In a real GPU kernel this would use atomicCAS. Here we emulate the
    semantics with tensor reads and writes, which are correct under
    neurOS's cooperative (non-preemptive) scheduling model.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or default_device()
        self._locked = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._owner = torch.full((1,), -1, dtype=torch.int32, device=self.device)
        self.acquisitions = 0
        self.contentions = 0

    def acquire(self, pid: int) -> bool:
        """Try to acquire the mutex.

        Args:
            pid: Process ID attempting acquisition.

        Returns:
            True if the lock was acquired, False if already held by
            another process.
        """
        if self._locked.item() == 0:
            self._locked.fill_(1)
            self._owner.fill_(pid)
            self.acquisitions += 1
            return True
        # Already locked -- record contention
        self.contentions += 1
        return False

    def release(self, pid: int) -> bool:
        """Release the mutex.

        Only the owning process can release. Attempting to release a
        lock not owned by *pid* is a no-op that returns False.

        Args:
            pid: Process ID attempting release.

        Returns:
            True if the lock was released, False if *pid* is not the owner.
        """
        if self._owner.item() == pid:
            self._locked.fill_(0)
            self._owner.fill_(-1)
            return True
        return False

    def is_locked(self) -> bool:
        """Return True if the mutex is currently held."""
        return self._locked.item() == 1

    def owner(self) -> int:
        """Return the PID of the lock holder, or -1 if unlocked."""
        return self._owner.item()

    def stats(self) -> Dict:
        """Return mutex statistics."""
        return {
            "locked": self.is_locked(),
            "owner": self.owner(),
            "acquisitions": self.acquisitions,
            "contentions": self.contentions,
            "contention_rate": (
                self.contentions / (self.acquisitions + self.contentions)
                if (self.acquisitions + self.contentions) > 0
                else 0.0
            ),
        }

    def __repr__(self) -> str:
        state = "locked" if self.is_locked() else "unlocked"
        return (f"TensorMutex({state}, owner={self.owner()}, "
                f"acq={self.acquisitions}, cont={self.contentions})")


# ═══════════════════════════════════════════════════════════════════════════════
# TensorSemaphore -- GPU-resident counting semaphore
# ═══════════════════════════════════════════════════════════════════════════════

class TensorSemaphore:
    """GPU-resident counting semaphore.

    The count is a single int32 tensor on GPU. Acquire decrements the
    count if positive; release increments it or wakes a waiting process.

    Processes that fail to acquire are placed in a FIFO wait queue.
    When a release occurs and there are waiters, the first waiter is
    woken (returned from ``release``) rather than incrementing the count.
    """

    def __init__(self, initial_count: int = 1,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self._count = torch.tensor(
            [initial_count], dtype=torch.int32, device=self.device
        )
        self._initial = initial_count
        self._waiters: List[int] = []
        self.total_waits = 0
        self.total_acquires = 0
        self.total_releases = 0

    def acquire(self, pid: int) -> bool:
        """Attempt to decrement the semaphore count.

        Args:
            pid: Process ID attempting acquisition.

        Returns:
            True if the count was positive and decremented (acquired).
            False if the count was zero -- *pid* is added to the wait queue.
        """
        if self._count.item() > 0:
            self._count -= 1
            self.total_acquires += 1
            return True
        self._waiters.append(pid)
        self.total_waits += 1
        return False

    def release(self) -> Optional[int]:
        """Increment the semaphore count or wake a waiter.

        If processes are waiting, the first waiter is removed from the
        queue and its PID is returned (the count stays unchanged -- the
        permit goes directly to the woken process).

        Returns:
            PID of the woken waiter, or None if no one was waiting
            (in which case the count is incremented).
        """
        self.total_releases += 1
        if self._waiters:
            woken = self._waiters.pop(0)
            self.total_acquires += 1
            return woken
        self._count += 1
        return None

    def count(self) -> int:
        """Return the current semaphore count."""
        return self._count.item()

    @property
    def waiters(self) -> List[int]:
        """Return the list of waiting PIDs (read-only copy)."""
        return list(self._waiters)

    def stats(self) -> Dict:
        """Return semaphore statistics."""
        return {
            "count": self.count(),
            "initial": self._initial,
            "waiters": len(self._waiters),
            "total_acquires": self.total_acquires,
            "total_releases": self.total_releases,
            "total_waits": self.total_waits,
        }

    def __repr__(self) -> str:
        return (f"TensorSemaphore(count={self.count()}/{self._initial}, "
                f"waiters={len(self._waiters)}, acq={self.total_acquires})")


# ═══════════════════════════════════════════════════════════════════════════════
# TensorBarrier -- GPU-resident N-process barrier
# ═══════════════════════════════════════════════════════════════════════════════

class TensorBarrier:
    """GPU-resident N-process barrier synchronization.

    All *num_parties* processes must call ``arrive`` before the barrier
    breaks. When the last process arrives, the barrier resets its count,
    increments the generation counter, and returns True to signal
    completion.

    The arrival count and generation are GPU-resident int tensors.
    """

    def __init__(self, num_parties: int,
                 device: Optional[torch.device] = None):
        if num_parties < 1:
            raise ValueError(
                f"num_parties must be >= 1, got {num_parties}"
            )
        self.device = device or default_device()
        self._count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._generation = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.num_parties = num_parties
        self._arrived_pids: List[int] = []
        self.total_completions = 0

    def arrive(self, pid: int) -> bool:
        """Record a process arriving at the barrier.

        Args:
            pid: Process ID that has reached the barrier.

        Returns:
            True if this arrival breaks the barrier (all parties have
            arrived). False if still waiting for more processes.
        """
        self._count += 1
        self._arrived_pids.append(pid)

        if self._count.item() >= self.num_parties:
            # Barrier broken -- reset for next generation
            self._count.fill_(0)
            self._generation += 1
            self._arrived_pids.clear()
            self.total_completions += 1
            return True
        return False

    def generation(self) -> int:
        """Return the current generation (incremented each time the barrier breaks)."""
        return self._generation.item()

    def waiting(self) -> int:
        """Return the number of processes that have arrived but not yet released."""
        return self._count.item()

    @property
    def arrived_pids(self) -> List[int]:
        """Return the list of PIDs that have arrived in this generation (read-only copy)."""
        return list(self._arrived_pids)

    def stats(self) -> Dict:
        """Return barrier statistics."""
        return {
            "num_parties": self.num_parties,
            "waiting": self.waiting(),
            "generation": self.generation(),
            "total_completions": self.total_completions,
        }

    def __repr__(self) -> str:
        return (f"TensorBarrier(parties={self.num_parties}, "
                f"waiting={self.waiting()}/{self.num_parties}, "
                f"gen={self.generation()}, done={self.total_completions})")


# ═══════════════════════════════════════════════════════════════════════════════
# TensorRWLock -- GPU-resident reader-writer lock
# ═══════════════════════════════════════════════════════════════════════════════

class TensorRWLock:
    """GPU-resident reader-writer lock.

    Multiple readers can hold the lock concurrently, but a writer
    requires exclusive access. The reader count and writer identity
    are GPU-resident int32 tensors.

    Semantics:
        - ``acquire_read`` succeeds if no writer is active.
        - ``acquire_write`` succeeds if no readers and no writer are active.
        - Readers and writers are mutually exclusive.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or default_device()
        self._readers = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._writer = torch.full(
            (1,), -1, dtype=torch.int32, device=self.device
        )
        self._reader_pids: List[int] = []
        self.read_acquires = 0
        self.write_acquires = 0
        self.read_contentions = 0
        self.write_contentions = 0

    def acquire_read(self, pid: int) -> bool:
        """Acquire a read lock.

        Succeeds if no writer is currently active. Multiple readers
        may hold the lock simultaneously.

        Args:
            pid: Process ID requesting read access.

        Returns:
            True if the read lock was acquired, False if a writer is active.
        """
        if self._writer.item() == -1:
            self._readers += 1
            self._reader_pids.append(pid)
            self.read_acquires += 1
            return True
        self.read_contentions += 1
        return False

    def release_read(self, pid: int) -> bool:
        """Release a read lock.

        Args:
            pid: Process ID releasing read access.

        Returns:
            True if the read lock was released, False if *pid* did not
            hold a read lock.
        """
        if pid in self._reader_pids:
            self._reader_pids.remove(pid)
            self._readers -= 1
            return True
        return False

    def acquire_write(self, pid: int) -> bool:
        """Acquire an exclusive write lock.

        Succeeds only if there are no active readers and no active writer.

        Args:
            pid: Process ID requesting write access.

        Returns:
            True if the write lock was acquired, False otherwise.
        """
        if self._writer.item() == -1 and self._readers.item() == 0:
            self._writer.fill_(pid)
            self.write_acquires += 1
            return True
        self.write_contentions += 1
        return False

    def release_write(self, pid: int) -> bool:
        """Release the write lock.

        Only the owning writer can release.

        Args:
            pid: Process ID releasing write access.

        Returns:
            True if the write lock was released, False if *pid* is not
            the current writer.
        """
        if self._writer.item() == pid:
            self._writer.fill_(-1)
            return True
        return False

    def active_readers(self) -> int:
        """Return the number of active readers."""
        return self._readers.item()

    def active_writer(self) -> int:
        """Return the PID of the active writer, or -1 if none."""
        return self._writer.item()

    def stats(self) -> Dict:
        """Return reader-writer lock statistics."""
        return {
            "active_readers": self.active_readers(),
            "active_writer": self.active_writer(),
            "reader_pids": list(self._reader_pids),
            "read_acquires": self.read_acquires,
            "write_acquires": self.write_acquires,
            "read_contentions": self.read_contentions,
            "write_contentions": self.write_contentions,
        }

    def __repr__(self) -> str:
        writer = self.active_writer()
        if writer != -1:
            return (f"TensorRWLock(writer={writer}, "
                    f"r_acq={self.read_acquires}, w_acq={self.write_acquires})")
        readers = self.active_readers()
        return (f"TensorRWLock(readers={readers}, "
                f"r_acq={self.read_acquires}, w_acq={self.write_acquires})")


# ═══════════════════════════════════════════════════════════════════════════════
# SyncManager -- Central registry for named synchronization objects
# ═══════════════════════════════════════════════════════════════════════════════

class SyncManager:
    """Central registry managing named synchronization objects.

    Provides create/get/destroy operations for all four primitive types.
    Names are unique across the entire manager -- a mutex and a semaphore
    cannot share the same name.

    All primitives are allocated on the same GPU device.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or default_device()
        self._mutexes: Dict[str, TensorMutex] = {}
        self._semaphores: Dict[str, TensorSemaphore] = {}
        self._barriers: Dict[str, TensorBarrier] = {}
        self._rwlocks: Dict[str, TensorRWLock] = {}

    # ─── Mutex operations ──────────────────────────────────────────────────

    def create_mutex(self, name: str) -> TensorMutex:
        """Create a named mutex.

        Args:
            name: Unique name for the mutex.

        Returns:
            The newly created TensorMutex.

        Raises:
            ValueError: If a synchronization object with *name* already exists.
        """
        self._check_name_available(name)
        mutex = TensorMutex(device=self.device)
        self._mutexes[name] = mutex
        logger.debug(f"[SyncManager] Created mutex: {name}")
        return mutex

    def get_mutex(self, name: str) -> Optional[TensorMutex]:
        """Look up a mutex by name. Returns None if not found."""
        return self._mutexes.get(name)

    # ─── Semaphore operations ──────────────────────────────────────────────

    def create_semaphore(self, name: str,
                         count: int = 1) -> TensorSemaphore:
        """Create a named counting semaphore.

        Args:
            name: Unique name for the semaphore.
            count: Initial permit count (default 1 for binary semaphore).

        Returns:
            The newly created TensorSemaphore.

        Raises:
            ValueError: If a synchronization object with *name* already exists.
        """
        self._check_name_available(name)
        sem = TensorSemaphore(initial_count=count, device=self.device)
        self._semaphores[name] = sem
        logger.debug(f"[SyncManager] Created semaphore: {name} (count={count})")
        return sem

    def get_semaphore(self, name: str) -> Optional[TensorSemaphore]:
        """Look up a semaphore by name. Returns None if not found."""
        return self._semaphores.get(name)

    # ─── Barrier operations ────────────────────────────────────────────────

    def create_barrier(self, name: str,
                       parties: int) -> TensorBarrier:
        """Create a named N-process barrier.

        Args:
            name: Unique name for the barrier.
            parties: Number of processes that must arrive before the
                barrier breaks.

        Returns:
            The newly created TensorBarrier.

        Raises:
            ValueError: If a synchronization object with *name* already exists.
        """
        self._check_name_available(name)
        barrier = TensorBarrier(num_parties=parties, device=self.device)
        self._barriers[name] = barrier
        logger.debug(f"[SyncManager] Created barrier: {name} (parties={parties})")
        return barrier

    def get_barrier(self, name: str) -> Optional[TensorBarrier]:
        """Look up a barrier by name. Returns None if not found."""
        return self._barriers.get(name)

    # ─── RWLock operations ─────────────────────────────────────────────────

    def create_rwlock(self, name: str) -> TensorRWLock:
        """Create a named reader-writer lock.

        Args:
            name: Unique name for the RWLock.

        Returns:
            The newly created TensorRWLock.

        Raises:
            ValueError: If a synchronization object with *name* already exists.
        """
        self._check_name_available(name)
        rwlock = TensorRWLock(device=self.device)
        self._rwlocks[name] = rwlock
        logger.debug(f"[SyncManager] Created rwlock: {name}")
        return rwlock

    def get_rwlock(self, name: str) -> Optional[TensorRWLock]:
        """Look up a reader-writer lock by name. Returns None if not found."""
        return self._rwlocks.get(name)

    # ─── Destroy ───────────────────────────────────────────────────────────

    def destroy(self, name: str) -> bool:
        """Remove a synchronization object by name from any registry.

        Args:
            name: Name of the object to destroy.

        Returns:
            True if an object was found and removed, False if the name
            was not registered.
        """
        for registry in (self._mutexes, self._semaphores,
                         self._barriers, self._rwlocks):
            if name in registry:
                del registry[name]
                logger.debug(f"[SyncManager] Destroyed: {name}")
                return True
        return False

    # ─── Listing ───────────────────────────────────────────────────────────

    def list_all(self) -> Dict[str, str]:
        """Return a mapping of name to type for every registered object."""
        result: Dict[str, str] = {}
        for name in self._mutexes:
            result[name] = "mutex"
        for name in self._semaphores:
            result[name] = "semaphore"
        for name in self._barriers:
            result[name] = "barrier"
        for name in self._rwlocks:
            result[name] = "rwlock"
        return result

    # ─── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return aggregate statistics across all synchronization objects."""
        total_mutex_acq = sum(m.acquisitions for m in self._mutexes.values())
        total_mutex_cont = sum(m.contentions for m in self._mutexes.values())
        total_sem_acq = sum(s.total_acquires for s in self._semaphores.values())
        total_sem_waits = sum(s.total_waits for s in self._semaphores.values())
        total_barrier_comp = sum(
            b.total_completions for b in self._barriers.values()
        )
        total_read_acq = sum(r.read_acquires for r in self._rwlocks.values())
        total_write_acq = sum(r.write_acquires for r in self._rwlocks.values())

        return {
            "mutexes": len(self._mutexes),
            "semaphores": len(self._semaphores),
            "barriers": len(self._barriers),
            "rwlocks": len(self._rwlocks),
            "total_objects": (
                len(self._mutexes) + len(self._semaphores)
                + len(self._barriers) + len(self._rwlocks)
            ),
            "mutex_acquisitions": total_mutex_acq,
            "mutex_contentions": total_mutex_cont,
            "semaphore_acquires": total_sem_acq,
            "semaphore_waits": total_sem_waits,
            "barrier_completions": total_barrier_comp,
            "rwlock_read_acquires": total_read_acq,
            "rwlock_write_acquires": total_write_acq,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"SyncManager(mutexes={s['mutexes']}, "
                f"sems={s['semaphores']}, "
                f"barriers={s['barriers']}, "
                f"rwlocks={s['rwlocks']})")

    # ─── Internal ──────────────────────────────────────────────────────────

    def _check_name_available(self, name: str) -> None:
        """Raise ValueError if *name* is already used by any primitive."""
        all_names = set(self._mutexes) | set(self._semaphores) | \
            set(self._barriers) | set(self._rwlocks)
        if name in all_names:
            raise ValueError(
                f"Synchronization object '{name}' already exists"
            )
