"""GPU-Native Memory Protection Unit.

Per-process bounds checking, permission enforcement, guard pages, and stack
canary validation -- all backed by GPU-resident tensors. The MPU runs access
checks as vectorized tensor operations so that every region for a process is
tested in a single parallel pass (no per-region Python loop for bounds).

Architecture:
    Region tables:  [max_processes, max_regions] GPU tensors for start/end/perms
    Guard pages:    boolean mask identifying guard regions (zero-permission)
    Stack canaries: per-process random int64 values generated on GPU

Protection model:
    - R=1, W=2, X=4 bitmask per region (matches ARM64 EL0 conventions)
    - Standard process layout: text (R+X), data (R+W), heap (R+W), stack (R+W)
    - Guard page below stack base to catch stack overflow
    - Random stack canary for corruption detection

Key design decisions:
    - All region metadata lives on GPU -- zero CPU-GPU sync for access checks
    - Guard page detection is a first-class concern, not an afterthought
    - Violation history is capped (CPU-side list) to avoid unbounded growth
    - Region names are CPU-side metadata only; hot path uses tensor ops
"""

import time
import torch
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

from .device import default_device

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

PAGE_SIZE = 4096          # 4KB pages, matching ARM64 convention

# Permission bitmask values
PERM_READ = 1             # R
PERM_WRITE = 2            # W
PERM_EXEC = 4             # X

# Common permission combinations
PERM_RX = PERM_READ | PERM_EXEC     # 5 — text segments
PERM_RW = PERM_READ | PERM_WRITE    # 3 — data/heap/stack
PERM_NONE = 0                        # guard pages

# Maximum violation history kept in memory
MAX_VIOLATION_HISTORY = 1024


# ═══════════════════════════════════════════════════════════════════════════════
# Access Violation Record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AccessViolation:
    """Record of a denied memory access.

    Attributes:
        pid: Process that attempted the access.
        address: Virtual address that was accessed.
        access_type: Kind of access -- "read", "write", or "execute".
        reason: Why the access was denied:
            "bounds"         -- address not in any active region
            "permission"     -- region exists but permission bit not set
            "guard_page"     -- address hit a guard page
            "stack_overflow" -- address hit the guard page below a stack
        timestamp: Wall-clock time of the violation (time.time()).
    """
    pid: int
    address: int
    access_type: str
    reason: str
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════════════
# Process Memory Region Descriptor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessMemoryRegion:
    """A contiguous memory region owned by a process.

    Attributes:
        start: Base address of the region.
        size: Size in bytes.
        permissions: Bitmask -- R=1, W=2, X=4.
        name: Human-readable label ("text", "data", "heap", "stack", "guard").
    """
    start: int
    size: int
    permissions: int
    name: str


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Protection Unit
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryProtectionUnit:
    """GPU-resident per-process bounds checking and permission enforcement.

    All region metadata is stored as GPU tensors so that access checks are
    vectorized across every region of a process in a single pass.  Guard pages,
    stack canaries, and violation tracking are integrated into the hot path.

    Typical lifecycle::

        mpu = MemoryProtectionUnit()
        mpu.setup_process(pid=1,
                          text_start=0x1000, text_size=0x2000,
                          data_start=0x4000, data_size=0x1000,
                          heap_start=0x8000, heap_size=0x4000,
                          stack_start=0xF000, stack_size=0x2000)

        violation = mpu.check_access(pid=1, address=0x1000, access_type="read")
        assert violation is None  # allowed

        violation = mpu.check_access(pid=1, address=0x1000, access_type="write")
        assert violation is not None  # text is R+X, not W

        mpu.teardown_process(pid=1)
    """

    def __init__(self, max_processes: int = 256,
                 max_regions_per_process: int = 16,
                 device: Optional[torch.device] = None):
        """Initialize the Memory Protection Unit.

        Args:
            max_processes: Maximum number of concurrent processes.
            max_regions_per_process: Maximum memory regions per process.
            device: PyTorch device.  Defaults to neurOS default (MPS > CUDA > CPU).
        """
        self.device = device or default_device()
        self.max_processes = max_processes
        self.max_regions = max_regions_per_process

        # ── Per-process region tables (GPU tensors) ──────────────────────
        self.regions_start = torch.zeros(
            max_processes, max_regions_per_process,
            dtype=torch.int64, device=self.device,
        )
        self.regions_end = torch.zeros(
            max_processes, max_regions_per_process,
            dtype=torch.int64, device=self.device,
        )
        self.regions_perms = torch.zeros(
            max_processes, max_regions_per_process,
            dtype=torch.int32, device=self.device,
        )
        self.regions_valid = torch.zeros(
            max_processes, max_regions_per_process,
            dtype=torch.bool, device=self.device,
        )

        # Region names -- CPU-side metadata only, not in the hot path
        self.regions_names: Dict[tuple, str] = {}

        # ── Guard page tracking ──────────────────────────────────────────
        self.guard_pages = torch.zeros(
            max_processes, max_regions_per_process,
            dtype=torch.bool, device=self.device,
        )

        # ── Stack canary values (random int64, generated on GPU) ─────────
        self.stack_canaries = torch.zeros(
            max_processes, dtype=torch.int64, device=self.device,
        )
        self.canary_initialized = torch.zeros(
            max_processes, dtype=torch.bool, device=self.device,
        )

        # ── Statistics ───────────────────────────────────────────────────
        self.total_checks: int = 0
        self.total_violations: int = 0
        self.violations: List[AccessViolation] = []

        logger.info(
            f"[MPU] Initialized: {max_processes} processes x "
            f"{max_regions_per_process} regions on {self.device}"
        )

    # ─── Process Setup / Teardown ──────────────────────────────────────────

    def setup_process(self, pid: int, text_start: int, text_size: int,
                      data_start: int, data_size: int,
                      heap_start: int, heap_size: int,
                      stack_start: int, stack_size: int) -> None:
        """Initialize memory regions for a process with standard layout.

        Creates five regions:
            0: text  (R+X)  -- executable code
            1: data  (R+W)  -- initialized/uninitialized data
            2: heap  (R+W)  -- dynamic allocations
            3: stack (R+W)  -- call stack
            4: guard (none) -- one page below stack to detect overflow

        Also generates a random 64-bit stack canary on GPU.

        Args:
            pid: Process ID (must be < max_processes).
            text_start: Base address of text segment.
            text_size: Size of text segment in bytes.
            data_start: Base address of data segment.
            data_size: Size of data segment in bytes.
            heap_start: Base address of heap.
            heap_size: Initial heap size in bytes.
            stack_start: Base address of stack.
            stack_size: Stack size in bytes.
        """
        self._add_region(pid, 0, text_start, text_size, PERM_RX, "text")
        self._add_region(pid, 1, data_start, data_size, PERM_RW, "data")
        self._add_region(pid, 2, heap_start, heap_size, PERM_RW, "heap")
        self._add_region(pid, 3, stack_start, stack_size, PERM_RW, "stack")
        # Guard page: one page below stack base with zero permissions
        self._add_region(pid, 4, stack_start - PAGE_SIZE, PAGE_SIZE, PERM_NONE, "guard")
        self.guard_pages[pid, 4] = True

        # Random canary -- combine two 32-bit halves for full 64-bit range.
        # torch.randint upper bound must fit in int64, so we avoid 2**63.
        hi = torch.randint(0, 2**31, (1,), dtype=torch.int64, device=self.device)
        lo = torch.randint(0, 2**31, (1,), dtype=torch.int64, device=self.device)
        self.stack_canaries[pid] = (hi << 32) | lo
        self.canary_initialized[pid] = True

        logger.debug(
            f"[MPU] Process {pid}: text={text_start:#x}+{text_size:#x} "
            f"data={data_start:#x}+{data_size:#x} "
            f"heap={heap_start:#x}+{heap_size:#x} "
            f"stack={stack_start:#x}+{stack_size:#x}"
        )

    def teardown_process(self, pid: int) -> None:
        """Remove all memory regions for a terminated process.

        Clears the region table row, guard page flags, and canary state.

        Args:
            pid: Process ID to tear down.
        """
        self.regions_valid[pid] = False
        self.guard_pages[pid] = False
        self.canary_initialized[pid] = False
        # Clean up CPU-side name metadata
        for idx in range(self.max_regions):
            self.regions_names.pop((pid, idx), None)
        logger.debug(f"[MPU] Process {pid}: torn down")

    def _add_region(self, pid: int, idx: int, start: int, size: int,
                    perms: int, name: str) -> None:
        """Write a single region entry into the GPU tensor tables.

        Args:
            pid: Process ID.
            idx: Region slot index within the process.
            start: Region start address.
            size: Region size in bytes.
            perms: Permission bitmask (R=1, W=2, X=4).
            name: Human-readable region label.
        """
        self.regions_start[pid, idx] = start
        self.regions_end[pid, idx] = start + size
        self.regions_perms[pid, idx] = perms
        self.regions_valid[pid, idx] = True
        self.regions_names[(pid, idx)] = name

    # ─── Access Checking ───────────────────────────────────────────────────

    def check_access(self, pid: int, address: int,
                     access_type: str) -> Optional[AccessViolation]:
        """Check whether a memory access is permitted.

        Uses GPU tensor operations to test all regions in parallel:
            1. Vectorized bounds check (address in [start, end) for all regions)
            2. Guard page detection
            3. Permission bitmask test

        Args:
            pid: Process performing the access.
            address: Virtual address being accessed.
            access_type: One of "read", "write", "execute".

        Returns:
            None if the access is allowed; an AccessViolation if denied.
        """
        self.total_checks += 1

        # Vectorized bounds check across all regions for this process
        addr_t = torch.tensor(address, dtype=torch.int64, device=self.device)
        in_bounds = (
            (addr_t >= self.regions_start[pid])
            & (addr_t < self.regions_end[pid])
            & self.regions_valid[pid]
        )

        # No region contains this address
        if not in_bounds.any():
            return self._record_violation(pid, address, access_type, "bounds")

        # Check guard page hits first (they have zero permissions anyway,
        # but we want a more specific violation reason)
        matching_indices = in_bounds.nonzero(as_tuple=True)[0]
        for idx in matching_indices:
            if self.guard_pages[pid, idx]:
                region_name = self.regions_names.get((pid, idx.item()), "")
                reason = "stack_overflow" if region_name == "guard" else "guard_page"
                return self._record_violation(pid, address, access_type, reason)

        # Permission check on the first matching region
        perm_bit = _ACCESS_TYPE_TO_BIT.get(access_type, 0)
        region_idx = int(matching_indices[0].item())
        perms = self.regions_perms[pid, region_idx].item()

        if not (perms & perm_bit):
            return self._record_violation(pid, address, access_type, "permission")

        return None  # Access allowed

    def _record_violation(self, pid: int, address: int,
                          access_type: str, reason: str) -> AccessViolation:
        """Create a violation record, store it, and return it."""
        violation = AccessViolation(
            pid=pid,
            address=address,
            access_type=access_type,
            reason=reason,
            timestamp=time.time(),
        )
        self.total_violations += 1
        self.violations.append(violation)
        # Cap history to prevent unbounded growth
        if len(self.violations) > MAX_VIOLATION_HISTORY:
            self.violations = self.violations[-MAX_VIOLATION_HISTORY:]
        logger.debug(
            f"[MPU] Violation: pid={pid} addr={address:#x} "
            f"{access_type} -> {reason}"
        )
        return violation

    # ─── Stack Canary ──────────────────────────────────────────────────────

    def get_canary(self, pid: int) -> int:
        """Get the stack canary value for a process.

        Args:
            pid: Process ID.

        Returns:
            The 64-bit canary value.
        """
        return int(self.stack_canaries[pid].item())

    def check_canary(self, pid: int, current_value: int) -> bool:
        """Check whether the stack canary is intact.

        Args:
            pid: Process ID.
            current_value: The value read from the canary location.

        Returns:
            True if the canary matches (stack is intact), False if corrupted.
        """
        if not self.canary_initialized[pid]:
            return True
        return current_value == self.stack_canaries[pid].item()

    # ─── Heap Management ───────────────────────────────────────────────────

    def grow_heap(self, pid: int, additional_bytes: int) -> bool:
        """Grow the heap region for a process (analogous to brk/sbrk).

        Extends the heap's end address by the given amount.  The heap region
        is at slot index 2 by convention from setup_process().

        Args:
            pid: Process ID.
            additional_bytes: Number of bytes to add to the heap.

        Returns:
            True if the heap was grown successfully, False if the heap region
            is not active.
        """
        heap_idx = 2  # Convention from setup_process()
        if not self.regions_valid[pid, heap_idx]:
            return False
        self.regions_end[pid, heap_idx] += additional_bytes
        logger.debug(
            f"[MPU] Process {pid}: heap grown by {additional_bytes} bytes "
            f"-> end={self.regions_end[pid, heap_idx].item():#x}"
        )
        return True

    # ─── Region Queries ────────────────────────────────────────────────────

    def add_region(self, pid: int, start: int, size: int,
                   perms: int, name: str) -> bool:
        """Add a new memory region to a process.

        Finds the first unused region slot and writes the new region into it.

        Args:
            pid: Process ID.
            start: Region start address.
            size: Region size in bytes.
            perms: Permission bitmask (R=1, W=2, X=4).
            name: Human-readable region label.

        Returns:
            True if the region was added, False if no free slots remain.
        """
        for idx in range(self.max_regions):
            if not self.regions_valid[pid, idx]:
                self._add_region(pid, idx, start, size, perms, name)
                return True
        logger.warning(f"[MPU] Process {pid}: no free region slots")
        return False

    def list_regions(self, pid: int) -> List[ProcessMemoryRegion]:
        """List all active memory regions for a process.

        Args:
            pid: Process ID.

        Returns:
            List of ProcessMemoryRegion descriptors for each active region.
        """
        regions = []
        for idx in range(self.max_regions):
            if self.regions_valid[pid, idx]:
                regions.append(ProcessMemoryRegion(
                    start=int(self.regions_start[pid, idx].item()),
                    size=int(
                        (self.regions_end[pid, idx]
                         - self.regions_start[pid, idx]).item()
                    ),
                    permissions=int(self.regions_perms[pid, idx].item()),
                    name=self.regions_names.get((pid, idx), f"region_{idx}"),
                ))
        return regions

    # ─── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return MPU statistics.

        Returns:
            Dict with active_processes, total_regions, total_checks,
            total_violations, and recent_violations count.
        """
        active_processes = int(self.regions_valid.any(dim=1).sum().item())
        total_regions = int(self.regions_valid.sum().item())
        return {
            "active_processes": active_processes,
            "total_regions": total_regions,
            "total_checks": self.total_checks,
            "total_violations": self.total_violations,
            "recent_violations": len(self.violations[-5:]),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoryProtectionUnit(procs={s['active_processes']}, "
            f"regions={s['total_regions']}, "
            f"checks={s['total_checks']}, "
            f"violations={s['total_violations']})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Module-Level Lookup Table
# ═══════════════════════════════════════════════════════════════════════════════

_ACCESS_TYPE_TO_BIT: Dict[str, int] = {
    "read": PERM_READ,
    "write": PERM_WRITE,
    "execute": PERM_EXEC,
}
