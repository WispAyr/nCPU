"""neurOS: GPU-Native Neural Operating System.

Every component -- MMU, scheduler, cache, filesystem, shell, assembler,
compiler -- is a trained neural network running entirely on GPU.

Usage:
    from ncpu.os.neuros import NeurOS
    os = NeurOS()
    os.boot()
"""

from .boot import NeurOS
from .assembler import NeuralAssembler, ClassicalAssembler
from .compiler import NeuralCompiler
from .watchdog import NeuralWatchdog
from .sync import SyncManager, TensorMutex, TensorSemaphore, TensorBarrier, TensorRWLock
from .protection import MemoryProtectionUnit

__all__ = [
    "NeurOS", "NeuralAssembler", "ClassicalAssembler", "NeuralCompiler",
    "NeuralWatchdog", "SyncManager", "TensorMutex", "TensorSemaphore",
    "TensorBarrier", "TensorRWLock", "MemoryProtectionUnit",
]
