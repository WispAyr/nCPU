"""ncpu.os — Neural and GPU Operating Systems.

Subpackages:
    ncpu.os.neuros  — neurOS: GPU-native neural operating system (trained models)
    ncpu.os.gpu     — GPU UNIX OS: compiled C running on Metal GPU shaders

For backward compatibility, neurOS symbols are re-exported here so that
``from ncpu.os import NeurOS`` continues to work unchanged.

Module-level imports like ``from ncpu.os.mmu import NeuralMMU`` are also
preserved via sys.modules aliasing.
"""

import importlib
import sys

# Re-export everything from neurOS so existing imports are unaffected
from .neuros import *                          # noqa: F401,F403
from .neuros import (                          # explicit for static analysis
    NeurOS, NeuralAssembler, ClassicalAssembler, NeuralCompiler,
    NeuralWatchdog, SyncManager, TensorMutex, TensorSemaphore,
    TensorBarrier, TensorRWLock, MemoryProtectionUnit,
)

# Backward compatibility: alias ncpu.os.<module> -> ncpu.os.neuros.<module>
# so that `from ncpu.os.mmu import NeuralMMU` still works.
_NEUROS_MODULES = [
    "boot", "assembler", "compiler", "language", "cache", "scheduler",
    "mmu", "tlb", "interrupts", "filesystem", "process", "ipc",
    "watchdog", "sync", "protection", "syscalls", "device", "tui", "shell",
]

for _mod_name in _NEUROS_MODULES:
    _full_old = f"ncpu.os.{_mod_name}"
    _full_new = f"ncpu.os.neuros.{_mod_name}"
    if _full_old not in sys.modules:
        try:
            _mod = importlib.import_module(_full_new)
            sys.modules[_full_old] = _mod
        except ImportError:
            pass

del _mod_name, _full_old, _full_new, _NEUROS_MODULES
try:
    del _mod
except NameError:
    pass

__all__ = [
    "NeurOS", "NeuralAssembler", "ClassicalAssembler", "NeuralCompiler",
    "NeuralWatchdog", "SyncManager", "TensorMutex", "TensorSemaphore",
    "TensorBarrier", "TensorRWLock", "MemoryProtectionUnit",
]
__version__ = "0.1.0"
