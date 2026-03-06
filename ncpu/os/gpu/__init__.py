"""GPU UNIX OS — Compiled C running on Apple Silicon Metal GPU.

Pipeline: C source -> aarch64-elf-gcc -> raw binary -> Metal compute shader -> Python I/O

Modules:
    runner      — C compilation, syscall handling, process management
    filesystem  — In-memory UNIX filesystem with pipe support
    demo        — Interactive multi-process UNIX shell demo
    shell       — Simple interactive shell demo
"""

from .runner import compile_c, compile_c_from_string, run, make_syscall_handler
from .runner import ProcessManager, run_multiprocess
from .filesystem import GPUFilesystem, PipeBuffer

__all__ = [
    "compile_c", "compile_c_from_string", "run", "make_syscall_handler",
    "ProcessManager", "run_multiprocess",
    "GPUFilesystem", "PipeBuffer",
]
