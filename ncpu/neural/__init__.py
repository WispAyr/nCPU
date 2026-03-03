"""Full Neural CPU: every component replaced by trained neural networks.

This is the complete neural CPU where ALU, decoder, register file, memory
operations, shifts, comparisons, and math functions are all implemented as
trained PyTorch models running on GPU.

Requires: torch, numpy

Quick start:
    from ncpu.neural import NeuralCPU
    cpu = NeuralCPU()
"""

try:
    from .cpu import NeuralCPU
    __all__ = ["NeuralCPU"]
except ImportError as e:
    import warnings
    warnings.warn(
        f"ncpu.neural requires torch and numpy: {e}. "
        "Install with: pip install ncpu[model]"
    )
    __all__ = []
