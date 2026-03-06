"""Device management for neurOS.

Shared device selection and tensor utilities used by all neurOS components.
Follows the same MPS > CUDA > CPU priority as ncpu.neural.cpu.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Module-level default device (lazy-init on first import)
_default_device = None


def default_device() -> torch.device:
    """Get the default device, caching the result."""
    global _default_device
    if _default_device is None:
        _default_device = get_device()
        logger.info(f"[neurOS] Device: {_default_device}")
    return _default_device
