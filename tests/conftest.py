"""pytest configuration for nCPU tests."""

import sys
from pathlib import Path

# Ensure the package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
