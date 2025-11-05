"""PathSearch.dataset package initializer.

Expose commonly used dataset classes for convenient imports like:
    from dataset import DHMCLUADDataset
"""
from typing import List

__all__: List[str] = []

try:
    from .DHMCLUADDataset import DHMCLUADDataset  # type: ignore
    __all__.append("DHMCLUADDataset")
except Exception:
    # best-effort import; avoid crashing on IDE analysis if module import fails
    pass

try:
    from .AnonymousDataset import AnonymousDataset  # type: ignore
    __all__.append("AnonymousDataset")
except Exception:
    pass
