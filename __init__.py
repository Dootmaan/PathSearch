"""PathSearch package initializer.

Keep this file minimal so the package can be imported by IDEs and tools.
"""
__all__ = ["config"]

try:
    # import central config if available
    from . import config  # type: ignore
except ImportError:
    # If config is not yet present or import fails, provide a lightweight fallback
    class _DummyConfig:
        DATA_DIR = "./data"
        CACHE_DIR = "./cache_all"
        RESULTS_DIR = "./temp_result"
        MODEL_DEFAULT = "pathsearch"

    config = _DummyConfig()

__version__ = "0.1"
