"""Central configuration for PathSearch.

This module exposes defaults and reads environment variables to allow
overriding paths without editing source files.
"""
import os
from pathlib import Path

# Repository-local defaults
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('PATHSEARCH_DATA_DIR', REPO_ROOT / 'data'))
CACHE_DIR = Path(os.environ.get('PATHSEARCH_CACHE_DIR', REPO_ROOT / 'cache_all'))
RESULTS_DIR = Path(os.environ.get('PATHSEARCH_RESULTS_DIR', REPO_ROOT / 'temp_result'))

# Labels CSVs and other dataset files can be pointed to via env vars.
DHMC_LUAD_CSV = Path(os.environ.get('DHMC_LUAD_CSV', REPO_ROOT / 'EasyMIL' / 'dataset_csv' / 'DHMC-LUAD.csv'))

# Default model path/name (Hugging Face repo or local path)
MODEL_DEFAULT = os.environ.get('PATHSEARCH_MODEL_DEFAULT', 'pathsearch')

# Optional external weights directory (use env to override)
MILIP_WEIGHTS_PATH = os.environ.get('PATHSEARCH_MILIP_WEIGHTS', '')

def ensure_dirs():
    """Create cache/results directories if they do not exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    'REPO_ROOT', 'DATA_DIR', 'CACHE_DIR', 'RESULTS_DIR',
    'DHMC_LUAD_CSV', 'MODEL_DEFAULT', 'ensure_dirs'
]
