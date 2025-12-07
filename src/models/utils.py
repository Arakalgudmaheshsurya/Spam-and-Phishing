"""
src/models/utils.py
-------------------
Shared helpers for loading models from disk with simple caching.
"""

from pathlib import Path
from typing import Any, Optional, Dict
from joblib import load as joblib_load
import joblib

_MODEL_CACHE: Dict[str, Any] = {}

def _safe_exists(path: Path) -> bool:
    try:
        return path.is_file()
    except Exception:
        return False

def load_joblib_model(path: Path) -> Optional[Any]:
    """
    Load a joblib / pickle model with a tiny cache.
    Returns None if the path doesn't exist.
    """
    global _MODEL_CACHE
    key = str(path.resolve())

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if not _safe_exists(path):
        return None

    model = joblib_load(path)
    _MODEL_CACHE[key] = model
    return model
