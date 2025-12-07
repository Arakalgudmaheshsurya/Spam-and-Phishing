"""
src/models/text_model.py
------------------------
Wrapper around the trained TF-IDF + LogisticRegression text model.

Model file is expected at: data/models/text_model.joblib
"""

from pathlib import Path
from typing import Optional
import numpy as np

from src.models.utils import load_joblib_model

DEFAULT_TEXT_MODEL_PATH = Path("data/models/text_model.joblib")


def get_text_model(path: Path = DEFAULT_TEXT_MODEL_PATH):
    """
    Returns the loaded text model pipeline, or None if not found.
    """
    return load_joblib_model(path)


def score_text(text: str, path: Path = DEFAULT_TEXT_MODEL_PATH) -> Optional[float]:
    """
    Returns phishing probability for a given text blob in [0,1],
    or None if the model file is missing.
    """
    model = get_text_model(path)
    if model is None:
        return None

    try:
        probs = model.predict_proba([text])
        return float(probs[0, 1])
    except Exception:
        return None
