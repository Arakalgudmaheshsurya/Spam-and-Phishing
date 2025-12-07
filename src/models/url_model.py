"""
src/models/url_model.py
-----------------------
Wrapper for URL-level model (LightGBM/RandomForest) trained by train_url_model.py

Model file: data/models/url_model.pkl

Input: list of url_feats dicts from pipeline:
  {"url": "...", "length": ..., "dots": ..., "has_ip": 0/1,
   "contains_login": 0/1, "has_unicode": 0/1}
"""

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from src.models.utils import load_joblib_model

DEFAULT_URL_MODEL_PATH = Path("data/models/url_model.pkl")


def get_url_model(path: Path = DEFAULT_URL_MODEL_PATH):
    return load_joblib_model(path)


def score_urls(
    url_feats: List[Dict],
    path: Path = DEFAULT_URL_MODEL_PATH,
) -> Optional[List[float]]:
    """
    Returns a list of phishing probabilities (0-1) per URL, or None if model missing.
    """
    if not url_feats:
        return []

    model = get_url_model(path)
    if model is None:
        return None

    df = pd.DataFrame(url_feats)
    # ensure required columns exist
    for col in ["length", "dots", "has_ip", "contains_login", "has_unicode"]:
        if col not in df.columns:
            df[col] = 0

    X = df[["length", "dots", "has_ip", "contains_login", "has_unicode"]]

    try:
        probs = model.predict_proba(X)[:, 1]
        return [float(x) for x in probs]
    except Exception:
        return None
