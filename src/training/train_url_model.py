"""
src/training/train_url_model.py
-------------------------------
Train a URL-level phishing detector from features exported by the pipeline.

Expected input:
- A directory of JSON files produced by src/inference/pipeline.extract_all(...)
  Typically saved under data/features/*.json

Labeling rules (in order of precedence):
1) If the JSON has a top-level key "label" (str) or "y" (int/bool), it's used.
   Accepted positive strings: {"phish","phishing","spam","malware","malware_lure","bec","qr-phish"}
   Accepted negative strings: {"ham","legit","benign"}
2) Else, filename prefix heuristic:
   - startswith in {"phish_", "phishing_", "spam_", "malware_", "qr_", "qrphish_"} => positive
   - startswith in {"ham_", "legit_"} => negative

Each email JSON may contain multiple URLs (field: "url_feats": list[dict]).
We treat each URL as one training row (email label applied to all URLs in that email).

Output:
- Trained model -> data/models/url_model.pkl
- Report metrics -> data/processed/url_model_report.json

Install (from your ML requirements file):
    pip install -r requirements-ml.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from joblib import dump

# Prefer LightGBM if available; otherwise fall back to RandomForest
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    _HAS_LGBM = False


POS_LABELS = {"phish", "phishing", "spam", "malware", "malware_lure", "bec", "qr-phish", "brand-impersonation"}
NEG_LABELS = {"ham", "legit", "benign"}


def infer_label_from_json(j: Dict[str, Any], fname: str) -> int:
    # JSON-defined label
    if "y" in j:
        y = j["y"]
        if isinstance(y, bool):
            return int(y)
        if isinstance(y, (int, float)):
            return 1 if int(y) > 0 else 0
        if isinstance(y, str):
            return 1 if y.lower().strip() in POS_LABELS else 0

    if "label" in j and isinstance(j["label"], str):
        lab = j["label"].lower().strip()
        if lab in POS_LABELS:
            return 1
        if lab in NEG_LABELS:
            return 0

    # Filename heuristic
    base = Path(fname).name.lower()
    for pref in ("phish_", "phishing_", "spam_", "malware_", "qr_", "qrphish_", "qr-phish_"):
        if base.startswith(pref):
            return 1
    for pref in ("ham_", "legit_"):
        if base.startswith(pref):
            return 0

    # Default to negative if unknown
    return 0


def load_feature_rows(features_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    files = sorted(list(features_dir.glob("*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON feature files found in: {features_dir}")

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # also allow JSON lines (one per email)
            with open(fp, "r", encoding="utf-8") as f:
                lines = [json.loads(x) for x in f if x.strip()]
                if len(lines) == 1:
                    data = lines[0]
                else:
                    # Treat each line as a separate email object
                    for d in lines:
                        y = infer_label_from_json(d, fp.name)
                        for uf in d.get("url_feats", []):
                            rows.append({
                                "url": uf.get("url", ""),
                                "length": uf.get("length", np.nan),
                                "dots": uf.get("dots", np.nan),
                                "has_ip": int(bool(uf.get("has_ip", False))),
                                "contains_login": int(bool(uf.get("contains_login", False))),
                                "has_unicode": int(bool(uf.get("has_unicode", False))),
                                "email_file": fp.name,
                                "label": y,
                            })
                    continue  # go to next file

        y = infer_label_from_json(data, fp.name)
        for uf in data.get("url_feats", []):
            rows.append({
                "url": uf.get("url", ""),
                "length": uf.get("length", np.nan),
                "dots": uf.get("dots", np.nan),
                "has_ip": int(bool(uf.get("has_ip", False))),
                "contains_login": int(bool(uf.get("contains_login", False))),
                "has_unicode": int(bool(uf.get("has_unicode", False))),
                "email_file": fp.name,
                "label": y,
            })

    if not rows:
        raise ValueError("No URL feature rows were found. Ensure your JSON files contain 'url_feats'.")

    df = pd.DataFrame(rows)
    # Drop obvious junk (empty URL rows)
    df = df[df["url"].astype(str).str.len() > 0].reset_index(drop=True)
    return df


def build_model(n_features: int):
    """
    Construct a small training pipeline.
    - imputer + (optional) scaler for numeric columns
    - LightGBM (preferred) or RandomForest
    """
    num_cols = ["length", "dots"]
    bin_cols = ["has_ip", "contains_login", "has_unicode"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), num_cols),
            ("bin", Pipeline([("impute", SimpleImputer(strategy="most_frequent"))]), bin_cols),
        ],
        remainder="drop",
    )

    if _HAS_LGBM:
        clf = LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            objective="binary",
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    feature_names = num_cols + bin_cols
    assert len(feature_names) == n_features
    return pipe


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, y_prob)
    except Exception:
        pr = float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "roc_auc": round(float(roc), 4) if roc == roc else None,
        "pr_auc": round(float(pr), 4) if pr == pr else None,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "threshold": 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Train URL-level phishing model from extracted features.")
    parser.add_argument("--features_dir", default="data/features", type=str, help="Directory with JSON feature files.")
    parser.add_argument("--models_dir", default="data/models", type=str, help="Directory to save trained model.")
    parser.add_argument("--processed_dir", default="data/processed", type=str, help="Directory to save reports.")
    parser.add_argument("--test_size", default=0.2, type=float, help="Holdout fraction.")
    parser.add_argument("--random_state", default=42, type=int)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)
    processed_dir = Path(args.processed_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading features from {features_dir} ...")
    df = load_feature_rows(features_dir)

    # Basic sanity
    print(f"[+] Loaded {len(df)} URL rows from {df['email_file'].nunique()} emails.")
    X = df[["length", "dots", "has_ip", "contains_login", "has_unicode"]].copy()
    y = df["label"].astype(int).values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if y.sum() and y.sum() != len(y) else None
    )

    # Model
    model = build_model(n_features=X.shape[1])

    # Fit
    print("[+] Training model ...")
    model.fit(X_train, y_train)

    # Predict
    print("[+] Evaluating ...")
    if hasattr(model.named_steps["clf"], "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Some tree models (rare) may not expose predict_proba
        y_prob = model.decision_function(X_test)
        # scale to [0,1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

    y_pred = (y_prob >= 0.5).astype(int)
    report = evaluate(y_test, y_prob, y_pred)

    # Save artifacts
    model_path = models_dir / "url_model.pkl"
    dump(model, model_path)
    report_path = processed_dir / "url_model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[+] Saved model -> {model_path}")
    print(f"[+] Saved report -> {report_path}")
    print("[+] Done.")


if __name__ == "__main__":
    main()
