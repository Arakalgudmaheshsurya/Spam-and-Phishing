"""
Train a baseline text model (TF-IDF + LogisticRegression) for phishing/spam detection.

Input: data/processed/text_corpus.csv (id, subject, body, label)
Output:
  - data/models/text_model.joblib   (pipeline: TfidfVectorizer -> LogisticRegression)
  - data/processed/text_model_report.json
"""

import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from joblib import dump

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/text_corpus.csv")
    ap.add_argument("--models_dir", default="data/models")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.corpus)

    # Force subject/body to be strings
    df["subject"] = df["subject"].astype(str).replace("nan", "")
    df["body"]    = df["body"].astype(str).replace("nan", "")

    # Labels
    df["label"] = df["label"].astype(int)
    print("Label distribution:", df["label"].value_counts().to_dict())


# ---- NEW: downsample to something manageable on a laptop ----
    # target per class (you can adjust, e.g. 20000 or 50000)
    target_per_class = 20000

    small_parts = []
    for label_val, group in df.groupby("label"):
        n = min(len(group), target_per_class)
        print(f"[i] Sampling {n} rows from class {label_val} (out of {len(group)})")
        small_parts.append(group.sample(n=n, random_state=args.random_state))

    df = pd.concat(small_parts, ignore_index=True)
    print("New label distribution (after sampling):", df["label"].value_counts().to_dict())


    
    # -------------------------------------------------------------
    X = df["subject"] + " " + df["body"]
    y = df["label"].values

    # Check we actually have both classes
    unique_labels = sorted(set(y.tolist()))
    if len(unique_labels) < 2:
        print(f"[!] Only one class present in labels: {unique_labels}")
        print("[!] You need both ham (0) and phishing/spam (1) examples in text_corpus.csv.")
        return

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if 0 < y.sum() < len(y) else None,
    )


    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1,2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )),
        ("lr", LogisticRegression(max_iter=2000, n_jobs=-1))
    ])

    pipe.fit(Xtr, ytr)

    yprob = pipe.predict_proba(Xte)[:,1]
    yhat  = (yprob >= 0.5).astype(int)

    report = classification_report(yte, yhat, output_dict=True)
    # add PR-AUC/ROC-AUC
    try: report["pr_auc"] = average_precision_score(yte, yprob)
    except: report["pr_auc"] = None
    try: report["roc_auc"] = roc_auc_score(yte, yprob)
    except: report["roc_auc"] = None

    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.processed_dir); processed_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "text_model.joblib"
    dump(pipe, model_path)

    with open(processed_dir / "text_model_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[✅] Saved model -> {model_path}")
    print(f"[✅] Saved report -> {processed_dir/'text_model_report.json'}")

if __name__ == "__main__":
    main()
