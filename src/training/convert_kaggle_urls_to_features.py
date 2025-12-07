"""
Convert Kaggle-style URL CSVs into our pipeline feature JSON files.

Input CSV must have:
- a 'url' column
- a label column (one of: label, target, result, status, type) with values like
  {phishing, spam, malicious, bad, 1} or {benign, good, safe, 0}

Usage:
  python src/training/convert_kaggle_urls_to_features.py \
      --csv data/raw/kaggle_urls.csv \
      --outdir data/features \
      --prefix kaggle

This produces many small JSON files compatible with train_url_model.py.
"""

import argparse
import json
from pathlib import Path
import pandas as pd

POS = {"phish","phishing","malicious","malware","bad","spam","1","true","yes"}
NEG = {"benign","legit","good","safe","0","false","no"}

LABEL_CANDIDATES = ["label","target","result","status","type","class"]

def normalize_label(v):
    s = str(v).strip().lower()
    if s in POS:
        return 1
    if s in NEG:
        return 0
    # numeric fallback
    try:
        return 1 if float(s) > 0 else 0
    except Exception:
        return 0

def basic_url_feats(u: str):
    return {
        "url": u,
        "length": len(u),
        "dots": u.count("."),
        "has_ip": 1 if u.startswith(("http://","https://")) and u.split("/")[2].replace(".","").isdigit() else 0,
        "contains_login": 1 if any(k in u.lower() for k in ["login","signin","verify","update","password"]) else 0,
        "has_unicode": 1 if any(ord(c)>127 for c in u) else 0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="data/features")
    ap.add_argument("--prefix", default="kaggle")
    ap.add_argument("--limit", type=int, default=0, help="optional cap for quick tests")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "url" not in df.columns:
        raise ValueError("Input CSV must have a 'url' column.")

    label_col = None
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            label_col = c; break
    if label_col is None:
        raise ValueError("Could not find a label column (tried: {}).".format(", ".join(LABEL_CANDIDATES)))

    if args.limit and len(df) > args.limit:
        df = df.sample(args.limit, random_state=42)

    # Group into small batches so each JSON looks like one 'email' with several URLs
    # (helps re-use the same trainer without modification)
    batch_size = 10
    batch = []
    batch_idx = 0

    def write_batch(rows, y):
        nonlocal batch_idx
        if not rows: return
        data = {
            "label": "phishing" if y==1 else "ham",
            "urls": [r["url"] for r in rows],
            "url_feats": [basic_url_feats(r["url"]) for r in rows],
            "text": "",
            "html_len": 0,
            "images": [],
        }
        fname = f"{args.prefix}_{'phish' if y==1 else 'ham'}_{batch_idx:06d}.json"
        (outdir / fname).write_text(json.dumps(data, indent=2), encoding="utf-8")
        batch_idx += 1

    # Split by label so each batch file is cleanly labeled
    for yval, group in df.groupby(df[label_col].apply(normalize_label)):
        urls = [{"url": u} for u in group["url"].astype(str).tolist() if isinstance(u, str) and u.strip()]
        for i in range(0, len(urls), batch_size):
            write_batch(urls[i:i+batch_size], int(yval))

    print(f"✅ Wrote feature JSONs to: {outdir}")

if __name__ == "__main__":
    main()
