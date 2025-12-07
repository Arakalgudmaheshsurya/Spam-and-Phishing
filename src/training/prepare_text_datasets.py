"""
Unify multiple Kaggle text/email datasets into one clean CSV.

Input root (default: data/raw/text_datasets/) can contain folders with CSVs that
have any of these columns:
  - subject, body, text, message, content
  - label columns: {label, target, class, is_spam, category, type}
    (values like: phishing/spam/ham/legit/0/1/true/false)

Output: data/processed/text_corpus.csv with columns:
  id, subject, body, label  (label: 1=phish/spam, 0=ham/legit)
"""

import argparse, re
from pathlib import Path
import pandas as pd

POS = {"phish","phishing","spam","malicious","malware","bad","1","true","yes"}
NEG = {"ham","legit","benign","good","safe","0","false","no"}

LABEL_CANDS = ["label","target","class","is_spam","category","type","result","status"]
SUBJECT_CANDS = ["subject","Subject","SUBJECT"]
BODY_CANDS = ["body","text","message","content","Body","Text","MESSAGE","CONTENT"]

def norm_label(v):
    s = str(v).strip().lower()
    if s in POS: return 1
    if s in NEG: return 0
    # numeric fallback
    try: return 1 if float(s) > 0 else 0
    except: return 0

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def clean_text(x):
    if pd.isna(x): return ""
    s = str(x)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def process_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # pick label
    lab_col = pick_col(df, LABEL_CANDS)
    if lab_col is None:
        raise ValueError(f"No label column found in {csv_path}")
    df["label"] = df[lab_col].apply(norm_label)

    sub_col = pick_col(df, SUBJECT_CANDS)
    body_col = pick_col(df, BODY_CANDS)
    if body_col is None:
        # if nothing obvious, try any text-like longest column
        textish = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
        body_col = textish

    df_out = pd.DataFrame({
        "subject": df[sub_col].apply(clean_text) if sub_col else "",
        "body": df[body_col].apply(clean_text),
        "label": df["label"].astype(int)
    })
    # drop empty rows
    df_out = df_out[(df_out["subject"].str.len() + df_out["body"].str.len()) > 0]
    return df_out.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="data/raw/text_datasets", help="Folder containing dataset folders/CSVs")
    ap.add_argument("--out_csv", default="data/processed/text_corpus.csv")
    ap.add_argument("--limit_per_csv", type=int, default=0, help="sample per CSV for quick tests")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv in list(in_root.rglob("*.csv")):
        try:
            df = process_csv(csv)
            if args.limit_per_csv and len(df) > args.limit_per_csv:
                df = df.sample(args.limit_per_csv, random_state=42)
            df["source"] = csv.relative_to(in_root).as_posix()
            frames.append(df)
            print(f"[+] {csv} -> {len(df)} rows")
        except Exception as e:
            print(f"[!] Skipped {csv}: {e}")

    if not frames:
        raise SystemExit("No CSVs processed. Check your datasets.")

    full = pd.concat(frames, ignore_index=True)
    full.insert(0, "id", range(1, len(full)+1))
    full.to_csv(out_csv, index=False)
    print(f"[✅] Wrote {len(full)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
