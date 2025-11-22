
# audioShieldNet/asnet_3/audioshieldnet/data/asvspoof21_prepare.py

import os, pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "real": 0, "bonafide": 0, "bona_fide": 0, "genuine": 0, "0": 0, 0: 0,
    "fake": 1, "spoof": 1, "attack": 1, "1": 1, 1: 1,
}

def _map_label(x):
    s = str(x).strip().lower()
    if s in LABEL_MAP: return LABEL_MAP[s]
    try: return int(s)
    except: raise ValueError(f"Unknown label {x!r}")

def _read_meta(meta_csv: str, root_dir: str) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    # normalize headers
    cols = {c.lower(): c for c in df.columns}
    if "filepath" not in cols and "path" in cols:
        df = df.rename(columns={cols["path"]: "filepath"})
    if "filepath" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{meta_csv}: expected columns 'path/filepath' and 'label'. Found: {df.columns.tolist()}")

    root_abs = os.path.abspath(root_dir)
    def _rel(p):
        p = os.path.normpath(str(p))
        return os.path.relpath(p, root_abs) if os.path.isabs(p) and p.startswith(root_abs) else p

    df["filepath"] = df["filepath"].astype(str).apply(_rel)
    df["label"] = df["label"].apply(_map_label).astype(int)
    return df[["filepath", "label"]].reset_index(drop=True)

def make_splits(meta_csv: str, root_dir: str, out_dir: str, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    df = _read_meta(meta_csv, root_dir)

    # stratified 80/10/10
    df_train, df_tmp = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"], shuffle=True
    )
    df_val, df_test = train_test_split(
        df_tmp, test_size=0.5, random_state=seed, stratify=df_tmp["label"], shuffle=True
    )

    tr_csv = os.path.join(out_dir, "ASVspoof21_train.csv")
    va_csv = os.path.join(out_dir, "ASVspoof21_val.csv")
    te_csv = os.path.join(out_dir, "ASVspoof21_test.csv")

    df_train.to_csv(tr_csv, index=False)
    df_val.to_csv(va_csv, index=False)
    df_test.to_csv(te_csv, index=False)

    print(f"[ASV21][prepare] train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")
    print(f"[ASV21][prepare] wrote:\n  {tr_csv}\n  {va_csv}\n  {te_csv}")
    return {"train_csv": tr_csv, "val_csv": va_csv, "test_csv": te_csv}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir",   required=True, help="e.g., /scratch/.../ASVspoof2021")
    ap.add_argument("--meta_csv",   required=True, help="e.g., /scratch/.../ASVspoof2021/metadata_LA.csv")
    ap.add_argument("--out_dir",    default=None,  help="default: <root_dir>/combined")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(os.path.abspath(args.root_dir), "combined")
    outs = make_splits(args.meta_csv, args.root_dir, out_dir, seed=args.seed)
    print(json.dumps(outs, indent=2))




# python audioShieldNet/asnet_3/audioshieldnet/data/asvspoof21_prepare.py \
#   --root_dir /scratch/xxxxxx/projects/deepfake/dataset/audio/ASVspoof2021 \
#   --meta_csv /scratch/xxxxxx/projects/deepfake/dataset/audio/ASVspoof2021/metadata_LA.csv
# # outputs to: <root>/combined/ASVspoof21_{train,val,test}.csv
