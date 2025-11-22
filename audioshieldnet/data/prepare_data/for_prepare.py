# audioShieldNet/asnet_1/audioshieldnet/data/for_prepare.py
# python audioShieldNet/asnet_2/audioshieldnet/data/for_prepare.py \
#   --root_dir /scratch/xxxxxx/projects/deepfake/dataset/audio/FOR \
#   --metadata_dir /scratch/xxxxxx/projects/deepfake/dataset/audio/FOR/metadata \
#   --variants for-norm for-original for-rerec for-2sec \
#   --out_dir /scratch/xxxxxx/projects/deepfake/dataset/audio/FOR/metadata/combined


import os, pandas as pd

LABEL_MAP = {
    "real": 0, "bonafide": 0, "bona_fide": 0, "genuine": 0, "0": 0, 0: 0,
    "fake": 1, "spoof": 1, "attack": 1, "1": 1, 1: 1,
}

def _map_label(x):
    s = str(x).strip().lower()
    if s in LABEL_MAP: return LABEL_MAP[s]
    try: return int(s)
    except: raise ValueError(f"Unknown label {x!r}")

def _read_one_csv(path_csv: str, root_dir: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    cols = {c.lower(): c for c in df.columns}
    if "filepath" not in cols and "path" in cols:
        df = df.rename(columns={cols["path"]: "filepath"})
    if "filepath" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path_csv}: expected columns 'path/filepath' and 'label'")
    df["filepath"] = df["filepath"].astype(str)

    # make relative under root_dir when possible
    root_abs = os.path.abspath(root_dir)
    def _rel(p):
        p = os.path.normpath(p)
        if os.path.isabs(p) and p.startswith(root_abs):
            return os.path.relpath(p, root_abs)
        return p
    df["filepath"] = df["filepath"].apply(_rel)
    df["label"] = df["label"].apply(_map_label).astype(int)
    return df[["filepath","label"]]

def combine_for_variants(root_dir: str,
                         metadata_dir: str,
                         variants: list,
                         out_dir: str = None):
    """
    Combines FOR per-variant CSVs into 3 unified CSVs.
    Returns dict with keys: train_csv, val_csv, test_csv.
    """
    if variants == "all" or variants is None:
        variants = ["for-norm", "for-original", "for-rerec", "for-2sec"]
    assert isinstance(variants, (list, tuple)) and len(variants) > 0

    md = os.path.abspath(metadata_dir)
    out_dir = os.path.abspath(out_dir or os.path.join(md, "combined"))
    os.makedirs(out_dir, exist_ok=True)

    def _csv_path(variant: str, split: str):
        # e.g., metadata/FOR/for-norm/metadata_FOR_for-norm_training.csv
        return os.path.join(md, variant, f"metadata_FOR_{variant}_{split}.csv")

    splits = {"training":"train", "validation":"val", "testing":"test"}
    buckets = {k: [] for k in splits.values()}

    for v in variants:
        for src, dst in splits.items():
            p = _csv_path(v, src)
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            buckets[dst].append(_read_one_csv(p, root_dir))

    outs = {}
    for dst, parts in buckets.items():
        cat = pd.concat(parts, ignore_index=True)
        out_csv = os.path.join(out_dir, f"FOR_{'-'.join(variants)}_{dst}.csv")
        cat.to_csv(out_csv, index=False)
        outs[f"{dst}"] = out_csv
        print(f"[FOR][prepare] wrote {dst}: {out_csv}  (N={len(cat)})")

    return outs

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--metadata_dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["for-norm","for-original","for-rerec","for-2sec"])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    outs = combine_for_variants(args.root_dir, args.metadata_dir, args.variants, args.out_dir)
    print(json.dumps(outs, indent=2))
