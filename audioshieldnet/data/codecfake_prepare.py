# audioShieldNet/asnet_1/audioshieldnet/data/prepare_data/codecfake_prepare.py


import os, sys, glob, pandas as pd
from sklearn.model_selection import train_test_split

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
REAL_NAMES = {"genuine", "real", "bonafide", "bona_fide"}

def _is_audio(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in AUDIO_EXTS

def _rel_under(root_abs: str, p: str) -> str:
    p = os.path.normpath(p)
    return os.path.relpath(p, root_abs) if os.path.isabs(p) and p.startswith(root_abs) else p

def crawl_codecfake(root_dir: str,
                    include_generators=None,
                    exclude_generators=None,
                    follow_symlinks=False):
    """
    Walks <root_dir> (your CodecFake directory) and builds a dataframe with columns:
      filepath, label  (0=genuine, 1=fake), src (folder name)
    Any top-level subfolder whose name is in REAL_NAMES is treated as real; all others → fake.
    You can restrict with include_generators / exclude_generators (by folder name).
    """
    root_abs = os.path.abspath(root_dir)
    include = set(x.lower() for x in include_generators) if include_generators else None
    exclude = set(x.lower() for x in exclude_generators) if exclude_generators else set()

    rows = []
    for item in sorted(os.listdir(root_abs)):
        full = os.path.join(root_abs, item)
        if not os.path.isdir(full):
            continue
        name = item.lower()
        if include and name not in include:
            continue
        if name in exclude:
            continue

        label = 0 if name in REAL_NAMES else 1
        # collect audio recursively
        for p, _, files in os.walk(full, followlinks=follow_symlinks):
            for f in files:
                if _is_audio(f):
                    abspath = os.path.join(p, f)
                    rel = _rel_under(root_abs, abspath)
                    rows.append((rel, label, item))

    if not rows:
        raise RuntimeError(f"No audio files detected under: {root_dir}")
    df = pd.DataFrame(rows, columns=["filepath", "label", "src"])
    return df

def make_splits(root_dir: str,
                out_dir: str = None,
                include_generators=None,
                exclude_generators=None,
                seed: int = 42):
    """
    Stratified 80/10/10 split → writes CSVs into <root_dir>/combined by default.
    Returns dict with train_csv, val_csv, test_csv.
    """
    root_abs = os.path.abspath(root_dir)
    out_dir = os.path.abspath(out_dir or os.path.join(root_abs, "combined"))
    os.makedirs(out_dir, exist_ok=True)

    df = crawl_codecfake(root_abs, include_generators, exclude_generators)

    # stratified 80/10/10 by label (keeps real/fake ratio)
    tr, tmp = train_test_split(df, test_size=0.2, random_state=seed,
                               stratify=df["label"], shuffle=True)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed,
                              stratify=tmp["label"], shuffle=True)

    tr_csv = os.path.join(out_dir, "CodecFake_train.csv")
    va_csv = os.path.join(out_dir, "CodecFake_val.csv")
    te_csv = os.path.join(out_dir, "CodecFake_test.csv")
    tr[["filepath","label"]].to_csv(tr_csv, index=False)
    va[["filepath","label"]].to_csv(va_csv, index=False)
    te[["filepath","label"]].to_csv(te_csv, index=False)

    print(f"[CodecFake][prepare] N={len(df)} → train={len(tr)}  val={len(va)}  test={len(te)}")
    print(f"[CodecFake][prepare] wrote:\n  {tr_csv}\n  {va_csv}\n  {te_csv}")
    return {"train_csv": tr_csv, "val_csv": va_csv, "test_csv": te_csv}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help=".../dataset/audio/CodecFake")
    ap.add_argument("--out_dir",  default=None)
    ap.add_argument("--include",  nargs="*", default=None, help="limit to these subfolders (by name)")
    ap.add_argument("--exclude",  nargs="*", default=None, help="exclude these subfolders (by name)")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()
    outs = make_splits(args.root_dir, args.out_dir, args.include, args.exclude, seed=args.seed)
    print(json.dumps(outs, indent=2))



# python audioShieldNet/asnet_3/audioshieldnet/data/prepare_data/codecfake_prepare.py \
#   --root_dir /scratch/xxxxxx/projects/deepfake/dataset/audio/CodecFake
