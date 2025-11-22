# audioShieldNet/asnet_1/audioshieldnet/data/wavefake_prepare.py


import os, glob, random, math, pandas as pd
from typing import Iterable, List, Dict

LABEL_MAP = {
    "real": 0, "bonafide": 0, "genuine": 0, "0": 0, 0: 0,
    "fake": 1, "spoof": 1, "attack": 1, "1": 1, 1: 1,
}
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a")

def _rel_under(root_abs: str, p: str) -> str:
    p = os.path.normpath(p)
    if os.path.isabs(p) and p.startswith(root_abs):
        return os.path.relpath(p, root_abs)
    return p  # may remain absolute if outside root

def _collect_files(roots: Iterable[str]) -> List[str]:
    out = []
    for r in roots or []:
        r = os.path.abspath(r)
        if os.path.isdir(r):
            # include nested dirs
            for ext in AUDIO_EXTS:
                out.extend(glob.glob(os.path.join(r, "**", f"*{ext}"), recursive=True))
        elif os.path.isfile(r) and r.lower().endswith(AUDIO_EXTS):
            out.append(r)
    return sorted(set(out))

def _strlist(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def make_wavefake_csvs(
    root_dir: str,
    *,
    fake_roots: Iterable[str] = None,   # defaults to <root>/generated_audio/*
    real_roots: Iterable[str] = None,   # optional; if empty, no real added
    split: Dict[str, float] = None,     # e.g., {"train":0.8,"val":0.1,"test":0.1}
    seed: int = 42,
    out_dir: str = None,
) -> Dict[str, str]:
    root_abs = os.path.abspath(root_dir)
    if fake_roots is None:
        fake_roots = glob.glob(os.path.join(root_abs, "generated_audio", "*"))
    fake_files = _collect_files(_strlist(fake_roots))
    real_files = _collect_files(_strlist(real_roots))

    if not fake_files:
        raise FileNotFoundError(f"No fake audio found under: {fake_roots}")

    print(f"[WaveFake][prepare] fake={len(fake_files)}  real={len(real_files)}")

    rows = []
    for p in fake_files:
        rows.append({"filepath": _rel_under(root_abs, p), "label": 1})
    for p in real_files:
        rows.append({"filepath": _rel_under(root_abs, p), "label": 0})

    df = pd.DataFrame(rows, columns=["filepath","label"])

    # Shuffle + split
    rnd = random.Random(seed)
    idxs = list(range(len(df)))
    rnd.shuffle(idxs)

    split = split or {"train": 0.8, "val": 0.1, "test": 0.1}
    assert abs(sum(split.values()) - 1.0) < 1e-6, "split ratios must sum to 1.0"

    n = len(df)
    n_train = int(math.floor(n * split["train"]))
    n_val = int(math.floor(n * split["val"]))
    n_test = n - n_train - n_val

    i_train = idxs[:n_train]
    i_val   = idxs[n_train:n_train+n_val]
    i_test  = idxs[n_train+n_val:]

    out_dir = os.path.abspath(out_dir or os.path.join(root_abs, "metadata", "combined"))
    os.makedirs(out_dir, exist_ok=True)

    def _dump(name, ids):
        outp = os.path.join(out_dir, f"WaveFake_{name}.csv")
        df.iloc[ids][["filepath","label"]].to_csv(outp, index=False)
        print(f"[WaveFake][prepare] wrote {name}: {outp} (N={len(ids)})")
        return outp

    return {
        "train_csv": _dump("train", i_train),
        "val_csv":   _dump("val",   i_val),
        "test_csv":  _dump("test",  i_test),
    }

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--fake_roots", nargs="*", default=None,
                    help="Folders/files with generated audio. Default: <root>/generated_audio/*")
    ap.add_argument("--real_roots", nargs="*", default=None,
                    help="Optional folders/files with genuine audio (e.g., LJSpeech wavs).")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=0.1)
    args = ap.parse_args()
    splits = {"train": args.train, "val": args.val, "test": args.test}
    outs = make_wavefake_csvs(args.root_dir,
                              fake_roots=args.fake_roots,
                              real_roots=args.real_roots,
                              split=splits,
                              seed=args.seed,
                              out_dir=args.out_dir)
    print(json.dumps(outs, indent=2))


# python audioShieldNet/asnet_2/audioshieldnet/data/wavefake_prepare.py \
#   --root_dir /scratch/xxxxxx/projects/deepfake/dataset/audio/waveFake \
#   --real_roots /scratch/xxxxxx/voice_cloning/datasets/LJSpeech/LJSpeech-1.1/wavs \
#   --out_dir /scratch/xxxxxx/projects/deepfake/dataset/audio/waveFake/combined

