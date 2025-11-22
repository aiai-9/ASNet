# audioShieldNet/asnet_3/audioshieldnet/data/prepare_data/librisevoc_prepare.py

import os, re, json, pandas as pd
from typing import Tuple

LABEL_MAP = {
    "real": 0, "bonafide": 0, "bona_fide": 0, "genuine": 0, "0": 0, 0: 0,
    "fake": 1, "spoof": 1, "attack": 1, "1": 1, 1: 1,
}

def _map_label(x) -> int:
    s = str(x).strip().lower()
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    try:
        return int(s)
    except Exception:
        # common LibriSeVoc: label is second column 0/1; if missing, infer from path
        p = s
        if re.search(r"(bonafide|bona[_-]?fide|genuine|/gt/)", p): return 0
        if re.search(r"(spoof|fake|attack|wavegrad|wavenet|wavernn|melgan|parallel_wave_gan|diffwave)", p): return 1
        raise ValueError(f"Unknown label value: {x!r}")

def _rel_under(root_abs: str, p: str) -> str:
    p = os.path.normpath(str(p))
    if os.path.isabs(p) and p.startswith(root_abs):
        return os.path.relpath(p, root_abs)
    return p

def _read_list(list_path: str, root_dir: str) -> pd.DataFrame:
    """
    Supports typical LibriSeVoc lines:
      /abs/.../gt/xxx.wav,0,gt
      /abs/.../wavegrad/xxx_gen.wav,1,wavegrad
    Also handles space/tab separated.
    """
    rows = []
    with open(list_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [x.strip() for x in re.split(r"[,\t ]+", ln) if x.strip()]
            if len(parts) == 1:
                p, y = parts[0], _map_label(parts[0])
            else:
                p, y = parts[0], _map_label(parts[1])
            rows.append((_rel_under(os.path.abspath(root_dir), p), int(y)))
    df = pd.DataFrame(rows, columns=["filepath", "label"])
    return df

def prepare_lsv(root_dir: str,
                lists_dir: str = None,
                train_list: str = "train.list",
                val_list:   str = "dev.list",
                test_list:  str = "test.list",
                out_dir:    str = None) -> Tuple[str, str, str]:
    root_abs = os.path.abspath(root_dir)
    lists_dir = lists_dir or os.path.join(root_abs, "lists")
    out_dir = os.path.abspath(out_dir or os.path.join(root_abs, "combined"))
    os.makedirs(out_dir, exist_ok=True)

    def _resolve(name: str) -> str:
        # prefer lists_dir, fall back to lists_2 if present
        p1 = os.path.join(lists_dir, name)
        p2 = os.path.join(os.path.dirname(lists_dir), "lists_2", name)
        if os.path.isfile(p1): return p1
        if os.path.isfile(p2): return p2
        raise FileNotFoundError(f"Could not find {name} in {lists_dir} or lists_2/")

    tr = _read_list(_resolve(train_list), root_abs)
    va = _read_list(_resolve(val_list),   root_abs)
    try:
        te = _read_list(_resolve(test_list),  root_abs)
    except FileNotFoundError:
        te = pd.DataFrame(columns=["filepath", "label"])

    tr_csv = os.path.join(out_dir, "LibriSeVoc_train.csv")
    va_csv = os.path.join(out_dir, "LibriSeVoc_val.csv")
    te_csv = os.path.join(out_dir, "LibriSeVoc_test.csv")
    tr.to_csv(tr_csv, index=False)
    va.to_csv(va_csv, index=False)
    te.to_csv(te_csv, index=False)

    print(f"[LSV][prepare] train={len(tr)}  val={len(va)}  test={len(te)}")
    print(f"[LSV][prepare] wrote:\n  {tr_csv}\n  {va_csv}\n  {te_csv}")
    return tr_csv, va_csv, te_csv

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--lists_dir", default=None)
    ap.add_argument("--train_list", default="train.list")
    ap.add_argument("--val_list",   default="dev.list")
    ap.add_argument("--test_list",  default="test.list")
    ap.add_argument("--out_dir",    default=None)
    args = ap.parse_args()
    outs = prepare_lsv(args.root_dir, args.lists_dir, args.train_list, args.val_list, args.test_list, args.out_dir)
    print(json.dumps({"train_csv": outs[0], "val_csv": outs[1], "test_csv": outs[2]}, indent=2))

