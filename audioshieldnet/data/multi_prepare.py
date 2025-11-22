# audioShieldNet/asnet_3/audioshieldnet/data/multi_prepare.py

# audioshieldnet/data/multi_prepare.py
import os, json
import pandas as pd

from typing import List, Dict, Tuple

LABEL_MAP = {
    "real": 0, "bonafide": 0, "bona_fide": 0, "genuine": 0, "0": 0, 0: 0,
    "fake": 1, "spoof": 1, "attack": 1, "1": 1, 1: 1,
}

def _map_label(x):
    s = str(x).strip().lower()
    if s in LABEL_MAP: return LABEL_MAP[s]
    try: return int(s)
    except: raise ValueError(f"Unknown label {x!r}")

def _read_idx(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    cols_lower = {c.lower(): c for c in df.columns}

    # Normalize: prefer 'filepath'; if only 'path' exists, rename it.
    if "filepath" not in cols_lower:
        if "path" in cols_lower:
            df = df.rename(columns={cols_lower["path"]: "filepath"})
        else:
            raise ValueError(f"{path_csv}: expected a 'filepath' or 'path' column")

    if "label" not in cols_lower:
        raise ValueError(f"{path_csv}: expected a 'label' column")

    df["filepath"] = df[cols_lower.get("filepath", "filepath")].astype(str)
    df["label"] = df[cols_lower["label"]].apply(_map_label).astype(int)
    return df[["filepath", "label"]]


def _glob_in(dir_: str, key: str) -> str:
    cands = [f for f in os.listdir(dir_) if key in f.lower() and f.lower().endswith(".csv")]
    if not cands:
        raise FileNotFoundError(f"No *{key}*.csv in {dir_}")
    if len(cands) > 1:
        # pick the longest name (often the “combined”/most explicit one)
        cands.sort(key=len, reverse=True)
    return os.path.join(dir_, cands[0])

def _train_val_test_from_source(src: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df or empty df)."""
    if "csv_dir" in src:
        d = os.path.abspath(src["csv_dir"])
        tr = _read_idx(_glob_in(d, "train"))
        va = _read_idx(_glob_in(d, "val"))
        te = _read_idx(_glob_in(d, "test")) if any("test" in f.lower() for f in os.listdir(d)) else pd.DataFrame(columns=["filepath","label"])
        return tr, va, te

    # explicit file paths
    if "train_csv" in src and "val_csv" in src:
        tr = _read_idx(src["train_csv"])
        va = _read_idx(src["val_csv"])
        te = _read_idx(src["test_csv"]) if "test_csv" in src and os.path.isfile(src["test_csv"]) else pd.DataFrame(columns=["filepath","label"])
        return tr, va, te

    # single full CSV → split 80/10/10
    if "csv" in src and os.path.isfile(src["csv"]):
        full = _read_idx(src["csv"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(full); n_tr = int(0.8*n); n_va = int(0.1*n)
        tr = full.iloc[:n_tr].copy()
        va = full.iloc[n_tr:n_tr+n_va].copy()
        te = full.iloc[n_tr+n_va:].copy()
        return tr, va, te

    raise ValueError(f"Bad source spec: {src}")

def combine_sources(sources: List[Dict], out_dir: str):
    """
    sources: list of dicts (examples)
      - {csv_dir: "/path/to/combined"}    # contains *train.csv, *val.csv, *test.csv
      - {train_csv: "...", val_csv: "...", test_csv: "..."}
      - {csv: "/path/to/full.csv"}        # auto 80/10/10 split

    Writes multi_train.csv/multi_val.csv/(multi_test.csv) with dataset_id/name.
    """
    os.makedirs(out_dir, exist_ok=True)

    tr_parts, va_parts, te_parts = [], [], []
    for ds_id, src in enumerate(sources):
        ds_name = src.get("name", f"ds{ds_id}")
        tr, va, te = _train_val_test_from_source(src)
        for df in (tr, va, te):
            if not df.empty:
                df["dataset_id"] = ds_id
                df["dataset_name"] = ds_name
        tr_parts.append(tr)
        va_parts.append(va)
        if not te.empty:
            te_parts.append(te)

    def _write(cat: pd.DataFrame, name: str):
        out = os.path.join(out_dir, f"{name}.csv")
        cat.to_csv(out, index=False)
        print(f"[multi][prepare] wrote {name}: {out} (N={len(cat)})")
        return out

    tr_all = pd.concat(tr_parts, ignore_index=True).sample(frac=1.0, random_state=123)
    va_all = pd.concat(va_parts, ignore_index=True)
    outs = {
        "train_csv": _write(tr_all, "multi_train"),
        "val_csv":   _write(va_all, "multi_val"),
    }
    if te_parts:
        te_all = pd.concat(te_parts, ignore_index=True)
        outs["test_csv"] = _write(te_all, "multi_test")
    return outs

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sources", required=True,
                    help="JSON list of sources; each can include name and one of {csv_dir, (train_csv,val_csv[,test_csv]), csv}")
    args = ap.parse_args()
    outs = combine_sources(json.loads(args.sources), args.out_dir)
    print(json.dumps(outs, indent=2))
