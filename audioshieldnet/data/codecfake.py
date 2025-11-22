# audioShieldNet/asnet_1/audioshieldnet/data/codecfake.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from audioshieldnet.data.audioshield_dataset import (
    AudioShieldDataset,
    read_index_table,
    make_dataloader,
    compute_class_weights,
)

def _emit_tmp_csv(df: pd.DataFrame, tag: str) -> str:
    """
    Save a normalized (filepath,label) DataFrame to a temp CSV and return its path.
    """
    out_dir = "data_list/_auto_split"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"codecfake_{tag}.csv")
    # keep only required columns, and enforce types
    norm = df[["filepath", "label"]].copy()
    norm["filepath"] = norm["filepath"].astype(str)
    norm["label"] = norm["label"].astype(int)
    norm.to_csv(out_path, index=False)
    return out_path


def _rel_under(root_dir: str, maybe_abs: str) -> str:
    """
    Make path relative to root_dir if it points inside root_dir.
    Otherwise return it as-is.
    """
    p = os.path.normpath(str(maybe_abs))
    r = os.path.abspath(root_dir)
    if os.path.isabs(p) and p.startswith(r):
        return os.path.relpath(p, r)
    return p


def _normalize_paths_to_root(df: pd.DataFrame, root_dir: str) -> pd.DataFrame:
    """
    Ensure 'filepath' column is relative to root_dir where possible.
    """
    df = df.copy()
    df["filepath"] = df["filepath"].astype(str).apply(lambda p: _rel_under(root_dir, p))
    return df


def build_dataloaders(cfg):
    """
    Expected cfg.data keys for CodecFake:
      data:
        name: codecfake
        root_dir: /scratch/.../CodecFake
        # Option A: single index with (filepath,label) for all data (we split 80/20)
        index: /scratch/.../CodecFake/metadata_CodecFake_full.csv
        # Option B: explicit val index (skip split)
        # val_index: /scratch/.../CodecFake/val.csv
        sr: 16000           # default if missing
        max_secs: 6.0       # default if missing

    Returns: (dl_tr, dl_va, class_info_dict)
    """
    dcfg     = cfg["data"]
    root_dir = dcfg["root_dir"]
    sr       = int(dcfg.get("sr", 16000))
    max_secs = float(dcfg.get("max_secs", 6.0))

    idx  = dcfg.get("index", None)
    vidx = dcfg.get("val_index", None)
    if idx is None:
        raise KeyError("CodecFake config requires data.index (path to a CSV/LIST/TXT with 'filepath,label').")

    # ---- Read & normalize the training index ----
    # Supports CSV/TSV/LIST/TXT via read_index_table()
    all_df = read_index_table(idx)
    all_df = _normalize_paths_to_root(all_df, root_dir)

    # ---- Split or use explicit val ----
    if vidx is None:
        # 80/20 stratified split
        use_strat = all_df["label"].nunique() >= 2
        tr_df, va_df = train_test_split(
            all_df, test_size=0.2,
            random_state=cfg["train"].get("seed", 42),
            stratify=all_df["label"] if use_strat else None,
            shuffle=True
        )
    else:
        # read explicit val index as well
        va_df = read_index_table(vidx)
        va_df = _normalize_paths_to_root(va_df, root_dir)
        tr_df = all_df

    # ---- Emit temp CSVs for the generic dataset class ----
    tr_csv = _emit_tmp_csv(tr_df, "train")
    va_csv = _emit_tmp_csv(va_df, "val")

    print(f"[INFO][CodecFake] train index: {tr_csv}  (N={len(tr_df)})")
    print(f"[INFO][CodecFake] val   index: {va_csv}  (N={len(va_df)})")

    # ---- Build datasets ----
    ds_tr = AudioShieldDataset(
        tr_csv, root_dir, sr, max_secs,
        train_mode=True, center_crop_eval=True, fail_policy="zero"
    )
    ds_va = AudioShieldDataset(
        va_csv, root_dir, sr, max_secs,
        train_mode=False, center_crop_eval=True, fail_policy="zero"
    )

    # ---- Balanced sampler on train ----
    labels_np = np.array([int(ds_tr.df.iloc[i]["label"]) for i in range(len(ds_tr))])
    class_counts, sample_weights = compute_class_weights(labels_np)
    print(f"[INFO][CodecFake] class counts real={class_counts[0]} fake={class_counts[1]}")

    bs = int(cfg["train"]["batch_size"])
    steps_per_epoch = cfg["train"].get("steps_per_epoch", None)
    num_samples = (int(steps_per_epoch) * bs) if steps_per_epoch is not None else len(sample_weights)

    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    # ---- DataLoaders (shared helper sets workers/pin/prefetch, etc.) ----
    dl_tr = make_dataloader(ds_tr, cfg, sampler=sampler, shuffle=False, tag="train")
    dl_va = make_dataloader(ds_va, cfg, sampler=None,    shuffle=False, tag="val")

    return dl_tr, dl_va, {"real": int(class_counts[0]), "fake": int(class_counts[1])}
