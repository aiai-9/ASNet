# audioShieldNet/asnet_1/audioshieldnet/data/asvspoof.py

import os, numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

from audioshieldnet.data.audioshield_dataset import (
    AudioShieldDataset,
    make_dataloader,
    compute_class_weights,
)


def _normalize_label_col(df):
    if "filepath" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "filepath"})
    if "label" not in df.columns:
        raise ValueError(f"CSV must have a 'label' column. Found: {df.columns.tolist()}")
    LABEL_MAP = {"real":0,"bonafide":0,"bona_fide":0,"genuine":0,"0":0,0:0,
                 "fake":1,"spoof":1,"attack":1,"1":1,1:1}
    def _map_lbl(x):
        s = str(x).lower()
        if s in LABEL_MAP: return LABEL_MAP[s]
        try: return int(x)
        except: raise ValueError(f"Unknown label: {x}")
    df["label"] = df["label"].apply(_map_lbl)
    return df

def build_dataloaders(cfg):
    meta_csv = cfg['data']['train_csv']; root_dir = cfg['data']['root_dir']
    df = pd.read_csv(meta_csv); df = _normalize_label_col(df)
    
    # df = df.head(100000)  # safety cap
    
    pc = "filepath"
    mask_train_dev = df[pc].astype(str).str.contains("LA_train", case=False) | df[pc].astype(str).str.contains("LA_dev", case=False)
    subset = df[mask_train_dev].copy() if mask_train_dev.any() else df.copy()

    def _norm_path(p):
        p = os.path.normpath(str(p))
        if os.path.isabs(p) and p.startswith(root_dir): return os.path.relpath(p, root_dir)
        return p
    subset["filepath"] = subset[pc].apply(_norm_path)

    use_strat = subset["label"].nunique() >= 2
    tr_df, va_df = train_test_split(subset, test_size=0.2, random_state=cfg['train'].get('seed', 42),
                                    stratify=subset["label"] if use_strat else None, shuffle=True)

    os.makedirs("data_list/_auto_split", exist_ok=True)
    tr_csv = "data_list/_auto_split/train_tmp.csv"; va_csv = "data_list/_auto_split/val_tmp.csv"
    tr_df[["filepath","label"]].to_csv(tr_csv, index=False)
    va_df[["filepath","label"]].to_csv(va_csv, index=False)
    print(f"[INFO] Auto-split → train={len(tr_df)}  val={len(va_df)}")
    

    ds_tr = AudioShieldDataset(
        tr_csv, root_dir, cfg['data']['sr'], cfg['data']['max_secs'],
        train_mode=True,  center_crop_eval=True, fail_policy="zero"
    )
    ds_va = AudioShieldDataset(
        va_csv, root_dir, cfg['data']['sr'], cfg['data']['max_secs'],
        train_mode=False, center_crop_eval=True, fail_policy="zero"
    )

    labels_np = np.array([int(ds_tr.df.iloc[i]["label"]) for i in range(len(ds_tr))])
    class_counts, sample_weights = compute_class_weights(labels_np)
    print(f"[INFO] Train class counts real={class_counts[0]} fake={class_counts[1]}")

    bs = int(cfg['train']['batch_size'])
    steps_per_epoch = cfg['train'].get('steps_per_epoch', None)
    num_samples = (int(steps_per_epoch)*bs) if steps_per_epoch is not None else len(sample_weights)

    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    dl_tr = make_dataloader(ds_tr, cfg, sampler=sampler, shuffle=False, tag="train")
    dl_va = make_dataloader(ds_va, cfg, sampler=None,    shuffle=False, tag="val")

    return dl_tr, dl_va, {"counts": class_counts}
