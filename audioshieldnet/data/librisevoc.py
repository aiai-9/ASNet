# audioShieldNet/asnet_3/audioshieldnet/data/librisevoc_split.py

import os
import random
import numpy as np
from typing import Optional, List, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from audioshieldnet.data.audioshield_dataset import (
    AudioShieldDataset,
    mute_worker,
    compute_class_weights,   # returns (class_counts, sample_weights) given labels
)
from audioshieldnet.data.prepare_data.librisevoc_prepare import prepare_lsv


# -------------------------
# Determinism helpers
# -------------------------
def _get_seed_from_cfg(cfg, default: int = 42) -> int:
    try:
        return int(((cfg.get("train", {}) or {}).get("seed", default)))
    except Exception:
        return int(default)

def _build_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def _worker_init_seeded(worker_id: int, base_seed: int):
    # Keep your existing "mute" behavior and add deterministic RNGs
    try:
        mute_worker(worker_id)
    except Exception:
        pass
    wseed = (int(base_seed) + int(worker_id)) % (2**32)
    np.random.seed(wseed)
    random.seed(wseed)


# -------------------------
# Label utilities (AudioShieldDataset or Subset)
# -------------------------
def _labels_from_dataset(ds) -> np.ndarray:
    """
    Return labels as np.int array for AudioShieldDataset or Subset(AudioShieldDataset).
    Falls back to empty array if unknown dataset type.
    """
    if ds is None:
        return np.array([], dtype=int)

    # Direct AudioShieldDataset with a .df
    if hasattr(ds, "df"):
        return ds.df["label"].to_numpy(dtype=int)

    # Subset wrapping an AudioShieldDataset (handle nested subsets)
    if isinstance(ds, Subset):
        base = ds.dataset
        if hasattr(base, "df") and hasattr(ds, "indices"):
            base_labels = base.df["label"].to_numpy(dtype=int)
            idx = np.asarray(ds.indices, dtype=int)
            return base_labels[idx]
        labs = _labels_from_dataset(base)
        if labs.size and hasattr(ds, "indices"):
            idx = np.asarray(ds.indices, dtype=int)
            return labs[idx]

    # Unknown dataset type
    return np.array([], dtype=int)

def _split_counts(ds) -> Tuple[int, int]:
    """Returns (n_real, n_fake) for AudioShieldDataset or Subset(...) safely."""
    labs = _labels_from_dataset(ds)
    if labs.size == 0:
        return (0, 0)
    cnts = np.bincount(labs, minlength=2)
    return int(cnts[0]), int(cnts[1])


# -------------------------
# Misc utilities
# -------------------------
def _print_counts(tag: str, n_real: int, n_fake: int, n_total: Optional[int] = None):
    total = (n_total if n_total is not None else (n_real + n_fake))
    prior_fake = (n_fake / total) if total > 0 else 0.0
    print(f"[INFO][LibriSeVoc][{tag:>8}] total={total:6d}  real={n_real:6d}  fake={n_fake:6d}  prior(fake)={prior_fake:.4f}")

def _rng(seed: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(None if seed is None else int(seed))

def _stratified_pick_indices(labels: np.ndarray, frac: float, seed: Optional[int]) -> List[int]:
    """
    Pick approximately `frac` of indices per class (min 1 if that class exists).
    Returns a list of absolute row indices into the dataset.
    """
    rs = _rng(seed)
    n = labels.size
    if n == 0 or frac <= 0:
        return []
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    def _pick(idxs: np.ndarray) -> np.ndarray:
        if idxs.size == 0:
            return idxs
        k = max(1, int(np.floor(frac * idxs.size)))
        if k >= idxs.size:
            return idxs.copy()
        return rs.choice(idxs, size=k, replace=False)

    pick0 = _pick(idx0)
    pick1 = _pick(idx1)
    picked = np.concatenate([pick0, pick1])
    rs.shuffle(picked)
    return picked.tolist()


# -------------------------
# DataLoader factory (seeded, deterministic workers)
# -------------------------
def _make_loader(dataset, cfg, *, sampler=None, shuffle=False, tag="train", generator: Optional[torch.Generator] = None):
    tcfg = cfg.get("train", {}) or {}
    bs   = int(tcfg.get("batch_size", 32))
    nw   = int(tcfg.get("num_workers", 16))
    pf   = int(tcfg.get("prefetch_factor", 8))
    pw   = bool(tcfg.get("persistent_workers", True))
    pin  = bool(tcfg.get("pin_memory", True))

    seed = _get_seed_from_cfg(cfg, 42)
    def _wif(worker_id: int):
        _worker_init_seeded(worker_id, seed)

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=pf,
        worker_init_fn=_wif,
        generator=generator,  # critical for determinism
    )


# -------------------------
# Builders
# -------------------------
def build_dataloaders(cfg):
    """
    Returns (dl_tr, dl_va, dl_cal) for training-time use on LibriSeVoc.
    - TRAIN: uniform shuffle or WeightedRandomSampler (cfg.data.sampler).
    - VAL:   sequential loader (shuffle=False).
    - CAL:   optional explicit cal split (data.cal_csv). For CAL-from-TEST, use build_testloader().
    Also fills cfg['data']['split_counts'] with counts for logging.
    """
    d = cfg["data"]
    root_dir = d["root_dir"]
    sr       = int(d.get("sr", 22050))
    max_secs = float(d.get("max_secs", 6.0))

    # Prepare CSVs (auto or explicit)
    if bool(d.get("auto_prepare", True)):
        tr_csv, va_csv, te_csv = prepare_lsv(
            root_dir=root_dir,
            lists_dir=d.get("lists_dir"),
            train_list=d.get("train_list", "train.list"),
            val_list=d.get("val_list", "dev.list"),
            test_list=d.get("test_list", "test.list"),
            out_dir=d.get("out_dir"),
        )
        # Let explicit overrides win if provided
        tr_csv = d.get("train_csv", tr_csv)
        va_csv = d.get("val_csv", va_csv)
        if d.get("test_csv") is None and te_csv is not None:
            d["test_csv"] = te_csv
    else:
        tr_csv, va_csv = d["train_csv"], d["val_csv"]

    cal_csv = d.get("cal_csv", None)

    # Datasets
    ds_tr  = AudioShieldDataset(tr_csv, root_dir, sr, max_secs, train_mode=True, fail_policy="skip")
    ds_va  = AudioShieldDataset(va_csv, root_dir, sr, max_secs, train_mode=False)
    ds_cal = AudioShieldDataset(cal_csv, root_dir, sr, max_secs, train_mode=False) if cal_csv else None

    # Seeded generator shared by all loaders/samplers
    seed = _get_seed_from_cfg(cfg, 42)
    gen  = _build_torch_generator(seed)

    # Sampler / weights for TRAIN only
    labels_tr = _labels_from_dataset(ds_tr)
    class_counts_tr, sample_w_tr = compute_class_weights(labels_tr)

    bs = int(cfg["train"]["batch_size"])
    steps = cfg["train"].get("steps_per_epoch", None)
    num_samples = (int(steps) * bs) if steps is not None else len(sample_w_tr)

    sampler_type = (d.get("sampler", "uniform") or "uniform").lower()
    if sampler_type in ("weighted", "class_balanced", "balanced"):
        sampler_tr = WeightedRandomSampler(
            weights=sample_w_tr,
            num_samples=num_samples,
            replacement=True,
            generator=gen
        )
        shuffle_tr = False
    else:
        sampler_tr = None
        shuffle_tr = True

    # Loaders (all get the same generator)
    dl_tr  = _make_loader(ds_tr, cfg, sampler=sampler_tr, shuffle=shuffle_tr, tag="train", generator=gen)
    dl_va  = _make_loader(ds_va, cfg, sampler=None, shuffle=False, tag="val", generator=gen)
    dl_cal = _make_loader(ds_cal, cfg, sampler=None, shuffle=False, tag="cal", generator=gen) if ds_cal else None

    # Counts & priors summary
    tr_real, tr_fake = _split_counts(ds_tr)
    va_real, va_fake = _split_counts(ds_va)
    cal_real, cal_fake = _split_counts(ds_cal) if ds_cal else (0, 0)

    _print_counts("train", tr_real, tr_fake)
    _print_counts("val",   va_real, va_fake)
    if ds_cal:
        _print_counts("cal", cal_real, cal_fake)
    _print_counts("sum", tr_real + va_real + cal_real, tr_fake + va_fake + cal_fake)

    # Store counts for downstream use (bias init, logs, etc.)
    cfg.setdefault("data", {}).setdefault("prior_counts", {"real": tr_real, "fake": tr_fake})
    cfg["data"].setdefault("split_counts", {})
    cfg["data"]["split_counts"].update({
        "train": {"real": tr_real, "fake": tr_fake},
        "val":   {"real": va_real, "fake": va_fake},
    })
    if ds_cal:
        cfg["data"]["split_counts"]["cal"] = {"real": cal_real, "fake": cal_fake}

    return dl_tr, dl_va, dl_cal


def build_testloader(cfg):
    """
    Build TEST (and optional CAL-from-TEST) for LibriSeVoc in a deterministic way.
    Controlled by:
      data.test_csv
      data.cal_csv (explicit cal) OR
      data.cal_from_test_frac (float, 0..1) + data.cal_exclude_from_test (bool)
    Returns: dl_te, dl_cal (dl_cal may be None)
    """
    d = cfg["data"]
    test_csv = d.get("test_csv", None)
    if not test_csv:
        print("[INFO][LibriSeVoc][   test] no test_csv provided – returning (None, None)")
        return None, None

    root_dir = d["root_dir"]
    sr = int(d.get("sr", 22050))
    max_secs = float(d.get("max_secs", 6.0))

    # Full TEST dataset
    ds_te_full = AudioShieldDataset(test_csv, root_dir, sr, max_secs, train_mode=False)
    te_real, te_fake = _split_counts(ds_te_full)
    _print_counts("test(full)", te_real, te_fake)

    # Deterministic generator shared by CAL/TEST loaders
    seed = _get_seed_from_cfg(cfg, 42)
    gen  = _build_torch_generator(seed)

    # Decide calibration strategy
    cal_csv = d.get("cal_csv", None)
    cal_frac = float(d.get("cal_from_test_frac", 0.0) or 0.0)
    cal_exclude = bool(d.get("cal_exclude_from_test", False))

    dl_cal = None
    cal_indices: List[int] = []

    if (cal_csv is None) and (cal_frac > 0.0):
        labels_te = _labels_from_dataset(ds_te_full)
        cal_indices = _stratified_pick_indices(labels_te, cal_frac, seed)
        if len(cal_indices) > 0:
            ds_cal = Subset(ds_te_full, cal_indices)
            dl_cal = _make_loader(ds_cal, cfg, sampler=None, shuffle=False, tag="cal(test)", generator=gen)
            # Report cal counts
            labs_cal = labels_te[np.array(cal_indices, dtype=int)]
            cal_counts = np.bincount(labs_cal, minlength=2)
            cal_real, cal_fake = int(cal_counts[0]), int(cal_counts[1])
            _print_counts("cal(test)", cal_real, cal_fake)
            # record in cfg
            cfg.setdefault("data", {}).setdefault("split_counts", {})
            cfg["data"]["split_counts"]["cal"] = {"real": cal_real, "fake": cal_fake}
        else:
            print("[INFO][LibriSeVoc][    cal] cal_from_test_frac>0 but no indices were sampled (tiny test or extreme skew)")

    # Build TEST (optionally excluding CAL indices)
    if cal_exclude and cal_indices:
        all_idx = np.arange(len(ds_te_full))
        mask = np.ones_like(all_idx, dtype=bool)
        mask[np.array(cal_indices, dtype=int)] = False
        kept = all_idx[mask]
        ds_te = Subset(ds_te_full, kept.tolist())
        # Recompute counts for the kept subset
        labels_te = _labels_from_dataset(ds_te_full)
        kept_labs = labels_te[kept]
        kept_counts = np.bincount(kept_labs, minlength=2)
        te_real, te_fake = int(kept_counts[0]), int(kept_counts[1])
        _print_counts("test(kept)", te_real, te_fake)
    else:
        ds_te = ds_te_full
        # counts already printed

    dl_te = _make_loader(ds_te, cfg, sampler=None, shuffle=False, tag="test", generator=gen)

    # Store test counts
    cfg.setdefault("data", {}).setdefault("split_counts", {})
    cfg["data"]["split_counts"]["test"] = {"real": te_real, "fake": te_fake}

    return dl_te, dl_cal
