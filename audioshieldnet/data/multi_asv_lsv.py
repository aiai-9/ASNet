# audioShieldNet/asnet_6/audioshieldnet/data/multi.py

import random
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
    Subset,
    ConcatDataset,
)

from audioshieldnet.data.audioshield_dataset import (
    AudioShieldDataset,
    mute_worker,
    compute_class_weights,
)


# =========================
# Determinism helpers
# =========================
def _get_seed_from_cfg(cfg: Dict[str, Any], default: int = 42) -> int:
    try:
        return int(((cfg.get("train", {}) or {}).get("seed", default)))
    except Exception:
        return int(default)


def _build_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _worker_init_seeded(worker_id: int, base_seed: int):
    # Keep existing mute behavior + deterministic RNGs
    try:
        mute_worker(worker_id)
    except Exception:
        pass
    wseed = (int(base_seed) + int(worker_id)) % (2**32)
    np.random.seed(wseed)
    random.seed(wseed)


# =========================
# Label utilities
# =========================
def _labels_from_dataset(ds) -> np.ndarray:
    """
    Return labels as np.int array for:
      - AudioShieldDataset (has .df)
      - Subset(AudioShieldDataset or nested)
      - ConcatDataset([...]) of any of the above
    Falls back to empty array if unknown dataset type.
    """
    if ds is None:
        return np.array([], dtype=int)

    # Direct AudioShieldDataset with .df
    if hasattr(ds, "df"):
        return ds.df["label"].to_numpy(dtype=int)

    from torch.utils.data import Subset as TorchSubset
    from torch.utils.data import ConcatDataset as TorchConcat

    # Subset wrapping another dataset (possibly nested)
    if isinstance(ds, TorchSubset):
        base = ds.dataset
        idx = np.asarray(ds.indices, dtype=int)
        base_labels = _labels_from_dataset(base)
        if base_labels.size == 0:
            return np.array([], dtype=int)
        return base_labels[idx]

    # ConcatDataset – concatenate labels from all children
    if isinstance(ds, TorchConcat):
        parts = [_labels_from_dataset(child) for child in ds.datasets]
        if not parts:
            return np.array([], dtype=int)
        parts = [p for p in parts if p.size > 0]
        if not parts:
            return np.array([], dtype=int)
        return np.concatenate(parts)

    # Unknown dataset type
    return np.array([], dtype=int)


def _split_counts(ds) -> Tuple[int, int]:
    """Returns (n_real, n_fake) for AudioShieldDataset / Subset / ConcatDataset safely."""
    labs = _labels_from_dataset(ds)
    if labs.size == 0:
        return (0, 0)
    cnts = np.bincount(labs, minlength=2)
    return int(cnts[0]), int(cnts[1])


# =========================
# Misc utilities
# =========================
def _print_counts(
    tag: str,
    n_real: int,
    n_fake: int,
    n_total: Optional[int] = None,
    prefix: str = "Multi",
):
    total = (n_total if n_total is not None else (n_real + n_fake))
    prior_fake = (n_fake / total) if total > 0 else 0.0
    print(
        f"[INFO][{prefix}][{tag:>12}] total={total:6d}  "
        f"real={n_real:6d}  fake={n_fake:6d}  prior(fake)={prior_fake:.4f}"
    )


def _make_loader(
    dataset,
    cfg,
    *,
    sampler=None,
    shuffle=False,
    tag="train",
    generator: Optional[torch.Generator] = None,
):
    tcfg = cfg.get("train", {}) or {}
    bs = int(tcfg.get("batch_size", 32))
    nw = int(tcfg.get("num_workers", 16))
    pf = int(tcfg.get("prefetch_factor", 8))
    pw = bool(tcfg.get("persistent_workers", True))
    pin = bool(tcfg.get("pin_memory", True))

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
        generator=generator,
    )


# =========================
# Core builders
# =========================
def _extract_csv_paths(spec: Dict[str, Any], global_root: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Helper to flexibly support:
      - train_csv / val_csv / test_csv
      - OR train / val / test
    Returns dict with keys: root_dir, train_csv, val_csv, test_csv (any may be None).
    """
    root_dir = spec.get("root_dir", global_root)

    # allow both naming styles
    train_csv = spec.get("train_csv") or spec.get("train")
    val_csv   = spec.get("val_csv") or spec.get("val")
    test_csv  = spec.get("test_csv") or spec.get("test")

    return {
        "root_dir": root_dir,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
    }


def build_dataloaders(cfg):
    """
    Multi-dataset training loader.

    Expected config structure (example):

    data:
      name: multi
      sr: 16000
      n_fft: 1024
      hop: 256
      n_mels: 80
      max_secs: 6.0

      # default root_dir if a sub-dataset doesn't override it
      root_dir: /scratch/.../dataset_root

      # how to sample:
      #   'uniform'  → shuffle over concatenated dataset
      #   'weighted' → class-balanced sampling via compute_class_weights
      sampler: weighted

      datasets:
        - id: lsv
          root_dir: /.../LibriSeVoc
          train_csv: /.../LibriSeVoc_train.csv
          val_csv:   /.../LibriSeVoc_val.csv

        - id: asv21
          root_dir: /.../ASVspoof2021
          train_csv: /.../ASVspoof21_train.csv
          val_csv:   /.../ASVspoof21_val.csv

        - id: for
          root_dir: /.../FOR
          train: /.../FOR_..._train.csv     # 'train' also accepted
          val:   /.../FOR_..._val.csv       # 'val' also accepted

    Returns:
        dl_tr, dl_va, dl_cal  (dl_cal is None by default for multi)
    """
    d = cfg["data"]
    sr = int(d.get("sr", 16000))
    max_secs = float(d.get("max_secs", 6.0))

    global_root = d.get("root_dir", None)
    ds_specs = d.get("datasets", [])
    if not ds_specs:
        raise ValueError("data.name='multi' but no data.datasets list provided in config.")

    train_datasets = []
    val_datasets = []

    # Build each sub-dataset from its csv paths
    for spec in ds_specs:
        ds_id = str(spec.get("id", "ds")).strip()
        paths = _extract_csv_paths(spec, global_root)

        root_dir = paths["root_dir"]
        tr_csv = paths["train_csv"]
        va_csv = paths["val_csv"]

        if root_dir is None or tr_csv is None or va_csv is None:
            raise ValueError(
                f"[multi] dataset entry {ds_id!r} missing root_dir/train/val paths. "
                f"Got root_dir={root_dir}, train={tr_csv}, val={va_csv}"
            )

        ds_tr = AudioShieldDataset(tr_csv, root_dir, sr, max_secs, train_mode=True)
        ds_va = AudioShieldDataset(va_csv, root_dir, sr, max_secs, train_mode=False)

        tr_real, tr_fake = _split_counts(ds_tr)
        va_real, va_fake = _split_counts(ds_va)

        _print_counts(f"{ds_id}-train", tr_real, tr_fake, prefix="Multi")
        _print_counts(f"{ds_id}-val", va_real, va_fake, prefix="Multi")

        train_datasets.append(ds_tr)
        val_datasets.append(ds_va)

    # Concatenate all sub-datasets for joint training
    ds_tr_all = ConcatDataset(train_datasets)
    ds_va_all = ConcatDataset(val_datasets)
    ds_cal = None  # optional global calibration, not used here

    # Deterministic RNG for loader & sampler
    seed = _get_seed_from_cfg(cfg, 42)
    gen = _build_torch_generator(seed)

    # --- sampler / weights for TRAIN only ---
    labels_tr = _labels_from_dataset(ds_tr_all)
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
            generator=gen,
        )
        shuffle_tr = False
    else:
        sampler_tr = None
        shuffle_tr = True

    # --- loaders ---
    dl_tr = _make_loader(
        ds_tr_all,
        cfg,
        sampler=sampler_tr,
        shuffle=shuffle_tr,
        tag="train",
        generator=gen,
    )
    dl_va = _make_loader(
        ds_va_all,
        cfg,
        sampler=None,
        shuffle=False,
        tag="val",
        generator=gen,
    )
    dl_cal = None

    # --- global counts & priors summary ---
    tr_real, tr_fake = _split_counts(ds_tr_all)
    va_real, va_fake = _split_counts(ds_va_all)

    _print_counts("train(all)", tr_real, tr_fake, prefix="Multi")
    _print_counts("val(all)", va_real, va_fake, prefix="Multi")
    _print_counts("sum(all)", tr_real + va_real, tr_fake + va_fake, prefix="Multi")

    # Store counts & priors for downstream use (loss, bias init, logs)
    prior = {"real": tr_real, "fake": tr_fake}
    cfg.setdefault("data", {}).setdefault("prior_counts", prior)
    cfg["data"].setdefault("split_counts", {})
    cfg["data"]["split_counts"].update(
        {
            "train": {"real": tr_real, "fake": tr_fake},
            "val": {"real": va_real, "fake": va_fake},
        }
    )

    return dl_tr, dl_va, dl_cal


def build_testloader(cfg):
    """
    Optional: build a joint TEST loader over all sub-datasets that provide a test path.
    Uses the same 'datasets' block as build_dataloaders.

    Returns:
        dl_te, dl_cal  (dl_cal is always None for multi here)
    """
    d = cfg["data"]
    sr = int(d.get("sr", 16000))
    max_secs = float(d.get("max_secs", 6.0))
    global_root = d.get("root_dir", None)
    ds_specs = d.get("datasets", [])
    if not ds_specs:
        print("[INFO][Multi][   test] no data.datasets provided – returning (None, None)")
        return None, None

    test_datasets = []
    for spec in ds_specs:
        ds_id = str(spec.get("id", "ds")).strip()
        paths = _extract_csv_paths(spec, global_root)
        root_dir = paths["root_dir"]
        te_csv = paths["test_csv"]

        if root_dir is None or te_csv is None:
            # silently skip datasets without test split
            print(f"[INFO][Multi][{ds_id}-test] skipped (no root_dir or test path).")
            continue

        ds_te = AudioShieldDataset(te_csv, root_dir, sr, max_secs, train_mode=False)
        te_real, te_fake = _split_counts(ds_te)
        _print_counts(f"{ds_id}-test", te_real, te_fake, prefix="Multi")
        test_datasets.append(ds_te)

    if not test_datasets:
        print("[INFO][Multi][   test] no sub-datasets with test_csv/test – returning (None, None)")
        return None, None

    ds_te_all = ConcatDataset(test_datasets)
    te_real, te_fake = _split_counts(ds_te_all)
    _print_counts("test(all)", te_real, te_fake, prefix="Multi")

    seed = _get_seed_from_cfg(cfg, 42)
    gen = _build_torch_generator(seed)

    dl_te = _make_loader(
        ds_te_all,
        cfg,
        sampler=None,
        shuffle=False,
        tag="test",
        generator=gen,
    )
    dl_cal = None

    cfg.setdefault("data", {}).setdefault("split_counts", {})
    cfg["data"]["split_counts"]["test"] = {"real": te_real, "fake": te_fake}

    return dl_te, dl_cal
