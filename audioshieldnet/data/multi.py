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

# ============================================================
# Determinism helpers
# ============================================================
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
    """
    Keep existing mute behavior + deterministic RNGs per worker.
    """
    try:
        mute_worker(worker_id)
    except Exception:
        pass
    wseed = (int(base_seed) + int(worker_id)) % (2**32)
    np.random.seed(wseed)
    random.seed(wseed)


# ============================================================
# Label utilities (AudioShieldDataset / Subset / ConcatDataset)
# ============================================================
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
        non_empty = [p for p in parts if p.size > 0]
        return np.concatenate(non_empty) if non_empty else np.array([], dtype=int)

    # Unknown dataset type
    return np.array([], dtype=int)


def _split_counts(ds) -> Tuple[int, int]:
    """Returns (n_real, n_fake) for AudioShieldDataset / Subset / ConcatDataset safely."""
    labs = _labels_from_dataset(ds)
    if labs.size == 0:
        return (0, 0)
    cnts = np.bincount(labs, minlength=2)
    return int(cnts[0]), int(cnts[1])


# ============================================================
# Misc utilities
# ============================================================
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
        f"[INFO][{prefix}][{tag:>8}] total={total:6d}  "
        f"real={n_real:6d}  fake={n_fake:6d}  prior(fake)={prior_fake:.4f}"
    )


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


def _make_loader(
    dataset,
    cfg: Dict[str, Any],
    *,
    sampler=None,
    shuffle: bool = False,
    tag: str = "train",
    generator: Optional[torch.Generator] = None,
):
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
        generator=generator,
    )


# ============================================================
# BUILDERS
# ============================================================
def build_dataloaders(cfg):
    """
    Multi-dataset training loader.

    Expected YAML structure (example):

    data:
      name: multi
      sr: 16000
      max_secs: 6.0
      root_dir: /scratch/.../dataset_root   # default if a sub-dataset doesn't override

      sampler: weighted  # or "uniform"

      datasets:
        - id: lsv
          root_dir: /scratch/.../LibriSeVoc
          train_csv: /scratch/.../LibriSeVoc_train.csv
          val_csv:   /scratch/.../LibriSeVoc_val.csv
          test_csv:  /scratch/.../LibriSeVoc_test.csv

        - id: asv21
          root_dir: /scratch/.../ASVspoof2021
          train_csv: /scratch/.../ASVspoof21_train.csv
          val_csv:   /scratch/.../ASVspoof21_val.csv
          test_csv:  /scratch/.../ASVspoof21_test.csv

        - id: wavefake
          root_dir: /scratch/.../waveFake
          train_csv: /scratch/.../WaveFake_train.csv
          val_csv:   /scratch/.../WaveFake_val.csv
          test_csv:  /scratch/.../WaveFake_test.csv

    Returns:
        dl_tr, dl_va, dl_cal
    """
    d = cfg["data"]
    sr       = int(d.get("sr", 16000))
    max_secs = float(d.get("max_secs", 6.0))

    ds_specs = d.get("datasets", [])
    if not ds_specs:
        raise ValueError("data.name='multi' but no data.datasets list provided in config.")

    train_datasets: List[AudioShieldDataset] = []
    val_datasets:   List[AudioShieldDataset] = []

    # Build each sub-dataset and log its stats
    for spec in ds_specs:
        ds_id     = str(spec.get("id", "ds")).strip()
        root_dir  = spec.get("root_dir", d.get("root_dir", None))
        tr_csv    = spec.get("train_csv", spec.get("train", None))
        va_csv    = spec.get("val_csv",   spec.get("val", None))

        if root_dir is None or tr_csv is None or va_csv is None:
            raise ValueError(f"[multi] dataset entry {ds_id!r} missing root_dir/train_csv/val_csv.")

        ds_tr = AudioShieldDataset(tr_csv, root_dir, sr, max_secs, train_mode=True)
        ds_va = AudioShieldDataset(va_csv, root_dir, sr, max_secs, train_mode=False)

        tr_real, tr_fake = _split_counts(ds_tr)
        va_real, va_fake = _split_counts(ds_va)

        _print_counts(f"{ds_id}-train", tr_real, tr_fake, prefix="Multi")
        _print_counts(f"{ds_id}-val",   va_real, va_fake,  prefix="Multi")

        train_datasets.append(ds_tr)
        val_datasets.append(ds_va)

    # Concatenate all sub-datasets for joint training
    ds_tr_all = ConcatDataset(train_datasets)
    ds_va_all = ConcatDataset(val_datasets)
    ds_cal    = None   # optional global CAL from VAL/TEST if desired

    # Deterministic RNG for loader & sampler
    seed = _get_seed_from_cfg(cfg, 42)
    gen  = _build_torch_generator(seed)

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
    dl_tr  = _make_loader(ds_tr_all, cfg, sampler=sampler_tr, shuffle=shuffle_tr, tag="train", generator=gen)
    dl_va  = _make_loader(ds_va_all, cfg, sampler=None,        shuffle=False,      tag="val",   generator=gen)
    dl_cal = None  # you can add a global calibration CSV if you want

    # --- global counts & priors summary ---
    tr_real, tr_fake = _split_counts(ds_tr_all)
    va_real, va_fake = _split_counts(ds_va_all)

    _print_counts("train(all)", tr_real, tr_fake, prefix="Multi")
    _print_counts("val(all)",   va_real, va_fake, prefix="Multi")
    _print_counts("sum(all)",   tr_real + va_real, tr_fake + va_fake, prefix="Multi")

    # Store counts & priors for downstream use
    prior = {"real": tr_real, "fake": tr_fake}
    cfg.setdefault("data", {}).setdefault("prior_counts", prior)
    cfg["data"].setdefault("split_counts", {})
    cfg["data"]["split_counts"].update({
        "train": {"real": tr_real, "fake": tr_fake},
        "val":   {"real": va_real, "fake": va_fake},
    })

    return dl_tr, dl_va, dl_cal


def build_testloader(cfg):
    """
    Multi-dataset TEST builder.

    For each entry in data.datasets, if it has:
      - test_csv  (preferred), or
      - test      (alias),

    we build an AudioShieldDataset for TEST and then ConcatDataset over all
    available test splits.

    Optional global calibration-from-TEST:

      data.cal_from_test_frac: float in [0,1]  (0.0 = disabled)
      data.cal_exclude_from_test: bool

    Returns:
      dl_te, dl_cal  (dl_cal may be None)
    """
    d = cfg.get("data", {}) or {}
    sr       = int(d.get("sr", 16000))
    max_secs = float(d.get("max_secs", 6.0))

    ds_specs = d.get("datasets", [])
    if not ds_specs:
        print("[INFO][Multi][   test] no data.datasets list; returning (None, None).")
        return None, None

    test_datasets: List[AudioShieldDataset] = []

    # Build each available TEST split
    for spec in ds_specs:
        ds_id     = str(spec.get("id", "ds")).strip()
        root_dir  = spec.get("root_dir", d.get("root_dir", None))
        te_csv    = spec.get("test_csv", spec.get("test", None))

        if root_dir is None or te_csv is None:
            # silently skip datasets without TEST; training/VAL still work
            continue

        ds_te = AudioShieldDataset(te_csv, root_dir, sr, max_secs, train_mode=False)
        te_real, te_fake = _split_counts(ds_te)
        _print_counts(f"{ds_id}-test", te_real, te_fake, prefix="Multi")
        test_datasets.append(ds_te)

    if not test_datasets:
        print("[INFO][Multi][   test] no test_csv/test found in any dataset; returning (None, None).")
        return None, None

    ds_te_full = ConcatDataset(test_datasets)
    te_real, te_fake = _split_counts(ds_te_full)
    _print_counts("test(full)", te_real, te_fake, prefix="Multi")

    # Deterministic RNG shared by TEST/CAL loaders
    seed = _get_seed_from_cfg(cfg, 42)
    gen  = _build_torch_generator(seed)

    # Decide calibration strategy from TEST (global)
    cal_frac    = float(d.get("cal_from_test_frac", 0.0) or 0.0)
    cal_exclude = bool(d.get("cal_exclude_from_test", False))

    dl_cal: Optional[DataLoader] = None
    cal_indices: List[int] = []

    if cal_frac > 0.0:
        labels_te = _labels_from_dataset(ds_te_full)
        cal_indices = _stratified_pick_indices(labels_te, cal_frac, seed)
        if len(cal_indices) > 0:
            ds_cal = Subset(ds_te_full, cal_indices)
            dl_cal = _make_loader(ds_cal, cfg, sampler=None, shuffle=False, tag="cal(test)", generator=gen)

            labs_cal = labels_te[np.asarray(cal_indices, dtype=int)]
            cal_counts = np.bincount(labs_cal, minlength=2)
            cal_real, cal_fake = int(cal_counts[0]), int(cal_counts[1])
            _print_counts("cal(test)", cal_real, cal_fake, prefix="Multi")

            cfg.setdefault("data", {}).setdefault("split_counts", {})
            cfg["data"]["split_counts"]["cal"] = {"real": cal_real, "fake": cal_fake}
        else:
            print("[INFO][Multi][    cal] cal_from_test_frac>0 but no indices were sampled (tiny test or extreme skew)")

    # Build TEST (optionally excluding CAL indices)
    if cal_exclude and cal_indices:
        all_idx = np.arange(len(ds_te_full))
        mask = np.ones_like(all_idx, dtype=bool)
        mask[np.asarray(cal_indices, dtype=int)] = False
        kept = all_idx[mask]
        ds_te = Subset(ds_te_full, kept.tolist())

        labels_te_full = _labels_from_dataset(ds_te_full)
        kept_labs = labels_te_full[kept]
        kept_counts = np.bincount(kept_labs, minlength=2)
        te_real, te_fake = int(kept_counts[0]), int(kept_counts[1])
        _print_counts("test(kept)", te_real, te_fake, prefix="Multi")
    else:
        ds_te = ds_te_full
        # counts already printed as test(full)

    dl_te = _make_loader(ds_te, cfg, sampler=None, shuffle=False, tag="test", generator=gen)

    # Store test counts in cfg
    cfg.setdefault("data", {}).setdefault("split_counts", {})
    cfg["data"]["split_counts"]["test"] = {"real": te_real, "fake": te_fake}

    return dl_te, dl_cal
