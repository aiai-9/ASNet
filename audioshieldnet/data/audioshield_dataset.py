# audioShieldNet/asnet_2/audioshieldnet/data/audioshield_dataset.py


import os
import random
import warnings
import re
from functools import lru_cache
from typing import Tuple, Optional, List

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import importlib
from typing import Callable, Tuple, Optional, Dict, Any

# Map cfg.data.name -> module path that provides build_dataloaders/build_testloader
_DATA_MODULES = {
    "asvspoof21_split": "audioshieldnet.data.asvspoof21_split",
    "librisevoc_split": "audioshieldnet.data.librisevoc_split",
    # add more datasets here as you create them
}

def _load_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise RuntimeError(f"[data] Could not import module '{name}'.") from e

def resolve_data_builders(cfg: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Returns (build_dataloaders, build_testloader) callables based on cfg['data']['name'].
    """
    d = (cfg.get("data") or {})
    data_name = str(d.get("name", "")).strip().lower()
    if not data_name:
        raise RuntimeError("[data] cfg.data.name is missing/empty; cannot resolve data builders.")

    module_path = _DATA_MODULES.get(data_name)
    if module_path is None:
        # Helpful hint: show known keys
        known = ", ".join(sorted(_DATA_MODULES.keys()))
        raise RuntimeError(f"[data] Unknown dataset name '{data_name}'. Known: {known}")

    mod = _load_module(module_path)

    try:
        build_dataloaders = getattr(mod, "build_dataloaders")
        build_testloader  = getattr(mod, "build_testloader")
    except AttributeError:
        raise RuntimeError(f"[data] Module '{module_path}' must define build_dataloaders and build_testloader")

    return build_dataloaders, build_testloader


__all__ = [
    "AudioShieldDataset",
    "read_index_table",
    "mute_worker",
    "make_dataloader",
    "compute_class_weights",
]




# ----------------------
# Shared worker init (used by all loaders)
# ----------------------
def mute_worker(_):
    """Silence noisy third-party warnings inside DataLoader workers."""
    # Silence harmless Pydantic v2 warnings about Field() attributes
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", message="StreamingMediaDecoder has been deprecated")


    warnings.filterwarnings(
        "ignore",
        message="The 'repr' attribute with value",
        category=UserWarning,
        module="pydantic._internal._generate_schema",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'frozen' attribute with value",
        category=UserWarning,
        module="pydantic._internal._generate_schema",
    )

    
    try:
        from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
        warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
    except Exception:
        warnings.filterwarnings("ignore", category=UserWarning,
                                module="pydantic._internal._generate_schema")

    warnings.filterwarnings("ignore",
                            message=r"The 'repr' attribute .* Field\(\) .* no effect")
    warnings.filterwarnings("ignore",
                            message=r"The 'frozen' attribute .* Field\(\) .* no effect")
    

# ----------------------
# Small shared utilities
# ----------------------
def compute_class_weights(labels_np):
    """Return (class_counts, sample_weights) for WeightedRandomSampler."""
    import numpy as np
    class_counts = np.bincount(labels_np, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels_np]
    return class_counts, sample_weights



def make_dataloader(dataset,
                    cfg: dict,
                    *,
                    sampler=None,
                    shuffle: bool = False,
                    tag: str = "train"):
    """
    Build a DataLoader with cfg.train knobs and the shared mute_worker().
    """
    tcfg = cfg.get("train", {}) or {}
    bs   = int(tcfg.get("batch_size", 32))
    nw   = int(tcfg.get("num_workers", 16))
    pf   = int(tcfg.get("prefetch_factor", 4))
    pw   = bool(tcfg.get("persistent_workers", True))
    pin  = bool(tcfg.get("pin_memory", False))  # default False like your current code

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=pf,
        worker_init_fn=mute_worker,
    )

# ----------------------
# Helpers
# ----------------------

def _norm_seconds(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


LABEL_MAP = {
    "real": 0, "bonafide": 0, "bona_fide": 0, "genuine": 0, "0": 0, 0: 0,
    "fake": 1, "spoof": 1, "attack": 1, "1": 1, 1: 1,
}

def _map_label(v) -> int:
    s = str(v).strip().lower()
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    # Try int
    try:
        return 1 if int(s) == 1 else 0
    except Exception:
        raise ValueError(f"Unknown label value: {v!r} (expected one of {sorted(set(LABEL_MAP.keys()))})")


def _infer_label_from_path(path_like: str) -> Optional[int]:
    """
    Best-effort inference if no explicit label in a .list line:
      - if the path contains 'bonafide|bona_fide|genuine' => 0
      - if the path contains 'spoof|fake|attack' => 1
    Returns None if we can't decide.
    """
    p = str(path_like).lower()
    if re.search(r"(bonafide|bona[_-]?fide|genuine)", p):
        return 0
    if re.search(r"(spoof|fake|attack)", p):
        return 1
    return None


# --------- Fast, cached decode + resample (per-process LRU) ----------
@lru_cache(maxsize=64)
def _cached_decode_and_resample(key: str) -> torch.Tensor:
    """
    key = "<abs_path>::<target_sr>"
    Returns mono waveform [T] float32 in [-1, 1]
    """
    path, sr = key.rsplit("::", 1)
    sr = int(sr)

    # Try soundfile (fast for FLAC) then fallback to torchaudio
    try:
        import soundfile as sf
        data, r = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 1:
            wav = torch.from_numpy(data)[None, :]  # [1,T]
        else:
            # [T,C] -> [C,T]
            wav = torch.from_numpy(data.T).contiguous()
    except Exception as e_sf:
        try:
            wav, r = torchaudio.load(path)  # [C,T], dtype float32 or int
            if not torch.is_floating_point(wav):
                wav = wav.float() / max(1.0, float(torch.iinfo(wav.dtype).max))
        except Exception as e_ta:
            raise RuntimeError(
                f"Audio decode failed for: {path}\n"
                f"soundfile error: {e_sf}\n"
                f"torchaudio error: {e_ta}"
            )

    # Resample if needed
    if r != sr:
        wav = torchaudio.functional.resample(wav, r, sr)
    # Mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).contiguous()  # [T]


def _safe_load_audio(abs_path: str, sr: int) -> torch.Tensor:
    """Wrapper that hits the LRU cache and returns [T] float32 in [-1, 1]."""
    return _cached_decode_and_resample(f"{abs_path}::{sr}")


def _center_crop(wav: torch.Tensor, max_len: int) -> torch.Tensor:
    T = wav.numel()
    if T <= max_len:
        out = torch.zeros(max_len, dtype=wav.dtype)
        out[:T] = wav
        return out
    start = (T - max_len) // 2
    return wav[start:start + max_len].contiguous()


def _random_crop(wav: torch.Tensor, max_len: int) -> torch.Tensor:
    T = wav.numel()
    if T <= max_len:
        out = torch.zeros(max_len, dtype=wav.dtype)
        out[:T] = wav
        return out
    start = random.randint(0, T - max_len)
    return wav[start:start + max_len].contiguous()


# ----------------------
# Index readers (CSV/TSV/LIST)
# ----------------------

def _read_csv_like(path: str) -> pd.DataFrame:
    path = os.path.abspath(path)
    sep = "," if path.lower().endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    if "filepath" not in cols and "path" in cols:
        df = df.rename(columns={cols["path"]: "filepath"})
        cols["filepath"] = "filepath"
    if "filepath" not in df.columns:
        raise ValueError(f"{path}: missing 'filepath' column (found: {df.columns.tolist()})")
    if "label" not in df.columns:
        raise ValueError(f"{path}: missing 'label' column (found: {df.columns.tolist()})")
    df["filepath"] = df["filepath"].astype(str)
    df["label"] = df["label"].apply(_map_label).astype(int)
    return df[["filepath", "label"]].copy()


def _read_list_like(path: str) -> pd.DataFrame:
    """
    Support common .list formats used by LibriSeVoc/others:
      - "rel/or/abs/path.wav,label"
      - "rel/or/abs/path.wav label"
      - "rel/or/abs/path.wav\tlabel"
      - "rel/or/abs/path.wav"  (label inferred from path if possible)
      - "label rel/or/abs/path.wav" (rare; we detect if the first token is 0/1)
    """
    rows: List[Tuple[str, int]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # CSV-ish
            if "," in line and "\t" not in line and " " not in line:
                p, lab = line.split(",", 1)
                rows.append((p.strip(), _map_label(lab)))
                continue

            # Tokenized by whitespace or tabs
            toks = re.split(r"[\t ]+", line)
            toks = [t for t in toks if t]
            if len(toks) == 1:
                p = toks[0]
                lab = _infer_label_from_path(p)
                if lab is None:
                    raise ValueError(
                        f"{path}: one-column list but label is not inferable from path='{p}'. "
                        "Provide labels explicitly (two columns) or rename dirs to include "
                        "'bonafide/genuine' or 'spoof/fake/attack'."
                    )
                rows.append((p, lab))
            elif len(toks) >= 2:
                # Decide if format is: (file,label) or (label,file)
                left, right = toks[0], toks[1]
                # If left looks like a label
                if re.fullmatch(r"[01]|real|bonafide|bona[_-]?fide|genuine|fake|spoof|attack", left.lower()):
                    lab = _map_label(left)
                    p = right
                else:
                    p = left
                    lab = _map_label(right)
                rows.append((p, lab))
            else:
                continue

    df = pd.DataFrame(rows, columns=["filepath", "label"])
    df["filepath"] = df["filepath"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def read_index_table(index_path: str) -> pd.DataFrame:
    """
    Auto-detect index format from path extension:
      - .csv/.tsv -> pandas reader
      - .list/.lst/.txt -> flexible line parser
    """
    index_path = os.path.abspath(index_path)
    ext = os.path.splitext(index_path)[1].lower()
    if ext in [".csv", ".tsv"]:
        return _read_csv_like(index_path)
    if ext in [".list", ".lst", ".txt"]:
        return _read_list_like(index_path)
    # Fallback: try CSV then LIST
    try:
        return _read_csv_like(index_path)
    except Exception:
        return _read_list_like(index_path)


# ----------------------
# Dataset
# ----------------------

class AudioShieldDataset(Dataset):
    """
    Generic dataset for audio deepfake/ASVspoof-style binary classification.

    Accepts an "index file" (CSV/TSV/LIST) with at least:
      - 'filepath' (relative to root_dir or absolute)
      - 'label'    (0/1 or {real/bonafide/genuine, fake/spoof/attack})

    Features:
      - soundfile→torchaudio robust I/O (+LRU cache)
      - resampling + mono
      - train (random crop) vs eval (center crop)
      - corrupt file handling via `fail_policy`: "zero" | "skip" | "raise"
      - counters for skipped/zeroed files: `bad_count`, `skipped_count`

    Returns: (wav[T], label[int], rel_path[str])
    """

    def __init__(
        self,
        index_path: str,
        root_dir: str,
        sr: int,
        max_secs: float,
        *,
        train_mode: bool = True,
        center_crop_eval: bool = True,
        fail_policy: str = "zero",    # "zero" | "skip" | "raise"
        allowed_exts: Optional[Tuple[str, ...]] = (".wav", ".flac", ".mp3", ".m4a"),
        # optional: filter by extension for list files that include non-audio entries
    ):
        super().__init__()
        self.index_path = os.path.abspath(index_path)
        self.root_dir = os.path.abspath(root_dir)
        self.sr = int(sr)
        self.max_secs = _norm_seconds(max_secs)
        self.train_mode = bool(train_mode)
        self.center_crop_eval = bool(center_crop_eval)
        self.fail_policy = str(fail_policy).lower()
        self.allowed_exts = tuple(allowed_exts) if allowed_exts else None

        # Per-dataset counters
        self.bad_count = 0       # decode failures → zeroed or skipped
        self.skipped_count = 0   # if fail_policy == "skip"

        # Read + normalize the index table
        df = read_index_table(self.index_path)

        # Filter by extension if requested
        if self.allowed_exts:
            def _keep(p: str) -> bool:
                _, e = os.path.splitext(str(p))
                return e.lower() in self.allowed_exts
            keep = df["filepath"].astype(str).apply(_keep)
            dropped = int((~keep).sum())
            if dropped > 0:
                print(f"[DATA] Dropped {dropped} rows due to extension filter {self.allowed_exts}")
            df = df[keep].reset_index(drop=True)

        # Store normalized paths
        df["filepath"] = df["filepath"].astype(str)
        self.df = df[["filepath", "label"]].reset_index(drop=True)

        # Build a list of indices if we might skip items
        self._valid_idx_cache = None  # lazily built if fail_policy == "skip"

    def __len__(self):
        if self.fail_policy == "skip" and self._valid_idx_cache is not None:
            return len(self._valid_idx_cache)
        return len(self.df)

    def _rel_abs(self, rel_or_abs: str) -> Tuple[str, str]:
        """
        Returns (abs_path, rel_path-under-root-if-possible).
        """
        p = os.path.normpath(rel_or_abs)
        if os.path.isabs(p):
            if p.startswith(self.root_dir):
                return p, os.path.relpath(p, self.root_dir)
            return p, os.path.basename(p)
        return os.path.join(self.root_dir, p), p

    def _crop(self, wav: torch.Tensor) -> torch.Tensor:
        if self.max_secs <= 0:
            return wav.contiguous()
        max_len = int(self.max_secs * self.sr)
        if self.train_mode:
            return _random_crop(wav, max_len)
        if self.center_crop_eval:
            return _center_crop(wav, max_len)
        # fallback: random even in eval if center disabled
        return _random_crop(wav, max_len)

    def _get_row(self, raw_idx: int) -> Tuple[str, int]:
        row = self.df.iloc[raw_idx]
        return str(row["filepath"]), int(row["label"])

    def __getitem__(self, idx):
        # If we are skipping failed items, map visible idx -> real df index
        raw_idx = idx
        if self.fail_policy == "skip" and self._valid_idx_cache is not None:
            raw_idx = self._valid_idx_cache[idx]

        rel_path, lab = self._get_row(raw_idx)
        abs_path, rel = self._rel_abs(rel_path)

        try:
            wav = _safe_load_audio(abs_path, self.sr)  # [T]
            wav = self._crop(wav)
            return wav, lab, rel
        except Exception as e:
            # Handle failures per policy
            self.bad_count += 1
            msg = f"[DATA] Failed to read '{abs_path}' ({type(e).__name__}: {e})"
            if self.fail_policy == "raise":
                raise RuntimeError(msg)

            if self.fail_policy == "skip":
                # Build cache on first failure, if needed
                if self._valid_idx_cache is None:
                    self._valid_idx_cache = list(range(len(self.df)))
                # Remove this raw_idx if present
                if raw_idx in self._valid_idx_cache:
                    self._valid_idx_cache.remove(raw_idx)
                self.skipped_count += 1
                # Pick another sample (simple fallback: wrap around)
                new_len = len(self._valid_idx_cache)
                if new_len == 0:
                    # Nothing left—return a zero tensor to avoid hard crash
                    z = torch.zeros(int(self.max_secs * self.sr) if self.max_secs > 0 else self.sr, dtype=torch.float32)
                    return z, lab, rel
                return self.__getitem__(idx % new_len)

            # fail_policy == "zero": return a zero-padded segment
            T = int(self.max_secs * self.sr) if self.max_secs > 0 else self.sr
            z = torch.zeros(max(1, T), dtype=torch.float32)
            return z, lab, rel
