# audioShieldNet/asnet_2/audioshieldnet/data/audioshield_dataset.py
# audioshield_dataset.py
import os, random
from functools import lru_cache
import torch, torchaudio, pandas as pd
from torch.utils.data import Dataset

@lru_cache(maxsize=4096)
def _cached_decode_and_resample(key: str) -> torch.Tensor:
    path, sr = key.rsplit("::", 1)
    sr = int(sr)
    # try soundfile (fast for flac), fallback to torchaudio
    try:
        import soundfile as sf
        data, r = sf.read(path, dtype="float32", always_2d=False)
        wav = torch.from_numpy(data if data.ndim == 1 else data.T).float()
        if wav.ndim == 1: wav = wav[None, :]
    except Exception:
        wav, r = torchaudio.load(path)  # [C,T]
    if r != sr:
        wav = torchaudio.functional.resample(wav, r, sr)
    if wav.size(0) > 1:                  # mono
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).contiguous()   # [T]

def _safe_load_audio(abs_path: str, sr: int) -> torch.Tensor:
    return _cached_decode_and_resample(f"{abs_path}::{sr}")

class AudioShieldDataset(Dataset):
    """
    Unified dataset for AudioShieldNet.
    Expects a CSV with columns:
      - filepath (abs or relative to root_dir)
      - label    (any of: real/bonafide/genuine/0 -> 0; fake/spoof/attack/1 -> 1)
    Returns: (wav[T], label[int], rel_path[str])
    """
    LABEL_MAP = {
        "real":0, "bonafide":0, "bona_fide":0, "genuine":0, "0":0, 0:0,
        "fake":1, "spoof":1, "attack":1, "1":1, 1:1
    }

    def __init__(self, csv_path: str, root_dir: str, sr: int, max_secs: float):
        self.df = pd.read_csv(csv_path)
        self.root_dir = os.path.abspath(root_dir)
        self.sr = int(sr)
        self.max_secs = float(max_secs)

        if "filepath" not in self.df.columns and "path" in self.df.columns:
            self.df.rename(columns={"path": "filepath"}, inplace=True)

        if "label" not in self.df.columns:
            raise ValueError(f"CSV must have 'label' column, got: {self.df.columns.tolist()}")

        # normalize columns
        self.df["filepath"] = self.df["filepath"].astype(str)
        def _map_lbl(x):
            s = str(x).lower()
            if s in self.LABEL_MAP: return self.LABEL_MAP[s]
            try: return int(x)
            except: raise ValueError(f"Unknown label: {x}")
        self.df["label"] = self.df["label"].map(_map_lbl).astype(int)

    def __len__(self): return len(self.df)

    def _abs_rel(self, p: str):
        p = os.path.normpath(p)
        if os.path.isabs(p):
            if p.startswith(self.root_dir):
                return p, os.path.relpath(p, self.root_dir)
            return p, os.path.basename(p)
        ap = os.path.join(self.root_dir, p)
        return ap, p

    def _crop_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        if self.max_secs <= 0: return wav
        max_len = int(self.max_secs * self.sr)
        T = wav.numel()
        if T == max_len: return wav
        if T < max_len:
            out = torch.zeros(max_len, dtype=wav.dtype)
            out[:T] = wav
            return out
        start = random.randint(0, T - max_len)
        return wav[start:start+max_len].contiguous()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        abs_p, rel_p = self._abs_rel(row["filepath"])
        wav = _safe_load_audio(abs_p, self.sr)
        wav = self._crop_or_pad(wav)
        return wav, int(row["label"]), rel_p
