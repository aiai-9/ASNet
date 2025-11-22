# audioShieldNet/asnet_1/audioshieldnet/security/pseudo_ood.py
import torch
import torch.nn.functional as F
import numpy as np

# optional (unused in this lightweight impl, but fine to keep)
try:
    import torchaudio  # noqa: F401
    import torchaudio.functional as AF  # noqa: F401
except Exception:
    torchaudio = None
    AF = None

from audioshieldnet.security.sanitize import mp3_roundtrip


# ----------------------------
# helpers
# ----------------------------
def _maybe_to_mono(x: torch.Tensor) -> torch.Tensor:
    """[B,T] or [B,1,T] -> [B,T]"""
    if x.dim() == 3 and x.size(1) == 1:
        return x[:, 0, :]
    return x


def _fir_lowpass_kernel(cutoff_hz: float, sr: int, width: int, device, dtype):
    """
    Tiny sinc low-pass kernel with Hamming window.
    width should be odd; we auto-fix if needed and also clamp for very short T.
    """
    width = int(width)
    if width < 3:
        width = 3
    if width % 2 == 0:
        width += 1
    t = torch.arange(-(width // 2), width // 2 + 1, device=device, dtype=dtype)
    fc = float(cutoff_hz) / float(sr)  # normalized cutoff (0..0.5)
    # sinc
    h = torch.where(t == 0, torch.tensor(2 * fc, device=device, dtype=dtype),
                    torch.sin(2 * torch.pi * fc * t) / (torch.pi * t))
    # hamming window
    w = 0.54 - 0.46 * torch.cos(2 * torch.pi * (torch.arange(h.numel(), device=device, dtype=dtype) / (h.numel() - 1)))
    h = h * w
    h = h / h.sum().clamp_min(1e-8)
    return h.view(1, 1, -1)


def _lowpass(x: torch.Tensor, sr: int, cutoff_hz: float, width: int = 101) -> torch.Tensor:
    if cutoff_hz is None:
        return x
    B, T = x.shape
    width = min(int(width), max(3, (T // 4) * 2 + 1))  # keep kernel modest vs T; ensure odd
    if width % 2 == 0:
        width += 1
    k = _fir_lowpass_kernel(cutoff_hz, sr, width, x.device, x.dtype)
    return F.conv1d(x.unsqueeze(1), k, padding=width // 2).squeeze(1)


def _highpass(x: torch.Tensor, sr: int, cutoff_hz: float, width: int = 101) -> torch.Tensor:
    if cutoff_hz is None:
        return x
    # highpass = x - lowpass(x, cutoff)
    return x - _lowpass(x, sr, cutoff_hz, width=width)


def _bandpass(x: torch.Tensor, sr: int, lo: float, hi: float, width: int = 101) -> torch.Tensor:
    """
    crude bandpass: lowpass(hi) then highpass(lo)
    """
    if (lo is None) and (hi is None):
        return x
    y = x
    if hi is not None:
        y = _lowpass(y, sr, hi, width=width)
    if lo is not None:
        y = _highpass(y, sr, lo, width=width)
    return y


def _bandstop(x: torch.Tensor, sr: int, lo: float, hi: float, width: int = 101) -> torch.Tensor:
    """
    crude bandstop: x - bandpass(x, lo, hi)
    knocks out a mid band
    """
    bp = _bandpass(x, sr, lo, hi, width=width)
    return x - bp


def _time_stretch(wav: torch.Tensor, rate: float = 0.9) -> torch.Tensor:
    """
    cheap linear-resample stretch, then crop/pad back to T
    """
    B, T = wav.shape
    new_len = max(1, int(round(T * float(rate))))
    stretched = F.interpolate(wav.unsqueeze(1), size=new_len, mode="linear", align_corners=False).squeeze(1)
    if stretched.size(1) >= T:
        return stretched[:, :T]
    return F.pad(stretched, (0, T - stretched.size(1)))


def _pitch_shift(wav: torch.Tensor, sr: int, n_semitones: int = 2) -> torch.Tensor:
    """
    approximate pitch shift: resample by r=2^(n/12) then resize back to T
    """
    B, T = wav.shape
    r = float(2 ** (n_semitones / 12.0))
    new_len = max(1, int(round(T / r)))
    shifted = F.interpolate(wav.unsqueeze(1), size=new_len, mode="linear", align_corners=False).squeeze(1)
    if shifted.size(1) >= T:
        return shifted[:, :T]
    return F.pad(shifted, (0, T - shifted.size(1)))


def _add_white_noise(wav: torch.Tensor, snr_db: float = 5.0) -> torch.Tensor:
    sig_p = (wav ** 2).mean(dim=1, keepdim=True).clamp_min(1e-8)
    snr = 10 ** (snr_db / 10.0)
    noise_p = sig_p / snr
    noise = torch.randn_like(wav) * noise_p.sqrt()
    return wav + noise


def _telephone_bandpass(wav: torch.Tensor, sr: int) -> torch.Tensor:
    # ~300–3400 Hz (classic telephone)
    return _bandpass(wav, sr, lo=300.0, hi=3400.0, width=101)


# ----------------------------
# registry
# ----------------------------
OODS = {
    "mp3":       lambda wav, sr, step: mp3_roundtrip(wav),
    "time":      lambda wav, sr, step: _time_stretch(wav, rate=0.9 if (step % 2 == 0) else 1.1),
    "pitch":     lambda wav, sr, step: _pitch_shift(wav, sr, n_semitones=(2 if (step % 2 == 0) else -2)),
    # knock out a mid band to induce spectral weirdness
    "bandstop":  lambda wav, sr, step: _bandstop(wav, sr, lo=300.0, hi=3000.0, width=101),
    # narrow telephone bandwidth
    "telephone": lambda wav, sr, step: _telephone_bandpass(wav, sr),
    # SNR schedule: slightly cleaner over very long training; clamp min SNR
    "noise":     lambda wav, sr, step: _add_white_noise(wav, snr_db=max(3.0, 8.0 - (step / 10000.0))),
}

# optional: keep older YAMLs working
ALIASES = {
    "telband": "telephone",
    "noise10": "noise",
}


class PseudoOODSampler:
    """
    Lightweight curriculum: start with a small set, expand transform types over steps.
    Keeps names consistent with your repo & YAML.
    """
    def __init__(self, sr: int = 16000, enabled_types=None, curriculum: bool = True):
        self.sr = int(sr)
        self.enabled = list(enabled_types) if enabled_types else list(OODS.keys())
        self.curriculum = bool(curriculum)

    def __call__(self, wav: torch.Tensor, step: int = 0):
        """
        wav: [B,T] float in [-1,1]; returns (wav_ood, type_name)
        """
        wav = _maybe_to_mono(wav)
        types = self.enabled
        if self.curriculum and len(types) > 1:
            # widen palette every ~1000 steps
            k = min(len(types), 1 + int(step // 1000))
            pool = types[:k]
        else:
            pool = types
        # resolve alias to canonical key if present
        choice = np.random.choice(pool)
        choice_canon = ALIASES.get(choice, choice)
        fn = OODS.get(choice_canon, OODS["mp3"])
        out = fn(wav, self.sr, int(step))
        out = _maybe_to_mono(out)
        return out, choice
