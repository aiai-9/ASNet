# audioshieldnet/security/sanitize.py
import torch
import torchaudio

def bandpass(x: torch.Tensor, sr: int = 16000, low: float = 50.0, high: float = 7600.0):
    # x: [B, T]
    out = []
    for i in range(x.shape[0]):
        xi = x[i:i+1]
        xi = torchaudio.functional.highpass_biquad(xi, sr, low)
        xi = torchaudio.functional.lowpass_biquad(xi, sr, high)
        out.append(xi)
    return torch.cat(out, dim=0)

def mp3_roundtrip(x: torch.Tensor, sr: int = 16000):
    # cheap codec proxy: resample down and up
    y = torchaudio.functional.resample(x, orig_freq=sr, new_freq=sr // 2)
    y = torchaudio.functional.resample(y, orig_freq=sr // 2, new_freq=sr)
    return y.clamp(-1.0, 1.0)
