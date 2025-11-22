# audioShieldNet/asnet_6/audioshieldnet/models/asn_features.py

import torch
import torch.nn.functional as F
import torchaudio

class ASNFeatures(torch.nn.Module):
    """
    Extracts:
      - log-Mel amplitude
      - Mel-mapped phase-derivative
    Returns [B, M, T] tensors, per-utterance normalized.
    """
    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop: int = 256, n_mels: int = 80):
        super().__init__()
        self.sr, self.n_fft, self.hop, self.n_mels = sr, n_fft, hop, n_mels
        self.window = torch.hann_window(n_fft)
        self.mel = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=sr, n_stft=n_fft // 2 + 1
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor):
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        win = self.window.to(wav.device)
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop, window=win, return_complex=True)  # [B,F,T]
        mag   = spec.abs() + 1e-8
        phase = torch.angle(spec)
        mel_mag = self.mel(mag)
        logmel  = torch.log(mel_mag + 1e-6)

        dphi = torch.diff(phase, dim=-1)
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        mag_ = mag[..., 1:]
        ph_mel = self.mel(mag_ * dphi.abs()) / (self.mel(mag_) + 1e-8)
        ph_mel = F.pad(ph_mel, (1, 0))

        def norm(x):
            mu = x.mean(dim=(-2, -1), keepdim=True)
            sd = x.std(dim=(-2, -1), keepdim=True) + 1e-5
            return (x - mu) / sd

        return norm(logmel), norm(ph_mel)
