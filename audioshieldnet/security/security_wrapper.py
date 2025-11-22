# audioshieldnet/security/security_wrapper.py
import torch
from .trust import suspicious_flags
from .sanitize import bandpass, mp3_roundtrip

class SecureDetector(torch.nn.Module):
    def __init__(self, base_model, feats, temp_scaler=None, energy_thr= -3.0, abstain_band=(0.4, 0.6)):
        super().__init__()
        self.base = base_model
        self.feats = feats
        self.temp = temp_scaler
        self.energy_thr = float(energy_thr)
        self.abstain_band = tuple(abstain_band)

    @torch.no_grad()
    def forward(self, wav, sanitize=True, smooth=False, smooth_n=8, sigma=0.002):
        # wav: [B, T] waveform
        x = bandpass(wav) if sanitize else wav
        logmel, phmel = self.feats(x)
        logits, _ = self.base(logmel, phmel, target=None)
        probs, E, susp = suspicious_flags(logits, energy_thr=self.energy_thr, temp_scaler=self.temp, tau=self.abstain_band)
        # Optionally smooth by averaging noisy copies
        if smooth:
            scores = [probs]
            for _ in range(smooth_n):
                xn = (x + sigma * torch.randn_like(x)).clamp(-1.0, 1.0)
                lm, pm = self.feats(xn)
                logits_n, _ = self.base(lm, pm, target=None)
                scores.append(torch.sigmoid(logits_n).detach())
            probs = torch.stack(scores, dim=0).mean(dim=0)
        # decision: -1 abstain, 0 real, 1 fake
        decision = torch.where(susp, -1, (probs > 0.5).long())
        return {"logits": logits, "prob_fake": probs, "energy": E, "suspicious": susp, "decision": decision}
