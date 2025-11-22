# audioShieldNet/asnet_1/audioshieldnet/security/attacks.py

import torch
from torch.nn import functional as F
from audioshieldnet.models.asn_features import ASNFeatures

def _compute_features_grad(feats: ASNFeatures, wav):
    win = feats.window.to(wav.device)
    spec = torch.stft(wav, n_fft=feats.n_fft, hop_length=feats.hop, window=win, return_complex=True)
    mag = spec.abs() + 1e-8
    phase = torch.angle(spec)
    mel_mag = feats.mel(mag)
    logmel = torch.log(mel_mag + 1e-6)
    dphi = torch.diff(phase, dim=-1)
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
    mag_ = mag[..., 1:]
    ph_mel = feats.mel(mag_ * dphi.abs()) / (feats.mel(mag_) + 1e-8)
    ph_mel = F.pad(ph_mel, (1, 0))
    def _norm(x):
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sd = x.std(dim=(-2, -1), keepdim=True) + 1e-5
        return (x - mu) / sd
    return _norm(logmel), _norm(ph_mel)

def fgsm_attack(x, y, model, feats, loss_fn, eps: float):
    prev_mode = model.training
    model.train(True)
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        x_adv = x.detach().clone().requires_grad_(True)
        logmel, phmel = _compute_features_grad(feats, x_adv)
        logits, _ = model(logmel, phmel, target=None)
        loss = loss_fn(logits, y.float())
        loss.backward()
        x_adv = (x_adv + eps * x_adv.grad.sign()).clamp(-1.0, 1.0).detach()
    model.train(prev_mode)
    return x_adv

def pgd_attack(x, y, model, feats, loss_fn, eps: float, alpha: float, steps: int):
    prev_mode = model.training
    model.train(True)
    with torch.enable_grad():
        x0 = x.detach()
        x_adv = x0.clone()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logmel, phmel = _compute_features_grad(feats, x_adv)
            logits, _ = model(logmel, phmel, target=None)
            loss = loss_fn(logits, y.float())
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
                x_adv = x_adv.clamp(-1.0, 1.0)
            x_adv = x_adv.detach()
    model.train(prev_mode)
    return x_adv
