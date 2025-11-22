# audioShieldNet/asnet_1/audioshieldnet/utils/augs.py

import torch, random
def add_white_noise(wav, snr_db):
    noise = torch.randn_like(wav)
    sig_p = wav.pow(2).mean(dim=-1, keepdim=True)
    noise_p = noise.pow(2).mean(dim=-1, keepdim=True) + 1e-8
    target_noise_p = sig_p / (10 ** (snr_db / 10.0))
    noise = noise * (target_noise_p.sqrt() / noise_p.sqrt())
    return (wav + noise).clamp(-1, 1)

def spec_augment(mel, T=30, p_t=0.3, F=8, p_f=0.3):
    B, M, Tlen = mel.shape
    if p_t > 0 and T > 0 and random.random() < p_t and Tlen > 1:
        t = random.randint(1, min(T, Tlen - 1))
        t0 = random.randint(0, Tlen - t)
        mel[:, :, t0:t0 + t] = 0
    if p_f > 0 and F > 0 and random.random() < p_f and M > 1:
        f = random.randint(1, min(F, M - 1))
        f0 = random.randint(0, M - f)
        mel[:, f0:f0 + f, :] = 0
    return mel
