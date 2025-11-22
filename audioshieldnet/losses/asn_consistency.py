# audioShieldNet/asnet_6/audioshieldnet/losses/asn_consistency.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def _cosine_distance(u, v):
    u = F.normalize(u, dim=-1)
    v = F.normalize(v, dim=-1)
    return 1.0 - (u * v).sum(dim=-1)

class _LocalHead(nn.Module):
    def __init__(self, in_ch=1, hidden=8, out_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
        )
    def forward(self, x): return self.net(x)

class ASNConsistencyLoss(nn.Module):
    """
    Label-aware, band-wise, multi-scale STC loss + coherence.
    Inputs: A_map, P_map as [B,1,M,T], labels y in {0,1}.
    """
    def __init__(self, win_lengths=(80, 160), hop=40, margin=0.2, spoof_weight=0.5,
                 sr=16000, hop_length=256, tv_weight=0.0,
                 mel_sample=16, max_time_windows=64, seed=123):
        super().__init__()
        self.win_lengths = win_lengths
        self.hop = hop
        self.margin = margin
        self.spoof_weight = spoof_weight
        self.sr = sr; self.hop_length = hop_length
        self.tv_weight = tv_weight
        self.mel_sample = mel_sample
        self.max_time_windows = max_time_windows
        self._seed = int(seed)
        self.headA = _LocalHead(1,8,8)
        self.headP = _LocalHead(1,8,8)

    def _frames(self, T, w_frames, hop_frames):
        idx = []
        t = 0
        while t + w_frames <= T:
            idx.append((t, t + w_frames)); t += hop_frames
        if not idx: idx = [(0, T)]
        return idx

    def forward(self, A_map, P_map, y, H_phase=None):
        B, _, M, T = A_map.shape
        A_loc = self.headA(A_map)  # [B,C,M,T]
        P_loc = self.headP(P_map)  # [B,C,M,T]

        g = torch.Generator(device=A_map.device); g.manual_seed(self._seed); self._seed += 1
        if self.mel_sample and self.mel_sample < M:
            idx_m = torch.randperm(M, generator=g, device=A_map.device)[:self.mel_sample]
        else:
            idx_m = torch.arange(M, device=A_map.device)

        def ms_to_frames(ms):
            samples = int(self.sr * (ms / 1000.0))
            return max(3, samples // self.hop_length)
        hop_frames = ms_to_frames(self.hop)

        all_dists = []
        loss_real, loss_spoof, count = 0.0, 0.0, 0

        for w_ms in self.win_lengths:
            w_frames = ms_to_frames(w_ms)
            frames = self._frames(T, w_frames, hop_frames)
            if self.max_time_windows and len(frames) > self.max_time_windows:
                inds = torch.linspace(0, len(frames)-1, steps=self.max_time_windows).long()
                frames = [frames[i.item()] for i in inds]

            for m in idx_m.tolist():
                a_m = A_loc[:, :, m, :]
                p_m = P_loc[:, :, m, :]
                for (t0, t1) in frames:
                    u = a_m[:, :, t0:t1].flatten(1)
                    v = p_m[:, :, t0:t1].flatten(1)
                    d = _cosine_distance(u, v)
                    all_dists.append(d)
                    if (y == 0).any(): loss_real  = loss_real  + d[y == 0].mean()
                    if (y == 1).any(): loss_spoof = loss_spoof + F.relu(self.margin - d[y == 1]).mean()
                    count += 1

        if count == 0:
            return torch.tensor(0.0, device=A_map.device, requires_grad=True), torch.zeros(B, device=A_map.device)

        loss_real /= max(1, count)
        loss_spoof /= max(1, count)
        stc_loss = loss_real + self.spoof_weight * loss_spoof

        d_stack = torch.stack(all_dists, dim=0)
        coh_score = 1.0 - d_stack.mean(dim=0)

        tv = 0.0
        if self.tv_weight > 0.0 and H_phase is not None:
            tv = (H_phase[:, 1:, :] - H_phase[:, :-1, :]).abs().mean() * self.tv_weight

        return stc_loss + tv, coh_score
