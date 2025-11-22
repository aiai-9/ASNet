# audioShieldNet/asnet_6/audioshieldnet/models/asnnet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_blocks import EncoderLite
from torch.nn.utils import spectral_norm

class MelProjector(nn.Module):
    def __init__(self, n_mels: int, out_channels: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(1, 1, kernel_size=(5, 5), padding=(2, 2), bias=False)
        self.pw = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)
        self.collapse = nn.Conv2d(out_channels, out_channels, kernel_size=(n_mels, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                 # [B,1,M,T]
        y = torch.relu(self.dw(x))         # [B,1,M,T]
        y = self.pw(y)                     # [B,C,M,T]
        y = self.collapse(y).squeeze(2)    # [B,C,T]
        return y

class ASNNet(nn.Module):
    def __init__(self, n_mels: int = 80, cons_weight: float = 0.0,
                 enc_hidden: int = 128, emb_dim: int = 128, proj_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.amp_proj = MelProjector(n_mels, out_channels=proj_channels)
        self.phs_proj = MelProjector(n_mels, out_channels=proj_channels)
        self.amp_enc = EncoderLite(in_channels=proj_channels, hidden=enc_hidden, embed_dim=emb_dim)
        self.phs_enc = EncoderLite(in_channels=proj_channels, hidden=enc_hidden, embed_dim=emb_dim)
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(emb_dim * 2, 128)),
            nn.ReLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 1)),
        )
        self.cons_weight = cons_weight

    def forward(self, logmel: torch.Tensor, phmel: torch.Tensor, target: torch.Tensor | None = None):
        amp_seq = self.amp_proj(logmel)
        phs_seq = self.phs_proj(phmel)
        e_a = self.amp_enc(amp_seq)
        e_p = self.phs_enc(phs_seq)
        na = F.normalize(e_a, dim=-1); np_ = F.normalize(e_p, dim=-1)
        coherence = (na - np_).pow(2).sum(dim=-1).mean()
        logits = self.classifier(torch.cat([e_a, e_p], dim=-1)).squeeze(-1)

        aux = {
            "coherence": coherence,
            # >>> expose stream embeddings so trainer can run CMRA/InfoNCE
            "z_spec": e_a,     # amplitude stream embedding
            "z_pros": e_p,     # phase/prosodic stream embedding
            # EncoderLite doesn't expose temporal tokens; use None (trainer handles it)
            "tokA": None,
            "tokP": None,
        }

        if target is None:
            return logits, aux
        bce  = F.binary_cross_entropy_with_logits(logits, target.float())
        aux.update({"bce": bce, "loss": bce})
        return logits, aux
