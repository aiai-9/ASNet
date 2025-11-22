# audioShieldNet/asnet_5/audioshieldnet/models/asnnet_secdual.py
# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F

# ----------------------------
# Gradient Reversal (for GRL)
# ----------------------------
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): 
        ctx.lam = float(lam)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g): 
        return -ctx.lam * g, None

# ----------------------------
# Cross-Attention (token-to-token)
# ----------------------------
class CrossAttn(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.h = heads

    def forward(self, xa, xb):  # xa queries, xb keys/values
        # xa: [B, Ta, D], xb: [B, Tb, D]
        B, Ta, D = xa.shape
        Tb = xb.shape[1]
        h = self.h
        d = D // h
        q = self.q(xa).view(B, Ta, h, d).transpose(1, 2)      # [B,h,Ta,d]
        k = self.k(xb).view(B, Tb, h, d).transpose(1, 2)      # [B,h,Tb,d]
        v = self.v(xb).view(B, Tb, h, d).transpose(1, 2)      # [B,h,Tb,d]
        att = (q @ k.transpose(-1, -2)) / (d ** 0.5)          # [B,h,Ta,Tb]
        w = att.softmax(dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, Ta, D)  # [B,Ta,D]
        return self.proj(out)                                  # [B,Ta,D]

# ----------------------------
# Attention Pool over time
# ----------------------------
class AttnPool1D(nn.Module):
    """
    Input:  x [B, T, C]
    Output: z [B, C], tokens [B, T, C] (for cross-attn)
    """
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_heads, dim))   # H learnable queries
        nn.init.normal_(self.q, std=0.02)
        self.key  = nn.Linear(dim, dim, bias=False)
        self.val  = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        H = self.q.shape[0]
        k = self.key(x)                                      # [B,T,C]
        v = self.val(x)                                      # [B,T,C]
        q = self.q.unsqueeze(0).expand(B, H, C)              # [B,H,C]
        att = torch.einsum("bhc,btc->bht", q, k) / (C ** 0.5)
        w = att.softmax(dim=-1)                              # [B,H,T]
        pooled = torch.einsum("bht,btc->bhc", w, v)          # [B,H,C]
        pooled = pooled.mean(dim=1)                          # [B,C]
        return self.proj(pooled), x                          # z, tokens

# ----------------------------
# TCN residual block
# ----------------------------
class TCNBlock(nn.Module):
    """
    Depthwise-dilated temporal conv + pointwise conv, with residual.
    Input:  [B, C, T] -> Output: [B, C, T]
    """
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = dilation
        self.dw   = nn.Conv1d(channels, channels, kernel_size=3, padding=pad,
                              dilation=dilation, groups=channels, bias=False)
        self.pw   = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm1d(channels)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        return y + residual

# ----------------------------
# Encoder: Mel projector -> TCN stack -> AttnPool
# ----------------------------
class EncoderTCN1D(nn.Module):
    """
    Treat mel bins as input channels, do 1x1 (mel->C) projection, TCN over time,
    then attention pooling over time to get a fixed embedding.
    """
    def __init__(self, n_mels: int, hidden_ch: int = 128, emb: int = 128,
                 tcn_layers: int = 4, tcn_dilations=None, dropout: float = 0.2, attn_heads: int = 1):
        super().__init__()
        if tcn_dilations is None:
            tcn_dilations = [1, 2, 4, 8]
        assert len(tcn_dilations) == tcn_layers, "tcn_layers must match len(tcn_dilations)"

        # mel projection (learned instead of mean collapse)
        self.mel_proj = nn.Conv1d(n_mels, hidden_ch, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm1d(hidden_ch)
        self.act0 = nn.GELU()

        # TCN stack
        blocks = []
        for d in tcn_dilations:
            blocks.append(TCNBlock(hidden_ch, dilation=int(d), dropout=dropout))
        self.tcn = nn.Sequential(*blocks)

        # token projection for attn (time-major tokens)
        self.post = nn.Sequential(
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pool = AttnPool1D(hidden_ch, num_heads=attn_heads)
        self.head = nn.Linear(hidden_ch, emb)

    def forward(self, mel):           # mel: [B, n_mels, T]
        y = self.mel_proj(mel)        # [B, C, T]
        y = self.act0(self.bn0(y))    # [B, C, T]
        y = self.tcn(y)               # [B, C, T]
        y = self.post(y)              # [B, C, T]
        tokens = y.transpose(1, 2)    # [B, T, C]
        pooled, tokens = self.pool(tokens)  # pooled: [B,C]
        z = self.head(pooled)         # [B, emb]
        return z, tokens              # tokens for cross-attn

# ----------------------------
# SecDual with TCN+AttnPool encoders
# ----------------------------
class SecDual(nn.Module):
    """
    Security-aware dual-stream:
      - amplitude & phase encoders (Mel->TCN->AttnPool)
      - cross attention (A<->P) on token sequences
      - fused embedding -> classifier
      - vocoder head via GRL (adversarial invariance)
      - optional temp head for eval-time calibration
    """
    def __init__(self,
                 n_mels=80,
                 emb=128,
                 n_vocoders=8,
                 grl_lambda=0.2,
                 dropout=0.2,
                 tcn_layers=4,
                 tcn_dilations=(1,2,4,8),
                 hidden_ch=128,
                 attn_heads=1):
        super().__init__()
        self.encA = EncoderTCN1D(n_mels, hidden_ch, emb,
                                 tcn_layers=tcn_layers,
                                 tcn_dilations=list(tcn_dilations),
                                 dropout=dropout, attn_heads=attn_heads)
        self.encP = EncoderTCN1D(n_mels, hidden_ch, emb,
                                 tcn_layers=tcn_layers,
                                 tcn_dilations=list(tcn_dilations),
                                 dropout=dropout, attn_heads=attn_heads)

        self.ca_AP = CrossAttn(emb, heads=attn_heads)   # A queries P
        self.ca_PA = CrossAttn(emb, heads=attn_heads)   # P queries A

        self.fuse = nn.Sequential(
            nn.Linear(emb * 2, hidden_ch), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ch, emb)
        )
        self.cls = nn.Sequential(
            nn.Linear(emb, hidden_ch), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ch, 1)
        )
        self.temp_head = nn.Sequential(
            nn.Linear(emb, emb // 2), nn.GELU(),
            nn.Linear(emb // 2, 1)
        )
        self.voc_head = nn.Sequential(
            nn.Linear(emb, emb // 2), nn.GELU(),
            nn.Linear(emb // 2, n_vocoders)
        )
        self.grl_lambda = grl_lambda
        self.n_mels = n_mels
        self.emb = emb

    def forward(self, logmel, phmel, target=None, return_feat=False, eval_temp=False):
        """
        logmel, phmel: [B, n_mels, T]  (no channel dim; mel is channels)
        """
        # --- encoders: learned mel projection + TCN + attn pool ---
        zA, tokA = self.encA(logmel)     # zA:[B,emb], tokA:[B,T,C]
        zP, tokP = self.encP(phmel)

        # --- cross attention on single tokens (use pooled token as 1-length sequence) ---
        # Using pooled embeddings as a single query/key/value token works well for compact models.
                # >>> PATCH: token-level cross-attention (use real temporal tokens)
        # cap tokens for speed and stability
        Ta = min(64, tokA.shape[1]) if tokA is not None else 1
        Tp = min(64, tokP.shape[1]) if tokP is not None else 1
        tokA_s = tokA[:, :Ta, :] if tokA is not None else zA.unsqueeze(1)  # [B,Ta,C] / [B,1,C]
        tokP_s = tokP[:, :Tp, :] if tokP is not None else zP.unsqueeze(1)  # [B,Tp,C] / [B,1,C]

        # A queries P, and P queries A
        cAP_tokens = self.ca_AP(tokA_s, tokP_s)      # [B,Ta,C]
        cPA_tokens = self.ca_PA(tokP_s, tokA_s)      # [B,Tp,C]
        cAP = cAP_tokens.mean(dim=1)                 # [B,C]
        cPA = cPA_tokens.mean(dim=1)                 # [B,C]
        cross = 0.5 * (cAP + cPA)                    # [B,C]

        # fuse pooled and cross-modal signal
        z = self.fuse(torch.cat([zA, zP], dim=-1)) + cross  # [B,emb]
        # <<< PATCH


        # --- classifier logits ---
        logits = self.cls(z).squeeze(-1)                      # [B]

        # --- aux dict (ALWAYS expose z_spec/z_pros for CMRA) ---
        out = {
            "embedding": z,
            # expose stream embeddings so trainer's CMRA can see them
            "z_spec": zA,     # spectral/amplitude stream embedding
            "z_pros": zP,     # prosodic/phase stream embedding
            # optional but handy for analysis/ablation:
            "tokA": tokA, 
            "tokP": tokP,
        }

        if eval_temp:
            # temperature > 1; clamp upper bound for safety
            T_hat = torch.softplus(self.temp_head(z)) + 1.0
            logits = logits / T_hat.clamp_max(10.0)
            out["T_hat"] = T_hat

        # Vocoder head through GRL (domain-adversarial)
        z_grl = GRL.apply(z, self.grl_lambda)
        voc_logits = self.voc_head(z_grl)
        out["voc_logits"] = voc_logits

        # Keep return_feat for backward-compatibility (redundant now, but harmless)
        if return_feat:
            out["zA"], out["zP"] = zA, zP

        if target is None:
            return logits, out
        return logits, out
