# audioShieldNet/asnet_6/audioshieldnet/models/asn.py

import torch
import torch.nn as nn
from .asn_features import ASNFeatures
from .asnnet_model import ASNNet
from .asnnet_secdual import SecDual  # NEW

def build_model(cfg, device):
    feats = ASNFeatures(**{k: cfg['data'][k] for k in ['sr','n_fft','hop','n_mels']}).to(device)

    # NEW: switchable variants
    variant = str(cfg.get("model", {}).get("variant", "asnnet")).lower()
    if variant == "secdual":
        net = SecDual(
            emb=cfg['model'].get('emb', 128),
            n_vocoders=cfg['model'].get('n_vocoders', 8),
            grl_lambda=cfg['model'].get('grl_lambda', 0.2),
            dropout=cfg['model'].get('dropout', 0.2),
        ).to(device)
    else:
        net = ASNNet(n_mels=cfg['data']['n_mels'],
                     cons_weight=0.0,
                     dropout=cfg['model'].get('dropout', 0.1)).to(device)

    # Initialize last binary logit bias from class prior (works for either variant)
    prior = (cfg.get("data", {}) or {}).get("prior_counts", None)
    if prior:
        tot = max(1, prior.get("real", 0) + prior.get("fake", 0))
        p1  = max(1.0 / tot, prior.get("fake", 0) / tot)
        b0  = float(torch.log(torch.tensor(p1/(1-p1))))
        with torch.no_grad():
            last = None
            for mod in reversed(list(net.modules())):
                if isinstance(mod, nn.Linear) and mod.out_features == 1:
                    last = mod; break
            if last is not None and last.bias is not None:
                last.bias.fill_(b0)

    return feats, net
