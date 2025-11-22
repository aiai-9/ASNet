# audioShieldNet/asnet_1/audioshieldnet/utils/ema.py

import torch
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
    @torch.no_grad()
    def restore(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}
