# audioShieldNet/asnet_1/audioshieldnet/utils/eval_compat.py

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc as sk_auc

@torch.no_grad()
def eval_auc_eer(model, feats, loader, device="cuda"):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Each batch must be at least (wav, label).")
        wav = batch[0].to(device, non_blocking=True)
        y   = batch[1]
        logmel, phmel = feats(wav)
        logits, _ = model(logmel, phmel)
        prob = torch.sigmoid(logits).cpu().numpy()
        y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        ys.append(y_np); ps.append(prob)

    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    if y_true.size and y_prob.size:
        fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob.astype(float))
        auc = float(sk_auc(fpr, tpr))
        fnr = 1.0 - tpr
        idx = int(np.argmin(np.abs(fnr - fpr)))
        eer = float(max(fpr[idx], fnr[idx]))
    else:
        auc, eer = float('nan'), float('nan')
    return auc, eer
