# audioshieldnet/security/backdoor_scan.py
import numpy as np
from sklearn.cluster import KMeans
import torch

@torch.no_grad()
def embedding_scan(model, feats, loader, device, head_fn=None, k=4):
    """
    Collect embeddings (penultimate) across the loader and run k-means.
    head_fn: if provided, should be call (model, feats, wav) -> embedding tensor
    Returns dict with cluster ids and stats.
    """
    embs = []
    labels = []
    model.eval()
    for wav, y, *rest in loader:
        wav = wav.to(device)
        logmel, phmel = feats(wav)
        if head_fn is None:
            # Expect the network to expose forward_features() or return embedding when called with target=None
            if hasattr(model, "forward_features"):
                z = model.forward_features(logmel, phmel)
            else:
                # fallback: call model and use concatenated penultimate vectors (if available)
                logits, aux = model(logmel, phmel, target=None)
                # Try to recover embedding from aux or internal attr 'last_embedding'
                if "embedding" in aux:
                    z = aux["embedding"]
                else:
                    # final fallback: use logits repeated to match dims (not ideal)
                    z = torch.unsqueeze(torch.sigmoid(logits), -1).cpu()
        else:
            z = head_fn(model, feats, wav)
        embs.append(z.cpu())
        labels.append(y.cpu())
    if not embs:
        return {"k": k, "n": 0, "clusters": []}
    embs = torch.cat(embs).numpy()
    labels = torch.cat(labels).numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embs)
    clusters = [{"label_counts": dict(zip(*np.unique(labels[km.labels_==i], return_counts=True))),
                 "size": int((km.labels_==i).sum())} for i in range(k)]
    return {"k": k, "n": embs.shape[0], "clusters": clusters, "centers_shape": km.cluster_centers_.shape}
