# audioShieldNet/asnet_6/audioshieldnet/engine/evaluator.py

import os, json, time, warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

# Project deps
from audioshieldnet.security.ood import energy_score
from audioshieldnet.security.attacks import fgsm_attack, pgd_attack
from audioshieldnet.security.pseudo_ood import PseudoOODSampler
from audioshieldnet.security.plots import (
    plot_abstain_tradeoff,
    plot_energy_histograms_png,
    plot_risk_coverage_png,
)
from audioshieldnet.utils.risk_coverage import risk_coverage_from_energy, ece_on_subset
from audioshieldnet.utils.metrics import expected_calibration_error



# --------------------
# Legacy helpers (kept for compatibility)
# --------------------
def auc_eer(net, feats, dl, device, invert=False):
    """Returns (auc, eer, (y_true, y_prob)) using shared collect_probs."""
    y_true, y_prob, _ = collect_probs(feats, net, dl, device, invert=invert, need_paths=False)
    auc = roc_auc_score(y_true.astype(int), y_prob.astype(float)) if y_true.size else float('nan')
    fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob.astype(float)) if y_true.size else (None, None, None)
    if fpr is None:
        return float('nan'), float('nan'), (y_true, y_prob)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return float(auc), float(eer), (y_true, y_prob)


def adversarial_eval_legacy(feats, net, loader, device, loss_fn, adv_cfg):
    """Kept for backward compatibility with helper imports elsewhere."""
    from audioshieldnet.engine.evaluator import fnr_at_tpr95 as _fnr95
    if not adv_cfg.get('use_adv', False):
        return np.array([]), np.array([]), float('nan')

    eps_list = adv_cfg.get("adv_eps", [0.001])
    eps = float(eps_list[-1] if isinstance(eps_list, (list, tuple)) and len(eps_list) else eps_list)
    steps = int(adv_cfg.get("adv_steps", 1))
    alpha = float(adv_cfg.get("adv_alpha", 0.0)) or (eps / max(1, 4))

    ys_adv, ps_adv = [], []
    net.eval()
    for batch in tqdm(loader, desc="[eval] adversarial", unit="batch", leave=False):
        wav, y = batch[0], batch[1]
        wav = wav.to(device, non_blocking=True)
        y   = (y.to(device, non_blocking=True) if torch.is_tensor(y) else torch.tensor(y, device=device))
        if steps <= 1:
            wav_adv = fgsm_attack(wav, y, net, feats, lambda lg,t: loss_fn(lg,t,epoch=None), eps=eps)
        else:
            wav_adv = pgd_attack(wav, y, net, feats, lambda lg,t: loss_fn(lg,t,epoch=None), eps=eps, alpha=alpha, steps=steps)
        logmel_a, phmel_a = feats(wav_adv)
        logits_a, _ = net(logmel_a, phmel_a, target=None)
        ps_adv.append(torch.sigmoid(logits_a).detach().cpu().numpy())
        ys_adv.append(y.detach().cpu().numpy())

    y_true_adv = np.concatenate(ys_adv).reshape(-1) if ps_adv else np.array([])
    y_prob_adv = np.concatenate(ps_adv).reshape(-1) if ps_adv else np.array([])
    fnr95 = _fnr95(y_true_adv, y_prob_adv) if y_prob_adv.size else float('nan')
    return y_true_adv, y_prob_adv, fnr95

def adversarial_eval_with_curves(
    feats,
    net,
    loader,
    device,
    loss_fn,
    adv_cfg,
    outdir: Optional[str] = None,
    temperature: float = 1.0,
) -> Tuple[float, float, float, Dict[str, str]]:
    """
    Same as adversarial_eval, but also returns ROC/PR curves for adversarial
    examples if outdir is provided.

    Returns: rob_auc, rob_eer, fnr95, fig_paths_adv (possibly empty dict)
    """
    if not adv_cfg.get("use_adv", False):
        return float("nan"), float("nan"), float("nan"), {}

    eps, alpha, steps = _resolve_eps_alpha(adv_cfg)
    ys_adv, ps_adv = [], []
    net.eval()

    for batch in tqdm(loader, desc="[eval] adversarial(curves)", unit="batch", leave=False):
        if len(batch) < 2:
            continue
        wav, y = batch[0], batch[1]
        wav = wav.to(device, non_blocking=True)
        y   = (y.to(device, non_blocking=True) if torch.is_tensor(y) else torch.tensor(y, device=device))

        attack_loss = lambda lg, tgt: loss_fn(lg, tgt, epoch=None)
        if steps <= 1:
            wav_adv = fgsm_attack(wav, y, net, feats, attack_loss, eps=eps)
        else:
            wav_adv = pgd_attack(wav, y, net, feats, attack_loss, eps=eps, alpha=alpha, steps=steps)

        lm_a, pm_a = feats(wav_adv)
        logits_a, _ = net(lm_a, pm_a, target=None)
        if temperature and abs(temperature - 1.0) > 1e-6:
            logits_a = logits_a / float(temperature)
        ps_adv.append(torch.sigmoid(logits_a).detach().cpu().numpy().reshape(-1))
        ys_adv.append(y.detach().float().cpu().numpy().reshape(-1))

    y_true_adv = np.concatenate(ys_adv) if ys_adv else np.array([])
    y_prob_adv = np.concatenate(ps_adv) if ps_adv else np.array([])

    if y_true_adv.size == 0 or np.unique(y_true_adv).size < 2:
        return float("nan"), float("nan"), float("nan"), {}

    auc, eer, _, _, _ = auc_eer_core(y_true_adv, y_prob_adv)
    fnr95 = fnr_at_tpr95(y_true_adv, y_prob_adv)

    fig_paths_adv: Dict[str, str] = {}
    if outdir is not None and y_true_adv.size and np.unique(y_true_adv).size > 1:
        # reuse the existing ROC/PR helper, in a separate subfolder "adv"
        adv_dir = os.path.join(outdir, "adv")
        fig_paths_adv = plot_roc_pr(y_true_adv, y_prob_adv, adv_dir)

    return float(auc), float(eer), float(fnr95), fig_paths_adv



def suspicious_fraction(feats, net, loader, device, T_energy, tau_susp):
    es = []
    with torch.no_grad():
        for batch in loader:
            wav = batch[0].to(device, non_blocking=True)
            logmel, phmel = feats(wav)
            logits,_ = net(logmel, phmel, target=None)
            E = energy_score(logits, T=T_energy)
            es.append(E.cpu().numpy())
    es = np.concatenate(es).reshape(-1) if es else np.array([])
    return float((es >= tau_susp).mean()) if es.size else 0.0


# --------------------
# Robust checkpoint load
# --------------------
def load_checkpoint_into_model(ckpt_path: str, model: nn.Module, device: str = "cuda") -> Dict:
    """
    Loads a checkpoint saved by save_checkpoint(...). Accepts several formats:
      { 'model': state_dict, ... }  OR a raw state_dict (rare)
    Returns the loaded dict for reference.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    pkg = torch.load(ckpt_path, map_location=device)

    # Typical format from your save_checkpoint
    state = None
    if isinstance(pkg, dict):
        if "model" in pkg and isinstance(pkg["model"], dict):
            state = pkg["model"]
        elif "state_dict" in pkg and isinstance(pkg["state_dict"], dict):
            state = pkg["state_dict"]
    if state is None and isinstance(pkg, dict):
        # heuristic: try to detect a state-dict like mapping
        keys = list(pkg.keys())
        looks_like_state = all(isinstance(k, str) for k in keys) and any(k.endswith("weight") or k.endswith("bias") for k in keys)
        if looks_like_state:
            state = pkg

    if state is None:
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path} (keys={list(pkg.keys())[:10]})")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        warnings.warn(f"[load_checkpoint_into_model] missing={len(missing)} unexpected={len(unexpected)}")

    return pkg


# --------------------
# Basic metrics
# --------------------
def fnr_at_tpr95(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """FNR at TPR=0.95 on positive class=1."""
    if y_true.size == 0 or y_prob.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, thr = roc_curve(y_true.astype(int), y_prob.astype(float))
    idx = int(np.nanargmin(np.abs(tpr - 0.95)))
    fnr = 1.0 - tpr[idx]
    return float(fnr)


def collect_probs(
    feats, net: nn.Module, loader, device: str, invert: bool = False, need_paths: bool = False,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:

    """
    Returns (y_true, y_prob, paths?)
    - y_prob is sigmoid(logits) unless invert=True (then 1 - sigmoid).
    - If the dataset yields paths (batch[2]), we propagate them.
    """
    net.eval()
    ys, ps, paths = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[eval] clean", unit="batch", leave=False):
            if len(batch) < 2:
                continue
            wav, y = batch[0], batch[1]
            wav = wav.to(device, non_blocking=True)
            y   = (y.to(device, non_blocking=True) if torch.is_tensor(y) else torch.tensor(y, device=device))
            lm, pm = feats(wav)
            logits, _ = net(lm, pm, target=None)
            if temperature and abs(temperature - 1.0) > 1e-6:
                logits = logits / float(temperature)
            p = torch.sigmoid(logits)

            if invert:
                p = 1.0 - p

            ys.append(y.detach().float().cpu().numpy().reshape(-1))
            ps.append(p.detach().float().cpu().numpy().reshape(-1))

            if need_paths and len(batch) >= 3:
                bpaths = batch[2]
                if isinstance(bpaths, (list, tuple)):
                    paths.extend([str(x) for x in bpaths])
                else:
                    paths.extend([None] * y.numel())

    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    return y_true, y_prob, (paths if need_paths else None)


def auc_eer_core(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Returns: auc, eer, thr_eer, thr_f1, f1_best
    """
    if y_true.size == 0 or y_prob.size == 0 or np.unique(y_true).size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    auc = roc_auc_score(y_true.astype(int), y_prob.astype(float))

    fpr, tpr, thr = roc_curve(y_true.astype(int), y_prob.astype(float))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr_eer = float(thr[idx]) if thr is not None and thr.size else float("nan")

    # fast F1 threshold scan
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_true.astype(int), (y_prob >= t).astype(int)) for t in grid]
    best_idx = int(np.argmax(f1s)) if len(f1s) else 0
    thr_f1 = float(grid[best_idx]) if len(grid) else float("nan")
    f1_best = float(f1s[best_idx]) if len(f1s) else float("nan")

    return float(auc), float(eer), thr_eer, thr_f1, f1_best


def ece_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0 or y_prob.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(expected_calibration_error(y_true, y_prob))


# --------------------
# Adversarial evaluation
# --------------------
def _resolve_eps_alpha(adv_cfg: Dict) -> Tuple[float, float, int]:
    eps_list = adv_cfg.get("adv_eps", [0.001])
    eps = float(eps_list[-1] if isinstance(eps_list, (list, tuple)) and len(eps_list) else eps_list)
    steps = int(adv_cfg.get("adv_steps", 1))
    alpha = float(adv_cfg.get("adv_alpha", 0.0)) or (eps / max(1, 4))
    return eps, alpha, steps


def adversarial_eval(feats, net, loader, device, loss_fn, adv_cfg, temperature: float = 1.0) -> Tuple[float, float, float]:
    """
    Returns: rob_auc, rob_eer, fnr95
    """
    if not adv_cfg.get('use_adv', False):
        return float("nan"), float("nan"), float("nan")

    eps, alpha, steps = _resolve_eps_alpha(adv_cfg)
    ys_adv, ps_adv = [], []
    net.eval()

    for batch in tqdm(loader, desc="[eval] adversarial", unit="batch", leave=False):
        if len(batch) < 2:
            continue
        wav, y = batch[0], batch[1]
        wav = wav.to(device, non_blocking=True)
        y   = (y.to(device, non_blocking=True) if torch.is_tensor(y) else torch.tensor(y, device=device))

        attack_loss = lambda lg, tgt: loss_fn(lg, tgt, epoch=None)
        if steps <= 1:
            wav_adv = fgsm_attack(wav, y, net, feats, attack_loss, eps=eps)
        else:
            wav_adv = pgd_attack(wav, y, net, feats, attack_loss, eps=eps, alpha=alpha, steps=steps)

        lm_a, pm_a = feats(wav_adv)
        logits_a, _ = net(lm_a, pm_a, target=None)
        if temperature and abs(temperature - 1.0) > 1e-6:
            logits_a = logits_a / float(temperature)
        ps_adv.append(torch.sigmoid(logits_a).detach().cpu().numpy().reshape(-1))
        ys_adv.append(y.detach().float().cpu().numpy().reshape(-1))

    y_true_adv = np.concatenate(ys_adv) if ys_adv else np.array([])
    y_prob_adv = np.concatenate(ps_adv) if ps_adv else np.array([])

    if y_true_adv.size == 0 or np.unique(y_true_adv).size < 2:
        return float("nan"), float("nan"), float("nan")

    auc, eer, _, _, _ = auc_eer_core(y_true_adv, y_prob_adv)
    fnr95 = fnr_at_tpr95(y_true_adv, y_prob_adv)
    return float(auc), float(eer), float(fnr95)


# --------------------
# Energy / OOD eval
# --------------------
def energy_arrays(feats, net, loader, device, T_energy: float, temperature: float = 1.0) -> np.ndarray:

    net.eval()
    E = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[eval] energy(ID)", unit="batch", leave=False):
            wav = batch[0].to(device, non_blocking=True)
            lm, pm = feats(wav)
            lg, _ = net(lm, pm, target=None)
            if temperature and abs(temperature - 1.0) > 1e-6:
                lg = lg / float(temperature)
            E.append(energy_score(lg, T=T_energy).detach().cpu().numpy())
    return np.concatenate(E) if E else np.array([])


def energy_arrays_ood(
    feats, net, loader, device, T_energy: float, cfg: Dict, temperature: float = 1.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    sec = (cfg.get("security", {}) or {})
    ood_cfg = (sec.get("ood_push", {}) or {})
    if not bool(ood_cfg.get("use", False)):
        return {}, {}

    types = ood_cfg.get("types", ["mp3"])
    sr_from_cfg = int((cfg.get("data", {}) or {}).get("sr", 16000))
    sampler = PseudoOODSampler(sr=sr_from_cfg, enabled_types=types, curriculum=False)

    id_E = energy_arrays(feats, net, loader, device, T_energy, temperature=temperature)

    per_type_E = {}
    per_type_auroc = {}

    with torch.no_grad():
        for t in types:
            Es = []
            for batch in tqdm(loader, desc=f"[eval] energy(OOD:{t})", unit="batch", leave=False):
                wav = batch[0].to(device, non_blocking=True)
                wav_ood, _ = sampler(wav, step=0)
                lm_o, pm_o = feats(wav_ood)
                lg_o, _ = net(lm_o, pm_o, target=None)
                if temperature and abs(temperature - 1.0) > 1e-6:
                    lg_o = lg_o / float(temperature)
                Es.append(energy_score(lg_o, T=T_energy).detach().cpu().numpy())

            oE = np.concatenate(Es) if Es else np.array([])
            per_type_E[t] = oE

            if id_E.size and oE.size and not np.allclose(id_E, oE):
                y_mix = np.concatenate([np.zeros_like(id_E), np.ones_like(oE)])
                s_mix = np.concatenate([id_E, oE])  # higher ⇒ more OOD
                try:
                    per_type_auroc[t] = float(roc_auc_score(y_mix, s_mix))
                except Exception:
                    per_type_auroc[t] = float("nan")

    return per_type_E, per_type_auroc


# --------------------
# Polarity choice (match Trainer logic)
# --------------------
def choose_polarity_from_auc(feats, net, loader, device) -> Tuple[bool, float, float]:
    y_p, p_p, _ = collect_probs(feats, net, loader, device, invert=False, need_paths=False)
    y_i, p_i, _ = collect_probs(feats, net, loader, device, invert=True,  need_paths=False)

    auc_p = roc_auc_score(y_p.astype(int), p_p.astype(float)) if y_p.size else float("nan")
    auc_i = roc_auc_score(y_i.astype(int), p_i.astype(float)) if y_i.size else float("nan")
    invert = bool(auc_i > auc_p + 1e-9)
    return invert, float(auc_p), float(auc_i)


# --------------------
# Writers / plots
# --------------------
def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def write_predictions_csv(out_csv: str, paths: Optional[List[str]], y_true: np.ndarray,
                          y_prob: np.ndarray, energies: Optional[np.ndarray],
                          threshold: float):
    _ensure_outdir(os.path.dirname(out_csv))
    with open(out_csv, "w") as f:
        header = ["index", "path", "label", "pred", "prob"]
        if energies is not None:
            header.append("energy")
        f.write(",".join(header) + "\n")
        N = int(y_true.shape[0]) if y_true.size else 0
        for i in range(N):
            prob = float(y_prob[i])
            pred = int(prob >= threshold)
            row = [
                str(i),
                (("" if paths is None else (paths[i] if paths and i < len(paths) and paths[i] is not None else ""))),
                str(int(y_true[i])),
                str(pred),
                f"{prob:.6f}",
            ]
            if energies is not None:
                row.append(f"{float(energies[i]):.6f}")
            f.write(",".join(row) + "\n")


def write_json(obj: Dict, path: str):
    _ensure_outdir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_confusion_png(y_true: np.ndarray, y_prob: np.ndarray, thr: float, out_png: str, title: str):
    _ensure_outdir(os.path.dirname(out_png))
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real(0)", "fake(1)"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=180)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{title} (thr={thr:.3f})")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    return cm.tolist()


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, outdir: str) -> Dict[str, str]:
    _ensure_outdir(outdir)
    # ROC
    fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob.astype(float))
    roc_auc = roc_auc_score(y_true.astype(int), y_prob.astype(float))
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f}")
    ax.plot([0,1],[0,1], ls="--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(outdir, "roc_curve.png")
    plt.savefig(roc_path); plt.close(fig)

    # PR
    prec, rec, _ = precision_recall_curve(y_true.astype(int), y_prob.astype(float))
    ap = average_precision_score(y_true.astype(int), y_prob.astype(float))
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    ax.plot(rec, prec, lw=2, label=f"AP={ap:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall Curve"); ax.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(outdir, "pr_curve.png")
    plt.savefig(pr_path); plt.close(fig)

    return {"roc": roc_path, "pr": pr_path}


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, outdir: str, bins: int = 15) -> str:
    _ensure_outdir(outdir)
    prob_true, prob_pred = calibration_curve(y_true.astype(int), y_prob.astype(float), n_bins=bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label="Model")
    ax.plot([0,1], [0,1], ls="--", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical probability")
    ax.set_title("Reliability Curve (Calibration)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "reliability_curve.png")
    plt.savefig(path); plt.close(fig)
    return path


def tau_sweep_and_plot(y_true: np.ndarray, y_prob: np.ndarray, energies: np.ndarray,
                        outdir: str, tau_min: float = -2.0, tau_max: float = 1.0,
                        num: int = 61) -> Dict:
    """
    For tau in [tau_min, tau_max], compute kept-subset AUC/EER and plot curves.
    Returns a dict with arrays and best picks by AUC and by EER.
    """
    _ensure_outdir(outdir)
    taus = np.linspace(tau_min, tau_max, num)
    kept_auc, kept_eer, kept_cov = [], [], []

    for tau in taus:
        keep = energies < tau
        cov = keep.mean() if energies.size else np.nan
        if keep.any():
            yk, pk = y_true[keep], y_prob[keep]
            if yk.size > 1 and np.unique(yk).size > 1:
                auc, eer, _, _, _ = auc_eer_core(yk, pk)
            else:
                auc, eer = np.nan, np.nan
        else:
            auc, eer = np.nan, np.nan
        kept_auc.append(auc); kept_eer.append(eer); kept_cov.append(cov)

    kept_auc = np.array(kept_auc, dtype=float)
    kept_eer = np.array(kept_eer, dtype=float)
    kept_cov = np.array(kept_cov, dtype=float)

    # Best picks
    with np.errstate(invalid="ignore"):
        idx_auc = int(np.nanargmax(kept_auc)) if np.any(~np.isnan(kept_auc)) else 0
        idx_eer = int(np.nanargmin(kept_eer)) if np.any(~np.isnan(kept_eer)) else 0

    best_auc = {
        "tau": float(taus[idx_auc]),
        "auc": float(kept_auc[idx_auc]) if kept_auc.size else float("nan"),
        "eer": float(kept_eer[idx_auc]) if kept_eer.size else float("nan"),
        "coverage": float(kept_cov[idx_auc]) if kept_cov.size else float("nan"),
        "index": int(idx_auc),
    }
    best_eer = {
        "tau": float(taus[idx_eer]),
        "auc": float(kept_auc[idx_eer]) if kept_auc.size else float("nan"),
        "eer": float(kept_eer[idx_eer]) if kept_eer.size else float("nan"),
        "coverage": float(kept_cov[idx_eer]) if kept_cov.size else float("nan"),
        "index": int(idx_eer),
    }

    # Plot
    fig, ax1 = plt.subplots(figsize=(6.0, 4.6), dpi=160)
    ax1.plot(taus, kept_auc, lw=2, label="AUC(kept)")
    ax1.set_xlabel("τ (energy threshold)"); ax1.set_ylabel("AUC (kept)")
    ax2 = ax1.twinx()
    ax2.plot(taus, kept_eer, lw=2, ls="--", label="EER(kept)")
    ax2.set_ylabel("EER (kept)")

    # Mark bests
    ax1.axvline(best_auc["tau"], ls=":", lw=1.5)
    ax2.axvline(best_eer["tau"], ls=":", lw=1.5)

    fig.suptitle("τ Sweep: Kept-Subset Metrics vs Threshold")
    fig.tight_layout()
    path = os.path.join(outdir, "tau_sweep.png")
    plt.savefig(path); plt.close(fig)

    return {
        "taus": taus.tolist(),
        "kept_auc": kept_auc.tolist(),
        "kept_eer": kept_eer.tolist(),
        "kept_coverage": kept_cov.tolist(),
        "best_by_auc": best_auc,
        "best_by_eer": best_eer,
        "figure": path,
    }

def _fit_temperature_on_loader(
    feats, net: nn.Module, loader, device: str, max_iter: int = 200, init_T: float = 1.0
) -> float:
    """
    Fit scalar temperature by minimizing NLL on a calibration/VAL loader.
    Returns temperature > 0.
    """
    net.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) < 2:
                continue
            wav, y = batch[0], batch[1]
            wav = wav.to(device, non_blocking=True)
            y = (y.to(device, non_blocking=True) if torch.is_tensor(y) else torch.tensor(y, device=device)).float()
            lm, pm = feats(wav)
            lg, _ = net(lm, pm, target=None)
            logits_list.append(lg.detach())
            labels_list.append(y.detach())

    if not logits_list:
        return 1.0

    logits = torch.cat(logits_list, dim=0).flatten()
    labels = torch.cat(labels_list, dim=0).flatten()

    # Optimize T via a softplus-reparameterized scalar to keep T>0
    s = torch.tensor(np.log(np.exp(init_T - 1.0) - 1.0) if init_T > 1.0 else 0.0, dtype=torch.float32, requires_grad=True, device=logits.device)
    opt = torch.optim.LBFGS([s], lr=0.1, max_iter=min(max_iter, 200), line_search_fn="strong_wolfe")

    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        T = torch.nn.functional.softplus(s) + 1e-6
        loss = bce(logits / T, labels)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T = float(torch.nn.functional.softplus(s).item() + 1e-6)
    except Exception:
        # Fallback small Adam if LBFGS fails
        s2 = torch.tensor(s.detach().cpu().item(), dtype=torch.float32, requires_grad=True, device=logits.device)
        opt2 = torch.optim.Adam([s2], lr=0.01)
        for _ in range(100):
            opt2.zero_grad()
            Tt = torch.nn.functional.softplus(s2) + 1e-6
            loss = bce(logits / Tt, labels)
            loss.backward()
            opt2.step()
        T = float(torch.nn.functional.softplus(s2).item() + 1e-6)

    # Clamp to a reasonable range to avoid extreme calibration
    T = max(0.5, min(5.0, T))
    return T


# --------------------
# Top-level one-shot evaluation
# --------------------
def evaluate_on_loader(
    cfg: Dict,
    feats,
    net: nn.Module,
    dl,                     # eval loader (VAL or TEST)
    device: str,
    ckpt_dir: str,
    loss_fn,                # criterion used (for adversarial loss)
    result_subdir: str = "result_test",
    temperature: float = 1.0,
) -> Dict:
    """
    Runs full eval on test loader and saves under ckpt_dir/result_subdir/:
      - predictions_test.csv (index,path,label,pred,prob,energy?)
      - figures/:
          * roc_curve.png, pr_curve.png
          * reliability_curve.png
          * tau_sweep.png
          * confusion_f1.png, confusion_eer.png
          * energy_hist_*.png
          * risk_coverage_*.png
          * abstain_vs_fa.png
          * adv/roc_curve.png, adv/pr_curve.png
      - report.json
    """
    result_dir = os.path.join(ckpt_dir, result_subdir)
    figs_dir   = os.path.join(result_dir, "figures")
    _ensure_outdir(result_dir)
    _ensure_outdir(figs_dir)

    eval_cfg = (cfg.get("eval", {}) or {})
    sec      = (cfg.get("security", {}) or {})
    triage   = (sec.get("triage", {}) or {})
    T_energy = float(triage.get("T", 1.0))
    force_inv = eval_cfg.get("force_invert", None)

    # --------------------
    # Polarity & clean probs
    # --------------------
    if isinstance(force_inv, bool):
        invert = force_inv
        auc_as_is, auc_inv = float("nan"), float("nan")
    else:
        invert, auc_as_is, auc_inv = choose_polarity_from_auc(feats, net, dl, device)

    y_true, y_prob, paths = collect_probs(
        feats, net, dl, device,
        invert=invert,
        need_paths=True,
        temperature=temperature,
    )

    # Clean metrics
    auc, eer, thr_eer, thr_f1, f1_best = auc_eer_core(y_true, y_prob)
    ece = ece_metric(y_true, y_prob)

    # --------------------
    # Energies (ID) + abstain + risk–coverage
    # --------------------
    E_id = energy_arrays(feats, net, dl, device, T_energy, temperature=temperature)
    tau_susp = float(triage.get("tau_susp_energy", -0.25))
    keep_mask = (E_id < tau_susp) if E_id.size else np.array([], dtype=bool)

    auc_kept = eer_kept = ece_kept = float("nan")
    abstain_frac = float("nan")
    aurc_eer = aurc_1ma = float("nan")
    rc_paths: Dict[str, str] = {}
    abstain_tradeoff_png: str = ""

    if E_id.size and y_true.size and y_prob.size:
        if keep_mask.any():
            yk, pk = y_true[keep_mask], y_prob[keep_mask]
            if yk.size > 1 and np.unique(yk).size > 1:
                auc_kept, eer_kept, _, _, _ = auc_eer_core(yk, pk)
            ece_kept = ece_on_subset(yk, pk)

        abstain_frac = 1.0 - float(keep_mask.mean())

        # Risk–coverage curves (EER and 1–AUC)
        covs_eer,  risks_eer,  aurc_eer  = risk_coverage_from_energy(
            y_true, y_prob, E_id, metric="eer"
        )
        covs_1ma, risks_1ma, aurc_1ma   = risk_coverage_from_energy(
            y_true, y_prob, E_id, metric="1-auc"
        )

        rc_paths = plot_risk_coverage_png(
            np.asarray(covs_eer),  np.asarray(risks_eer),
            np.asarray(covs_1ma), np.asarray(risks_1ma),
            figs_dir,
            prefix="risk_coverage",
        )

        # Abstain vs fake-accept tradeoff
        abstain_tradeoff_png = os.path.join(figs_dir, "abstain_vs_fa.png")
        plot_abstain_tradeoff(y_true, y_prob, E_id, abstain_tradeoff_png)

    # --------------------
    # OOD energy (per-type) + energy histograms
    # --------------------
    per_type_E, per_type_auroc = energy_arrays_ood(
        feats, net, dl, device, T_energy, cfg, temperature=temperature
    )

    energy_hist_paths: Dict[str, str] = {}
    if E_id.size:
        try:
            energy_hist_paths = plot_energy_histograms_png(
                E_id,
                per_type_E if isinstance(per_type_E, dict) else {},
                figs_dir,
            )
        except Exception:
            energy_hist_paths = {}

    # --------------------
    # Adversarial robustness (metrics + ROC/PR on adv examples)
    # --------------------
    rob_auc, rob_eer, fnr95, adv_figs = adversarial_eval_with_curves(
        feats, net, dl, device, loss_fn, sec, outdir=figs_dir, temperature=temperature
    )

    # --------------------
    # CSV threshold / predictions
    # --------------------
    csv_src = str(eval_cfg.get("csv_threshold_source", "f1")).lower()
    if csv_src == "eer" and not np.isnan(thr_eer):
        csv_thr = float(thr_eer)
        csv_tag = "eer"
    elif csv_src == "config" and eval_cfg.get("use_config_threshold", False):
        csv_thr = float(eval_cfg.get("threshold", 0.5))
        csv_tag = "config"
    else:
        csv_thr = float(thr_f1 if not np.isnan(thr_f1) else 0.5)
        csv_tag = "f1"

    out_csv = os.path.join(result_dir, "predictions_test.csv")
    write_predictions_csv(
        out_csv,
        paths,
        y_true,
        y_prob,
        energies=E_id if E_id.size else None,
        threshold=csv_thr,
    )

    # --------------------
    # Classic figures: confusion, ROC/PR, reliability, tau sweep
    # --------------------
    png_f1  = os.path.join(figs_dir, "confusion_f1.png")
    png_eer = os.path.join(figs_dir, "confusion_eer.png")
    cm_f1  = plot_confusion_png(
        y_true, y_prob,
        thr_f1 if not np.isnan(thr_f1) else 0.5,
        png_f1,
        "Confusion @F1-thr",
    )
    cm_eer = plot_confusion_png(
        y_true, y_prob,
        thr_eer if not np.isnan(thr_eer) else 0.5,
        png_eer,
        "Confusion @EER-thr",
    )

    fig_paths = plot_roc_pr(y_true, y_prob, figs_dir)
    rel_path  = plot_reliability(
        y_true, y_prob, figs_dir,
        bins=int(eval_cfg.get("reliability_bins", 15)),
    )
    fig_paths["reliability"] = rel_path

    tau_cfg = (triage.get("tau_sweep", {}) or {})
    tau_min = float(tau_cfg.get("min", -2.0))
    tau_max = float(tau_cfg.get("max",  1.0))
    tau_num = int(tau_cfg.get("num",  61))
    tau_result = {}
    if E_id.size:
        tau_result = tau_sweep_and_plot(
            y_true, y_prob, E_id, figs_dir,
            tau_min=tau_min, tau_max=tau_max, num=tau_num,
        )

    # --------------------
    # Build report.json
    # --------------------
    report = {
        "polarity": {
            "forced": force_inv if isinstance(force_inv, bool) else False,
            "invert_used": bool(invert),
            "auc_as_is": auc_as_is,
            "auc_inverted": auc_inv,
        },
        "clean": {
            "auc": auc,
            "eer": eer,
            "ece": ece,
            "thr_eer": thr_eer,
            "thr_f1": thr_f1,
            "f1_best": f1_best,
        },
        "abstain_energy": {
            "T": T_energy,
            "tau_susp": tau_susp,
            "abstain_frac": abstain_frac,
            "auc_kept": auc_kept,
            "eer_kept": eer_kept,
            "ece_kept": ece_kept,
            "aurc_eer": aurc_eer,
            "aurc_1minusauc": aurc_1ma,
        },
        "ood_energy": {
            "per_type_auroc": per_type_auroc,
        },
        "adversarial": {
            "auc": rob_auc,
            "eer": rob_eer,
            "fnr_at_tpr95": fnr95,
            "eps_used": sec.get("adv_eps", None),
            "steps":    sec.get("adv_steps", None),
            "alpha":    sec.get("adv_alpha", None),
        },
        "counts": {
            "N":   int(y_true.size),
            "pos": int((y_true == 1).sum()) if y_true.size else 0,
            "neg": int((y_true == 0).sum()) if y_true.size else 0,
        },
        "confusion": {
            "matrix_f1": cm_f1,
            "matrix_eer": cm_eer,
            "labels": ["real(0)", "fake(1)"],
        },
        "files": {
            "result_dir":      os.path.abspath(result_dir),
            "predictions_csv": os.path.abspath(out_csv),
            "figures": {
                # clean performance
                "roc_curve_png":         os.path.abspath(fig_paths["roc"]),
                "pr_curve_png":          os.path.abspath(fig_paths["pr"]),
                "reliability_curve_png": os.path.abspath(fig_paths["reliability"]),
                "tau_sweep_png":         os.path.abspath(tau_result.get("figure", "")),
                "confusion_f1_png":      os.path.abspath(png_f1),
                "confusion_eer_png":     os.path.abspath(png_eer),

                # adversarial curves
                "adv_roc_curve_png": os.path.abspath(adv_figs.get("roc", "")) if isinstance(adv_figs, dict) else "",
                "adv_pr_curve_png":  os.path.abspath(adv_figs.get("pr", ""))  if isinstance(adv_figs, dict) else "",

                # energy histograms (ID + ID vs OOD types)
                "energy_hist_pngs": {
                    k: os.path.abspath(v) for k, v in (energy_hist_paths or {}).items()
                },

                # risk–coverage curves
                "risk_coverage_eer_png":   os.path.abspath(rc_paths.get("eer", ""))   if isinstance(rc_paths, dict) else "",
                "risk_coverage_1mauc_png": os.path.abspath(rc_paths.get("1mauc", "")) if isinstance(rc_paths, dict) else "",

                # abstain vs fake-accept trade-off
                "abstain_vs_fa_png": os.path.abspath(abstain_tradeoff_png) if abstain_tradeoff_png else "",
            },
        },
        "csv_threshold": {"source": csv_tag, "value": csv_thr},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_json = os.path.join(result_dir, "report.json")
    write_json(report, out_json)
    return report
