# audioShieldNet/asnet_6/audioshieldnet/security/plots.py

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from sklearn.metrics import (
    roc_curve,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Optional W&B import
try:
    import wandb
    from wandb import Image
except Exception:
    wandb = None
    Image = None


# ===========================================================
# 📊 CONFUSION MATRIX & METRIC VISUALIZATION
# ===========================================================

def _safe_log(run, payload: dict):
    """Helper to safely log to W&B."""
    if run is None:
        return
    try:
        run.log(payload, commit=False)
    except Exception:
        pass


def compute_eer_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute probability threshold corresponding to Equal Error Rate (EER)."""
    if y_true.size == 0 or y_prob.size == 0:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true.astype(int), y_prob.astype(float))
    if thr is None or thr.size == 0:
        return 0.5
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    if 0 <= idx < thr.size:
        return float(thr[idx])
    return 0.5


def log_val_confusion_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics_dir: str,
    epoch: int,
    wandb_run: Optional[object] = None,
    threshold: Optional[float] = None,
    also_log_fixed_05: bool = False,
    label_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute and log confusion matrix, precision, recall, and F1 for validation.
    - If threshold=None → compute EER threshold automatically.
    - Optionally logs a second CM at fixed 0.5 for comparison.
    """
    os.makedirs(metrics_dir, exist_ok=True)

    if y_true.size == 0 or y_prob.size == 0:
        print("[WARN] Empty y_true or y_prob — skipping confusion matrix.")
        return {}

    # Determine threshold
    thr = compute_eer_threshold(y_true, y_prob) if (threshold is None or np.isnan(threshold)) else float(threshold)
    y_pred = (y_prob >= thr).astype(int)

    # Compute metrics
    cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true.astype(int), y_pred, average="binary", zero_division=0
    )

    # Print summary
    print(f"[VAL][thr={thr:.3f}] Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  (EER-mode)" if threshold is None
          else f"[VAL][thr={thr:.3f}] Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

    # Save confusion matrix image
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names if label_names else [0, 1])
    disp.plot(cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix @ thr={thr:.2f} (Epoch {epoch})")
    plt.tight_layout()
    cm_path = os.path.join(metrics_dir, f"confmat_epoch{epoch:03d}.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # Log to WandB
    if wandb_run is not None and Image is not None:
        _safe_log(wandb_run, {
            "val/conf_threshold_used": thr,
            "val/precision": prec,
            "val/recall": rec,
            "val/f1": f1,
            "val/confusion_matrix": Image(cm_path)
        })

    # Optional comparison at fixed 0.5
    if also_log_fixed_05:
        y_pred_05 = (y_prob >= 0.5).astype(int)
        cm05 = confusion_matrix(y_true.astype(int), y_pred_05, labels=[0, 1])
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.0))
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm05, display_labels=label_names if label_names else [0, 1])
        disp2.plot(cmap="Greens", colorbar=False, ax=ax2)
        ax2.set_title(f"Confusion Matrix @ thr=0.50 (Epoch {epoch})")
        plt.tight_layout()
        cm05_path = os.path.join(metrics_dir, f"confmat_fixed05_epoch{epoch:03d}.png")
        fig2.savefig(cm05_path, dpi=150)
        plt.close(fig2)
        if wandb_run is not None and Image is not None:
            _safe_log(wandb_run, {"val/confusion_matrix@0.5": Image(cm05_path)})

    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "threshold": float(thr)}


# ===========================================================
# 📉 Abstain vs. False Accept Tradeoff
# ===========================================================
def plot_abstain_tradeoff(y_true, y_prob, energy_scores, out_png):
    """
    Plot abstain fraction vs false-accept rate for a range of energy thresholds.
    """
    thr_energy = np.percentile(energy_scores, np.linspace(0, 100, 50))
    fa = []
    abst = []
    for thr in thr_energy:
        abst_frac = float((energy_scores >= thr).mean())
        preds = (y_prob > 0.5).astype(int)
        fa_rate = float(((preds == 0) & (y_true == 1)).sum()) / max(1, (y_true == 1).sum())
        fa.append(fa_rate)
        abst.append(abst_frac)

    plt.figure(figsize=(6, 4))
    plt.plot(abst, fa, marker="o")
    plt.xlabel("Abstain fraction")
    plt.ylabel("Fake accepted rate (FA)")
    plt.title("Abstain vs Fake-Accept tradeoff")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ===========================================================
# 📈 Curriculum Schedule Visualization
# ===========================================================
def _get_nested(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def curriculum_schedule_from_cfg(cfg: Dict[str, Any], total_epochs: int):
    """
    Parse curriculum gates and create boolean activation masks for each module.
    """

    sam_start = int(_get_nested(cfg, "train.sam_start_epoch", _get_nested(cfg, "train.stability_warmup_epochs", 0)))
    energy_start = int(_get_nested(cfg, "train.energy_start_epoch", 9999))
    ood_start = int(_get_nested(cfg, "train.ood_start_epoch", 9999))
    adv_start = int(_get_nested(cfg, "train.adv_start_epoch", _get_nested(cfg, "train.adv_warmup_epochs", 0)))
    swa_frac = float(_get_nested(cfg, "optim.swa.start_epoch", 0.8))
    swa_start_ep = max(0, min(total_epochs - 1, int(round(total_epochs * swa_frac))))

    use_sam = bool(_get_nested(cfg, "optim.use_sam", False))
    use_energy = bool(_get_nested(cfg, "security.energy_reg", False))
    use_ood = bool(_get_nested(cfg, "security.ood_push.use", False))
    use_adv = bool(_get_nested(cfg, "security.use_adv", False))
    use_swa = bool(_get_nested(cfg, "optim.swa.enable", False))

    def on_after(start_ep, enabled=True):
        m = np.zeros(total_epochs, dtype=int)
        if enabled and start_ep < total_epochs:
            m[start_ep:] = 1
        return m

    masks = {
        "SAM": on_after(sam_start, use_sam),
        "Energy": on_after(energy_start, use_energy),
        "OOD": on_after(ood_start, use_ood),
        "Adv": on_after(adv_start, use_adv),
        "SWA": on_after(swa_start_ep, use_swa),
    }

    schedule = {
        "sam_start": sam_start,
        "energy_start": energy_start,
        "ood_start": ood_start,
        "adv_start": adv_start,
        "swa_start_epoch": swa_start_ep,
    }
    return schedule, masks


def plot_curriculum_schedule(schedule: Dict[str, int], masks: Dict[str, "np.ndarray"], total_epochs: int, save_path: str):
    """Visualize curriculum activation schedule."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = np.arange(total_epochs)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    order = ["SAM", "Energy", "OOD", "Adv", "SWA"]
    y_offset = 0
    y_gap = 1.2

    for name in order:
        if name not in masks:
            continue
        y = masks[name] + y_offset
        ax.step(x, y, where="post", linewidth=2, label=name)
        start_key = {
            "SAM": "sam_start",
            "Energy": "energy_start",
            "OOD": "ood_start",
            "Adv": "adv_start",
            "SWA": "swa_start_epoch",
        }[name]
        start_ep = int(schedule.get(start_key, total_epochs))
        if 0 <= start_ep < total_epochs:
            ax.scatter([start_ep], [y_offset], s=30)
            ax.text(start_ep, y_offset + 0.15, f"start={start_ep}", fontsize=8, ha="left", va="bottom")
        y_offset += y_gap

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Module (stacked)")
    ax.set_title("Robust Curriculum Schedule")
    ax.set_xlim(0, total_epochs - 1)
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.legend(ncol=5, loc="upper left", fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def log_curriculum_schedule(cfg: Dict[str, Any], total_epochs: int, metrics_dir: str, epoch: int, wandb_run: Optional[object] = None):
    """Save and log curriculum schedule to W&B."""
    schedule, masks = curriculum_schedule_from_cfg(cfg, total_epochs)
    out_path = os.path.join(metrics_dir, f"curriculum_epoch{epoch:03d}.png")
    png = plot_curriculum_schedule(schedule, masks, total_epochs, out_path)
    if wandb_run is not None and Image is not None:
        _safe_log(wandb_run, {"val/curriculum_plot": Image(png)})
    return png

# ===========================================================
# 🔋 Energy Histograms (PNG for offline reporting)
# ===========================================================
def plot_energy_histograms_png(
    id_E: np.ndarray,
    ood_E_by_type: Optional[Dict[str, np.ndarray]],
    outdir: str,
    prefix: str = "energy_hist"
) -> Dict[str, str]:
    """
    Save static PNGs for ID energy and ID vs each OOD type.
    Returns a dict mapping names -> file paths.
    """
    os.makedirs(outdir, exist_ok=True)
    paths: Dict[str, str] = {}

    # ID-only histogram
    if isinstance(id_E, np.ndarray) and id_E.size:
        plt.figure(figsize=(5.0, 4.0))
        plt.hist(id_E, bins=50, alpha=0.8)
        plt.xlabel("Energy score")
        plt.ylabel("Count")
        plt.title("ID energy distribution")
        plt.tight_layout()
        p_id = os.path.join(outdir, f"{prefix}_id.png")
        plt.savefig(p_id, dpi=150)
        plt.close()
        paths["id"] = p_id

    # Overlay ID vs each OOD type
    if ood_E_by_type:
        for t, arr in ood_E_by_type.items():
            if not isinstance(arr, np.ndarray) or not arr.size or not isinstance(id_E, np.ndarray) or not id_E.size:
                continue
            plt.figure(figsize=(5.0, 4.0))
            plt.hist(id_E,  bins=50, alpha=0.6, label="ID")
            plt.hist(arr,   bins=50, alpha=0.6, label=f"OOD-{t}")
            plt.xlabel("Energy score")
            plt.ylabel("Count")
            plt.title(f"Energy: ID vs OOD-{t}")
            plt.legend()
            plt.tight_layout()
            p_ood = os.path.join(outdir, f"{prefix}_id_vs_{t}.png")
            plt.savefig(p_ood, dpi=150)
            plt.close()
            paths[f"id_vs_{t}"] = p_ood

    return paths


def log_energy_histograms(
    wandb_run,
    id_E: np.ndarray,
    ood_E_by_type: Optional[Dict[str, np.ndarray]] = None,
    prefix: str = "val",
    epoch: Optional[int] = None,
):
    """
    Log energy histograms for ID and each OOD type + mean/std and mean gap.
    Safe to call even if some arrays are empty.
    """
    if wandb_run is None:
        return

    payload = {}
    if isinstance(id_E, np.ndarray) and id_E.size:
        try:
            payload[f"{prefix}/E_id_hist"] = wandb.Histogram(id_E.tolist())
        except Exception:
            pass
        payload[f"{prefix}/E_id_mean"] = float(np.mean(id_E))
        payload[f"{prefix}/E_id_std"]  = float(np.std(id_E))

    if ood_E_by_type:
        for t, arr in ood_E_by_type.items():
            if not isinstance(arr, np.ndarray) or not arr.size:
                continue
            try:
                payload[f"{prefix}/E_ood_hist/{t}"] = wandb.Histogram(arr.tolist())
            except Exception:
                pass
            payload[f"{prefix}/E_ood_mean/{t}"] = float(np.mean(arr))
            payload[f"{prefix}/E_ood_std/{t}"]  = float(np.std(arr))
            if isinstance(id_E, np.ndarray) and id_E.size:
                payload[f"{prefix}/E_gap_mean/{t}"] = float(np.mean(arr) - np.mean(id_E))

    if payload:
        if epoch is not None:
            payload["epoch"] = int(epoch)
        wandb_run.log(payload, commit=False)
        
# ===========================================================
# 📉 Risk–Coverage curves (PNG for offline reporting)
# ===========================================================
def plot_risk_coverage_png(
    covs_eer: np.ndarray,
    risks_eer: np.ndarray,
    covs_1mauc: np.ndarray,
    risks_1mauc: np.ndarray,
    outdir: str,
    prefix: str = "risk_coverage"
) -> Dict[str, str]:
    """
    Save RC(EER) and RC(1-AUC) as static PNGs.
    Returns dict {"eer": path_eer, "1mauc": path_1mauc}.
    """
    os.makedirs(outdir, exist_ok=True)
    paths: Dict[str, str] = {}

    # RC(EER)
    if isinstance(covs_eer, np.ndarray) and covs_eer.size and \
       isinstance(risks_eer, np.ndarray) and risks_eer.size:
        plt.figure(figsize=(5.5, 4.0))
        plt.plot(covs_eer, risks_eer, marker="o")
        plt.xlabel("Coverage")
        plt.ylabel("Risk (EER)")
        plt.title("Risk–Coverage (EER)")
        plt.grid(True, axis="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        p_eer = os.path.join(outdir, f"{prefix}_eer.png")
        plt.savefig(p_eer, dpi=150)
        plt.close()
        paths["eer"] = p_eer

    # RC(1-AUC)
    if isinstance(covs_1mauc, np.ndarray) and covs_1mauc.size and \
       isinstance(risks_1mauc, np.ndarray) and risks_1mauc.size:
        plt.figure(figsize=(5.5, 4.0))
        plt.plot(covs_1mauc, risks_1mauc, marker="o")
        plt.xlabel("Coverage")
        plt.ylabel("Risk (1 – AUC)")
        plt.title("Risk–Coverage (1 – AUC)")
        plt.grid(True, axis="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        p_1m = os.path.join(outdir, f"{prefix}_1mauc.png")
        plt.savefig(p_1m, dpi=150)
        plt.close()
        paths["1mauc"] = p_1m

    return paths



def log_risk_coverage_dashboard(
    wandb_run,
    covs_eer, risks_eer, aurc_eer,
    covs_1mauc, risks_1mauc, aurc_1mauc,
    prefix: str = "val"
):
    """
    Side-by-side RC lines for EER and 1-AUC with AURC scalars.
    """
    if wandb_run is None:
        return
    try:
        tbl1 = wandb.Table(data=list(zip(covs_eer.tolist(), risks_eer.tolist())),
                           columns=["coverage", "risk_eer"])
        tbl2 = wandb.Table(data=list(zip(covs_1mauc.tolist(), risks_1mauc.tolist())),
                           columns=["coverage", "risk_1mauc"])
        wandb_run.log({
            f"{prefix}/RC(EER)":  wandb.plot.line(tbl1, "coverage", "risk_eer",  title="Risk–Coverage (EER)"),
            f"{prefix}/RC(1-AUC)": wandb.plot.line(tbl2, "coverage", "risk_1mauc", title="Risk–Coverage (1–AUC)"),
            f"{prefix}/aurc_eer": float(aurc_eer),
            f"{prefix}/aurc_1minusauc": float(aurc_1mauc),
        }, commit=False)
    except Exception:
        pass
