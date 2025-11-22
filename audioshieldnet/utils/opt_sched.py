# audioShieldNet/asnet_4/audioshieldnet/utils/opt_sched.py


from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, SequentialLR, LinearLR
)

# ============================================================
#  SWA helpers  (supports two-input feature extractors)
# ============================================================

def swa_update_bn_two_input(loader, swa_model, feats, device: str, max_batches: int = 200):
    """Refresh BN running stats for a two-input model using the train loader."""
    import torch.nn as nn
    swa_model.train()

    # Reset BN stats if stale
    for m in swa_model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if hasattr(m, "running_mean") and m.running_mean is not None:
                m.running_mean.zero_()
            if hasattr(m, "running_var") and m.running_var is not None:
                m.running_var.fill_(1)
            if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()

    n = 0
    with torch.no_grad():
        for batch in loader:
            wav = batch[0].to(device, non_blocking=True)
            logmel, phmel = feats(wav)
            _ = swa_model(logmel, phmel, target=None)
            n += 1
            if n >= max_batches:
                break


# ============================================================
#  Optimizer builder
# ============================================================

def _param_groups_no_wd(model: torch.nn.Module, weight_decay: float):
    """Split params so bias/Norm layers get 0 weight decay."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(cfg: Dict, model: torch.nn.Module) -> Tuple[torch.optim.Optimizer, bool]:
    """
    Returns: (optimizer, use_sam)
    - AdamW with bias/Norm excluded from weight decay
    - Optionally wraps with SAM if cfg.optim.use_sam = true
    """
    tcfg = cfg.get("train", {}) or {}
    ocfg = cfg.get("optim", {}) or {}
    use_sam = bool(ocfg.get("use_sam", False))
    wd = float(tcfg.get("weight_decay", 1e-4))
    lr = float(tcfg.get("lr", 2e-4))

    param_groups = _param_groups_no_wd(model, wd)
    base_opt = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

    if use_sam:
        from audioshieldnet.models.sam_optimizer import SAM
        rho_val = float(ocfg.get("sam_rho", 0.05))
        for g in base_opt.param_groups:
            g.setdefault("rho", rho_val)
            g.setdefault("adaptive", False)
        opt = SAM(model.parameters(), base_optimizer=base_opt, rho=rho_val, adaptive=False)
        return opt, True

    return base_opt, False


# ============================================================
#  Scheduler builder  (all PyTorch-2.5-safe)
# ============================================================

def build_scheduler(cfg: Dict, optimizer: torch.optim.Optimizer,
                    steps_per_epoch: int, total_epochs: int) -> Tuple[torch.optim.lr_scheduler._LRScheduler | None, bool]:
    """
    Returns:
      (scheduler, sched_per_batch)
        sched_per_batch = True  → call .step() each batch
        sched_per_batch = False → call .step() each epoch
    """
    ocfg = (cfg.get("optim", {}) or {}).get("scheduler", {}) or {}
    name = (ocfg.get("name", "cosine") or "cosine").lower()

    opt_for_sched = getattr(optimizer, "base_optimizer", optimizer)

    steps_per_epoch = max(1, int(steps_per_epoch))
    total_epochs = max(1, int(total_epochs))
    total_steps = max(1, steps_per_epoch * total_epochs)

    if name == "none":
        return None, False

    # ---------- CosineAnnealingLR (per-batch with warmup)
    if name == "cosine":
        warmup_steps = int(ocfg.get("warmup_steps", 1000))
        min_lr = float(ocfg.get("min_lr", 1e-6))
        warmup_steps = min(warmup_steps, max(1, total_steps // 2))

        warm = LinearLR(opt_for_sched,
                        start_factor=float(ocfg.get("warmup_start_factor", 1e-3)),
                        end_factor=1.0,
                        total_iters=max(1, warmup_steps),
                        last_epoch=-1)
        cos = CosineAnnealingLR(opt_for_sched,
                                T_max=max(1, total_steps - warmup_steps),
                                eta_min=min_lr,
                                last_epoch=-1)
        sched = SequentialLR(opt_for_sched, schedulers=[warm, cos],
                             milestones=[warmup_steps])
        return sched, True  # per-batch

    # ---------- CosineAnnealingLR (per-epoch)
    if name in ("cosine_epoch", "cosine_ep"):
        warmup_epochs = int(ocfg.get("warmup_epochs", max(1, int(0.1 * total_epochs))))
        min_lr = float(ocfg.get("min_lr", 1e-6))
        warmup_epochs = min(warmup_epochs, max(1, total_epochs // 2))

        warm = LinearLR(opt_for_sched,
                        start_factor=float(ocfg.get("warmup_start_factor", 1e-3)),
                        end_factor=1.0,
                        total_iters=max(1, warmup_epochs),
                        last_epoch=-1)
        cos = CosineAnnealingLR(opt_for_sched,
                                T_max=max(1, total_epochs - warmup_epochs),
                                eta_min=min_lr,
                                last_epoch=-1)
        sched = SequentialLR(opt_for_sched, schedulers=[warm, cos],
                             milestones=[warmup_epochs])
        return sched, False  # per-epoch

    # ---------- OneCycleLR (per-batch)
    if name == "onecycle":
        max_lr = float(ocfg.get("onecycle_max_lr",
                     (cfg.get("train", {}) or {}).get("lr", 2e-4) * 2.0))
        t_steps = max(2, total_steps)
        sched = OneCycleLR(opt_for_sched,
                           max_lr=max_lr,
                           total_steps=t_steps,
                           pct_start=float(ocfg.get("pct_start", 0.1)),
                           anneal_strategy=str(ocfg.get("anneal_strategy", "cos")),
                           div_factor=float(ocfg.get("div_factor", 25.0)),
                           final_div_factor=float(ocfg.get("final_div_factor", 1e4)),
                           last_epoch=-1)
        return sched, True

    # ---------- ReduceLROnPlateau (per-epoch)
    if name == "plateau":
        factor = float(ocfg.get("plateau_factor", 0.5))
        patience = int(ocfg.get("plateau_patience", 2))
        cooldown = int(ocfg.get("plateau_cooldown", 0))
        min_lr = float(ocfg.get("min_lr", 0.0))
        thresh_mode = str(ocfg.get("threshold_mode", "rel"))
        thresh = float(ocfg.get("threshold", 1e-4))
        sched = ReduceLROnPlateau(opt_for_sched,
                                  mode="max",
                                  factor=factor,
                                  patience=patience,
                                  threshold=thresh,
                                  threshold_mode=thresh_mode,
                                  cooldown=cooldown,
                                  min_lr=min_lr,
                                  verbose=False)
        return sched, False

    # ---------- Fallback
    return None, False


# ============================================================
#  SWA toggles
# ============================================================

def swa_is_active(cfg: Dict) -> bool:
    return bool(((cfg.get("optim", {}) or {}).get("swa", {}) or {}).get("enable", False))


def swa_should_update(cfg: Dict, epoch: int, total_epochs: int) -> bool:
    sc = ((cfg.get("optim", {}) or {}).get("swa", {}) or {})
    start_frac = float(sc.get("start_epoch", 0.8))
    start_ep = max(1, int(start_frac * total_epochs))
    return epoch >= start_ep


def swa_make_model(model: torch.nn.Module):
    from torch.optim.swa_utils import AveragedModel
    return AveragedModel(model)


def swa_update_bn(loader, swa_model, device: str):
    from torch.optim.swa_utils import update_bn
    swa_model.train()
    update_bn(loader, swa_model, device=device)


# ============================================================
#  ε-curriculum helper
# ============================================================

def schedule_adv_eps(cfg: Dict, global_step: int, total_steps: int) -> float:
    """Allow adv_eps to be scalar or a list for curriculum across training horizon."""
    sec = cfg.get("security", {}) or {}
    eps = sec.get("adv_eps", 0.001)
    if isinstance(eps, (list, tuple)):
        q = global_step / max(1.0, float(total_steps))
        idx = min(int(q * len(eps)), len(eps) - 1)
        return float(eps[idx])
    return float(eps)
