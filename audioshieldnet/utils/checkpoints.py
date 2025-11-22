# audioShieldNet/asnet_6/audioshieldnet/utils/checkpoints.py

import os, glob, torch, math
from typing import Optional, Tuple, List, Dict, Any


def _opt_group_fingerprint(opt):
    try:
        groups = getattr(opt, "param_groups", [])
        return {
            "n_groups": len(groups),
            "wd": [float(g.get("weight_decay", 0.0)) for g in groups],
            "lr": [float(g.get("lr", 0.0)) for g in groups],
        }
    except Exception:
        return {"n_groups": None, "wd": [], "lr": []}


def save_checkpoint(path, net, opt, ema, epoch, global_step, best_value, cfg):
    pkg = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict() if hasattr(opt, "state_dict") else None,
        "opt_fingerprint": _opt_group_fingerprint(opt) if opt is not None else None,
        "ema_shadow": getattr(ema, "shadow", None),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_value": None if best_value is None else float(best_value),
        "cfg": cfg,
    }
    torch.save(pkg, path)

import os
import torch

def try_resume(ckpt_path, net, opt, ema, device):
    """
    Try to resume training from `ckpt_path`.

    Returns:
      (start_epoch, global_step, best_value)

    If anything looks wrong (missing file, corrupted file, bad format), we
    print a warning and fall back to training from scratch instead of crashing.
    """
    # 1) No file → clean start
    if not os.path.isfile(ckpt_path):
        print(f"[RESUME] No checkpoint at {ckpt_path} → training from scratch.")
        return 0, 0, None

    # 2) Load checkpoint robustly
    try:
        pkg = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"[RESUME][WARN] Failed to load checkpoint {ckpt_path}: {e}")
        print("[RESUME][WARN] Ignoring checkpoint and starting from scratch.")
        return 0, 0, None

    # 3) Resolve model state dict from different possible formats
    state = None
    if isinstance(pkg, dict):
        if "model" in pkg and isinstance(pkg["model"], dict):
            state = pkg["model"]
        elif "state_dict" in pkg and isinstance(pkg["state_dict"], dict):
            state = pkg["state_dict"]
        else:
            # heuristic: maybe the whole dict is a raw state_dict
            keys = list(pkg.keys())
            looks_like_state = (
                keys
                and all(isinstance(k, str) for k in keys)
                and any(k.endswith("weight") or k.endswith("bias") for k in keys)
            )
            if looks_like_state:
                state = pkg

    if state is None:
        print(f"[RESUME][WARN] Checkpoint format unsupported or missing model weights: {ckpt_path}")
        print("[RESUME][WARN] Starting from scratch.")
        return 0, 0, None

    # 4) Load model weights (non-strict so we don’t crash on small arch changes)
    try:
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"[RESUME][WARN] Missing keys when loading model: {len(missing)}")
        if unexpected:
            print(f"[RESUME][WARN] Unexpected keys when loading model: {len(unexpected)}")
    except Exception as e:
        print(f"[RESUME][WARN] Could not load model weights from {ckpt_path}: {e}")
        print("[RESUME][WARN] Starting from scratch.")
        return 0, 0, None

    # 5) Optimizer (tolerant to group mismatch) — your original logic, but safe
    if pkg.get("optimizer") is not None and opt is not None:
        try:
            cur_fp = _opt_group_fingerprint(opt)
            old_fp = pkg.get("opt_fingerprint", None)
            if old_fp and old_fp.get("n_groups", None) != cur_fp.get("n_groups", None):
                print(
                    f"[RESUME] Optimizer groups differ "
                    f"(ckpt={old_fp.get('n_groups')} vs cur={cur_fp.get('n_groups')}) "
                    f"→ skipping optimizer state."
                )
            else:
                opt.load_state_dict(pkg["optimizer"])
        except ValueError as e:
            # Typical: "different number of parameter groups"
            print(f"[RESUME] Skipping optimizer state: {e}")
        except Exception as e:
            print(f"[RESUME] Could not load optimizer state (ignored): {e}")

    # 6) EMA
    if pkg.get("ema_shadow") is not None and ema is not None:
        try:
            ema.shadow = {k: v.clone().to(device) for k, v in pkg["ema_shadow"].items()}
        except Exception as e:
            print(f"[RESUME][WARN] Could not restore EMA shadow (ignored): {e}")

    start_epoch = int(pkg.get("epoch", 0))
    global_step = int(pkg.get("global_step", 0))
    best_value  = pkg.get("best_value", None)

    print(
        f"[RESUME] Loaded checkpoint {ckpt_path} "
        f"(epoch={start_epoch}, global_step={global_step}, best_value={best_value})."
    )
    return start_epoch, global_step, best_value


def metric_better(a, b, mode):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return False
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return True
    return (a > b) if mode == "max" else (a < b)


def prune_topk(ckpt_dir, keep_top_k):
    snaps = sorted(glob.glob(os.path.join(ckpt_dir, "topk", "*.ckpt")))
    if len(snaps) <= keep_top_k:
        return
    for p in snaps[:len(snaps) - keep_top_k]:
        try:
            os.remove(p)
        except Exception:
            pass


def load_weights_into(net, ckpt_path, device, prefer_ema=True):
    ckpt = torch.load(ckpt_path, map_location=device)

    # choose weights block (prefer EMA if present)
    state = None
    if prefer_ema:
        # trainer saves ema as "ema_shadow"
        if isinstance(ckpt.get("ema_shadow"), dict):
            state = ckpt["ema_shadow"]
        elif isinstance(ckpt.get("ema"), dict):
            # other runs may have saved {"ema": {"shadow": {...}}} or direct dict
            state = ckpt["ema"].get("shadow", ckpt["ema"])

    if state is None:
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("net")

    if state is None:
        raise KeyError(
            f"No state_dict found in {ckpt_path} (looked for 'ema_shadow','ema','model','state_dict','net')."
        )

    # strip possible "module." from DDP
    fixed = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    try:
        net.load_state_dict(fixed, strict=True)
    except RuntimeError:
        net.load_state_dict(fixed, strict=False)

    return ckpt


# ============================================================
# NEW: Helpers for checkpoint selection (best/last/topkN/auto)
# ============================================================

def list_sorted_topk(ckpt_dir: str) -> Tuple[List[str], str, str]:
    """
    Inspect checkpoints/topk/*.ckpt, sort them by stored best_value
    using cfg['train']['best_metric'] and best_mode ('min'/'max').

    Returns:
        paths_sorted: list of ckpt paths (best → worst)
        metric_name:  metric used for ranking
        best_mode:    'min' or 'max'
    """
    topk_dir = os.path.join(ckpt_dir, "topk")
    if not os.path.isdir(topk_dir):
        raise FileNotFoundError(f"[ckpt/topk] Directory does not exist: {topk_dir}")

    files = [f for f in os.listdir(topk_dir) if f.endswith(".ckpt")]
    if not files:
        raise FileNotFoundError(f"[ckpt/topk] No .ckpt files in {topk_dir}")

    entries = []
    metric_name = "auc"
    best_mode = "max"

    for fname in files:
        fpath = os.path.join(topk_dir, fname)
        try:
            state = torch.load(fpath, map_location="cpu")
        except Exception:
            # If load fails, push it to the bottom
            entries.append((float("inf"), "min", fpath))
            continue

        cfg = state.get("cfg", {}) if isinstance(state, dict) else {}
        train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
        metric_name = str(train_cfg.get("best_metric", "auc")).lower()
        best_mode = str(train_cfg.get("best_mode", "max")).lower()

        # For these metrics, lower is better
        if metric_name in ["eer", "ece", "fnr95_fgsm", "susp_frac"]:
            best_mode = "min"
        else:
            best_mode = "max"

        best_value = state.get("best_value", None)
        if best_value is None:
            best_value = float("inf") if best_mode == "min" else float("-inf")

        entries.append((float(best_value), best_mode, fpath))

    if not entries:
        raise FileNotFoundError(f"[ckpt/topk] No usable checkpoints in {topk_dir}")

    # Assume best_mode is consistent; use from first entry
    _, mode, _ = entries[0]
    if mode == "min":
        entries.sort(key=lambda x: x[0])  # smallest is best
    else:
        entries.sort(key=lambda x: x[0], reverse=True)  # largest is best

    paths_sorted = [p for _, _, p in entries]
    return paths_sorted, metric_name, mode


def resolve_checkpoint_default(ckpt_dir: str, override: Optional[str]) -> Tuple[str, str]:
    """
    Old behavior:
      - if override is a file, use it
      - else best.ckpt, else last.ckpt

    Returns: (path, label) where label is 'override' | 'best' | 'last'
    """
    if override and os.path.isfile(override):
        return override, "override"

    best = os.path.join(ckpt_dir, "best.ckpt")
    last = os.path.join(ckpt_dir, "last.ckpt")

    if os.path.isfile(best):
        return best, "best"
    if os.path.isfile(last):
        return last, "last"

    raise FileNotFoundError(
        f"[ckpt] No checkpoint found at {best} or {last} (override via --ckpt)."
    )


def resolve_checkpoint_by_choice(
    ckpt_dir: str,
    override: Optional[str],
    eval_cfg: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Decide which checkpoint to use based on:
      1) CLI override (--ckpt) → highest priority
      2) eval.ckpt_choose in YAML:
           'auto'  → best.ckpt else last.ckpt
           'best'  → best.ckpt
           'last'  → last.ckpt
           'topk1' → best model in checkpoints/topk (rank 1)
           'topk2' → 2nd-best model in checkpoints/topk
           'topk3' → 3rd-best model in checkpoints/topk
    Returns: (path, label) where label is 'override', 'best', 'last', 'topk1', etc.
    """
    # CLI override wins
    if override and os.path.isfile(override):
        return override, "override"

    choice = str(eval_cfg.get("ckpt_choose", "auto")).lower().strip()

    # Auto: old behavior (best → last)
    if choice == "auto":
        return resolve_checkpoint_default(ckpt_dir, override=None)

    # Explicit BEST / LAST
    if choice == "best":
        path = os.path.join(ckpt_dir, "best.ckpt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[ckpt_choose=best] Not found: {path}")
        return path, "best"

    if choice == "last":
        path = os.path.join(ckpt_dir, "last.ckpt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[ckpt_choose=last] Not found: {path}")
        return path, "last"

    # topkN → Nth best in checkpoints/topk
    if choice.startswith("topk"):
        idx_str = choice.replace("topk", "")
        try:
            rank = int(idx_str)
        except ValueError:
            raise ValueError(f"[eval.ckpt_choose] Invalid topk index: {choice}")

        if rank < 1:
            raise ValueError(f"[eval.ckpt_choose] topk index must be >=1, got {rank}")

        paths_sorted, _, _ = list_sorted_topk(ckpt_dir)
        if rank > len(paths_sorted):
            raise FileNotFoundError(
                f"[ckpt_choose={choice}] Only {len(paths_sorted)} top-k checkpoints available in {os.path.join(ckpt_dir,'topk')}."
            )

        path = paths_sorted[rank - 1]  # 1-based rank
        return path, f"topk{rank}"

    raise ValueError(f"[eval.ckpt_choose] Unsupported value: {choice}")
