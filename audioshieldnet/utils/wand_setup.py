# audioshieldnet/utils/wand_setup.py
from __future__ import annotations
import os
import numbers
import numpy as np
import torch
import wandb


def to_serializable(x, tensor_limit: int = 1024, array_limit: int = 1024):
    """Convert nested dict/list structures so W&B can safely serialize them."""
    if isinstance(x, (str, bool, numbers.Number)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= array_limit else f"<ndarray shape={x.shape} dtype={x.dtype}>"
    if isinstance(x, torch.Tensor):
        n = x.numel()
        return x.detach().cpu().tolist() if n <= tensor_limit else f"<tensor shape={tuple(x.shape)} dtype={x.dtype}>"
    if isinstance(x, dict):
        return {k: to_serializable(v, tensor_limit, array_limit) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_serializable(v, tensor_limit, array_limit) for v in x]
    try:
        return str(x)
    except Exception:
        return "<unserializable>"


def wandb_log_safe(wb: wandb.wandb_sdk.wandb_run.Run, payload: dict, commit: bool = False):
    """Log a dict to W&B while avoiding raw arrays/tensors in the payload."""
    def s(v):
        if isinstance(v, (str, bool, numbers.Number)) or v is None: return v
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, np.ndarray): return f"<ndarray shape={v.shape} dtype={v.dtype}>"
        if isinstance(v, torch.Tensor): return f"<tensor shape={tuple(v.shape)} dtype={v.dtype}>"
        return v  # Tables/plots/artifacts are fine
    wb.log({k: s(v) for k, v in payload.items()}, commit=commit)


def save_npz_artifact(wb: wandb.wandb_sdk.wandb_run.Run,
                      out_dir: str,
                      epoch: int,
                      arrays: dict,
                      artifact_name: str = "risk_coverage_curves",
                      artifact_type: str = "metrics") -> str:
    """
    Save arrays into a compressed .npz and upload as a W&B artifact.
    Returns the saved path.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{artifact_name}_{epoch}.npz")
    np.savez_compressed(path, **arrays)
    art = wandb.Artifact(artifact_name, type=artifact_type)
    art.add_file(path)
    wb.log_artifact(art)
    return path


def init_wandb(cfg: dict, wcfg: dict, outdir: str, run_id: str | None = None,
               watch: str | None = None, init_timeout: int = 300):
    """
    Initialize W&B with a sanitized config and safe defaults.
    """
    settings = wandb.Settings(init_timeout=init_timeout, code_dir=None)
    cfg_serializable = to_serializable(cfg)

    try:
        run = wandb.init(
            project=wcfg.get('project', 'audioshieldnet'),
            entity=wcfg.get('entity', None),
            name=wcfg.get('run_name', None),
            config=cfg_serializable,          # sanitized cfg
            dir=outdir,
            id=run_id,
            resume="allow",
            settings=settings,
        )
    except Exception as e:
        os.environ["WANDB_MODE"] = "offline"
        print(f"[W&B] Online init failed ({e}). Falling back to OFFLINE mode.")
        run = wandb.init(
            project=wcfg.get('project', 'audioshieldnet'),
            entity=wcfg.get('entity', None),
            name=wcfg.get('run_name', None),
            config=cfg_serializable,          # sanitized cfg
            dir=outdir,
            id=run_id,
            resume="allow",
            settings=settings,
        )

    # NOTE: Do NOT call wandb.watch() here; do it in the caller with a real model instance.
    return run
