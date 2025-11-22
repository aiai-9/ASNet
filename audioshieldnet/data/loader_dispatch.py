# audioshieldnet/data/loader_dispatch.py
from __future__ import annotations
import importlib
from typing import Callable, Tuple, Optional, Dict, Any

# --------------------------------------------------------------------------------------
# Canonical registry: friendly names/aliases -> python module path
# Update the right-hand side strings if your module paths differ.
# --------------------------------------------------------------------------------------
_REGISTRY: Dict[str, str] = {
    # Canonical names
    "asvspoof21_split": "audioshieldnet.data.asvspoof21_split",
    "librisevoc_split": "audioshieldnet.data.librisevoc_split",
    "wavefake":   "audioshieldnet.data.wavefake",     # <-- adjust if needed
    "codecfake_split":      "audioshieldnet.data.codecfake_split",    # <-- adjust if needed
    "for":              "audioshieldnet.data.fakeOrReal",   # <-- adjust if needed
    "multi":            "audioshieldnet.data.multi",

    # Short aliases → canonical
    "asvspoof":         "audioshieldnet.data.asvspoof21_split",
    "librisevoc":       "audioshieldnet.data.librisevoc_split",
    "wavefake":         "audioshieldnet.data.wavefake",
    "codec":            "audioshieldnet.data.codecfake_split",
    "for":              "audioshieldnet.data.fakeOrReal",
    "fakeorreal":       "audioshieldnet.data.fakeOrReal",
    "fake_or_real":     "audioshieldnet.data.fakeOrReal",
    "multi_mix":        "audioshieldnet.data.multi",
    "mix":              "audioshieldnet.data.multi",
}

def register_dataset(name: str, module_path: str) -> None:
    """
    Runtime extension: register/override a dataset name or alias.
    Example:
        register_dataset("wavefake", "myproj.data.wavefake_data")
    """
    key = _norm(name)
    _REGISTRY[key] = module_path

def _norm(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")

def _load_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise RuntimeError(f"[data] Could not import module '{name}'.") from e

def _module_has_builders(mod) -> bool:
    return all(hasattr(mod, fn) and callable(getattr(mod, fn))
               for fn in ("build_dataloaders", "build_testloader"))

def _fallback_to_multi_if_possible(cfg: Dict[str, Any]):
    """
    If cfg suggests multi-source training (sources/auto_prepare/out_dir),
    try to use the 'multi' combiner.
    """
    d = (cfg.get("data", {}) or {})
    has_multi_signals = bool(d.get("sources")) or d.get("auto_prepare") or d.get("out_dir")
    if has_multi_signals:
        mod_path = _REGISTRY.get("multi", "audioshieldnet.data.multi")
        mod = _load_module(mod_path)
        if _module_has_builders(mod):
            return mod
    return None

def _resolve_module_from_cfg(cfg: Dict[str, Any]):
    d = (cfg.get("data", {}) or {})

    # 1) Highest priority: explicit module path in YAML
    mod_path = d.get("module", None)
    if isinstance(mod_path, str) and mod_path.strip():
        mod = _load_module(mod_path)
        if not _module_has_builders(mod):
            raise RuntimeError(f"[data] Module '{mod_path}' does not define "
                               f"build_dataloaders/build_testloader.")
        return mod

    # 2) Named dataset with aliases
    data_name = d.get("name", None)
    if isinstance(data_name, str) and data_name.strip():
        key = _norm(data_name)
        mod_path = _REGISTRY.get(key)
        if mod_path:
            mod = _load_module(mod_path)
            if _module_has_builders(mod):
                return mod
            # If alias points to missing module but multi-mix is configured, try multi fallback
            mod_fallback = _fallback_to_multi_if_possible(cfg)
            if mod_fallback:
                return mod_fallback
            raise RuntimeError(f"[data] Dataset '{data_name}' → '{mod_path}' imported, "
                               f"but it lacks required builders.")
        # Unknown name → attempt multi fallback if multi signals exist
        mod_fallback = _fallback_to_multi_if_possible(cfg)
        if mod_fallback:
            return mod_fallback

        known = ", ".join(sorted(_REGISTRY.keys()))
        raise RuntimeError(f"[data] Unknown dataset name '{data_name}'. Known: {known}\n"
                           f"Tip: set data.module: audioshieldnet.data.multi if you use mixed sources.")

    # 3) Nothing specified: try multi if sources hint is present
    mod_fallback = _fallback_to_multi_if_possible(cfg)
    if mod_fallback:
        return mod_fallback

    known = ", ".join(sorted(_REGISTRY.keys()))
    raise RuntimeError("[data] Neither data.module nor data.name specified, and no multi-source hints found.\n"
                       f"Known names: {known}\n"
                       f"Tip: set data.module: audioshieldnet.data.multi or provide data.name.")

def resolve_data_builders(cfg: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Returns (build_dataloaders, build_testloader) based on cfg['data'].
    Priority:
      1) data.module (explicit)
      2) data.name with aliases (registry)
      3) fallback to 'multi' if sources/auto_prepare/out_dir exist
    """
    mod = _resolve_module_from_cfg(cfg)
    b_tr = getattr(mod, "build_dataloaders", None)
    b_te = getattr(mod, "build_testloader", None)
    if not (callable(b_tr) and callable(b_te)):
        raise RuntimeError(f"[data] Module {getattr(mod, '__name__', str(mod))} is missing required builders.")
    return b_tr, b_te
