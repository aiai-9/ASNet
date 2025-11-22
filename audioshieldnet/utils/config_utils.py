# audioShieldNet/asnet_1/audioshieldnet/utils/config_utils.py
"""
Utility helpers for YAML configs (e.g., placeholder and environment variable expansion).
"""

import os

def resolve_placeholders(obj, mapping):
    """
    Recursively replace ${var} placeholders and expand environment variables in a nested dict/list.

    Example:
        mapping = {"ckpt_dir": "asn_asv21_2"}
        resolve_placeholders({"path": "exp/${ckpt_dir}"}, mapping)
        -> {"path": "exp/asn_asv21_2"}
    """
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(v, mapping) for v in obj]
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace(f"${{{k}}}", str(v))
        return os.path.expandvars(obj)
    return obj
