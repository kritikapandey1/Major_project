"""Load configs/default.yaml, apply a profile, env substitutions and CLI overrides.

Typical use (scripts get this via src/utils/cli.py):

    cfg = load_config(profile="local", overrides=["train.batch_size=8"])
    features_dir = get_path(cfg, "features")
"""

from __future__ import annotations

import copy
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")

# Path keys that hold folders/files (resolved to absolute paths).
_ROOT_KEYS = ("work_root", "persist_root", "samples", "model_cache")
# Logical name -> (root key, sub-dir key) used by get_path().
_SUBDIRS = {
    "raw": ("work_root", "raw_dir"),
    "processed": ("work_root", "processed_dir"),
    "features": ("work_root", "features_dir"),
    "splits": ("work_root", "splits_dir"),
    "checkpoints": ("persist_root", "checkpoints_dir"),
    "logs": ("persist_root", "logs_dir"),
}
REQUIRED_SECTIONS = ("seed", "device", "paths", "runtime", "preprocess", "models", "fusion", "train")


class ConfigError(ValueError):
    """Raised for any invalid or inconsistent configuration."""


def detect_profile() -> str:
    """Pick the profile automatically: DFG_PROFILE env var, else colab if on Colab, else local."""
    if os.environ.get("DFG_PROFILE"):
        return os.environ["DFG_PROFILE"]
    if "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    return "local"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: `override` merged into `base` recursively (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _substitute_env(node: Any) -> Any:
    """Replace ${env:NAME,default} in every string of a nested structure."""
    if isinstance(node, dict):
        return {k: _substitute_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_env(v) for v in node]
    if isinstance(node, str):
        def repl(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is None:
                raise ConfigError(f"environment variable {name} is not set and has no default")
            return default
        return _ENV_PATTERN.sub(repl, node)
    return node


def apply_override(cfg: dict, override: str) -> None:
    """Apply one 'dotted.key=value' override in place. The key must already exist."""
    if "=" not in override:
        raise ConfigError(f"override '{override}' must look like key.sub=value")
    dotted, raw_value = override.split("=", 1)
    keys = dotted.strip().split(".")
    node = cfg
    for i, key in enumerate(keys[:-1]):
        if not isinstance(node.get(key), dict):
            raise ConfigError(f"override '{dotted}': '{'.'.join(keys[:i + 1])}' is not a config section")
        node = node[key]
    if keys[-1] not in node:
        raise ConfigError(f"override '{dotted}': unknown key '{keys[-1]}' "
                          f"(known: {', '.join(sorted(node))})")
    node[keys[-1]] = yaml.safe_load(raw_value)  # "8" -> 8, "true" -> True, "[a,b]" -> list


def _resolve_paths(cfg: dict) -> None:
    """Turn relative root paths into absolute paths under the repo root.

    A path with a root or drive (``/content/x``, ``C:\\data``, ``\\\\server\\share``) is kept as is.
    Checking ``anchor`` instead of ``is_absolute()`` matters on Windows, where ``/content/x``
    has no drive letter and would otherwise be glued onto the repo folder.
    """
    for key in _ROOT_KEYS:
        p = Path(cfg["paths"][key]).expanduser()
        cfg["paths"][key] = str(p if p.anchor else (REPO_ROOT / p).resolve())


def _validate(cfg: dict) -> None:
    """Check required sections and a few values that later steps rely on."""
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"config is missing sections: {missing}")
    if cfg["device"] not in ("auto", "cpu", "cuda"):
        raise ConfigError(f"device must be auto, cpu or cuda (got {cfg['device']!r})")
    pre = cfg["preprocess"]
    for key in ("num_frames", "face_size", "mouth_size", "sample_rate"):
        if not isinstance(pre[key], int) or pre[key] <= 0:
            raise ConfigError(f"preprocess.{key} must be a positive integer (got {pre[key]!r})")
    fus = cfg["fusion"]
    if fus["d_model"] % fus["num_heads"] != 0:
        raise ConfigError(f"fusion.d_model ({fus['d_model']}) must be divisible by "
                          f"fusion.num_heads ({fus['num_heads']})")
    if not 0.0 <= fus["modality_dropout"] < 1.0:
        raise ConfigError("fusion.modality_dropout must be in [0, 1)")
    ratios = cfg.get("splits", {}).get("ratios")
    if ratios and abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ConfigError(f"splits.ratios must sum to 1 (got {sum(ratios.values())})")


def load_config(path: str | Path | None = None, profile: str | None = None,
                overrides: list[str] | None = None, device: str | None = None) -> dict:
    """Load the config file and return the fully resolved config dict.

    Args:
        path: YAML file; defaults to configs/default.yaml.
        profile: profile name; None = detect_profile().
        overrides: list of 'dotted.key=value' strings.
        device: if given, replaces cfg['device'] (the --device flag).
    """
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    profiles = raw.pop("profiles", {}) or {}
    profile = profile or detect_profile()
    if profile not in profiles:
        raise ConfigError(f"unknown profile '{profile}' (available: {', '.join(sorted(profiles))})")
    cfg = deep_merge(raw, profiles[profile] or {})
    cfg["profile"] = profile

    cfg = _substitute_env(cfg)
    for ov in overrides or []:
        apply_override(cfg, ov)
    if device is not None:
        cfg["device"] = device
    _validate(cfg)
    _resolve_paths(cfg)
    logger.debug("Loaded config %s with profile %s", path, profile)
    return cfg


def get_path(cfg: dict, name: str, create: bool = False) -> Path:
    """Absolute path for a logical folder name.

    Names: the roots (work_root, persist_root, samples, model_cache) and
    raw, processed, features, splits, checkpoints, logs.
    """
    if name in _ROOT_KEYS:
        p = Path(cfg["paths"][name])
    elif name in _SUBDIRS:
        root_key, sub_key = _SUBDIRS[name]
        p = Path(cfg["paths"][root_key]) / cfg["paths"][sub_key]
    else:
        raise ConfigError(f"unknown path name '{name}' "
                          f"(known: {', '.join(list(_ROOT_KEYS) + list(_SUBDIRS))})")
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_device(requested: str = "auto") -> str:
    """Map 'auto' / 'cpu' / 'cuda' to an actual device string, with a clear error for bad requests."""
    import torch  # imported here so config loading never needs torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError("--device cuda requested but no CUDA GPU is available "
                          "(this laptop is CPU-only; use Colab for GPU runs)")
    if requested not in ("cpu", "cuda"):
        raise ConfigError(f"device must be auto, cpu or cuda (got {requested!r})")
    return requested
