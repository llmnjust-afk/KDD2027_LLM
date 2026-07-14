"""Centralized configuration loading from YAML files."""

from __future__ import annotations

import os
from pathlib import Path
from copy import deepcopy
from typing import Any

import yaml

_CACHE: dict[str, dict] = {}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"


def load_config(name: str = "default") -> dict[str, Any]:
    """Load a YAML config file by name (without extension).

    Cached. Merges `default.yaml` as a base when a non-default name is given.
    """
    if name in _CACHE:
        return deepcopy(_CACHE[name])

    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    if name != "default":
        base = load_config("default")
        cfg = _deep_merge(base, cfg)

    _CACHE[name] = cfg
    return deepcopy(cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def get_path(key: str, cfg: dict | None = None) -> Path:
    """Resolve a path from the `paths` section, relative to project root."""
    cfg = cfg or load_config()
    raw = cfg["paths"][key]
    p = Path(raw)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def ensure_dirs(cfg: dict | None = None) -> None:
    """Create all directories listed under `paths`."""
    cfg = cfg or load_config()
    for v in cfg["paths"].values():
        p = Path(v)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)


def env_or_cfg(env_key: str, cfg: dict, *path_keys: str, default=None) -> Any:
    """Read from environment first, then nested config, then default."""
    val = os.environ.get(env_key)
    if val is not None:
        return val
    node = cfg
    for k in path_keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node if node is not None else default
