# -*- coding: utf-8 -*-
"""YAML configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_params(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Empty YAML files return an empty dictionary.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            "The top level of the YAML configuration must be a mapping."
        )

    return config


def load_model_params(
    path: str | Path,
    model_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load fixed parameters and hyperparameter grid for one model type.
    """
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must be a non-empty string.")

    config = load_params(path)

    if model_type not in config:
        raise ValueError(
            f"Model type {model_type!r} is missing from {Path(path)}."
        )

    section = config[model_type]

    if not isinstance(section, Mapping):
        raise ValueError(
            f"Configuration section {model_type!r} must be a mapping."
        )

    fixed = section.get("fixed") or {}
    grid = section.get("grid") or {}

    if not isinstance(fixed, Mapping):
        raise ValueError(f"{model_type}.fixed must be a mapping.")
    if not isinstance(grid, Mapping):
        raise ValueError(f"{model_type}.grid must be a mapping.")

    return dict(fixed), dict(grid)


def merge_overrides(
    base: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Deep-copy a base configuration and apply top-level overrides.
    """
    if not isinstance(base, Mapping):
        raise TypeError("base must be a mapping.")
    if override is not None and not isinstance(override, Mapping):
        raise TypeError("override must be a mapping or None.")

    merged = deepcopy(dict(base))
    merged.update(dict(override or {}))
    return merged


def validate_required(
    config: Mapping[str, Any],
    required_keys: Sequence[str],
) -> None:
    """
    Raise an error when required configuration keys are missing.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")

    missing = [key for key in required_keys if key not in config]

    if missing:
        raise ValueError(
            "Missing required configuration keys: "
            + ", ".join(map(str, missing))
        )
