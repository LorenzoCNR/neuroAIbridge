# -*- coding: utf-8 -*-
"""Small, explicit persistence helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def _prepare_parent(path: Path) -> None:
    """Create the parent directory of an output file."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Serialize JSON-compatible data and return the written path."""
    path = Path(path)
    _prepare_parent(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_pickle(
    data: Any,
    path: str | Path,
    *,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> Path:
    """Serialize a Python object with pickle."""
    path = Path(path)
    _prepare_parent(path)

    with path.open("wb") as file:
        pickle.dump(data, file, protocol=protocol)

    return path


def load_pickle(path: str | Path) -> Any:
    """
    Load a trusted pickle file.

    Warning
    -------
    Pickle can execute arbitrary code. Only load files from trusted sources.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with path.open("rb") as file:
        return pickle.load(file)
