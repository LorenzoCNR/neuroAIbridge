# -*- coding: utf-8 -*-
"""
Compact object-inspection helpers for scientific Python workflows.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch


def _brief(value: Any, max_length: int = 120) -> str:
    """Return a compact one-line description of a Python object."""
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device}, requires_grad={value.requires_grad})"
        )

    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"

    if isinstance(value, dict):
        keys = list(value)[:6]
        suffix = "..." if len(value) > 6 else ""
        return f"dict(len={len(value)}, keys={keys}{suffix})"

    if isinstance(value, (list, tuple, set)):
        preview = list(value)[:4]
        types = sorted({type(item).__name__ for item in preview})
        return (
            f"{type(value).__name__}(len={len(value)}, "
            f"head_types={types})"
        )

    if inspect.ismodule(value):
        return f"module({value.__name__})"

    if inspect.isclass(value):
        return f"class({value.__module__}.{value.__name__})"

    if callable(value):
        name = getattr(value, "__name__", type(value).__name__)
        module = getattr(value, "__module__", "?")
        return f"callable({module}.{name})"

    try:
        text = str(value)
    except Exception:
        text = f"<unprintable {type(value).__name__}>"

    if len(text) > max_length:
        text = text[: max_length - 3] + "..."

    return f"{type(value).__name__}({text})"


def explore_obj(
    obj: Any,
    *,
    show_private: bool = False,
    max_attrs: int = 60,
) -> None:
    """
    Print a structured, read-only summary of an object.

    This helper is intended for interactive debugging in Spyder, IPython,
    notebooks, and the Python console.
    """
    if not isinstance(max_attrs, int) or max_attrs < 1:
        raise ValueError("max_attrs must be a positive integer.")

    cls = obj.__class__

    print(f"object: {cls.__module__}.{cls.__name__}")
    print(f"bases: {[base.__name__ for base in cls.__bases__]}")
    print(f"mro: {[entry.__name__ for entry in cls.__mro__]}")

    inner = getattr(obj, "model", None)
    if inner is not None:
        inner_cls = inner.__class__
        print(
            "wrapped model: "
            f"{inner_cls.__module__}.{inner_cls.__name__}"
        )

    if hasattr(obj, "__dict__"):
        keys = list(obj.__dict__)[:max_attrs]
        print(f"instance attributes: {keys}")

    names = [
        name
        for name in dir(obj)
        if not (name.startswith("__") and name.endswith("__"))
    ]

    if not show_private:
        names = [name for name in names if not name.startswith("_")]

    print("attributes:")

    for name in names[:max_attrs]:
        try:
            value = getattr(obj, name)
            description = _brief(value)
        except Exception as error:
            description = f"<inaccessible: {error}>"

        print(f"  - {name}: {description}")
