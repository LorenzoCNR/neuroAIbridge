# -*- coding: utf-8 -*-
"""Project path construction and environment setup."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .io import save_json


LOGGER = logging.getLogger(__name__)


def find_project_root(
    start: str | Path | None = None,
    markers: tuple[str, ...] = ("pyproject.toml", ".git"),
) -> Path:
    """
    Search upward for a directory containing a project marker.
    """
    current = Path(start or Path.cwd()).resolve()

    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate

    raise FileNotFoundError(
        f"Could not find a project root from {current} "
        f"using markers {markers}."
    )


def project_paths(
    data_root: str | Path,
    project_name: str,
    *,
    create: bool = False,
) -> tuple[Path, Path]:
    """
    Return canonical input and output directories for one project.

    Layout:
        <data_root>/projects/<project_name>/input_
        <data_root>/projects/<project_name>/output_
    """
    if not isinstance(project_name, str) or not project_name.strip():
        raise ValueError("project_name must be a non-empty string.")

    project_directory = (
        Path(data_root).expanduser().resolve()
        / "projects"
        / project_name
    )
    input_directory = project_directory / "input_"
    output_directory = project_directory / "output_"

    if create:
        input_directory.mkdir(parents=True, exist_ok=True)
        output_directory.mkdir(parents=True, exist_ok=True)

    return input_directory, output_directory


def dump_experiment_config(
    output_dir: str | Path,
    config: dict,
    filename: str = "config.json",
) -> Path:
    """Write a reproducibility snapshot to an output directory."""
    if not filename:
        raise ValueError("filename must be non-empty.")

    return save_json(config, Path(output_dir) / filename)


def setup_paths(
    project_root: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    pipe_path: str | Path | None = None,
    change_dir: bool = False,
    parents: bool = False,
) -> dict[str, Path | None]:
    """
    Resolve project, data, output, and optional external-code paths.

    ``parents=False`` preserves the historical behavior and requires data and
    output directories to exist. ``parents=True`` creates missing directories.
    """
    root = Path(project_root).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {root}")

    def resolve_directory(
        value: str | Path,
        *,
        name: str,
    ) -> Path:
        directory = Path(value).expanduser()

        if not directory.is_absolute():
            directory = root / directory

        directory = directory.resolve()

        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(
                f"{name} is not a directory: {directory}"
            )

        if not directory.exists():
            if parents:
                directory.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"{name} does not exist: {directory}"
                )

        return directory

    resolved_data = resolve_directory(data_dir, name="data_dir")
    resolved_output = resolve_directory(out_dir, name="out_dir")

    resolved_pipeline = None

    if pipe_path is not None:
        resolved_pipeline = Path(pipe_path).expanduser()

        if not resolved_pipeline.is_absolute():
            resolved_pipeline = root / resolved_pipeline

        resolved_pipeline = resolved_pipeline.resolve()

        if not resolved_pipeline.is_dir():
            raise FileNotFoundError(
                f"pipe_path does not exist: {resolved_pipeline}"
            )

        pipeline_string = str(resolved_pipeline)

        if pipeline_string not in sys.path:
            sys.path.insert(0, pipeline_string)

    if change_dir:
        os.chdir(root)

    LOGGER.info("Project root: %s", root)
    LOGGER.info("Data directory: %s", resolved_data)
    LOGGER.info("Output directory: %s", resolved_output)

    return {
        "project_root": root,
        "data_dir": resolved_data,
        "out_dir": resolved_output,
        "pipeline_path": resolved_pipeline,
    }
