# -*- coding: utf-8 -*-
"""
Human-readable project artifact registry.

ProjectStore creates a stable input/output workspace and records generated
artifacts in ``project.json``. It is deliberately lightweight: the manifest
is plain JSON rather than a database.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_INPUT_KINDS = {"raw", "external", "interim", "processed", "note"}
_OUTPUT_KINDS = {"model", "embed", "figure", "log"}
_VALID_KINDS = _INPUT_KINDS | _OUTPUT_KINDS


class ProjectStore:
    """Manage project directories and a JSON artifact manifest."""

    def __init__(
        self,
        data_root: str | Path,
        out_root: str | Path,
        project_name: str,
    ) -> None:
        if not isinstance(project_name, str) or not project_name.strip():
            raise ValueError("project_name must be a non-empty string.")

        self.data_root = Path(data_root).expanduser().resolve()
        self.out_root = Path(out_root).expanduser().resolve()
        self.project_name = project_name

        self.project_input = (
            self.data_root
            / "inputs"
            / "projects"
            / project_name
        )
        self.project_output = (
            self.out_root
            / "outputs"
            / "projects"
            / project_name
        )

        for directory in (
            "raw",
            "external",
            "interim",
            "processed",
            "notes",
        ):
            (self.project_input / directory).mkdir(
                parents=True,
                exist_ok=True,
            )

        for directory in (
            "models",
            "embeds",
            "figures",
            "logs",
        ):
            (self.project_output / directory).mkdir(
                parents=True,
                exist_ok=True,
            )

        self.manifest_path = self.project_input / "project.json"
        self.manifest = self._load_or_create_manifest()

    def _load_or_create_manifest(self) -> dict[str, Any]:
        """Load an existing manifest or create a new one."""
        if not self.manifest_path.exists():
            manifest = {
                "project": self.project_name,
                "artifacts": [],
            }
            self._write_manifest(manifest)
            return manifest

        try:
            manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON manifest: {self.manifest_path}"
            ) from error

        if not isinstance(manifest, dict):
            raise ValueError("Project manifest must contain a JSON object.")

        manifest.setdefault("project", self.project_name)
        manifest.setdefault("artifacts", [])
        return manifest

    def _write_manifest(
        self,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        """Atomically write the current manifest."""
        data = self.manifest if manifest is None else manifest
        temporary_path = self.manifest_path.with_suffix(".json.tmp")

        temporary_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(self.manifest_path)

    def root_in(self) -> Path:
        """Return the input workspace root."""
        return self.project_input

    def root_out(self) -> Path:
        """Return the output workspace root."""
        return self.project_output

    def register_artifact(
        self,
        kind: str,
        name: str,
        rel_path: str,
        tags: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        *,
        require_exists: bool = False,
    ) -> dict[str, Any]:
        """
        Add an artifact entry to the manifest and return it.
        """
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}."
            )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError("rel_path must be a non-empty string.")

        resolved = (
            self.resolve_input(rel_path)
            if kind in _INPUT_KINDS
            else self.resolve_output(rel_path)
        )

        if require_exists and not resolved.exists():
            raise FileNotFoundError(
                f"Artifact file does not exist: {resolved}"
            )

        entry = {
            "kind": kind,
            "name": name,
            "path": rel_path,
            "tags": list(tags or []),
            "meta": dict(meta or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.manifest["artifacts"].append(entry)
        self._write_manifest()
        return entry

    def list_artifacts(
        self,
        kind: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """List manifest entries, optionally filtering by kind or tag."""
        artifacts = list(self.manifest.get("artifacts", []))

        if kind is not None:
            artifacts = [
                artifact
                for artifact in artifacts
                if artifact.get("kind") == kind
            ]

        if tag is not None:
            artifacts = [
                artifact
                for artifact in artifacts
                if tag in artifact.get("tags", [])
            ]

        return artifacts

    def resolve_input(self, rel_path: str | Path) -> Path:
        """Resolve a path and ensure it remains inside the input root."""
        return self._safe_resolve(self.project_input, rel_path)

    def resolve_output(self, rel_path: str | Path) -> Path:
        """Resolve a path and ensure it remains inside the output root."""
        return self._safe_resolve(self.project_output, rel_path)

    @staticmethod
    def _safe_resolve(
        root: Path,
        rel_path: str | Path,
    ) -> Path:
        """Prevent relative paths from escaping a project workspace."""
        path = Path(rel_path)

        if path.is_absolute():
            raise ValueError("rel_path must be relative.")

        resolved = (root / path).resolve()

        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Path escapes project root: {rel_path}"
            ) from error

        return resolved
