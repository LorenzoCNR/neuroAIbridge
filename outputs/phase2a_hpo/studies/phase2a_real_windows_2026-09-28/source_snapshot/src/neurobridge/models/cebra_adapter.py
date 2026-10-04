# -*- coding: utf-8 -*-
"""Adapter around the official CEBRA estimator.

The adapter keeps CEBRA-specific code in one module and provides a small,
validated API for single-session and multi-session workflows.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.io import savemat

try:
    from cebra import CEBRA
    from cebra.data.helper import OrthogonalProcrustesAlignment
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    CEBRA = None  # type: ignore[assignment]
    OrthogonalProcrustesAlignment = None  # type: ignore[assignment]
    _CEBRA_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _CEBRA_IMPORT_ERROR = None


Array = np.ndarray
ArrayOrList = Union[Array, Sequence[Array]]


class CebraAdapter:
    """Thin wrapper around the official :class:`cebra.CEBRA` estimator.

    Parameters
    ----------
    params:
        Keyword arguments forwarded directly to ``CEBRA(**params)``.

    Notes
    -----
    CEBRA is treated as an optional dependency at module-import time. Creating
    ``CebraAdapter`` still requires the package to be installed.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        if CEBRA is None:
            raise ImportError(
                "CEBRA is required to create CebraAdapter. Install it with "
                "`pip install cebra`."
            ) from _CEBRA_IMPORT_ERROR

        self.params: Dict[str, Any] = dict(params or {})
        self.model = CEBRA(**self.params)

        self._last_Z: Optional[np.ndarray] = None
        self._is_multi = False
        self._n_sessions = 0
        self._is_fitted = False

    def fit(
        self,
        X: ArrayOrList,
        y: Optional[ArrayOrList] = None,
    ) -> "CebraAdapter":
        """Fit CEBRA on one session or a sequence of sessions.

        Parameters
        ----------
        X:
            A single array or a sequence of session arrays.
        y:
            Optional auxiliary variables. In multi-session mode this must be a
            sequence parallel to ``X``.
        """
        if _is_session_sequence(X):
            X_list = [_validate_feature_array(x, name=f"X[{i}]") for i, x in enumerate(X)]

            if len(X_list) == 0:
                raise ValueError("X must contain at least one session.")

            if y is None:
                self.model.fit(X_list)
            else:
                if not _is_session_sequence(y):
                    raise TypeError(
                        "In multi-session mode, y must be a list or tuple "
                        "parallel to X."
                    )

                y_list = [
                    _ensure_2d(_as_float32(yy), name=f"y[{i}]")
                    for i, yy in enumerate(y)
                ]

                if len(X_list) != len(y_list):
                    raise ValueError(
                        "X and y must contain the same number of sessions, "
                        f"got {len(X_list)} and {len(y_list)}."
                    )

                for session_index, (X_session, y_session) in enumerate(
                    zip(X_list, y_list)
                ):
                    if len(X_session) != len(y_session):
                        raise ValueError(
                            f"Session {session_index} has inconsistent lengths: "
                            f"len(X)={len(X_session)}, len(y)={len(y_session)}."
                        )

                self.model.fit(X_list, y_list)

            self._is_multi = True
            self._n_sessions = len(X_list)
            self._is_fitted = True
            return self

        X_array = _validate_feature_array(X, name="X")

        if y is None:
            self.model.fit(X_array)
        else:
            if _is_session_sequence(y):
                raise TypeError(
                    "For single-session input X, y must be a single array."
                )
            y_array = _ensure_2d(_as_float32(y), name="y")
            if len(X_array) != len(y_array):
                raise ValueError(
                    "X and y must contain the same number of samples, "
                    f"got {len(X_array)} and {len(y_array)}."
                )
            self.model.fit(X_array, y_array)

        self._is_multi = False
        self._n_sessions = 1
        self._is_fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        session_id: Optional[int] = None,
    ) -> np.ndarray:
        """Transform samples into a CEBRA embedding."""
        self._require_fitted()
        X_array = _validate_feature_array(X, name="X")

        if self._is_multi:
            session_id = self._validate_session_id(session_id)
            Z = self.model.transform(X_array, session_id=session_id)
        else:
            if session_id is not None:
                warnings.warn(
                    "session_id is ignored in single-session mode.",
                    UserWarning,
                    stacklevel=2,
                )
            Z = self.model.transform(X_array)

        embedding = _validate_embedding_array(Z)
        self._last_Z = embedding
        return embedding

    def transform_all_sessions(
        self,
        X_list: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """Transform all sessions after multi-session fitting."""
        self._require_fitted()

        if not self._is_multi:
            raise ValueError(
                "transform_all_sessions requires a multi-session fitted model."
            )

        if not _is_session_sequence(X_list):
            raise TypeError("X_list must be a list or tuple of session arrays.")

        if len(X_list) != self._n_sessions:
            raise ValueError(
                f"Expected {self._n_sessions} sessions, got {len(X_list)}."
            )

        return [
            self.transform(X_session, session_id=session_id)
            for session_id, X_session in enumerate(X_list)
        ]

    def fit_transform(
        self,
        X: ArrayOrList,
        y: Optional[ArrayOrList] = None,
        session_id: Optional[int] = None,
    ) -> np.ndarray:
        """Fit the estimator and transform one selected session."""
        self.fit(X, y=y)

        if _is_session_sequence(X):
            selected_session = self._validate_session_id(session_id)
            return self.transform(
                X[selected_session],
                session_id=selected_session,
            )

        return self.transform(X)

    def save_npz(
        self,
        output_dir: Union[str, Path],
        Z: np.ndarray,
        extras: Optional[Mapping[str, Any]] = None,
        filename: str = "cebra_embedding.npz",
    ) -> Path:
        """Save an embedding and optional metadata as a compressed NPZ file."""
        output_path = _prepare_output_path(output_dir, filename, suffix=".npz")
        payload = _build_payload(Z, extras)
        np.savez_compressed(output_path, **payload)
        return output_path

    def save_mat(
        self,
        output_dir: Union[str, Path],
        Z: np.ndarray,
        extras: Optional[Mapping[str, Any]] = None,
        filename: str = "cebra_results.mat",
    ) -> Path:
        """Save an embedding and optional metadata as a MATLAB MAT file."""
        output_path = _prepare_output_path(output_dir, filename, suffix=".mat")
        payload = _build_payload(Z, extras)
        savemat(str(output_path), payload)
        return output_path

    @property
    def last_embedding(self) -> Optional[np.ndarray]:
        """Return the most recently computed embedding, if available."""
        return self._last_Z

    @property
    def is_multi_session(self) -> bool:
        """Return whether the fitted estimator uses multiple sessions."""
        return self._is_multi

    @property
    def n_sessions(self) -> int:
        """Return the number of fitted sessions."""
        return self._n_sessions

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("CebraAdapter must be fitted before transform.")

    def _validate_session_id(self, session_id: Optional[int]) -> int:
        if session_id is None:
            raise ValueError("session_id is required in multi-session mode.")
        if not isinstance(session_id, int):
            raise TypeError("session_id must be an integer.")
        if not 0 <= session_id < self._n_sessions:
            raise ValueError(
                f"session_id must be between 0 and {self._n_sessions - 1}, "
                f"got {session_id}."
            )
        return session_id


def _is_session_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _as_float32(x: Any) -> np.ndarray:
    """Convert an array-like value to a finite float32 NumPy array."""
    array = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("Input arrays must contain only finite values.")
    return array


def _validate_feature_array(x: Any, *, name: str) -> np.ndarray:
    """Validate one CEBRA feature array."""
    array = _as_float32(x)
    if array.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be two- or three-dimensional, got shape {array.shape}."
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return array


def _ensure_2d(y: Any, *, name: str = "y") -> np.ndarray:
    """Return labels with shape ``(n_samples, n_features)``."""
    array = _as_float32(y)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be one- or two-dimensional, got shape {array.shape}."
        )
    return array


def _validate_embedding_array(Z: Any) -> np.ndarray:
    embedding = _as_float32(Z)
    if embedding.ndim != 2:
        raise ValueError(
            "CEBRA transform must return a two-dimensional embedding, "
            f"got shape {embedding.shape}."
        )
    return embedding


def _prepare_output_path(
    output_dir: Union[str, Path],
    filename: str,
    *,
    suffix: str,
) -> Path:
    if not filename:
        raise ValueError("filename must not be empty.")

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    output_path = path / filename
    if output_path.suffix.lower() != suffix:
        output_path = output_path.with_suffix(suffix)
    return output_path


def _build_payload(
    Z: np.ndarray,
    extras: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"Z": _validate_embedding_array(Z)}
    if extras:
        for key, value in extras.items():
            if key == "Z":
                raise ValueError("extras must not overwrite the reserved key 'Z'.")
            payload[key] = value
    return payload


def procrustes_align_to(
    Z_ref: np.ndarray,
    Z: np.ndarray,
    y_ref: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    top_k: int = 5,
    subsample: Optional[int] = None,
) -> np.ndarray:
    """Align one embedding to a reference with CEBRA Procrustes alignment."""
    _require_alignment_dependency()
    reference = _validate_embedding_array(Z_ref)
    target = _validate_embedding_array(Z)

    if reference.shape[1] != target.shape[1]:
        raise ValueError(
            "Reference and target embeddings must have the same dimension, "
            f"got {reference.shape[1]} and {target.shape[1]}."
        )

    aligner = OrthogonalProcrustesAlignment(
        top_k=top_k,
        subsample=subsample,
    )

    if y_ref is None and y is None:
        return _as_float32(aligner.fit_transform(reference, target))

    if y_ref is None or y is None:
        raise ValueError("y_ref and y must either both be provided or both be None.")

    reference_labels = _ensure_2d(y_ref, name="y_ref")
    target_labels = _ensure_2d(y, name="y")

    return _as_float32(
        aligner.fit_transform(
            reference,
            target,
            reference_labels,
            target_labels,
        )
    )


def procrustes_align_all(
    embeddings: Mapping[str, np.ndarray],
    labels: Optional[Mapping[str, Optional[np.ndarray]]],
    ref_name: str,
    top_k: int = 5,
    subsample: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Align all named embeddings to one named reference embedding."""
    if ref_name not in embeddings:
        raise ValueError(f"Reference embedding {ref_name!r} was not found.")

    labels = labels or {}
    reference = _validate_embedding_array(embeddings[ref_name])
    reference_labels = labels.get(ref_name)

    aligned: Dict[str, np.ndarray] = {ref_name: reference}
    for name, embedding in embeddings.items():
        if name == ref_name:
            continue

        aligned[name] = procrustes_align_to(
            reference,
            embedding,
            y_ref=reference_labels,
            y=labels.get(name),
            top_k=top_k,
            subsample=subsample,
        )

    return aligned


def _require_alignment_dependency() -> None:
    if OrthogonalProcrustesAlignment is None:
        raise ImportError(
            "CEBRA is required for Procrustes alignment. Install it with "
            "`pip install cebra`."
        ) from _CEBRA_IMPORT_ERROR
