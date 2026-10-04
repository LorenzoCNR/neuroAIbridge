from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from neurobridge.experiments.frozen_sensitivity import verify_frozen_checkpoint


def test_checkpoint_discrepancy_is_strict_by_default_and_recorded_explicitly():
    raw = np.arange(18, dtype=float).reshape(6, 3) + 1
    values = {"embedding_raw": raw, "embedding_unit": raw / np.linalg.norm(raw, axis=1, keepdims=True)}
    for key in ("trial_id", "time_id", "global_time_id", "labels", "progress", "lag_valid"):
        values[key] = np.arange(6)
    windows = dict(values, X_windows=raw[:, None, :])
    model = SimpleNamespace(transform=lambda x: x + 0.1)
    args = ("embedding.npz", values, windows, SimpleNamespace(latent_dim=3), {"model": "pca"}, "pca.joblib")
    with patch("neurobridge.experiments.frozen_sensitivity.joblib.load", return_value=model):
        with pytest.raises(ValueError, match="does not reproduce"):
            verify_frozen_checkpoint(*args)
        diagnostics = {}
        assert verify_frozen_checkpoint(*args, diagnostics=diagnostics) == 6
        assert diagnostics["checkpoint_reproduced_within_tolerance"] is False
        assert diagnostics["max_absolute_difference"] == pytest.approx(0.1)
        # Recording numerical differences must not bypass metadata validation.
        windows["labels"] = np.arange(6) + 1
        with pytest.raises(ValueError, match="metadata mismatch"):
            verify_frozen_checkpoint(*args, diagnostics={})
