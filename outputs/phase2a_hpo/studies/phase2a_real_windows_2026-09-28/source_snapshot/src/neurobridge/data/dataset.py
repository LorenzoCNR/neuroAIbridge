from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch
from torch.utils.data import Dataset


class TemporalWindowDataset(Dataset):
    """
    PyTorch dataset for windowed multivariate time series.

    Each sample is one temporal window plus metadata needed to construct
    distances/similarities inside a batch.
    """

    def __init__(
            self,
            X_windows,
            time_id,
            global_time_id,
            trial_id,
            labels_windows: Optional[object] = None,
            extra_metadata: Optional[Mapping[str, object]] = None):
        self.X_windows = torch.as_tensor(X_windows, dtype=torch.float32)
        self.time_id = torch.as_tensor(time_id, dtype=torch.float32)
        self.global_time_id = torch.as_tensor(global_time_id, dtype=torch.float32)
        self.trial_id = torch.as_tensor(trial_id, dtype=torch.long)

        if labels_windows is None:
            self.labels_windows = None
        else:
            labels_tensor = torch.as_tensor(labels_windows)
            if labels_tensor.dtype.is_floating_point or labels_tensor.ndim > 1:
                self.labels_windows = labels_tensor.float()
            else:
                self.labels_windows = labels_tensor.long()

        self.extra_metadata = {}
        if extra_metadata is not None:
            for key, values in extra_metadata.items():
                if key in {
                    "x",
                    "time_id",
                    "global_time_id",
                    "trial_id",
                    "label",
                }:
                    raise ValueError(
                        f"extra metadata key {key!r} is reserved"
                    )
                self.extra_metadata[key] = torch.as_tensor(values)

        n_windows = self.X_windows.shape[0]
        if self.time_id.shape[0] != n_windows:
            raise ValueError("time_id must have one value per window")
        if self.global_time_id.shape[0] != n_windows:
            raise ValueError("global_time_id must have one value per window")
        if self.trial_id.shape[0] != n_windows:
            raise ValueError("trial_id must have one value per window")
        if self.labels_windows is not None and self.labels_windows.shape[0] != n_windows:
            raise ValueError("labels_windows must have one value per window")
        for key, values in self.extra_metadata.items():
            if values.shape[0] != n_windows:
                raise ValueError(
                    f"extra metadata {key!r} must have one value per window"
                )

    def __len__(self):
        return self.X_windows.shape[0]

    def __getitem__(self, idx):
        sample = {
            "x": self.X_windows[idx],
            "time_id": self.time_id[idx],
            "global_time_id": self.global_time_id[idx],
            "trial_id": self.trial_id[idx],
        }

        if self.labels_windows is not None:
            sample["label"] = self.labels_windows[idx]

        for key, values in self.extra_metadata.items():
            sample[key] = values[idx]

        return sample
