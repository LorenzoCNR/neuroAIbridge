"""Trial-safe CEBRA-Behavior sampling for discrete auxiliary labels."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset


class CEBRASupervisedWindowDataset(Dataset):
    """Construct CEBRA reference/positive/negative windows from labels.

    This is the discrete-label analogue of :class:`CEBRATripletWindowDataset`.
    Reference and negative samples are drawn from the supplied split, while a
    positive is sampled from the same discrete label as the reference. The
    positive and negative assignments are resampled between epochs.
    """

    def __init__(
        self,
        dataset,
        candidate_indices: Iterable[int],
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        labels = getattr(dataset, "labels_windows", None)
        if labels is None:
            raise ValueError("CEBRA supervised sampling requires discrete labels")
        labels = torch.as_tensor(labels).view(-1)
        if labels.shape[0] != len(dataset):
            raise ValueError("dataset labels must have one value per window")
        if labels.dtype.is_floating_point:
            raise ValueError("CEBRA supervised sampling requires integer labels")

        candidates = [int(index) for index in candidate_indices]
        if not candidates:
            raise ValueError("candidate_indices must not be empty")

        if valid_mask is None:
            valid = torch.ones(len(dataset), dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid_mask, dtype=torch.bool).view(-1)
            if valid.shape[0] != len(dataset):
                raise ValueError("valid_mask must have one value per dataset window")

        candidate_set = {
            index
            for index in candidates
            if 0 <= index < len(dataset) and bool(valid[index])
        }
        if not candidate_set:
            raise ValueError("no valid candidate windows remain")

        # CEBRA's discrete sampler requires at least two samples for each
        # label from which a positive can be drawn.
        groups: dict[int, list[int]] = {}
        for index in sorted(candidate_set):
            groups.setdefault(int(labels[index]), []).append(index)
        groups = {label: values for label, values in groups.items() if len(values) >= 2}
        if not groups:
            raise ValueError("no label has at least two candidate windows")

        anchors = [index for values in groups.values() for index in values]
        self.dataset = dataset
        self.anchor_indices = torch.tensor(sorted(anchors), dtype=torch.long)
        self.anchor_labels = labels[self.anchor_indices].long()
        self.label_groups = {
            label: torch.tensor(values, dtype=torch.long)
            for label, values in groups.items()
        }
        self.negative_pool = torch.tensor(sorted(candidate_set), dtype=torch.long)
        self.resample()

    def __len__(self) -> int:
        return int(self.anchor_indices.numel())

    def resample(self, *, generator: torch.Generator | None = None) -> None:
        """Resample same-label positives and independent negatives."""
        positives = torch.empty_like(self.anchor_indices)
        for label, group in self.label_groups.items():
            mask = self.anchor_labels == label
            anchors = self.anchor_indices[mask]
            permutation = torch.randperm(len(group), generator=generator)
            selected = group[permutation]
            # Keep the positive distinct from its anchor while preserving the
            # same-label conditional distribution.
            same = selected == anchors
            if torch.any(same):
                selected[same] = group[(permutation[same] + 1) % len(group)]
            positives[mask] = selected

        draws = torch.randint(
            self.negative_pool.numel(),
            (len(self),),
            generator=generator,
        )
        self.positive_indices = positives
        self.negative_indices = self.negative_pool[draws]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        anchor = self.anchor_indices[index]
        positive = self.positive_indices[index]
        negative = self.negative_indices[index]
        return {
            "reference_x": self.dataset.X_windows[anchor],
            "positive_x": self.dataset.X_windows[positive],
            "negative_x": self.dataset.X_windows[negative],
        }
