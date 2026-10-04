"""Trial-safe CEBRA-style sampling for cached temporal windows."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset


class CEBRATripletWindowDataset(Dataset):
    """Construct reference/positive/negative windows from center indices.

    The source dataset already contains trial-safe windows. This wrapper keeps
    the same representation but samples positives by a fixed temporal offset
    within the same trial and draws negatives from the same split. Windows are
    therefore cached, while the *pairing* follows the CEBRA sampler semantics.
    """

    def __init__(
        self,
        dataset,
        candidate_indices: Iterable[int],
        *,
        offset: int,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if offset <= 0:
            raise ValueError("offset must be strictly positive")

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
            index for index in candidates
            if 0 <= index < len(dataset) and bool(valid[index])
        }
        if not candidate_set:
            raise ValueError("no valid candidate windows remain")

        # Metadata are center coordinates, so the map is independent of the
        # flattened storage order used by the cached windows.
        lookup = {
            (int(dataset.trial_id[index]), int(dataset.time_id[index])): index
            for index in candidate_set
        }

        anchors: list[int] = []
        positives: list[int] = []
        for index in sorted(candidate_set):
            key = (int(dataset.trial_id[index]), int(dataset.time_id[index]) + offset)
            positive = lookup.get(key)
            if positive is None:
                continue
            anchors.append(index)
            positives.append(positive)

        if not anchors:
            raise ValueError(
                "no reference/positive pairs exist in the supplied split for "
                f"offset={offset}"
            )

        self.dataset = dataset
        self.anchor_indices = torch.tensor(anchors, dtype=torch.long)
        self.positive_indices = torch.tensor(positives, dtype=torch.long)
        self.negative_pool = torch.tensor(sorted(candidate_set), dtype=torch.long)

        # Draw once per epoch-sized dataset. This avoids Python-side random
        # work in __getitem__ while retaining an independent negative for each
        # reference. A later epoch can call resample_negatives() if desired.
        self.resample_negatives()

    def __len__(self) -> int:
        return int(self.anchor_indices.numel())

    def resample_negatives(self, *, generator: torch.Generator | None = None) -> None:
        draws = torch.randint(
            self.negative_pool.numel(),
            (len(self),),
            generator=generator,
        )
        negatives = self.negative_pool[draws]
        # Avoid trivial self-pairs where possible without rejection sampling.
        if self.negative_pool.numel() > 2:
            same_anchor = negatives == self.anchor_indices
            same_positive = negatives == self.positive_indices
            replace = same_anchor | same_positive
            negatives[replace] = self.negative_pool[(draws[replace] + 1) % self.negative_pool.numel()]
        self.negative_indices = negatives

    def resample(self, *, generator: torch.Generator | None = None) -> None:
        """Resample the independent negatives (common sampler interface)."""
        self.resample_negatives(generator=generator)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        anchor = self.anchor_indices[index]
        positive = self.positive_indices[index]
        negative = self.negative_indices[index]
        return {
            "reference_x": self.dataset.X_windows[anchor],
            "positive_x": self.dataset.X_windows[positive],
            "negative_x": self.dataset.X_windows[negative],
        }
