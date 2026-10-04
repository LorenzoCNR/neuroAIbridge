from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_cosine_logits(z, temperature=0.1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = F.normalize(z, dim=-1)
    return z @ z.T / temperature


def supervised_infonce_loss(z, labels, temperature=0.1):
    """
    Supervised contrastive / InfoNCE-style loss.

    Positives are samples with the same label inside the batch.
    """
    if z.ndim != 2:
        raise ValueError("z must have shape (batch, embedding_dim)")

    labels = labels.view(-1)
    if labels.shape[0] != z.shape[0]:
        raise ValueError("labels must have one value per embedding")

    logits = pairwise_cosine_logits(z, temperature=temperature)
    batch_size = z.shape[0]
    device = z.device

    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye

    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    valid = positive_mask.sum(dim=1) > 0
    if not torch.any(valid):
        return z.sum() * 0.0

    selected_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    loss_per_sample = -selected_log_prob.sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return loss_per_sample[valid].mean()


def masked_infonce_loss(z, positive_mask, temperature=0.1):
    """
    InfoNCE-style loss from an explicit positive-pair mask.

    This covers supervised labels, temporal offsets, or any sampler that marks
    anchor-positive pairs without requiring a scalar class label.
    """
    if z.ndim != 2:
        raise ValueError("z must have shape (batch, embedding_dim)")
    if positive_mask.shape != (z.shape[0], z.shape[0]):
        raise ValueError("positive_mask must have shape (batch, batch)")

    batch_size = z.shape[0]
    device = z.device
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = positive_mask.to(device=device, dtype=torch.bool) & ~eye

    logits = pairwise_cosine_logits(z, temperature=temperature)
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    valid = positive_mask.sum(dim=1) > 0
    if not torch.any(valid):
        return z.sum() * 0.0

    selected_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    loss_per_sample = -selected_log_prob.sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return loss_per_sample[valid].mean()


def time_offset_infonce_loss(z, trial_id, time_id, offset=10, temperature=0.1):
    """
    Unsupervised temporal InfoNCE loss.

    Positives are windows from the same trial whose time identifiers differ by
    `offset`. This is the compact NeuroBridge analogue of a CEBRA-time setup:
    no behavioral label is required.
    """
    trial_id = trial_id.view(-1)
    time_id = time_id.view(-1)
    if trial_id.shape[0] != z.shape[0] or time_id.shape[0] != z.shape[0]:
        raise ValueError("trial_id and time_id must have one value per embedding")
    if offset <= 0:
        raise ValueError("offset must be positive")

    same_trial = trial_id[:, None].eq(trial_id[None, :])
    temporal_offset = torch.abs(time_id[:, None] - time_id[None, :]).eq(float(offset))
    positive_mask = same_trial & temporal_offset
    return masked_infonce_loss(z, positive_mask, temperature=temperature)


def cebra_infonce_loss(reference, positive, negative, temperature=1.0):
    """InfoNCE on an explicit CEBRA-style reference/positive/negative batch.

    ``reference`` and ``positive`` contain one embedding per sampled pair. The
    rows of ``negative`` are the independent negative samples for the batch;
    every reference is contrasted with all of them. This is the objective used
    by the CEBRA solvers after their sampler has constructed the three inputs:
    the positive similarity is the alignment term and only the negative
    similarities enter the ``logsumexp`` uniformity term.
    Embeddings are L2-normalized here so the function remains correct when it is
    used with an encoder whose final normalization is disabled for diagnostics.
    """
    for name, value in (("reference", reference), ("positive", positive),
                        ("negative", negative)):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must have shape (batch, embedding_dim)")
    if reference.shape != positive.shape or negative.shape != reference.shape:
        raise ValueError("reference, positive and negative must have identical shapes")
    if reference.shape[0] < 1:
        raise ValueError("the batch must contain at least one sample")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    reference = F.normalize(reference, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)

    positive_logits = (reference * positive).sum(dim=-1) / temperature
    negative_logits = (reference @ negative.T) / temperature
    return -positive_logits.mean() + torch.logsumexp(negative_logits, dim=1).mean()


def soft_contrastive_loss(z, similarity, temperature=0.1, eps=1e-8):
    """
    Soft structured contrastive loss.

    similarity[i, j] is the desired soft relationship between samples i and j.
    The diagonal is ignored. Rows are normalized into soft target distributions.
    """
    if z.ndim != 2:
        raise ValueError("z must have shape (batch, embedding_dim)")
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix")
    if similarity.shape[0] != z.shape[0]:
        raise ValueError("similarity batch size must match z")

    similarity = similarity.to(device=z.device, dtype=z.dtype)
    batch_size = z.shape[0]
    eye = torch.eye(batch_size, dtype=torch.bool, device=z.device)

    targets = similarity.masked_fill(eye, 0.0).clamp_min(0.0)
    row_sum = targets.sum(dim=1, keepdim=True)
    valid = row_sum.squeeze(1) > eps

    if not torch.any(valid):
        return z.sum() * 0.0

    targets = targets / row_sum.clamp_min(eps)

    logits = pairwise_cosine_logits(z, temperature=temperature)
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    weighted_log_prob = torch.where(targets > 0, targets * log_prob, torch.zeros_like(log_prob))
    loss_per_sample = -weighted_log_prob.sum(dim=1)
    return loss_per_sample[valid].mean()
