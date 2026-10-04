# -*- coding: utf-8 -*-
"""
Temporal neural encoders used by NeuroBridge.

All encoders in this module follow the same public interface:

Input
-----
x : torch.Tensor
    Shape: (batch_size, window_size, n_features)

Output
------
z : torch.Tensor
    Shape: (batch_size, embedding_dim)

The module currently provides four alternatives:

- TemporalCNNEncoder:
    1D temporal convolutions followed by global average pooling.

- CEBRAOffset10Encoder:
    CEBRA-compatible ten-bin valid-convolution encoder with skip connections.

- TemporalMLPEncoder:
    Simple baseline that flattens the full temporal window.

- TemporalLSTMEncoder:
    Recurrent baseline based on the final LSTM hidden state.

- TemporalTransformerEncoder:
    Transformer baseline with sinusoidal positional encoding and
    temporal mean pooling.

The optional L2 normalization makes the embeddings directly compatible
with cosine-similarity-based contrastive objectives.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------
# Shared validation utilities
# ---------------------------------------------------------------------

def _validate_positive_int(name: str, value: int) -> None:
    """
    Validate an integer hyperparameter that must be strictly positive.

    Parameters
    ----------
    name:
        Name used in the error message.
    value:
        Value to validate.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_dropout(dropout: float) -> None:
    """
    Validate a dropout probability.

    PyTorch expects dropout in the half-open interval [0, 1).
    """
    if not isinstance(dropout, (int, float)):
        raise TypeError(
            f"dropout must be a real number, got {type(dropout).__name__}."
        )

    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(
            f"dropout must satisfy 0 <= dropout < 1, got {dropout}."
        )


def _validate_temporal_input(
    x: torch.Tensor,
    *,
    n_features: int,
    window_size: int | None = None,
) -> None:
    """
    Validate the common input format used by all temporal encoders.

    Expected shape:
        (batch_size, window_size, n_features)
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"x must be a torch.Tensor, got {type(x).__name__}."
        )

    if x.ndim != 3:
        raise ValueError(
            "x must have shape "
            "(batch_size, window_size, n_features), "
            f"got {tuple(x.shape)}."
        )

    if x.size(-1) != n_features:
        raise ValueError(
            f"Expected {n_features} input features, got {x.size(-1)}."
        )

    if window_size is not None and x.size(1) != window_size:
        raise ValueError(
            f"Expected window size {window_size}, got {x.size(1)}."
        )


def _normalize_embedding(
    z: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Optionally L2-normalize each embedding vector.

    With normalization enabled, every row has approximately unit norm.
    This is useful because the dot product then corresponds to cosine
    similarity, which is used by the contrastive losses in NeuroBridge.
    """
    if normalize:
        return F.normalize(z, p=2, dim=-1)

    return z


# ---------------------------------------------------------------------
# Positional encoding used by the Transformer encoder
# ---------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Add deterministic sinusoidal position information to a sequence.

    A Transformer self-attention layer does not know temporal order by
    itself. This module adds a distinct position-dependent vector to
    every time step before the sequence enters the Transformer.

    Parameters
    ----------
    model_dim:
        Feature dimension used internally by the Transformer.
    max_length:
        Maximum supported temporal window length.

    Input shape
    -----------
    (batch_size, window_size, model_dim)

    Output shape
    ------------
    (batch_size, window_size, model_dim)
    """

    def __init__(
        self,
        model_dim: int,
        max_length: int = 4096,
    ) -> None:
        super().__init__()

        _validate_positive_int("model_dim", model_dim)
        _validate_positive_int("max_length", max_length)

        # Position indices:
        # shape -> (max_length, 1)
        positions = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Frequencies used by the standard Transformer positional encoding.
        # shape -> (ceil(model_dim / 2),)
        even_dimensions = torch.arange(
            0,
            model_dim,
            2,
            dtype=torch.float32,
        )
        frequencies = torch.exp(
            -math.log(10000.0) * even_dimensions / model_dim
        )

        # Complete positional-encoding table:
        # shape -> (max_length, model_dim)
        encoding = torch.zeros(
            max_length,
            model_dim,
            dtype=torch.float32,
        )

        encoding[:, 0::2] = torch.sin(positions * frequencies)

        # For odd model_dim values, the cosine branch has one column fewer.
        cosine_width = encoding[:, 1::2].shape[1]
        if cosine_width > 0:
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[:cosine_width]
            )

        # A buffer:
        # - moves with the model between CPU and GPU;
        # - is not optimized as a trainable parameter;
        # - is omitted from checkpoints because it can be reconstructed.
        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional information to the projected sequence."""
        if x.ndim != 3:
            raise ValueError(
                "Positional encoding expects shape "
                "(batch_size, window_size, model_dim), "
                f"got {tuple(x.shape)}."
            )

        sequence_length = x.size(1)
        maximum_length = self.encoding.size(1)

        if sequence_length > maximum_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"max_length={maximum_length}."
            )

        position_values = self.encoding[:, :sequence_length].to(
            device=x.device,
            dtype=x.dtype,
        )

        return x + position_values


# ---------------------------------------------------------------------
# CNN encoder
# ---------------------------------------------------------------------

class TemporalCNNEncoder(nn.Module):
    """
    CEBRA-inspired 1D temporal convolutional encoder.

    The model applies convolutions across time. Neurons/features are
    treated as input channels.

    External input:
        (batch_size, window_size, n_features)

    Internal Conv1d input:
        (batch_size, n_features, window_size)

    Output:
        (batch_size, embedding_dim)

    Notes
    -----
    The current symmetric-padding rule requires an odd kernel size.

    Examples:
        kernel_size=3 -> accepted
        kernel_size=5 -> accepted
        kernel_size=4 -> rejected

    With an odd kernel and padding=kernel_size // 2, every convolution
    preserves the temporal length exactly.
    """

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        n_layers: int = 3,
        dropout: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        _validate_positive_int("n_features", n_features)
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("hidden_dim", hidden_dim)
        _validate_positive_int("kernel_size", kernel_size)
        _validate_positive_int("n_layers", n_layers)
        _validate_dropout(dropout)

        # IMPORTANT:
        # We reject EVEN kernels.
        #
        # For stride=1 and dilation=1:
        # output_length = input_length + 2*padding - kernel_size + 1
        #
        # With kernel_size=3 and padding=1:
        # output_length = input_length
        #
        # With kernel_size=4 and padding=2:
        # output_length = input_length + 1
        #
        # Therefore an odd kernel gives exact symmetric "same length"
        # padding with padding=kernel_size // 2.
        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd when using symmetric padding. "
                f"Got kernel_size={kernel_size}."
            )

        self.n_features = n_features
        self.normalize = normalize

        padding = kernel_size // 2

        self.input_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GELU(),
        )

        # Residual temporal blocks preserve local information while increasing
        # the receptive field. All convolutions keep the window length fixed.
        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_layers - 1)
            ]
        )

        # Global average pooling converts every temporal window into one
        # hidden vector. This means the contrastive loss receives one
        # embedding per window, not one embedding per time step.
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        # Final projection into the requested representation dimension.
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of temporal neural windows."""
        _validate_temporal_input(
            x,
            n_features=self.n_features,
        )

        # Conv1d expects channels before time:
        # (B, T, N) -> (B, N, T)
        x_channels_first = x.transpose(1, 2)

        hidden_sequence = self.input_layer(x_channels_first)
        for block in self.residual_blocks:
            hidden_sequence = hidden_sequence + block(hidden_sequence)

        # (B, hidden_dim, T) -> (B, hidden_dim, 1)
        pooled = self.pool(hidden_sequence)

        # Remove the singleton temporal dimension:
        # (B, hidden_dim, 1) -> (B, hidden_dim)
        pooled = pooled.squeeze(-1)

        embedding = self.projection(pooled)

        return _normalize_embedding(
            embedding,
            normalize=self.normalize,
        )


class CEBRAOffset10Encoder(nn.Module):
    """Minimal local implementation of CEBRA's ``offset10-model``.

    The encoder uses valid temporal convolutions and three residual blocks.  A
    ten-bin context produces one temporal output, hence one embedding per
    sampled center.  When the NeuroBridge cache contains wider windows (21 bins
    in the current protocol), the centered ten-bin context is selected before
    applying the CEBRA architecture.
    """

    context_size = 10

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 3,
        hidden_dim: int = 32,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        _validate_positive_int("n_features", n_features)
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("hidden_dim", hidden_dim)
        self.n_features = n_features
        self.normalize = normalize

        self.input_layer = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=2),
            nn.GELU(),
        )
        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.output_layer = nn.Conv1d(hidden_dim, embedding_dim, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_temporal_input(x, n_features=self.n_features)
        if x.size(1) < self.context_size:
            raise ValueError(
                f"CEBRAOffset10Encoder requires at least {self.context_size} "
                f"time bins, got {x.size(1)}."
            )

        start = (x.size(1) - self.context_size) // 2
        context = x[:, start:start + self.context_size].transpose(1, 2)
        hidden = self.input_layer(context)
        for block in self.residual_blocks:
            hidden = hidden[..., 1:-1] + block(hidden)

        embedding = self.output_layer(hidden).squeeze(-1)
        if embedding.ndim != 2:
            raise RuntimeError(
                "CEBRAOffset10Encoder expected one temporal output; "
                f"got shape {tuple(embedding.shape)}."
            )
        return _normalize_embedding(embedding, normalize=self.normalize)


# ---------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------

class TemporalMLPEncoder(nn.Module):
    """
    Simple non-convolutional baseline.

    The complete temporal window is flattened and passed through a
    two-layer MLP. This model does not encode an explicit temporal
    inductive bias and is useful as a comparison against the CNN.
    """

    def __init__(
        self,
        window_size: int,
        n_features: int,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        _validate_positive_int("window_size", window_size)
        _validate_positive_int("n_features", n_features)
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("hidden_dim", hidden_dim)

        self.window_size = window_size
        self.n_features = n_features
        self.normalize = normalize

        flattened_dim = window_size * n_features

        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a fixed-length temporal window with an MLP."""
        _validate_temporal_input(
            x,
            n_features=self.n_features,
            window_size=self.window_size,
        )

        embedding = self.net(x)

        return _normalize_embedding(
            embedding,
            normalize=self.normalize,
        )


# ---------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------

class TemporalLSTMEncoder(nn.Module):
    """
    Recurrent baseline for temporal neural windows.

    The final hidden state of the top LSTM layer summarizes the full
    temporal window and is projected into the embedding space.
    """

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        _validate_positive_int("n_features", n_features)
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("hidden_dim", hidden_dim)
        _validate_positive_int("num_layers", num_layers)
        _validate_dropout(dropout)

        self.n_features = n_features
        self.normalize = normalize

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch applies recurrent-layer dropout only when there is
            # more than one stacked LSTM layer.
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch using the final hidden LSTM state."""
        _validate_temporal_input(
            x,
            n_features=self.n_features,
        )

        # h_n shape:
        # (num_layers, batch_size, hidden_dim)
        _, (h_n, _) = self.lstm(x)

        # Final hidden state of the top recurrent layer:
        final_hidden = h_n[-1]

        embedding = self.projection(final_hidden)

        return _normalize_embedding(
            embedding,
            normalize=self.normalize,
        )


# ---------------------------------------------------------------------
# Transformer baseline
# ---------------------------------------------------------------------

class TemporalTransformerEncoder(nn.Module):
    """
    Transformer baseline for temporal neural windows.

    Processing pipeline:
        neural features
        -> linear projection
        -> sinusoidal positional encoding
        -> Transformer encoder
        -> temporal mean pooling
        -> embedding projection

    The positional encoding is required because self-attention alone
    does not explicitly know the temporal order of the input samples.
    """

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 16,
        model_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_length: int = 4096,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        _validate_positive_int("n_features", n_features)
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("model_dim", model_dim)
        _validate_positive_int("n_heads", n_heads)
        _validate_positive_int("n_layers", n_layers)
        _validate_positive_int("max_length", max_length)
        _validate_dropout(dropout)

        if model_dim % n_heads != 0:
            raise ValueError(
                "model_dim must be divisible by n_heads, "
                f"got model_dim={model_dim} and n_heads={n_heads}."
            )

        self.n_features = n_features
        self.normalize = normalize

        # Convert each neural feature vector into the Transformer dimension.
        self.input_projection = nn.Linear(
            in_features=n_features,
            out_features=model_dim,
        )

        # Add explicit temporal-order information.
        self.position_encoding = SinusoidalPositionalEncoding(
            model_dim=model_dim,
            max_length=max_length,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        self.projection = nn.Linear(model_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of temporal windows with self-attention."""
        _validate_temporal_input(
            x,
            n_features=self.n_features,
        )

        projected = self.input_projection(x)
        ordered = self.position_encoding(projected)
        contextualized = self.encoder(ordered)

        # One embedding per temporal window.
        pooled = contextualized.mean(dim=1)
        embedding = self.projection(pooled)

        return _normalize_embedding(
            embedding,
            normalize=self.normalize,
        )
