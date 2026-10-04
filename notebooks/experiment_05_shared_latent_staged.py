"""Cell-by-cell runner for the compartmental shared-latent benchmark.

Open this file as a Python notebook in VS Code/Jupyter.  Each stage can be
rerun independently because it consumes the artifacts written by the previous
stage rather than retraining or regenerating data implicitly.
"""

# %% [markdown]
# # Shared latent center-out benchmark
#
# The benchmark has two reporting branches:
#
# * ``full_sample``: descriptive, all trials used for fitting;
# * ``held_out``: trial split, train/validation/test and final test metrics.
#
# The place-field experiment is separate and is not called here.

# %% Imports and protocol
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists() and (PROJECT_ROOT.parent / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurobridge.experiments.staged_shared_latent import (
    SharedLatentStageConfig,
    stage_evaluate,
    stage_fit,
    stage_generate,
    stage_plot_lag_profile,
    stage_plot,
    stage_plot_raw,
    stage_plot_training_history,
    stage_profile,
    stage_quality_report,
    stage_transform,
    stage_windows,
)

CONFIG = SharedLatentStageConfig(
    name="shared_latent_center_out",
    latent_dim=3,
    n_trials=200,
    n_conditions=8,
    trial_length=200,
    dt=0.02,
    n_neurons_A=160,
    n_neurons_B=120,
    lag_bins=10,
    cebra_time_offset=10,
    window_size=21,
    rate_scale=1.0,
    overdispersion=4.0,
    refractory_mean_bins=1,
    refractory_std_bins=0.25,
    batch_size=1024,
    max_iterations=2000,
    early_stopping_patience=0,
    seed=42,
    run_label="output_2026-08-27_comparison_2000",
)

# %% Stage 1: generate shared latent and spikes
DATA_ARTIFACT = stage_generate(PROJECT_ROOT, CONFIG)
print(DATA_ARTIFACT)

# %% Stage 2: build trial-safe windows and one fixed split
WINDOW_ARTIFACT = stage_windows(PROJECT_ROOT, CONFIG)
print(WINDOW_ARTIFACT)

# %% Stage 3: fit one selected model/loss/branch
# Change only these selectors to rerun a different branch without regenerating
# data or windows.
SUBJECT = "A"
MODEL = "cnn1d"        # "cnn1d", "transformer", or "pca"
LOSS = "cebra_time"      # "soft", "infonce", "cebra_time", or "cebra_behavior" (ignored by PCA)
BRANCH = "held_out"     # "full_sample" or "held_out"

MODEL_ARTIFACT = stage_fit(
    PROJECT_ROOT,
    CONFIG,
    subject=SUBJECT,
    model_name=MODEL,
    loss_name=LOSS,
    branch=BRANCH,
)
print(MODEL_ARTIFACT)

# %% Stage 4: transform all windows with the frozen model
EMBEDDING_ARTIFACT = stage_transform(
    PROJECT_ROOT,
    CONFIG,
    subject=SUBJECT,
    model_name=MODEL,
    loss_name=LOSS,
    branch=BRANCH,
)
print(EMBEDDING_ARTIFACT)

# %% Stage 5: evaluate both subjects and the lag
METRICS_ARTIFACT = stage_evaluate(
    PROJECT_ROOT,
    CONFIG,
    model_name=MODEL,
    loss_name=LOSS,
    branch=BRANCH,
)
print(METRICS_ARTIFACT)

# %% Stage 6: plot saved embeddings and metrics
FIGURE_ARTIFACT = stage_plot(
    PROJECT_ROOT,
    CONFIG,
    model_name=MODEL,
    loss_name=LOSS,
    branch=BRANCH,
)
print(FIGURE_ARTIFACT)

# %% Stage 6b: plot training and lag diagnostics from saved artifacts
if MODEL != "pca":
    LOSS_FIGURE_ARTIFACT = stage_plot_training_history(
        PROJECT_ROOT,
        CONFIG,
        model_name=MODEL,
        loss_name=LOSS,
        branch=BRANCH,
    )
    LAG_FIGURE_ARTIFACT = stage_plot_lag_profile(
        PROJECT_ROOT,
        CONFIG,
        model_name=MODEL,
        loss_name=LOSS,
        branch=BRANCH,
    )
    print(LOSS_FIGURE_ARTIFACT)
    print(LAG_FIGURE_ARTIFACT)

# %% Stage 6c: plot the saved embeddings without unit-sphere normalization
RAW_FIGURE_ARTIFACT = stage_plot_raw(
    PROJECT_ROOT,
    CONFIG,
    model_name=MODEL,
    loss_name=LOSS,
    branch=BRANCH,
)
print(RAW_FIGURE_ARTIFACT)

# %% Stage 7: compact PyTorch computational report (no full refit)
PROFILE_ARTIFACT = stage_profile(PROJECT_ROOT, CONFIG)
print(PROFILE_ARTIFACT)

# %% Stage 8: evaluation-only quality report from frozen artifacts
# This keeps latent recovery, task decoding, temporal structure, held-out
# generalization, robustness, and efficiency as separate metric categories.
QUALITY_ARTIFACT = stage_quality_report(PROJECT_ROOT, CONFIG)
print(QUALITY_ARTIFACT)
