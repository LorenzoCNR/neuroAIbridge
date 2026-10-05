# NeuroBridge Documentation

This directory is the canonical scientific and technical documentation for
NeuroBridge. It is designed to be readable directly on GitHub or from a local
clone: the project website is optional.

## Start Here

Read the documents in this order:

1. [Generative model](01_GENERATIVE_MODEL.md) explains the task latent, neural
   population map, firing rates, spike-count emission, and linear-track place
   fields.
2. [Data and temporal windows](02_DATA_AND_WINDOWS.md) explains the tensors,
   metadata, padding, trial boundaries, train/test split, and what one model
   observation represents.
3. [Learning objectives](03_LEARNING_OBJECTIVES.md) derives the soft structured
   contrastive loss, distinguishes its two temperatures, and states what is
   supervised, self-supervised, and computationally expensive.
4. [Encoders](04_ENCODERS.md) explains PCA, CNN1D, MLP, LSTM, and Transformer
   processing at both the algorithmic and tensor level.
5. [Evaluation and multiple subjects](05_EVALUATION_AND_MULTISUBJECT.md)
   explains RSA, Procrustes alignment, lag recovery, and the limits of causal
   interpretation.
6. [Experiments and evidence](06_EXPERIMENTS_AND_LIMITATIONS.md) documents the
   frozen synthetic reference, the exploratory real-monkey run, generated
   artifacts, current claims, and missing experiments.
7. [Python pipeline map](07_PYTHON_PIPELINE_MAP.md) maps executable entry
   points to implementation modules, stage dependencies, validation rules,
   caches, and result artifacts.

For a cell-by-cell reproduction guide, see
[the notebook guide](../notebooks/EXPERIMENTS.md).

## Documentation Layers

NeuroBridge has three documentation layers:

| Location | Purpose | Intended reader |
|---|---|---|
| `docs/*.md` | Canonical scientific and technical explanation | GitHub visitors, collaborators, reviewers |
| `docs/source/*.rst` | Sphinx source used to build the searchable documentation website | Documentation build system |
| `docs/archive/` | Historical drafts retained for traceability | Maintainers only |

Files ending in `.rst` use **reStructuredText**, the markup format consumed by
Sphinx. They play a role similar to Markdown files, but are primarily build
sources. A reader should not need to inspect them to understand the project:
the complete explanation is available in the Markdown documents listed above.

## What Is Implemented

The package currently includes:

- circular and linear controlled motor-task latents;
- heterogeneous neural population mappings;
- Poisson spike-count emission plus optional overdispersion, bursting, and
  refractory mechanisms;
- centered temporal windows that never cross trial boundaries;
- PCA, CNN1D, MLP, LSTM, and Transformer encoders;
- soft structured contrastive, supervised InfoNCE, and temporal-offset
  objectives;
- held-out RSA and Procrustes recovery metrics on the synthetic branch;
- subject-specific neural mappings, imposed temporal lag, and lag-aware
  alignment utilities.

The staged Synthetic v1 reference compares PCA, CNN1D, and Transformer across
four objectives and keeps latent recovery, decoding, lag, robustness, and
efficiency distinct. The local real-data branch
`src/neurobridge/experiments/real_monkey.py` runs the same model/objective grid
on one Area-2 reaching session and writes trial-safe held-out and descriptive
full-sample artifacts under
`outputs/real_monkey_area2_active_staged_2026-09-22/`. It has no biological
latent ground truth, second subject, or cross-session generalization yet.

The older notebooks remain reproducible examples of the original simulator
workflow. They are not the sole definition of the current staged benchmark;
the current protocol and output locations are recorded in
`docs/06_EXPERIMENTS_AND_LIMITATIONS.md`.

## Which documents are authoritative?

The numbered files in this directory and the updated Obsidian notes with the
suffix `_UPDATED_2026-09-17` are the active documentation. The following are
historical by design and should not be overwritten: the Synthetic v1 report,
its parameter sheet, the shared-latent chapter draft, and everything under
`docs/archive/`. The older copies without the `UPDATED` suffix in
`legacy_ai_for_all/Miei doc/miei_doc_neurobridge/` were removed from the active
working set because their content is superseded by the updated notes; they are
not a second source of truth.

## Build The Searchable Site

The documentation website is generated from `docs/source/`:

```powershell
python -m pip install -e ".[docs]"
docs\make.bat html
```

The local result is `docs/_build/html/index.html`. Building the website is not
required to read any canonical document in this directory.
