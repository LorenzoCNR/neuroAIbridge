# NeuroBridge Documentation

This directory contains the project's scientific and technical documentation.
The GitHub repository is `LorenzoCNR/neuroAIbridge`; NeuroBridge is the project
name used in prose and figures.

## Start with the current study

1. [Experiments, results, and limitations](06_EXPERIMENTS_AND_LIMITATIONS.md)
   maps the frozen evidence and states what each result does—and does not—show.
2. [Python pipeline map](07_PYTHON_PIPELINE_MAP.md) traces the staged code,
   validation rules, caches, and output dependencies.
3. Exact scientific settings and ancestry are recorded in the relevant frozen
   spec, CSV/JSON manifest, and provenance beside each output branch. Those
   artifacts, rather than prose, are the authority for exact values.

## Scientific background

- [Generative model](01_GENERATIVE_MODEL.md): task latent, population mapping,
  firing rates, and spike-count generation.
- [Data and temporal windows](02_DATA_AND_WINDOWS.md): trial structure,
  metadata, padding, splits, and the frozen Real window choice.
- [Learning objectives](03_LEARNING_OBJECTIVES.md): definitions and
  supervision supplied by each objective.
- [Encoders](04_ENCODERS.md): PCA and temporal neural encoders.
- [Evaluation and multiple subjects](05_EVALUATION_AND_MULTISUBJECT.md):
  geometry, decoding, temporal correspondence, and causal-interpretation
  limits.
- [Notebook guide](../notebooks/EXPERIMENTS.md): scope of the executable
  simulator tutorials and historical reference runs.

## What is implemented

The package contains controlled circular/linear simulators, temporal-window
datasets, PCA/CNN1D/Transformer encoders, four neural objectives, and staged
evaluation. The final study keeps whole-process latent recovery, task
accessibility, temporal structure, held-out generalization, robustness,
diagnostics, and compute as separate dimensions.

The current frozen Real study is indexed under
`outputs/final_thesis_v1/`. It uses one Area-2 reaching recording, a
validation-selected 201-bin window, TOTAL65 and the fixed operational
somatotopic A/B partition, plus frozen neural models and downstream evaluation.
The recording has no observed biological latent ground truth or second
independent animal/session; see the current-results guide for the precise
design and caveats.

The older four notebooks and the older staged synthetic workflow are useful
reproducible examples, but they are not the source of truth for the final HPO
selection or Real evaluation protocol.

## Documentation layers and history

| Location | Role |
|---|---|
| `docs/*.md` | Active scientific explanation and current evidence map |
| `docs/source/*.rst` | Source for the optional searchable Sphinx site |
| `docs/archive/` | Historical drafts retained for traceability |
| `NEUROBRIDGE_CURRENT_STATE_AUDIT.md` | Read-only snapshot dated 2026-09-24; not current status |

`docs/06_EXPERIMENTS_AND_LIMITATIONS.md` and
`docs/07_PYTHON_PIPELINE_MAP.md` are the current narrative guides. The
2026-09-24 audit retains historically correct statuses for that date; its
`MISSING` entries must not be read as the status of today's outputs.

## Build the searchable documentation site

The site is optional; all canonical explanations are available as Markdown.

```powershell
python -m pip install -e ".[docs]"
docs\make.bat html
```

The local build is `docs/_build/html/index.html`.
