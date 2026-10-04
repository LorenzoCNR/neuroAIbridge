# Experiments, Results, And Current Evidence

This document records the frozen shared-latent benchmark and the first
real-data benchmark. The detailed, step-by-step synthetic account is the
historical Obsidian report:
`legacy_ai_for_all/Miei doc/Obsidian Vault/05_progetti/Neuro_Bridge/NeuroBridge - Rapporto completo esperimento shared latent(2).md`.

## Reference run

The reproducible output directory is
`outputs/output_2026-08-27_comparison_2000/`.

| Item | Current value |
|---|---:|
| subjects | A: 160 neurons; B: 120 neurons |
| trials per condition | 25 |
| conditions | 8 circular directions |
| bins per trial | 200 |
| bin width | 20 ms |
| imposed subject-B lag | 10 bins |
| trial split | 70% train / 20% test / 10% validation |
| window | 21 bins, stride 1 |
| embedding | 3 dimensions |
| neural optimizer budget | 2000 AdamW steps per combination |
| seed | 42 |

The frozen artifact records 140 train / 40 test / 20 validation trials. The
current source dataclass defaults encode 17/5/3 trials per direction (136/40/24);
this provenance mismatch is intentionally not repaired here because changing
the split would invalidate the existing reference artifacts. Future synthetic
runs must choose and record one split explicitly.

The generator produces the complete shared latent process `Z_shared` and
subject-specific neural observations. `Z_shared` is not supplied as a
regression target during neural fitting; it is used for post-hoc recovery
metrics. Internal simulator components are retained only as secondary
diagnostics.

## Compared representations and objectives

The benchmark contains PCA, CNN1D, and Transformer encoders. Neural objectives
are:

- `soft`: pairwise structured contrastive loss using only normalized temporal
  distance and circular condition distance, with weights 0.5/0.5;
- `infonce`: supervised same-condition InfoNCE;
- `cebra_time`: sampled triplets with a same-trial positive at offset 10;
- `cebra_behavior`: sampled triplets with a same-condition positive and no
  explicit temporal offset.

PCA is fit on centered flattened training windows. CNN1D and Transformer use
the same `(batch, 21, neurons)` window interface and produce 3-D embeddings.

## Leakage control and stages

The split is made at trial level before windows are constructed. All overlapping
windows from one trial therefore remain in one partition. The staged pipeline
is:

1. `stage_generate`: latent, subject maps, spikes, and diagnostics;
2. `stage_windows`: windows and metadata;
3. `stage_fit`: one independent fit per subject/model/loss;
4. `stage_transform`: cached embeddings (`embedding_raw`, `embedding_unit`);
5. `stage_evaluate`: Procrustes, RSA, and lag profiles;
6. `stage_plot` / `stage_plot_raw`: normalized and unnormalized figures;
7. `stage_profile`: PyTorch profiler microbenchmarks.

Each stage writes a separate cache directory and can be rerun without
recomputing previous stages. The current run has 36 combinations, 36 metric
CSVs, profiler output in `stage07_profiles/`, and a separate quality report in
`stage08_quality_report/`.

## Current evidence

Held-out recovery of the complete `Z_shared` is strongest for PCA overall
(A/B Procrustes $R^2$ 0.737/0.682). The best held-out lag recovery is obtained by
the explicit time-offset objective: `cebra_time` estimates +11 bins (CNN1D)
and +10 bins (Transformer) for the imposed +10-bin lag. The soft objective is
cheaper and can be competitive in RSA, but it does not robustly recover the
lag. The reported recovery values refer to `Z_shared` as a whole.

The full-sample branch is descriptive, matching common neuroscience practice;
the held-out branch is the generalization check required for an AI evaluation.
All normalized plots use `embedding_unit`. The raw plots use `embedding_raw`
and do not impose a spherical constraint.

## Limitations

- single generator family and seed;
- synthetic shared latent and imposed lag, so the result is a controlled test,
  not evidence for a biological common manifold;
- neural objectives inject different amounts of supervision;
- stride-one windows are highly correlated, although trial-level splitting
  prevents train/test window leakage;
- a 3-D bottleneck may underrepresent intrinsic/embedding dimensionality;
- profiler timings are short microbenchmarks and should not replace the
  end-to-end timings stored in `compute.json`.

## Exploratory real-monkey benchmark (2026-09-22)

The real branch is a separate, cacheable pipeline in
`src/neurobridge/experiments/real_monkey.py`. It uses the local active Area-2
reaching recording, not the simulator, and therefore has no known latent `Z`
for recovery scoring.

| Item | Current value |
|---|---:|
| trials | 193 |
| bins per trial | 600 |
| neural channels | 65 |
| direction classes | 8 |
| trial split | 134 train / 20 validation / 39 test |
| window | 21 bins, stride 1, center padding |
| models | PCA, CNN1D, Transformer |
| neural objectives | soft, supervised InfoNCE, Time Contrastive Blocks, Behavior Contrastive Blocks |
| optimizer budget | 2000 steps for each neural combination |
| output root | `outputs/real_monkey_area2_active_staged_2026-09-22/` |

The staged output contains data and windows, model checkpoints, frozen raw/unit
embeddings, metric CSVs, and raw-vs-unit figures. The held-out branch is the
generalization check: models are fitted on training trials and metrics are
reported on test trials. The `full_sample` branch fits on all trials and is
descriptive; its metric implementation still reports the same held-out test
rows for comparability, so it must not be presented as an independent
generalization estimate.

Real-data metrics are deliberately separated into task decoding (direction,
progress, position, velocity), input-alignment diagnostics (linear CKA),
noise robustness, and participation ratio. CKA here compares an embedding to
the input window, not to a latent ground truth, and is not an overall quality
score. The branch currently does not include a biological lag estimate,
lag-lag matrix, cross-session/animal generalization, or latent recovery.

For the numerical table and interpretation of the first run, see the separate
report `legacy_ai_for_all/Miei doc/Obsidian Vault/05_progetti/Neuro_Bridge/NeuroBridge - Rapporto real monkey 2026-09-22.md`.

## Output map

Use these locations instead of searching all generated folders:

- **Synthetic reference:** `outputs/output_2026-08-27_comparison_2000/`.
- **Synthetic audits:** `outputs/synthetic_v1_raw_unit_cka_2026-09-19/`,
  `outputs/audit_2026-09-19_raw_unit_cka_*`, and
  `outputs/audit_2026-09-20_lag_shuffle_*`.
- **Real-monkey reference:** `outputs/real_monkey_area2_active_staged_2026-09-22/`.
- **Summary table:** `outputs/results_recap_2026-09-22/results_recap_2026-09-22.xlsx`.

The remaining `outputs/output_2026-08-26_*` and
`outputs/output_2026-08-27_timing_100/` directories are historical runs, not
the current reference. They are retained for traceability and should not be
used as the default source for new claims.
