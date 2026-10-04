# Real-data results: branch-aware report

## Coverage and protocol

- Completed baseline real-data runs: **9** (training seeds 42, 123, 456; populations total_65, A, B).
- Stage-5 branches: **held_out** and **full_sample** for every run (18 run/branch combinations).
- Each run/branch has **18 Stage-5 CSV artifacts**: point metrics, representation geometry, and the existing trial-bootstrap summaries.
- The real-data frozen schedule is `max_iterations=4000`, validation interval 150, minimum 750 updates, patience 5, and relative minimum delta 0.001; channel-partition seed is 42. Model/objective names, measured values, raw/unit representations, scope/split metadata, and available bootstrap intervals are kept run-wise in [REAL_BRANCH_METRICS.csv](../tables/REAL_BRANCH_METRICS.csv).
- No model was retrained for this reporting repair; the table was assembled from completed manifests and cached Stage-5 outputs.

## How to read the branches

- **held_out** is the primary evidence for held-out accessibility/generalization.
- **full_sample** is explicitly marked `descriptive_in_sample_representation`: its representation was fit using all trials. Its decoder rows are descriptive and must not be presented as an independent generalization estimate.
- Full-sample CKA and window-level participation-ratio diagnostics use all windows; their held-out versions use test windows. Trial-prototype geometry/RSA and its prototype-level participation-ratio diagnostic use 193 trials in `full_sample` and the recorded test trials in `held_out`. These are distinct records; the table preserves their categories, references, and scopes rather than pooling them.
- `raw` and `unit` are separate rows. RSA rows retain their behavioral reference geometry and existing bootstrap intervals. Decoder robustness rows retain their original category and perturbation fields.

## Metric contents

This is an aggregation of existing metrics only: task/behavior decoding, post-hoc embedding-noise robustness, linear CKA to input windows, participation ratio, and behavioral-geometry RSA. No metric is introduced, recomputed with a new definition, or averaged across seeds in this table.

Controlled-lag R10 artifacts remain in their separate run folders and are not
mixed into this R0 baseline table.

## Run manifests

- `clean_rebuild_2026-09-23_seed42_real_0_A` - [manifest](../../runs/clean_rebuild_2026-09-23_seed42_real_0_A/run_manifest.json)
- `clean_rebuild_2026-09-23_seed42_real_0_B` - [manifest](../../runs/clean_rebuild_2026-09-23_seed42_real_0_B/run_manifest.json)
- `v2_corrected_2026-09-23_seed42_real_total_65` - [manifest](../../runs/v2_corrected_2026-09-23_seed42_real_total_65/run_manifest.json)
- `v2_corrected_2026-09-23_seed123_real_0_A` - [manifest](../../runs/v2_corrected_2026-09-23_seed123_real_0_A/run_manifest.json)
- `v2_corrected_2026-09-23_seed123_real_0_B` - [manifest](../../runs/v2_corrected_2026-09-23_seed123_real_0_B/run_manifest.json)
- `v2_corrected_2026-09-23_seed123_real_total_65` - [manifest](../../runs/v2_corrected_2026-09-23_seed123_real_total_65/run_manifest.json)
- `v2_corrected_2026-09-23_seed456_real_0_A` - [manifest](../../runs/v2_corrected_2026-09-23_seed456_real_0_A/run_manifest.json)
- `v2_corrected_2026-09-23_seed456_real_0_B` - [manifest](../../runs/v2_corrected_2026-09-23_seed456_real_0_B/run_manifest.json)
- `v2_corrected_2026-09-23_seed456_real_total_65` - [manifest](../../runs/v2_corrected_2026-09-23_seed456_real_total_65/run_manifest.json)

## Full-sample figures

The evaluation stage generated the branch-matched RSA figure for every real
run; these are linked from their immutable run folders:

- `clean_rebuild_2026-09-23_seed42_real_0_A` - [full-sample behavioral-geometry RSA](../../runs/clean_rebuild_2026-09-23_seed42_real_0_A/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)
- `clean_rebuild_2026-09-23_seed42_real_0_B` - [full-sample behavioral-geometry RSA](../../runs/clean_rebuild_2026-09-23_seed42_real_0_B/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)
- `v2_corrected_2026-09-23_seed42_real_total_65` - [full-sample behavioral-geometry RSA](../../runs/v2_corrected_2026-09-23_seed42_real_total_65/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)
- `v2_corrected_2026-09-23_seed123_real_0_A` - [full-sample behavioral-geometry RSA](../../runs/v2_corrected_2026-09-23_seed123_real_0_A/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)
- `v2_corrected_2026-09-23_seed123_real_0_B` - [full-sample behavioral-geometry RSA](../../runs/v2_corrected_2026-09-23_seed123_real_0_B/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)
- `v2_corrected_2026-09-23_seed123_real_total_65` - [full-sample behavioral-geometry RSA](../../runs/v2_corrected_2026-09-23_seed123_real_total_65/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)
- `v2_corrected_2026-09-23_seed456_real_0_A` - [full-sample behavioral-geometry RSA](../../runs/v2_corrected_2026-09-23_seed456_real_0_A/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)
- `v2_corrected_2026-09-23_seed456_real_0_B` - [full-sample behavioral-geometry RSA](../../runs/v2_corrected_2026-09-23_seed456_real_0_B/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)
- `v2_corrected_2026-09-23_seed456_real_total_65` - [full-sample behavioral-geometry RSA](../../runs/v2_corrected_2026-09-23_seed456_real_total_65/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)

Embedding/trajectory and raw-versus-unit figures remain alongside these under
each run's `stage06_figures/<branch>/` directory.

## Existing synthetic table

The pre-existing [MAIN_RESULTS_TABLE.csv](../tables/MAIN_RESULTS_TABLE.csv) is preserved unchanged; it contains the synthetic seed-42 aggregation. This real-data table is separate so its branch scopes and evaluation provenance remain explicit.
