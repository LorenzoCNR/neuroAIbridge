# NeuroBridge Python Pipeline Map

This is a source-oriented guide to how the Python code is connected. It
describes the repository as it exists on 2026-10-07; it is not a new
experimental protocol and does not replace the frozen configuration,
provenance manifests, or scientific explanations in `docs/01`–`docs/06`.

The repository contains several generations of work. A script being present
does not make it part of the current final analysis. For current results, read
the relevant report and provenance file under `outputs/` and follow the parent
hashes. Do not infer that an old notebook, exploratory runner, or adapter was
used to produce the frozen results.

## The dependency chain in one view

```text
simulation generator OR preprocessed real recording
        │
        ├── trial split and trial-safe temporal windows + metadata
        │
        ├── train-only PCA / neural-model fitting
        │          └── validation, early stopping, checkpoint + history
        │
        ├── frozen transform / embedding export
        │          └── raw embedding + unit embedding + trial/time/condition IDs
        │
        ├── evaluation on the declared split and common valid support
        │          ├── geometry / latent or cross-population structure
        │          ├── task accessibility / decoding
        │          ├── temporal alignment / lag
        │          └── diagnostics and computational performance
        │
        └── uncertainty and null analyses of frozen outputs
                   └── tables, figures, provenance, and claim-level summaries
```

The source package provides reusable building blocks. `src/neurobridge/experiments/`
contains experiment-specific orchestration, while `tools/` contains command-line
entry points, final-study adapters, and downstream analysis scripts. Most
`tools/` scripts import shared functions rather than implementing a second
model stack.

## 1. Reusable package layers

| Layer | Main files | What flows through it |
|---|---|---|
| Simulation | `data/sim/Lat_traj_generator.py`, `data/sim/builders.py`, `data/sim/Spikes_generator.py` | Task latent trajectories become population-specific drives/rates and simulated spike counts; the multi-population simulator also supports a controlled digital shift. |
| Data and windows | `data/io.py`, `data/preprocess.py`, `data/dataset.py`, `sampling/f_windows.py` | Load/preprocess arrays, split whole trials, and build centered windows without crossing trial boundaries. Each sample carries trial/time/label metadata and a validity mask. |
| Sampling and loss targets | `sampling/batch_similarity.py`, `sampling/positive_weights.py`, `sampling/labelled.py`, `losses/infonce.py` | Convert time/condition/behavior metadata into pairwise similarities, positive masks/weights, or valid triplets; objective code consumes these targets. |
| Encoders | `models/temporal_cnn.py`, `models/blocks.py` | PCA is handled by experiment code; neural encoders include temporal CNN and Transformer plus other reusable architectures. Frozen final embeddings are 3-D. |
| Optimization and transform | `train/loop.py` | `train_steps` / `train_epoch` handle pairwise objectives; `train_triplet_steps` / `train_triplet_epoch` handle triplets; `encode_windows` transforms batches; `profile_training_steps` measures short compute profiles. |
| Evaluation | `eval/representation.py`, `eval/knn_decoder.py` | Geometry/alignment functions include Procrustes, RSA-style distance correlation, CKA, and lagged alignment; decoder helpers handle held-out task prediction. |
| Plotting and paths | `viz/plots.py`, `viz/manifold_plots.py`, `utils/paths.py`, `utils/project_store.py`, `utils/config.py`, `utils/io.py` | Shared figure routines, project/output paths, configuration parsing, and serialized artifacts. |

The metadata contract is important: `trial_id`, `time_id`, condition/behavior
labels, `valid_mask`, and split identity must stay aligned row-for-row with each
embedding. Lag and cross-population comparisons match explicit trial/time
coordinates on the common valid support; they must not reconstruct those
coordinates from a presumed number or order of windows.

## 2. Synthetic staged benchmark

The cacheable staged reference is implemented in
`src/neurobridge/experiments/staged_shared_latent.py`. Its public driver is
`run_staged_benchmark`; the stages can also be invoked separately:

1. `stage_generate` calls the simulation builders and stores the latent,
   population observations, validity information, configuration, and parent
   provenance in `stage01_data/`.
2. `stage_windows` makes a trial-level split before materializing centered
   windows and metadata in `stage02_windows/`. `build_windows` and
   `TemporalWindowDataset` implement the underlying window/sample contract.
3. `stage_fit` creates the train/validation loaders, constructs a PCA or neural
   encoder, evaluates the configured objective, records train/validation
   history, and writes the selected/stopping checkpoints under `stage03_fit/`.
   Validation controls checkpoint selection/early stopping; held-out test
   observations are not model-selection inputs.
4. `stage_transform` / `stage_encode` load the checkpoint and write cached
   `embedding_raw` and `embedding_unit` with explicit metadata in
   `stage04_embeddings/`.
5. `stage_evaluate` writes branch-aware scientific metric tables under
   `stage05_metrics/`. Held-out is the generalization branch; full-sample
   metrics, where defined, are descriptive and are not a substitute for
   held-out evidence.
6. `stage_plot`, `stage_plot_raw`, `stage_plot_ground_truth_latent`,
   `stage_plot_training_history`, and `stage_plot_lag_profile` render distinct
   views from cached embeddings, latent, or history. `stage_profile` records
   computational measurements; `stage_quality_report` assembles the separate
   quality dimensions.

The stage directory is derived from `SharedLatentStageConfig` and a protocol
fingerprint. A valid cached parent is reused when its identity/configuration
matches; a changed upstream dependency requires its downstream children to be
revalidated. A plot-only change should not retrain a model. Never copy an
artifact between settings just because its filename matches.

Earlier notebook-style task experiments are also present in
`src/neurobridge/experiments/synthetic_task_suite.py` and `notebooks/`. They are
useful executable examples, but the staged run and its recorded configuration
are the reference for staged-benchmark results.

## 3. Real Area-2 branches: keep them distinct

There are three related but non-interchangeable code paths:

* `experiments/real_monkey.py` is the earlier cacheable natural-recording
  pipeline. Its `prepare_real_monkey`, `stage_windows`, `fit_pca_embeddings`,
  `fit_neural_model`, `transform_neural_embeddings`, `evaluate_pca`,
  `evaluate_neural`, and plotting functions form a staged path. It is useful
  implementation history, not by itself the frozen final-thesis selection.
* `experiments/real_monkey_validated.py` contains validation-aware training,
  cache/provenance checks, geometry and decoding evaluation, and diagnostic
  utilities. `run_natural_monkey_suite` coordinates that branch.
* The frozen final-study path below uses Phase-2A safe bundles and the
  `tools/final_thesis_*.py` adapters. Its immutable spec and output manifests,
  rather than earlier exploratory defaults, define the final Real study.

The real recording supplies behavior labels and neural activity, not a known
biological latent `Z`. Consequently, Real results concern accessibility,
cross-population consistency under the specified partition, temporal
correspondence, and stability; they are not latent-ground-truth recovery.

## 4. Phase-2A validation-only HPO and frozen final Real fits

The relevant entry points are `tools/phase2a_*.py` and
`tools/final_thesis_real_hpo.py`, `tools/final_thesis_real_fit.py`,
`tools/final_thesis_freeze_real.py`, and `tools/final_thesis_run_real.py`.
Core logic lives in `experiments/phase2a_hpo.py`,
`phase2a_safe_inputs.py`, `phase2a_window_inputs.py`, and
`phase2a_safe_fit.py`.

### Preparation and HPO selection

1. `phase2a_hpo.load_sealed_candidates` verifies the sealed candidate-table
   digest; `plan_initial` / `resolved_config` enumerate immutable trial IDs,
   domains, populations, architecture/objective, seeds, and output paths.
2. `phase2a_safe_inputs.extract_one` / `extract_all` materialize train and
   validation only from the already-preprocessed parent observations and
   frozen split. Test is allowed only as partition metadata for exclusion at
   extraction; the safe fit reader receives the child bundle, not the complete
   source container.
3. `phase2a_safe_fit.load_safe_bundle`, `_loaders`, and `fit_trial` pass only
   those safe observations to the optimizer. `early_stop_transition` and the
   frozen validation schedule choose a checkpoint from validation loss. The
   qualification gate checks finite outputs and frozen collapse/eligibility
   rules. Test metrics, `Z` recovery, and downstream scientific metrics are not
   available to candidate selection.
4. `ranked_search_candidates`, `extension_required`, and
   `select_finalist` implement the frozen candidate ranking/extension rules.
   Synthetic and Real are selected separately, and Real windows are ranked
   within their declared architecture/objective/window cells, not by pooling
   raw losses across different objectives.

### Freeze and final fits

`tools/final_thesis_freeze_real.py` verifies the HPO ledgers and extension
provenance, writes winner/window tables, and seals
`outputs/final_thesis_v1/freeze/REAL_FINAL_EXPERIMENT_SPEC.json`. The frozen
spec contains the single selected Real window and final trial enumeration.
`tools/final_thesis_real_fit.py` validates each trial/config/spec/hash contract
and writes immutable trial records, training history, checkpoints, and
validation artifacts. `tools/final_thesis_run_real.py` is the resume-aware
orchestrator: it checks the frozen spec, reuses only explicitly authorized
seed-1101 HPO checkpoints, runs the remaining frozen seed fits, and refreshes
the rolling fit table without changing completed trials.

PCA is generated separately by `tools/generate_real_pca_reference.py` as a
deterministic reference, rather than being repeated as if it were a stochastic
neural training seed.

## 5. Frozen embeddings, core evaluation, and downstream branches

`tools/final_thesis_core_metrics.py` is both a source/provenance audit and the
core embedding/metric exporter:

* `--audit` validates the frozen run slots, source/split/partition hashes,
  checkpoint ancestry, embedding inventory, and evaluation support before
  analysis.
* `--run` verifies/reuses valid frozen outputs, exports the embedding index and
  metadata, computes core metric rows and summaries, and records provenance.
  It does not retrain encoders. `raw` and `unit` are distinct representations
  and remain separate in the index and tables.

The core branch is under
`outputs/final_thesis_v1/final_evaluation/core_metrics/`. The primary
generalization evidence is held-out. Full-sample decoding, if present, is
explicitly descriptive/in-sample; it must not be presented as held-out
generalization. The core scientific dimensions remain separate: whole-process
geometry, task accessibility, temporal alignment, diagnostics, and compute.

The Real train-fit PCA reference is a separate deterministic baseline. Its
frozen embeddings and PCA models live under
`outputs/final_thesis_v1/final_evaluation/real_pca_reference/`; the downstream
evaluator `tools/evaluate_real_pca_reference.py` writes its metrics beneath
`real_pca_reference/metrics/`. It reuses the saved PCA projections, applies the
same train/validation/test probe discipline and existing metric definitions,
and does not modify `core_metrics/CORE_METRICS_LONG.csv` or retrain neural
encoders. Held-out PCA accessibility and matched A/B consistency are separate
from all-valid descriptive diagnostics; raw and unit results remain distinct.

`tools/final_thesis_downstream_eval.py` loads the frozen embedding index/support
and runs downstream branches. Its `run_controlled_lag` and `run_null_controls`
consume the same declared test support and save checkpointed analysis slots;
they do not fit encoders. `tools/final_thesis_uncertainty.py` separately
aggregates individual training seeds and resamples complete held-out trials;
windows are not treated as independent bootstrap units.

Later controlled-lag versions are separate namespaces, not replacements for
the original result. `tools/final_real_controlled_lag_v3_2sd.py` reuses frozen
embeddings and recomputes lag profiles on a broader range; its constants,
parent hashes, and output directory identify that version. The corrected
results reside in `controlled_lag_v3_2sd_corrected/`. The lag-revision helpers
in `experiments/final_real_lag_revision.py` define safe shifts/window support
for the corresponding versioned branch. R0 is relative correspondence; a
digital B-only shift tests response to a known intervention. Neither is a
causal or biological synaptic-delay estimate.

The current gap-closure scripts are additional frozen-output analyses, each in
its own versioned directory:

| Script | Function in the analysis chain | Output namespace |
|---|---|---|
| `tools/final_gap_closure.py` | Trial-level bootstrap of synthetic whole-latent geometry. | `gap_closure_v2/synthetic_z_bootstrap/` |
| `tools/final_gap_closure_completion.py` | Reconstructs/verifies the exact synthetic source and completes whole-`Z`, stratified-BACC, and label-shuffle extensions. | `gap_closure_v2/` sub-branches |
| `tools/final_gap_closure_merge_recovery.py` | Merges already checkpointed shuffle results after validating identities; it is a recovery/aggregation path, not a new model fit. | `gap_closure_v2/label_shuffle_holm_3000/` |
| `tools/final_gap_bootstrap_completion.py`, `tools/final_gap_real_rsa_bootstrap.py` | Additional frozen-input bootstrap summaries for specified metrics. | `gap_closure_v2/` bootstrap branches |
| `tools/final_gap_temporal_null.py` | Block-based temporal-order null on frozen Synthetic held-out embeddings. | `gap_closure_v1/temporal_null/` |
| `tools/final_gap_multiplicity.py` | Applies the prespecified Holm families to existing p-values. | `gap_closure_v1/claim_inference/` |
| `tools/final_gap_c8.py` | Train-defined C8 condition-centroid generator evaluated on held-out synthetic centroids. | `gap_closure_v1/symmetry_c8/` |

Each branch records parent hashes and settings. Intermediate replicate/checkpoint
files may be kept locally but excluded from Git when they are too large; compact
summaries, reports, seed tables, figures, and provenance are the review surface.

## 6. Separate Frosolone-inspired Synthetic analysis

`tools/frosolone_pipeline1_synthetic.py` is a downstream methodological
adaptation; it does not retrain the NeuroBridge encoders and is not part of the
core metric table. `read_inputs` loads the cached synthetic neural arrays,
validity masks, labels, configuration, and frozen split while deliberately
not loading latent truth for this discrimination task.

For each synthetic population, it fits PCA on valid training observations,
scans component-by-time one-way ANOVA effect sizes to select the NMA interval
using training trials only, and then holds that PCA basis, selected component,
and interval fixed. It constructs one-vs-rest CSP features for the full valid
interval and for the selected interval, selects mutual-information features
using training labels, and fits a quadratic discriminant classifier on train
features. Validation and held-out test are reported separately. The held-out
NMA profile is a post-selection description; it is not used to move the
selected interval. Tables, confusion matrices, centroid-distance plots,
selected-window diagnostics, and a report are written to the separate
`outputs/frosolone_synthetic_pipeline1_seed42_complete/` branch. This is an
adaptation to simulated neural counts, not a literal reproduction of an EEG
paper.

**Interpretation caution:** the current interval selection takes the minimum
raw ANOVA p-value over the scanned component/time candidates; that search is
adaptive and the p-values are not, by themselves, multiplicity-adjusted
confirmatory evidence. Treat NMA as a train-only feature/window selection step
and rely on held-out decoder performance for generalization. The exact number
of scan candidates and full details are recorded in the run's report/tables.

## 7. Artifact, split, cache, and provenance rules

* A split is made at the complete-trial level before overlapping windows are
  formed. All rows for one trial belong to exactly one split.
* Training fits use train observations. Validation is used for checkpoint,
  early-stop, or frozen HPO decisions only where the protocol explicitly says
  so. Test outcomes are downstream evaluation, never model/HPO selection.
* A metric branch consumes frozen checkpoints/embeddings and the same declared
  support. Downstream changes do not justify encoder retraining.
* `raw` and row-unit-normalized embedding matrices are separate objects. A
  normalization choice is not silently selected after looking at the result.
* Immutable branches write once. Existing artifacts are accepted only when
  identity and hashes match; a conflicting file is a stop/error, not an
  overwrite instruction.
* Resumable analyses checkpoint units of work and verify parent/config/code
  hashes before reuse. A downstream configuration change should invalidate only
  its dependent stage; any broader invalidation needs explicit dependency
  evidence.
* Manifests should retain run/trial ID, population, architecture, objective,
  seed, split, config/data/split/partition hashes, parent artifact IDs/hashes,
  and code/provenance where applicable.
* Compact outputs are versioned under `outputs/`. `.gitignore` excludes raw
  neural arrays, binary caches, interactive HTML, and named high-volume
  replicate/checkpoint files; those local files are not deleted by Git rules.

## 8. Tests and where to investigate failures

Tests are grouped by contracts rather than one monolithic runner:

* `tests/test_p0_contracts.py`, `test_learning_components.py`,
  `test_similarity.py`, and `test_representation_eval.py` cover package-level
  data/loss/model/evaluation pieces.
* `tests/test_phase2a_hpo.py`, `test_phase2a_safe_execution.py`,
  `test_phase2a_somatotopic.py`, and `test_phase2a_window_campaign.py` cover
  sealed candidates, safe-input boundaries, population/window mapping, and
  resume/provenance contracts.
* `tests/test_final_thesis_*.py`, `test_final_real_embeddings.py`, and
  `test_final_real_lag_revision.py` cover frozen final-study manifests,
  embedding invariants, and lag revision support.
* `tests/test_final_gap_closure.py` covers the gap-closure helpers; the
  analysis scripts also perform runtime checks against frozen parent metrics
  and hash manifests.

For exact scientific definitions, consult `docs/03_LEARNING_OBJECTIVES.md`,
`docs/05_EVALUATION_AND_MULTISUBJECT.md`, and the versioned protocol/provenance
files beside each output. This map explains which Python components connect;
the frozen artifacts remain the authority for what was actually run.
