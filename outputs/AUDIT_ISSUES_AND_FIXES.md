# Audit issues and fixes — clean rebuild 2026-09-23

This log is append-only for this rebuild. It was created after inventory and
before editing project source. No model training has started yet. The
repository worktree was already dirty at inspection (421 porcelain status
lines); those changes are user-owned and are not reverted or attributed to
this rebuild.

## Baseline and provenance

- Output inventory at start: `outputs/` existed and contained **0 files**.
- Raw real data: `data/monkey_reaching_preload_smth_40/macaque_data.jl`, SHA-256
  `bf6b9eab46d8daaa8f6a9e69c7df1dbd9e337c60ee38af520042d3047c244d13`.
- Verified source dimensions: 193 trials × 600 bins × 65 channels; direction
  labels 0–7; position and velocity each have two columns. Loader documentation
  identifies 1-ms bins and 40-ms smoothing.
- Code commit before corrections: `9efbbb5fbe73404d790b4052011e292fdc09ea07`.
- The working tree was dirty before this task. To preserve the exact pre-edit
  code state, these SHA-256 hashes were captured before corrections:

  | File | Pre-correction SHA-256 |
  |---|---|
  | `src/neurobridge/experiments/real_monkey.py` | `9C2B6F1EAC664B388DCAEA89C9CAA7A7C1D2B6E556FF16910F452A1D5253F1B7` |
  | `src/neurobridge/experiments/real_monkey_validated.py` | `6FA7ADC5FD37AB9B96F9AC9582A5FCE2309B4EEB032DE765BA9F8EF597C1BEF6` |
  | `src/neurobridge/experiments/staged_shared_latent.py` | `D7618493478D8C1C77BA2B9CE0EBFEDCFDF964AA6372B40D38DAF6F962868265` |
  | `src/neurobridge/train/loop.py` | `9356D0231E93618918049940F9895191A0E31BFA14127F91F15E193526CAF823` |
  | `src/neurobridge/losses/infonce.py` | `9427914FC50F24E6DCD6EA5DF01C5ED687D4AAD3CF65A9C3A673F91F67F2838C` |
  | `src/neurobridge/sampling/cebra_time.py` | `2CA95D3CFA25CB0E52F8EF030D102B6DBA865B8C432BE4CFAD64CA928146ED1D` |
  | `src/neurobridge/sampling/cebra_behavior.py` | `2FF98FDB048CC6CDC1A4A5F4F9E6436431B29E18FC947DC7B3B990F2008EBD29` |

## Permanent issue table

| ID | Severity | Component | What was wrong | Evidence | Scientific impact | Fix | Existing results affected | Rerun required? |
|---|---|---|---|---|---|---|---|---|
| NB-RB-01 | CRITICAL | Monkey cold start | New runner requires a deleted parent cache/PCA instead of rebuilding from raw data | `run_natural_monkey_suite()` calls `reuse_natural_input_cache()` and `reuse_train_only_pca()`; output inventory is empty | Experiment cannot start under the explicit clean-rebuild requirement | Build stage-1 data/split/windows directly from raw data; fit PCA in the new run | No current files; all deleted prior outputs remain unavailable | Yes, all new fits; no old artifact is reused |
| NB-RB-02 | MAJOR | Real schedule | Real validated default is 5,000, while the current user-specified rebuild protocol is 4,000 | `ValidationSchedule.max_steps=5000`; validation interval 150, minimum 750, patience 5, relative delta .001 | Violates the newly frozen common cap; changes stopping opportunities and compute | Set real max updates to 4,000; retain the specified real validation settings | No current files | Yes, new real fits |
| NB-RB-03 | MAJOR | Real training batches | Training loaders allow a final partial batch, so candidate count may shrink below B=1024 | Three `DataLoader(... drop_last=False)` paths in `fit_validated_neural_model`; train K is measured per update | Changes the effective contrastive candidate pool on tail updates | Use full training batches and log omitted tail count; fail if fewer than 1,024 eligible anchors | Any older result using partial tails; those outputs were deleted | Yes for affected new fits |
| NB-RB-04 | MAJOR | Real paired settings | Active real pipeline fits all 65 channels together; it has no fixed 32/33 population split or R0/R10 pair | `real_monkey.stage_windows()` uses all `spikes_active`; no shift/validity-mask setting exists | Cross-population alignment and controlled-lag claims cannot be estimated | Deterministic 32/33 split recorded once; same trials/channels for R0 and R10; non-circular +10-bin B shift with invalid edge mask | No current files | Yes, new R0/R10 fits |
| NB-RB-05 | MAJOR | Synthetic protocol/validation | Current staged defaults are B=256, 2,000 updates, patience=0; triplet validation is resampled at checkpoints and loaders keep partial tails | `SharedLatentStageConfig`; `stage_fit()` resamples validation triplets and uses `drop_last=False` | Protocol differs from the current brief; validation-based checkpoint comparisons are stochastic and K can vary | New run config B=1024/cap=4,000; validation 400/minimum 800/patience 3/delta .001; freeze validation triplets and keep full batches | Prior files were deleted; no old metric is imported | Yes, new synthetic fits |
| NB-RB-06 | MAJOR | Synthetic lag curves | Candidate lags are scored on different trial/time supports | `stage_evaluate()` calls `lagged_alignment_by_trial_time()` without `common_support=True` | Lag-score differences can reflect different observations, not only lag | Use one strict intersection support for every candidate lag; save support and counts | Any prior lag curves; deleted and not reused | Recompute all new lag curves from new embeddings |
| NB-RB-07 | MAJOR | Full-sample branch | Validated real runner hardcodes held-out only; required full-sample models need N_best updates from fresh initialization | Manifest says full sample not run; runner uses `branch="held_out"` | Missing complete-manifold descriptive analysis; full-sample cannot be mislabeled as generalization | Add separate all-trials branch trained from scratch for exactly held-out N_best; no validation series | No current files | Yes, full-sample fits and downstream outputs |
| NB-RB-08 | MINOR | MATLAB handoff | Cached NPZs contain some metadata, but there is no MATLAB-friendly, explicit raw/unit embedding export with complete run metadata | No active MATLAB exporter found; user needs lag-lag matrices downstream | Manual reconstruction risks row/time/trial misalignment | Export per-fit `.mat` containing raw/unit embedding, trial/time/global-time, behavior, validity, split and setting/population/model/objective/seed/channels | No current files | No retraining beyond the new embeddings; export follows each embedding |
| NB-RB-09 | MAJOR | Presentation/compute reporting | No unified presentation bundle or complete PyTorch compute table/plots exists | No central `presentation_ready` writer or `COMPUTE_PERFORMANCE.csv`; per-fit records omit inference throughput in at least the synthetic path | Compute efficiency and representation quality are hard to compare; output is not presentation-ready | Assemble progressive CSV/summary and high-resolution figures, plus measured per-fit train/inference resources and CNN-vs-Transformer comparisons | No current files | Recompute/export from new fits; no extra training |
| NB-RB-10 | MAJOR | Synthetic cache/resume | Cache validation compares internal config names against a public config that has renamed those entries | `_public_config()` stores `time_offset_bins=10` and `temporal_objective_temperature=1.0`; `check_cached_config()` looks for `cebra_time_offset` and `cebra_temperature`, reproducing two mismatches | Normal stage re-entry can stop at `_stage_dir()` when a stage-1 cache exists, so the synthetic pipeline cannot reliably resume at later stages | Make cache validation understand the public aliases while preserving strict value checks | No current scientific result; synthetic outputs were empty at inspection | Code correction only; no retraining |
| NB-RB-11 | MINOR | Window validity contract | Metadata validator checks only center-bin validity although window generation requires every sample in the window to be valid | `pytest -q tests/test_p0_contracts.py` failed at `validate_metadata()` on a fixture with patterned invalid bins; generated `lag_valid` is a strict-window mask | Qualification reports a false metadata inconsistency and cannot pass for the intended boundary mask | Validate the complete window support against source `valid_*` bins | No model/embedding/metric output affected; qualification check only | Code/test correction only; no retraining |
| NB-RB-12 | CRITICAL | Validated monkey embedding stage | Validated fit and legacy transform use the same model directory but incompatible cache schemas; transform invokes the legacy fitter again | Validated config is top-level public fields plus schedule/objective; legacy fitter expects a nested `config` object and therefore rejects it | R0/R10 training can finish but cannot proceed to embedding/metrics; restarting at transform repeats the same failure | Allow transform to consume the already selected validated checkpoint directly, with cache provenance checks | No current results; training artifacts produced before this stage can be reused | Code correction; if a fit has already completed, resume at transform and do not retrain |
| NB-RB-13 | MINOR | Resume manifest/progress | Real runner drops prior completed units on restart and reports a total equal to only one branch although it appends held-out and full-sample units | `completed=[]` after reading the prior manifest; loop appends two strings but `total_fits=len(models)*len(objectives)` | Progress can restart at zero or exceed 100%, misleading the user but not affecting model results | Restore completed units for a matching run and report two branch-fits per model/objective | Manifest/progress reporting only; no scientific artifact affected | Code/reporting correction only; no retraining |
| NB-RB-14 | MINOR | Figure cache | The main neural figure cache keys only on file existence; a changed plot parameter can leave the main plot stale while companion plots are rewritten | `plot_neural_embeddings()` returns the existing main PNG without checking DPI or parent hashes, then regenerates shared views | Figure bundle can mix rendering settings; no training/embedding or metric effect | Record figure settings and parent hashes; reuse only a matching bundle or require explicit plot-only regeneration | Figures only; no scientific result affected | Plot regeneration only; no retraining or embedding generation |
| NB-RB-15 | MAJOR | Top-level phase manifest | Separate synthetic then real invocations are rejected because the manifest treats an absent synthetic channel hash and later real channel hash as incompatible protocols | `_load_or_create_manifest()` compares the entire expected protocol dict, including nullable `channel_partition_sha256` | Cannot resume across requested top-level phases from separate invocations | Merge a newly available canonical partition hash while enforcing equality when both invocations have one | Manifest/phase bookkeeping only; no scientific result affected | Code/reporting correction only; no retraining |
| NB-RB-16 | MAJOR | Full-sample fit cache | Cache validation requires `full_sample_updates`, but the full-sample config writer omits that key, so every complete full-sample cache is treated as a protocol mismatch | `fit_validated_neural_model()` compares `cached.get("full_sample_updates") == full_sample_updates` at line 626; the config writer at line 921 omits the key; the failed qualification's config JSON has no `full_sample_updates` despite its checkpoint being trained for 1 update | Full-sample checkpoint cannot be reused on restart; the suite fails at cache validation instead of resuming downstream | Persist the already-used `full_sample_updates` in the full-sample config metadata; keep strict cache validation | Failed qualification run `resume_qualification_20260923T181133_351249Z`; produced checkpoint/embedding remain intact but suite resume did not complete; no V2 scientific outputs are affected | Code correction, then qualification rerun with new run ID; no scientific retraining |
| NB-RB-17 | MINOR | Synthetic trajectory figures | The raw/unit trajectory plotting helper skips direction groups after an invalid reshape | During `clean_rebuild_2026-09-23_seed42_synthetic`, `plot_direction_averaged_embedding()` in `src/neurobridge/viz/manifold_plots.py:759` printed `cannot reshape array of size 2700 into shape (200,3)` (and analogous 2550/190 cases) for labels 1–8 | Some requested direction-averaged trajectories may be absent from those figures; model fitting and already computed metrics are separate stages and are not changed by this plotting exception | After the active training run, correct the grouping/support logic and regenerate only affected synthetic figures; do not retrain | Current synthetic raw/unit trajectory figures in this V2 run may be incomplete; embeddings/checkpoints/metrics are not invalidated by this warning | Plot-only correction and figure regeneration; no retraining, subject to confirming no downstream metric consumes this helper |
| NB-RB-18 | MAJOR | Real total population | Runner only creates 32-channel A and 33-channel B configs; no all-65 run exists | `run_seed42_rebuild()` only passes `A_channel_indices` and `B_channel_indices` to `canonical_config()` | No natural-real representation result using the complete recorded population | Add a distinct resumable 65-channel run root using the same frozen suite | Existing Synthetic/R0 A/B results unaffected | New 65-channel fits required |
| NB-RB-19 | MAJOR | Training-seed coverage | Orchestrator has only seed 42 for Synthetic and real data | Literal `SEED=42`; no iteration over `{42,123,456}` | No multi-seed mean/SD or cross-seed consistency result; seed-42 artifacts remain valid | Add training seeds 123 and 456 with separate run roots; keep Synthetic data/split seeds fixed at 42 | Existing seed-42 runs reused read-only | New fits for missing seeds only |
| NB-RB-20 | MAJOR | Controlled real lag | REAL-10 shifts both A and B and independently fits each setting | Runner passes `imposed_shift_bins=10` to both populations and calls `run_natural_monkey_suite()` for REAL-10 | Shift and refitting are confounded; current REAL-10 cannot support controlled R0-to-R10 lag claims | Preserve current artifacts; new R10 transform-only branch shifts B and reuses R0 checkpoints | Synthetic/R0 unaffected; independent REAL-10 outputs retained but excluded from controlled-lag claim | New shifted embeddings/lag metrics only; no retraining |
| NB-RB-21 | MAJOR | Controlled lag evaluator draft | First draft passed already-offset B coordinates with a zero-lag request | `_lag_curves_on_joint_support()` supplied `time_id_other=time+lag` and then `lags=(0,)` | No output existed yet; if used, non-zero candidate pairs would be missing | Pass the frozen candidate lag list using original B coordinates over joint support | No existing artifacts affected | Code-only correction and focused lag test; no retraining |
| NB-RB-22 | MAJOR | Real lag scan range | REAL-0 optima saturate the frozen `[-20,+20]` scan, so REAL-10 peak delta is not resolved | Seed42 controlled scan: 15/16 R0 and 16/16 R10 peaks at a scan boundary; observed deltas only 0 or 3; independent transform check confirms B10(t+10)=B0(t) within `8.94e-8` for CNN/soft | Digital shift is implemented, but this scan cannot establish lag recovery for current A/B embeddings; no representation-quality metric is affected | No code correction under the frozen range; report boundary flags and non-conclusive lag. Widening the grid would be a separately approved protocol decision | Only seed42 controlled-lag interpretation; no checkpoint/embedding/decoding/geometry results invalidated | No retraining or recomputation; retain current metrics and boundary diagnostics |
| NB-RB-23 | MINOR | Lag artifact provenance | Lag CSV rows lack explicit configuration hashes; aggregate provenance does not map every row to its exact A/B/R10 config hash | `lag_curves_r0_vs_r10.csv` and `lag_recovery_summary.csv` contain seed/model/objective/branch, while the JSON sidecar lists parent artifact hashes but no per-config hash mapping | Values are unchanged, but exact config-to-row traceability is incomplete | Add a new provenance sidecar mapping each seed/population/model/objective/branch to config and parent hashes; retain existing CSVs unchanged | Seed42 lag values remain valid; metadata only is incomplete | Provenance-only update; no metric recomputation or retraining |
| NB-RB-24 | MINOR | Corrected V2 progress manifest | Population subphase states could remain `running` after their run IDs completed during an `all` invocation | `_ensure_real_population()` records `real_total_65` or `real_A/B`, while `run_corrected_v2_rebuild()` only closes the top-level `all` phase | Progress JSON could misstate phase completion; no training or scientific output changes | Mark each subphase complete after its requested seed set completed; preserve every run record | Corrected V2 progress manifest only | Manifest-only update; no recomputation or retraining; resolved after main process exit |
| NB-RB-25 | MAJOR | Real `full_sample` metric stage | Real-data orchestration creates full-sample fits and embeddings but runs scientific metric evaluation only for `held_out` | Seed42 real total-65, A, and B each have 18 held-out metric CSVs and no `stage05_metrics/full_sample/`; their full-sample embeddings/checkpoints exist. Synthetic seed123 has 18 CSVs in each branch | Missing in-sample/descriptive monkey results; held-out generalization metrics remain valid and must stay primary | Run the existing branch-aware evaluators on cached full-sample embeddings/checkpoints; keep decoding explicitly descriptive/in-sample; do not add metric definitions or retrain | Missing full-sample PCA/neural metrics, bootstrap-decoding outputs, and trial-geometry/RSA outputs for completed seed42 real runs; same omission would affect future real seeds absent correction | Evaluation-only recomputation from cached artifacts; no model retraining |
| NB-RB-26 | MINOR / REPORTING ONLY | Real geometry aggregation | Legacy held-out `representation_geometry.csv` rows omit explicit branch/scope and use `n_test_trials` rather than common trial-count field | Example seed42 A geometry file lacks branch/scope columns and records 39 test trials; sibling metric CSV identifies `held_out` | RSA values remain valid, but detached aggregate rows can lose their held-out scope | Reporting-only table normalizes branch from parent folder, evaluation scope from frozen branch semantics, and trial count from existing column; source CSVs stay unchanged | Presentation aggregation only; no metric or model artifact invalidated | Reporting-only; no recomputation or retraining |

## Issue details before correction

### NB-RB-01 — cold-start dependency on deleted parent artifacts

- **Where:** `src/neurobridge/experiments/real_monkey_validated.py`,
  `run_natural_monkey_suite()`, `reuse_natural_input_cache()`,
  `reuse_train_only_pca()`.
- **Current behavior:** runner requires parent run
  `real_monkey_area2_active_staged_2026-09-23_8000`; absence raises before
  training. PCA is also copied from that parent.
- **Expected behavior:** create a new immutable run from the existing raw
  source; no deleted model/cache/embedding/metric is a dependency.
- **Why problematic:** software/delivery blocker; it is not a scientific
  failure and does not justify using stale data.
- **Direct evidence:** `outputs/` contained zero files; raw source exists and
  `_load_source()` successfully reports 193×600×65 activity.
- **Scientific consequence / affected results:** no new checkpoint, embedding,
  metric, lag result, null, robustness, or figure can be produced until the
  input stages are rebuilt. No present output is scientifically affected.
- **Severity:** CRITICAL.
- **Required action:** code correction to input preparation, then new run.
- **Proposed minimal fix:** invoke source load, trial split and window-building
  stages under fresh run IDs; fit PCA from the train portion for held-out and
  separately from all trials for full-sample.
- **Classification:** BUG FIX restoring the requested clean-start behavior.
- **Before/after:** pending implementation and evidence.

### NB-RB-02 — real update cap differs from the frozen rebuild schedule

- **Where:** `src/neurobridge/experiments/real_monkey_validated.py`,
  `ValidationSchedule` and `canonical_config()`.
- **Current behavior:** default cap is 5,000; interval 150; minimum 750;
  patience 5; relative minimum delta .001.
- **Expected behavior:** cap 4,000, retaining interval 150, minimum 750,
  patience 5 and relative minimum delta .001.
- **Why problematic:** protocol inconsistency; a different update budget changes
  stopping opportunities and compute, even though it does not redefine the
  loss itself.
- **Direct evidence:** source defaults above; latest user instruction and
  attached protocol explicitly request 4,000.
- **Scientific consequence / affected results:** neural model/checkpoints,
  embeddings, all downstream metrics and figures from this rebuild.
- **Severity:** MAJOR.
- **Required action:** retrain new real configurations; never overwrite old
  runs.
- **Proposed minimal fix:** change only the real schedule cap to 4,000 and
  version run IDs/config manifests.
- **Classification:** NEW PROTOCOL DECISION explicitly specified by the user,
  not a bug fix.
- **Before/after:** pending implementation and evidence.

### NB-RB-03 — variable effective candidate count in real training tails

- **Where:** `fit_validated_neural_model()` in
  `src/neurobridge/experiments/real_monkey_validated.py`, all three training
  `DataLoader` branches.
- **Current behavior:** `drop_last=False`; the final batch can contain fewer
  than 1,024 anchors. The objective then sees fewer candidates than in normal
  updates. The fit logs a histogram but does not prevent this.
- **Expected behavior:** every optimizer update uses B=1,024 and constant
  objective-specific K (1,023 for pairwise self-excluding losses; 1,024 for
  explicit candidate-pool losses). Validation is already deterministic,
  full-batch, and fixed across checks.
- **Why problematic:** protocol inconsistency that changes the effective
  contrastive task on the tail update; not a change to the requested B.
- **Direct evidence:** the three current training loaders use
  `drop_last=False`; `_batch_loss()` derives K from actual batch size.
- **Scientific consequence / affected results:** one or more training updates,
  stopping checkpoint, best checkpoint, and downstream metrics/plots for the
  affected fit.
- **Severity:** MAJOR.
- **Required action:** retrain any configuration that used a partial update.
- **Proposed minimal fix:** `drop_last=True`; record eligible and dropped
  anchors and assert every observed update is exactly 1,024.
- **Classification:** BUG FIX restoring the stated fixed-candidate protocol.
- **Before/after:** pending implementation and evidence.

### NB-RB-04 — no paired real pseudo-populations or controlled shift

- **Where:** `src/neurobridge/experiments/real_monkey.py`, `_load_source()` and
  `stage_windows()`; current validated runner.
- **Current behavior:** all 65 channels form one input population. No explicit
  channel partition, R0/R10 condition, shift mask, or paired export exists.
- **Expected behavior:** deterministic disjoint A=32/B=33 channel subsets;
  identical subset/trial IDs in R0 and R10; R10 applies B[t]=B[t−10] without
  wrap; invalid shifted support is excluded.
- **Why problematic:** the real-data cross-population/controlled-lag question
  is not currently represented. R0 is not a known biological zero lag, and
  R10 is not a causal intervention.
- **Direct evidence:** raw source has exactly 65 channels; current loader passes
  the complete array to windows and sets all `lag_valid=True`.
- **Scientific consequence / affected results:** all cross-population geometry
  and lag curves for real settings; no effect on synthetic outputs.
- **Severity:** MAJOR.
- **Required action:** implement the explicitly requested R0/R10 setting and
  train the new paired runs.
- **Proposed minimal fix:** one seeded, saved 32/33 partition; copy exact split
  and trial IDs into both settings; apply shift to B only and save a mask.
- **Classification:** NEW PROTOCOL DECISION specified by the user (the exact
  reproducible channel assignment will be seed-42 permutation, recorded in the
  manifest). It is not a claim of biological delay.
- **Before/after:** pending implementation and evidence.

### NB-RB-05 — synthetic config, stochastic validation, and tail batches

- **Where:** `SharedLatentStageConfig` and `stage_fit()` in
  `src/neurobridge/experiments/staged_shared_latent.py`.
- **Current behavior:** defaults are B=256, 2,000 updates, patience 0; loader
  branches allow partial batches; triplet validation assignments are
  resampled after each validation checkpoint; full-sample history writes
  `validation_loss=train_loss` when validation is absent.
- **Expected behavior:** B=1,024; cap 4,000; fixed validation every 400 updates
  after minimum 800; patience 3 and relative improvement .001; fixed validation
  anchors/pairs/candidates and constant K; full-sample has no validation value.
- **Why problematic:** new protocol mismatch, stochastic checkpoint ranking,
  varying objective size, and misleading train=validation records.
- **Direct evidence:** config defaults and `stage_fit()` sampler/loader/history
  code; 140 synthetic training trials × 200 bins and 20 validation trials ×
  200 bins do not divide evenly by 1,024.
- **Scientific consequence / affected results:** all synthetic model selection,
  embeddings and derived scores; the fake full-sample validation is reporting,
  not a real held-out estimate.
- **Severity:** MAJOR.
- **Required action:** train new fits after schedule and sampler corrections;
  old outputs were deleted and are not reused.
- **Proposed minimal fix:** extend the staged config/trainer with the requested
  cadence/minimum/relative criterion, freeze validation triplets once, enforce
  full B, record N_best, then train full-sample exactly N_best from a fresh
  seed-initialized model.
- **Classification:** requested schedule is a NEW PROTOCOL DECISION; freezing
  validation, fixed K, and removing fake validation are BUG FIXES.
- **Before/after:** pending implementation and evidence.

### NB-RB-06 — synthetic lag scores use lag-dependent support

- **Where:** `stage_evaluate()` in
  `src/neurobridge/experiments/staged_shared_latent.py`, call to
  `lagged_alignment_by_trial_time()`.
- **Current behavior:** `common_support` defaults to false, so pairs absent at
  a candidate lag are skipped separately for that lag.
- **Expected behavior:** one strict shared trial/time support for every lag in
  the requested scan, with the exact indices and pair count saved.
- **Why problematic:** candidate lag scores are not based on the same sample.
- **Direct evidence:** function default is `common_support=False`; call site
  does not override it.
- **Scientific consequence / affected results:** synthetic lag curves,
  estimated lag, peak margin/sharpness, and derived lag comparisons only.
- **Severity:** MAJOR.
- **Required action:** recompute lag metrics and figures from existing/new
  embeddings; no model retraining is needed for this issue alone.
- **Proposed minimal fix:** call with `common_support=True` after applying each
  population's validity mask; persist the common support definition/indices.
- **Classification:** BUG FIX restoring comparable lag evaluation.
- **Before/after:** pending implementation and evidence.

### NB-RB-07 — full-sample branch is absent from the validated real runner

- **Where:** `run_natural_monkey_suite()` and `_write_run_manifest()` in
  `src/neurobridge/experiments/real_monkey_validated.py`.
- **Current behavior:** the runner only trains/transforms/evaluates
  `held_out`; its manifest explicitly says full sample was not run.
- **Expected behavior:** train a separate descriptive full-sample model from
  fresh initialization for exactly N_best updates learned from the paired
  held-out fit; no validation split/curve is fabricated.
- **Why problematic:** missing required descriptive manifold/trajectory
  analysis, distinct from held-out generalization.
- **Direct evidence:** branch is hardcoded to `held_out`; manifest says
  “not run”.
- **Scientific consequence / affected results:** full-sample embeddings, plots,
  collapse/geometry/lag descriptions are missing; held-out scores remain
  conceptually separate.
- **Severity:** MAJOR.
- **Required action:** run the additional fresh full-sample fits and downstream
  metrics after held-out N_best is known.
- **Proposed minimal fix:** create a distinct full-sample stage that reuses only
  raw/window inputs, not model/checkpoint weights.
- **Classification:** NEW PROTOCOL DECISION specified by the user.
- **Before/after:** pending implementation and evidence.

### NB-RB-08 — no explicit MATLAB lag-lag export

- **Where:** existing embedding transforms under `real_monkey.py` and
  `staged_shared_latent.py`.
- **Current behavior:** caches are NPZ-oriented; there is no per-fit MATLAB
  export contract with all alignment keys and run metadata.
- **Expected behavior:** `.mat` per setting/branch/population/model/objective/
  seed with raw/unit embeddings, trial/time/global-time IDs, behavior fields,
  validity mask, split labels, channels, and setting/population/model/objective/
  seed metadata.
- **Why problematic:** MATLAB lag-lag analysis can silently misalign flattened
  observations if row metadata is reconstructed manually.
- **Direct evidence:** active repository search found no `.m` pipeline/exporter;
  transform routines write NPZ.
- **Scientific consequence / affected results:** downstream MATLAB lag-lag
  work only; no effect on the Python fit or its metrics.
- **Severity:** MINOR.
- **Required action:** export each new embedding; no additional training.
- **Proposed minimal fix:** use `scipy.io.savemat()` after embedding generation,
  preserving row order and metadata verbatim.
- **Classification:** requested output addition, not a scientific protocol
  change.
- **Before/after:** pending implementation and round-trip verification.

### NB-RB-09 — presentation and compute bundle missing

- **Where:** experiment output assembly; current synthetic and real plotters
  write per-run figures but no unified delivery table/summary.
- **Current behavior:** no `presentation_ready/` bundle, central results table,
  compute-performance CSV, measured inference throughput, or CNN/Transformer
  compute comparison plots. Some existing histories/figures are reusable.
- **Expected behavior:** progressive output to the requested structured folder;
  compute remains separate from representation quality.
- **Why problematic:** output is not directly usable for the presentation and
  efficiency cannot be reported independently.
- **Direct evidence:** output tree was empty; source contains per-fit plotting
  but no central exporter.
- **Scientific consequence / affected results:** delivery/reporting only; does
  not alter trained representations.
- **Severity:** MAJOR for the deadline deliverable.
- **Required action:** generate tables/summary/figures from new run artifacts.
- **Proposed minimal fix:** re-use the existing raw/unit, trajectory, loss, and
  lag plotters; add one presentation assembler and save raster/vector versions.
- **Classification:** requested delivery addition, not a change to the scientific
  metric definitions.
- **Before/after:** pending implementation and evidence.

### NB-RB-10 — synthetic public-config alias breaks modular cache reuse

- **Where:** `src/neurobridge/experiments/audit_contracts.py`,
  `check_cached_config()`; called from `_stage_dir()` and synthetic stage
  cache guards in `src/neurobridge/experiments/staged_shared_latent.py`.
- **Current behavior:** the synthetic serializer writes `time_offset_bins`
  and `temporal_objective_temperature`, while the generic cache validator
  iterates dataclass keys `cebra_time_offset` and `cebra_temperature` and sees
  missing values.
- **Expected behavior:** validate the same numerical protocol values under
  the stable public names, while still rejecting actual protocol changes.
- **Why problematic:** software/cache-resume bug; this directly undermines
  stage-wise restartability. It does not change the science of a successful
  fit, but can prevent later stages from using a valid earlier cache.
- **Direct evidence:** a live comparison of `asdict(SharedLatentStageConfig)`
  with `_public_config()` reported exactly
  `[('cebra_time_offset', 10, None, 'time_offset_bins'),
  ('cebra_temperature', 1.0, None, 'temporal_objective_temperature')]`.
  Thus `check_cached_config()` would raise a cache mismatch on those two
  keys for the current public cache format.
- **Scientific consequence / affected results:** no current outputs existed;
  no trained result or metric is scientifically contaminated. Without the
  fix, synthetic window/model/embedding/metric/figure stages may fail to
  resume from their valid cache.
- **Severity:** MAJOR.
- **Required action:** code correction only; no retraining.
- **Proposed minimal fix:** add an explicit alias map in the shared cache
  validator and compare each saved public field to the corresponding
  dataclass value; retain strict mismatch errors for all other settings.
- **Classification:** BUG FIX to restore the declared modular cache behavior.
- **Before/after:** pending implementation and tests.

### NB-RB-11 — window validity validator assumes center-only support

- **Where:** `src/neurobridge/experiments/audit_contracts.py`,
  `validate_metadata()`; contrasted with `_window_subject()` in
  `src/neurobridge/experiments/staged_shared_latent.py`.
- **Current behavior:** `_window_subject()` correctly marks a window valid
  only when all samples in its temporal context are valid, but the validator
  compares `lag_valid` only to validity at the center bin.
- **Expected behavior:** the saved window mask must equal the conjunction of
  source-bin validity across the complete configured window; centered padding
  outside the recording is invalid.
- **Why problematic:** software/qualification inconsistency, not a training
  or metric flaw. It prevents the relevant contract test from passing on
  correct strict-window masks.
- **Direct evidence:** after adding the cache-alias regression test,
  `pytest -q tests/test_p0_contracts.py` reported 1 failed / 5 passed. The
  failure was `ValueError: lag_valid disagrees with generative metadata` on
  the existing 3-bin-window test fixture with `valid_A = ids % 3 != 0`.
- **Scientific consequence / affected results:** no fitted model, embedding,
  metric, or figure; only the metadata qualification check is affected.
- **Severity:** MINOR.
- **Required action:** correct the validity validator and test its strict
  boundary behavior; no retraining.
- **Proposed minimal fix:** compute the expected per-center mask as `all()`
  over the configured window offsets, treating out-of-range padded bins as
  invalid, then compare it to `lag_valid`.
- **Classification:** BUG FIX aligning the executable contract to the
  intended trial-safe strict-window definition.
- **Before/after:** pending implementation and evidence.

### NB-RB-12 — validated monkey checkpoint cannot enter legacy transform cache path

- **Where:** `fit_validated_neural_model()` in
  `src/neurobridge/experiments/real_monkey_validated.py`; then
  `transform_neural_embeddings()` and `fit_neural_model()` in
  `src/neurobridge/experiments/real_monkey.py`.
- **Current behavior:** the validated runner writes the fit under
  `stage03_models/<branch>/<model>_<objective>/model.pt` and serializes a
  top-level config containing public protocol fields, schedule, branch, model,
  and objective. The legacy transformer calls `fit_neural_model()` before
  encoding; that fitter uses the same directory but expects `config.json` to
  contain a nested `config` object equal to the older schema.
- **Expected behavior:** transform the already selected checkpoint returned
  by the validated fit, without launching a second fitter or changing the
  validation-selected model.
- **Why problematic:** software integration/cache-schema bug. It blocks the
  intended fit -> embedding stage and defeats resumption from a completed
  checkpoint; it does not make an otherwise completed fit scientifically
  invalid.
- **Direct evidence:** validated cache writer stores `"config":
  legacy._public_config(config)` plus `"schedule"`, `"objective"`, etc.; the
  legacy writer stores `{"config": ..., "branch": ..., "model": ..., 
  "loss": ...}` and its cache check compares `cached.get("config", {})` with
  the entire serialized legacy config. The call chain in
  `run_natural_monkey_suite()` passes the validated fit directly to the legacy
  transform, which calls the incompatible fitter on the same model path.
- **Scientific consequence / affected results:** without correction, the
  paired real run cannot reliably produce embeddings or downstream metrics.
  There are no current outputs; a completed validated checkpoint can be
  preserved and reused.
- **Severity:** CRITICAL.
- **Required action:** code correction; resume from the embedding stage after
  the fix, without retraining completed fits.
- **Proposed minimal fix:** add an explicit optional checkpoint input to the
  transform function; when supplied, skip the legacy fitter and validate the
  checkpoint/config metadata against the requested model, objective, branch,
  and run config. Record the checkpoint identity in embedding metadata.
- **Classification:** BUG FIX restoring the staged validated workflow.
- **Before/after:** pending implementation and evidence.

### NB-RB-13 — real progress manifest is not resumable and branch denominator is wrong

- **Where:** `run_natural_monkey_suite()` in
  `src/neurobridge/experiments/real_monkey_validated.py`.
- **Current behavior:** a compatible old manifest is read for validation,
  then `completed` is reset to an empty list. Each model/objective appends one
  held-out and one full-sample record, but progress divides by only
  `len(models) * len(objectives)`.
- **Expected behavior:** retain valid completed units for the same protocol
  and use a denominator that counts both trained branches.
- **Why problematic:** logging/resume bookkeeping defect only. Stage caches
  still prevent completed checkpoints from being retrained, but the displayed
  completion can restart at zero or exceed the total.
- **Direct evidence:** source control flow at suite initialization and the two
  `completed.append()` calls; the loop's `total_fits` expression is half of
  the actual branch-fit count.
- **Scientific consequence / affected results:** none; only manifest and
  progress interpretation.
- **Severity:** MINOR.
- **Required action:** code/reporting correction; no model or metric rerun.
- **Proposed minimal fix:** restore/deduplicate `completed_fits` from the
  compatible manifest and set expected work units to two per model/objective.
- **Classification:** BUG FIX to the resume ledger, not a protocol change.
- **Before/after:** pending implementation and evidence.

### NB-RB-14 — plot cache does not include rendering configuration

- **Where:** `plot_neural_embeddings()` and `_plot_shared_embedding_views()` in
  `src/neurobridge/experiments/real_monkey.py`.
- **Current behavior:** if the main PNG exists, the function returns it
  without checking the requested plot configuration, while regenerating the
  companion figures. This can mix old and new rendering parameters in one
  apparent figure bundle.
- **Expected behavior:** cache/reuse the whole figure bundle only when plot
  settings and all parent artifact hashes match; otherwise make an explicit
  plot-only regeneration without touching fit or embedding artifacts.
- **Why problematic:** logging/visualization cache defect only; it does not
  change scientific metrics or representation quality.
- **Direct evidence:** branch guarded only by `output.exists()` and
  `not force`; no DPI or parent-artifact metadata was previously available.
- **Scientific consequence / affected results:** figures only. Checkpoints,
  embeddings, and metrics are unaffected.
- **Severity:** MINOR.
- **Required action:** plot-only cache correction and regeneration.
- **Proposed minimal fix:** record plotting parameters and parent hashes in a
  figure provenance sidecar; permit cache reuse only for a matching sidecar.
- **Classification:** BUG FIX; no scientific protocol change.
- **Before/after:** pending implementation and evidence.

### NB-RB-15 — phase-specific runs cannot merge into one resumable manifest

- **Where:** `_load_or_create_manifest()` in
  `src/neurobridge/experiments/presentation_rebuild.py`.
- **Current behavior:** a synthetic-only invocation writes a null channel
  partition hash, while a later real-only invocation computes the canonical
  hash and fails whole-dictionary equality.
- **Expected behavior:** phase selection is independent; later phases extend
  the same run manifest, while a non-null canonical partition hash must remain
  identical once recorded.
- **Why problematic:** resume-manifest software bug only; it does not affect
  training or scientific outcomes.
- **Direct evidence:** `expected_protocol["channel_partition_sha256"]` depends
  on the current phase, then the saved and expected protocol dictionaries are
  compared for exact equality.
- **Scientific consequence / affected results:** no model or metric; only
  phase progress tracking and restartability.
- **Severity:** MAJOR.
- **Required action:** code correction and a two-phase manifest test.
- **Proposed minimal fix:** compare invariant protocol fields, reject
  conflicting non-null partition hashes, and fill a previously null hash when
  the first real phase starts.
- **Classification:** BUG FIX; no protocol or model setting changes.
- **Before/after:** pending implementation and evidence.

### NB-RB-16 — full-sample fit cache omits its update count

- **Where:** `fit_validated_neural_model()` in
  `src/neurobridge/experiments/real_monkey_validated.py`, its full-sample cache
  comparison around line 626 and config serialization around line 921.
- **Current behavior:** a completed full-sample fit is written without the
  `full_sample_updates` field in `config.json`, but a later cache check
  requires that field to equal the requested update count. Since
  `cached.get("full_sample_updates")` returns `None`, it rejects the valid
  cache and raises `FileExistsError`.
- **Expected behavior:** persist the update count used by the fit, then reuse
  the full-sample checkpoint when all protocol fields including that count
  match.
- **Why problematic:** checkpoint-resume software bug. It does not alter the
  optimization itself, but it prevents modular restart after the full-sample
  fit and can block all subsequent stages.
- **Direct evidence:** run
  `resume_qualification_20260923T181133_351249Z` successfully prepared
  raw/windows, reused the held-out checkpoint after simulated interruption,
  reused held-out embeddings, and verified selective invalidation. It then
  failed with `FileExistsError: cached validated fit uses a different
  protocol` at line 629. The full-sample config JSON has `branch=full_sample`
  and schedule `max_steps=1`, but no `full_sample_updates`; the writer at line
  921 likewise omits that field while the cache check requires it.
- **Scientific consequence / affected results:** existing full-sample
  checkpoint and embedding remain scientifically unchanged; only cache reuse
  and downstream resume are blocked. No V2 scientific outputs exist yet.
- **Severity:** MAJOR.
- **Required action:** code correction to persist this already-used metadata,
  then rerun the qualification under a new run ID. No scientific retraining.
- **Proposed minimal fix:** add `"full_sample_updates": full_sample_updates`
  to the serialized fit config. Keep strict equality checking.
- **Classification:** BUG FIX restoring intended stage-resume behavior; no
  protocol change.
- **Before/after:** pending harness correction and passing rerun evidence.

### NB-RB-17 — synthetic direction-averaged plot drops groups on reshape errors

- **Where:** `plot_direction_averaged_embedding()` in
  `src/neurobridge/viz/manifold_plots.py`, around the trial-average reshape;
  called by the synthetic stage-6 raw and unit-sphere figure generators.
- **Current behavior:** the helper prints a reshape error and continues to the
  next label when the selected embedding rows do not divide into the
  hard-coded expected points-per-trial shape.
- **Expected behavior:** group rows using the saved trial/time/condition
  support actually associated with the selected embedding rows, and either
  draw each group or explicitly report why a group is excluded.
- **Why problematic:** plotting/aggregation implementation issue only, unless
  evidence later shows this helper feeds a scientific metric (none found so
  far). It does not alter model fitting or the stored embeddings.
- **Direct evidence:** the active synthetic run printed
  `Errore di reshape per label 1: cannot reshape array of size 2700 into shape
  (200,3)` and analogous errors for labels 2–8; another subset printed
  `cannot reshape array of size 2550 into shape (190,3)`. The exception is
  caught and the function `continue`s, so affected traces are skipped.
- **Scientific consequence / affected results:** possible missing
  direction-averaged traces in raw and unit-sphere figures only. No
  checkpoint, embedding, decoding, geometry, lag, or robustness metric is
  invalidated by this call path.
- **Severity:** MINOR.
- **Required action:** after V2 training, make a plot-only correction and
  regenerate affected figures; do not retrain.
- **Proposed minimal fix:** use per-trial/time metadata to select complete
  trajectory segments and validate their support before averaging, instead of
  reshaping by the current hard-coded trial-length assumption.
- **Classification:** BUG FIX for plotting; no scientific protocol change.
- **Before/after:** pending figure-only correction and visual verification.

### NB-RB-18 — no full 65-channel natural-real analysis branch

- **Where:** `run_seed42_rebuild()` in
  `src/neurobridge/experiments/presentation_rebuild.py`, lines 155–186.
- **Current behavior:** the real-data loop only constructs population A from
  `A_channel_indices` and population B from `B_channel_indices`; there is no
  configuration using all source channels.
- **Expected behavior:** add a separately labeled `REAL_TOTAL_65` run using
  all 65 recorded channels, while retaining A/B as distinct pseudopopulation
  analyses.
- **Why problematic:** experiment-design omission; current “full_sample” is
  the all-trials branch, not a full-channel population. This is not a defect
  in the completed A/B fits.
- **Direct evidence:** the only population keys passed to
  `canonical_config()` are `A_channel_indices` and `B_channel_indices`;
  `_load_source()` defaults to all channels only when `channel_indices=None`,
  which the current runner never passes.
- **Scientific consequence / affected results:** no full-population neural
  representation results currently exist. Synthetic and A/B checkpoints,
  embeddings, metrics, and figures are unaffected.
- **Severity:** MAJOR.
- **Required action:** add a new, resumable full-65 run branch; do not
  overwrite or invalidate existing artifacts.
- **Proposed minimal fix:** pass all channel indices to a distinct
  `REAL_TOTAL_65` configuration and output root/run ID, preserving the same
  architectures, objectives, schedule, metrics, and split rules.
- **Classification:** implementation of the explicitly requested design;
  no change to frozen model or optimization protocol.
- **Before/after:** pending implementation and qualification evidence.

### NB-RB-19 — only seed 42 is orchestrated

- **Where:** `SEED` and `RUN_STAMP` in
  `src/neurobridge/experiments/presentation_rebuild.py`, lines 27–28; seed
  assignment for synthetic at lines 137–142 and real at lines 167–175.
- **Current behavior:** the entry point exposes only seed 42, and synthetic
  `seed`, `split_seed`, and `training_seed` are all set to 42. Real configs
  likewise use the single canonical seed.
- **Expected behavior:** complete the frozen training-seed set `{42, 123,
  456}` for Synthetic and real populations. For Synthetic, keep the same
  generated latent/data and vary only the training seed, as clarified by the
  user; do not conflate data-generation variation with training variability.
- **Why problematic:** incomplete replication/statistical coverage, not a
  failure of the already completed seed-42 run.
- **Direct evidence:** literal `SEED = 42`; the runner creates a single
  synthetic run ID and passes that same seed to generation, split, and
  training; the real loop has no seed iteration.
- **Scientific consequence / affected results:** seed-42 outputs remain
  valid as single-seed results, but three-seed means/SD and cross-seed
  consistency are unavailable until the missing runs are completed.
- **Severity:** MAJOR.
- **Required action:** add separate resumable run IDs/output roots for seeds
  123 and 456; do not rerun or overwrite seed 42.
- **Proposed minimal fix:** make training seed explicit in runner
  configuration and run identity, keep Synthetic data/split seed fixed for
  identical latent/data, and preserve all other frozen settings.
- **Classification:** completion of the already specified seed protocol;
  no change to the frozen seed set or other protocol settings.
- **Before/after:** pending implementation and qualification evidence.

### NB-RB-20 — REAL-10 shifts both populations and independently fits models

- **Where:** setting/population loop in `run_seed42_rebuild()` in
  `src/neurobridge/experiments/presentation_rebuild.py`, lines 155–185;
  shift application in `_load_source()` in
  `src/neurobridge/experiments/real_monkey.py`, lines 330–345.
- **Current behavior:** for `real_10`, `shift` is 10 and the same
  `imposed_shift_bins=shift` is passed to both A and B. The runner then calls
  `run_natural_monkey_suite()` independently for each population/setting,
  which fits models for the shifted setting rather than evaluating the R0
  model on a controlled perturbation.
- **Expected behavior:** retain the exact R0 models, leave A unchanged, shift
  only B by +10 bins without wrap, propagate strict validity masks, and
  evaluate embeddings/lag using those same R0 checkpoints.
- **Why problematic:** synthetic-versus-monkey/control-condition protocol
  inconsistency and confounding of imposed shift with refitting. These
  outputs cannot isolate the effect of a B-only digital delay.
- **Direct evidence:** runner passes shift 10 in both population configs;
  `_load_source()` shifts whichever selected channel population is supplied;
  active progress logs show optimizer steps for `real_10_A`, proving that
  this branch trains independently.
- **Scientific consequence / affected results:** the current REAL-10 model,
  embedding, metric, and figure artifacts are not admissible as controlled
  R0-to-R10 lag-recovery evidence. Existing R0 and Synthetic results are
  unaffected. Preserve current REAL-10 artifacts as distinct historical
  outputs; do not silently relabel them as the controlled perturbation.
- **Severity:** MAJOR for the controlled-lag claim.
- **Required action:** let the already active job finish and preserve it;
  create a distinct REAL-10 evaluation branch derived from R0 checkpoints,
  with no new fitting.
- **Proposed minimal fix:** reuse each R0 A/B checkpoint; transform A on
  unchanged data; create shifted B inputs and validity metadata, transform
  with the corresponding R0 B checkpoint, then evaluate on common valid
  trial/time support.
- **Classification:** correction to the explicitly specified controlled
  perturbation; no architecture, objective, optimizer, metric, seed, or
  qualification change.
- **Before/after:** pending implementation and qualification evidence.

### NB-RB-21 — first controlled-lag helper draft mis-keyed shifted pairs

- **Where:** `_lag_curves_on_joint_support()` in the new
  `src/neurobridge/experiments/real_monkey_lag_eval.py`, initial draft before
  execution.
- **Current behavior:** the draft supplied B coordinates already offset by
  the candidate lag, then called `lagged_alignment_by_trial_time()` with lag
  zero. The helper would look for `(trial, reference_time)` in B rather than
  `(trial, reference_time + candidate_lag)`, so non-zero pairs would be
  unmatched.
- **Expected behavior:** use the existing metadata-aware lag helper with its
  actual candidate lag range, after restricting A reference rows to the
  intersection valid for every lag in both R0 and R10.
- **Why problematic:** downstream lag-evaluation software bug. It would
  produce missing/invalid candidate matches, not a model-training problem.
- **Direct evidence:** inspected call passes `time_id_other = time + lag`
  alongside `lags=(0,)`; the helper constructs its lookup from B time IDs
  then searches using `reference_time + 0`.
- **Scientific consequence / affected results:** no run artifacts or
  scientific outputs; the new helper had not yet been executed. If left,
  controlled REAL-0/REAL-10 lag curves could be invalid.
- **Severity:** MAJOR for the new lag evaluator only.
- **Required action:** code correction and focused synthetic lag test; no
  retraining or recomputation of existing results.
- **Proposed minimal fix:** pass candidate `LAGS` directly to the existing
  alignment function using the shared valid reference rows and original B
  trial/time coordinates.
- **Classification:** BUG FIX in new downstream evaluation; no protocol or
  metric-definition change.
- **Before/after:** `offset B coordinates + request lag zero` -> pass original
  B coordinates and the frozen candidate lag list after intersecting valid
  support -> candidate pair cardinality is exactly equal at all 41 lags.
- **Resolution evidence:** `tests/test_real_monkey_lag_eval.py` verifies
  synthetic R0 best lag 0 and B-only R10 best lag +10; the focused suite
  passed.

### NB-RB-22 â€” boundary-saturated seed-42 real lag scan

- **Where:** frozen candidate set `LAGS` in
  `src/neurobridge/experiments/lag_shuffle.py` (`-20` through `+20`) and
  summary generation in `run_controlled_lag_evaluation()` in
  `src/neurobridge/experiments/real_monkey_lag_eval.py`.
- **Current behavior:** the scan returns a best candidate at a range endpoint
  for most REAL-0 and every REAL-10 seed-42 configuration.
- **Expected behavior:** the best candidate is interior to the scan for a
  resolved lag estimate; an endpoint result is flagged as censored and not
  treated as a resolved estimate.
- **Why problematic:** scientific/statistical limitation, not a software
  shift defect. The frozen scan cannot localize optima beyond its endpoints.
- **Direct evidence:** across 16 architecture/objective/branch combinations,
  15/16 REAL-0 peaks and 16/16 REAL-10 peaks are at `-20` or `+20`; observed
  `delta_l_hat` values are only 0 or 3, with none equal to the imposed +10.
  Separately, CNN/soft shifted B unit embeddings satisfy
  `B_R10(trial,t+10) = B_R0(trial,t)` within maximum absolute error
  `8.94e-8` over 110,010 matched valid rows, confirming the input transform.
- **Scientific consequence / affected results:** seed-42 lag recovery is
  non-conclusive under the frozen scan. This does not invalidate embeddings,
  decoding, representational geometry, or the digital perturbation itself.
- **Severity:** MAJOR for interpretation of controlled lag only.
- **Required action:** no correction under the explicitly frozen metric; keep
  boundary flags and report non-conclusive estimates. Extending the candidate
  range is a new protocol decision and is not authorized here.
- **Proposed minimal fix:** none within this protocol; an approved future
  study could pre-specify a wider candidate-lag range.
- **Classification:** scientific limitation; no BUG FIX and no new protocol
  decision has been made.
- **Before/after:** not corrected; the existing output is retained and
  explicitly interpreted as boundary-censored.

### NB-RB-23 â€” lag metric rows need explicit config-hash linkage

- **Where:** `lag_curves_r0_vs_r10.csv`, `lag_recovery_summary.csv`, and
  `lag_recovery_provenance.json` written by
  `run_controlled_lag_evaluation()` in
  `src/neurobridge/experiments/real_monkey_lag_eval.py`.
- **Current behavior:** CSV rows identify seed/model/objective/branch and
  setting, while the JSON sidecar contains a flat mapping of parent hashes.
  It does not explicitly map each metric row to both R0 population configs
  and the shifted R10 config hashes.
- **Expected behavior:** every lag metric record must be traceable to the
  exact config, split, channel partition, embedding, and checkpoint parents.
- **Why problematic:** logging/provenance incompleteness only; it does not
  change metric values, selected lag, fit, or embedding.
- **Direct evidence:** current CSV headers omit `config_sha256`; the existing
  provenance JSON `parent_artifacts` has artifact hashes but no per-row
  config-hash tuple. Individual shifted-embedding metadata does have its
  config hash.
- **Scientific consequence / affected results:** current seed-42 lag values
  remain numerically valid, but row-level config traceability is incomplete.
  No checkpoint, embedding, decoder, geometry metric, or figure values are
  affected.
- **Severity:** MINOR.
- **Required action:** provenance-only code correction; no metric
  recomputation or retraining.
- **Proposed minimal fix:** write an additional versioned sidecar mapping
  each lag row to config hashes and parent checkpoint/window/embedding hashes;
  leave the already-generated CSVs and figures unchanged.
- **Classification:** logging/provenance BUG FIX; no scientific protocol
  decision.
- **Before/after:** pending sidecar generation and hash validation.

### NB-RB-24 â€” corrected V2 progress subphases can remain running

- **Where:** `_ensure_real_population()` and `run_corrected_v2_rebuild()` in
  `src/neurobridge/experiments/presentation_rebuild.py`.
- **Current behavior:** individual run records are marked complete, but
  population subphase names (`real_total_65`, `real_A`, `real_B`) are set to
  `running` and are not closed when their seed loop finishes.
- **Expected behavior:** close each requested subphase as complete only after
  all of its requested run IDs are complete.
- **Why problematic:** manifest/status reporting only; no model or metric
  behavior is changed.
- **Direct evidence:** `_record_corrected_run()` writes the per-run complete
  record while the top-level runner only assigns `phase_status[phase]` at
  entry/finalization.
- **Scientific consequence / affected results:** corrected V2 manifest status
  only; all artifact hashes and scientific outputs remain unchanged.
- **Severity:** MINOR.
- **Required action:** update and validate the progress manifest after the
  active pipeline completes; no recomputation or retraining.
- **Proposed minimal fix:** set the relevant subphase to complete after each
  seed loop and preserve every completed run entry.
- **Classification:** reporting BUG FIX only.
- **Before/after:** `all=complete` and all 14 run records were complete while
  six requested subphases remained `running` -> updated only those six
  `phase_status` fields after exit 0 -> all seven phase statuses and all 14
  run records now read `complete`; no scientific artifact was modified.

### NB-RB-25 — real full-sample metric evaluation was not orchestrated

- **Where:** `src/neurobridge/experiments/real_monkey_validated.py`,
  `run_natural_monkey_suite()` (pre-correction lines 1539-1650), and the
  branch-aware evaluation functions in `src/neurobridge/experiments/real_monkey.py`.
- **Current behavior:** the suite fits/transforms PCA and neural models for
  both `held_out` and `full_sample`, but calls `evaluate_pca()`,
  `evaluate_neural()`, `compute_trial_bootstrap_decoding()`, and
  `compute_representation_geometry()` only for `held_out`. The
  `stage05_metrics/full_sample/` directory and its metric files are absent
  for real seed42 total-65/A/B runs.
- **Expected behavior:** evaluate both frozen branches wherever the existing
  metric is scientifically applicable. `held_out` remains the primary
  representation-generalization evidence. `full_sample` metrics are clearly
  marked descriptive/in-sample; full-sample global geometry diagnostics use
  all available trials. Do not introduce metric definitions or refit models.
- **Why scientifically problematic:** this is an evaluation-orchestration
  omission (software/protocol-output bug), not a training or metric-formula
  failure. It leaves the requested in-sample descriptive comparison missing
  and can make reports appear to contain only held-out results.
- **Direct evidence:** seed42 real total-65 and existing A/B each contain 18
  CSV files under `stage05_metrics/held_out` and none under
  `stage05_metrics/full_sample`; cached `stage04_embeddings/full_sample/`
  artifacts and full-sample checkpoints exist. Synthetic seed123 has 18 CSVs
  in each of `held_out` and `full_sample`. Pre-correction source hash for
  `real_monkey_validated.py` is
  `B4E881AE0127DC746840FE9308D909D307BD7E325F62B095DBBD17D4410CDE31`;
  HEAD before this correction is
  `9efbbb5fbe73404d790b4052011e292fdc09ea07` (workspace already dirty).
- **Scientific consequence / affected results:** no full-sample monkey
  descriptive metrics exist for these runs yet. Existing held-out metrics,
  trained checkpoints, embeddings, and full-sample figures are not invalidated.
  No full-sample decoding score may be presented as generalization evidence.
- **Severity:** MAJOR (missing requested analysis branch; not a corrupted
  trained model).
- **Required action:** code correction and evaluation-only recomputation from
  cached full-sample embeddings/checkpoints. No refit or retraining.
- **Proposed minimal fix:** invoke the existing branch-aware evaluation
  stages on cached full-sample artifacts and add explicit
  `descriptive_in_sample_representation` scope metadata; use all valid
  full-sample trial prototypes for existing global geometry metrics. Preserve
  held-out files and formulas.
- **Classification:** BUG FIX restoring the frozen two-branch output protocol;
  no new protocol decision or metric.
- **Before/after:** real run had both `held_out` and `full_sample` fits and
  embeddings but only held-out metric calls -> evaluate-only stage now consumes
  cached full-sample embeddings and checkpoints -> each seed42 real total-65/A/B
  root has 18 CSVs in `stage05_metrics/full_sample/`; global CKA/participation
  ratio use `all_windows`, and RSA geometry uses all 193 trials. Decoding rows
  carry `descriptive_in_sample_representation` and retain `evaluation_split=test`
  as a descriptive probe, not generalization evidence. All 156 full-sample
  model-stage file hashes across the three roots (including checkpoints) were
  unchanged. Focused suite:
  12 tests passed; `py_compile` passed.
- **Source provenance:** HEAD remained
  `9efbbb5fbe73404d790b4052011e292fdc09ea07` (no commit created). Before hashes:
  `real_monkey.py` `A0B45B23381974283449B29C2A09256E07D1B6B41CB47B629033EC9C6D86EA02`,
  `real_monkey_validated.py` `B4E881AE0127DC746840FE9308D909D307BD7E325F62B095DBBD17D4410CDE31`,
  `presentation_rebuild.py` `79FA63C37B86BB6A6F73F57D151D1800817E405FF673024089BCD7A1DF4F1E24`;
  after hashes: `real_monkey.py`
  `2492A299C9767AB1BA20290DDBB8006E54E730C07255A7064C4FD3132DEA3328`,
  `real_monkey_validated.py`
  `8D4A2542E83CE7AAC3869F77F81E2E29EB3B5414689E96861D47694FF4DDE25E`,
  `presentation_rebuild.py`
  `AE7ABA052E8D61818D1AD456B1CA5F1A9B0971F650132D9335B464CCF1723652`.

### NB-RB-26 — held-out geometry CSVs omit branch/scope columns

- **Where:** `stage05_metrics/held_out/representation_geometry.csv` in the
  completed real run roots; generated by the prior geometry-evaluation path.
- **Current behavior:** geometry rows contain RSA values, `n_test_trials`, and
  their bootstrap intervals, but omit explicit `branch`, `evaluation_scope`,
  and the common `n_evaluation_trials` field. Neighboring held-out metric CSVs
  do carry `branch=held_out`.
- **Expected behavior:** all aggregate rows should preserve branch and
  evaluation scope, while identifying geometry trial counts consistently.
- **Why problematic:** logging/reporting inconsistency only; it prevents a
  uniform downstream aggregation and could make RSA scope ambiguous when
  detached from its directory. Metric values, trial support, and model
  artifacts are not changed.
- **Direct evidence:** for example,
  `clean_rebuild_2026-09-23_seed42_real_0_A/stage05_metrics/held_out/representation_geometry.csv`
  starts with columns `model,objective,representation,category,metric,...`
  (no branch/scope columns) and records `n_test_trials=39`; the sibling
  `cnn1d_soft.csv` records `branch=held_out`.
- **Scientific consequence / affected results:** RSA/behavioral geometry
  estimates remain numerically valid; only branch-aware reporting/aggregation
  is affected. No training, embeddings, decoding, CKA, robustness, or lag
  outputs are invalidated.
- **Severity:** MINOR / REPORTING ONLY.
- **Required action:** code correction in the downstream aggregator only; no
  recomputation of metrics, embeddings, or fits.
- **Proposed minimal fix:** when reading the existing geometry CSV, take the
  branch from its explicit `stage05_metrics/<branch>/` parent directory and
  map `n_test_trials` to the aggregate table's count field. Preserve the
  original metric files unchanged.
- **Classification:** BUG FIX to downstream reporting; no protocol decision
  and no metric-definition change.
- **Before/after:** held-out RSA rows had trial count but no row-level branch
  scope -> aggregate uses the artifact's held-out directory and existing test
  trial count -> output table carries `branch=held_out`,
  `evaluation_scope=held_out_representation_generalization`, and consistent trial-count
  metadata without rewriting Stage-5 sources.
- **Resolution evidence:** `tools/aggregate_real_branch_metrics.py` completed
  against all 9 real baseline runs and emitted 6,876 point-metric rows in
  `outputs/presentation_ready/tables/REAL_BRANCH_METRICS.csv`; each of the 18
  run/branch combinations had the expected 18 Stage-5 CSVs. Source Stage-5 CSVs
  were not rewritten.

## Corrections and run provenance

### Correction record — corrected real-data design (implementation in progress)

| Issue | Before | Correction | Current evidence / affected artifacts |
|---|---|---|---|
| NB-RB-18 | Runner generated only A/B (32/33 channels). | Added a separate `total_65` config with `channel_indices=0..64` and unique run roots. | Config test confirms 65 indices; raw/windows cache prepared with shape `(115800,21,65)` and 111,940 strictly valid windows. Full-65 model training has not started yet. Existing Synthetic/A/B outputs remain read-only. |
| NB-RB-19 | Only training seed 42 was orchestrated. | Added explicit `{42,123,456}` iteration; synthetic data seed and split stay fixed at 42 while only `training_seed` varies. | Config test confirms fixed synthetic data/split seeds and training seeds `(42,123,456)`. Seed-42 existing runs remain unchanged; missing seed fits have not started. |
| NB-RB-20 | REAL-10 shifted both A and B and fit new models. | Added a distinct transform-only R10 branch: B input shifts +10 bins without wrap, A remains the R0 embedding, R0 B checkpoint is reused, and lag curves share strict trial/time support. | Completed 16 model/objective/branch transforms with no fit; trial/time/behavior/split metadata matched. Existing independent REAL-10 fits are preserved but excluded from controlled-lag claims. |
| NB-RB-21 | First downstream draft mismatched offset B coordinates with zero candidate lag. | Corrected to use original B coordinates with the existing metadata-aware lag helper over all candidate lags. | 15 focused/regression tests passed; no result from the faulty draft was created. |
| NB-RB-22 | Frozen scan saturates at its endpoints on seed-42 real embeddings. | No metric change; report boundary censoring and no resolved +10 recovery conclusion. | 15/16 R0 and 16/16 R10 maxima are boundary peaks; `delta_l_hat` only 0 or 3. Shift equivalence passed for CNN/soft; no retraining is indicated. |
| NB-RB-25 | Real Stage-5 wrote only `held_out` metrics despite saved full-sample fits/embeddings. | Added cache-only branch-aware full-sample metric evaluation; no metric definitions or training settings changed. | All 9 real baseline runs (seeds 42/123/456 × total_65/A/B) now have 18 Stage-5 CSVs in each branch; RSA uses 193 full-sample trials. The main process exited 0; evaluation-only completion created 18 files each for seed456 A/B and reused already-complete files elsewhere. No fit was repeated; original model-stage/checkpoint hashes remained unchanged in the earlier seed42 comparison. |
| NB-RB-26 | Legacy held-out geometry rows omitted row-level branch/scope and used a different trial-count column. | Reporting-only aggregation uses the `held_out` parent branch, frozen held-out geometry scope, test split, and existing `n_test_trials`; source metric CSVs are unchanged. | All 9 real baseline runs and both branches appear in `REAL_BRANCH_METRICS.csv` (6,876 rows); 18/18 Stage-5 CSVs were found for each run/branch. |

No frozen model, objective, optimizer, training schedule, metric, channel
partition, training seed set, or qualification threshold was changed. The
three new run roots are separate from the already-complete Synthetic/R0 A/B
roots. Controlled-lag R10 creates embeddings/metrics only and performs no fit.
The active independent REAL-10 A/B fits were not interrupted and will be
preserved as historical, non-controlled outputs.

Each correction will append its actual code diff/evidence here. New run IDs
will be versioned under `outputs/runs/`; the old identifiers
`output_2026-08-27_comparison_2000`,
`real_monkey_area2_validated_v1_2026-09-23_5000`, and
`real_monkey_area2_active_staged_2026-09-23_8000` are historical source/default
labels only. Their outputs are absent and will not be overwritten or reused.

## Phase-2A Synthetic execution issue (2026-09-28; recorded before correction)

| ID | Severity | Component | What was wrong | Evidence | Scientific impact | Fix | Existing results affected | Rerun required? |
|---|---|---|---|---|---|---|---|---|
| NB-P2A-01 | MAJOR | Synthetic HPO checkpoint output path | A long immutable trial ID makes the checkpoint path 262 characters on Windows; `torch.save` could not open it. This is an artifact-path/software issue, not a scientific protocol change. | `phase2a_safe_fit.py::_fit_new`, `torch.save` at line 352; trial `...synthetic-A-transformer-time_contrastive_blocks-c0-s1101-ee7828b9fbc31683` has `FAILED_RUNTIME`, error `[enforce fail at inline_container.cc:745] . open file failed with error code: 3`; its checkpoint path is 262 characters, whereas a successful Transformer/InfoNCE checkpoint path is 246. | That one fitted configuration lacks a saved checkpoint/validation diagnostic and cannot enter candidate ranking as a valid result. No completed fit, embedding, metric, or V2 artifact is changed. | Planned: preserve the failed trial; run an immutable new attempt under a shorter output/trial ID, with identical config/data/seed/source hashes and explicit old-to-new mapping. Test checkpoint/diagnostic persistence before treating it as resolved. | Failed trial only; completed 40 Synthetic slots remain valid. | One configuration needs a new fit because no checkpoint was saved; other long-path yet-unstarted slots should use short IDs proactively. |

The subsequent in-flight Synthetic fit was intentionally interrupted when this
issue was identified, before it produced `result.json`; its immutable partial
record is preserved and must also be mapped to a new attempt. No Real HPO fit
had started. This is a technical artifact-path correction only: no frozen
architecture, objective, temperature, optimizer, seed, sampling, split, or
early-stopping rule is changed.

**NB-P2A-01 correction and before/after evidence.**

```diff
old: planned checkpoint path length 262; PyTorch save raised RuntimeError error code 3;
     original result is FAILED_RUNTIME and has no checkpoint
-> correction: preserve that result; map the identical configuration to new
     immutable trial ID p2a-syn-tt-A-c0-s1101-ee7828b9fbc31683 with a
     195-character checkpoint path
-> new: result status ELIGIBLE at 2000 updates; best_validation_checkpoint.pt
     exists and the result records artifact hashes
```

The mapping and prior-result hashes are in
`outputs/phase2a_hpo/studies/phase2a_real_windows_2026-09-28/synthetic_only_short_path_manifest.json`.
The original `FAILED_RUNTIME` result was not overwritten. This correction
affects output naming only; the unchanged configuration hash is
`ee7828b9fbc31683...` as encoded in both IDs and the immutable trial records.

## Final Real one-window amendment (2026-09-29; declared before final training)

| ID | Severity | Component | What was wrong | Evidence | Scientific impact | Fix / classification | Existing results affected | Rerun required? |
|---|---|---|---|---|---|---|---|---|
| NB-FT-01 | MAJOR protocol decision | Real HPO selection | Original Phase-2A finalist procedure would use seeds 1101/1201/1301; user instead explicitly closed HPO after seed-1101 search and paired extension. This is not a software bug. | 384 initial and 108 extension result records verified; extension: 90 eligible, 18 near-collapse, 0 failed; no finalist fit exists. | Hyperparameter selection robustness across training seeds is unmeasured. It does not invalidate existing fits or constitute test leakage. | NEW PROTOCOL DECISION: select the all-three-population eligible lowest mean validation-loss candidate in each window/architecture/objective cell using seed 1101; no finalist HPO fits. Rule sealed before cross-window selection in `outputs/final_thesis_v1/freeze/REAL_SELECTION_RULE_v1.json`. | HPO initial/extension retained; prospective final model specification changes. | No HPO rerun. |
| NB-FT-02 | MAJOR scientific limitation | Real cross-window validation ranking | The frozen Real validation loader excludes invalid centered-window rows; changing window size also changes temporal support. | `phase2a_safe_fit.py::_loaders` selects `lag_valid` rows for Real; safe bundle manifests show valid validation centers 11,600 (21), 11,200 (41), 9,600 (121), 8,000 (201), despite the same validation trial IDs. | Selected W* is an operational choice under the approved rank rule, not isolated evidence that its input span is intrinsically superior. Per-window HPO checkpoint validity is unaffected. | NEW PROTOCOL DECISION, not bug fix: use the already stored validation losses and rank only within each architecture/objective; document unequal support. No silent common-support re-evaluation. | W* interpretation and resulting final-72 design; no prior checkpoint, embedding, metric or test result invalidated. | No retraining; a different scientific question would require a separately approved validation estimator. |
| NB-FT-03 | MAJOR provenance/software gate | New final fit adapter `tools/final_thesis_real_fit.py::_check_identity` | Initial draft checked the final spec file hash and safe output path but did not bind the requested trial to one of the spec's authorized final slots. | Read-only code review before any final fit; a same-root arbitrary trial could satisfy the old checks. | Could admit an unplanned configuration into the final output namespace; no existing scientific result was produced by this draft. | BUG FIX pending: require exact slot membership and matching config/path/seed/population/window before safe-bundle loading. | New adapter only; 492 HPO fits unaffected. | Code correction/test only; zero retraining. |
| NB-FT-04 | MAJOR software gate | `tools/final_thesis_run_real.py::run`, smoke result and full-run gate | The new runner checks `optimizer_updates == 1` after a one-update smoke. The unchanged successful `_fit_new` result records `stopping_update`, not `optimizer_updates`; `optimizer_updates` occurs only as `None` in failure records. | `src/neurobridge/experiments/phase2a_safe_fit.py` success result at lines 386-399 versus failure result at lines 220-231; review before any final fit. A successful smoke would be falsely rejected. | Prevents the 48 new final fits from starting; does not change optimization, HPO selection, or any existing scientific result. | BUG FIX pending: gate and display the recorded `stopping_update == 1`, with success status and artifact checks; focused test. | New runner only; 492 HPO fits and 24 reusable checkpoints unaffected. | Code correction/test only; zero retraining. |
| NB-FT-05 | MINOR reporting/resume | `tools/final_thesis_run_real.py::run`, rolling final-fit table | On restart, the runner initializes its in-memory results with 24 HPO-reused records only and immediately replaces `FINAL_MODEL_FITS.csv`, temporarily dropping any already-completed new final-fit rows until replay finishes. | Read-only code review of the new runner before final fit; `_rolling_table` precedes the new-slot loop. | Checkpoints and embeddings are not changed or retrained, but the rolling report can remain incomplete after interruption and misstate progress. | BUG FIX pending: pre-verify and reconcile completed immutable new-slot records before the first rolling-table write; focused resume test. | New runner/report only; HPO artifacts unaffected. | Reporting code/test only; zero retraining. |

**NB-FT-03/04/05 corrections (before final campaign).**

```diff
NB-FT-03: adapter accepted an unlisted trial in the final namespace
-> require unique immutable final slot and exact config/path/seed/window match
-> 6 adapter boundary tests pass; no fit existed before the correction

NB-FT-04: successful one-update smoke reported stopping_update, but runner read optimizer_updates
-> check stopping_update == 1 and verify success status plus all four artifact hashes
-> real one-update smoke passed: status ELIGIBLE, one optimizer update, checkpoint/history/validation diagnostic saved

NB-FT-05: restart briefly rewrote rolling table with only 24 HPO-reused rows
-> pre-verify all existing new-slot result directories before first table write
-> completed immutable rows are carried forward on restart; no fit is repeated
```

Focused final freeze/adapter/runner/HPO tests: 15 passed, including two
mocked runner integration tests showing successful smoke gating and 26-row
reconciliation (24 reused plus two pre-existing) before the first resumed
table write, with no repeated fit. Final Real dry-run resolved
72 slots, 24 reused HPO checkpoints and 48 new fits at window 201, with no test
loading. The smoke trial is isolated under `outputs/final_thesis_v1/smoke/` and
is not counted among the 72 final model instances.

| ID | Severity | Component | What was wrong | Evidence | Scientific impact | Fix / classification | Existing results affected | Rerun required? |
|---|---|---|---|---|---|---|---|---|
| NB-FT-06 | MAJOR provenance gate before held-out evaluation | `outputs/final_thesis_v1/freeze/REAL_FINAL_EXPERIMENT_SPEC.json` | The immutable fit specification references the original split and safe-bundle hashes, but does not itself enumerate exact train/validation/test trial IDs, held-out evaluation IDs, preprocessing identity, or raw/unit normalization convention. | Independent provenance review of the frozen spec and its parent manifests before any test evaluation; fit adapter itself loads train/validation-only bundles. | Final fitting is train/validation safe, but a later evaluation must not infer the held-out support or transform convention after seeing results. No test outcomes were opened for selection. | BUG/REPORTING-PROVENANCE COMPLETION: create a new immutable companion evaluation specification, bound by SHA-256 to the existing fit spec, before opening test data. Preserve the existing fit spec. | Prospective embedding/metrics only; 492 HPO artifacts and 72-slot fit design unchanged. | Metadata/provenance generation only; no retraining. |
| NB-FT-07 | MAJOR downstream scientific choice | W201 held-out metric support in `src/neurobridge/experiments/real_monkey_validated.py` | Existing V2 RSA grid is `linspace(40,569,10)`; four of its ten points (40, 98, 510, 569) lie outside W201 unpadded-valid centers 100..499. Existing V2 decoding paths can include all split rows without a window-valid mask. | Code inspection and W201 safe manifest: 400 valid centers of 600 per trial; no held-out metric was computed for the new final models. | All-600 versus valid-only support changes the estimand, especially for temporal and RSA results; selecting a support after seeing test outcomes would compromise the firewall. Training/HPO remains valid. | NEW EVALUATION PROTOCOL DECISION required before opening test: explicitly freeze per-metric support and any RSA grid adaptation; no silent change to V2 metric definition. | Prospective final embeddings/metrics only, not checkpoints or HPO. | No retraining; downstream evaluation may need a dedicated configuration. |
| NB-FT-08 | MINOR downstream reporting | New `tools/final_real_embeddings.py::export_slot` manifest | The first exporter draft accepts a completed final fit with status `INELIGIBLE_NEAR_COLLAPSE` but does not copy that status into the embedding manifest. | Read-only review of code before any real embedding export; no final embedding artifact exists. | A downstream aggregator could mistake a diagnostically exported collapsed fit for an eligible representation. Checkpoint/training is unchanged. | BUG FIX pending: add parent fit status and near-collapse flag to immutable embedding manifest, and require downstream aggregation to respect eligibility. | Prospective embedding manifests only; existing HPO/final checkpoints unaffected. | Metadata-only; no retraining or embedding recomputation. |
| NB-FT-09 | MINOR artifact-resume gate | New `tools/final_real_embeddings.py::export_slot` | The first exporter draft creates the final output directory before writing three arrays and manifest. An interruption leaves a partial final directory, and restart refuses that slot rather than resuming. | Code-path review before any real export; no export directory exists yet. | Training and scientific definitions unaffected, but a long 72-slot embedding campaign cannot automatically recover one interrupted export. | BUG FIX pending: write to a per-attempt staging directory and atomically publish only after all files/hashes are complete; preserve any interrupted staging artifact for diagnosis. | Prospective embedding outputs only; all checkpoints unaffected. | Exporter code/test only; no retraining. |

**NB-FT-08/09 corrections (before any real embedding export).**

```diff
old: immutable embedding manifest omitted fit eligibility and near-collapse status
-> add fit_status and near_collapse from the parent fit result
-> downstream can distinguish diagnostic collapsed embeddings from eligible ones

old: final export directory became visible before its three NPZ files and manifest completed
-> write and hash all files in a per-attempt staging directory, then atomically rename
-> completed slots are hash-checked and reused; an interrupted staging attempt remains separate
```

The 20 focused final-phase tests passed. The exporter publication/reuse test uses
generated data and mocked inference, including an `INELIGIBLE_NEAR_COLLAPSE`
parent. No Real test observation or final embedding was opened by these tests.

## Final-evaluation downstream qualification (2026-10-01)

| ID | Severity | Component | What was wrong | Evidence | Scientific impact | Fix / classification | Existing results affected | Rerun required? |
|---|---|---|---|---|---|---|---|---|
| NB-FEV-001 | MAJOR software blocker | `tools/final_thesis_uncertainty.py::_fit_probe_predictions` | Real `behavior_alphas` is serialized as a Python-literal dictionary in the frozen CSV, while the uncertainty runner initially accepted JSON only. | Real pilot failed parsing the recorded single-quoted mapping before any bootstrap output was written; parser regression test passes after JSON + `ast.literal_eval` handling. | Prevented Real uncertainty evaluation; did not alter or invalidate core metrics, embeddings, or fitted encoders. | BUG FIX: parse the existing frozen representation safely without changing alpha selection or probe semantics. | No existing scientific outputs; only the new uncertainty runner. | Rerun uncertainty evaluation only; no retraining. |
| NB-FEV-002 | MINOR numerical / validation | `tools/final_thesis_uncertainty.py` probe-equivalence check | A global one-thread BLAS cap produced tiny float32 differences from core progress-R² values and was initially mistaken for a metric mismatch. | Default thread settings reproduced all 51 Synthetic raw/unit checks exactly (0 mismatches); one-thread mode differed by at most 1.13249e-6 with identical data, split, alpha, and formula. | Floating-point implementation noise only; no scientific metric changed. | BUG FIX: remove the non-protocol global thread cap; preserve the frozen estimator and definition. | No core results; only bootstrap qualification. | Focused test/pilot repeat only; no encoder retraining. |
| NB-FEV-003 | MINOR computational efficiency | `tools/final_thesis_uncertainty.py` embedding cache | Initial preparation loaded separate duplicate arrays for raw/unit and retained all slots, increasing memory use unnecessarily. | Initial RSS was about 2.57 GB; streaming/shared loading reduced measured working set to about 0.71 GB (about 72% lower); 244/244 Synthetic raw/unit probe checks passed under the streaming path. | Resource risk only; no changes to samples, metrics, or model artifacts. | IMPLEMENTATION-ONLY FIX: share each loaded artifact and release it slot-by-slot; use bounded atomic cell checkpoints. | No existing core outputs or model artifacts. | No training; continue the planned bootstrap once safe resources are available. |
| NB-FEV-004 | MAJOR resume/provenance limitation | `tools/final_thesis_null_controls.py` campaign runner | The active null campaign holds completed slot results in memory and does not persist per-slot checkpoints until the branch finishes. | At 105/244 completed slots the agent confirmed results were in the parent process RAM only; stopping the run would discard them. The active process remained CPU-active with no observed OOM, and was therefore left running. | No core metric is changed; an interruption could force recomputation of this new null branch and lose the rolling state. | No change to the active run (would discard completed work). After completion, record for future resume: atomically persist/hash each finished slot and skip verified slots on restart. | Active Block-2 null campaign only; no prior scientific output. | No rerun if current campaign completes; future resume improvement is code-only. |
| NB-FEV-005 | MINOR numerical validation / hardening | `tools/final_thesis_downstream_eval.py::run_pairing_null` observed-vs-core guard | Runtime guard allowed `rtol=1e-8, atol=1e-10`, while the regression fixture and intended equivalence check use `1e-12`; observed values happened to match exactly, but the production guard was looser than the test. | Independent review found the tolerance discrepancy at the observed pairing comparison. The frozen-core sweep reported 384/384 exact matches with maximum absolute delta 0.0; this issue was recorded before correction. | No observed scientific value differs; the looser guard could fail to catch a future numerical/implementation drift. | BUG FIX / hardening only: align production guard with the frozen regression tolerance; no metric formula or sample changes. | No existing core outputs; affects only qualification of the new null branch. | Rerun focused tests and read-only observed-vs-core qualification; no model retraining. |
| NB-FEV-006 | MAJOR software/resume bug | `tools/final_thesis_downstream_eval.py::run_null_controls`, duplicate legacy label loop at lines 793–800 before checkpointed runner at lines 851–857 | The runner executed a full uncheckpointed label-shuffle pool in RAM, then would execute the new checkpointed label-shuffle pool a second time. The initial active session had no output checkpoint directory. | At campaign start, the exact process used eight CPU workers and completed 15/244 label slots in memory; no checkpoint/output existed. Source inspection showed the old loop preceding `_run_checkpointed_null_slots`, guaranteeing a duplicate 244-slot pass. | No model/core metric or protocol result was altered, but the run would waste a full label-shuffle pass and could lose work on interruption, undermining the promised resume behavior. | BUG FIX: remove the obsolete loop and run each label/pair slot only through the write-once, hash-verified checkpoint runner. Preserve seed/support/replicate definitions. | Only the new Block-2 null campaign; no pre-existing scientific results or core outputs. | Restart Block 2 from the beginning; the 15 RAM-only slots are not reusable. No retraining. |

**NB-FEV-004 implementation follow-up (2026-10-01).** The original in-memory null attempt exited on the observed/core assertion before writing branch artifacts; its computed arrays were not retained. The runner now has atomic per-slot/cell checkpoints and hash-verified resume tests. No second null campaign had started at the time this follow-up was recorded. The initial attempt is not a completed scientific result.

**NB-FEV-006 stop-count clarification (2026-10-01).** The initial status message observed 15/244 label slots completed; the final session output before Ctrl+C reported 30/244. All 30 were RAM-only, with no published output or checkpoint, and the parent plus all eight worker PIDs were confirmed stopped. This replaces the preliminary count for accounting; no scientific result is retained or counted.

## Real controlled-lag protocol amendment (2026-10-02; recorded before correction)

| ID | Severity | Component | What was wrong / changed | Evidence | Scientific impact | Fix / classification | Existing results affected | Rerun required? |
|---|---|---|---|---|---|---|---|---|
| NB-FEV-007 | MAJOR / NEW PROTOCOL DECISION | Real Block 4 controlled digital lag | The prior controlled-lag branch imposed only +10 bins on B and searched absolute lags -20..+20. The user now specifies four Real B-only interventions: +100, +120, +160, and +200 bins. | `tools/final_thesis_downstream_eval.py::run_controlled_lag` sets `shift = 10`; its scorer imports `LAGS = -20..20`. Existing `REAL_LAG_PROVENANCE.json` records only +10. Current branch contains the three CSV/JSON records but the provenance-listed figures are absent. | The old +10 outputs do not answer the amended robustness question. An absolute scan over -20..+20 cannot locate peaks shifted by +100..+200. The old encoders and all non-lag metrics remain valid. | NEW PROTOCOL DECISION: retain the existing 41-point local scan width/resolution as residual offsets -20..+20 around each known imposed delay; report physical lag as `imposed_shift + residual_offset`. Use one lag-independent held-out reference support shared by R0 and all four interventions. Shift B only, without wrap, and remap the exact frozen R0 B embeddings; do not retrain. Preserve the old +10 branch as superseded provenance rather than deleting it. | Only the old +10 controlled-lag outputs are superseded for the amended Real robustness claim. Core metrics, frozen checkpoints/embeddings, Synthetic results, and null/uncertainty inputs are unaffected. | Recompute Block 4 lag curves/summaries/figures from frozen embeddings only; no encoder training or embedding regeneration. |

**NB-FEV-007 implementation evidence (pending).** Before/after target: `B_R10(t+10)=B_R0(t)` and one absolute `[-20,+20]` scan -> B-only no-wrap shifts `{100,120,160,200}` with physical candidate grids `{d-20,...,d+20}` on the same lag-independent held-out reference rows -> report each intervention's recovered absolute peak and `Delta l_hat` versus its known digital shift. The centered local grid preserves the previous scan's resolution and makes the all-intervention common-support condition feasible with W201; boundary peaks remain censored/unresolved and are not called resolved lag estimates. No result has been interpreted under this amendment yet.

| NB-FEV-008 | MAJOR artifact-path inconsistency | `outputs/final_thesis_v1/final_evaluation/core_metrics/EMBEDDING_INDEX.csv` and downstream readers | All 122 indexed `manifest_path` values point to absent `core_metrics/embeddings/...` locations, so `load_core_context()` cannot load the frozen representations needed by pairing nulls or lag evaluation. | Read-only hash audit found exact alternate copies for all 122 rows under `embedding_process_plots/embeddings` (50 Synthetic under `synthetic/<trial_id>`, 72 Real under `<trial_id>`): all indexed manifest hashes and all per-artifact hashes match; 0 mismatches; 0.344 GiB total. | Downstream work is blocked by location only. Core metrics, embeddings, model weights, trial support, and definitions are intact and unchanged. | BUG FIX / artifact restoration: copy the 122 hash-verified files to the exact already-recorded index paths; do not edit the index or overwrite an existing target. Keep the alternate copies intact. | Null pairing, uncertainty bootstrap, and controlled lag readers; no core result values. | Restore/copy artifacts only; no retraining, embedding recomputation, or metric recomputation. |

**NB-FEV-008 evidence before restoration (2026-10-02).** The current filesystem had 122/122 index manifests missing from their recorded locations and 122/122 hash-identical copies available at the alternate location. No copy has yet been made at the time of this record.

**NB-FEV-008 correction evidence (2026-10-02).** After confirming that all 122 canonical target directories were absent, copied (not moved) each verified alternate directory into the exact path already recorded by `EMBEDDING_INDEX.csv`. `load_core_context()` then passed its full hash/manifest checks and loaded 122 rows: 50 Synthetic and 72 Real. Alternate originals remain intact. No index, core metric, embedding content, model, split, or scientific setting was changed.

| NB-FEV-009 | MAJOR uncertainty-analysis dependency | `tools/final_thesis_uncertainty.py::run_uncertainty` -> `final_thesis_core_metrics._synthetic_latents` | The Block-3 runner requires the original Synthetic generator config and complete `shared_data.npz` at `outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic/stage01_data/`; that parent run directory is absent, so the pilot exits before bootstrap or output publication. | Exact traceback: `FileNotFoundError` for `stage01_data/config.json`. The checked-in/final evaluation support and embedding artifacts exist, but the referenced latent container/config do not. | Synthetic latent-geometry uncertainty (Procrustes/RSA vs Z) cannot be bootstrapped from aggregate core metrics alone. Real and Synthetic decoding/lag computations may remain possible from frozen embedding metadata, but the current monolithic runner stops before them. No core metric or embedding is invalidated. | Pending: search only for an exact hash-verified archived source or an already sealed deterministic reconstruction recipe. If none exists, do not substitute plotted/reconstructed values silently; mark only dependent Synthetic-Z bootstrap outputs MISSING and run scientifically independent supported uncertainty cells with explicit partial status. | Block-3 uncertainty only; core tables and Blocks 1–2 unchanged. | No retraining. Bootstrap-only work may be partial unless exact parent source is recovered. |

**NB-FEV-009 evidence before correction (2026-10-02).** `python tools/final_thesis_uncertainty.py --pilot --replicates 1000` exited with code 1 before creating any output. The missing `SYN_RUN` parent was not regenerated or replaced during diagnosis.

## Real controlled-lag amendment 2 (2026-10-02; recorded before implementation)

| ID | Severity | Component | What was wrong / changed | Evidence | Scientific impact | Fix / classification | Existing results affected | Rerun required? |
|---|---|---|---|---|---|---|---|---|
| NB-FEV-010 | MAJOR / NEW PROTOCOL DECISION | Real controlled-lag branch and B encoder training | The prior amendment NB-FEV-007 specified `{100,120,160,200}` and frozen-encoder remapping only. The user now narrows the imposed shifts to `{100,160,200}` ms and requests both (a) shifted B data passed through frozen R0 B encoders and (b) new B encoders trained from random initialization on the shifted train/validation data. | Frozen spec selects W=201 ms, held-out split hash `e1c115...`; B_DISTAL safe bundle has 92,400 rows (154 train/validation trials, 600 bins each) and explicitly excludes test. `data/monkey_reaching_preload_smth_40/macaque_data.jl` SHA-256 is `bf6b9e...`, matching the frozen raw-source hash in the bundle. Existing 72 final encoders/embeddings remain available. The current lag scorer is fixed at `-20..+20`, which cannot evaluate these interventions. | The frozen-only branch measures a digital-shift response/encoder robustness, not training under shifted observations. The retrained B branch is needed for the requested synthetic-parallel question. A full absolute curve on the existing W201 common-support contract would have no rows across the complete `-20..+220` range; use the already recorded 41-point local residual curve `d+[-20,+20]` per known intervention, with a common reference support across all three shifts and censor peaks at the local boundaries. This is a local controlled-intervention response, not blind lag discovery or a biological delay estimate. | NEW PROTOCOL DECISION: preserve the old +10 branch and frozen R0 artifacts; create an immutable separate branch for shifts `{100,160,200}`. For each, (a) shift B only with no wrap and evaluate with the frozen R0 B checkpoint, and (b) train a fresh B encoder per architecture/objective/seed using the exact frozen HPO winner, W201, training schedule, split and seeds; reuse the unchanged R0 A encoder. Keep trial-level train/validation/test firewall and the existing local residual-grid rule. Store separate fit, shifted-bundle, embedding, curve, summary, figure and provenance artifacts. | Old +10 outputs remain historical; core metrics and all existing R0 checkpoints/embeddings remain unchanged. Only the controlled-lag scientific branch is supplemented/superseded for the requested delays. | Three shifts × 2 architectures × 4 objectives × 3 seeds = 72 new B fits, plus frozen-encoder evaluation from the same R0 parents. No A retraining or HPO. Run only after exact split/window and shifted-window equivalence qualification passes. |

**NB-FEV-010 evidence before implementation (2026-10-02).** The source and design were inspected read-only. `nvidia-smi` showed no Python compute process; existing Python processes had unchanged CPU time over a 5-second check. No training was started. The exact split and window-level shift equivalence still require executable qualification before any fit.

| NB-FEV-011 | MINOR software/path bug | `tools/final_real_controlled_lag_v2.py::_prepare_trainval_bundle` and test/provenance helpers | The new runner formed `src/neurobridge/...` source paths relative to the project root without the `src` component, so hashing the extractor failed on its first qualification pass. | Exact exception: `FileNotFoundError` for `<repo>\neurobridge\experiments\final_real_lag_revision.py`; the actual source is under `<repo>\src\neurobridge\experiments\`. The frozen Real spec/source/split checks passed before the failure. No shifted bundle, training run, embedding, metric, or figure was created; only the new branch spec manifest exists. | Blocks safe-input provenance and therefore the new lag qualification; no scientific result or existing output is affected. | BUG FIX: resolve new module hashes relative to the repository's `src/` directory; preserve the existing new-branch spec and verify it remains identical. | Only the new controlled-lag-v2 input preparation; all previous branches and frozen artifacts unaffected. | Correct the path and rerun qualification; no fit rerun because none started. |

| NB-FEV-012 | MINOR resume/qualification bug | `tools/final_real_controlled_lag_v2.py::main` qualification manifest comparison | Resume compared the entire freshly computed qualification object against an immutable prior record, including nondeterministic inference wall-clock seconds and throughput. | `--run-all --resume` exited before fit 1 with `existing qualification record differs; preserving it`; prior `QUALIFICATION.json` stores per-delay `inference_seconds` and `inference_valid_windows_per_second`, which necessarily vary across invocations. Fit result count remained 0/72. | No scientific metric, checkpoint, or existing result is affected; this blocks safe resumption despite deterministic input/embedding equivalence passing. | BUG FIX: preserve the original qualification record and compare only stable qualification invariants (source/split/partition, trial/window counts, lag grids, row support, and embedding equivalence within the frozen tolerance); treat runtime/throughput as provenance measurements, not identity fields. | New controlled-lag-v2 branch only; no core or legacy artifacts. | Runner comparison fix and qualification rerun; zero fit reruns because no fit began. |

**NB-FEV-012 correction evidence (2026-10-02).** The resume comparator now ignores only `inference_seconds` and `inference_valid_windows_per_second`, while requiring exact source/split/partition/window/support/device facts and both prior/current raw/unit embedding errors to remain within the pre-existing `2e-5` equivalence tolerance. The immutable first `QUALIFICATION.json` is retained. Four focused tests pass, including changed-source rejection and error-over-tolerance rejection; a second `--qualify-only` completed successfully against the existing record. No fit has started yet.

```diff
old: compare all qualification fields, including per-run inference timing -> reject valid resume
-> correction: compare stable invariants and the frozen numeric equivalence bound; retain first qualification record
-> new: repeated qualification passes without rewriting provenance or weakening source/support checks
```

| NB-FEV-013 | MINOR software/interface bug | `src/neurobridge/experiments/final_real_lag_fit.py::_load_trainval_bundle` vs `tools/final_real_controlled_lag_v2.py::_prepare_trainval_bundle` | The fit adapter expected `windows.npz`, while the trusted preparer publishes `trainval_windows.npz`. | Exact fit traceback: `ProtocolViolation: shifted train/validation bundle is incomplete`; directory inspection confirmed the bundle has `manifest.json`, `split.json`, and `trainval_windows.npz`, with no `windows.npz`. Qualification had passed; fit count stayed 0/72 and GPU remained at ~15% desktop utilization with no Python model process. | Prevents the authorized fresh-B campaign from starting; no scientific outputs or frozen results are affected. | BUG FIX: make the adapter consume the exact prepared filename `trainval_windows.npz`; keep the immutable safe bundle unchanged and preserve all hash/firewall checks. | New controlled-lag-v2 runner only; all prior data, checkpoints, embeddings, core metrics and +10 branch unaffected. | Adapter/test fix and resume; zero fit reruns because no optimizer step/artifact was created. |

**NB-FEV-013 correction evidence (2026-10-02).** The fit adapter now reads the exact existing `trainval_windows.npz` filename; the bundle itself was not modified. Five focused tests pass, including a temporary hash-verified train/validation bundle loaded by the adapter, and `py_compile` passes. No optimizer step/checkpoint/result was created by the failed attempt.

```diff
old: fit adapter expected `windows.npz` -> rejected prepared immutable train/validation bundle
-> correction: consume `trainval_windows.npz` as emitted by the trusted extractor
-> new: file name, manifest hash, safe split, trial IDs and no-test firewall are checked on the existing bundle
```

| NB-FEV-014 | MINOR reporting/runner bug | `tools/final_real_controlled_lag_v2.py::main` post-fit report assembly | After the first fit, execution referenced undefined variable `summaries` instead of the in-scope `summary_rows`. | Exact traceback: `NameError: name 'summaries' is not defined` at the final report block. The completed fit record is `lagv2-refit-w201-B_DISTAL-cnn1d-soft-d100-s1101-a39957d74dcb`: `ELIGIBLE`, selected update 150, stopped at 1500, 64.06 s; best/stopping checkpoints, history, validation embedding and refit test embedding are present with recorded hashes. Campaign count is 1/72. | Training and exported embeddings are valid; the exception occurred after fit/status persistence, shifted embedding creation and provenance write. It prevented final/rolling report publication and curves are correctly pending until complete shift triplets exist. | BUG FIX: remove the redundant undefined-variable/CSV-reload branch and pass the in-memory `summary_rows` directly to the report builder. Resume and reuse the verified first fit and embeddings. | New controlled-lag-v2 report/aggregation only; no frozen R0 or old +10 artifact. | Report-code correction and resume; do not retrain the completed fit. |

**NB-FEV-014 correction evidence (2026-10-02).** Removed the redundant undefined-variable conditional; `summary_rows` is passed directly to the report builder. The runner resumed the first fit and completed successfully at 1/72; `REAL_CONTROLLED_LAG_V2_REPORT.md` now exists. The existing first-fit `result.json` SHA-256 before/after resume is identical (`5804A763...BAC1`), confirming no retraining or overwrite. Five focused tests and syntax compilation pass.

```diff
old: finish fit/embedding work, then raise `NameError` while assembling report
-> correction: use already computed `summary_rows` directly
-> new: resume reuses fit 1/72 and emits the rolling status report cleanly
```

| NB-FEV-015 | MINOR / COSMETIC-REPORTING ONLY | `tools/final_real_controlled_lag_v2.py::_plot_results`, first controlled-lag figure | The figure-level legend and suptitle occupy the same top-center region, so the legend overlaps the title/subtitle. | Direct visual inspection of `outputs/final_thesis_v1/final_evaluation/controlled_lag_v2/figures/REAL_CONTROLLED_LAG_CURVES.png` after all 72 fits: overlapping title/legend is visible; plotted curve values and axes are present. | Figure readability only; no fit, embedding, lag score, eligibility, or scientific table is changed. | BUG FIX (presentation only): reserve separate vertical space for title and legend, regenerate only the figure files from the frozen CSVs. | The six `REAL_CONTROLLED_LAG_CURVES.{png,pdf,svg}` and `REAL_CONTROLLED_LAG_SHIFT_RECOVERY.{png,pdf,svg}` renderings only; no model/checkpoint/embedding/metric result. | Regenerate figures only; no retraining or metric recomputation. |

**NB-FEV-015 correction evidence (2026-10-03; pending).** Issue recorded before modifying the plotting layout. Corrected render and visual QA will be appended after regeneration from the already completed curve/summary tables.

| NB-FEV-016 | MINOR scientific reporting / visualization | `tools/final_real_controlled_lag_v2.py::_plot_results`, controlled-shift recovery figure | The plot shows only interior/identifiable delta points, but does not display how many estimates were omitted as unresolved/censored or the number of boundary R0 peaks. | `REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv`: 4/144 intervention rows have identifiable deltas; 140/144 do not; 126/144 intervention peaks and 23/24 R0 baseline peaks are at search boundaries. Current recovery PNG contains only four plotted points and no omitted-case counts. | Could make sparse identifiable cases look representative and obscure that most lag estimates are censored/unresolved; underlying per-seed CSV values remain intact. | REPORTING-ONLY FIX: add an explicit count annotation to the recovery figure; retain the same interior-only scatter and do not treat censored estimates as resolved. | `REAL_CONTROLLED_LAG_SHIFT_RECOVERY.{png,pdf,svg}` only; no fit, embedding, curve value, or eligibility changes. | Regenerate figures only; no retraining or metric recomputation. |

**NB-FEV-016 correction evidence (2026-10-03; pending).** The issue is recorded before changing the figure. Regenerate the annotation from the completed summary table, then verify it reports 4/144 identifiable and the censored/unresolved counts above.

**NB-FEV-015/016 correction evidence (2026-10-03).** Re-rendered only the six PNG/PDF/SVG presentation files from the existing per-seed CSVs. Visual QA confirms the curve-figure legend is on its own row above all eight panel titles, without overlap; the recovery plot visibly reports 4/144 identifiable, 140/144 unresolved/censored, 126/144 intervention boundary peaks, and 23/24 R0 boundary peaks. All six figure files are non-empty. The existing curve, summary, and fit-status CSV SHA-256 values match the input hashes recorded in postprocess provenance; all 72 fit records and both sets of 72 embedding archives pass integrity verification. Postprocessing reports `training_performed=false`, `metric_recomputation=false`, `core_metrics_modified=false`, and `raw_or_test_input_containers_loaded=false` (the saved test embeddings were opened only for structural/hash integrity checks, not for evaluating or changing metrics). No checkpoint, embedding, observed metric, split, or scientific protocol was changed.

Status: NB-FEV-015 and NB-FEV-016 are resolved by the evidence above; the earlier pending notes are retained as discovery-time history.

| NB-FEV-017 | MAJOR / NEW PROTOCOL DECISION | Real controlled-lag evaluation grid in `tools/final_real_controlled_lag_v2.py` and `src/neurobridge/experiments/final_real_lag_revision.py` | R0 was scanned only over -20..+20 ms and imposed delays over `d±20` ms, despite the source dataset documenting Gaussian spike smoothing with 40-ms standard deviation. | The checked-in monkey-reaching loader documents 1-ms bins and a Gaussian kernel with 40-ms std (lines 58-59; calls `smooth_spk(40)`); the frozen lag spec fixes residual offsets at only ±20 ms. | The current maxima often occur at the ends of a search interval shorter than the smoothing scale, so the existing outputs cannot establish whether peaks are localized on a two-sigma temporal span. This is a search-range limitation, not proof that the lag exceeds a trial or proof that the model failed. | NEW PROTOCOL DECISION (explicitly requested): add a separate lag-only re-evaluation using R0 offsets [-80,+80] ms and intervention grids `d+[-80,+80]` ms (2× the documented 40-ms smoothing SD), 1-ms resolution, and one common held-out reference support across all compared lags/branches/shifts; maintain same-trial matching and no wrap. Do not change, delete, or overwrite the previous branch. | Only controlled-lag curve/summary/statistics/figures and their interpretation; all 72 fits, checkpoints, embeddings, fit statuses, HPO, core metrics and non-lag results are out of scope and remain immutable. | Recompute lag evaluation and derived tables/figures from existing frozen held-out embeddings only; no raw preprocessing, embedding regeneration, retraining or refit. |

| NB-FEV-018 | MINOR software / import-path bug | New `tools/final_real_controlled_lag_v3_2sd.py` module import | Running the script by path adds the source folder but not the repository root to `sys.path`, so the shared lag helpers under the tools namespace cannot be imported. | First invocation exited before loading artifacts with `ModuleNotFoundError: No module named 'tools'`; no new output/staging directory, metric, training or artifact was created. | Blocks the requested post-hoc lag re-evaluation only; all parent outputs remain unchanged. | BUG FIX: add the resolved repository root to `sys.path` before importing the shared read-only helpers. No scientific setting changes. | New v3_2sd runner only; no checkpoint, embedding, metric or v2 artifact. | Code correction and rerun of the lag-only script; no training or embedding regeneration. |

| NB-FEV-019 | MINOR software / mask-qualification bug | New `tools/final_real_controlled_lag_v3_2sd.py::main` support invariant | The first support check incorrectly required every shifted embedding's valid coordinates to equal the unshifted R0 coordinates, although no-wrap shifts correctly invalidate different edge rows. | The rerun passed imports and hash verification, then stopped at `A/B valid held-out support differs across model slots` while comparing an intentionally shifted mask to R0. No lag curve, metric table, staging output, training, or embedding generation occurred. | This was an over-restrictive software assertion, not a data or protocol failure. The correct invariant is equality across seeds/architectures within each same shift/branch, followed by intersection across all conditions for common support. | BUG FIX: compare validity coordinate sets within each condition key (R0-A, R0-B, or branch×shift); intersect those condition-specific sets to construct the identical reference support. Preserve no-wrap exclusions and the frozen test IDs. | New v3_2sd runner only; v2 tables, checkpoints, embeddings and all core metrics remain unchanged. | Correct support check; rerun lag-only evaluation; no refitting or embedding regeneration. |

| NB-FEV-020 | MINOR software / local counting bug | New `tools/final_real_controlled_lag_v3_2sd.py::main` common-support report | The per-trial support-count comprehension referenced an undefined local name after successfully constructing the 1,560-row common support. | The second rerun passed source/hash checks and support intersection, then raised `NameError: name 't' is not defined` while formatting the per-trial counts. It stopped before score calculation and before creating any output/staging directory. | Reporting/checking only; no scientific calculation or parent artifact was affected. | BUG FIX: count reference tuples by their trial component using a non-shadowing local variable. | New v3_2sd runner only; no old result or model artifact. | Fix the support-count expression and resume the same lag-only computation; no retraining. |

| NB-FEV-021 | COSMETIC / terminal reporting | New `tools/final_real_controlled_lag_v3_2sd.py` final status print | The command printed a Unicode sigma symbol to a Windows CP1252 console after atomically publishing all results, and the console raised an encoding exception. | All expected tables, report, figures and `PROVENANCE.json` exist in the new branch; provenance records training=false and the computed row counts. The exception traceback is at the final print after `os.replace`; no output hash or computation failed. | Terminal exit message only; scientific outputs are complete and unaffected. | BUG FIX: use ASCII-only terminal status text. No re-analysis; retain the execution-source hash recorded in the generated provenance. | New v3_2sd console message only. | No metric, model, embedding, or figure rerun; code-only cosmetic correction. |

| NB-FEV-022 | MAJOR reporting / eligibility propagation | New two-sigma lag summaries and plots | First-pass summaries marked all frozen-R0 pairs as not near-collapse without carrying forward the A/B parent fit statuses; retrained-pair flags considered only the refit B, not the frozen A anchor. | Core embedding index records three near-collapse parent fits in the A/B populations: B_DISTAL Transformer/InfoNCE seed 1301 and A_PROXIMAL Transformer/behavior-contrastive seeds 1201 and 1301. Cross-checking the 49 interior-peak Δ rows shows 11 involve a near-collapse parent/refit; only 38 are fit-eligible. | The raw Procrustes scores remain valid descriptive measurements, but the first-pass plot/count of 49 could overstate interpretable learned-representation recovery by not marking those 11 cases. No model or lag score is changed by the correction. | BUG FIX / metadata propagation: combine A-anchor, R0-B, and (for the refit branch) shifted-B eligibility; keep the 49 geometric interior cases, label 11 as near-collapse/ineligible, report the 38 eligible cases separately, and regenerate eligibility-aware summaries/figures in a corrected immutable sibling branch. | First-pass v3_2sd eligibility columns and derived aggregates/figures only; all raw per-seed lag curve values and parent artifacts remain preserved and hash-verified. | Recompute/aggregate from the already frozen embeddings or first-pass curves; no training or embedding generation. |

| NB-FEV-023 | COSMETIC / terminal reporting | New `tools/final_real_controlled_lag_v3_2sd.py` completion print | One Unicode delta character remained in the second completion-status line after the corrected immutable output had been published. | The corrected folder contains all tables, report, figures and provenance; the process traceback is solely a CP1252 encoding error at the final `print`, after `os.replace`. No result-writing or hash step failed. | Terminal status only; published scientific outputs are complete and unaffected. | BUG FIX: replace the remaining Unicode character in terminal text with ASCII. The recorded execution-source hash remains the hash of the exact script that generated the artifacts. | Console completion line only; no scientific output, parent fit, checkpoint, embedding or core metric. | No recomputation; code-only stdout fix. |

| NB-FEV-024 | MINOR audit cross-check correction | Eligibility audit for `controlled_lag_v3_2sd_corrected` | The pre-run cross-check recorded 11 ineligible / 38 eligible among 49 geometrically identifiable deltas; the executed, row-level eligibility propagation yields 13 / 36. | The corrected `REAL_LAG_SUMMARY_2SD.csv` has 49 geometric interior deltas: 36 with `delta_fit_eligible=True`, 13 flagged. The 13 consist of 9 frozen-R0 rows (3 each at shifts 100/160/200) and 4 refit rows (0/2/2 by shift); all 13 carry `INELIGIBLE_NEAR_COLLAPSE`. | The 49 raw geometric delta estimates are unchanged. Eligibility-aware counts/figures must use 13 flagged and 36 eligible, not the earlier hand cross-check of 11/38. | AUDIT CORRECTION: retain NB-FEV-022 as the discovery record, but supersede its preliminary 11/38 count with this row-level verified 13/36 result. No score, eligibility rule, or protocol setting is changed. | Corrected v3_2sd eligibility summaries/aggregates/report/figures; first-pass branch remains historical and unchanged. | No training or curve recomputation required; corrected branch already contains the verified row-level result. |

```diff
old: curve legend overlapped the title; recovery plot omitted the number of censored cases
-> correction: reserve a separate one-row legend area and annotate identifiability/boundary counts
-> new: legible curve panels and explicit 4/144, 140/144, 126/144, 23/24 context; all per-seed inputs hash-verified unchanged
```

**NB-FEV-022/024 correction evidence (2026-10-03).** Published the corrected immutable sibling branch at `outputs/final_thesis_v1/final_evaluation/controlled_lag_v3_2sd_corrected/`. It reports 49 geometric interior Δ estimates, 13 near-collapse/ineligible, and 36 fit-eligible; 95/144 remain unresolved/censored. The per-seed curve table has 27,048 rows and matches every corresponding first-pass curve value exactly (0 changed values). All 17 pre-provenance output hashes pass; the 168-row summary, 1,560-row common support (39 trials × 40 rows), and no-wrap support invariants pass. Visual QA of the PNGs confirms the recovery plot shows all 49 interior cases split into eligible and flagged, and the sensitivity plot shows eligible counts 3/9/21 for frozen B and 1/5/15 for refit B at radii ±50/±60/±80. The preliminary NB-FEV-022 count of 11/38 is retained as discovery history but superseded by the verified 13/36 cross-check in NB-FEV-024. No encoder, embedding, checkpoint, core metric, or parent artifact was changed.

**NB-FEV-023 correction evidence (2026-10-03).** Replaced the remaining non-ASCII delta symbol in the terminal-only completion line; `python -m py_compile tools/final_real_controlled_lag_v3_2sd.py` passes. The published `PROVENANCE.json` records the SHA-256 of the exact executed source before this stdout-only edit; reversing that one-character edit reconstructs the recorded source hash. No scientific outputs were rerun or modified for this cosmetic correction.

**NB-FEV-023 wording clarification (2026-10-03).** The completion line contained two non-ASCII delta symbols, not one; both were replaced with ASCII text. Exact byte-level reversal reconstructs the execution-source SHA-256 `a5cd1f40f2aa44e1cbc46d73d7d4f54fa157a44d294bf7c81a16d4061cc2395e` recorded in the branch provenance. The earlier singular wording is corrected here; the discrepancy is cosmetic only.
