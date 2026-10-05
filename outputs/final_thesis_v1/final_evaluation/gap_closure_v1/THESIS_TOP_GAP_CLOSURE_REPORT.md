# NeuroBridge thesis gap closure — 2026-10-04

## Outcome

This is a new, versioned **downstream-only** branch. No encoder was trained, no HPO reopened, no test outcome used for model selection, and no frozen checkpoint, split, winner, core metric, existing null, uncertainty status, controlled-lag result, or Frosolone artifact was changed. Raw and unit embeddings remain separate throughout.

The one incomplete **parent dependency** is the trial-bootstrap CI for whole synthetic latent-process `Z` geometry (Procrustes R² and RSA Spearman): its exact frozen source container is absent locally. This is **not** a reason to retrain a model. The existing core point estimates remain available, but valid trial-level CIs against `Z` require restoring the exact parent file with SHA-256 `079e283ab896c147cff12e5f089f9080e89418925eb6629da4a76af1c50baab8`.

## DONE / PARTIAL / MISSING

| Item | State | Evidence / limit |
|---|---|---|
| Frozen core and held-out support | DONE (reused) | `../core_metrics/PROVENANCE.json`, `EVALUATION_SUPPORT_MANIFEST.json`, 122 indexed embedding slots; verified by the existing loader and SHA-256 gates. |
| Existing label-shuffle and A/B trial-pair nulls | DONE (reused) | `../null_controls/NULL_LABEL_SHUFFLE.csv`, `NULL_AB_TRIAL_PAIRING.csv`; not recomputed. Their old `NULL_TEMPORAL.csv` correctly remains `NOT_COMPUTED` as historical output. |
| Training-seed variability | DONE (reused) | `../uncertainty/TRAINING_SEED_VARIABILITY.csv` (336 rows). Seed 1101 is HPO-selected; 1201/1301 are independent refits. No n=3 population CI is claimed. |
| Complete-trial bootstrap | PARTIAL | `trial_bootstrap/TRIAL_BOOTSTRAP_PRIMARY_METRICS.csv`: 340 rows, 1000 deterministic attempted draws per dataset, 100 Synthetic BACC, 144 Real BACC, 48 Real A/B Procrustes, 48 Synthetic A/B `S_true(+10)`; all with point estimate, bootstrap SE, percentile CI and parent embedding hash. Synthetic `Z` geometry missing for the exact-parent reason. |
| Additional planned primary bootstrap metrics | DONE except Synthetic Z geometry | `trial_bootstrap_completion_v1/TRIAL_BOOTSTRAP_ADDITIONAL_METRICS.csv`: 436 rows (100 Synthetic progress R², 144 Real position R², 144 Real velocity R², 48 Real A/B linear CKA), all with 1000 finite draws. `trial_bootstrap_real_rsa_v1/REAL_AB_RSA_TRIAL_BOOTSTRAP.csv`: 48 Real A/B RSA Spearman rows, each with 1000 finite draws. Thus 824 downstream CI rows are complete across the three versioned branches. |
| Temporal-order null | DONE for the prespecified Synthetic primary statistic | `temporal_null/NULL_TEMPORAL_SUMMARY.csv` (48 comparisons) and `NULL_TEMPORAL_REPLICATES.csv` (48,000 values), 1000 seeds; within-trial 10-bin B-block derangements, fixed A, fixed held-out same-trial support, raw/unit separately. Real R0 is not assigned a biological true lag and is not turned into a confirmatory test. |
| Claim/multiplicity freeze | DONE, retrospective | `FINAL_CLAIM_FREEZE.md` was written before the new calculations. The V2 outcomes were already known, so this is not a prospective preregistration. Temporal-null Holm family covers all 48 architecture/objective/seed/view comparisons. |
| Existing null p-value multiplicity | DONE (new interpretation, old results unchanged) | `claim_inference/HOLM_PRIMARY_NULL_FAMILIES.csv` and `HOLM_FAMILY_SUMMARY.csv` apply the frozen five-family Holm rule to the existing and new permutation results. |
| Synthetic C8 extension | DONE, exploratory | `symmetry_c8/C8_EQUIVARIANCE_HELD_OUT.csv`: 100 model/population/view rows, including deterministic PCA; 36 descriptive seed-variability groups; zero non-finite failures. The 45-degree proper-rotation generator is fitted on train centroids only, evaluated on held-out centroids; generator eighth-power closure error < `2e-15`. |
| Existing corrected Real lag | DONE (reused, exploratory) | `../controlled_lag_v3_2sd_corrected/REAL_LAG_SUMMARY_2SD.csv` and report; 49/144 geometric interior intervention deltas, 95/144 unresolved. Not recalculated here. |
| Frosolone synthetic Pipeline 1 | DONE as existing separate analysis; source rerun currently limited | `../../../frosolone_synthetic_pipeline1_seed42_complete/provenance.json` and `REPRODUCTION_CHAIN.md`; its referenced source container is the same missing exact file. No Frosolone results were modified. |

## Numerical reading, without collapsing dimensions

- Bootstrap median 95% interval widths across cells: Synthetic condition Balanced Accuracy **0.062**; Real direction Balanced Accuracy **0.137**; Real A/B Procrustes R² **0.279**; Synthetic `S_true(+10)` **0.083**. These are held-out-trial sampling intervals conditional on each frozen model/probe, not training-seed intervals.
- Additional median CI widths: Synthetic progress R² **0.055**; Real position R² **0.174**; Real velocity R² **0.131**; Real A/B linear CKA **0.198**; Real A/B RSA Spearman **0.186**. These metrics retain their existing definitions and support; they were not used to select a model.
- The 1000 unstratified trial draws yielded 968 finite eight-class Balanced Accuracy draws for Synthetic and 939 for Real. Missing-class draws were not silently redefined; the Real BACC rows carry `LOW_FINITE_REPLICATES`. Their percentile intervals are **conditional on all eight classes appearing in the bootstrap draw**, so nominal unconditional 95% coverage is not established. All 1000 geometry/lag draws were finite. The observed BACC itself is unchanged.
- Against the fixed within-trial block-order null, all 48 Synthetic `S_true(+10)` comparisons exceed their nulls. The smallest attainable empirical p with 1000 permutations is `0.000999`; Holm-adjusted p across 48 comparisons is `0.04795`. This supports **sensitivity to ordered within-trial correspondence under this null**, not exact lag recovery, a biological delay, or a causal link. The null destroys global order while retaining 10-bin local blocks, so it is not a comprehensive autocorrelation-matched null.
- Multiple-comparison correction materially changes the accessibility wording: all 100 Synthetic and 144 Real label-shuffle raw p-values are `<0.05`, but **none** passes the prespecified familywise Holm rule. With only 1000 permutations, the smallest possible adjusted p is `0.0999` for 100 Synthetic comparisons and `0.1439` for 144 Real comparisons. This is a p-value-resolution limitation, not evidence of chance-level decoding. For A/B Procrustes trial-pairing nulls, 44/48 Synthetic and 48/48 Real comparisons pass Holm; the temporal family passes 48/48. These are within their respective null definitions, not an omnibus representation-quality verdict.
- C8 condition-centroid equivariance is high for PCA (mean unit score **0.982**) and many neural objectives. Unit-view objective means across populations/seeds: CNN1D behavior **0.979**, InfoNCE **0.966**, time-contrastive blocks **0.977**, soft **−0.576**; Transformer behavior **0.984**, InfoNCE **0.985**, time-contrastive blocks **0.979**, soft **−0.992**. This is an exploratory condition-level symmetry diagnostic, not evidence that a model is globally superior. Individual seed/population values, raw view and train-fit diagnostics are in the CSV/figure. PCA has one deterministic record per population; neural objectives have three seeds.

## Method and integrity

The bootstrap resamples **complete held-out trials** with replacement; A/B draws are paired by the same trial IDs. Frozen validation-selected probes are refit on their original train split solely to reproduce the core predictions, then held fixed while test trials are resampled. No test probe fitting occurs. Exact core point estimates are asserted before publication. A 3×3 sufficient-statistics Procrustes implementation was checked against direct Procrustes on original and resampled data.

The temporal null keeps trial identity and condition labels, and permutes B's 13 contiguous ten-bin blocks independently inside each of the 40 held-out Synthetic trials; blocks are deranged, with no cross-trial mixing. It evaluates the **already defined** `S_true(+10)` on the same 5,200 reference coordinates. The existing A/B trial-pairing null asks a different question and is not substituted. No Real biological lag claim is made.

The C8 group acts as `(condition + g) mod 8`. Its embedding-space proper-rotation generator has angle 45 degrees around an axis inferred from **train** condition centroids, so the eighth power is identity. The held-out score aggregates all eight shifts. This tests centroid-level symmetry; independent stochastic trials do not provide exact samplewise group transforms. No C8 score fed back into HPO or model choice.

All new analysis subbranches contain immutable provenance with script hash, claim-freeze hash, parent hashes, support/split, seed policy, timestamp and artifact hashes. The Real RSA bootstrap also has 48 cell-level checkpoints for safe deterministic resume. The existing `../uncertainty/UNCERTAINTY_STATUS.json` deliberately remains `PARTIAL_BOOTSTRAP_PENDING`: updating it to complete would be false while Synthetic Z bootstrap is missing. The new `trial_bootstrap/UNCERTAINTY_STATUS.json` records this partial state.

## New artifacts

- `FINAL_CLAIM_FREEZE.md`; `AUDIT_ISSUES_AND_FIXES.md`; this report.
- `trial_bootstrap/`: `TRIAL_BOOTSTRAP_PRIMARY_METRICS.csv`, `TRIAL_BOOTSTRAP_SEEDS.csv`, `BOOTSTRAP_FAILURES.json`, `UNCERTAINTY_STATUS.json`, `PROVENANCE.json`.
- `trial_bootstrap_completion_v1/`: `TRIAL_BOOTSTRAP_ADDITIONAL_METRICS.csv`, `FAILURES.json`, `PROVENANCE.json`.
- `trial_bootstrap_real_rsa_v1/`: `REAL_AB_RSA_TRIAL_BOOTSTRAP.csv`, `BOOTSTRAP_SEEDS.csv`, `WORK_STATE.json`, 48 per-cell records, `PROVENANCE.json`.
- `temporal_null/`: `NULL_TEMPORAL_SUMMARY.csv`, `NULL_TEMPORAL_REPLICATES.csv`, `NULL_TEMPORAL_SEEDS.csv`, `PROVENANCE.json`, representative histogram in PNG/PDF.
- `symmetry_c8/`: `C8_EQUIVARIANCE_HELD_OUT.csv`, `C8_SEED_VARIABILITY.csv`, `C8_FAILURES.json`, `PROVENANCE.json`, all-seed dot plot in PNG/PDF.
- `claim_inference/`: `HOLM_PRIMARY_NULL_FAMILIES.csv`, `HOLM_FAMILY_SUMMARY.csv`, `PROVENANCE.json`.
- Code and focused tests: `tools/final_gap_closure.py`, `tools/final_gap_bootstrap_completion.py`, `tools/final_gap_real_rsa_bootstrap.py`, `tools/final_gap_temporal_null.py`, `tools/final_gap_c8.py`, `tools/final_gap_multiplicity.py`, `tests/test_final_gap_closure.py`.

## Verification

- 5 focused numerical tests passed: exact vs sufficient-statistics Procrustes, R² and CKA; temporal-null identity vs direct Procrustes; exact C8 orbit/group closure. `pytest` is unavailable in the active `cebra` environment, so the five test functions were invoked directly.
- Existing core loader verified 122 embedding slots and parent hashes; observed bootstrap/null point estimates were compared against frozen core metrics.
- Every listed artifact SHA-256 matches its subbranch provenance; all provenance files match the current claim-freeze hash. Frozen branches show no tracked modifications in `git status`.

## Remaining scientific work

**L3 (needed for strong defensible core claims):** restore the exact frozen Synthetic source container and compute the missing complete-trial Procrustes-vs-`Z` **and RSA Spearman-vs-`Z`** CIs in a new versioned continuation; decide prospectively whether Real eight-class BACC needs a class-stratified trial bootstrap or whether 939 finite draws with an explicit caveat suffice; either increase label-shuffle permutations in a new immutable branch to resolve the prespecified Holm families or explicitly present current accessibility null p-values as non-confirmatory; integrate the retrospective claim freeze into thesis text without presenting it as preregistration; preserve the Real controlled-lag boundary-censoring caveat. None requires encoder retraining.

**L4 / future work:** independent replication on another real dataset/subject, repeated somatotopic partitions if scientifically appropriate, more training seeds for population-level stochasticity, richer autocorrelation-preserving temporal nulls, and sample-level equivariance only if paired group-transformed data are generated. These are not prerequisites for interpreting the present frozen V2 results and must not be retrofitted as test-based tuning.
