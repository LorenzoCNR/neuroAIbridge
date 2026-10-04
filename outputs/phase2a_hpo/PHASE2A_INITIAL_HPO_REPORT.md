# Phase-2A initial HPO — automatic gate status

Status on 2026-09-28: **NO-GO before HPO training**. No fit or optimizer update was started; candidates 0–3 were not launched. Fits attempted/completed/failed: **0/0/0**. There are consequently no new candidate losses, collapse decisions, eligibility rankings, or extension-trigger decisions. The sealed 160-fit initial plan remains the existing `dry_run/initial_plan.json` and was not changed.

## Concrete firewall contradiction requiring clarification

The execution prompt requires both (a) extraction of exactly the frozen train/validation rows from the original data and (b) confirmation that *no test data/index/artifact was loaded* before the GO gate. The existing source containers are not train/validation-only:

- Synthetic: the existing Stage-1 `shared_data.npz` stores all 200 trials, and Stage-2 `windows_A.npz` / `windows_B.npz` and `split.json` store all partitions. `staged_shared_latent.py` `stage_windows` and `stage_fit` operate on those full caches (`stage_fit`, lines 753–757).
- Real: `data/monkey_reaching_preload_smth_40/macaque_data.jl` is a single joblib recording. `_load_source` in `real_monkey.py` loads the full recording before splitting it; `fit_validated_neural_model` in `real_monkey_validated.py`, lines 635–642, reads the full window cache and split unless a filtered bundle is supplied.

A trusted **preparation** step can read the original complete source/split, extract and seal only train/validation windows, and hand that filtered bundle to an HPO fit adapter. The fit adapter can then be strictly unable to access test rows or indices. But that preparation step would have read a complete container; it cannot truthfully satisfy the prompt's literal global statement that no test-bearing data/index/artifact was loaded at all. Conversely, forbidding even trusted preparation from reading the complete source/split prevents proving that a new bundle preserves the frozen trial assignments and data realization.

An explicit interpretation is required: does the no-test-loading rule apply to the **HPO fit and candidate-selection path** while allowing a one-time trusted extractor to read the complete original source/split, or does it prohibit the extractor as well? The former preserves the intended validation/test firewall if the extractor writes no test rows to HPO inputs and is independently tested; the latter needs a pre-existing train/validation-only source that is not currently established.

## Gate and work performed

No Phase-2A source code or V2 file was modified in this turn. No test outcomes or scientific metrics were opened. No new automated test or smoke fit was run, because the input-boundary contract is unresolved. Prior dry-run tests and candidate checksum results are in `PHASE2A_HPO_RUNNER_DRYRUN_REPORT.md`; they are not represented as satisfying this new fit gate. Fit-level provenance/resume/failure accounting and safe-input hashes are therefore still pending.

Automatic GO conditions are **not met**. The next action depends only on the firewall interpretation above; no scientific hyperparameter or protocol choice is being reopened.

## Subsequent execution and temporal-window amendment (2026-09-28)

The user subsequently authorized a trusted extractor. Five immutable 21-bin train/validation-only bundles were created. Sixteen focused tests passed, two one-update smoke fits passed, and the old 160-fit GO gate passed. The original initial campaign was then started.

Before the new 41/121/201-bin Real amendment arrived, **10 Synthetic population-level fits had completed** (all recorded `ELIGIBLE`) and an eleventh Synthetic fit was in progress. The running process was interrupted immediately on receipt of the amendment. The 11th trial directory contains an incomplete attempt without `result.json`; it is not counted as completed. **No Real 21-bin HPO fit was started.** All old artifacts remain preserved under `studies/phase2a_frozen_2026-09-28/`.

Consequently, the amendment's premise that *no Phase-2A fit had yet been executed* is factually false. It cannot be recorded as a pre-training amendment. The Real window design can still be versioned prospectively relative to **Real HPO**, but whether the 10 unaffected Synthetic checkpoints should be reused in the new 352-result study or retained only as a separate pilot is a provenance choice. The current campaign is stopped pending that explicit choice; neither old outputs nor V2 were overwritten. No test scientific outcome was inspected.

## Rolling cell: synthetic/cnn1d/soft (8/448 slots)

Validation-only candidate/population results and ranking are in `studies/phase2a_real_windows_2026-09-28/cells/synthetic_cnn1d_soft.json`; eligible ranking: [(2, 6.910511096318563), (3, 6.911315759023031), (1, 6.912679195404053), (0, 6.91291626294454)]; extension: PENDING_PAIRED_ARCHITECTURE. No test or scientific outcome used for selection.

## Rolling cell: synthetic/transformer/soft (16/448 slots)

Validation-only candidate/population results and ranking are in `studies/phase2a_real_windows_2026-09-28/cells/synthetic_transformer_soft.json`; eligible ranking: [(3, 6.907838582992554), (2, 6.908499717712402), (1, 6.91126012802124), (0, 6.91136876742045)]; extension: NO_EXTENSION. No test or scientific outcome used for selection.

## Synthetic-only initial campaign completed (2026-09-28)

The amended study's **64/64 Synthetic initial slots** completed with valid
results: 10 earlier Synthetic fits were hash-verified and reused, and 54 slots
were completed in the amended study. All eight architecture/objective cells
have validation-only candidate rankings in
`studies/phase2a_real_windows_2026-09-28/cells/`; none triggers extension
under the frozen rule. The machine-readable 64-row table is
`studies/phase2a_real_windows_2026-09-28/rolling_fit_results.csv`; the
immutable completion record is `synthetic_only_summary.json` in that study.

One long Windows checkpoint path caused a preserved `FAILED_RUNTIME` attempt;
it was rerun with the identical config/input/seed under a shorter immutable
trial ID and completed. The issue and before/after evidence are in
`outputs/AUDIT_ISSUES_AND_FIXES.md` (NB-P2A-01), and the exact ID mapping is in
`synthetic_only_short_path_manifest.json`. Interrupted partial attempts were
also preserved and never counted as completed results.

**Real HPO completed fits: 0.** The user requested a pause before Real to
provide a potentially non-random channel A/B definition; no Real fit will be
started against the prepared random-partition bundles until that direction is
resolved. Synthetic finalist-seed fits and post-HPO scientific evaluation are
not claimed complete by this initial-campaign report.

## Final Phase-2A initial-campaign status (2026-09-29; authoritative update)

The earlier NO-GO, rolling, and pause entries above are retained as historical
provenance. This section supersedes their *current-status* statements. No V2
artifact or frozen scientific setting was changed.

### Synthetic

- The 64/64 initial population-level fits were hash-verified and reused, not
  retrained. All eight architecture/objective cells had `NO_EXTENSION`.
- The 64/64 new finalist fits (root seeds 1201 and 1301; A and B) completed:
  zero technical failures and one ineligible near-collapse fit. Eight final
  winners were selected by the frozen validation-only rule. CNN1D winners:
  soft candidate 2, infonce candidate 3, time_contrastive_blocks candidate 2,
  behavior_contrastive_blocks candidate 2. Transformer winners: soft candidate
  3, infonce candidate 3, time_contrastive_blocks candidate 2,
  behavior_contrastive_blocks candidate 2.
- Exact finalist hyperparameters, per-seed/population losses, eligibility,
  selected/stopping updates, checkpoint hashes and provenance are in
  `studies/phase2a_real_windows_2026-09-28/synthetic_finalists/finalist_results.csv`,
  `synthetic_finalists/winners/*.json`, and
  `synthetic_finalists/completion_summary.json`.

### Real, canonical somatotopic partition

- Partition `real_somatotopic_v1` is recorded in
  `partitions/real_somatotopic_v1.json` (SHA-256
  `027ca4b9a6ba53471b72cf93a11e9514968af4399df0dc347de4c44973b98025`).
  A_PROXIMAL has 32 units; B_DISTAL has 33, with disjoint complete coverage of
  the 65 features and no electrode split. TOTAL65 is unchanged. The historical
  random partition was preserved but not used for these fits. A/B denotes
  proximal-/distal-enriched operational subpopulations, not discrete modules.
- All four centered Real windows (21, 41, 121, 201; positive_offset 10) passed
  the train/validation-only safe-bundle and smoke-fit gate. The 384/384 planned
  initial population-level fits (candidates 0–3, root seed 1101) completed.
  There were **310 ELIGIBLE, 74 INELIGIBLE_NEAR_COLLAPSE, zero technical
  failures**. By window:

  | Window | Eligible | Near-collapse | Failed |
  | --- | ---: | ---: | ---: |
  | 21 | 81 | 15 | 0 |
  | 41 | 78 | 18 | 0 |
  | 121 | 74 | 22 | 0 |
  | 201 | 77 | 19 | 0 |

- The 384 trial-level records are in
  `studies/phase2a_real_somatotopic_v1_2026-09-28/real_initial_results.csv`.
  The 96 window × architecture × objective × population summaries are in
  `REAL_INITIAL_BY_POPULATION.csv`; the 32 validation-only cell rankings are in
  `REAL_INITIAL_CELL_RANKINGS.csv`, with full candidate/population records in
  `cells/*.json`. The immutable completion record is `initial_summary.json`.
  The summary's rolling-table SHA-256 matches the final 384-row CSV.
- The frozen *paired-architecture* extension trigger is required in 9 of 16
  window/objective pairs: time_contrastive_blocks and
  behavior_contrastive_blocks at **all four** windows, plus infonce at window
  **121**. The other 7 pairs have `NO_EXTENSION`. Exact rankings and decisions
  are in `paired_extension/*.json` and `REAL_INITIAL_CELL_RANKINGS.csv`.
- The user-requested thermal pause followed fit 290. Its result and artifact
  hashes were verified; no partial fit was left. After the GPU limit was
  externally set to **240 W** and verified, fits 1–290 were reused and fits
  291–384 completed. The pause provenance is in
  `THERMAL_PAUSE_STATE_2026-09-29.json`. The power-cap change is an execution
  environment change, **not an HPO hyperparameter or protocol change**;
  training-time/throughput comparisons across the pause should be stratified
  or clearly labelled by power-cap regime.

All 384 Real result records, trial/config links and hashed checkpoint/diagnostic
artifacts were checked after completion; no missing or mismatched artifact was
found. Candidate ranking and eligibility used only frozen train/validation
bundles. No test data, Z recovery, decoding, lag, RSA, CKA or Procrustes
outcome was used for HPO selection. Real extension candidates 4–5, Real
finalist seeds 1201/1301, and post-HPO scientific/test evaluation were **not
run** in this stage.

Provenance limitation: the local preprocessed Real container does not carry
original unit-ID metadata, so its feature-index-to-unit-ID mapping cannot be
independently reconstructed from that container. The exact user-frozen mapping
was internally checked and preserved in the partition artifact.
