# Phase-2A HPO runner — implementation and dry-run report

Date: 2026-09-28. Scope: implementation and validation without any fit, optimizer update, test-data access, or V2 modification.

## Files created or modified

- `src/neurobridge/experiments/phase2a_hpo.py` — sealed-candidate reader, trial planner, validation-only qualification/selection functions, immutable-manifest and output guards, fit-free CLI logic. New file; no V2 source modified.
- `tools/phase2a_hpo.py` — dry-run-only entry point. New file.
- `tests/test_phase2a_hpo.py` — six focused tests. New file.
- `outputs/phase2a_hpo/dry_run/initial_plan.json` — immutable dry-run plan, including all initial resolved configs and planned paths. New file.
- This report. New file.

No existing V2 directory or artifact was written. The HPO code's only CLI mode is `--dry-run`; it has no fit entry point.

## Tests and dry run

- `C:\Users\loren\miniconda3\python.exe -m pytest tests/test_phase2a_hpo.py -q`: **6 passed** (4.26 s). The `cebra` Python environment lacked `pytest`, so its initial test invocation could not run; the test suite was then run with the base Python installation.
- `C:\Users\loren\miniconda3\python.exe tools/phase2a_hpo.py --dry-run`: **passed**. Zero fit-hook calls and zero optimizer updates by construction.
- Initial enumeration: **160 population-level fits**, comprising 64 Synthetic and 96 Real; 80 CNN and 80 Transformer; 40 per objective; 32 per population (`Synthetic A/B`, `Real TOTAL65/A/B`). This represents 64 initial candidate triples across 16 architecture/objective/domain cells, each evaluated on its domain's populations.
- Only candidates **0–3** occur in the initial plan. Candidate 0 has exact V2 numerical values. Candidates **4–5** are sealed but reserved; extension/finalist trigger logic is exercised by tests, not executed as fits.

## Invariants and provenance

- The 20 frozen candidate-table data lines matched SHA-256 `196d0b24251ed43f2f781af635a3cb1c37abf43e5e6e7d8ed66cd261aa616f6c`. A one-digit alteration was rejected in a test. Sobol points are **not** regenerated at runtime.
- The canonical channel-partition file matched SHA-256 `dc3ec6c7339cb69ab9607b0ce0126f159fdc86ba5a54946871b11859c09476bf`: A=32, B=33, disjoint, full 65-channel coverage. The same verified indices are assigned to all Real candidate/architecture/seed plans.
- Search root seed is 1101; Synthetic A/B effective seeds are 1101/1102; all Real populations use 1101. Dataset/split/channel-partition seeds are recorded separately as 42. Domain-specific frozen early-stopping settings and fixed model/data-shape settings are in each resolved config.
- Every planned trial has a deterministic ID and canonical config SHA-256. The dry-run plan records the source-file hashes and combined source-tree SHA-256 `b72b84c2328f24c8465fdb9ee0b71009508017dd402665bf223b3be4f29a8970`. The protocol file is included in the source list. Immutable-manifest tests verify exact-content resume and rejection of a changed payload; output-path tests reject V2 writes.
- The selection ledger accepts only validation loss, status, and stopping/checkpoint updates, plus cell/seed identity; extra test/scientific-result fields are rejected. Validation geometry rejects non-validation rows, shape/cardinality failures, nonfinite/non-unit embeddings, and uses the frozen three-condition near-collapse rule. Synthetic and Real candidate ranking is separate; qualification requires all populations, then all three HPO seeds. Paired extension is checked in tests.
- The dry-run references no test indices, test embeddings, Stage-5 outcomes, Z recovery, lag, decoding, RSA, CKA, or Procrustes. No existing test artifact was opened for candidate selection.

## Discrepancy / blocker before fits

This is a **dry-run planner and validation/selection core, not an authorized fit-capable runner**. Data/split/content-parent hashes for HPO-safe train/validation inputs remain `PENDING_HPO_SAFE_TRAIN_VALIDATION_INPUTS`; no safe input bundles or source snapshot for executable fits were generated. Thus the required fit-level provenance, resume from checkpoint, and failure accounting have not been integration-tested and cannot be claimed complete.

The current V2 fit helpers cannot be reused as-is under the requested hard firewall: `staged_shared_latent.py:753–757` invokes `stage_windows`, loads the full window cache and split, and validates the split including test; `real_monkey_validated.py:635–642` loads the full real window dataset and full split before selecting train/validation rows. Connecting those hooks would let HPO code load test-bearing inputs. No scientific setting needs changing, but a train/validation-only input boundary and fit adapter must be implemented and dry-run/integration-tested before any Phase-2A fit. The existing V2 code and outputs were left unchanged.

## GO / NO-GO

**GO** for the fit-free enumeration, candidate seal, pairing/seed checks, path protection, and pure validation-selection logic tested here. **NO-GO for starting initial Phase-2A fits** until the hard validation/test firewall is satisfied by safe train/validation-only inputs and an HPO-specific adapter with complete parent hashes, source snapshot, checkpoint resume/invalidation, and fit-failure accounting. No model was trained in this task.
