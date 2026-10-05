# NeuroBridge gap-closure V2 — verified status

**Date:** 2026-10-05
**Scope:** downstream-only completion using the frozen V2 embeddings, checkpoints, split/support manifests, and metric definitions. No encoder training, HPO, model selection, or changes to observed core metrics were performed. `raw` and `unit` views remain separate.

## Outcome

The exact missing Synthetic parent was deterministically reconstructed and matched its frozen SHA-256. The held-out whole-latent-`Z` trial bootstrap is complete. The planned 3,000-permutation label-shuffle extension is complete and Holm-adjusted. A class-stratified complete-trial BACC bootstrap is documented as the preferred sampling interval, with the old unstratified result retained as sensitivity. **The requested experiment work is now stopped; move to thesis writing.**

No per-component `M`/`eta` recovery was added: the Synthetic geometry bootstrap compares each 3-D embedding to the full latent process `Z` jointly.

## DONE / PARTIAL / MISSING

| Item | Status | Evidence / interpretation |
|---|---|---|
| Frozen Synthetic source container | **DONE — exact reconstruction** | `reconstructed_source/outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic/stage01_data/shared_data.npz`; SHA-256 `079e283ab896c147cff12e5f089f9080e89418925eb6629da4a76af1c50baab8`, exactly the expected frozen hash. Config file SHA `fdd80c6e0bf918883b58968401bb2d73faea21336d886694abdce41c5bc0ea25`; canonical config SHA `53fe5f508f0c23b3db0a487bf2da5332423227b0c0d56109ad61d39d4f8b1dd7`; generator SHA `7f98e8d63c073b82f79fcdcdb7f8dc07763596741b8464bd500907e4e0c524f8`. Recorded in `RECONSTRUCTED_SYNTHETIC_PARENT_PROVENANCE.json`. |
| Synthetic whole-`Z` trial bootstrap | **DONE** | `synthetic_z_bootstrap/SYNTHETIC_Z_TRIAL_BOOTSTRAP.csv` (300 summaries) and `_REPLICATES.csv` (300,000 draws), plus seeds and provenance. 1,000 complete held-out trial resamples, 40 trials per draw, A/B populations, all three training seeds, PCA reference, raw/unit separately. Metrics: Procrustes R², RSA Spearman, and RSA Pearson versus full 3-D `Z`. The observed points reproduce frozen core values within `1e-10`; all 300 summaries have 1,000 finite draws. No encoder retraining. |
| Other frozen primary trial bootstraps | **DONE — reused** | Existing V1 branches remain intact: `gap_closure_v1/trial_bootstrap/` (340 rows), `trial_bootstrap_completion_v1/` (436), and `trial_bootstrap_real_rsa_v1/` (48). They cover the planned Synthetic/Real accessibility, geometry, behavior, CKA, lag-score, and Real A/B RSA quantities. All 436 additional metrics and 48 Real A/B RSA rows have 1,000 finite draws. |
| Preferred BACC sampling interval | **DONE** | `stratified_bacc_bootstrap/STRATIFIED_BACC_TRIAL_BOOTSTRAP.csv` (244 rows; 100 Synthetic + 144 Real), seeds, decision note, and provenance. It resamples whole held-out trials within each observed class, preserving observed class trial counts; each row has 1,000 finite replicates. Observed BACC values match the earlier frozen point estimates exactly (0 mismatches). |
| Seed-dot + trial-bootstrap-CI figures | **MISSING (presentation only)** | The numerical tables are complete, but no new uncertainty figures were generated in this closure. This does not require retraining; figures can be generated later from the frozen CSVs if needed. |
| Label-shuffle null resolution / Holm | **DONE** | `label_shuffle_holm_3000/NULL_LABEL_SHUFFLE_EXTENSION_REPLICATES_2000.csv` (488,000 new rows), `NULL_LABEL_SHUFFLE_SUMMARY_3000.csv` (244 rows), `HOLM_LABEL_FAMILY_RESULTS_3000.csv`, `HOLM_LABEL_FAMILY_SUMMARY_3000.csv`, seeds, provenance. The original 1,000 permutations were reused; 2,000 new permutations per slot used seeds 731000–732999. Total is 3,000 per each of 244 slots. |
| Claim freeze / temporal-order null / C8 extension | **DONE within the prior V1 scope** | `gap_closure_v1/FINAL_CLAIM_FREEZE.md` is retrospective, not a prospective preregistration. `temporal_null/` evaluates the frozen Synthetic `S_true(+10)` statistic; `symmetry_c8/` is exploratory. These branches were reused, not recalculated. A biological true lag is not assigned to Real R0. |
| Overall bootstrap status | **DONE for the frozen primary metric set** | The historical `uncertainty/UNCERTAINTY_STATUS.json` now points to all completed additive branches. Its previous content is preserved as `UNCERTAINTY_STATUS_PRE_GAP_CLOSURE_V2.json`. Training-seed variability remains separate (336 rows); three seeds are descriptive stability evidence, not a precise population CI. |
| Real controlled lag / Frosolone | **DONE as existing separate results; not rerun here** | `controlled_lag_v3_2sd_corrected/` and `frosolone_synthetic_pipeline1_seed42_complete/` were not modified. Current task required stopping after the bootstrap/null closure. |
| Real temporal-order null | **NOT APPLICABLE to the requested closure / not newly run** | The temporal-null branch remains Synthetic-focused; no new Real temporal-null definition was introduced. R10 is not part of this closure. |

## Statistical reading

The stratified BACC decision is **yes** for the preferred interval because Balanced Accuracy uses the recall of every fixed task class, while an ordinary trial bootstrap sometimes omits a class. The earlier unstratified bootstrap had 968/1,000 finite Synthetic draws and 939/1,000 finite Real draws. The stratified version avoids undefined draws but estimates uncertainty **conditional on the observed class trial counts**; it does not quantify class-prevalence uncertainty. Frozen probes were not refit inside bootstrap replicates, and no point estimate changed. The unstratified intervals remain available as sensitivity output. Median CI widths are 0.06159 → 0.05546 for Synthetic and 0.13737 → 0.11996 for Real; this width comparison is descriptive, not a model-performance comparison.

For label shuffles, all 244 observed BACC values exceeded all 3,000 corresponding null draws. Thus all raw one-sided empirical p-values are at the attainable floor `1/3001 = 0.00033322`. Under the frozen families, Holm gives:

| Family | Tests | Holm p-values < .05 | Adjusted p (all tests) |
|---|---:|---:|---:|
| Synthetic condition BACC | 100 | 100 | 0.0333222 |
| Real direction BACC | 144 | 144 | 0.0479840 |

The Real family result is close to .05 because of its 144-test family size and the finite-permutation resolution. This supports label accessibility under the specified trial-label null; it does **not** collapse accessibility, latent-process recovery, geometry, lag, or computational efficiency into one representation-quality claim.

## Issue discovered and resolved

After all 244 permutation slots had completed, the first merge attempt stopped with a `KeyError` for the deterministic PCA slot: the frozen CSV represented its absent training seed as `""`, while the new merge key converted JSON `null` to `"None"`. This was an aggregation bug only. It affected no permutation, embedding, metric point estimate, checkpoint, or training run.

The issue was recorded before correction in `AUDIT_ISSUES_AND_FIXES.md` (GC2-01). `tools/final_gap_closure_merge_recovery.py` normalized JSON null to the frozen empty-string key and merged the already complete checkpoints. **Zero permutation slots were recomputed.** The original failing campaign source/hash remains recorded in its immutable campaign manifest; the recovery helper hash is recorded in the final provenance.

## Integrity and checks

- The 244 checkpoint files each passed the merge-time checks: correct campaign hash, exactly 2,000 numeric values, all finite; 488,000 extension rows were then generated.
- The 300-summary Synthetic-Z table, 244-summary stratified-BACC table, and all label-null output artifact hashes match their provenance manifests.
- Existing core/parent hashes were verified against the extension campaign manifest. The historical 1,000-permutation CSV and summary were read-only inputs; the existing `core_metrics/`, checkpoints, split, HPO winners, `gap_closure_v1/`, and controlled-lag branch were not overwritten.
- Git branch/HEAD before and after this correction: `main` / `a73558826e55c610b568a86be45fa49adb563484`; no commit was created. The campaign retains the original driver hash; the additive recovery helper hash is in `label_shuffle_holm_3000/PROVENANCE.json`.
- `python -m py_compile tools/final_gap_closure_merge_recovery.py` passed. The recovery run's built-in invariants, explicit CSV count/status checks, observed-BACC equality check, and provenance hash checks passed. No general test suite or new plot generation was run.
- One-time downstream probe refits occurred only for the requested label-shuffle replicates. **No encoder/model was retrained.**

## Remaining thesis work — writing, not more experiments

**L3 / needed for defensible thesis claims:** integrate the versioned outputs, claim hierarchy, finite-permutation floor, conditional BACC interval, `n=3` seed limitation, Real lag censoring, and Synthetic-vs-Real distinctions into the thesis; describe the claim freeze as retrospective. The only non-numerical item left from the broader uncertainty request is the seed-dot + bootstrap-CI figure set; it can be produced from these frozen tables without retraining or rerunning experiments.

**L4 / future work, not a blocker:** independent real-subject/data replication, more training seeds, repeated somatotopic partitions, and richer autocorrelation-preserving temporal nulls. Do not use these as post-hoc selection criteria for the present frozen models.
