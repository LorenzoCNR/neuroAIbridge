# NeuroBridge claim freeze — gap_closure_v1

Frozen before the new bootstrap, temporal-order null, and C8 calculations on 2026-10-04. This is a **retrospective analysis freeze**, not a prospective preregistration: the V2 core results and earlier null/lag results have already been inspected. No model or hyperparameter can be chosen using these analyses.

## Claim tiers

| Tier | Claim and estimand | Data and representation |
|---|---|---|
| PRIMARY CONFIRMATORY (within the frozen V2 comparisons; retrospective designation) | Synthetic geometry: Procrustes R² of the **whole** simulated latent process Z. Real cross-population geometry: A/B Procrustes R². | Held-out trials; raw and unit are two prespecified, separately reported views. Neither view is selected by its outcome. |
| PRIMARY CONFIRMATORY (same qualification) | Accessibility: condition Balanced Accuracy (Synthetic), direction Balanced Accuracy (Real), with frozen train-fitted/validation-selected linear probes. | Held-out trials; raw and unit separately. |
| PRIMARY CONFIRMATORY (same qualification) | Synthetic temporal fidelity: Procrustes lag score at the known imposed +10-bin shift on lag-independent common support. | Held-out A/B; raw and unit separately. |
| SECONDARY | Synthetic RSA Spearman vs Z and progress R²; Real A/B RSA Spearman and linear CKA, continuous-behavior R²; lag peak and peak margin. | Held-out; raw and unit separately. |
| EXPLORATORY | Real R0 relative lag and controlled digital shifts; C8 condition-centroid equivariance; Frosolone adaptation; architecture/objective rank narratives. | No biological/causal-delay interpretation; no winner selection. |

The three primary dimensions—geometry, accessibility, temporal fidelity—remain distinct. A high score in one does not substitute for another. Near-collapse/fit eligibility is reported, never used to silently remove an unfavorable result. Seed 1101 is the HPO-selected checkpoint, while 1201/1301 are independent refits; n=3 is descriptive stability, not a population confidence interval.

## Inference and multiplicity

The prespecified observational unit for sampling uncertainty is the complete held-out trial, resampled within each dataset. A/B bootstrap draws use the same trial identities on both sides. Training seeds are never pooled into a trial bootstrap. The percentile 95% intervals describe held-out-trial sampling conditional on each frozen encoder/probe; they do not include training-seed or HPO-selection uncertainty.

Permutation p-values are interpreted within separate families: (1) Synthetic condition-label accessibility, (2) Real direction-label accessibility, (3) Synthetic A/B pairing geometry, (4) Real A/B pairing geometry, and (5) Synthetic temporal-order correspondence. Within each family, apply Holm correction across all evaluated architecture × objective × seed × population (where applicable) × representation comparisons for the stated primary metric. Secondary metrics and Real controlled-lag profiles are descriptive/exploratory; no uncorrected p-value is labeled confirmatory. A result is not called robust merely because one seed or one view passes. Report the complete family and adjusted p-values, including flagged fits, and distinguish eligibility-aware summaries.

The temporal-order null permutes B embedding **time blocks within each held-out trial**, independently by trial, while leaving A, B trial identity, task condition, trial length, and the lag-independent common coordinate support unchanged. Use deterministic seeds and a fixed block length of 10 bins; no test result may tune block length. This destroys global temporal alignment while retaining local within-block order, but it may not preserve all temporal autocorrelation or yield a clean null for a maximized lag score. Therefore the primary null comparison is the frozen score at the known Synthetic +10 shift; S_max and estimated lag are descriptive. Real R0 has no known true lag and remains exploratory.

The C8 extension is separate from V2. The group C8 acts on condition labels by `g·c=(c+g) mod 8`; the corresponding latent action rotates the first two direction coordinates by `2πg/8` and leaves progress unchanged. Because stochastic trial trajectories are not exact matched group orbits, evaluate **condition-centroid** equivariance only, not sample-level equivariance: learn a single orthogonal generator from training-set centroids, compose its powers, and score held-out centroid correspondence across all eight shifts. Report the score alongside the same construction on PCA and Z if frozen Z is available. No C8 result changes model selection.

## Integrity gates

All new outputs are versioned under `gap_closure_v1`. Frozen core metric tables, checkpoints, split files, HPO winners, and existing figures are read-only. Each analysis must record parent artifact hashes, code hash, settings, split, random seeds, time, and non-finite/failed cases. A missing exact frozen parent is a missing result, not license to regenerate an approximate parent.
