# Frosolone-inspired Pipeline 1 on NeuroBridge Synthetic

## Scope and result

This is a separate, downstream synthetic analysis. It reuses the cached seed-42 synthetic data and frozen trial split. **No neural encoder was retrained.** The analysis fits train-only PCA/NMA and a fixed QDA decoder; held-out test trials are evaluated only after all feature choices are fixed.

## What was run

The staged cache is `C:\Users\loren\Desktop\USB_Content\New project\Neuro_Bridge\outputs\runs\clean_rebuild_2026-09-23_seed42_synthetic`. The frozen configuration reports 200 trials, 8 balanced task conditions, 200 time bins at 0.02 s/bin, a 3-D simulated latent, 160 A units, 120 B units, and centered input windows of 21 bins. The existing split was reused unchanged: 140 train, 20 validation, 40 test trials, with five test trials per condition. A and B were analyzed separately; they were not pooled. The script loads only raw counts, labels, and validity masks; it does not materialize `Z_A`, `Z_B`, or `Z_shared`.

The analysis sequence was:

1. Read the cached raw neural count tensors `X_A`/`X_B`, task-condition labels, validity masks, frozen config, and split metadata. Record SHA-256 hashes in `provenance.json`.
2. Fit a 3-component PCA basis separately for A and B, using only valid time bins from TRAIN trials. The component count matches the configured simulated latent dimensionality; the true `Z` values were not used to fit PCA or choose a window.
3. For each PCA component and valid time bin, perform one-way ANOVA across the eight task conditions using TRAIN trials only. Choose the single component/time point with the smallest raw p-value and center a 21-bin interval there (0.42 s). The search yields 180 candidate time points per component. For B, the frozen leading invalid bins from the imposed lag are excluded. After fixing this window, compute the same time/component profile and eta-squared on TEST trials only as a held-out reproducibility check; it does not modify the window.
4. On PCA-score trajectories, fit a multiclass one-vs-rest CSP bank independently over the full valid trial interval and the selected window, using TRAIN trials only. Retain the high- and low-variance CSP filters for each of the eight classes; extract log-normalized projected variances; concatenate the full and selected-window features (Pipeline-1 feature fusion). Rank the fused features by mutual information on TRAIN only, retain eight, then fit fixed unregularized QDA.
5. As a context reference, fit the same QDA with only full-interval features. Report validation and test decoding. The held-out class-centroid distance matrices use only the already train-fitted PCA transform and are descriptive; they did not choose the NMA window or classifier.

## NMA-selected windows

| Population | Selected PC | Center (bin) | Center (s) | Window [start, stop) | TRAIN raw ANOVA p | TRAIN eta-squared in selected PC/window | TEST eta-squared in same PC/window |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 3 | 185 | 3.700 | [175, 196) | 0.000306 | 0.067 | 0.244 |
| B | 2 | 42 | 0.840 | [32, 53) | 7.74e-05 | 0.060 | 0.194 |

The paired profile plots show eta-squared over time on TRAIN and untouched TEST trials, with the TRAIN-selected window shaded. Held-out effect size inside that fixed window is the central check that class separation carries over; no test significance threshold or test-based window search is used.

## Held-out decoding

| Population | Procedure | Accuracy | Balanced accuracy | Test trials |
|---|---|---:|---:|---:|
| A | full_interval_reference | 0.150 | 0.150 | 40 |
| A | pipeline1_full_plus_NMA_interval | 0.150 | 0.150 | 40 |
| B | full_interval_reference | 0.075 | 0.075 | 40 |
| B | pipeline1_full_plus_NMA_interval | 0.075 | 0.075 | 40 |

Balanced accuracy is the mean recall across task conditions; with the balanced held-out split it equals accuracy.

The held-out set contains 40 trials (five per condition), so chance balanced accuracy is 0.125 and estimates are coarse. In this run, Pipeline 1 scored 0.150 for A and 0.075 for B; the full-interval reference scored the same values. Thus the classifier result does **not** show a held-out decoding gain. At the selected PC/window, held-out mean eta-squared was 0.244 for A and 0.194 for B, compared with 0.067 and 0.060 on TRAIN. This indicates measurable class-mean separation in the fixed held-out interval, but with five trials per condition it is noisy; it does not override the near-chance CSP/QDA result. The profiles and confusion matrices are the evidence, not a claim of a successful classifier.

## What is and is not a reproduction

The paper has a baseline Pipeline 0 and six NMA-derived variants (Pipelines 1-6). Pipeline 1 fuses classifier features extracted from the full temporal interval and the globally most class-separable interval. The paper's NMA uses PCA-derived component trajectories and timewise one-way ANOVA; pairwise Tukey tests support its pairwise-class variants, not the global Pipeline 1 window. It does **not** select by Euclidean distances. Its downstream EEG stack is filter-bank CSP, mutual-information feature selection, and QDA. See Methods, Sections 2.5-2.6 (PDF pages 7-8).

Here, the NMA interval-selection logic and Pipeline-1 feature-fusion idea are retained. CSP, mutual-information feature ranking, and QDA are also included, but adapted: there is no EEG filter bank (the synthetic sampling rate is 50 Hz, with a 25-Hz Nyquist frequency, and the observations are spike counts rather than EEG rhythms); CSP is one-vs-rest over the eight synthetic conditions and is fitted to the 3-D PCA-score trajectories. One low- and one high-variance filter are kept per class because the PCA input has three dimensions. Eight MI-ranked features are retained so QDA covariance is estimable with the available per-class training trials; this is a sample-limited adaptation of the paper's `m=2`, `D=4*K` setting. This is therefore a **Frosolone-inspired adaptation**, not an exact replication of the EEG pipeline. Pairwise Euclidean distances between held-out condition centroids in PCA-score space are provided as a separate descriptive result, not as the NMA selection rule.

## Interpretation limits

- The minimum raw p-value was selected over multiple component-time tests, following the paper's raw separability criterion. These p-values are selection diagnostics, not confirmatory significance claims; no multiplicity correction was used to choose the window.
- The saved split adjusts the nominal per-condition defaults to keep the requested global proportions: it contains 17-18 TRAIN, 5 TEST, and 2-3 VALIDATION trials per class. This exact saved split was preserved.
- The synthetic task classes and 3-D ground truth make this a controlled method check, not evidence that the same method generalizes to EEG or real monkey data.
- `Z` is intentionally excluded from PCA/NMA/decoder selection. Any future latent-recovery analysis must remain a separate post-hoc evaluation.
- The validation split is reported but is not used for tuning; no settings were chosen after seeing validation/test performance.
- The full-interval reference is a context baseline, not an additional Frosolone pipeline.
- Pipeline-1 held-out classification is near chance in this single split; positive eta-squared in the selected window is not sufficient to claim effective decoding.

## Files

- `tables/selected_nma_windows.csv`: selected component/time/window and training statistics.
- `tables/nma_pc_time_tests_*.csv`, `tables/nma_all_populations.csv`: train-only component/time ANOVA scan.
- `tables/nma_heldout_pc_time_profile_*.csv`: descriptive held-out profiles under the train-selected PCA basis/window.
- `tables/decoding_metrics.csv`: validation/test accuracy and balanced accuracy.
- `tables/pipeline1_train_feature_selection_*.csv`: train-only MI feature ranking and selection.
- `tables/confusion_*.csv`: held-out confusion matrices.
- `tables/test_centroid_distances_*.csv`: held-out class-centroid Euclidean distances in PCA-score space.
- `figures/`: NMA profiles, held-out confusion matrices, distance heatmaps, and decoding comparison (PNG + PDF).
- `provenance.json`: input/code hashes and the resolved procedure.
- `PAPER_RECAP.md`: paper methods/results summary with section, figure, and table references.

## References

Frosolone, M., Prevete, R., Ognibeni, L., Giugliano, S., Apicella, A., Pezzulo, G., & Donnarumma, F. (2024). Enhancing EEG-Based MI-BCIs with Class-Specific and Subject-Specific Features Detected by Neural Manifold Analysis. Sensors, 24(19), 6110. https://doi.org/10.3390/s24196110

Source PDF in this repository: `legacy_ai_for_all/Bibliog_Code/Frosolone, Prevete et al,Enhancing EEG-Based MI-BCIs with Class-Specific and, Subject... 2024.pdf`.
