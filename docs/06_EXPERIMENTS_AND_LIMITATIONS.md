# Experiments, Results, and Limitations

**State checked against the local repository on 2026-10-07.** This page is a
navigation and interpretation guide, not a substitute for metric tables. For
exact values use the versioned CSVs and their provenance; for code dependencies
see [the Python pipeline map](07_PYTHON_PIPELINE_MAP.md).

## Current frozen study

The study has two scientifically distinct branches:

- **Synthetic:** a simulator supplies the complete latent process, allowing
  direct evaluation of whole-process representation recovery as well as
  decoding and temporal structure.
- **Real Area-2 reaching:** one recording supplies neural activity and
  behavior, but no known biological latent process. Results therefore concern
  task accessibility, cross-population consistency, robustness, temporal
  correspondence, and compute—not recovery of an unobserved real-data latent.

### Real protocol frozen for the final neural fits

The frozen Real experiment spec is
`outputs/final_thesis_v1/freeze/REAL_FINAL_EXPERIMENT_SPEC.json`.

- HPO compared windows `{21, 41, 121, 201}` and selected **201 bins** by the
  prespecified validation-rank rule. This is an operational selection on the
  frozen validation data, not evidence that 201 ms is intrinsically optimal.
- The recording is sampled at 1 ms. The positive training offset remains 10
  bins; controlled digital interventions are separate downstream analyses.
- The 65 channels are evaluated as `TOTAL65` and a fixed operational
  somatotopic split: `A_PROXIMAL` (32 proximal-enriched units) and `B_DISTAL`
  (33 distal-enriched units). This is a gradient-based partition, not a claim
  of two discrete anatomical modules or two brains.
- HPO is closed. The spec records 492 Real population-level HPO fits and says
  that multi-seed robustness of hyperparameter selection was not performed.
  No test/scientific metric was used for selecting the window or HPO winner.
- The frozen neural final grid has 72 fit slots:
  2 architectures (CNN1D/Transformer) × 4 objectives × 3 populations × 3
  training seeds (1101/1201/1301). Seed 1101 reuses validation-selected HPO
  checkpoints; 1201 and 1301 are independent final training seeds.
- Real PCA is a deterministic, separate train-fit reference, not a neural
  model multiplied across seeds. Full-sample refits are deferred and
  descriptive only.

The Synthetic output rows in the final metric table include seeds 1101, 1201,
and 1301. The interpretation of any particular row should be taken from its
run/provenance record rather than inferred from seed number alone.

## Where the results are

All paths below are relative to the repository root.

| Branch | Main artifacts | What they answer |
|---|---|---|
| Core evaluation | `outputs/final_thesis_v1/final_evaluation/core_metrics/` | Separate Synthetic whole-`Z` geometry/accessibility/lag summaries and Real task-accessibility/A–B consistency summaries, with long tables, embedding index, support manifest, and provenance. Held-out results are the primary generalization evidence. |
| Real PCA reference | `outputs/final_thesis_v1/final_evaluation/real_pca_reference/metrics/` | Matched Real PCA held-out accessibility and A/B consistency plus separate all-valid descriptive rows. It reuses the frozen train-fit PCA projections. |
| Null controls | `outputs/final_thesis_v1/final_evaluation/null_controls/` | Label-shuffle accessibility and random A/B trial-pairing nulls, with replicate tables and provenance. The root `NULL_TEMPORAL.csv` explicitly marks its generic temporal-order null as `NOT_COMPUTED`. |
| Synthetic temporal-order null extension | `outputs/final_thesis_v1/final_evaluation/gap_closure_v1/temporal_null/` | A separate block-based temporal-order null on frozen Synthetic held-out embeddings. It is not the missing generic/Real temporal-null row above. |
| Trial uncertainty | `outputs/final_thesis_v1/final_evaluation/uncertainty/` and `gap_closure_v1/`, `gap_closure_v2/` | The status manifest reports completed 1000-replicate whole-trial bootstrap coverage for the frozen primary metrics, keeping trial-sampling uncertainty separate from training-seed variability. The status also records that bootstrap-CI figures were not published in this branch. |
| Controlled Real lag | `outputs/final_thesis_v1/final_evaluation/controlled_lag_v3_2sd_corrected/` | Corrected ±80 ms search around digital B shifts of 100, 160, and 200 ms, with frozen-R0-B and B-refit branches. The report gives 49/144 geometrically interior shift estimates, of which 36 meet fit-eligibility requirements; the rest are unresolved/censored or ineligible. |
| Embedding plots | `outputs/final_thesis_v1/final_evaluation/embedding_process_plots/` | Raw and unit-normalized Synthetic/Real embedding plots with trial/time metadata. `PLOT_INDEX.csv` is the held-out index and `PLOT_INDEX_ALL.csv` the all-valid index. The local indices currently contain 502 rows each. |
| Presentation subset | `outputs/presentation_ready/` | Curated tables and selected static figures for presentation; it is not the complete local output archive. |
| Frosolone-inspired analysis | `outputs/frosolone_synthetic_pipeline1_seed42_complete/` | Separate Synthetic downstream NMA/ANOVA/CSP-style discrimination analysis using frozen neural data/embeddings; it is not part of the core metric table or a literal EEG-paper reproduction. |

The separate Real HPO ledgers, frozen winner tables, selection rule, and
complete model spec are under `outputs/phase2a_hpo/` and
`outputs/final_thesis_v1/freeze/`. The path map and script relationships are in
`docs/07_PYTHON_PIPELINE_MAP.md`.

## Interpretation rules

- Keep whole-process recovery, downstream decoding, temporal structure,
  held-out generalization, robustness, collapse/dimensionality diagnostics,
  and computational efficiency as separate claims. No single metric is an
  overall representation-quality score.
- `embedding_raw` and `embedding_unit` are distinct frozen representations;
  report them separately rather than selecting whichever looks better.
- Held-out scores are the primary generalization evidence. Full-sample
  refitting, when present, is descriptive/in-sample. The all-valid embedding
  plots project valid rows with the frozen held-out-trained encoder; they are
  not an independent full-sample refit.
- Training seeds 1101/1201/1301 describe training variability. Three seeds do
  not provide a precise population-level confidence interval. Whole-trial
  bootstrap intervals quantify held-out trial-sampling uncertainty conditional
  on frozen encoders/probes; they are not a replacement for seed variability.
- Real R0 measures relative temporal correspondence. A digital shift measures
  response to an imposed intervention. Neither estimates a biological,
  synaptic, or causal delay between brains.
- In the corrected controlled-lag report, 49/144 is a geometric
  interior-peak count; only 36/144 also satisfy the frozen fit-eligibility
  criteria. These are not 49 successful general lag recoveries.

## Remaining scientific limits

- Real results come from one Area-2 recording/session and one fixed
  somatotopic partition. Independent sessions/animals and repeated channel
  partitions are not represented by the current canonical result.
- There is no ground-truth latent process for the monkey recording, so Real
  metrics cannot establish recovery of a biological latent manifold.
- HPO winner selection used the frozen validation protocol at seed 1101; its
  multi-seed selection robustness was not run. Do not use test metrics to
  revisit that choice.
- The generic temporal-order null in `null_controls/NULL_TEMPORAL.csv` remains
  uncomputed. The separate Synthetic block-null extension does not fill that
  Real/generic gap.
- The Real lag profile contains boundary-censored or fit-ineligible cases;
  retain the full curves and avoid interpreting only the resolved subset as a
  universal result.
- Three training seeds are a limited stability sample, and the Frosolone-
  inspired branch is a separate methodological extension.

## Historical/reference material

The four introductory simulator notebooks, the original staged Synthetic
reference, older 2000-step outputs, and the 2026-09-22 exploratory Real branch
are reproducible historical/reference material. They are not the final
validation-selected Real protocol and must not be combined with its metrics
without a separate, explicit comparison. The 2026-09-24
`NEUROBRIDGE_CURRENT_STATE_AUDIT.md` is a dated historical audit; its former
`MISSING` statuses are not current status claims.
