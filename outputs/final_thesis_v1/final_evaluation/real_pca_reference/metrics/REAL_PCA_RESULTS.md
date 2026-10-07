# Real-data PCA reference results

This is a separate downstream evaluation of the frozen W201 PCA reference. The PCA basis was fit on training trials only; the saved PCA embeddings were reused. No neural encoder was retrained and the frozen core metric tables were not modified.

## Evaluation

- Populations: TOTAL65, A_PROXIMAL (32 channels), B_DISTAL (33 channels).
- Raw and unit-normalized embeddings are evaluated separately.
- Held-out accessibility probes are fit on train, selected on validation, and scored on test, using the same probe grids and metric functions as the frozen core evaluator.
- Held-out A/B consistency uses matched trial/time points and the frozen common test support.
- All-valid diagnostics and A/B consistency are descriptive only; they are not generalization evidence.
- No latent-ground-truth recovery is available for real data. PCA was not added to the controlled-lag model grid, whose frozen protocol covers the neural encoders.

## Files

- `REAL_PCA_PRIMARY_HELD_OUT.csv`: compact held-out accessibility and A/B consistency results.
- `REAL_PCA_METRICS_LONG.csv`: all held-out and all-valid descriptive metrics.
- `REAL_PCA_PROBE_SELECTION.csv`: validation-selected linear probe parameters.
- `PROVENANCE.json`: source hashes, definitions, supports, and output hashes.

Embedding figures already exist in `outputs/final_thesis_v1/final_evaluation/embedding_process_plots/real/` and `outputs/presentation_ready/figures/real/seed42/total_65/`; this branch adds metrics, not replacement plots.
