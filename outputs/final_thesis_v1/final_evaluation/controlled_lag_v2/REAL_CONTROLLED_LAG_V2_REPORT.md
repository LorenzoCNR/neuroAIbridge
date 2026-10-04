# Real controlled-lag V2

This is a separate, immutable controlled digital-intervention branch. It does not modify the frozen Real R0 fits or core metrics.

## Protocol

- Canonical Real window: 201 ms.
- B-only delays: 100, 160, 200 ms; no circular wrap; invalid centered windows excluded.
- R0 scans -20..+20 ms; each intervention scans its predeclared local 41-point curve d-20..d+20 ms.
- Curves use one held-out trial/time reference support shared by R0 and every shift in both branches.
- Frozen branch reuses the saved B R0 embedding rows; refit branch trains only B from random initialization. A remains the paired frozen R0 anchor.
- Frozen hyperparameters, optimizer, batch, early stopping, channel partition, split, objectives, architectures, and seeds are inherited unchanged from the sealed Real spec.
- No test-based HPO, model choice, or hyperparameter decision. No biological/causal lag interpretation.

## Completion

- Planned fresh B fits: 72.
- Fit statuses: {'ELIGIBLE': 67, 'INELIGIBLE_NEAR_COLLAPSE': 5}.
- Identifiable intervention deltas: 4/144; unresolved/censored deltas: 140/144.
- Boundary peaks: R0 23/24; interventions 126/144.
- Identifiable cases (descriptive only): transformer/behavior_contrastive_blocks/seed 1201/frozen_R0_encoder: shift 100 ms, delta 100.0 ms; transformer/behavior_contrastive_blocks/seed 1201/frozen_R0_encoder: shift 160 ms, delta 160.0 ms; transformer/behavior_contrastive_blocks/seed 1201/frozen_R0_encoder: shift 200 ms, delta 200.0 ms; transformer/behavior_contrastive_blocks/seed 1201/retrained_B_encoder: shift 100 ms, delta 108.0 ms
- Near-collapse fits are retained and flagged: transformer/behavior_contrastive_blocks/seed 1101/shift 100 ms; transformer/behavior_contrastive_blocks/seed 1301/shift 160 ms; transformer/behavior_contrastive_blocks/seed 1101/shift 200 ms; transformer/behavior_contrastive_blocks/seed 1201/shift 200 ms; transformer/behavior_contrastive_blocks/seed 1301/shift 200 ms
- Seed summaries use n=3 and are descriptive (mean/SD/range); no inferential p-values or population confidence intervals are claimed.
- Per-seed lag summary: `REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv`; seed aggregates: `REAL_LAG_SEED_AGGREGATES.csv`.
- Per-seed lag curves: `REAL_LAG_CURVES_R0_FROZEN_REFIT.csv`; curve mean/SD by seed: `REAL_LAG_CURVE_SEED_AGGREGATES.csv`.
- Fit-level rows: `REFIT_B_FIT_STATUS.csv`; descriptive fit aggregate: `REFIT_B_FIT_SUMMARY.csv`.

## Provenance

- Frozen spec SHA-256: `e017ffbebe9ac579e4e0cc575a511d1c83110fcf0bfee22639cb3853eec8ec8b`.
- Raw source SHA-256: `bf6b9eab46d8daaa8f6a9e69c7df1dbd9e337c60ee38af520042d3047c244d13`.
- Frozen split SHA-256: `e1c115122c0424c124e0268bebd2a1d976a081564288538e785269ed9c5a666a`.
- Somatotopic partition SHA-256: `027ca4b9a6ba53471b72cf93a11e9514968af4399df0dc347de4c44973b98025`.
- Frozen encoder equivalence qualification (max absolute errors):

```json
{
  "frozen_encoder_input_equivalence": {
    "0": {
      "cuda_available": true,
      "device": "cuda",
      "gpu": "NVIDIA GeForce RTX 3080",
      "inference_seconds": 0.7987625000532717,
      "inference_valid_windows_per_second": 19530.21079352072,
      "pytorch": "2.13.0+cu130",
      "raw_max_abs_error": 5.21540641784668e-08,
      "unit_max_abs_error": 2.6971101760864258e-06,
      "valid_embedding_rows_compared": 15600
    },
    "100": {
      "cuda_available": true,
      "device": "cuda",
      "gpu": "NVIDIA GeForce RTX 3080",
      "inference_seconds": 0.4152530001010746,
      "inference_valid_windows_per_second": 28175.5941489939,
      "pytorch": "2.13.0+cu130",
      "raw_max_abs_error": 4.470348358154297e-08,
      "unit_max_abs_error": 2.6971101760864258e-06,
      "valid_embedding_rows_compared": 11700
    },
    "160": {
      "cuda_available": true,
      "device": "cuda",
      "gpu": "NVIDIA GeForce RTX 3080",
      "inference_seconds": 0.3525313001591712,
      "inference_valid_windows_per_second": 26550.833913964154,
      "pytorch": "2.13.0+cu130",
      "raw_max_abs_error": 4.470348358154297e-08,
      "unit_max_abs_error": 2.6971101760864258e-06,
      "valid_embedding_rows_compared": 9360
    },
    "200": {
      "cuda_available": true,
      "device": "cuda",
      "gpu": "NVIDIA GeForce RTX 3080",
      "inference_seconds": 0.25892839999869466,
      "inference_valid_windows_per_second": 30124.15787545639,
      "pytorch": "2.13.0+cu130",
      "raw_max_abs_error": 4.470348358154297e-08,
      "unit_max_abs_error": 2.6971101760864258e-06,
      "valid_embedding_rows_compared": 7800
    }
  },
  "frozen_split_sha256_verified": "e1c115122c0424c124e0268bebd2a1d976a081564288538e785269ed9c5a666a",
  "local_lag_curves_ms": {
    "100": [
      80,
      120
    ],
    "160": [
      140,
      180
    ],
    "200": [
      180,
      220
    ]
  },
  "partition_sha256_verified": "027ca4b9a6ba53471b72cf93a11e9514968af4399df0dc347de4c44973b98025",
  "rewindowed_trainval_trial_checks": 2960,
  "source_sha256_verified": "bf6b9eab46d8daaa8f6a9e69c7df1dbd9e337c60ee38af520042d3047c244d13",
  "test_rows_absent_from_fit_bundle": true,
  "test_trial_count": 39
}
```

## Figures

- `REAL_CONTROLLED_LAG_CURVES.png`
- `REAL_CONTROLLED_LAG_CURVES.pdf`
- `REAL_CONTROLLED_LAG_CURVES.svg`
- `REAL_CONTROLLED_LAG_SHIFT_RECOVERY.png`
- `REAL_CONTROLLED_LAG_SHIFT_RECOVERY.pdf`
- `REAL_CONTROLLED_LAG_SHIFT_RECOVERY.svg`

Per-seed curves and estimates are in `REAL_LAG_CURVES_R0_FROZEN_REFIT.csv` and `REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv`; fresh-fit status is in `REFIT_B_FIT_STATUS.csv`.
Post-processing hashes and proof of zero training/metric recomputation: `REAL_CONTROLLED_LAG_V2_POSTPROCESS_PROVENANCE_8fe279ec49fc.json`.

A boundary peak is censored/unresolved. `Delta lag` is compared to the imposed digital shift only when both the R0 and intervention peaks are interior. Most estimates are unresolved here; the four identifiable rows are concentrated in Transformer/behavior-contrastive seed 1201, so they are not broad evidence of robustness. These analyses measure response to a digital temporal intervention, not biological delay or causality.
