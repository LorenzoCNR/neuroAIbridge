# Real controlled lag: ±2σ re-evaluation

## What was changed

The earlier ±20 ms scan is retained unchanged as historical output. This separate branch recomputes lag curves only, reusing already frozen held-out A/B embeddings. No encoder was trained or rerun; no embeddings or checkpoints were regenerated.

The repository dataset-loader documentation states 1-ms bins and Gaussian spike smoothing with σ=40 ms. The primary scan therefore spans ±80 ms (2σ), sampled every 1 ms: R0 uses -80..+80 ms, and imposed delays use d+[-80,+80] ms. This radius is a search span, not a confidence interval.

## Support and protocol

- Imposed B shifts: 100, 160, 200 ms; both the frozen-R0 and retrained-B branches are retained.
- Same-trial, no-wrap matching; common held-out support across every lag, shift and branch: 1560 rows (39 test trials × 40 time points).
- The ±50/±60/±80 sensitivity table uses the same conservative support defined by the full ±80 scan.
- Metric: existing unit-embedding Procrustes R² lag profile. These are relative temporal correspondence and digital-intervention response, not biological or causal delays.

## Results

- R0 boundary peaks: 14/24.
- Intervention boundary peaks: 79/144.
- Δ estimates identifiable only when both R0 and intervention maxima are interior: 49/144; unresolved/censored: 95/144.
- Of the 49 geometrically interior Δ estimates, 13 involve a near-collapse/ineligible parent fit; 36 are fit-eligible. Geometric identifiability is not the same as fit eligibility.
- Fit eligibility remains unchanged for the B-refit campaign: 72 fits, 67 eligible and 5 near-collapse. Pair-level eligibility also requires the frozen A and R0-B anchor fits to be eligible.
- Sensitivity counts use the fixed ±80-ms common support; each cell is geometric interior / fit-eligible interior, out of 72:

| Search radius | B frozen | B refit |
|---:|---:|---:|
| ±50 | 9/72; 3/72 | 1/72; 1/72 |
| ±60 | 18/72; 9/72 | 6/72; 5/72 |
| ±80 | 30/72; 21/72 | 19/72; 15/72 |

- All per-seed lag scores are retained, including flagged fits; eligibility-aware plots/aggregates exclude flagged fits from eligible summaries.
- Seed mean/SD/range are descriptive for n=3; no inferential p-values or confidence intervals are claimed.

## Outputs

- REAL_LAG_CURVES_2SD.csv
- REAL_LAG_SUMMARY_2SD.csv
- REAL_LAG_SEED_AGGREGATES_2SD.csv
- REAL_LAG_CURVE_SEED_AGGREGATES_2SD.csv
- LAG_GRID_SENSITIVITY_50_60_80MS.csv
- LAG_GRID_SENSITIVITY_SUMMARY.csv
- REFIT_FIT_QUALIFICATION_COUNTS.csv
- figures/REAL_LAG_CURVES_2SD.png
- figures/REAL_LAG_CURVES_2SD.pdf
- figures/REAL_LAG_CURVES_2SD.svg
- figures/REAL_LAG_SHIFT_RECOVERY_2SD.png
- figures/REAL_LAG_SHIFT_RECOVERY_2SD.pdf
- figures/REAL_LAG_SHIFT_RECOVERY_2SD.svg
- figures/LAG_GRID_SENSITIVITY_50_60_80MS.png
- figures/LAG_GRID_SENSITIVITY_50_60_80MS.pdf
- figures/LAG_GRID_SENSITIVITY_50_60_80MS.svg
- PROVENANCE.json

## Integrity

Only frozen embedding artifacts, trial/time/validity metadata, manifests and fit-status records were read. No raw neural container, behavior labels, optimizer, training routine, HPO result, or core metric table was used to select lag settings.
