# Phase-2A Real temporal-window amendment

Approved 2026-09-28 after 10 Synthetic HPO fits were completed and before any Real HPO fit. The 10 complete Synthetic fits remain valid and are reused with their original immutable provenance. An interrupted 11th Synthetic attempt is preserved but is not a completed fit.

Synthetic remains window 21 (dt 0.02 s). Real R0 has four explicit centered-window conditions: 21 (V2 continuity), 41 (~40 ms), 121 (~120 ms), 201 (~200 ms) at the source's 1-ms bins. No other window is authorized. The four Real conditions receive identical sealed candidate triples, architectures, objectives, seed mapping, channel partition and frozen early stopping, but are ranked separately. Real positive offset remains 10 bins. No test or scientific outcome may select a window or HPO candidate.

Initial result slots: 64 Synthetic plus 4 × 96 Real = **448**. Ten Synthetic slots already have complete immutable checkpoints; the remaining slots require execution. Candidate indices 0–3 only in the initial campaign. Candidates 4–5 and finalist seeds remain governed by the unchanged conditional rules and are not part of the initial 448.

The trusted extractor may open frozen complete Stage-1 Real data and split metadata solely to build centered windows from already-preprocessed neural values and retain train/validation trials. It estimates no normalization or preprocessing transform, evaluates no test trial, and writes no test row or test ID to HPO-safe bundles. The fit/qualification/selection modules may open only those safe bundles.

This is prospective relative to **all Real HPO training**, not relative to the 10 earlier Synthetic fits. Both source histories remain explicit; no V2 or earlier HPO artifact is overwritten.
