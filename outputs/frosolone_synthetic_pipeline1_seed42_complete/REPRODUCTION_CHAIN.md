# Reproduction chain and reference files

This index records the complete provenance chain for the separate Synthetic NMA / Pipeline-1-inspired analysis. The staged source run and paper PDF are referenced in place; neither is copied or modified.

## Source paper

- Citation: Frosolone, M., Prevete, R., Ognibeni, L., Giugliano, S., Apicella, A., Pezzulo, G., & Donnarumma, F. (2024). Enhancing EEG-Based MI-BCIs with Class-Specific and Subject-Specific Features Detected by Neural Manifold Analysis. Sensors, 24(19), 6110. https://doi.org/10.3390/s24196110
- Local PDF: `C:\Users\loren\Desktop\USB_Content\New project\Neuro_Bridge\legacy_ai_for_all\Bibliog_Code\Frosolone, Prevete et al,Enhancing EEG-Based MI-BCIs with Class-Specific and, Subject... 2024.pdf`
- SHA-256: `f9cd42f77a9aa583a3a5092d593c74784e8552b50f24b04be0844e71b38fe1be`
- Relevant method locations: Sections 2.1-2.6 (PDF pp. 4-9); result datasets Sections 3.1-3.2 (pp. 9-12); Tables 1-5 and Figures 1-11. The detailed map and reported results are in `PAPER_RECAP.md`.

## Data and protocol parents (read-only)

- Staged source run: `C:\Users\loren\Desktop\USB_Content\New project\Neuro_Bridge\outputs\runs\clean_rebuild_2026-09-23_seed42_synthetic`
- Neural/count container: `C:\Users\loren\Desktop\USB_Content\New project\Neuro_Bridge\outputs\runs\clean_rebuild_2026-09-23_seed42_synthetic\stage01_data\shared_data.npz` (SHA-256 `079e283ab896c147cff12e5f089f9080e89418925eb6629da4a76af1c50baab8`)
- Frozen configuration: `C:\Users\loren\Desktop\USB_Content\New project\Neuro_Bridge\outputs\runs\clean_rebuild_2026-09-23_seed42_synthetic\stage01_data\config.json` (SHA-256 `fdd80c6e0bf918883b58968401bb2d73faea21336d886694abdce41c5bc0ea25`)
- Frozen trial split: `C:\Users\loren\Desktop\USB_Content\New project\Neuro_Bridge\outputs\runs\clean_rebuild_2026-09-23_seed42_synthetic\stage02_windows\split.json` (SHA-256 `9d1b9efbb1f5b2fe744013efa4a7482350e7a30800b88602e8a68c2ffe64430a`)
- Split counts: {"test": 40, "train": 140, "validation": 20}
- Data fields loaded: `labels`, `X_A`, `X_B`, `valid_A`, `valid_B`. Latent truth arrays (`Z*`) are deliberately not loaded. Test IDs are retained only to keep the frozen partition and for final held-out reporting; no test observations fit PCA, NMA window selection, CSP, feature selection, or QDA.

## Analysis implementation

- Script: `C:\Users\loren\Desktop\USB_Content\New project\Neuro_Bridge\tools\frosolone_pipeline1_synthetic.py`
- Script SHA-256: `d8c06723083efad4a634dfae92e7f82420b4a41ff729130cde954b73f1b2de07`
- Analysis protocol, decisions, findings, limits: `PIPELINE1_SYNTHETIC_ANALYSIS.md`
- Paper methods/results recap: `PAPER_RECAP.md`
- Machine-readable provenance and resolved settings: `provenance.json`
- Neural model retraining: **none**. This analysis fits a downstream PCA basis and CSP/QDA decoder using the cached synthetic observations; it does not refit any NeuroBridge neural encoder.

## Exact reproduction command

Run from repository root, using a new output directory (the script refuses to overwrite):

```powershell
python tools/frosolone_pipeline1_synthetic.py --run-dir "C:\Users\loren\Desktop\USB_Content\New project\Neuro_Bridge\outputs\runs\clean_rebuild_2026-09-23_seed42_synthetic" --output-dir "outputs/frosolone_synthetic_pipeline1_seed42_rerun"
```

The command reads the source/cache files above, writes the report, tables, and figures below, and does not modify its parents.

## Generated deliverables

- `figures/confusion_A_pipeline1.pdf`
- `figures/confusion_A_pipeline1.png`
- `figures/confusion_B_pipeline1.pdf`
- `figures/confusion_B_pipeline1.png`
- `figures/heldout_balanced_accuracy.pdf`
- `figures/heldout_balanced_accuracy.png`
- `figures/nma_profile_A.pdf`
- `figures/nma_profile_A.png`
- `figures/nma_profile_B.pdf`
- `figures/nma_profile_B.png`
- `figures/test_centroid_distances_A_NMA.pdf`
- `figures/test_centroid_distances_A_NMA.png`
- `figures/test_centroid_distances_B_NMA.pdf`
- `figures/test_centroid_distances_B_NMA.png`
- `PAPER_RECAP.md`
- `PIPELINE1_SYNTHETIC_ANALYSIS.md`
- `provenance.json`
- `tables/confusion_A_full_interval_reference.csv`
- `tables/confusion_A_pipeline1_full_plus_NMA_interval.csv`
- `tables/confusion_B_full_interval_reference.csv`
- `tables/confusion_B_pipeline1_full_plus_NMA_interval.csv`
- `tables/decoding_metrics.csv`
- `tables/nma_all_populations.csv`
- `tables/nma_heldout_all_populations.csv`
- `tables/nma_heldout_pc_time_profile_A.csv`
- `tables/nma_heldout_pc_time_profile_B.csv`
- `tables/nma_pc_time_tests_A.csv`
- `tables/nma_pc_time_tests_B.csv`
- `tables/pipeline1_train_feature_selection_A.csv`
- `tables/pipeline1_train_feature_selection_B.csv`
- `tables/reference_train_feature_selection_A.csv`
- `tables/reference_train_feature_selection_B.csv`
- `tables/selected_nma_windows.csv`
- `tables/test_centroid_distances_A_full_valid_support.csv`
- `tables/test_centroid_distances_A_selected_NMA_window.csv`
- `tables/test_centroid_distances_B_full_valid_support.csv`
- `tables/test_centroid_distances_B_selected_NMA_window.csv`

All generated files (excluding the checksum list itself) are SHA-256 indexed in `SHA256SUMS.txt`.
