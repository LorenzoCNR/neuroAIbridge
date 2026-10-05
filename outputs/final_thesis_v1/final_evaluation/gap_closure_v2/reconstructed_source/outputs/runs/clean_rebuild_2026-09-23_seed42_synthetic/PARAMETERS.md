# Shared-latent benchmark — run parameters

Protocol fingerprint: `71d381cbff4b`
Run label: `clean_rebuild_2026-09-23_seed42_synthetic`

| Parameter | Value |
|---|---|
| `name` | `shared_latent_center_out` |
| `latent_dim` | `3` |
| `n_trials` | `200` |
| `n_conditions` | `8` |
| `trial_length` | `200` |
| `dt` | `0.02` |
| `n_neurons_A` | `160` |
| `n_neurons_B` | `120` |
| `lag_bins` | `10` |
| `window_size` | `21` |
| `stride` | `1` |
| `phi` | `0.4` |
| `noise_scale` | `0.05` |
| `baseline_mean` | `1.0` |
| `baseline_std` | `0.1` |
| `rate_scale` | `1.0` |
| `first_coordinates_multiplier` | `3.0` |
| `overdispersion` | `4.0` |
| `refractory_mean_bins` | `1` |
| `refractory_std_bins` | `0.25` |
| `train_per_condition` | `17` |
| `test_per_condition` | `5` |
| `validation_per_condition` | `3` |
| `train_fraction` | `0.7` |
| `test_fraction` | `0.2` |
| `validation_fraction` | `0.1` |
| `run_label` | `clean_rebuild_2026-09-23_seed42_synthetic` |
| `hidden_dim` | `64` |
| `cnn_layers` | `3` |
| `transformer_dim` | `64` |
| `transformer_heads` | `4` |
| `transformer_layers` | `2` |
| `transformer_dropout` | `0.1` |
| `batch_size` | `1024` |
| `learning_rate` | `0.001` |
| `weight_decay` | `0.0001` |
| `max_iterations` | `4000` |
| `max_epochs` | `100` |
| `validation_interval` | `400` |
| `min_iterations` | `800` |
| `relative_min_delta` | `0.001` |
| `early_stopping_patience` | `3` |
| `embedding_temperature` | `0.1` |
| `metadata_temperature` | `0.5` |
| `time_weight` | `0.5` |
| `condition_weight` | `0.5` |
| `seed` | `42` |
| `split_seed` | `42` |
| `training_seed` | `42` |
| `output_root` | `outputs/runs` |
| `time_offset_bins` | `10` |
| `temporal_objective_temperature` | `1.0` |

## Stage outputs

- `stage01_data/`: shared latent process and spike counts
- `stage02_windows/`: trial-safe windows and split
- `stage03_models/`: one fitted model per subject/branch/loss
- `stage04_embeddings/`: frozen transforms
- `stage05_metrics/`: recovery and lag metrics
- `stage06_figures/`: planar and spherical plots
- `stage07_profiles/`: computational profiles
