import numpy as np

from neurobridge.experiments.presentation_rebuild import _real_config, _synthetic_config
from neurobridge.experiments.real_monkey_lag_eval import (
    _lag_curves_on_joint_support,
)
from neurobridge.experiments.real_monkey_validated import make_random_channel_split


def test_total65_configuration_uses_all_channels_and_requested_training_seed():
    partition = make_random_channel_split(n_channels=65, seed=42)
    config = _real_config(partition, training_seed=123, population="total_65")
    assert config.population_name == "total_65"
    assert config.channel_indices == tuple(range(65))
    assert config.training_seed == 123
    assert config.channel_split_seed == 42


def test_synthetic_training_seeds_keep_latent_and_split_seeds_fixed():
    config_42 = _synthetic_config(42)
    config_123 = _synthetic_config(123)
    config_456 = _synthetic_config(456)
    assert {config_42.seed, config_123.seed, config_456.seed} == {42}
    assert {config_42.split_seed, config_123.split_seed, config_456.split_seed} == {42}
    assert (config_42.training_seed, config_123.training_seed, config_456.training_seed) == (42, 123, 456)


def test_controlled_lag_uses_shared_support_and_recovers_b_only_plus_ten():
    rng = np.random.default_rng(7)
    n_trials, trial_length = 8, 100
    trial_id = np.repeat(np.arange(n_trials), trial_length)
    time_id = np.tile(np.arange(trial_length), n_trials)
    base = rng.normal(size=(len(trial_id), 3)).astype(np.float32)
    b10 = np.zeros_like(base)
    valid_b10 = time_id >= 10
    for trial in range(n_trials):
        rows = np.flatnonzero(trial_id == trial)
        b10[rows[10:]] = base[rows[:-10]]
    split = np.full(len(trial_id), "test", dtype="U10")

    def values(embedding):
        return {
            "embedding_unit": embedding,
            "trial_id": trial_id,
            "time_id": time_id,
            "split": split,
        }

    scores_r0, scores_r10, n_common = _lag_curves_on_joint_support(
        values(base), values(base), values(b10),
        np.ones(len(base), dtype=bool),
        np.ones(len(base), dtype=bool),
        valid_b10,
        branch="held_out",
    )
    assert n_common > 0
    assert max(scores_r0, key=scores_r0.get) == 0
    assert max(scores_r10, key=scores_r10.get) == 10
