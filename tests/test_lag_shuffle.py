import numpy as np
import pytest

from neurobridge.experiments.lag_shuffle import shuffled_trial_ids


def test_trial_shuffle_is_a_derangement_and_preserves_time_membership():
    trial_ids = np.repeat(np.arange(5), 3)
    shuffled = shuffled_trial_ids(trial_ids, np.random.default_rng(7))
    assert np.array_equal(np.sort(np.unique(shuffled)), np.arange(5))
    mapping = {
        int(trial): int(shuffled[np.flatnonzero(trial_ids == trial)[0]])
        for trial in np.unique(trial_ids)
    }
    assert all(mapping[int(trial)] != int(trial) for trial in mapping)
    assert np.array_equal(np.bincount(shuffled), np.bincount(trial_ids))


def test_trial_shuffle_requires_multiple_trials():
    with pytest.raises(ValueError, match="two trials"):
        shuffled_trial_ids(np.zeros(3, dtype=int), np.random.default_rng(0))
